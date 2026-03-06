# equation_service.py
import asyncio
import base64
import httpx
import re
from io import BytesIO
from typing import List, Tuple
from app.core.config import settings
from app.models.document import Equation, Figure, Position
from app.core.logger import get_logger

logger = get_logger(__name__)

# Maximum concurrent Mathpix API calls
MAX_CONCURRENT = 5

# Figures larger than this (in bytes of base64 data) are likely photos, not equations
# ~50KB of base64 ≈ 37KB image — most equations are smaller than this
MAX_EQUATION_IMAGE_SIZE = 80_000

# Minimum aspect ratio (width/height) for equation candidates
# Equations tend to be wider than tall; photos are more square
MIN_ASPECT_RATIO = 0.3


def _is_likely_equation_image(fig: Figure) -> bool:
    """
    Quick heuristic to check if a figure image could plausibly be an equation.
    Filters out large photos, diagrams, and full-page images before calling Mathpix.
    """
    if not fig.image_base64:
        return False

    # Check raw base64 size — large images are almost certainly not equations
    b64_size = len(fig.image_base64)
    if b64_size > MAX_EQUATION_IMAGE_SIZE:
        return False

    # Try to check image dimensions
    try:
        from PIL import Image as PILImage
        img_data = base64.b64decode(fig.image_base64)
        img = PILImage.open(BytesIO(img_data))
        w, h = img.size

        # Very large images (> 800px in both dimensions) are likely photos
        if w > 800 and h > 800:
            return False

        # Very small images (< 20px) are likely artifacts
        if w < 20 or h < 20:
            return False

    except Exception:
        # If we can't check dimensions, allow it through
        pass

    return True


class MathpixService:
    """Service for extracting equations from images using Mathpix OCR API"""

    API_URL = "https://api.mathpix.com/v3/text"

    def __init__(self):
        self.app_id = settings.MATHPIX_APP_ID
        self.app_key = settings.MATHPIX_APP_KEY

    @property
    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_key)

    async def _call_mathpix(self, client: httpx.AsyncClient, image_base64: str) -> dict | None:
        """Send a single image to Mathpix and return the JSON response."""
        headers = {
            "app_id": self.app_id,
            "app_key": self.app_key,
        }
        payload = {
            "src": f"data:image/png;base64,{image_base64}",
            "formats": ["latex_styled", "text"],
            "math_inline_delimiters": ["$", "$"],
            "math_display_delimiters": ["$$", "$$"],
        }

        try:
            resp = await client.post(self.API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("Mathpix HTTP error %s: %s", e.response.status_code, e.response.text)
            return None
        except Exception as e:
            logger.error("Mathpix request failed: %s", e)
            return None

    @staticmethod
    def _has_math(result: dict) -> bool:
        """
        Check whether the Mathpix response indicates the image contains math.
        Uses multiple signals: latex_styled content, confidence, and math indicators.
        """
        latex = result.get("latex_styled", "")
        text = result.get("text", "")

        # If no content at all, skip
        if not latex and not text:
            return False

        # Check the content to analyze (prefer latex_styled, fallback to text)
        content = latex or text

        # Very short content is unlikely to be a meaningful equation
        if len(content.strip()) < 3:
            return False

        # LaTeX-specific math commands — strong indicators
        latex_commands = [
            '\\frac', '\\sum', '\\int', '\\sqrt', '\\alpha', '\\beta',
            '\\gamma', '\\delta', '\\theta', '\\pi', '\\omega', '\\lambda',
            '\\mu', '\\sigma', '\\phi', '\\rho', '\\epsilon', '\\eta',
            '\\cdot', '\\times', '\\div', '\\pm', '\\mp',
            '\\leq', '\\geq', '\\neq', '\\approx', '\\infty',
            '\\partial', '\\nabla', '\\vec', '\\hat', '\\bar', '\\dot',
            '\\left', '\\right', '\\begin', '\\end', '\\mathrm', '\\mathbf',
            '\\overline', '\\underline', '\\overbrace', '\\underbrace',
            '\\lim', '\\log', '\\ln', '\\sin', '\\cos', '\\tan',
        ]
        if any(cmd in content for cmd in latex_commands):
            return True

        # General math symbols — also strong indicators
        math_symbols = ['^', '_', '=', '≥', '≤', '≠', '≈', '±', '∞',
                        '∑', '∫', '∂', '√', '→', '⇒', '∈', '∉',
                        '⊂', '⊃', '∪', '∩', '∀', '∃']
        math_symbol_count = sum(1 for s in math_symbols if s in content)
        if math_symbol_count >= 2:
            return True

        # Single = with numbers/variables on both sides suggests an equation
        if '=' in content and len(content.strip()) > 5:
            return True

        return False

    @staticmethod
    def _extract_eq_number(latex: str) -> str | None:
        """Try to find an equation number like (6-4) or (1.2) in the LaTeX string."""
        match = re.search(r'\([\d][\d.\-]*\)', latex)
        return match.group(0) if match else None

    async def _process_single_figure(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        fig: Figure,
        idx: int,
        total: int,
    ) -> Tuple[Equation | None, Figure | None]:
        """Process a single figure through Mathpix, returning either an equation or the original figure."""
        async with semaphore:
            logger.info("Mathpix: processing figure %d/%d (%s)", idx + 1, total, fig.figure_id)
            result = await self._call_mathpix(client, fig.image_base64)

            if result is None:
                logger.warning("  → Mathpix returned no result, keeping as figure")
                return None, fig

            latex = result.get("latex_styled", "")
            text = result.get("text", "")
            confidence = result.get("confidence", None)
            confidence_rate = result.get("confidence_rate", None)
            logger.debug(
                "  → Mathpix response: confidence=%.3f, confidence_rate=%.3f, "
                "latex_styled=%r, text=%r",
                confidence or 0, confidence_rate or 0,
                (latex[:100] + "...") if len(latex) > 100 else latex,
                (text[:100] + "...") if len(text) > 100 else text,
            )

            if self._has_math(result):
                content = latex or text
                eq_number = self._extract_eq_number(content)
                eq = Equation(
                    equation_id=f"eq_mathpix_{fig.figure_id}",
                    latex=content,
                    image_base64=fig.image_base64,
                    position=fig.position,
                    number=eq_number,
                )
                logger.info("  → EQUATION detected: %s", content[:120])
                return eq, None
            else:
                logger.info("  → not an equation (latex=%r, text=%r)",
                            latex[:60] if latex else "", text[:60] if text else "")
                return None, fig

    async def extract_equations_from_figures(
        self, figures: List[Figure]
    ) -> Tuple[List[Equation], List[Figure]]:
        """
        Send each candidate figure to Mathpix to detect whether it contains an equation.
        Pre-filters figures by size/dimensions to skip obvious non-equations.
        Processes up to MAX_CONCURRENT figures in parallel.

        Returns:
            (equations_found, remaining_figures)
            - equations_found: Figure images that Mathpix identified as equations
            - remaining_figures: Figures that are NOT equations
        """
        if not self.is_configured:
            logger.warning("Mathpix API keys not configured — skipping image equation extraction")
            return [], figures

        # Pre-filter: only send plausible equation candidates to Mathpix
        candidates = []
        remaining_figures: List[Figure] = []

        for fig in figures:
            if _is_likely_equation_image(fig):
                candidates.append(fig)
            else:
                remaining_figures.append(fig)

        skipped = len(figures) - len(candidates)
        if skipped > 0:
            logger.info(
                "Pre-filter: %d/%d figures skipped (too large for equations), %d candidates remain",
                skipped, len(figures), len(candidates),
            )

        if not candidates:
            logger.info("No equation candidates after pre-filtering")
            return [], remaining_figures

        # Process candidates concurrently with a semaphore
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        equations: List[Equation] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                self._process_single_figure(client, semaphore, fig, idx, len(candidates))
                for idx, fig in enumerate(candidates)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Mathpix task failed: %s", result)
                continue
            eq, fig = result
            if eq is not None:
                equations.append(eq)
            if fig is not None:
                remaining_figures.append(fig)

        logger.info(
            "Mathpix done: %d equations extracted from %d candidates (%d total figures, %d skipped)",
            len(equations), len(candidates), len(figures), skipped,
        )
        return equations, remaining_figures
