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

def _is_likely_equation_image(fig: Figure) -> bool:
    """
    Check if a figure image is worth sending to Mathpix.
    Only filters out images that definitely cannot contain equations:
    empty images and tiny artifacts. All other images (including large
    tables, diagrams, etc.) are sent to Mathpix because equations can
    appear inside tables and complex figures.
    """
    if not fig.image_base64:
        return False

    # Filter out tiny artifacts
    try:
        from PIL import Image as PILImage
        img_data = base64.b64decode(fig.image_base64)
        img = PILImage.open(BytesIO(img_data))
        w, h = img.size

        # Very small images (< 20px) are likely artifacts / icons
        if w < 20 or h < 20:
            return False

    except Exception:
        pass

    return True


class MathpixService:
    """Service for extracting equations from images using Mathpix OCR API"""

    API_URL = "https://api.mathpix.com/v3/text"

    def __init__(self, app_id: str = "", app_key: str = ""):
        self.app_id = app_id or settings.MATHPIX_APP_ID
        self.app_key = app_key or settings.MATHPIX_APP_KEY

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
            code = e.response.status_code
            if code in (401, 403):
                raise RuntimeError(
                    "Mathpix API key is invalid or expired. "
                    "Please update your Mathpix credentials in Admin → API Keys."
                ) from e
            if code == 429:
                raise RuntimeError(
                    "Mathpix API quota exceeded. "
                    "Please upgrade your Mathpix plan or update credentials in Admin → API Keys."
                ) from e
            logger.error("Mathpix HTTP error %s: %s", code, e.response.text)
            return None
        except RuntimeError:
            raise  # Re-raise our own RuntimeErrors (401/429)
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
        confidence = result.get("confidence", None)

        # If no content at all, skip
        if not latex and not text:
            return False

        # If Mathpix returned empty latex_styled with zero confidence,
        # it did not recognize any math — the text field is just raw OCR
        if not latex and (confidence is None or confidence < 0.1):
            return False

        # Check the content to analyze (prefer latex_styled, fallback to text)
        content = latex or text

        # Very short content is unlikely to be a meaningful equation
        if len(content.strip()) < 3:
            return False

        # Reject table/tabular content — not equations
        if re.search(r'\\begin\{(?:tabular|table|array)\}', content):
            return False

        # Reject pure \text{...} content — just text, not equations
        text_stripped = re.sub(r'\\text\s*\{[^}]*\}', '', content).strip()
        if not text_stripped or len(text_stripped) < 3:
            return False

        # LaTeX-specific math commands — strong indicators
        # (excludes structural commands like \begin, \end, \left, \right,
        #  \mathrm, \mathbf which appear in tables and formatting)
        latex_commands = [
            '\\frac', '\\sum', '\\int', '\\sqrt', '\\alpha', '\\beta',
            '\\gamma', '\\delta', '\\theta', '\\pi', '\\omega', '\\lambda',
            '\\mu', '\\sigma', '\\phi', '\\rho', '\\epsilon', '\\eta',
            '\\cdot', '\\times', '\\div', '\\pm', '\\mp',
            '\\leq', '\\geq', '\\neq', '\\approx', '\\infty',
            '\\partial', '\\nabla', '\\vec', '\\hat', '\\bar', '\\dot',
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

    @staticmethod
    def _split_equations(content: str) -> List[str]:
        """Split Mathpix response into individual equations.

        Handles display math ($$...$$), inline math ($...$), and
        newline-separated equations. Merges continuation lines (starting
        with =) back into the previous equation so multiline derivations
        stay as one equation.
        """
        raw_parts: List[str] = []

        # 1. Extract display math blocks ($$...$$)
        display_parts = re.split(r'\$\$', content)
        remaining_text_parts = []
        for i, part in enumerate(display_parts):
            part_stripped = part.strip()
            if i % 2 == 1 and part_stripped:
                # Inside $$...$$ — split on \\ but merge continuations later
                sub_eqs = re.split(r'\\\\', part_stripped)
                for seq in sub_eqs:
                    seq = seq.strip()
                    if seq and len(seq) > 2:
                        raw_parts.append(seq)
            else:
                remaining_text_parts.append(part_stripped)

        # 2. Extract inline math ($...$)
        remaining = ' '.join(remaining_text_parts)
        inline_parts = re.split(r'\$', remaining)
        leftover_text = []
        for i, part in enumerate(inline_parts):
            part_stripped = part.strip()
            if i % 2 == 1 and part_stripped:
                if len(part_stripped) > 2:
                    raw_parts.append(part_stripped)
            else:
                leftover_text.append(part_stripped)

        # 3. From leftover, split by newlines and look for equation-like lines
        leftover = ' '.join(leftover_text)
        for line in leftover.split('\n'):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            has_math = bool(re.search(
                r'[=<>≤≥≈≠]|\\(?:frac|sqrt|sum|int|alpha|beta|gamma|delta|mu|epsilon|sigma|omega|vec|hat|bar)',
                line
            ))
            if has_math:
                raw_parts.append(line)

        if not raw_parts:
            return [content] if content.strip() else []

        # 4. Merge continuation lines — lines starting with = are part of
        #    the previous equation (multiline derivations)
        merged: List[str] = []
        for part in raw_parts:
            stripped = part.lstrip()
            is_continuation = stripped.startswith('=') or stripped.startswith('\\approx') or stripped.startswith('\\equiv')
            if is_continuation and merged:
                # Append to previous equation with newline
                merged[-1] = merged[-1] + ' \\\\ ' + part
            else:
                merged.append(part)

        return merged

    @staticmethod
    def _is_junk_mathpix_equation(latex: str) -> bool:
        """Filter out Mathpix results that are not real equations.

        Catches: plain numbers with units, single variables, LaTeX table
        markup, very short fragments, and text-only content.
        """
        stripped = latex.strip()

        # Empty or very short
        if len(stripped) < 4:
            return True

        # LaTeX table markup (multicolumn, hline, etc.)
        if re.search(r'\\?multicolumn|\\hline|\\begin\{tabular\}', stripped):
            return True

        # Plain number with optional unit — e.g. "384,300 km", "=5.68 km/s"
        # Remove leading = sign for check
        check = re.sub(r'^[=\s]+', '', stripped)
        # Pattern: optional minus, digits with commas/decimals, optional unit text
        if re.match(
            r'^-?[\d,]+(?:\.\d+)?(?:\s*(?:×|\\times)\s*10[\^{}\d]+)?\s*'
            r'(?:km|m|s|kg|rad|hrs?|hours?|days?|months?|[a-z]{1,3}(?:\s*/\s*[a-z]{1,3})?(?:\s*\^?\s*\d)?)?'
            r'(?:\s*/\s*[a-z]{1,3}(?:\s*\^?\s*\d)?)?\s*$',
            check, re.IGNORECASE
        ):
            return True

        # Just a variable name with no operator (e.g. "V∞Mars", "V∞", "planet")
        # Must have at least one relational/math operator to be a real equation
        clean = re.sub(r'\\(?:text|mathrm|mathbf)\{[^}]*\}', '', stripped)
        clean = re.sub(r'[_^{}\\]', '', clean).strip()
        has_operator = bool(re.search(
            r'[=<>≤≥≈≠+\-*/]|\\(?:frac|sqrt|sum|int|cdot|times|div|pm|leq|geq|neq|approx)',
            stripped
        ))
        if not has_operator and len(clean) < 20:
            return True

        # Pure text with no math symbols (e.g. "planet", "as above")
        alpha_only = re.sub(r'[^a-zA-Z\s]', '', clean)
        if alpha_only.strip() == clean and len(clean) < 30:
            return True

        return False

    async def _process_single_figure(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        fig: Figure,
        idx: int,
        total: int,
    ) -> Tuple[List[Equation], Figure | None]:
        """Process a single figure through Mathpix, returning equations found and/or the original figure."""
        async with semaphore:
            logger.info("Mathpix: processing figure %d/%d (%s)", idx + 1, total, fig.figure_id)
            result = await self._call_mathpix(client, fig.image_base64)

            if result is None:
                logger.warning("  → Mathpix returned no result, keeping as figure")
                return [], fig

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
                # Split into individual equations (handles tables with many equations)
                eq_parts = self._split_equations(content)
                equations = []
                for eq_idx, eq_latex in enumerate(eq_parts):
                    # Filter junk: plain numbers, units, table markup, single variables
                    if self._is_junk_mathpix_equation(eq_latex):
                        logger.debug("  → Mathpix junk filtered: %s", eq_latex[:80])
                        continue
                    eq_number = self._extract_eq_number(eq_latex)
                    eq = Equation(
                        equation_id=f"eq_mathpix_{fig.figure_id}_{eq_idx}",
                        latex=eq_latex,
                        position=fig.position,
                        number=eq_number,
                    )
                    equations.append(eq)
                logger.info("  → %d equation(s) detected from figure %s", len(equations), fig.figure_id)
                # Figure contained equations — still keep it as a figure too
                # (it's an image with equations, not purely an equation)
                return equations, fig
            else:
                logger.info("  → not an equation (latex=%r, text=%r)",
                            latex[:60] if latex else "", text[:60] if text else "")
                return [], fig

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
                "Pre-filter: %d/%d figures skipped (no image data or too small), %d candidates remain",
                skipped, len(figures), len(candidates),
            )

        if not candidates:
            logger.info("No equation candidates after pre-filtering")
            return [], remaining_figures

        # Process candidates concurrently with a semaphore
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        equations: List[Equation] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = [
                self._process_single_figure(client, semaphore, fig, idx, len(candidates))
                for idx, fig in enumerate(candidates)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Mathpix task failed: %s", result)
                continue
            eqs, fig = result
            equations.extend(eqs)
            if fig is not None:
                remaining_figures.append(fig)

        logger.info(
            "Mathpix done: %d equations extracted from %d candidates (%d total figures, %d skipped)",
            len(equations), len(candidates), len(figures), skipped,
        )
        return equations, remaining_figures
