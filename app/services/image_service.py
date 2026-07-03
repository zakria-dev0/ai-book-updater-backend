"""Image processing utilities using Pillow."""

import base64
import io
from typing import Tuple
from PIL import Image
from app.core.logger import get_logger

logger = get_logger(__name__)

VALID_FORMATS = {"PNG", "JPEG", "GIF", "TIFF", "BMP", "WEBP"}
MIN_DIMENSION = 50


class ImageService:
    """Static utility methods for image processing."""

    @staticmethod
    def generate_thumbnail(
        image_base64: str,
        max_size: Tuple[int, int] = (300, 300),
    ) -> str:
        """
        Create a thumbnail from a base64-encoded image.

        Returns base64-encoded PNG thumbnail.
        """
        try:
            image_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_data))
            img.thumbnail(max_size, Image.LANCZOS)

            # Convert to RGB if necessary (e.g., RGBA or palette modes)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="PNG", optimize=True)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error("Thumbnail generation failed: %s", e)
            return image_base64  # Return original on failure

    @staticmethod
    def get_image_metadata(image_base64: str) -> dict:
        """
        Extract metadata from a base64-encoded image.

        Returns dict with width, height, format, size_bytes, mode.
        """
        try:
            raw = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(raw))
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format or "UNKNOWN",
                "size_bytes": len(raw),
                "mode": img.mode,
            }
        except Exception as e:
            logger.error("Image metadata extraction failed: %s", e)
            return {
                "width": 0,
                "height": 0,
                "format": "UNKNOWN",
                "size_bytes": len(base64.b64decode(image_base64)) if image_base64 else 0,
                "mode": "UNKNOWN",
            }

    @staticmethod
    def validate_image(image_base64: str) -> Tuple[bool, str]:
        """
        Validate a base64-encoded image.

        Checks minimum resolution and valid format.
        Returns (is_valid, error_message).
        """
        try:
            raw = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(raw))

            fmt = (img.format or "").upper()
            if fmt and fmt not in VALID_FORMATS:
                return False, f"Unsupported image format: {fmt}. Supported: {', '.join(sorted(VALID_FORMATS))}"

            if img.width < MIN_DIMENSION or img.height < MIN_DIMENSION:
                return False, f"Image too small ({img.width}x{img.height}). Minimum: {MIN_DIMENSION}x{MIN_DIMENSION}"

            return True, ""
        except Exception as e:
            return False, f"Invalid image data: {str(e)}"

    @staticmethod
    def compress_for_storage(
        image_data: bytes,
        max_dimension: int = 1600,
        jpeg_quality: int = 85,
        skip_below_bytes: int = 300_000,
    ) -> bytes:
        """
        Downscale/re-encode raw image bytes before they get base64-stored in MongoDB.

        Images whose longest side already fits within max_dimension and whose
        size is already under skip_below_bytes are returned unchanged. Images
        with transparency/palette data (diagrams, screenshots) are kept as
        optimized PNG; everything else is re-encoded as JPEG, since that is
        where most of the size reduction comes from on photo-like scans.
        Falls back to the original bytes if compression fails or doesn't help
        (mirrors the "keep original on failure" behavior used elsewhere in
        this module).
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            needs_resize = max(img.width, img.height) > max_dimension
            if not needs_resize and len(image_data) < skip_below_bytes:
                return image_data

            if needs_resize:
                img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

            buffer = io.BytesIO()
            if img.mode in ("P", "RGBA", "LA"):
                if img.mode not in ("RGB", "RGBA", "L", "P"):
                    img = img.convert("RGBA")
                img.save(buffer, format="PNG", optimize=True)
            else:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)

            compressed = buffer.getvalue()
            return compressed if len(compressed) < len(image_data) else image_data
        except Exception as e:
            logger.error("Image compression failed, keeping original: %s", e)
            return image_data

    @staticmethod
    def convert_format(
        image_base64: str,
        target_format: str = "PNG",
    ) -> str:
        """
        Convert a base64-encoded image to a different format.

        Returns base64-encoded image in the target format.
        """
        try:
            raw = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(raw))

            if img.mode not in ("RGB", "L") and target_format.upper() in ("JPEG", "BMP"):
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format=target_format.upper())
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error("Image format conversion failed: %s", e)
            return image_base64
