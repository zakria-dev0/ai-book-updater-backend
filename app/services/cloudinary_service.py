"""Cloudinary image upload/delete service for figure storage."""

import base64
import cloudinary
import cloudinary.uploader
import cloudinary.api
from typing import Optional
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_configured = False

# Cache for DB credentials so we don't query MongoDB on every upload
_db_credentials: dict | None = None


async def _load_db_credentials():
    """Load Cloudinary credentials from MongoDB settings collection."""
    global _db_credentials
    try:
        from app.database.connection import get_database
        db = get_database()
        doc = await db.settings.find_one({"key": "cloudinary_credentials"})
        if doc and doc.get("value"):
            _db_credentials = doc["value"]
            return _db_credentials
    except Exception:
        pass
    return None


def _get_credentials() -> tuple[str, str, str]:
    """Return (cloud_name, api_key, api_secret) from DB cache or .env."""
    if _db_credentials:
        return (
            _db_credentials.get("cloud_name", ""),
            _db_credentials.get("api_key", ""),
            _db_credentials.get("api_secret", ""),
        )
    return (
        settings.CLOUDINARY_CLOUD_NAME,
        settings.CLOUDINARY_API_KEY,
        settings.CLOUDINARY_API_SECRET,
    )


def _ensure_configured():
    """Lazy-configure Cloudinary on first use."""
    global _configured
    if _configured:
        return
    cloud_name, api_key, api_secret = _get_credentials()
    if not cloud_name:
        raise RuntimeError(
            "Cloudinary is not configured. "
            "Set your Cloudinary credentials in Admin → API Keys, "
            "or add CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and "
            "CLOUDINARY_API_SECRET to your .env file."
        )
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True,
    )
    _configured = True


class CloudinaryService:
    """Upload and manage figure images on Cloudinary."""

    FOLDER = "chapter-figures"

    @staticmethod
    def reset_config():
        """Reset cached config so next call picks up new credentials."""
        global _configured, _db_credentials
        _configured = False
        _db_credentials = None

    @staticmethod
    async def load_from_db():
        """Pre-load credentials from DB (call once at startup or after update)."""
        await _load_db_credentials()
        global _configured
        _configured = False  # Force re-configure with new creds

    @staticmethod
    def is_configured() -> bool:
        cloud_name, api_key, api_secret = _get_credentials()
        return bool(cloud_name and api_key and api_secret)

    @staticmethod
    def upload_figure(
        image_base64: str,
        document_id: str,
        figure_id: str,
    ) -> dict:
        """Upload a base64-encoded image to Cloudinary.

        Returns dict with ``url`` and ``public_id``.
        Raises on quota/auth errors so the caller can surface the message.
        """
        _ensure_configured()

        public_id = f"{CloudinaryService.FOLDER}/{document_id}/{figure_id}"

        # Detect format from base64 header or default to png
        data_uri = f"data:image/png;base64,{image_base64}"

        try:
            result = cloudinary.uploader.upload(
                data_uri,
                public_id=public_id,
                overwrite=True,
                resource_type="image",
            )
            url = result.get("secure_url") or result.get("url", "")
            logger.info(
                "Uploaded figure %s/%s to Cloudinary (%d bytes)",
                document_id, figure_id, result.get("bytes", 0),
            )
            return {
                "url": url,
                "public_id": result.get("public_id", public_id),
            }
        except Exception as e:
            error_msg = str(e)
            if "usage limit" in error_msg.lower() or "quota" in error_msg.lower():
                raise RuntimeError(
                    "Cloudinary storage quota exceeded. "
                    "Please upgrade your Cloudinary plan to continue processing documents."
                ) from e
            if "unauthorized" in error_msg.lower() or "401" in error_msg:
                raise RuntimeError(
                    "Cloudinary API key is invalid or expired. "
                    "Please update your Cloudinary credentials."
                ) from e
            if "stale request" in error_msg.lower() or "reported time" in error_msg.lower():
                raise RuntimeError(
                    "Cloudinary rejected the request due to a clock synchronization issue. "
                    "Your server's system clock is out of sync. "
                    "Please sync your system time (Windows: Settings → Time & Language → Sync now) "
                    "and ensure the correct timezone is set, then retry."
                ) from e
            raise

    @staticmethod
    def delete_figure(public_id: str) -> bool:
        """Delete a single image from Cloudinary by its public_id."""
        _ensure_configured()
        try:
            result = cloudinary.uploader.destroy(public_id, resource_type="image")
            return result.get("result") == "ok"
        except Exception as e:
            logger.error("Failed to delete Cloudinary image %s: %s", public_id, e)
            return False

    @staticmethod
    def delete_document_figures(document_id: str) -> int:
        """Delete all figures for a document (entire folder)."""
        _ensure_configured()
        prefix = f"{CloudinaryService.FOLDER}/{document_id}"
        try:
            cloudinary.api.delete_resources_by_prefix(prefix, resource_type="image")
            try:
                cloudinary.api.delete_folder(prefix)
            except Exception:
                pass  # Folder may already be empty/gone
            logger.info("Deleted Cloudinary folder: %s", prefix)
            return 1
        except Exception as e:
            logger.error("Failed to delete Cloudinary folder %s: %s", prefix, e)
            return 0

    @staticmethod
    def get_thumbnail_url(image_url: str, width: int = 300, height: int = 300) -> str:
        """Generate a Cloudinary transformation URL for a thumbnail.

        Uses Cloudinary's on-the-fly image transformation — no server-side
        processing needed.
        """
        if not image_url or "cloudinary" not in image_url:
            return image_url
        # Insert transformation before /upload/ or /image/upload/
        return image_url.replace("/upload/", f"/upload/w_{width},h_{height},c_fit/")
