from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from app.core.security import get_current_user_dep
from app.database.connection import get_database
from app.database.repositories.document_repo import DocumentRepository
from app.models.document import Document, DocumentType
from app.utils.file_handler import validate_file, save_upload_file
from app.core.config import settings
from app.core.logger import get_logger
from app.core.rate_limit import limiter
import os

logger = get_logger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])


@router.post(
    "/",
    summary="Upload a DOCX document",
    responses={
        200: {"description": "Document uploaded successfully"},
        400: {"description": "Invalid file type or size"},
        401: {"description": "Not authenticated"},
        500: {"description": "File save error"},
    },
)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="DOCX file to upload"),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Upload a DOCX document for processing.

    - Only `.docx` files are accepted (PDF is out of scope for Milestone 1)
    - Maximum file size: 50 MB
    - Returns a `document_id` to track processing status
    """
    # Validate extension
    is_valid, message = validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)

    # Validate size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
        )

    # Save to disk
    try:
        filename, file_path = await save_upload_file(file, current_user["email"])
    except Exception as e:
        logger.error("Failed to save file for user %s: %s", current_user["email"], str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )

    # Determine type (only docx accepted; guard already done above)
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_type = DocumentType.DOCX if file_ext == ".docx" else DocumentType.PDF

    # Create DB record
    document = Document(
        filename=filename,
        original_filename=file.filename,
        file_type=file_type,
        file_path=file_path,
        user_id=current_user["email"],
    )
    repo = DocumentRepository(db)
    doc_id = await repo.create(document.model_dump(by_alias=True, exclude={"id"}))

    logger.info("Document uploaded: %s by %s", file.filename, current_user["email"])
    return {
        "document_id": doc_id,
        "filename": file.filename,
        "file_size_bytes": file_size,
        "status": "uploaded",
        "uploaded_at": document.uploaded_at.isoformat(),
    }
