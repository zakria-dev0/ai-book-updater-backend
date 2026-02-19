from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from app.core.security import get_current_user
from app.database.connection import get_database
from app.models.document import Document, DocumentType
from app.utils.file_handler import validate_file, save_upload_file
from app.core.config import settings
import os

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Upload a DOCX or PDF document for processing
    
    - **file**: DOCX or PDF file to upload
    - Returns: Document metadata including document_id
    """
    
    # Validate file
    is_valid, message = validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
        )
    
    # Save file
    try:
        filename, file_path = await save_upload_file(file, current_user["email"])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Determine file type
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_type = DocumentType.DOCX if file_ext == ".docx" else DocumentType.PDF
    
    # Create document record
    document = Document(
        filename=filename,
        original_filename=file.filename,
        file_type=file_type,
        file_path=file_path,
        user_id=current_user["email"]
    )
    
    # Save to database
    result = await db.documents.insert_one(document.model_dump(by_alias=True, exclude={"id"}))
    document.id = str(result.inserted_id)
    
    return {
        "document_id": document.id,
        "filename": document.original_filename,
        "status": document.status,
        "uploaded_at": document.uploaded_at.isoformat()
    }