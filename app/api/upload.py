from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from app.core.security import get_current_user
from app.database.connection import get_database
from app.models.document import Document, DocumentType
from app.utils.file_handler import validate_file, save_upload_file, save_upload_file_from_bytes
from app.core.config import settings
import os 

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    # Step 1: Validate filename/extension FIRST before reading anything
    is_valid, message = validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    
    # Step 2: Read entire file content into memory first
    # This avoids multipart boundary issues and lets us validate size
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )
    
    # Step 3: Check file size on actual bytes (not stream seek - that caused your boundary error)
    file_size = len(file_content)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty"
        )
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / 1024 / 1024:.1f}MB"
        )
    
    # Step 4: Save file to disk using the content we already read
    try:
        filename, file_path = await save_upload_file_from_bytes(
            file_content, file.filename, current_user["email"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Step 5: Only save to DB after file is confirmed saved on disk
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_type = DocumentType.DOCX if file_ext == ".docx" else DocumentType.PDF
    
    document = Document(
        filename=filename,
        original_filename=file.filename,
        file_type=file_type,
        file_path=file_path,
        user_id=current_user["email"]
    )
    
    try:
        result = await db.documents.insert_one(document.model_dump(by_alias=True, exclude={"id"}))
        document.id = str(result.inserted_id)
    except Exception as e:
        # If DB save fails, clean up the file we just saved
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save document record: {str(e)}"
        )
    
    return {
        "document_id": document.id,
        "filename": document.original_filename,
        "status": document.status,
        "uploaded_at": document.uploaded_at.isoformat()
    }