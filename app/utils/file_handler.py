import os
import uuid
import shutil
from uuid import uuid4
import aiofiles
from pathlib import Path
from fastapi import UploadFile, HTTPException
from app.core.config import settings

def ensure_directories():
    """Ensure all storage directories exist"""
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.PROCESSING_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)

def validate_file(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded file"""
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        return False, f"File type {file_ext} not allowed. Only {settings.ALLOWED_EXTENSIONS} are supported."
    
    # File size will be checked during upload
    return True, "Valid"

async def save_upload_file(file: UploadFile, user_id: str) -> tuple[str, str]:
    """Save uploaded file to storage"""
    ensure_directories()
    
    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid4()}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return unique_filename, file_path

def delete_file(file_path: str):
    """Delete a file from storage"""
    if os.path.exists(file_path):
        os.remove(file_path)

async def save_upload_file_from_bytes(
    file_content: bytes,
    original_filename: str,
    user_id: str
) -> tuple[str, str]:
    """Save file from bytes instead of stream - avoids multipart boundary issues"""
    file_ext = os.path.splitext(original_filename)[1].lower()
    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join("./storage/uploads", filename)
    
    os.makedirs("./storage/uploads", exist_ok=True)
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(file_content)
    
    return filename, file_path