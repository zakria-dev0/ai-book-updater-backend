from fastapi import APIRouter, Depends, HTTPException, status
from app.core.security import get_current_user
from app.database.connection import get_database
from app.models.document import DocumentStatus
from app.services.document_service import DOCXParser
from bson import ObjectId
from datetime import datetime

router = APIRouter(prefix="/documents", tags=["processing"])

@router.post("/{document_id}/process")
async def process_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Start processing a document to extract content
    
    - **document_id**: ID of the uploaded document
    - Returns: Processing status
    """
    
    # Fetch document from database
    document = await db.documents.find_one({"_id": ObjectId(document_id)})
    
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    # Check if user owns the document
    if document["user_id"] != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    
    # Only allow DOCX files
    if document["file_type"] != "docx":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only DOCX files are supported"
        )
    
    # Update status to processing
    await db.documents.update_one(
        {"_id": ObjectId(document_id)},
        {
            "$set": {
                "status": DocumentStatus.PROCESSING,
                "processing_started_at": datetime.utcnow(),
                "progress": 10,
                "current_stage": "document_extraction"
            }
        }
    )
    
    try:
        parser = DOCXParser(document["file_path"])
        
        # Extract content
        text, equations, figures, tables, metadata = parser.parse()
        
        # Update document with extracted content
        await db.documents.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "text_content": text,
                    "equations": [eq.model_dump() for eq in equations],
                    "figures": [fig.model_dump() for fig in figures],
                    "tables": [tbl.model_dump() for tbl in tables],
                    "metadata": metadata.model_dump(),
                    "status": DocumentStatus.COMPLETED,
                    "processing_completed_at": datetime.utcnow(),
                    "progress": 100,
                    "current_stage": "completed"
                }
            }
        )
        
        return {
            "document_id": document_id,
            "status": "completed",
            "message": "Document processing completed successfully"
        }
        
    except Exception as e:
        # Update status to error
        await db.documents.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": DocumentStatus.ERROR,
                    "error_message": str(e),
                    "progress": 0
                }
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    document = await db.documents.find_one({"_id": ObjectId(document_id)})
    
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    if document["user_id"] != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    
    return {
        "document_id": document_id,
        "status": document["status"],
        "progress": document.get("progress", 0),
        "current_stage": document.get("current_stage", ""),
        "message": document.get("error_message", "Processing..."),
        "changes_count": 0
    }


@router.get("/")
async def list_documents(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    cursor = db.documents.find({"user_id": current_user["email"]})
    documents = await cursor.to_list(length=100)
    
    for doc in documents:
        doc["id"] = str(doc.pop("_id"))
    
    return {"documents": documents}


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    document = await db.documents.find_one({"_id": ObjectId(document_id)})
    
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    if document["user_id"] != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    
    document["id"] = str(document.pop("_id"))
    return document
