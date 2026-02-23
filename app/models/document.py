from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ANALYZING = "analyzing"
    EXPORT_READY = "export_ready"
    ERROR = "error"

class DocumentType(str, Enum):
    DOCX = "docx"
    PDF = "pdf"

class Position(BaseModel):
    page: Optional[int] = None
    paragraph: Optional[int] = None
    line: Optional[int] = None

class Equation(BaseModel):
    equation_id: str
    latex: str
    raw_omml: Optional[str] = None  # Raw OMML XML for Mathpix/omml2latex conversion later
    image_base64: Optional[str] = None
    position: Position
    number: Optional[str] = None  # e.g., "(6-4)"

class Figure(BaseModel):
    figure_id: str
    caption: Optional[str] = None
    image_base64: str
    position: Position
    number: Optional[str] = None  # e.g., "Figure 6-5"

class Table(BaseModel):
    table_id: str
    caption: Optional[str] = None
    content: List[List[str]]  # 2D array of table cells
    position: Position
    number: Optional[str] = None  # e.g., "Table 6-1"

class ProcessingHistoryEntry(BaseModel):
    stage: str
    progress: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message: Optional[str] = None


class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    total_pages: int
    total_paragraphs: int
    total_equations: int
    total_figures: int
    total_tables: int

class Document(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    filename: str
    original_filename: str
    file_type: DocumentType
    file_path: str
    user_id: str
    status: DocumentStatus = DocumentStatus.UPLOADED
    
    # Extracted content
    text_content: Optional[str] = None
    equations: List[Equation] = []
    figures: List[Figure] = []
    tables: List[Table] = []
    metadata: Optional[DocumentMetadata] = None
    
    # Processing info
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Progress tracking
    progress: int = 0  # 0-100
    current_stage: Optional[str] = None
    processing_history: List[ProcessingHistoryEntry] = []
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "filename": "chapter-6.docx",
                "original_filename": "chapter-6.docx",
                "file_type": "docx",
                "status": "uploaded",
                "user_id": "user123"
            }
        }