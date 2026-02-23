from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ChangeStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


class ChangeType(str, Enum):
    DATA_UPDATE = "data_update"
    TECH_UPDATE = "tech_update"
    MISSION_UPDATE = "mission_update"
    COMPANY_UPDATE = "company_update"
    REGULATORY_UPDATE = "regulatory_update"
    IMAGE_UPDATE = "image_update"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FactualClaim(BaseModel):
    claim_id: str
    text: str
    claim_type: str  # e.g. "statistic", "date", "company_info", "mission", "technology"
    paragraph_idx: int
    page: Optional[int] = None
    entities: List[str] = []
    temporal_refs: List[str] = []
    is_outdated: bool = False


class ResearchResult(BaseModel):
    source_url: str
    source_title: str
    source_type: str = "commercial"  # government, academic, commercial
    published_date: Optional[str] = None
    author: Optional[str] = None
    snippet: str = ""
    relevance_score: float = 0.0


class ChangeProposal(BaseModel):
    change_id: str
    document_id: str
    claim_id: str
    old_content: str
    new_content: str
    change_type: ChangeType = ChangeType.DATA_UPDATE
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    sources: List[ResearchResult] = []
    paragraph_idx: int
    page: Optional[int] = None
    status: ChangeStatus = ChangeStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    reviewer_note: Optional[str] = None


class ChangeLog(BaseModel):
    log_id: str
    document_id: str
    total_claims: int = 0
    total_outdated: int = 0
    total_changes: int = 0
    claims: List[FactualClaim] = []
    changes: List[ChangeProposal] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
