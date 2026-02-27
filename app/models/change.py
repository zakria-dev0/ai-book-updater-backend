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
    CONSTELLATION_UPDATE = "constellation_update"
    STATISTICS_UPDATE = "statistics_update"
    SYSTEM_UPDATE = "system_update"
    REGULATION_UPDATE = "regulation_update"
    BUSINESS_MODEL_UPDATE = "business_model_update"
    HISTORICAL_CORRECTION = "historical_correction"


class ApprovalAction(str, Enum):
    APPROVE_AS_IS = "approve_as_is"
    APPROVE_WITH_EDIT = "approve_with_edit"
    REJECT = "reject"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CoreClaimStatus(str, Enum):
    FALSE = "false"
    OUTDATED = "outdated"
    INCOMPLETE = "incomplete"
    STILL_TRUE = "still_true"


# --- Focus Area definitions ---
class FocusArea(str, Enum):
    MISSIONS = "missions"
    CONSTELLATIONS = "constellations"
    TECHNOLOGY = "technology"
    STATISTICS = "statistics"
    COMPANIES = "companies"
    BUSINESS_PHILOSOPHY = "business_philosophy"
    HISTORICAL_FACTS = "historical_facts"
    ALL = "all"


# Mapping: claim_type (from GPT) -> which FocusArea it belongs to
CLAIM_TYPE_TO_FOCUS_AREA = {
    "mission": FocusArea.MISSIONS,
    "constellation": FocusArea.CONSTELLATIONS,
    "technology": FocusArea.TECHNOLOGY,
    "statistic": FocusArea.STATISTICS,
    "company_info": FocusArea.COMPANIES,
    "business_philosophy": FocusArea.BUSINESS_PHILOSOPHY,
    "date": FocusArea.HISTORICAL_FACTS,
    "historical": FocusArea.HISTORICAL_FACTS,
    "policy": FocusArea.BUSINESS_PHILOSOPHY,
    "regulation": FocusArea.BUSINESS_PHILOSOPHY,
    "citation": FocusArea.TECHNOLOGY,
}


# --- Writing Style Profile ---
class StyleProfile(BaseModel):
    grade_level: str = "college_senior"  # college_freshman/junior/senior/graduate
    technical_depth: str = "intermediate"  # introductory/intermediate/advanced
    tone: str = "formal_academic"  # conversational/formal_academic/authoritative
    sentence_complexity: str = "moderate"  # simple/moderate/complex
    terminology_level: str = "technical"  # basic/technical/highly_technical
    avg_sentence_length: int = 25
    passive_voice_usage: str = "moderate"  # rare/moderate/frequent


class FactualClaim(BaseModel):
    claim_id: str
    text: str
    claim_type: str  # e.g. "statistic", "date", "company_info", "mission", "technology", "constellation"
    paragraph_idx: int
    page: Optional[int] = None
    entities: List[str] = []
    temporal_refs: List[str] = []
    is_outdated: bool = False
    focus_area: Optional[str] = None  # resolved FocusArea value after classification


class ResearchResult(BaseModel):
    source_url: str
    source_title: str
    source_type: str = "commercial"  # government, academic, news, technical, commercial
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
    core_claim_status: Optional[str] = None  # false / outdated / incomplete / still_true
    # User edit fields
    user_edited_content: Optional[str] = None
    approval_action: Optional[str] = None  # approve_as_is / approve_with_edit / reject


class ChangeLog(BaseModel):
    log_id: str
    document_id: str
    total_claims: int = 0
    total_outdated: int = 0
    total_changes: int = 0
    claims: List[FactualClaim] = []
    changes: List[ChangeProposal] = []
    focus_areas: List[str] = []  # which focus areas were used
    style_profile: Optional[dict] = None  # document style profile used
    created_at: datetime = Field(default_factory=datetime.utcnow)
