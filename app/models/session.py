from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ── Session Status (pipeline stages) ────────────────────────────────────────

class SessionStatus(str, Enum):
    CREATED = "created"
    RULES_CONFIRMED = "rules_confirmed"
    OUTLINE_EXTRACTED = "outline_extracted"
    DIAGNOSTIC_COMPLETE = "diagnostic_complete"
    OPPORTUNITIES_SELECTED = "opportunities_selected"
    RESEARCH_PLANNED = "research_planned"
    RESEARCHING = "researching"
    RESEARCH_DONE = "research_done"
    EVIDENCE_REVIEWED = "evidence_reviewed"
    PATCHES_GENERATED = "patches_generated"
    EDITS_APPLIED = "edits_applied"
    AUDIT_COMPLETE = "audit_complete"
    EXPORTED = "exported"
    ERROR = "error"


# ── Stage 1: Editorial Rules ────────────────────────────────────────────────

class EditorialRules(BaseModel):
    date_cutoff: Optional[str] = None  # e.g. "2020-01-01"
    preserve_sections: List[str] = []  # section headings to skip
    voice_preservation: bool = True
    citation_style: str = "inline"  # inline, footnote, endnote
    confidence_threshold: float = 0.5  # 0.0–1.0
    allowed_source_types: List[str] = ["government", "academic", "news", "technical", "commercial"]
    excluded_topics: List[str] = []
    max_sentence_change_pct: float = 80.0  # max % of sentence that can change


# ── Stage 2: Outline ────────────────────────────────────────────────────────

class OutlineItem(BaseModel):
    id: str
    text: str
    level: int  # heading level (1, 2, 3...)
    in_scope: bool = True
    paragraph_index: int


# ── Stage 3: Diagnostic Issue Types ─────────────────────────────────────────

class IssueType(str, Enum):
    OUTDATED_FACT = "outdated_fact"
    BROKEN_CITATION = "broken_citation"
    DATE_REFERENCE = "date_reference"
    CHANGED_STATUS = "changed_status"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Stage 3/4: Update Opportunity ───────────────────────────────────────────

class UpdateOpportunity(BaseModel):
    opportunity_id: str
    session_id: str
    section_ref: str
    original_sentence: str
    issue_type: IssueType
    severity: Severity
    confidence: float
    brief_reason: str
    selected: bool = False


# ── Stage 5: Research Plan ──────────────────────────────────────────────────

class ResearchPlan(BaseModel):
    plan_id: str
    opportunity_id: str
    session_id: str
    search_queries: List[str] = []
    target_sources: List[str] = []
    facts_to_verify: List[str] = []
    approved: bool = False


# ── Stage 6: Evidence Item ──────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    evidence_id: str
    research_plan_id: str
    session_id: str
    source_url: str
    source_title: str
    excerpt: str
    relevance_score: float = 0.0
    accepted: Optional[bool] = None  # None = not yet decided


# ── Stage 7: Patch ──────────────────────────────────────────────────────────

class PatchStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


class Patch(BaseModel):
    patch_id: str
    opportunity_id: str
    session_id: str
    original_sentence: str
    revised_sentence: str
    citation: str = ""
    rationale: str = ""
    confidence: float = 0.0
    change_pct: float = 0.0
    status: PatchStatus = PatchStatus.PENDING
    editor_revision: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    section_ref: str = ""


# ── Stage 9: Dated Statement ────────────────────────────────────────────────

class TriggerType(str, Enum):
    EXPLICIT_DATE = "explicit_date"
    RELATIVE_TEMPORAL = "relative_temporal"
    FUTURE_TENSE = "future_tense"
    PLANNED_STATUS = "planned_status"


class DatedStatement(BaseModel):
    statement_id: str
    session_id: str
    sentence: str
    trigger_word: str
    trigger_type: TriggerType
    section_ref: str
    risk: Severity
    resolved: bool = False
    resolution_note: Optional[str] = None  # "still_current", "flag_for_patch", "acceptable"


# ── Diagnostic Summary ──────────────────────────────────────────────────────

class DiagnosticSummary(BaseModel):
    total_issues: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    by_type: dict = {}  # issue_type -> count


# ── Main Session Model ──────────────────────────────────────────────────────

class EditorialSession(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    document_id: str
    user_id: str
    status: SessionStatus = SessionStatus.CREATED
    rules: Optional[EditorialRules] = None
    outline: List[OutlineItem] = []
    diagnostic: Optional[DiagnosticSummary] = None
    working_doc_path: Optional[str] = None  # path to modified copy
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None

    class Config:
        populate_by_name = True
