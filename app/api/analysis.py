from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, BackgroundTasks, Request
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import re as _re
import uuid
from app.core.security import get_current_user_dep
from app.database.connection import get_database
from app.database.repositories.document_repo import DocumentRepository
from app.database.repositories.change_repo import ChangeRepository
from app.models.document import DocumentStatus
from app.models.change import ChangeStatus, ApprovalAction, FocusArea, ChangeType
from app.agents.orchestrator import run_analysis
from app.core.logger import get_logger
from app.core.rate_limit import limiter

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Analysis"])


# ------------------------------------------------------------------ #
# Request / Response models                                            #
# ------------------------------------------------------------------ #

class AnalyzeRequest(BaseModel):
    focus_areas: List[str] = Field(
        default=["all"],
        description=(
            "Focus areas to analyze. Options: missions, constellations, technology, "
            "statistics, companies, business_philosophy, historical_facts, all. "
            "Default: ['all'] (detect everything)."
        ),
    )


class ReviewRequest(BaseModel):
    action: str = Field(
        ...,
        description="Action: 'approve_as_is', 'approve_with_edit', or 'reject'",
    )
    note: str = Field(default="", description="Optional reviewer note")
    edited_content: Optional[str] = Field(
        default=None,
        description="User-edited content (required when action is 'approve_with_edit')",
    )


class BatchReviewRequest(BaseModel):
    action: str = Field(
        ...,
        description="Action to apply to all specified changes: 'approve_as_is' or 'reject'",
    )
    change_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific change IDs. If omitted, applies to ALL pending changes.",
    )
    note: str = Field(default="", description="Optional reviewer note")


class AiPromptRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="User prompt describing what content to generate (e.g., 'Add a paragraph about Starlink')",
        min_length=5,
        max_length=2000,
    )
    placement: str = Field(
        ...,
        description="Where to place the content: 'after_section', 'at_end', or 'replace_section'",
    )
    section_index: Optional[int] = Field(
        default=None,
        description="Index of the section for 'after_section' or 'replace_section' placement",
    )


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

async def _get_owned_document(
    document_id: str,
    current_user: dict,
    repo: DocumentRepository,
    lightweight: bool = True,
) -> dict:
    document = await repo.find_by_id(document_id, lightweight=lightweight)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    # Admins can access any document; regular users can only access their own
    if current_user.get("role") != "admin" and document["user_id"] != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return document


def _require_admin_role(current_user: dict):
    """Raise 403 if the current user is not an admin."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required. Only admins can approve or reject changes.",
        )


def _validate_focus_areas(focus_areas: List[str]) -> List[str]:
    """Validate that all provided focus areas are recognized."""
    valid = {fa.value for fa in FocusArea}
    invalid = [fa for fa in focus_areas if fa not in valid]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid focus areas: {invalid}. Valid options: {sorted(valid)}",
        )
    return focus_areas


# ------------------------------------------------------------------ #
# Trigger analysis                                                     #
# ------------------------------------------------------------------ #

async def _run_analysis_task(document_id: str, db, focus_areas: List[str]):
    """Background task: run the full AI analysis pipeline."""
    try:
        await run_analysis(document_id, db, focus_areas=focus_areas)
        logger.info("Background analysis completed for %s", document_id)
    except Exception as e:
        logger.error("Background analysis failed for %s: %s", document_id, e)
        # run_analysis already sets ERROR status on failure, but guard just in case
        try:
            doc_repo = DocumentRepository(db)
            await doc_repo.update_fields(document_id, {
                "status": DocumentStatus.ERROR,
                "error_message": f"Analysis failed: {str(e)}",
            })
        except Exception:
            pass


@router.post(
    "/{document_id}/analyze",
    summary="Trigger AI analysis pipeline",
    responses={
        200: {"description": "Analysis started"},
        400: {"description": "Document not processed yet or already analyzing"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
@limiter.limit("3/minute")
async def analyze_document(
    request: Request,
    document_id: str,
    background_tasks: BackgroundTasks,
    body: AnalyzeRequest = Body(default=AnalyzeRequest()),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Start the AI analysis pipeline in the background.

    Returns immediately. Poll `GET /documents/{id}/analysis/status` for progress.

    **focus_areas** (optional): Filter analysis to specific categories.
    Options: missions, constellations, technology, statistics, companies,
    business_philosophy, historical_facts, all (default).
    """
    doc_repo = DocumentRepository(db)
    # Need full doc to check text_content exists
    document = await _get_owned_document(document_id, current_user, doc_repo, lightweight=False)

    if document["status"] not in (
        DocumentStatus.COMPLETED, DocumentStatus.ERROR,
        DocumentStatus.EXPORT_READY, DocumentStatus.ANALYZING,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document must be processed before analysis. Current status: " + document["status"],
        )

    if not document.get("text_content"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no text content — process it first",
        )

    focus_areas = _validate_focus_areas(body.focus_areas)

    # Set status to 'analyzing' and reset progress BEFORE scheduling background task
    # This ensures the first poll sees the correct state
    await doc_repo.update_fields(document_id, {
        "status": DocumentStatus.ANALYZING,
        "progress": 0,
        "current_stage": "Starting analysis...",
    })

    background_tasks.add_task(_run_analysis_task, document_id, db, focus_areas)

    return {
        "document_id": document_id,
        "status": "analyzing",
        "message": "Analysis started. Poll GET /documents/{id}/analysis/status for progress.",
        "focus_areas": focus_areas,
    }


# ------------------------------------------------------------------ #
# Cancel analysis                                                      #
# ------------------------------------------------------------------ #

@router.post(
    "/{document_id}/analyze/cancel",
    summary="Cancel a running or stuck analysis",
    responses={
        200: {"description": "Analysis cancelled or already not analyzing"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def cancel_analysis(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Cancel a running or stuck analysis and reset the document status
    back to 'completed' so the user can access previous results or re-run.
    """
    doc_repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, doc_repo)

    current_status = document["status"]

    # If already not analyzing, return success with current status (no-op)
    if current_status != DocumentStatus.ANALYZING:
        logger.info(
            "Cancel requested for document %s but status is already '%s'",
            document_id, current_status,
        )
        return {
            "document_id": document_id,
            "status": current_status,
            "message": f"Document is not analyzing (current status: {current_status}). No action needed.",
        }

    await doc_repo.update_fields(document_id, {
        "status": DocumentStatus.COMPLETED,
    })

    logger.info("Analysis cancelled for document %s by user %s", document_id, current_user["email"])

    return {
        "document_id": document_id,
        "status": "completed",
        "message": "Analysis cancelled. Document status reset to completed.",
    }


# ------------------------------------------------------------------ #
# Analysis status                                                      #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/analysis/status",
    summary="Get analysis status",
    responses={
        200: {"description": "Current analysis status"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_analysis_status(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Check whether a document is being analyzed and get summary stats."""
    doc_repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, doc_repo)  # lightweight by default

    change_repo = ChangeRepository(db)
    changelog = await change_repo.find_changelog_by_document(document_id, summary_only=True)

    return {
        "document_id": document_id,
        "status": document["status"],
        "progress": document.get("progress", 0),
        "current_stage": document.get("current_stage", ""),
        "is_analyzed": changelog is not None,
        "style_profile": document.get("style_profile"),
        "analysis_summary": {
            "total_claims": changelog.get("total_claims", 0) if changelog else 0,
            "total_outdated": changelog.get("total_outdated", 0) if changelog else 0,
            "total_changes": changelog.get("total_changes", 0) if changelog else 0,
            "focus_areas": changelog.get("focus_areas", []) if changelog else [],
            "analyzed_at": changelog.get("created_at") if changelog else None,
        },
    }


# ------------------------------------------------------------------ #
# Claims                                                               #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/claims",
    summary="List identified claims",
    responses={
        200: {"description": "List of factual claims found in document"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_claims(
    document_id: str,
    outdated_only: bool = Query(default=False, description="Filter to only outdated claims"),
    focus_area: Optional[str] = Query(default=None, description="Filter claims by focus area"),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return all factual claims identified during analysis."""
    doc_repo = DocumentRepository(db)
    await _get_owned_document(document_id, current_user, doc_repo)

    change_repo = ChangeRepository(db)
    changelog = await change_repo.find_changelog_by_document(document_id)

    if not changelog:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No analysis results found — run POST /analyze first",
        )

    claims = changelog.get("claims", [])

    # Diagnostic: report if stored claims count differs from changelog metadata
    stored_total = changelog.get("total_claims", 0)
    actual_stored = len(claims)
    if stored_total != actual_stored and actual_stored > 0:
        logger.warning(
            "Claims count mismatch for document %s: changelog.total_claims=%d but "
            "actual stored claims=%d (claims_stored_separately=%s)",
            document_id, stored_total, actual_stored,
            changelog.get("claims_stored_separately", False),
        )

    if outdated_only:
        claims = [c for c in claims if c.get("is_outdated")]
    if focus_area:
        claims = [c for c in claims if c.get("focus_area") == focus_area]

    return {
        "document_id": document_id,
        "total_claims": len(claims),
        "claims": claims,
        "analysis_total_claims": stored_total,
        "analysis_total_outdated": changelog.get("total_outdated", 0),
    }


# ------------------------------------------------------------------ #
# Change proposals                                                     #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/changes",
    summary="List change proposals",
    responses={
        200: {"description": "Paginated list of change proposals"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def list_changes(
    document_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_filter: str = Query(default=None, description="Filter by status: pending, approved, rejected, applied"),
    change_type: str = Query(default=None, description="Filter by change type (e.g., constellation_update, mission_update)"),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return paginated change proposals for a document."""
    doc_repo = DocumentRepository(db)
    await _get_owned_document(document_id, current_user, doc_repo)

    change_repo = ChangeRepository(db)
    skip = (page - 1) * page_size

    if status_filter:
        changes = await change_repo.find_by_status(document_id, status_filter, skip, page_size)
        total = await change_repo.count_by_status(document_id, status_filter)
    else:
        changes = await change_repo.find_by_document(document_id, skip, page_size)
        total = await change_repo.count_by_document(document_id)

    # Client-side filter by change_type if specified
    if change_type:
        changes = [c for c in changes if c.get("change_type") == change_type]

    return {
        "document_id": document_id,
        "changes": changes,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
    }


# ------------------------------------------------------------------ #
# Batch review — approve all / reject all                              #
# NOTE: This must be defined BEFORE the {change_id} wildcard routes    #
# so that "batch" is not captured as a change_id.                      #
# ------------------------------------------------------------------ #

@router.put(
    "/{document_id}/changes/batch",
    summary="Batch approve or reject changes",
    responses={
        200: {"description": "Batch update completed"},
        400: {"description": "Invalid action"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def batch_review_changes(
    document_id: str,
    body: BatchReviewRequest = Body(...),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Bulk approve or reject multiple (or all pending) change proposals.
    **Admin only** — regular users cannot approve or reject changes.

    - **action**: 'approve_as_is' or 'reject'
    - **change_ids**: Optional list of IDs. If omitted, targets all pending changes.
    """
    _require_admin_role(current_user)
    doc_repo = DocumentRepository(db)
    await _get_owned_document(document_id, current_user, doc_repo)

    if body.action not in (ApprovalAction.APPROVE_AS_IS, ApprovalAction.REJECT):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch action must be 'approve_as_is' or 'reject'",
        )

    change_repo = ChangeRepository(db)

    if body.change_ids:
        target_ids = body.change_ids
    else:
        # Get all pending changes for this document
        pending = await change_repo.find_by_status(document_id, ChangeStatus.PENDING, skip=0, limit=5000)
        target_ids = [c.get("change_id") or c["id"] for c in pending if c.get("change_id") or c.get("id")]

    if not target_ids:
        return {"document_id": document_id, "updated": 0, "message": "No changes to update"}

    new_status = ChangeStatus.REJECTED if body.action == ApprovalAction.REJECT else ChangeStatus.APPROVED
    updated = await change_repo.batch_update_status(
        document_id=document_id,
        change_ids=target_ids,
        status=new_status.value,
        reviewer_note=body.note,
        approval_action=body.action,
    )

    logger.info("Batch %s %d changes for document %s", body.action, updated, document_id)
    return {
        "document_id": document_id,
        "action": body.action,
        "updated": updated,
        "message": f"Batch reviewed: {updated} changes {body.action}",
    }


@router.get(
    "/{document_id}/changes/{change_id}",
    summary="Get single change detail",
    responses={
        200: {"description": "Change proposal details"},
        403: {"description": "Not authorized"},
        404: {"description": "Change not found"},
    },
)
async def get_change(
    document_id: str,
    change_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Get full details of a single change proposal."""
    doc_repo = DocumentRepository(db)
    await _get_owned_document(document_id, current_user, doc_repo)

    change_repo = ChangeRepository(db)
    change = await change_repo.find_by_id(change_id)

    if not change:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Change not found")
    if change.get("document_id") != document_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Change not found")

    return change


# ------------------------------------------------------------------ #
# Review change — approve / approve_with_edit / reject                 #
# ------------------------------------------------------------------ #

@router.put(
    "/{document_id}/changes/{change_id}",
    summary="Review a change proposal (approve, edit, or reject)",
    responses={
        200: {"description": "Change status updated"},
        400: {"description": "Invalid action or missing edited_content"},
        403: {"description": "Not authorized"},
        404: {"description": "Change not found"},
    },
)
async def review_change(
    document_id: str,
    change_id: str,
    body: ReviewRequest = Body(...),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Review a change proposal with one of three actions:
    - **approve_as_is**: Accept the AI-generated new_content as-is
    - **approve_with_edit**: Accept with user modifications (provide edited_content)
    - **reject**: Reject this change proposal

    **Admin only** — regular users cannot approve or reject changes.

    When applying changes later (Milestone 4), the system will use
    user_edited_content if it exists, otherwise new_content.
    """
    _require_admin_role(current_user)
    doc_repo = DocumentRepository(db)
    await _get_owned_document(document_id, current_user, doc_repo)

    valid_actions = {a.value for a in ApprovalAction}
    if body.action not in valid_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action '{body.action}'. Must be one of: {sorted(valid_actions)}",
        )

    # Require edited_content for approve_with_edit
    if body.action == ApprovalAction.APPROVE_WITH_EDIT and not body.edited_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="edited_content is required when action is 'approve_with_edit'",
        )

    change_repo = ChangeRepository(db)
    change = await change_repo.find_by_id(change_id)

    if not change:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Change not found")
    if change.get("document_id") != document_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Change not found")

    # Determine new status
    if body.action == ApprovalAction.REJECT:
        new_status = ChangeStatus.REJECTED
    else:
        new_status = ChangeStatus.APPROVED

    # Update the change record
    await change_repo.update_status(
        change_id=change_id,
        status=new_status.value,
        reviewer_note=body.note,
        approval_action=body.action,
        user_edited_content=body.edited_content or "",
    )

    # Determine what content will be used
    final_content = body.edited_content if body.action == ApprovalAction.APPROVE_WITH_EDIT else change.get("new_content", "")

    return {
        "change_id": change_id,
        "status": new_status.value,
        "approval_action": body.action,
        "final_content": final_content if body.action != ApprovalAction.REJECT else None,
        "message": f"Change reviewed: {body.action}",
    }


# ------------------------------------------------------------------ #
# Changelog                                                            #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/changelog",
    summary="Get full change log",
    responses={
        200: {"description": "Full analysis changelog"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_changelog(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return the full change log for the most recent analysis run."""
    doc_repo = DocumentRepository(db)
    await _get_owned_document(document_id, current_user, doc_repo)

    change_repo = ChangeRepository(db)
    changelog = await change_repo.find_changelog_by_document(document_id)

    if not changelog:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No analysis results found — run POST /analyze first",
        )

    return changelog


@router.get(
    "/{document_id}/changelog/export",
    summary="Export change log as JSON",
    responses={
        200: {"description": "Exportable JSON changelog"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def export_changelog(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Export the full changelog including all claims, proposals, and sources."""
    doc_repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, doc_repo)

    change_repo = ChangeRepository(db)
    changelog = await change_repo.find_changelog_by_document(document_id)

    if not changelog:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No analysis results found — run POST /analyze first",
        )

    # Include document metadata for context
    return {
        "export_version": "2.0",
        "exported_at": datetime.utcnow().isoformat(),
        "document": {
            "document_id": document_id,
            "filename": document.get("original_filename", ""),
            "status": document.get("status", ""),
            "style_profile": document.get("style_profile"),
        },
        "changelog": changelog,
    }


# ------------------------------------------------------------------ #
# Section extraction helper                                            #
# ------------------------------------------------------------------ #

# Patterns that match typical section/chapter headings in textbooks
_HEADING_PATTERNS = [
    _re.compile(r'^(Chapter\s+\d+[\.:]\s*.+)$', _re.MULTILINE | _re.IGNORECASE),
    _re.compile(r'^(\d+\.\d+[\.:]\s*.+)$', _re.MULTILINE),
    _re.compile(r'^(\d+\.\d+\.\d+[\.:]\s*.+)$', _re.MULTILINE),
    _re.compile(r'^([A-Z][A-Z\s]{5,60})$', _re.MULTILINE),  # ALL-CAPS headings
    _re.compile(r'^((?:Section|Part)\s+\d+[\.:]\s*.+)$', _re.MULTILINE | _re.IGNORECASE),
]


def _extract_sections(text_content: str) -> list:
    """Extract section headings from document text content."""
    if not text_content:
        return []

    paragraphs = text_content.split('\n')
    sections = []
    seen_titles = set()

    for para_idx, para in enumerate(paragraphs):
        line = para.strip()
        if not line or len(line) > 120 or len(line) < 3:
            continue

        for pattern in _HEADING_PATTERNS:
            match = pattern.match(line)
            if match:
                title = match.group(1).strip()
                # Avoid duplicates and very short/generic headings
                if title not in seen_titles and len(title) > 3:
                    seen_titles.add(title)
                    sections.append({
                        "index": len(sections),
                        "title": title,
                        "paragraph_idx": para_idx,
                    })
                break

    # If no headings found, fall back to page-range sections
    if not sections:
        total_paras = len(paragraphs)
        chunk_size = max(total_paras // 10, 20)
        for i in range(0, total_paras, chunk_size):
            end = min(i + chunk_size, total_paras)
            # Use first non-empty line as preview
            preview = ""
            for j in range(i, min(i + 5, end)):
                if paragraphs[j].strip():
                    preview = paragraphs[j].strip()[:60]
                    break
            sections.append({
                "index": len(sections),
                "title": f"Section {len(sections) + 1} — {preview}..." if preview else f"Section {len(sections) + 1}",
                "paragraph_idx": i,
            })

    return sections


# ------------------------------------------------------------------ #
# GET sections — extract document sections for AI prompt placement     #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/sections",
    summary="Get document sections",
    responses={
        200: {"description": "List of detected sections/headings"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_document_sections(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Extract and return section headings from the document for AI prompt placement."""
    doc_repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, doc_repo, lightweight=False)

    text_content = document.get("text_content", "")
    if not text_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no text content",
        )

    sections = _extract_sections(text_content)
    return {"document_id": document_id, "sections": sections, "total": len(sections)}


# ------------------------------------------------------------------ #
# POST AI prompt — generate new content matching document style        #
# ------------------------------------------------------------------ #

_AI_PROMPT_SYSTEM = """\
You are an expert editor for a university-level aerospace engineering textbook.
Your task is to write NEW content for the textbook based on a user prompt.

CRITICAL REQUIREMENT: The new content MUST be INDISTINGUISHABLE from the rest of the textbook.
You will be given the document's style profile and surrounding context to match.

STYLE RULES:
- Match the exact tone, grade level, and technical depth of the document
- Use active voice predominantly (less than 10% passive)
- Average sentence length ~20 words, mix short (10-15) and long (25-35) sentences
- Define acronyms on first use: "Low Earth Orbit (LEO)"
- Include specific numbers with dual units: "550 km (342 mi)"
- Spell out numbers under ten; use numerals for 10+
- Include exact dates when known
- Use transitional phrases to connect ideas naturally
- Write flowing prose — NO bullet points
- End on a SPECIFIC FACT, never editorial commentary
- Never use "This highlights...", "This underscores...", "This demonstrates..."
- Never use hedging: "may have", "could potentially"
- Write 3 to 5 sentences MAXIMUM — concise and focused, no padding
- Include at least 2 specific data points

OUTPUT FORMAT:
Return a JSON object:
{{
  "new_content": "The generated paragraph(s) matching the textbook style",
  "summary": "One-sentence summary of what was added"
}}

Return ONLY valid JSON — no markdown fences, no extra text.
"""

_AI_PROMPT_USER_TEMPLATE = """\
TODAY'S DATE: {today_date}

DOCUMENT STYLE PROFILE:
{style_profile}

SURROUNDING CONTEXT (from the document near the target location):
\"\"\"{context}\"\"\"

USER REQUEST: {prompt}

PLACEMENT: {placement_description}

Write the new content matching the textbook's exact writing style.
The content should flow naturally with the surrounding text.
Return valid JSON only.
"""


@router.post(
    "/{document_id}/ai-prompt",
    summary="Generate AI content from user prompt",
    responses={
        200: {"description": "AI-generated change proposal created"},
        400: {"description": "Invalid request"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
@limiter.limit("5/minute")
async def generate_ai_prompt(
    request: Request,
    document_id: str,
    body: AiPromptRequest = Body(...),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Generate new content for the document based on a user prompt.
    The AI matches the document's writing style and creates a change proposal
    that can be reviewed (approved/rejected) like any other change.
    """
    from openai import AsyncOpenAI
    from app.core.config import settings
    import json

    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenAI API key not configured",
        )

    if body.placement not in ("after_section", "at_end", "replace_section"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="placement must be 'after_section', 'at_end', or 'replace_section'",
        )

    if body.placement in ("after_section", "replace_section") and body.section_index is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="section_index is required for 'after_section' and 'replace_section' placement",
        )

    doc_repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, doc_repo, lightweight=False)

    text_content = document.get("text_content", "")
    if not text_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no text content",
        )

    paragraphs = text_content.split('\n')
    sections = _extract_sections(text_content)

    # Determine context and placement info
    target_para_idx = len(paragraphs) - 1  # default: end
    old_content = ""
    placement_description = ""

    if body.placement == "at_end":
        target_para_idx = len(paragraphs) - 1
        placement_description = "Add this content at the END of the document, after all existing content."
        # Get last few paragraphs as context
        context_start = max(0, len(paragraphs) - 5)
        context = "\n".join(p for p in paragraphs[context_start:] if p.strip())

    elif body.placement == "after_section":
        if body.section_index is not None and body.section_index < len(sections):
            section = sections[body.section_index]
            target_para_idx = section["paragraph_idx"]
            placement_description = f"Add this content AFTER the section: \"{section['title']}\""
            # Get context around this section
            ctx_start = max(0, target_para_idx - 2)
            ctx_end = min(len(paragraphs), target_para_idx + 8)
            context = "\n".join(p for p in paragraphs[ctx_start:ctx_end] if p.strip())
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid section_index")

    elif body.placement == "replace_section":
        if body.section_index is not None and body.section_index < len(sections):
            section = sections[body.section_index]
            target_para_idx = section["paragraph_idx"]
            # Find the content until the next section
            next_para_idx = len(paragraphs)
            if body.section_index + 1 < len(sections):
                next_para_idx = sections[body.section_index + 1]["paragraph_idx"]
            old_content = "\n".join(
                p for p in paragraphs[target_para_idx:next_para_idx] if p.strip()
            )
            placement_description = f"REPLACE the section: \"{section['title']}\" with updated content."
            # Context: before and after the section being replaced
            ctx_before = "\n".join(p for p in paragraphs[max(0, target_para_idx - 3):target_para_idx] if p.strip())
            ctx_after = "\n".join(p for p in paragraphs[next_para_idx:min(len(paragraphs), next_para_idx + 3)] if p.strip())
            context = f"BEFORE:\n{ctx_before}\n\nSECTION BEING REPLACED:\n{old_content}\n\nAFTER:\n{ctx_after}"
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid section_index")

    # Build style profile string
    style_profile = document.get("style_profile", {})
    if isinstance(style_profile, dict) and style_profile:
        style_str = "\n".join(f"- {k}: {v}" for k, v in style_profile.items())
    else:
        style_str = "- Grade: Undergraduate STEM textbook\n- Tone: Authoritative yet accessible\n- Voice: Active preferred, ~20 words/sentence"

    # Call GPT-4o
    user_prompt = _AI_PROMPT_USER_TEMPLATE.format(
        today_date=datetime.now().strftime("%Y-%m-%d"),
        style_profile=style_str,
        context=context[:3000],  # Limit context size
        prompt=body.prompt,
        placement_description=placement_description,
    )

    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=settings.GPT_MODEL,
            messages=[
                {"role": "system", "content": _AI_PROMPT_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            timeout=120.0,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        new_content = data.get("new_content", "")
        summary = data.get("summary", "AI-generated content")

        if not new_content or len(new_content) < 20:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI failed to generate sufficient content. Please try again.",
            )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI returned invalid response. Please try again.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("AI prompt generation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI generation failed: {str(e)[:200]}",
        )

    # Create change proposal
    change_data = {
        "change_id": f"change_{uuid.uuid4().hex[:12]}",
        "document_id": document_id,
        "claim_id": f"ai_prompt_{uuid.uuid4().hex[:8]}",
        "old_content": old_content if body.placement == "replace_section" else f"[AI Prompt: {body.prompt[:100]}]",
        "new_content": new_content,
        "change_type": ChangeType.AI_PROMPT.value,
        "confidence": "high",
        "sources": [],
        "paragraph_idx": target_para_idx,
        "page": None,
        "status": ChangeStatus.PENDING.value,
        "created_at": datetime.utcnow(),
        "reviewer_note": f"AI Prompt: {body.prompt}",
        "core_claim_status": None,
        "approval_action": None,
        "user_edited_content": None,
        "ai_prompt_metadata": {
            "prompt": body.prompt,
            "placement": body.placement,
            "section_index": body.section_index,
            "summary": summary,
        },
    }

    change_repo = ChangeRepository(db)
    await change_repo.create(change_data)

    logger.info(
        "AI prompt change created for document %s by user %s: %s",
        document_id, current_user["email"], body.prompt[:80],
    )

    # Build a clean response dict (MongoDB insert adds _id ObjectId in-place)
    response_change = {
        "change_id": change_data["change_id"],
        "document_id": change_data["document_id"],
        "claim_id": change_data["claim_id"],
        "old_content": change_data["old_content"],
        "new_content": change_data["new_content"],
        "change_type": change_data["change_type"],
        "confidence": change_data["confidence"],
        "sources": change_data["sources"],
        "paragraph_idx": change_data["paragraph_idx"],
        "page": change_data["page"],
        "status": change_data["status"],
        "created_at": change_data["created_at"].isoformat(),
        "reviewer_note": change_data["reviewer_note"],
        "core_claim_status": change_data["core_claim_status"],
        "approval_action": change_data["approval_action"],
        "user_edited_content": change_data["user_edited_content"],
    }

    return {
        "change": response_change,
        "message": f"AI content generated: {summary}",
    }
