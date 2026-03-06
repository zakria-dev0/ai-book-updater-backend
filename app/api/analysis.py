from fastapi import APIRouter, Depends, HTTPException, status, Query, Body, BackgroundTasks, Request
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from app.core.security import get_current_user_dep
from app.database.connection import get_database
from app.database.repositories.document_repo import DocumentRepository
from app.database.repositories.change_repo import ChangeRepository
from app.models.document import DocumentStatus
from app.models.change import ChangeStatus, ApprovalAction, FocusArea
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
    if document["user_id"] != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return document


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

    - **action**: 'approve_as_is' or 'reject'
    - **change_ids**: Optional list of IDs. If omitted, targets all pending changes.
    """
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

    When applying changes later (Milestone 4), the system will use
    user_edited_content if it exists, otherwise new_content.
    """
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
