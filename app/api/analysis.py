from fastapi import APIRouter, Depends, HTTPException, status, Query
from datetime import datetime
from app.core.security import get_current_user_dep
from app.database.connection import get_database
from app.database.repositories.document_repo import DocumentRepository
from app.database.repositories.change_repo import ChangeRepository
from app.models.document import DocumentStatus
from app.models.change import ChangeStatus
from app.agents.orchestrator import run_analysis
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Analysis"])


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

async def _get_owned_document(
    document_id: str,
    current_user: dict,
    repo: DocumentRepository,
) -> dict:
    document = await repo.find_by_id(document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if document["user_id"] != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return document


# ------------------------------------------------------------------ #
# Trigger analysis                                                     #
# ------------------------------------------------------------------ #

@router.post(
    "/{document_id}/analyze",
    summary="Trigger AI analysis pipeline",
    responses={
        200: {"description": "Analysis completed"},
        400: {"description": "Document not processed yet or already analyzing"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
        500: {"description": "Analysis failed"},
    },
)
async def analyze_document(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Run the full AI analysis pipeline on a processed document:
    1. Content Analysis — identify factual claims via GPT-4o
    2. Research — search for updated info via Tavily
    3. Proposals — generate change suggestions
    4. Validation — quality checks on proposals
    """
    doc_repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, doc_repo)

    if document["status"] == DocumentStatus.ANALYZING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is already being analyzed",
        )

    if document["status"] not in (DocumentStatus.COMPLETED, DocumentStatus.ERROR):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document must be processed before analysis. Current status: " + document["status"],
        )

    if not document.get("text_content"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has no text content — process it first",
        )

    try:
        changelog = await run_analysis(document_id, db)
        return {
            "document_id": document_id,
            "status": "completed",
            "message": "AI analysis completed successfully",
            "summary": {
                "total_claims": changelog.total_claims,
                "total_outdated": changelog.total_outdated,
                "total_changes": changelog.total_changes,
            },
        }
    except Exception as e:
        logger.error("Analysis failed for %s: %s", document_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


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
    document = await _get_owned_document(document_id, current_user, doc_repo)

    change_repo = ChangeRepository(db)
    changelog = await change_repo.find_changelog_by_document(document_id)

    return {
        "document_id": document_id,
        "status": document["status"],
        "is_analyzed": changelog is not None,
        "analysis_summary": {
            "total_claims": changelog.get("total_claims", 0) if changelog else 0,
            "total_outdated": changelog.get("total_outdated", 0) if changelog else 0,
            "total_changes": changelog.get("total_changes", 0) if changelog else 0,
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
    if outdated_only:
        claims = [c for c in claims if c.get("is_outdated")]

    return {
        "document_id": document_id,
        "total_claims": len(claims),
        "claims": claims,
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

    return {
        "document_id": document_id,
        "changes": changes,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size if total else 0,
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


@router.put(
    "/{document_id}/changes/{change_id}",
    summary="Approve or reject a change",
    responses={
        200: {"description": "Change status updated"},
        400: {"description": "Invalid status"},
        403: {"description": "Not authorized"},
        404: {"description": "Change not found"},
    },
)
async def review_change(
    document_id: str,
    change_id: str,
    action: str = Query(..., description="Action: approve or reject"),
    note: str = Query(default="", description="Optional reviewer note"),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Approve or reject a change proposal."""
    doc_repo = DocumentRepository(db)
    await _get_owned_document(document_id, current_user, doc_repo)

    if action not in ("approve", "reject"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Action must be 'approve' or 'reject'",
        )

    change_repo = ChangeRepository(db)
    change = await change_repo.find_by_id(change_id)

    if not change:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Change not found")
    if change.get("document_id") != document_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Change not found")

    new_status = ChangeStatus.APPROVED if action == "approve" else ChangeStatus.REJECTED
    await change_repo.update_status(change_id, new_status.value, note)

    return {
        "change_id": change_id,
        "status": new_status.value,
        "message": f"Change {action}d successfully",
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
        "export_version": "1.0",
        "exported_at": datetime.utcnow().isoformat(),
        "document": {
            "document_id": document_id,
            "filename": document.get("original_filename", ""),
            "status": document.get("status", ""),
        },
        "changelog": changelog,
    }
