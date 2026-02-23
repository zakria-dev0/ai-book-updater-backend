from fastapi import APIRouter, Depends, HTTPException, status, Query
from bson import ObjectId
from datetime import datetime
from app.core.security import get_current_user_dep
from app.database.connection import get_database
from app.database.repositories.document_repo import DocumentRepository
from app.models.document import DocumentStatus
from app.services.document_service import DOCXParser
from app.services.equation_service import MathpixService
from app.utils.file_handler import delete_file
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

async def _update_stage(
    repo: DocumentRepository,
    document_id: str,
    stage: str,
    progress: int,
    message: str = "",
):
    """Write a progress snapshot to the DB and append to processing_history."""
    entry = {
        "stage": stage,
        "progress": progress,
        "timestamp": datetime.utcnow(),
        "message": message,
    }
    await repo.update_fields(document_id, {
        "current_stage": stage,
        "progress": progress,
    })
    await repo.push_history_entry(document_id, entry)
    logger.info("Document %s – stage: %s (%d%%)", document_id, stage, progress)


async def _get_owned_document(
    document_id: str,
    current_user: dict,
    repo: DocumentRepository,
) -> dict:
    """Fetch document and verify ownership; raises 404/403 as appropriate."""
    document = await repo.find_by_id(document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if document["user_id"] != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return document


# ------------------------------------------------------------------ #
# Process                                                              #
# ------------------------------------------------------------------ #

@router.post(
    "/{document_id}/process",
    summary="Start processing a document",
    responses={
        200: {"description": "Processing completed"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
        500: {"description": "Processing failed"},
    },
)
async def process_document(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Extract text, figures, and tables from an uploaded DOCX document.

    Progress is written to the database at every stage so callers can
    poll `GET /documents/{id}/status` for real-time updates.
    """
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    if document["status"] == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is already being processed",
        )

    try:
        # ── Stage 1: Initializing ──────────────────────────────────────
        await repo.update_fields(document_id, {
            "status": DocumentStatus.PROCESSING,
            "processing_started_at": datetime.utcnow(),
            "processing_history": [],
        })
        await _update_stage(repo, document_id, "Initializing", 5, "Starting document processing")

        if document["file_type"] != "docx":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only DOCX files are supported",
            )

        parser = DOCXParser(document["file_path"], use_nougat=False)

        # ── Stage 2: Extract text ──────────────────────────────────────
        await _update_stage(repo, document_id, "Extracting text", 20, "Extracting text content")
        text = parser._extract_text()

        # ── Stage 3: Extract figures ───────────────────────────────────
        await _update_stage(repo, document_id, "Extracting figures", 45, "Extracting embedded images")
        figures = parser._extract_figures()

        # ── Stage 4: Extract tables ────────────────────────────────────
        await _update_stage(repo, document_id, "Extracting tables", 65, "Extracting tables")
        tables = parser._extract_tables()

        # ── Stage 5: Extract OMML equations ─────────────────────────────
        await _update_stage(repo, document_id, "Extracting equations", 70, "Extracting OMML equations and converting to LaTeX")
        equations = parser._extract_equations()

        # ── Stage 5b: Extract equations from images via Mathpix ──────
        if settings.MATHPIX_APP_ID and settings.MATHPIX_APP_KEY:
            await _update_stage(
                repo, document_id,
                "Extracting equations from images", 78,
                "Using Mathpix OCR to detect equations in figures",
            )
            mathpix = MathpixService()
            mathpix_eqs, figures = await mathpix.extract_equations_from_figures(figures)
            equations.extend(mathpix_eqs)
            logger.info("Mathpix extracted %d equations from figures", len(mathpix_eqs))
        else:
            logger.info("Mathpix not configured — skipping image equation extraction")

        # ── Stage 6: Generate metadata ─────────────────────────────────
        await _update_stage(repo, document_id, "Generating metadata", 85, "Generating document metadata")
        parser.equations = equations
        parser.figures = figures
        parser.tables = tables
        metadata = parser._generate_metadata()

        # ── Stage 7: Persist & complete ───────────────────────────────
        await repo.update_fields(document_id, {
            "text_content": text,
            "equations": [eq.model_dump() for eq in equations],
            "figures": [fig.model_dump() for fig in figures],
            "tables": [tbl.model_dump() for tbl in tables],
            "metadata": metadata.model_dump(),
            "status": DocumentStatus.COMPLETED,
            "processing_completed_at": datetime.utcnow(),
        })
        await _update_stage(repo, document_id, "Completed", 100, "Document processing completed successfully")

        logger.info("Document %s processed successfully", document_id)
        return {
            "document_id": document_id,
            "status": "completed",
            "message": "Document processing completed successfully",
            "summary": {
                "total_pages": metadata.total_pages,
                "total_paragraphs": metadata.total_paragraphs,
                "total_figures": len(figures),
                "total_tables": len(tables),
                "title": metadata.title,
                "author": metadata.author,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Processing failed for document %s: %s", document_id, str(e))
        await repo.update_fields(document_id, {
            "status": DocumentStatus.ERROR,
            "error_message": str(e),
        })
        await _update_stage(repo, document_id, "Error", 0, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


# ------------------------------------------------------------------ #
# Status                                                               #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/status",
    summary="Get document processing status",
    responses={
        200: {"description": "Current status and progress"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_document_status(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Poll the processing status of a document.

    Returns status, progress percentage (0-100), current stage,
    and full processing history.
    """
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    return {
        "document_id": document_id,
        "status": document["status"],
        "progress": document.get("progress", 0),
        "current_stage": document.get("current_stage", ""),
        "error_message": document.get("error_message"),
        "processing_history": document.get("processing_history", []),
    }


# ------------------------------------------------------------------ #
# List documents                                                        #
# ------------------------------------------------------------------ #

@router.get(
    "/",
    summary="List all documents for the current user",
    responses={200: {"description": "Paginated list of documents"}},
)
async def list_documents(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Return a paginated list of all documents uploaded by the authenticated user.
    Results are sorted newest first.
    """
    repo = DocumentRepository(db)
    skip = (page - 1) * page_size
    documents = await repo.find_by_user(current_user["email"], skip=skip, limit=page_size)
    total = await repo.count_by_user(current_user["email"])

    return {
        "documents": documents,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


# ------------------------------------------------------------------ #
# Get document                                                          #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}",
    summary="Get full document details",
    responses={
        200: {"description": "Document details with all extracted content"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Retrieve all details and extracted content for a specific document."""
    repo = DocumentRepository(db)
    return await _get_owned_document(document_id, current_user, repo)


# ------------------------------------------------------------------ #
# Delete document                                                       #
# ------------------------------------------------------------------ #

@router.delete(
    "/{document_id}",
    summary="Delete a document",
    responses={
        200: {"description": "Document deleted"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Permanently delete a document record and its uploaded file from storage.
    """
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    # Remove physical file
    file_path = document.get("file_path")
    if file_path:
        delete_file(file_path)

    await repo.delete(document_id)
    logger.info("Document %s deleted by %s", document_id, current_user["email"])
    return {"message": "Document deleted successfully", "document_id": document_id}


# ------------------------------------------------------------------ #
# Content endpoints                                                     #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/content/text",
    summary="Get extracted text content",
    responses={
        200: {"description": "Extracted text"},
        400: {"description": "Document not yet processed"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_text_content(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return only the extracted plain-text content of a processed document."""
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    if document["status"] != DocumentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has not been processed yet",
        )

    return {
        "document_id": document_id,
        "text_content": document.get("text_content", ""),
    }


@router.get(
    "/{document_id}/content/figures",
    summary="Get extracted figures",
    responses={
        200: {"description": "List of extracted figures with base64 images"},
        400: {"description": "Document not yet processed"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_figures(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return extracted figures (base64 images, captions, positions) for a document."""
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    if document["status"] != DocumentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has not been processed yet",
        )

    figures = document.get("figures", [])
    return {
        "document_id": document_id,
        "total_figures": len(figures),
        "figures": figures,
    }


@router.get(
    "/{document_id}/content/tables",
    summary="Get extracted tables",
    responses={
        200: {"description": "List of extracted tables as 2-D arrays"},
        400: {"description": "Document not yet processed"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_tables(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return extracted tables (2-D cell arrays, captions, positions) for a document."""
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    if document["status"] != DocumentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has not been processed yet",
        )

    tables = document.get("tables", [])
    return {
        "document_id": document_id,
        "total_tables": len(tables),
        "tables": tables,
    }


@router.get(
    "/{document_id}/content/equations",
    summary="Get extracted equations",
    responses={
        200: {"description": "List of extracted equations with LaTeX and OMML"},
        400: {"description": "Document not yet processed"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_equations(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Return extracted equations (LaTeX, raw OMML, positions, numbers) for a document.

    Each equation includes:
    - **latex**: LaTeX representation converted from OMML
    - **raw_omml**: Original Office Math Markup Language XML (for re-processing)
    - **number**: Equation number as it appears in the document (e.g. "(6-4)")
    - **position**: Paragraph index where the equation was found
    """
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    if document["status"] != DocumentStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document has not been processed yet",
        )

    equations = document.get("equations", [])
    return {
        "document_id": document_id,
        "total_equations": len(equations),
        "equations": equations,
    }


# ------------------------------------------------------------------ #
# Equation batch processing                                            #
# ------------------------------------------------------------------ #

@router.post(
    "/{document_id}/equations/batch-process",
    summary="Re-process all equations via Mathpix",
    responses={
        200: {"description": "Batch processing completed"},
        400: {"description": "Document not processed or no figures"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def batch_process_equations(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Re-process all figures through Mathpix to detect equations.
    Replaces existing Mathpix-detected equations with fresh results.
    Requires MATHPIX_APP_ID and MATHPIX_APP_KEY in .env.
    """
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    if document["status"] not in (DocumentStatus.COMPLETED, DocumentStatus.ERROR):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document must be processed first",
        )

    if not settings.MATHPIX_APP_ID or not settings.MATHPIX_APP_KEY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mathpix API keys not configured",
        )

    from app.models.document import Figure, Equation

    # Rebuild Figure objects from stored data
    figures_raw = document.get("figures", [])
    all_equations = document.get("equations", [])

    # Also include figures that were previously classified as equations (re-scan them)
    figures = [Figure(**f) if isinstance(f, dict) else f for f in figures_raw]

    if not figures:
        return {
            "document_id": document_id,
            "message": "No figures to process",
            "equations_found": 0,
        }

    mathpix = MathpixService()
    new_eqs, remaining_figs = await mathpix.extract_equations_from_figures(figures)

    # Keep non-Mathpix equations (OMML-extracted), replace Mathpix ones
    omml_equations = [eq for eq in all_equations if not eq.get("equation_id", "").startswith("eq_mathpix_")]
    combined_equations = omml_equations + [eq.model_dump() for eq in new_eqs]

    await repo.update_fields(document_id, {
        "equations": combined_equations,
        "figures": [fig.model_dump() for fig in remaining_figs],
    })

    logger.info(
        "Batch equation processing for %s: %d new equations from %d figures",
        document_id, len(new_eqs), len(figures),
    )

    return {
        "document_id": document_id,
        "message": "Batch equation processing completed",
        "equations_found": len(new_eqs),
        "total_equations": len(combined_equations),
        "remaining_figures": len(remaining_figs),
    }


# ------------------------------------------------------------------ #
# Equation preview                                                     #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/equations/{equation_id}/preview",
    summary="Get equation LaTeX preview",
    responses={
        200: {"description": "Equation preview data"},
        404: {"description": "Equation or document not found"},
        403: {"description": "Not authorized"},
    },
)
async def get_equation_preview(
    document_id: str,
    equation_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Return equation data for rendering preview.
    Includes LaTeX string, base64 image (if available), and position info.
    The frontend can render the LaTeX using KaTeX or MathJax.
    """
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    equations = document.get("equations", [])
    equation = None
    for eq in equations:
        eq_id = eq.get("equation_id", "") if isinstance(eq, dict) else eq.equation_id
        if eq_id == equation_id:
            equation = eq
            break

    if not equation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Equation not found")

    eq_data = equation if isinstance(equation, dict) else equation.model_dump()

    return {
        "document_id": document_id,
        "equation_id": equation_id,
        "latex": eq_data.get("latex", ""),
        "image_base64": eq_data.get("image_base64"),
        "position": eq_data.get("position"),
        "number": eq_data.get("number"),
        "render_hint": "Use KaTeX or MathJax to render the LaTeX string on the frontend",
    }
