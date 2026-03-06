import os
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks, Request
from bson import ObjectId
from datetime import datetime
from app.core.security import get_current_user_dep
from app.database.connection import get_database
from app.database.repositories.document_repo import DocumentRepository
from app.models.document import DocumentStatus
from app.services.document_service import DOCXParser
from app.services.equation_service import MathpixService
from app.services.image_service import ImageService
from app.utils.file_handler import delete_file
from app.core.logger import get_logger
from app.core.config import settings
from app.core.rate_limit import limiter
from app.core.websocket import ws_manager

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

    # Broadcast via WebSocket
    try:
        await ws_manager.broadcast(document_id, {
            "type": "progress",
            "document_id": document_id,
            "status": "processing",
            "progress": progress,
            "current_stage": stage,
            "message": message,
            "timestamp": entry["timestamp"].isoformat(),
        })
    except Exception:
        pass  # Non-critical — don't fail processing if WS broadcast fails


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

async def _run_processing(document_id: str, db):
    """Background task: run the full DOCX processing pipeline."""
    repo = DocumentRepository(db)

    try:
        document = await repo.find_by_id(document_id)
        if not document:
            logger.error("Background processing: document %s not found", document_id)
            return

        parser = DOCXParser(document["file_path"], use_nougat=False)

        await _update_stage(repo, document_id, "Extracting text", 20, "Extracting text content")
        text = parser._extract_text()

        await _update_stage(repo, document_id, "Extracting figures", 45, "Extracting embedded images")
        figures = parser._extract_figures()

        await _update_stage(repo, document_id, "Extracting tables", 65, "Extracting tables")
        tables = parser._extract_tables()

        await _update_stage(repo, document_id, "Extracting equations", 70, "Extracting OMML equations and converting to LaTeX")
        equations = parser._extract_equations()

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

        await _update_stage(repo, document_id, "Generating metadata", 85, "Generating document metadata")
        parser.equations = equations
        parser.figures = figures
        parser.tables = tables
        metadata = parser._generate_metadata()

        para_to_page = {str(k): v for k, v in parser._para_to_page.items()}

        await repo.update_fields(document_id, {
            "text_content": text,
            "equations": [eq.model_dump() for eq in equations],
            "figures": [fig.model_dump() for fig in figures],
            "tables": [tbl.model_dump() for tbl in tables],
            "metadata": metadata.model_dump(),
            "para_to_page": para_to_page,
            "status": DocumentStatus.COMPLETED,
            "processing_completed_at": datetime.utcnow(),
        })
        await _update_stage(repo, document_id, "Completed", 100, "Document processing completed successfully")
        logger.info("Document %s processed successfully", document_id)

        # Broadcast completion
        try:
            await ws_manager.broadcast(document_id, {
                "type": "completed",
                "document_id": document_id,
                "status": "completed",
                "progress": 100,
                "current_stage": "Completed",
            })
        except Exception:
            pass

    except Exception as e:
        logger.error("Processing failed for document %s: %s", document_id, str(e))
        await repo.update_fields(document_id, {
            "status": DocumentStatus.ERROR,
            "error_message": str(e),
        })
        await _update_stage(repo, document_id, "Error", 0, str(e))

        # Broadcast error
        try:
            await ws_manager.broadcast(document_id, {
                "type": "error",
                "document_id": document_id,
                "status": "error",
                "error_message": str(e),
            })
        except Exception:
            pass


@router.post(
    "/{document_id}/process",
    summary="Start processing a document",
    responses={
        200: {"description": "Processing started"},
        400: {"description": "Invalid request"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
@limiter.limit("5/minute")
async def process_document(
    request: Request,
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Start processing a DOCX document in the background.

    Returns immediately. Poll `GET /documents/{id}/status` for progress.
    """
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    if document["status"] == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is already being processed",
        )

    if document["file_type"] != "docx":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only DOCX files are supported",
        )

    # Set status immediately and launch background task
    await repo.update_fields(document_id, {
        "status": DocumentStatus.PROCESSING,
        "processing_started_at": datetime.utcnow(),
        "processing_history": [],
    })
    await _update_stage(repo, document_id, "Initializing", 5, "Starting document processing")

    background_tasks.add_task(_run_processing, document_id, db)

    return {
        "document_id": document_id,
        "status": "processing",
        "message": "Processing started. Poll GET /documents/{id}/status for progress.",
    }


# ------------------------------------------------------------------ #
# Restart processing                                                   #
# ------------------------------------------------------------------ #

@router.post(
    "/{document_id}/reprocess",
    summary="Restart document processing",
    responses={
        200: {"description": "Re-processing started"},
        400: {"description": "Cannot reprocess in current state"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
@limiter.limit("3/minute")
async def reprocess_document(
    request: Request,
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Restart processing for a document that errored or was already completed.
    Clears previous extraction data and re-runs the pipeline.
    """
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    if document["status"] == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is currently being processed",
        )

    # Reset extraction data
    await repo.update_fields(document_id, {
        "status": DocumentStatus.PROCESSING,
        "processing_started_at": datetime.utcnow(),
        "processing_history": [],
        "text_content": None,
        "equations": [],
        "figures": [],
        "tables": [],
        "metadata": None,
        "error_message": None,
    })
    await _update_stage(repo, document_id, "Initializing", 5, "Restarting document processing")

    background_tasks.add_task(_run_processing, document_id, db)

    return {
        "document_id": document_id,
        "status": "processing",
        "message": "Re-processing started. Poll GET /documents/{id}/status for progress.",
    }


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
# Document stats                                                        #
# ------------------------------------------------------------------ #

@router.get(
    "/stats",
    summary="Get document status counts for the current user",
    responses={200: {"description": "Status counts"}},
)
async def document_stats(
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return counts of documents grouped by status for the authenticated user."""
    pipeline = [
        {"$match": {"user_id": current_user["email"]}},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
    ]
    results = await db.documents.aggregate(pipeline).to_list(length=20)
    status_counts = {r["_id"]: r["count"] for r in results if r["_id"]}
    total = sum(status_counts.values())
    return {
        "total": total,
        "uploaded": status_counts.get("uploaded", 0),
        "processing": status_counts.get("processing", 0),
        "completed": status_counts.get("completed", 0)
            + status_counts.get("export_ready", 0)
            + status_counts.get("analyzing", 0),
        "error": status_counts.get("error", 0),
    }


# ------------------------------------------------------------------ #
# Get document                                                          #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}",
    summary="Get document details",
    responses={
        200: {"description": "Document details"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def get_document(
    document_id: str,
    lightweight: bool = Query(default=True, description="Exclude heavy fields (text, figures, equations, tables)"),
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Retrieve document details. Use lightweight=false to include full extracted content."""
    repo = DocumentRepository(db)
    document = await repo.find_by_id(document_id, lightweight=lightweight)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if document["user_id"] != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return document


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
@limiter.limit("3/minute")
async def batch_process_equations(
    request: Request,
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

    # Find surrounding paragraph text for context
    paragraphs = document.get("text_content", "").split("\n")
    pos = eq_data.get("position") or {}
    para_idx = pos.get("paragraph")
    context = ""
    if para_idx is not None and 0 <= para_idx < len(paragraphs):
        full_para = paragraphs[para_idx]
        context = full_para[:200] + ("..." if len(full_para) > 200 else "")

    return {
        "document_id": document_id,
        "equation_id": equation_id,
        "latex": eq_data.get("latex", ""),
        "image_base64": eq_data.get("image_base64"),
        "has_image": eq_data.get("image_base64") is not None,
        "position": pos,
        "number": eq_data.get("number"),
        "context": context,
        "render_hint": "Use KaTeX or MathJax to render the LaTeX string on the frontend",
    }


# ------------------------------------------------------------------ #
# Document thumbnail (first-page preview)                              #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/thumbnail",
    summary="Get document first-page thumbnail",
    responses={
        200: {"description": "Thumbnail data"},
        404: {"description": "Document not found or no preview available"},
    },
)
async def get_document_thumbnail(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Generate a thumbnail from the first page of the uploaded document."""
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    file_path = document.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source file not found")

    try:
        import fitz  # PyMuPDF
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        # For DOCX files, we can't directly render — return empty
        if file_path.endswith(".docx"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thumbnail not available for DOCX")

        doc = fitz.open(file_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(0.5, 0.5))  # half resolution
        img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.thumbnail((300, 400))
        buf = BytesIO()
        img.save(buf, format="PNG")
        thumbnail_b64 = base64.b64encode(buf.getvalue()).decode()
        doc.close()

        return {"thumbnail": thumbnail_b64}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not generate thumbnail")


# ------------------------------------------------------------------ #
# Figure thumbnail                                                     #
# ------------------------------------------------------------------ #

@router.get(
    "/{document_id}/figures/{figure_id}/thumbnail",
    summary="Get figure thumbnail",
    responses={
        200: {"description": "Figure thumbnail data"},
        404: {"description": "Figure or document not found"},
        403: {"description": "Not authorized"},
    },
)
async def get_figure_thumbnail(
    document_id: str,
    figure_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return a thumbnail version of a figure (300x300 max)."""
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    figures = document.get("figures", [])
    figure = None
    for fig in figures:
        fig_id = fig.get("figure_id", "") if isinstance(fig, dict) else fig.figure_id
        if fig_id == figure_id:
            figure = fig
            break

    if not figure:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Figure not found")

    fig_data = figure if isinstance(figure, dict) else figure.model_dump()
    image_b64 = fig_data.get("image_base64")

    if not image_b64:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Figure has no image data")

    thumbnail_b64 = ImageService.generate_thumbnail(image_b64)
    metadata = ImageService.get_image_metadata(image_b64)

    return {
        "document_id": document_id,
        "figure_id": figure_id,
        "thumbnail_base64": thumbnail_b64,
        "caption": fig_data.get("caption"),
        "number": fig_data.get("number"),
        "original_metadata": metadata,
    }


# ------------------------------------------------------------------ #
# Clone / duplicate document                                          #
# ------------------------------------------------------------------ #

@router.post(
    "/{document_id}/clone",
    summary="Clone a document",
    responses={
        201: {"description": "Document cloned successfully"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def clone_document(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Create a copy of a document with a new ID. Copies metadata and extracted content but not analysis results."""
    repo = DocumentRepository(db)
    document = await _get_owned_document(document_id, current_user, repo)

    import copy
    import shutil

    clone = copy.deepcopy(document)
    clone.pop("_id", None)
    clone.pop("id", None)
    new_id = str(ObjectId())
    clone["_id"] = ObjectId(new_id)
    clone["original_filename"] = f"Copy of {document.get('original_filename', document.get('filename', 'document'))}"
    clone["uploaded_at"] = datetime.utcnow().isoformat()
    clone["status"] = document.get("status", "uploaded")

    # Copy physical file if exists
    src_path = document.get("file_path", "")
    if src_path and os.path.exists(src_path):
        ext = os.path.splitext(src_path)[1]
        dest_path = os.path.join(os.path.dirname(src_path), f"{new_id}{ext}")
        shutil.copy2(src_path, dest_path)
        clone["file_path"] = dest_path

    await db.documents.insert_one(clone)
    logger.info("Document %s cloned to %s by %s", document_id, new_id, current_user["email"])

    return {
        "message": "Document cloned successfully",
        "document_id": new_id,
        "original_id": document_id,
    }


# ------------------------------------------------------------------ #
# Archive / unarchive document                                        #
# ------------------------------------------------------------------ #

@router.post(
    "/{document_id}/archive",
    summary="Archive a document",
    responses={
        200: {"description": "Document archived"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def archive_document(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Mark a document as archived. Archived documents are hidden from the default list."""
    repo = DocumentRepository(db)
    await _get_owned_document(document_id, current_user, repo)
    await repo.update_fields(document_id, {"archived": True, "archived_at": datetime.utcnow().isoformat()})
    logger.info("Document %s archived by %s", document_id, current_user["email"])
    return {"message": "Document archived", "document_id": document_id}


@router.post(
    "/{document_id}/unarchive",
    summary="Unarchive a document",
    responses={
        200: {"description": "Document unarchived"},
        403: {"description": "Not authorized"},
        404: {"description": "Document not found"},
    },
)
async def unarchive_document(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Restore an archived document back to the active list."""
    repo = DocumentRepository(db)
    await _get_owned_document(document_id, current_user, repo)
    await repo.update_fields(document_id, {"archived": False, "archived_at": None})
    logger.info("Document %s unarchived by %s", document_id, current_user["email"])
    return {"message": "Document unarchived", "document_id": document_id}
