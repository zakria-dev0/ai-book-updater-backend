import os
import csv
import io
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from bson import ObjectId
from app.core.security import get_current_user_dep
from app.database.connection import get_database
from app.database.repositories.change_repo import ChangeRepository
from app.services.export_service import ExportService
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Export"])


@router.get(
    "/{document_id}/export/docx",
    summary="Download updated DOCX with approved changes applied",
    responses={
        200: {"description": "Updated DOCX file", "content": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document": {}}},
        404: {"description": "Document or export not found"},
    },
)
async def export_updated_docx(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Generate and download an updated DOCX file with all approved changes applied.
    """
    doc = await db.documents.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.get("role") != "admin" and doc.get("user_id") != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    # Get approved changes
    approved_changes = await db.changes.find({
        "document_id": document_id,
        "status": {"$in": ["approved", "applied"]},
    }).to_list(None)

    if not approved_changes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No approved changes to apply",
        )

    # Generate the updated file
    output_path = await ExportService.generate_updated_docx(doc, approved_changes)
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate updated document",
        )

    original_name = doc.get("original_filename", "document.docx")
    base_name = os.path.splitext(original_name)[0]
    download_name = f"{base_name}_updated.docx"

    return FileResponse(
        path=output_path,
        filename=download_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.get(
    "/{document_id}/export/docx-highlighted",
    summary="Download highlighted DOCX with changes marked in yellow/green",
    responses={
        200: {"description": "Highlighted DOCX file", "content": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document": {}}},
        404: {"description": "Document or export not found"},
    },
)
async def export_highlighted_docx(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """
    Generate and download a highlighted DOCX file.
    - Regular changes: highlighted in yellow
    - AI-generated content: highlighted in green
    """
    doc = await db.documents.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.get("role") != "admin" and doc.get("user_id") != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    approved_changes = await db.changes.find({
        "document_id": document_id,
        "status": {"$in": ["approved", "applied"]},
    }).to_list(None)

    if not approved_changes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No approved changes to apply",
        )

    output_path = await ExportService.generate_updated_docx(doc, approved_changes, highlighted=True)
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate highlighted document",
        )

    original_name = doc.get("original_filename", "document.docx")
    base_name = os.path.splitext(original_name)[0]
    download_name = f"{base_name}_highlighted.docx"

    return FileResponse(
        path=output_path,
        filename=download_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.get(
    "/{document_id}/export/preview",
    summary="Preview highlighted document as PDF in browser",
    responses={
        200: {"description": "PDF preview of highlighted document", "content": {"application/pdf": {}}},
        400: {"description": "No approved changes"},
        404: {"description": "Document not found"},
    },
)
async def preview_highlighted(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Return a PDF of the highlighted document for in-browser preview."""
    doc = await db.documents.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    if current_user.get("role") != "admin" and doc.get("user_id") != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")

    approved_changes = await db.changes.find({
        "document_id": document_id,
        "status": {"$in": ["approved", "applied"]},
    }).to_list(None)

    if not approved_changes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No approved changes to preview",
        )

    pdf_path = await ExportService.generate_preview_pdf(doc, approved_changes)
    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate preview",
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline"},
    )


@router.get(
    "/{document_id}/export/original",
    summary="Download the original uploaded document",
    responses={
        200: {"description": "Original uploaded file"},
        404: {"description": "Document or file not found"},
    },
)
async def download_original(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Download the original uploaded DOCX file before any changes."""
    doc = await db.documents.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if doc.get("user_id") != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your document")

    file_path = doc.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original file not found on disk")

    original_name = doc.get("original_filename", "document.docx")
    return FileResponse(
        path=file_path,
        filename=original_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.get(
    "/{document_id}/export/csv",
    summary="Export change log as CSV",
    responses={
        200: {"description": "CSV file of all changes"},
        400: {"description": "No changes found"},
        404: {"description": "Document not found"},
    },
)
async def export_csv(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Export all change proposals for a document as a CSV file."""
    doc = await db.documents.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if doc.get("user_id") != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your document")

    change_repo = ChangeRepository(db)
    changes = await change_repo.find_by_document(document_id, skip=0, limit=5000)

    if not changes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No changes found for this document",
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Change ID", "Type", "Status", "Page", "Confidence",
        "Original Content", "New Content", "User Edited Content",
        "Reviewer Note", "Sources", "Reviewed At",
    ])
    for c in changes:
        sources = c.get("sources", [])
        source_urls = "; ".join(s.get("url", "") for s in sources if isinstance(s, dict))
        writer.writerow([
            c.get("id", ""),
            c.get("change_type", ""),
            c.get("status", ""),
            c.get("page_number", ""),
            c.get("confidence", ""),
            c.get("original_content", ""),
            c.get("new_content", ""),
            c.get("user_edited_content", ""),
            c.get("reviewer_note", ""),
            source_urls,
            c.get("reviewed_at", ""),
        ])

    output.seek(0)
    original_name = doc.get("original_filename", "document")
    base_name = os.path.splitext(original_name)[0]

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{base_name}_changes.csv"'},
    )


@router.get(
    "/{document_id}/export/pdf",
    summary="Export change log as PDF",
    responses={
        200: {"description": "PDF file of change log"},
        400: {"description": "No changes found"},
        404: {"description": "Document not found"},
    },
)
async def export_pdf(
    document_id: str,
    current_user: dict = Depends(get_current_user_dep),
    db=Depends(get_database),
):
    """Export all change proposals for a document as a PDF report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table as RLTable, TableStyle

    doc_record = await db.documents.find_one({"_id": ObjectId(document_id)})
    if not doc_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    if doc_record.get("user_id") != current_user["email"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your document")

    change_repo = ChangeRepository(db)
    changes = await change_repo.find_by_document(document_id, skip=0, limit=5000)

    if not changes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No changes found for this document",
        )

    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=16, spaceAfter=6*mm)
    story.append(Paragraph("Change Log Report", title_style))

    # Document info
    filename = doc_record.get("original_filename", "Unknown")
    story.append(Paragraph(f"<b>Document:</b> {filename}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))

    approved = sum(1 for c in changes if c.get("status") in ("approved", "applied"))
    rejected = sum(1 for c in changes if c.get("status") == "rejected")
    story.append(Paragraph(
        f"<b>Total:</b> {len(changes)} | <b>Approved:</b> {approved} | <b>Rejected:</b> {rejected}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 8*mm))

    # Changes table
    header = ["#", "Page", "Type", "Status", "Original", "Updated"]
    table_data = [header]
    for i, c in enumerate(changes, 1):
        old_text = (c.get("original_content") or c.get("old_content") or "")[:120]
        new_text = (c.get("user_edited_content") or c.get("new_content") or "")[:120]
        table_data.append([
            str(i),
            str(c.get("page_number", c.get("page", "-"))),
            c.get("change_type", ""),
            c.get("status", ""),
            Paragraph(old_text, ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7, leading=9)),
            Paragraph(new_text, ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7, leading=9)),
        ])

    col_widths = [20, 30, 60, 50, 170, 170]
    tbl = RLTable(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tbl)

    pdf.build(story)
    buffer.seek(0)

    original_name = doc_record.get("original_filename", "document")
    base_name = os.path.splitext(original_name)[0]

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{base_name}_changelog.pdf"'},
    )
