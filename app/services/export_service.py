import os
import copy
from typing import Optional, List
from docx import Document as DocxDocument
from docx.shared import RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from app.core.logger import get_logger
from app.core.config import settings
from app.services.renumbering_service import RenumberingService

logger = get_logger(__name__)


def _add_highlight(run, color="yellow"):
    """Add highlight color to a run. Common colors: yellow, green, cyan."""
    rPr = run._element.get_or_add_rPr()
    highlight = OxmlElement("w:highlight")
    highlight.set(qn("w:val"), color)
    rPr.append(highlight)


def _is_heading(paragraph):
    """Check if a paragraph is a heading (by style name or outline level)."""
    style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
    if "heading" in style_name or "title" in style_name or "toc" in style_name:
        return True
    # Check for outline level in pPr
    pPr = paragraph._element.find(qn("w:pPr"))
    if pPr is not None:
        outlineLvl = pPr.find(qn("w:outlineLvl"))
        if outlineLvl is not None:
            return True
    return False


def _find_body_style_paragraph(doc_paragraphs, near_idx):
    """Find the nearest body text paragraph (non-heading, non-empty) near the given index.

    Used as a style reference so AI-inserted content matches body text formatting.
    """
    # Search forward from near_idx
    for i in range(near_idx, min(near_idx + 20, len(doc_paragraphs))):
        p = doc_paragraphs[i]
        if p.text.strip() and not _is_heading(p) and p.runs:
            return p
    # Search backward from near_idx
    for i in range(near_idx - 1, max(near_idx - 20, -1), -1):
        p = doc_paragraphs[i]
        if p.text.strip() and not _is_heading(p) and p.runs:
            return p
    return None


def _find_section_end(doc_paragraphs, start_idx):
    """Find the last body paragraph in the current section (before next heading).

    Returns a suitable insertion point so AI content goes at the end of a section,
    not in the middle of complex layout areas like chapter openers.
    """
    last_body = None
    for i in range(start_idx, len(doc_paragraphs)):
        p = doc_paragraphs[i]
        if i > start_idx and _is_heading(p):
            # Hit the next heading — stop here
            break
        if p.text.strip() and not _is_heading(p):
            last_body = p
    return last_body


def _insert_paragraph_after(paragraph, text, highlight_color=None, style_ref=None):
    """Insert new paragraph(s) after the given paragraph, optionally highlighted.

    Args:
        paragraph: The paragraph to insert after.
        text: Text to insert (may contain newlines for multi-paragraph content).
        highlight_color: Optional highlight color (e.g. "yellow", "green").
        style_ref: Optional paragraph to copy formatting from. If None, uses `paragraph`.
    """
    from docx.text.paragraph import Paragraph

    # Use style_ref for formatting, fall back to insertion target
    fmt_source = style_ref if style_ref is not None else paragraph

    lines = text.split("\n") if "\n" in text else [text]
    last_para = paragraph

    for line in lines:
        line = line.strip()
        if not line:
            continue

        new_p = OxmlElement("w:p")
        last_para._element.addnext(new_p)

        # Copy paragraph properties (alignment, spacing, indent, style) from style reference
        source_pPr = fmt_source._element.find(qn("w:pPr"))
        if source_pPr is not None:
            new_pPr = copy.deepcopy(source_pPr)
            new_p.insert(0, new_pPr)

        new_para = Paragraph(new_p, paragraph._parent)
        run = new_para.add_run(line)

        # Copy run-level formatting (font, size, bold, italic) from style reference
        if fmt_source.runs:
            source_rPr = fmt_source.runs[0]._element.find(qn("w:rPr"))
            if source_rPr is not None:
                new_rPr = copy.deepcopy(source_rPr)
                # Remove any existing highlight from source so we can set our own
                for old_hl in new_rPr.findall(qn("w:highlight")):
                    new_rPr.remove(old_hl)
                # Remove auto-created rPr and set the copied one
                existing_rPr = run._element.find(qn("w:rPr"))
                if existing_rPr is not None:
                    run._element.remove(existing_rPr)
                run._element.insert(0, new_rPr)

        if highlight_color:
            _add_highlight(run, highlight_color)

        last_para = new_para

    return last_para


class ExportService:
    """Service for generating updated DOCX files with approved changes applied."""

    @staticmethod
    async def generate_updated_docx(
        document_data: dict,
        approved_changes: list,
        db=None,
        highlighted: bool = False,
    ) -> Optional[str]:
        """
        Generate an updated DOCX file by applying approved changes.

        Args:
            document_data: The document record from MongoDB.
            approved_changes: List of approved ChangeProposal dicts.
            db: Database connection (unused for now, reserved for future use).
            highlighted: If True, highlight changed text in yellow/green.

        Returns:
            Path to the generated DOCX file, or None on failure.
        """
        file_path = document_data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            logger.error("Original file not found: %s", file_path)
            return None

        try:
            doc = DocxDocument(file_path)

            # Separate AI prompt changes (insertions) from regular changes (replacements)
            ai_prompt_changes = [
                c for c in approved_changes
                if c.get("change_type") == "ai_prompt"
                   and not c.get("old_content", "").startswith("[AI Prompt:")
                   or c.get("change_type") == "ai_prompt"
            ]
            regular_changes = [
                c for c in approved_changes
                if c.get("change_type") != "ai_prompt"
            ]

            # Sort regular changes by paragraph index (descending) so that
            # replacements don't shift indices for subsequent changes.
            sorted_changes = sorted(
                regular_changes,
                key=lambda c: c.get("paragraph_idx", 0),
                reverse=True,
            )

            replacements_made = 0
            insertions_made = 0

            # 1. Apply regular text replacements
            for change in sorted_changes:
                old_content = change.get("old_content", "")
                new_content = change.get("user_edited_content") or change.get("new_content", "")

                if not old_content or not new_content:
                    continue

                for paragraph in doc.paragraphs:
                    if old_content in paragraph.text:
                        if highlighted:
                            # Clear paragraph and add highlighted replacement
                            full_text = paragraph.text
                            new_full_text = full_text.replace(old_content, new_content)
                            # Clear all existing runs
                            for run in paragraph.runs:
                                run.text = ""
                            if paragraph.runs:
                                paragraph.runs[0].text = new_full_text
                                _add_highlight(paragraph.runs[0], "yellow")
                            replacements_made += 1
                        else:
                            # Plain replacement (no highlight)
                            for run in paragraph.runs:
                                if old_content in run.text:
                                    run.text = run.text.replace(old_content, new_content)
                                    replacements_made += 1
                                    break
                            else:
                                full_text = paragraph.text
                                if old_content in full_text:
                                    new_full_text = full_text.replace(old_content, new_content)
                                    if paragraph.runs:
                                        paragraph.runs[0].text = new_full_text
                                        for run in paragraph.runs[1:]:
                                            run.text = ""
                                        replacements_made += 1
                        break

            # 2. Apply AI prompt insertions (sorted descending by paragraph_idx)
            sorted_ai_changes = sorted(
                ai_prompt_changes,
                key=lambda c: c.get("paragraph_idx", 0),
                reverse=True,
            )

            for change in sorted_ai_changes:
                new_content = change.get("user_edited_content") or change.get("new_content", "")
                if not new_content:
                    continue

                para_idx = change.get("paragraph_idx", 0)
                placement = "at_end"
                metadata = change.get("ai_prompt_metadata", {})
                if metadata:
                    placement = metadata.get("placement", "at_end")

                old_content = change.get("old_content", "")
                is_replace = placement == "replace_section" and old_content and not old_content.startswith("[AI Prompt:")

                # Find a body text paragraph near the target for style reference
                style_ref = _find_body_style_paragraph(doc.paragraphs, min(para_idx, len(doc.paragraphs) - 1))

                if is_replace:
                    # Replace: find the old content and replace it
                    replaced = False
                    for paragraph in doc.paragraphs:
                        # Check if first line of old_content matches
                        first_line = old_content.split('\n')[0].strip()
                        if first_line and first_line in paragraph.text:
                            # Split new content into lines
                            content_lines = [l.strip() for l in new_content.split("\n") if l.strip()]
                            if content_lines:
                                # Put first line in the existing paragraph
                                for run in paragraph.runs:
                                    run.text = ""
                                if paragraph.runs:
                                    paragraph.runs[0].text = content_lines[0]
                                    if highlighted:
                                        _add_highlight(paragraph.runs[0], "green")
                                # Insert remaining lines as new paragraphs after
                                if len(content_lines) > 1:
                                    remaining = "\n".join(content_lines[1:])
                                    _insert_paragraph_after(
                                        paragraph,
                                        remaining,
                                        "green" if highlighted else None,
                                        style_ref=style_ref,
                                    )
                            replaced = True
                            insertions_made += 1
                            break
                    if not replaced:
                        # Fallback: insert at end
                        if doc.paragraphs:
                            _insert_paragraph_after(
                                doc.paragraphs[-1],
                                new_content,
                                "green" if highlighted else None,
                                style_ref=style_ref,
                            )
                            insertions_made += 1
                else:
                    # Insert: find the end of the current section to avoid
                    # disrupting complex layouts (chapter openers, columns, etc.)
                    safe_idx = min(para_idx, len(doc.paragraphs) - 1)
                    section_end = _find_section_end(doc.paragraphs, safe_idx)

                    if section_end:
                        target_para = section_end
                    elif safe_idx < len(doc.paragraphs):
                        target_para = doc.paragraphs[safe_idx]
                    else:
                        target_para = doc.paragraphs[-1] if doc.paragraphs else None

                    if target_para:
                        _insert_paragraph_after(
                            target_para,
                            new_content,
                            "green" if highlighted else None,
                            style_ref=style_ref,
                        )
                        insertions_made += 1

            # Save to output directory
            os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
            original_name = document_data.get("original_filename", "document.docx")
            base_name = os.path.splitext(original_name)[0]

            if highlighted:
                output_filename = f"{base_name}_highlighted.docx"
            else:
                output_filename = f"{base_name}_updated.docx"

            output_path = os.path.join(settings.OUTPUT_DIR, output_filename)

            # Validate references after applying changes
            text_content = document_data.get("text_content", "")
            figures = document_data.get("figures", [])
            tables = document_data.get("tables", [])
            equations = document_data.get("equations", [])

            if text_content:
                warnings = RenumberingService.validate_references(
                    text_content, figures, tables, equations
                )
                for warning in warnings:
                    logger.warning("Export reference check: %s", warning)

            doc.save(output_path)
            logger.info(
                "Generated %s DOCX: %s (%d replacements, %d insertions)",
                "highlighted" if highlighted else "clean",
                output_path,
                replacements_made,
                insertions_made,
            )
            return output_path

        except Exception as e:
            logger.error("Failed to generate updated DOCX: %s", str(e))
            return None

    @staticmethod
    async def generate_preview_pdf(
        document_data: dict,
        approved_changes: list,
    ) -> Optional[str]:
        """
        Generate a PDF preview that is pixel-perfect identical to the Highlighted DOCX.

        Steps:
            1. Generate the highlighted DOCX (reuses generate_updated_docx with highlighted=True)
            2. Convert that DOCX → PDF using docx2pdf (uses Microsoft Word on Windows)

        Returns path to the generated PDF file, or None on failure.
        """
        try:
            # 1. Generate the highlighted DOCX (same file the user would download)
            highlighted_docx_path = await ExportService.generate_updated_docx(
                document_data, approved_changes, highlighted=True
            )
            if not highlighted_docx_path or not os.path.exists(highlighted_docx_path):
                logger.error("Failed to generate highlighted DOCX for preview")
                return None

            # 2. Convert DOCX → PDF
            pdf_path = os.path.splitext(highlighted_docx_path)[0] + "_preview.pdf"

            try:
                from docx2pdf import convert
                convert(highlighted_docx_path, pdf_path)
            except Exception as e:
                logger.warning("docx2pdf failed (%s), trying LibreOffice fallback", e)
                # Fallback: try LibreOffice headless
                import subprocess
                result = subprocess.run(
                    [
                        "soffice", "--headless", "--convert-to", "pdf",
                        "--outdir", os.path.dirname(pdf_path),
                        highlighted_docx_path,
                    ],
                    capture_output=True, text=True, timeout=60,
                )
                # LibreOffice outputs to same dir with .pdf extension
                lo_pdf = os.path.splitext(highlighted_docx_path)[0] + ".pdf"
                if os.path.exists(lo_pdf) and lo_pdf != pdf_path:
                    os.rename(lo_pdf, pdf_path)
                if not os.path.exists(pdf_path):
                    logger.error("LibreOffice conversion also failed: %s", result.stderr)
                    return None

            if not os.path.exists(pdf_path):
                logger.error("PDF conversion produced no output")
                return None

            logger.info("Generated preview PDF: %s", pdf_path)
            return pdf_path

        except Exception as e:
            logger.error("Failed to generate preview PDF: %s", str(e))
            return None
