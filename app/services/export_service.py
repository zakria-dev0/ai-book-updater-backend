import os
from typing import Optional, List
from docx import Document as DocxDocument
from app.core.logger import get_logger
from app.core.config import settings
from app.services.renumbering_service import RenumberingService

logger = get_logger(__name__)


class ExportService:
    """Service for generating updated DOCX files with approved changes applied."""

    @staticmethod
    async def generate_updated_docx(
        document_data: dict,
        approved_changes: list,
        db=None,
    ) -> Optional[str]:
        """
        Generate an updated DOCX file by applying approved changes.

        Args:
            document_data: The document record from MongoDB.
            approved_changes: List of approved ChangeProposal dicts.
            db: Database connection (unused for now, reserved for future use).

        Returns:
            Path to the generated DOCX file, or None on failure.
        """
        file_path = document_data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            logger.error("Original file not found: %s", file_path)
            return None

        try:
            doc = DocxDocument(file_path)

            # Sort changes by paragraph index (descending) so that replacements
            # don't shift indices for subsequent changes.
            sorted_changes = sorted(
                approved_changes,
                key=lambda c: c.get("paragraph_idx", 0),
                reverse=True,
            )

            replacements_made = 0

            for change in sorted_changes:
                old_content = change.get("old_content", "")
                # Use user-edited content if available, otherwise use AI-generated
                new_content = change.get("user_edited_content") or change.get("new_content", "")

                if not old_content or not new_content:
                    continue

                # Try to find and replace the old content in document paragraphs
                for paragraph in doc.paragraphs:
                    if old_content in paragraph.text:
                        # Replace while preserving some formatting
                        for run in paragraph.runs:
                            if old_content in run.text:
                                run.text = run.text.replace(old_content, new_content)
                                replacements_made += 1
                                break
                        else:
                            # If the old_content spans multiple runs, do a full paragraph replacement
                            full_text = paragraph.text
                            if old_content in full_text:
                                new_full_text = full_text.replace(old_content, new_content)
                                # Clear all runs and set new text on first run
                                if paragraph.runs:
                                    paragraph.runs[0].text = new_full_text
                                    for run in paragraph.runs[1:]:
                                        run.text = ""
                                    replacements_made += 1
                        break  # Only replace first occurrence per change

            # Save to output directory
            os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
            original_name = document_data.get("original_filename", "document.docx")
            base_name = os.path.splitext(original_name)[0]
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
                "Generated updated DOCX: %s (%d replacements)",
                output_path,
                replacements_made,
            )
            return output_path

        except Exception as e:
            logger.error("Failed to generate updated DOCX: %s", str(e))
            return None
