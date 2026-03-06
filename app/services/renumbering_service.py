"""Cross-reference detection and validation for figures, tables, and equations."""

import re
from dataclasses import dataclass, field
from typing import List, Dict
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Reference patterns ────────────────────────────────────────────────────────

# Matches: Figure 2-3, Fig. 2-3, Fig 2.3
FIGURE_PATTERN = re.compile(
    r"\b(?:Figure|Fig\.?)\s+(\d+[\-\.]\d+)\b", re.IGNORECASE
)
# Matches: Table 2-3, Tbl. 2-3
TABLE_PATTERN = re.compile(
    r"\b(?:Table|Tbl\.?)\s+(\d+[\-\.]\d+)\b", re.IGNORECASE
)
# Matches: Equation (6-4), Eq. (6-4), (6-4) when preceded by "Equation"
EQUATION_PATTERN = re.compile(
    r"\b(?:Equation|Eq\.?)\s*\((\d+[\-\.]\d+)\)", re.IGNORECASE
)
# Standalone equation references like (6-4) in context
EQUATION_PAREN_PATTERN = re.compile(
    r"\((\d+[\-\.]\d+)\)"
)


@dataclass
class Reference:
    """A reference to a figure, table, or equation found in text."""
    ref_type: str       # "figure", "table", "equation"
    number: str         # e.g., "2-3", "6.4"
    position: int       # character position in text
    raw_text: str       # e.g., "Figure 2-3"
    paragraph_idx: int = 0


@dataclass
class ReferenceMap:
    """Map of all references in a document."""
    figures: Dict[str, List[int]] = field(default_factory=dict)
    tables: Dict[str, List[int]] = field(default_factory=dict)
    equations: Dict[str, List[int]] = field(default_factory=dict)


class RenumberingService:
    """Service for detecting and validating cross-references in documents."""

    @staticmethod
    def find_references(text: str, paragraph_idx: int = 0) -> List[Reference]:
        """
        Scan text for figure, table, and equation references.

        Returns a list of Reference objects found in the text.
        """
        refs: List[Reference] = []

        for match in FIGURE_PATTERN.finditer(text):
            refs.append(Reference(
                ref_type="figure",
                number=match.group(1),
                position=match.start(),
                raw_text=match.group(0),
                paragraph_idx=paragraph_idx,
            ))

        for match in TABLE_PATTERN.finditer(text):
            refs.append(Reference(
                ref_type="table",
                number=match.group(1),
                position=match.start(),
                raw_text=match.group(0),
                paragraph_idx=paragraph_idx,
            ))

        for match in EQUATION_PATTERN.finditer(text):
            refs.append(Reference(
                ref_type="equation",
                number=match.group(1),
                position=match.start(),
                raw_text=match.group(0),
                paragraph_idx=paragraph_idx,
            ))

        return refs

    @staticmethod
    def build_reference_map(text_content: str) -> ReferenceMap:
        """
        Build a map of all references in a document's text.

        Returns a ReferenceMap where each key is a number (e.g., "2-3")
        and the value is a list of paragraph indices where it's referenced.
        """
        ref_map = ReferenceMap()
        paragraphs = text_content.split("\n")

        for idx, para in enumerate(paragraphs):
            if not para.strip():
                continue

            refs = RenumberingService.find_references(para, paragraph_idx=idx)
            for ref in refs:
                if ref.ref_type == "figure":
                    ref_map.figures.setdefault(ref.number, []).append(idx)
                elif ref.ref_type == "table":
                    ref_map.tables.setdefault(ref.number, []).append(idx)
                elif ref.ref_type == "equation":
                    ref_map.equations.setdefault(ref.number, []).append(idx)

        return ref_map

    @staticmethod
    def validate_references(
        text_content: str,
        figures: list,
        tables: list,
        equations: list,
    ) -> List[str]:
        """
        Check that all referenced figures, tables, and equations exist.

        Returns a list of warning messages for broken references.
        """
        warnings: List[str] = []
        ref_map = RenumberingService.build_reference_map(text_content)

        # Build sets of defined numbers
        defined_figures = set()
        for fig in figures:
            num = fig.get("number") if isinstance(fig, dict) else getattr(fig, "number", None)
            if num:
                defined_figures.add(num)

        defined_tables = set()
        for tbl in tables:
            num = tbl.get("number") if isinstance(tbl, dict) else getattr(tbl, "number", None)
            if num:
                defined_tables.add(num)

        defined_equations = set()
        for eq in equations:
            num = eq.get("number") if isinstance(eq, dict) else getattr(eq, "number", None)
            if num:
                # Strip parentheses: "(6-4)" → "6-4"
                clean = num.strip("()")
                defined_equations.add(clean)

        # Check figures
        for num, para_indices in ref_map.figures.items():
            if num not in defined_figures:
                warnings.append(
                    f"Broken reference: Figure {num} referenced in paragraph(s) "
                    f"{para_indices} but not found in extracted figures"
                )

        # Check tables
        for num, para_indices in ref_map.tables.items():
            if num not in defined_tables:
                warnings.append(
                    f"Broken reference: Table {num} referenced in paragraph(s) "
                    f"{para_indices} but not found in extracted tables"
                )

        # Check equations
        for num, para_indices in ref_map.equations.items():
            if num not in defined_equations:
                warnings.append(
                    f"Broken reference: Equation ({num}) referenced in paragraph(s) "
                    f"{para_indices} but not found in extracted equations"
                )

        if warnings:
            logger.warning(
                "Reference validation found %d broken references", len(warnings)
            )

        return warnings

    @staticmethod
    def renumber_after_changes(
        text: str,
        old_number: str,
        new_number: str,
        ref_type: str = "figure",
    ) -> str:
        """
        Update all references to a specific figure/table/equation number in text.

        Args:
            text: The document text
            old_number: The current number (e.g., "2-3")
            new_number: The new number (e.g., "2-4")
            ref_type: "figure", "table", or "equation"

        Returns:
            Updated text with references renumbered.
        """
        if ref_type == "figure":
            pattern = re.compile(
                rf"\b((?:Figure|Fig\.?)\s+){re.escape(old_number)}\b",
                re.IGNORECASE,
            )
            text = pattern.sub(rf"\g<1>{new_number}", text)

        elif ref_type == "table":
            pattern = re.compile(
                rf"\b((?:Table|Tbl\.?)\s+){re.escape(old_number)}\b",
                re.IGNORECASE,
            )
            text = pattern.sub(rf"\g<1>{new_number}", text)

        elif ref_type == "equation":
            pattern = re.compile(
                rf"\b((?:Equation|Eq\.?)\s*\(){re.escape(old_number)}(\))",
                re.IGNORECASE,
            )
            text = pattern.sub(rf"\g<1>{new_number}\g<2>", text)

        return text
