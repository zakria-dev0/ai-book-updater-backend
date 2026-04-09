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
# Page number references: "page 145", "pages 10-15", "p. 42", "pp. 10-15"
PAGE_REF_PATTERN = re.compile(
    r"\b(?:pages?|pp?\.?)\s+(\d{1,5})(?:\s*[-–—]\s*(\d{1,5}))?\b",
    re.IGNORECASE,
)


@dataclass
class Reference:
    """A reference to a figure, table, or equation found in text."""
    ref_type: str       # "figure", "table", "equation", "page"
    number: str         # e.g., "2-3", "6.4", "145"
    position: int       # character position in text
    raw_text: str       # e.g., "Figure 2-3", "page 145"
    paragraph_idx: int = 0
    end_number: str = ""  # for page ranges: "147" in "pages 145-147"


@dataclass
class ReferenceMap:
    """Map of all references in a document."""
    figures: Dict[str, List[int]] = field(default_factory=dict)
    tables: Dict[str, List[int]] = field(default_factory=dict)
    equations: Dict[str, List[int]] = field(default_factory=dict)
    pages: Dict[str, List[int]] = field(default_factory=dict)


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

        for match in PAGE_REF_PATTERN.finditer(text):
            refs.append(Reference(
                ref_type="page",
                number=match.group(1),
                position=match.start(),
                raw_text=match.group(0),
                paragraph_idx=paragraph_idx,
                end_number=match.group(2) or "",
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
                elif ref.ref_type == "page":
                    page_key = f"{ref.number}-{ref.end_number}" if ref.end_number else ref.number
                    ref_map.pages.setdefault(page_key, []).append(idx)

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

    @staticmethod
    def renumber_after_insertion(
        text: str,
        existing_numbers: list,
        inserted_number: str,
        ref_type: str = "figure",
    ) -> tuple:
        """Renumber references after a NEW item is inserted at a given position.

        When a new Figure/Table is inserted at e.g. "3-2", all existing items
        with number >= "3-2" in the same chapter are incremented by 1:
            old 3-2 -> 3-3, old 3-3 -> 3-4, etc.

        Args:
            text: Full document text
            existing_numbers: All current numbers in document e.g. ["3-1","3-2","3-3"]
            inserted_number: Number at which new item is being inserted e.g. "3-2"
            ref_type: "figure", "table", or "equation"

        Returns:
            (updated_text, number_map) where number_map is {old: new}
        """
        num_pat = re.compile(r'^(\d+)([\-\.])(\d+)$')
        m = num_pat.match(inserted_number)
        if not m:
            return text, {}
        ins_chap, ins_sep, ins_seq = m.group(1), m.group(2), int(m.group(3))

        # Find items in the same chapter with seq >= inserted seq, in reverse
        # (reverse so renaming 3-2 -> 3-3 doesn't collide with the real 3-3
        #  which also needs to become 3-4)
        affected = []
        for num in existing_numbers:
            nm = num_pat.match(num)
            if not nm:
                continue
            chap, sep, seq = nm.group(1), nm.group(2), int(nm.group(3))
            if chap == ins_chap and sep == ins_sep and seq >= ins_seq:
                affected.append((seq, num))

        # Sort descending so we rename 3-4 -> 3-5 first, then 3-3 -> 3-4, then 3-2 -> 3-3
        affected.sort(key=lambda x: x[0], reverse=True)

        number_map = {}
        for seq, old_num in affected:
            new_num = f"{ins_chap}{ins_sep}{seq + 1}"
            number_map[old_num] = new_num
            text = RenumberingService.renumber_after_changes(
                text, old_num, new_num, ref_type
            )

        if number_map:
            logger.info(
                "Insertion renumbering (%s): inserted %s, shifted %d items — %s",
                ref_type, inserted_number, len(number_map), number_map,
            )

        return text, number_map

    @staticmethod
    def renumber_sequential(
        text: str,
        ref_type: str,
        existing_numbers: list,
        removed_numbers: set,
    ) -> tuple:
        """Renumber all references sequentially after items are added/removed.

        Args:
            text: Full document text
            ref_type: "figure", "table", or "equation"
            existing_numbers: Sorted list of all current numbers e.g. ["1-1", "1-2", "1-3"]
            removed_numbers: Set of numbers that were removed

        Returns:
            (updated_text, number_map) where number_map is {old: new}
        """
        if not removed_numbers:
            return text, {}

        # Group by chapter prefix
        num_pat = re.compile(r'^(\d+)([\-\.])(\d+)$')
        chapter_groups = {}

        for num in existing_numbers:
            m = num_pat.match(num)
            if m:
                chap, sep, seq = m.group(1), m.group(2), int(m.group(3))
                chapter_groups.setdefault((chap, sep), []).append((seq, num))

        number_map = {}
        for (chap, sep), items in chapter_groups.items():
            items.sort(key=lambda x: x[0])
            remaining = [(seq, num) for seq, num in items if num not in removed_numbers]
            for new_idx, (_, old_num) in enumerate(remaining, start=1):
                new_num = f"{chap}{sep}{new_idx}"
                if new_num != old_num:
                    number_map[old_num] = new_num

        # Apply all renumbering to text
        for old_num, new_num in number_map.items():
            text = RenumberingService.renumber_after_changes(
                text, old_num, new_num, ref_type
            )

        if number_map:
            logger.info("Sequential renumbering (%s): %d items renumbered — %s",
                       ref_type, len(number_map), number_map)

        return text, number_map

    @staticmethod
    def find_page_references(text: str) -> List[Reference]:
        """Find all hardcoded page number references in text.

        Returns list of Reference objects for page references like
        "page 145", "pages 10-15", "p. 42".
        """
        refs = []
        for match in PAGE_REF_PATTERN.finditer(text):
            refs.append(Reference(
                ref_type="page",
                number=match.group(1),
                position=match.start(),
                raw_text=match.group(0),
                end_number=match.group(2) or "",
            ))
        return refs
