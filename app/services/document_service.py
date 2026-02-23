# document_service.py
from app.utils.omml_to_latex import omml_to_latex
from docx import Document as DocxDocument
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph
from typing import List, Optional, Tuple
from app.models.document import Equation, Figure, Table, Position, DocumentMetadata
import base64
from io import BytesIO
from PIL import Image
import re
import fitz  # PyMuPDF
import tempfile
import os
from docx2pdf import convert

class DOCXParser:
    """Parser for DOCX documents with optional Nougat support via PDF conversion"""

    def __init__(self, file_path: str, use_nougat: bool = True):
        self.file_path = file_path
        self.use_nougat = use_nougat
        self.doc = DocxDocument(file_path)
        self.equations: List[Equation] = []
        self.figures: List[Figure] = []
        self.tables: List[Table] = []
        self.text_content = ""
        self._para_to_page = self._build_page_map()

    def _build_page_map(self) -> dict:
        """
        Build a mapping of paragraph index → page number by scanning for
        page break elements (<w:br w:type="page"/> and <w:lastRenderedPageBreak/>)
        in each paragraph's XML.
        """
        W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        para_to_page: dict[int, int] = {}
        current_page = 1

        for para_idx, para in enumerate(self.doc.paragraphs):
            para_to_page[para_idx] = current_page
            # Check for explicit page breaks: <w:br w:type="page"/>
            for br in para._element.findall(f'.//{{{W_NS}}}br'):
                if br.get(f'{{{W_NS}}}type') == 'page':
                    current_page += 1
                    para_to_page[para_idx] = current_page
            # Check for rendered page breaks: <w:lastRenderedPageBreak/>
            if para._element.findall(f'.//{{{W_NS}}}lastRenderedPageBreak'):
                current_page += 1
                para_to_page[para_idx] = current_page

        return para_to_page

    def _page_for_para(self, para_idx: Optional[int]) -> Optional[int]:
        """Get the page number for a given paragraph index."""
        if para_idx is None:
            return None
        return self._para_to_page.get(para_idx)

    def parse(self) -> Tuple[str, List[Equation], List[Figure], List[Table], DocumentMetadata]:
        """Parse DOCX document and extract all content"""

        if self.use_nougat:
            try:
                # Convert DOCX to PDF and use Nougat for better equation extraction
                print("Using Nougat mode: Converting DOCX to PDF...")
                return self._parse_with_nougat()
            except Exception as e:
                print(f"Nougat conversion failed, using standard DOCX parsing: {e}")
                return self._parse_standard()
        else:
            return self._parse_standard()

    def _parse_with_nougat(self) -> Tuple[str, List[Equation], List[Figure], List[Table], DocumentMetadata]:
        """Convert DOCX to PDF and use Nougat for extraction"""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
            pdf_path = tmp_pdf.name

        try:
            # Convert DOCX to PDF
            print(f"Converting {self.file_path} to PDF...")
            convert(self.file_path, pdf_path)
            print(f"Conversion complete: {pdf_path}")

            # Use PDFParser with Nougat
            pdf_parser = PDFParser(pdf_path, use_nougat=True)
            text, equations, figures, tables, metadata = pdf_parser.parse()

            # Store results
            self.text_content = text
            self.equations = equations
            self.figures = figures
            self.tables = tables

            return text, equations, figures, tables, metadata

        finally:
            # Clean up temporary PDF
            if os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                except:
                    pass

    def _parse_standard(self) -> Tuple[str, List[Equation], List[Figure], List[Table], DocumentMetadata]:
        """Standard DOCX parsing without Nougat"""
        # Extract text content
        self.text_content = self._extract_text()

        # Extract equations
        self.equations = self._extract_equations()

        # Extract figures (images)
        self.figures = self._extract_figures()

        # Extract tables
        self.tables = self._extract_tables()

        # Generate metadata
        metadata = self._generate_metadata()

        return self.text_content, self.equations, self.figures, self.tables, metadata
    
    def _extract_text(self) -> str:
        """Extract all text from document"""
        full_text = []
        for para in self.doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    
    def _extract_equations(self) -> List[Equation]:
        """
        Extract equations from DOCX by parsing OMML (Office Math Markup Language) XML.
        Finds all <m:oMath> elements which is where Word stores real equations.
        raw_omml is stored so it can later be converted to LaTeX via Mathpix or omml2latex.
        """
        from lxml import etree

        equations = []
        MATH_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
        equation_number_pattern = r'\([\d][\d+\-]*\)'  # e.g. (6-4), (1), (3+1)

        for para_idx, para in enumerate(self.doc.paragraphs):
            # Find every <m:oMath> block inside this paragraph
            # Covers both display equations (<m:oMathPara><m:oMath>)
            # and inline equations (<m:oMath> directly in a run)
            omath_elements = para._element.findall(f'.//{{{MATH_NS}}}oMath')

            for eq_idx, omath in enumerate(omath_elements):
                # Store raw OMML XML — ready for Mathpix or omml2latex conversion later
                omml_xml = etree.tostring(omath, encoding='unicode')

                # Build a best-effort plain-text representation from <m:t> leaf nodes
                t_elements = omath.findall(f'.//{{{MATH_NS}}}t')
                eq_text = ''.join(t.text or '' for t in t_elements).strip()

                # Look for an equation number like (6-4) in the surrounding paragraph text
                eq_number_match = re.search(equation_number_pattern, para.text)
                eq_number = eq_number_match.group(0) if eq_number_match else None

                # Convert OMML → LaTeX; fall back to plain text if conversion fails
                latex = omml_to_latex(omml_xml) or eq_text

                equations.append(Equation(
                    equation_id=f"eq_{para_idx}_{eq_idx}",
                    latex=latex,
                    raw_omml=omml_xml,
                    position=Position(page=self._page_for_para(para_idx), paragraph=para_idx),
                    number=eq_number
                ))

        return equations
    
    def _extract_figures(self) -> List[Figure]:
        """Extract images/figures from DOCX"""
        figures = []

        # Build relationship ID → paragraph index map
        # <a:blip r:embed="rIdX"> inside a <w:drawing> tells us which paragraph holds each image
        A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'
        R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        rel_id_to_para: dict = {}
        for para_idx, para in enumerate(self.doc.paragraphs):
            for blip in para._element.findall(f'.//{{{A_NS}}}blip'):
                r_embed = blip.get(f'{{{R_NS}}}embed')
                if r_embed:
                    rel_id_to_para[r_embed] = para_idx

        for idx, rel in enumerate(self.doc.part.rels.values()):
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    caption = self._find_figure_caption(idx)
                    para_pos = rel_id_to_para.get(rel.rId)

                    figure = Figure(
                        figure_id=f"fig_{idx}",
                        caption=caption,
                        image_base64=image_base64,
                        position=Position(page=self._page_for_para(para_pos), paragraph=para_pos),
                        number=self._extract_figure_number(caption) if caption else None
                    )
                    figures.append(figure)
                except Exception as e:
                    print(f"Error extracting figure {idx}: {e}")
                    continue

        return figures
    
    def _extract_tables(self) -> List[Table]:
        """Extract tables from DOCX"""
        tables = []

        # Walk ALL XML elements iteratively (avoids Python recursion limit on large docs).
        # Collect tables in document order alongside their para count.
        # Uses a list (not a dict with element keys) to avoid lxml proxy identity issues.
        tables_in_order = []   # list of (lxml_tbl_element, para_count) in document order
        para_count = 0

        # Iterative pre-order DFS: push children in reverse so left siblings pop first
        stack = list(reversed(list(self.doc.element.body)))
        while stack:
            element = stack.pop()
            local_tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            if local_tag == 'p':
                para_count += 1
            elif local_tag == 'tbl':
                tables_in_order.append((element, para_count))
            # Always recurse so nested tables inside cells are found too
            stack.extend(reversed(list(element)))

        # self.doc.tables returns tables in the same document (DFS) order,
        # so index i here matches index i in tables_in_order.
        for idx, table in enumerate(self.doc.tables):
            table_data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)

            caption = self._find_table_caption(idx)
            para_pos = tables_in_order[idx][1] if idx < len(tables_in_order) else None

            table_obj = Table(
                table_id=f"tbl_{idx}",
                caption=caption,
                content=table_data,
                position=Position(page=self._page_for_para(para_pos), paragraph=para_pos),
                number=self._extract_table_number(caption) if caption else None
            )
            tables.append(table_obj)

        return tables
    
    def _find_figure_caption(self, figure_idx: int) -> Optional[str]:
        """Find caption for a figure"""
        # Look through paragraphs for "Figure X-Y" pattern
        figure_pattern = r'Figure\s+\d+-\d+[:.]\s*(.+)'
        
        for para in self.doc.paragraphs:
            match = re.search(figure_pattern, para.text, re.IGNORECASE)
            if match:
                return para.text.strip()
        
        return None
    
    def _find_table_caption(self, table_idx: int) -> Optional[str]:
        """Find caption for a table"""
        table_pattern = r'Table\s+\d+-\d+[:.]\s*(.+)'
        
        for para in self.doc.paragraphs:
            match = re.search(table_pattern, para.text, re.IGNORECASE)
            if match:
                return para.text.strip()
        
        return None
    
    def _extract_figure_number(self, caption: Optional[str]) -> Optional[str]:
        """Extract figure number from caption"""
        if not caption:
            return None
        match = re.search(r'Figure\s+(\d+-\d+)', caption, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _extract_table_number(self, caption: Optional[str]) -> Optional[str]:
        """Extract table number from caption"""
        if not caption:
            return None
        match = re.search(r'Table\s+(\d+-\d+)', caption, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _get_page_count(self) -> int:
        """
        Count pages by counting explicit page-break XML elements (<w:br w:type="page"/>).
        Falls back to 1 if none are found.
        """
        W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        page_breaks = sum(
            1 for el in self.doc.element.body.iter(f'{{{W_NS}}}br')
            if el.get(f'{{{W_NS}}}type') == 'page'
        )
        return max(1, page_breaks + 1)

    def _generate_metadata(self) -> DocumentMetadata:
        """Generate document metadata including title and author from core properties"""
        props = self.doc.core_properties
        return DocumentMetadata(
            title=props.title or None,
            author=props.author or None,
            total_pages=self._get_page_count(),
            total_paragraphs=len(self.doc.paragraphs),
            total_equations=len(self.equations),
            total_figures=len(self.figures),
            total_tables=len(self.tables),
        )


class PDFParser:
    """Parser for PDF documents using Nougat OCR for scientific documents"""

    def __init__(self, file_path: str, use_nougat: bool = True):
        self.file_path = file_path
        self.use_nougat = use_nougat
        self.doc = fitz.open(file_path)
        self.equations: List[Equation] = []
        self.figures: List[Figure] = []
        self.tables: List[Table] = []
        self.text_content = ""
        self._nougat_model = None

    def parse(self) -> Tuple[str, List[Equation], List[Figure], List[Table], DocumentMetadata]:
        """Parse PDF document and extract all content"""
        if self.use_nougat:
            try:
                self._parse_with_nougat()
            except Exception as e:
                print(f"Nougat parsing failed, falling back to PyMuPDF: {e}")
                self._parse_with_pymupdf()
        else:
            self._parse_with_pymupdf()

        metadata = self._generate_metadata()
        return self.text_content, self.equations, self.figures, self.tables, metadata

    def _parse_with_nougat(self):
        """Use Nougat for advanced PDF parsing (equations, tables, etc.)"""
        try:
            from nougat import NougatModel
            from nougat.postprocessing import markdown_compatible
            from nougat.utils.checkpoint import get_checkpoint

            # Load model (lazy loading - only once)
            if self._nougat_model is None:
                print("Loading Nougat model...")
                checkpoint = get_checkpoint(None, model_tag="0.1.0-small")
                self._nougat_model = NougatModel.from_pretrained(checkpoint)
                print("Nougat model loaded")

            # Process PDF with Nougat
            print(f"Processing PDF with Nougat: {self.file_path}")
            output = self._nougat_model.predict(self.file_path)

            # Nougat returns markdown with LaTeX equations
            markdown_text = markdown_compatible(output)
            self.text_content = markdown_text

            # Extract equations from markdown (LaTeX is between $ or $$)
            self._extract_equations_from_markdown(markdown_text)

            # Extract figures using PyMuPDF (Nougat doesn't extract images)
            self._extract_figures_pymupdf()

            # Extract tables from markdown
            self._extract_tables_from_markdown(markdown_text)

        except ImportError:
            print("Nougat not installed, falling back to PyMuPDF")
            self._parse_with_pymupdf()
        except Exception as e:
            print(f"Nougat error: {e}")
            raise

    def _parse_with_pymupdf(self):
        """Fallback: Use PyMuPDF for basic PDF parsing"""
        self.text_content = self._extract_text_pymupdf()
        self.equations = self._extract_equations_pymupdf()
        self.figures = self._extract_figures_pymupdf()

    def _extract_text_pymupdf(self) -> str:
        """Extract all text from PDF using PyMuPDF"""
        full_text = []
        for page in self.doc:
            full_text.append(page.get_text())
        return "\n".join(full_text)

    def _extract_equations_pymupdf(self) -> List[Equation]:
        """Basic equation detection in PDF text"""
        equations = []
        equation_pattern = r'\(([\d+-]+)\)'
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            for line_idx, line in enumerate(text.split("\n")):
                if any(char in line for char in ['=', '∫', '∑', '√', 'μ', 'π']):
                    eq_number_match = re.search(equation_pattern, line)
                    eq_number = eq_number_match.group(0) if eq_number_match else None
                    equations.append(Equation(
                        equation_id=f"eq_p{page_num}_{line_idx}",
                        latex=line.strip(),
                        position=Position(page=page_num),
                        number=eq_number
                    ))
        return equations

    def _extract_figures_pymupdf(self) -> List[Figure]:
        """Extract images from PDF pages using PyMuPDF"""
        figures = []
        for page_num, page in enumerate(self.doc):
            for img_idx, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                    figures.append(Figure(
                        figure_id=f"fig_p{page_num}_{img_idx}",
                        image_base64=image_base64,
                        position=Position(page=page_num)
                    ))
                except Exception as e:
                    print(f"Error extracting PDF figure on page {page_num}: {e}")
                    continue
        return figures

    def _extract_equations_from_markdown(self, markdown: str):
        """Extract LaTeX equations from Nougat's markdown output"""
        # Match inline equations: $...$
        inline_pattern = r'\$([^$]+)\$'
        # Match display equations: $$...$$
        display_pattern = r'\$\$([^$]+)\$\$'

        # Extract display equations first
        for idx, match in enumerate(re.finditer(display_pattern, markdown)):
            latex = match.group(1).strip()
            # Calculate approximate position
            char_pos = match.start()
            line_num = markdown[:char_pos].count('\n') + 1
            page_num = self._estimate_page_from_line(line_num)

            self.equations.append(Equation(
                equation_id=f"eq_nougat_{idx}",
                latex=latex,
                position=Position(page=page_num, line=line_num),
                number=None
            ))

        # Extract inline equations
        offset = len(self.equations)
        for idx, match in enumerate(re.finditer(inline_pattern, markdown)):
            latex = match.group(1).strip()
            # Skip if already captured in display equations
            if not any(eq.latex == latex for eq in self.equations):
                # Calculate approximate position
                char_pos = match.start()
                line_num = markdown[:char_pos].count('\n') + 1
                page_num = self._estimate_page_from_line(line_num)

                self.equations.append(Equation(
                    equation_id=f"eq_nougat_inline_{idx}",
                    latex=latex,
                    position=Position(page=page_num, line=line_num),
                    number=None
                ))

    def _extract_tables_from_markdown(self, markdown: str):
        """Extract tables from Nougat's markdown output"""
        # Markdown tables have format:
        # | Header1 | Header2 |
        # |---------|---------|
        # | Cell1   | Cell2   |

        table_pattern = r'(\|.+\|[\r\n]+(?:\|[-:\s|]+\|[\r\n]+)?(?:\|.+\|[\r\n]*)+)'

        for idx, match in enumerate(re.finditer(table_pattern, markdown, re.MULTILINE)):
            table_text = match.group(1).strip()
            rows = [line.strip() for line in table_text.split('\n') if line.strip()]

            # Calculate approximate position
            char_pos = match.start()
            line_num = markdown[:char_pos].count('\n') + 1
            page_num = self._estimate_page_from_line(line_num)

            # Parse markdown table
            table_data = []
            for row in rows:
                if re.match(r'\|[-:\s|]+\|', row):  # Skip separator row
                    continue
                cells = [cell.strip() for cell in row.split('|')[1:-1]]
                table_data.append(cells)

            if table_data:
                self.tables.append(Table(
                    table_id=f"tbl_nougat_{idx}",
                    caption=None,
                    content=table_data,
                    position=Position(page=page_num, line=line_num),
                    number=None
                ))

    def _estimate_page_from_line(self, line_num: int) -> int:
        """Estimate page number from line number (rough approximation)"""
        # Assume ~40-50 lines per page on average
        LINES_PER_PAGE = 45
        page = max(0, (line_num - 1) // LINES_PER_PAGE)
        # Make sure page doesn't exceed total pages
        return min(page, len(self.doc) - 1) if len(self.doc) > 0 else 0

    def _generate_metadata(self) -> DocumentMetadata:
        """Generate PDF document metadata"""
        return DocumentMetadata(
            total_pages=len(self.doc),
            total_paragraphs=0,
            total_equations=len(self.equations),
            total_figures=len(self.figures),
            total_tables=len(self.tables)
        )