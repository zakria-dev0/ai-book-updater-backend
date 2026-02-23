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

class DOCXParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            self.doc = DocxDocument(file_path)
        except Exception as e:
            raise ValueError(f"Invalid or corrupted DOCX file: {str(e)}")
    
    def parse(self) -> Tuple[str, List[Equation], List[Figure], List[Table], DocumentMetadata]:
        """Parse DOCX document and extract all content"""
        
        # Initialize empty lists first
        self.equations = []
        self.figures = []
        self.tables = []
        self.text_content = ""
        
        # Extract text content
        try:
            self.text_content = self._extract_text()
        except Exception as e:
            print(f"Error extracting text: {e}")
            self.text_content = ""
        
        # Extract equations (placeholder for now - will implement with Mathpix later)
        try:
            self.equations = self._extract_equations()
        except Exception as e:
            print(f"Error extracting equations: {e}")
            self.equations = []
        
        # Extract figures (images)
        try:
            self.figures = self._extract_figures()
        except Exception as e:
            print(f"Error extracting figures: {e}")
            self.figures = []
        
        # Extract tables
        try:
            self.tables = self._extract_tables()
        except Exception as e:
            print(f"Error extracting tables: {e}")
            self.tables = []
        
        # Generate metadata
        try:
            metadata = self._generate_metadata()
        except Exception as e:
            print(f"Error generating metadata: {e}")
            metadata = DocumentMetadata(
                total_pages=0,
                total_paragraphs=0,
                total_equations=0,
                total_figures=0,
                total_tables=0
            )
        
        return self.text_content, self.equations, self.figures, self.tables, metadata
    
    def _extract_text(self) -> str:
        """Extract all text from document"""
        full_text = []
        for para in self.doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    
    def _extract_equations(self) -> List[Equation]:
        """
        Extract equations from DOCX
        Note: DOCX equations are complex - this is a basic implementation
        For production, we'll use Mathpix API in next task
        """
        equations = []
        equation_pattern = r'\(([\d+-]+)\)'  # Matches equation numbers like (6-4)
        
        for idx, para in enumerate(self.doc.paragraphs):
            # Check if paragraph contains equation-like content
            # This is a simplified approach - real implementation needs Mathpix
            if any(char in para.text for char in ['=', '∫', '∑', '√', 'μ', 'π']):
                # Extract equation number if present
                eq_number_match = re.search(equation_pattern, para.text)
                eq_number = eq_number_match.group(0) if eq_number_match else None
                
                equation = Equation(
                    equation_id=f"eq_{idx}",
                    latex=para.text,  # Placeholder - will be replaced by Mathpix
                    position=Position(paragraph=idx),
                    number=eq_number
                )
                equations.append(equation)
        
        return equations
    
    def _extract_figures(self) -> List[Figure]:
        """Extract images/figures from DOCX with position tracking"""
        figures = []
        
        # Map images to their positions in document
        image_rels = {}
        for rel in self.doc.part.rels.values():
            if "image" in rel.target_ref:
                image_rels[rel.rId] = rel
        
        # Find images in paragraphs and track position
        for para_idx, para in enumerate(self.doc.paragraphs):
            # Check for images in this paragraph
            for run in para.runs:
                if 'graphic' in run._element.xml:
                    # Extract image from this run
                    for rel_id, rel in image_rels.items():
                        try:
                            image_data = rel.target_part.blob
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                            
                            # Find caption nearby
                            caption = self._find_figure_caption_near_paragraph(para_idx)
                            
                            figure = Figure(
                                figure_id=f"fig_{len(figures)}",
                                caption=caption,
                                image_base64=image_base64,
                                position=Position(paragraph=para_idx),  # ✅ Now tracked
                                number=self._extract_figure_number(caption) if caption else None
                            )
                            figures.append(figure)
                            break
                        except:
                            continue
        
        return figures
    
    def _find_figure_caption_near_paragraph(self, para_idx: int) -> Optional[str]:
        """Find figure caption within +/- 2 paragraphs"""
        start = max(0, para_idx - 2)
        end = min(len(self.doc.paragraphs), para_idx + 3)
        
        for i in range(start, end):
            para_text = self.doc.paragraphs[i].text
            if re.search(r'Figure\s+\d+-\d+', para_text, re.IGNORECASE):
                return para_text.strip()
        
        return None
    
    def _extract_tables(self) -> List[Table]:
        """Extract tables from DOCX with position tracking"""
        tables = []
        
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        
        try:
            body_elements = list(self.doc.element.body)
            paragraph_counter = 0
            
            for idx, element in enumerate(body_elements):
                if isinstance(element, CT_P):  # Paragraph
                    paragraph_counter += 1
                
                elif isinstance(element, CT_Tbl):  # Table
                    try:
                        table = self.doc.tables[len(tables)]  # Get actual table object
                        
                        # Extract table data as 2D array
                        table_data = []
                        for row in table.rows:
                            row_data = [cell.text.strip() for cell in row.cells]
                            table_data.append(row_data)
                        
                        # Find caption
                        caption = self._find_table_caption_near_element(idx, body_elements)
                        
                        table_obj = Table(
                            table_id=f"tbl_{len(tables)}",
                            caption=caption,
                            content=table_data,
                            position=Position(paragraph=paragraph_counter),
                            number=self._extract_table_number(caption) if caption else None
                        )
                        tables.append(table_obj)
                    except Exception as e:
                        print(f"Error extracting table {len(tables)}: {e}")
                        continue
        except Exception as e:
            print(f"Error in table extraction: {e}")
        
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
    
    def _generate_metadata(self) -> DocumentMetadata:
        """Generate document metadata"""
        return DocumentMetadata(
            total_pages=len(self.doc.sections),  # Approximation
            total_paragraphs=len(self.doc.paragraphs),
            total_equations=len(self.equations),
            total_figures=len(self.figures),
            total_tables=len(self.tables)
        )
    
    def _find_table_caption_near_element(self, element_idx: int, body_elements) -> Optional[str]:
        """
        Find table caption near a table element
        
        Args:
            element_idx: Index of table in body elements
            body_elements: All body elements (paragraphs and tables)
            
        Returns:
            Caption text if found, None otherwise
        """
        from docx.oxml.text.paragraph import CT_P
        
        # Check 2 paragraphs before and after the table
        start_idx = max(0, element_idx - 2)
        end_idx = min(len(body_elements), element_idx + 3)
        
        for i in range(start_idx, end_idx):
            element = body_elements[i]
            if isinstance(element, CT_P):
                # Get paragraph text
                para_text = ""
                for text_element in element.itertext():
                    para_text += text_element
                
                # Check if it contains "Table"
                if re.search(r'Table\s+\d+-\d+', para_text, re.IGNORECASE):
                    return para_text.strip()
        
        return None


class PDFParser:
    """Parser for PDF documents using PyMuPDF"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc = fitz.open(file_path)
        self.equations: List[Equation] = []
        self.figures: List[Figure] = []
        self.tables: List[Table] = []
        self.text_content = ""

    def parse(self) -> Tuple[str, List[Equation], List[Figure], List[Table], DocumentMetadata]:
        """Parse PDF document and extract all content"""
        self.text_content = self._extract_text()
        self.equations = self._extract_equations()
        self.figures = self._extract_figures()
        metadata = self._generate_metadata()
        return self.text_content, self.equations, self.figures, self.tables, metadata

    def _extract_text(self) -> str:
        """Extract all text from PDF"""
        full_text = []
        for page in self.doc:
            full_text.append(page.get_text())
        return "\n".join(full_text)

    def _extract_equations(self) -> List[Equation]:
        """Basic equation detection in PDF text"""
        equations = []
        equation_pattern = r'\(([\d+-]+)\)'
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            for line in text.split("\n"):
                if any(char in line for char in ['=', '∫', '∑', '√', 'μ', 'π']):
                    eq_number_match = re.search(equation_pattern, line)
                    eq_number = eq_number_match.group(0) if eq_number_match else None
                    equations.append(Equation(
                        equation_id=f"eq_p{page_num}_{len(equations)}",
                        latex=line.strip(),
                        position=Position(page=page_num),
                        number=eq_number
                    ))
        return equations

    def _extract_figures(self) -> List[Figure]:
        """Extract images from PDF pages"""
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

    def _generate_metadata(self) -> DocumentMetadata:
        """Generate PDF document metadata"""
        return DocumentMetadata(
            total_pages=len(self.doc),
            total_paragraphs=0,
            total_equations=len(self.equations),
            total_figures=len(self.figures),
            total_tables=len(self.tables)
        )