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

class DOCXParser:
    """Parser for DOCX documents"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc = DocxDocument(file_path)
        self.equations: List[Equation] = []
        self.figures: List[Figure] = []
        self.tables: List[Table] = []
        self.text_content = ""
    
    def parse(self) -> Tuple[str, List[Equation], List[Figure], List[Table], DocumentMetadata]:
        """Parse DOCX document and extract all content"""
        
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
        """Extract images/figures from DOCX"""
        figures = []
        
        # Get all image relationships
        for idx, rel in enumerate(self.doc.part.rels.values()):
            if "image" in rel.target_ref:
                try:
                    # Get image data
                    image_data = rel.target_part.blob
                    
                    # Convert to base64
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    # Try to find caption (look for nearby text mentioning "Figure")
                    caption = self._find_figure_caption(idx)
                    
                    figure = Figure(
                        figure_id=f"fig_{idx}",
                        caption=caption,
                        image_base64=image_base64,
                        position=Position(),  # Will be refined
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
        
        for idx, table in enumerate(self.doc.tables):
            # Extract table data as 2D array
            table_data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)
            
            # Try to find caption
            caption = self._find_table_caption(idx)
            
            table_obj = Table(
                table_id=f"tbl_{idx}",
                caption=caption,
                content=table_data,
                position=Position(),  # Will be refined
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
    
    def _generate_metadata(self) -> DocumentMetadata:
        """Generate document metadata"""
        return DocumentMetadata(
            total_pages=len(self.doc.sections),  # Approximation
            total_paragraphs=len(self.doc.paragraphs),
            total_equations=len(self.equations),
            total_figures=len(self.figures),
            total_tables=len(self.tables)
        )