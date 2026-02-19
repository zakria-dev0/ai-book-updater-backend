import requests
import base64
from typing import List, Optional
from app.core.config import settings
from app.models.document import Equation, Position

class MathpixService:
    """Service for extracting equations using Mathpix OCR API"""
    
    def __init__(self):
        self.api_url = "https://api.mathpix.com/v3/text"
        self.app_id = settings.MATHPIX_APP_ID
        self.app_key = settings.MATHPIX_APP_KEY
    
    def extract_equations_from_image(self, image_base64: str, page_num: int = None) -> List[Equation]:
        """
        Extract equations from an image using Mathpix
        
        Args:
            image_base64: Base64 encoded image
            page_num: Page number for position tracking
            
        Returns:
            List of extracted equations
        """
        headers = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "src": f"data:image/png;base64,{image_base64}",
            "formats": ["latex_styled"],
            "metadata": {
                "include_asciimath": True,
                "include_latex": True
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            latex = result.get("latex_styled", "")
            
            if latex:
                equation = Equation(
                    equation_id=f"eq_mathpix_{page_num}",
                    latex=latex,
                    image_base64=image_base64,
                    position=Position(page=page_num)
                )
                return [equation]
            
            return []
            
        except Exception as e:
            print(f"Mathpix API error: {e}")
            return []
    
    def extract_equations_from_pdf(self, pdf_path: str) -> List[Equation]:
        """
        Extract equations from entire PDF document
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of all equations found
        """
        # This would process PDF page by page
        # For now, return empty list - will implement in next iteration
        return []

# Alternative: Nougat-based equation extraction (open-source)
class NougatEquationExtractor:
    """
    Alternative equation extractor using Nougat model
    (Open-source alternative to Mathpix)
    """
    
    def __init__(self):
        # Will implement Nougat integration if Mathpix quota is exhausted
        pass
    
    def extract_equations(self, file_path: str) -> List[Equation]:
        """Extract equations using Nougat"""
        # TODO: Implement Nougat-based extraction
        return []