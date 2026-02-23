# tests/test_document_service.py

import pytest
from app.services.document_service import DOCXParser

def test_docx_parser_text_extraction():
    """Test text extraction from DOCX"""
    parser = DOCXParser("tests/fixtures/sample.docx")
    text, _, _, _, metadata = parser.parse()
    
    assert text is not None
    assert len(text) > 0
    assert metadata.total_paragraphs > 0

def test_docx_parser_figure_extraction():
    """Test figure extraction from DOCX"""
    parser = DOCXParser("tests/fixtures/sample_with_images.docx")
    _, _, figures, _, metadata = parser.parse()
    
    assert len(figures) > 0
    assert metadata.total_figures == len(figures)

def test_docx_parser_table_extraction():
    """Test table extraction from DOCX"""
    parser = DOCXParser("tests/fixtures/sample_with_tables.docx")
    _, _, _, tables, metadata = parser.parse()
    
    assert len(tables) > 0
    assert metadata.total_tables == len(tables)