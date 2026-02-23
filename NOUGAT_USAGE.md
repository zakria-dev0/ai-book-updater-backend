# Nougat Integration Guide

## What is Nougat?
Nougat is Meta's neural OCR model specifically designed for scientific documents. It excels at:
- Extracting LaTeX equations from PDFs
- Converting tables to markdown format
- Understanding academic paper structure
- Better accuracy for mathematical notation

## ✨ NEW: Now Works with DOCX Files Too!

Your system now automatically converts DOCX files to PDF and uses Nougat for better equation extraction!

## How It Works in Your Project

### DOCXParser with Nougat (NEW!)

```python
from app.services.document_service import DOCXParser

# DOCX is automatically converted to PDF, then Nougat extracts equations
parser = DOCXParser("document.docx", use_nougat=True)  # Default
text, equations, figures, tables, metadata = parser.parse()

# Or use basic DOCX parsing (no PDF conversion)
parser = DOCXParser("document.docx", use_nougat=False)
```

**How it works:**
1. DOCX → PDF conversion (using docx2pdf)
2. PDF → Nougat processing (LaTeX equation extraction)
3. Returns properly formatted equations

### PDFParser with Nougat

The `PDFParser` class now supports two modes:

**1. Nougat Mode (Default)** - Advanced AI-powered extraction
```python
from app.services.document_service import PDFParser

# Use Nougat for better equation/table extraction
parser = PDFParser("scientific_paper.pdf", use_nougat=True)
text, equations, figures, tables, metadata = parser.parse()
```

**2. PyMuPDF Mode (Fallback)** - Traditional text extraction
```python
# Fallback to basic PyMuPDF extraction
parser = PDFParser("document.pdf", use_nougat=False)
text, equations, figures, tables, metadata = parser.parse()
```

## Features

### Extracted from Nougat:
- **Text**: Full markdown-formatted text
- **Equations**: LaTeX equations extracted from `$...$` and `$$...$$`
- **Tables**: Parsed from markdown tables
- **Figures**: Still uses PyMuPDF (Nougat doesn't extract images)

### Automatic Fallback:
If Nougat fails (missing model, errors, etc.), automatically falls back to PyMuPDF

## API Endpoint Usage

When you upload any document via `/api/v1/upload/` and process via `/api/v1/documents/{id}/process`:

```python
# In processing.py, both parsers use Nougat by default:

if document["file_type"] == "docx":
    parser = DOCXParser(document["file_path"])  # Converts to PDF, then uses Nougat
    text, equations, figures, tables, metadata = parser.parse()

elif document["file_type"] == "pdf":
    parser = PDFParser(document["file_path"])  # Uses Nougat directly
    text, equations, figures, tables, metadata = parser.parse()
```

**Result:** Both DOCX and PDF files now get AI-powered equation extraction! 🎉

## Model Download

**First Run:**
- Nougat will automatically download the model (~1.5GB) on first use
- Model is cached at: `~/.cache/huggingface/hub/`
- Subsequent runs will be much faster

## Performance Notes

- **First PDF**: ~30-60 seconds (model loading + processing)
- **Subsequent PDFs**: ~10-30 seconds per page
- **Memory**: Requires ~2-4GB RAM for model
- **Best for**: Scientific papers, academic documents, books with equations

## Disable Nougat (if needed)

To disable Nougat and use only PyMuPDF:

```python
# In app/api/processing.py, modify the PDFParser initialization:
parser = PDFParser(document["file_path"], use_nougat=False)
```

## Troubleshooting

**Model not downloading:**
```bash
# Manually trigger download
python -c "from nougat.utils.checkpoint import get_checkpoint; get_checkpoint(None, model_tag='0.1.0-small')"
```

**Memory issues:**
- Use smaller model: Change `model_tag="0.1.0-small"` in code
- Process PDFs page-by-page if needed

**Slow processing:**
- Nougat is compute-intensive
- Consider processing in background queue for production
- Use PyMuPDF mode for simple documents
