"""
Content extraction modules
"""
from .digital import extract_digital_page
from .ocr import extract_ocr_page, OCREngine, DocTREngine
from .tables import extract_tables_digital

# Conditional Tesseract import (requires separate installation)
try:
    from .ocr_tesseract import extract_ocr_page_tesseract, TesseractEngine
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    TesseractEngine = None
    extract_ocr_page_tesseract = None

__all__ = [
    'extract_digital_page',
    'extract_ocr_page',
    'OCREngine',
    'DocTREngine',
    'extract_tables_digital',
    'TesseractEngine',
    'extract_ocr_page_tesseract',
    'TESSERACT_AVAILABLE'
]
