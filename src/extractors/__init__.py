"""
Content extraction modules
"""

from .digital import extract_digital_page
from .ocr import DocTREngine, OCREngine, extract_ocr_page
from .tables import extract_tables_digital

# Conditional Tesseract import (requires separate installation)
try:
    from .ocr_tesseract import TesseractEngine, extract_ocr_page_tesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    TesseractEngine = None
    extract_ocr_page_tesseract = None

__all__ = [
    "TESSERACT_AVAILABLE",
    "DocTREngine",
    "OCREngine",
    "TesseractEngine",
    "extract_digital_page",
    "extract_ocr_page",
    "extract_ocr_page_tesseract",
    "extract_tables_digital",
]
