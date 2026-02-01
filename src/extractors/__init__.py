"""
Módulos de extração de conteúdo
"""
from .digital import extract_digital_page
from .ocr import extract_ocr_page, OCREngine
from .tables import extract_tables_digital

__all__ = [
    'extract_digital_page',
    'extract_ocr_page',
    'OCREngine',
    'extract_tables_digital'
]
