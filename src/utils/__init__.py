"""
Utilitários
"""
from .bbox import normalize_bbox, sort_blocks_by_position
from .text_normalizer import normalize_text, merge_hyphenated_words
from .ocr_postprocess import (
    clean_ocr_text,
    postprocess_ocr_text,
    fix_common_ocr_errors,
    remove_short_lines
)

__all__ = [
    'normalize_bbox',
    'sort_blocks_by_position',
    'normalize_text',
    'merge_hyphenated_words',
    'clean_ocr_text',
    'postprocess_ocr_text',
    'fix_common_ocr_errors',
    'remove_short_lines'
]
