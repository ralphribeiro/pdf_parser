"""DOCX extractor."""

from __future__ import annotations

from pathlib import Path

from src.formats._common import build_digital_document
from src.models.schemas import Document

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


def extract_docx(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Convert DOCX paragraphs into a single digital virtual page."""
    if DocxDocument is None:
        raise RuntimeError("DOCX support requires the python-docx package")

    source_path = Path(path)
    docx_document = DocxDocument(str(source_path))
    paragraphs = [paragraph.text for paragraph in docx_document.paragraphs]
    return build_digital_document(source_path, paragraphs, doc_id=doc_id)
