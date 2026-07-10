"""Plain text and Markdown extractors."""

from __future__ import annotations

from pathlib import Path

from src.formats._common import build_digital_document, split_paragraphs
from src.models.schemas import Document


def extract_plain_text(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Convert a TXT/MD file into a single digital virtual page."""
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8-sig")
    return build_digital_document(source_path, split_paragraphs(text), doc_id=doc_id)
