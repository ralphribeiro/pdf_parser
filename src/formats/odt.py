"""ODT extractor."""

from __future__ import annotations

from pathlib import Path

from src.formats._common import build_digital_document
from src.models.schemas import Document

try:
    from odf.opendocument import load
    from odf.teletype import extractText
    from odf.text import P
except ImportError:
    load = None
    extractText = None
    P = None


def extract_odt(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Convert ODT paragraphs into a single digital virtual page."""
    if load is None or extractText is None:
        raise RuntimeError("ODT support requires the odfpy package")

    source_path = Path(path)
    odt_document = load(str(source_path))
    paragraphs = [
        extractText(paragraph) for paragraph in odt_document.getElementsByType(P)
    ]
    return build_digital_document(source_path, paragraphs, doc_id=doc_id)
