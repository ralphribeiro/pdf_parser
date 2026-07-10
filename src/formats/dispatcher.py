"""Format dispatcher for non-PDF and PDF document processing."""

from __future__ import annotations

from pathlib import Path

from src.formats.registry import DocumentFormat, detect_format
from src.models.schemas import Document


def process_file(
    path: str | Path,
    *,
    doc_id: str | None = None,
    extract_tables: bool = True,
    use_gpu: bool | None = None,
    parallel: bool | None = None,
) -> Document:
    """Process any supported input file and return the standard Document model."""
    source_path = Path(path)
    document_format = detect_format(source_path)

    if document_format is DocumentFormat.PDF:
        import config
        from src.pipeline import DocumentProcessor

        processor = DocumentProcessor(use_gpu=use_gpu)
        use_parallel = (
            parallel
            if parallel is not None
            else getattr(config, "PARALLEL_ENABLED", True)
        )
        if use_parallel:
            return processor.process_document_parallel(
                str(source_path), doc_id=doc_id, extract_tables=extract_tables
            )
        return processor.process_document(
            str(source_path), doc_id=doc_id, extract_tables=extract_tables
        )

    if document_format is DocumentFormat.IMAGE:
        from src.formats.image import extract_image

        return extract_image(source_path, doc_id=doc_id)

    if document_format is DocumentFormat.DOCX:
        from src.formats.docx import extract_docx

        return extract_docx(source_path, doc_id=doc_id)

    if document_format is DocumentFormat.ODT:
        from src.formats.odt import extract_odt

        return extract_odt(source_path, doc_id=doc_id)

    if document_format is DocumentFormat.TEXT:
        from src.formats.plain_text import extract_plain_text

        return extract_plain_text(source_path, doc_id=doc_id)

    raise ValueError(f"Unsupported format: {document_format}")
