"""
Utilities to convert processed documents into semantic chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.schemas import Document


@dataclass
class TextChunk:
    # pylint: disable=too-many-instance-attributes
    """Chunk payload ready for embedding/vector indexing."""

    chunk_id: str
    job_id: str
    source_file: str
    page_number: int
    block_id: str
    block_type: str
    text: str
    confidence: float


def _table_to_text(rows: list[list[str]] | None) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for row in rows:
        row_text = " ".join(cell.strip() for cell in row if cell and cell.strip())
        if row_text:
            lines.append(row_text)
    return "\n".join(lines)


def build_chunks(document: Document, job_id: str) -> list[TextChunk]:
    """Flatten a Document into semantic chunks (text + table content)."""
    chunks: list[TextChunk] = []
    for page in document.pages:
        for block in page.blocks:
            text = (block.text or "").strip()
            if not text:
                text = _table_to_text(block.rows).strip()
            if not text:
                continue

            chunks.append(
                TextChunk(
                    chunk_id=f"{job_id}:{page.page}:{block.block_id}",
                    job_id=job_id,
                    source_file=document.source_file,
                    page_number=page.page,
                    block_id=block.block_id,
                    block_type=str(block.type),
                    text=text,
                    confidence=block.confidence,
                )
            )
    return chunks
