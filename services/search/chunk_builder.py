"""
Utilities to convert processed documents into semantic chunks.
"""

from __future__ import annotations

import re
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


MIN_CHUNK_CHARS = 50
MAX_TEXT_CHUNK_CHARS = 1200
MAX_TABLE_CHUNK_CHARS = 1200
MAX_TABLE_CHUNK_ROWS = 40

_LEADING_NOISE_RE = re.compile(r"^\.[A-Za-z0-9]{6,12}\s+")
_BOILERPLATE_PATTERNS = (
    re.compile(r"certid[aã]o de (publica[cç][aã]o|remessa) de rela[cç][aã]o", re.I),
    re.compile(r"c[oó]digo da certid[aã]o", re.I),
    re.compile(r"para conferir o original, acesse o site", re.I),
    re.compile(r"este documento [ée] c[oó]pia do original", re.I),
    re.compile(r"assinad[oa] digitalmente", re.I),
    re.compile(r"odanissa etnemlatigid", re.I),
    re.compile(r"otnemucod o rirefnoc", re.I),
)
_CERTIDAO_URL_RE = re.compile(r"https?://comunicaapi\.pje\.jus\.br/.*/certidao", re.I)


def _clean_text(text: str) -> str:
    """Strip leading OCR hash artifacts (e.g. '.dBmu9HEi CERTIDÃO...')."""
    return _LEADING_NOISE_RE.sub("", text)


def _table_to_chunks(rows: list[list[str]] | None) -> list[str]:
    """Split table rows into multiple semantic chunks."""
    if not rows:
        return []

    chunks: list[str] = []
    current_rows: list[str] = []
    current_chars = 0

    for row in rows:
        row_text = " ".join(cell.strip() for cell in row if cell and cell.strip())
        if not row_text:
            continue

        row_len = len(row_text) + 1
        next_chars = current_chars + row_len
        too_many_rows = len(current_rows) >= MAX_TABLE_CHUNK_ROWS
        too_many_chars = next_chars > MAX_TABLE_CHUNK_CHARS
        if current_rows and (too_many_rows or too_many_chars):
            chunks.append("\n".join(current_rows))
            current_rows = []
            current_chars = 0

        current_rows.append(row_text)
        current_chars += row_len

    if current_rows:
        chunks.append("\n".join(current_rows))
    return chunks


def _split_text_chunks(
    text: str, *, max_chars: int = MAX_TEXT_CHUNK_CHARS
) -> list[str]:
    """Split long text into smaller semantic chunks."""
    stripped = text.strip()
    if not stripped:
        return []

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    segments = lines or [stripped]

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for segment in segments:
        while len(segment) > max_chars:
            head = segment[:max_chars]
            split_at = head.rfind(" ")
            if split_at < max_chars // 2:
                split_at = max_chars
            piece = segment[:split_at].strip()
            if current:
                chunks.append("\n".join(current))
                current = []
                current_chars = 0
            if piece:
                chunks.append(piece)
            segment = segment[split_at:].strip()

        seg_len = len(segment) + 1
        if current and (current_chars + seg_len > max_chars):
            chunks.append("\n".join(current))
            current = []
            current_chars = 0

        if segment:
            current.append(segment)
            current_chars += seg_len

    if current:
        chunks.append("\n".join(current))
    return chunks


def _is_low_value_chunk(text: str) -> bool:
    """Heuristic filter for boilerplate chunks that hurt retrieval quality."""
    lowered = text.lower()
    match_count = sum(1 for pattern in _BOILERPLATE_PATTERNS if pattern.search(text))

    # Strong reversed boilerplate marker from OCR artifacts.
    if "otnemucod o rirefnoc" in lowered:
        return True

    if match_count >= 2:
        return True

    # Certidao URL stubs are usually metadata headers.
    if _CERTIDAO_URL_RE.search(text) and len(text) < 600:
        return True

    # Common non-semantic signature style blocks.
    return lowered.startswith("fls. ") and len(text) < 120


def build_chunks(document: Document, job_id: str) -> list[TextChunk]:
    """Flatten a Document into semantic chunks (text + table content).

    Chunks shorter than MIN_CHUNK_CHARS are skipped because they carry
    too little semantic signal (e.g. "fls. 24", page headers, short
    certidão titles).
    """
    chunks: list[TextChunk] = []
    for page in document.pages:
        for block in page.blocks:
            block_text = (block.text or "").strip()
            if block_text:
                candidates = _split_text_chunks(block_text)
            else:
                candidates = []
                for table_chunk in _table_to_chunks(block.rows):
                    candidates.extend(_split_text_chunks(table_chunk))

            total_parts = len(candidates)
            for idx, raw_text in enumerate(candidates, start=1):
                text = _clean_text(raw_text.strip())
                if len(text) < MIN_CHUNK_CHARS:
                    continue
                if _is_low_value_chunk(text):
                    continue

                part_suffix = f":part{idx}" if total_parts > 1 else ""
                chunks.append(
                    TextChunk(
                        chunk_id=f"{job_id}:{page.page}:{block.block_id}{part_suffix}",
                        job_id=job_id,
                        source_file=document.source_file,
                        page_number=page.page,
                        block_id=f"{block.block_id}{part_suffix}",
                        block_type=str(block.type),
                        text=text,
                        confidence=block.confidence,
                    )
                )
    return chunks
