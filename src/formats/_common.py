"""Helpers shared by non-PDF format extractors."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from src.models.schemas import Block, BlockType, Document, Page

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0


def build_digital_document(
    path: str | Path,
    paragraphs: Iterable[str],
    *,
    doc_id: str | None = None,
) -> Document:
    """Build a single virtual-page Document from ordered text paragraphs."""
    source_path = Path(path)
    clean_paragraphs = [p.strip() for p in paragraphs if p and p.strip()]
    blocks = _blocks_from_paragraphs(clean_paragraphs)

    page = Page(
        page=1,
        source="digital",
        blocks=blocks,
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
    )
    return Document(
        doc_id=doc_id or source_path.stem,
        source_file=source_path.name,
        total_pages=1,
        processing_date=datetime.now(),
        pages=[page],
    )


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs using one or more blank lines."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip():
            current.append(line.rstrip())
            continue
        if current:
            paragraphs.append("\n".join(current).strip())
            current = []
    if current:
        paragraphs.append("\n".join(current).strip())
    return paragraphs


def _blocks_from_paragraphs(paragraphs: list[str]) -> list[Block]:
    if not paragraphs:
        return []

    top_margin = 0.06
    bottom_margin = 0.06
    gap = 0.015
    usable_height = max(0.1, 1.0 - top_margin - bottom_margin)
    block_height = max(
        0.03,
        (usable_height - gap * (len(paragraphs) - 1)) / len(paragraphs),
    )

    blocks: list[Block] = []
    y1 = top_margin
    for idx, paragraph in enumerate(paragraphs, start=1):
        y2 = min(0.98, y1 + block_height)
        blocks.append(
            Block(
                block_id=f"p1_b{idx}",
                type=BlockType.PARAGRAPH,
                text=paragraph,
                bbox=[0.08, y1, 0.92, y2],
                confidence=1.0,
            )
        )
        y1 = y2 + gap
    return blocks
