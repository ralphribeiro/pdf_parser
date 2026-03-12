"""
Tests for Document -> semantic chunks conversion.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.schemas import Block, BlockType, Document, Page


def _doc_with_blocks(*, blocks):
    return Document(
        doc_id="doc-1",
        source_file="invoice.pdf",
        total_pages=1,
        processing_date=datetime.now(),
        pages=[Page(page=1, source="digital", blocks=blocks)],
    )


class TestChunkBuilder:
    def test_builds_text_chunk_from_paragraph(self):
        from services.search.chunk_builder import build_chunks

        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="Primeiro parágrafo",
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=0.99,
                )
            ]
        )
        chunks = build_chunks(document=doc, job_id="job-1")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "job-1:1:p1_b1"
        assert chunks[0].text == "Primeiro parágrafo"
        assert chunks[0].source_file == "invoice.pdf"

    def test_builds_text_chunk_from_table_rows(self):
        from services.search.chunk_builder import build_chunks

        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_t1",
                    type=BlockType.TABLE,
                    text=None,
                    rows=[["item", "valor"], ["taxa", "5%"]],
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=1.0,
                )
            ]
        )
        chunks = build_chunks(document=doc, job_id="job-1")
        assert len(chunks) == 1
        assert "item valor" in chunks[0].text
        assert "taxa 5%" in chunks[0].text

    def test_skips_empty_chunks(self):
        from services.search.chunk_builder import build_chunks

        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="   ",
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=0.8,
                ),
                Block(
                    block_id="p1_t1",
                    type=BlockType.TABLE,
                    text=None,
                    rows=[],
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=1.0,
                ),
            ]
        )
        chunks = build_chunks(document=doc, job_id="job-1")
        assert not chunks
