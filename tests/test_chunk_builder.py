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

        long_text = (
            "Primeiro parágrafo do documento com texto "
            "suficiente para indexação semântica"
        )
        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text=long_text,
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=0.99,
                )
            ]
        )
        chunks = build_chunks(document=doc, document_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc-1:1:p1_b1"
        assert "Primeiro parágrafo" in chunks[0].text
        assert chunks[0].source_file == "invoice.pdf"

    def test_builds_text_chunk_from_table_rows(self):
        from services.search.chunk_builder import build_chunks

        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_t1",
                    type=BlockType.TABLE,
                    text=None,
                    rows=[
                        ["item descricao completa do produto", "valor total em reais"],
                        ["taxa mensal referente ao serviço", "5% ao mês"],
                    ],
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=1.0,
                )
            ]
        )
        chunks = build_chunks(document=doc, document_id="doc-1")
        assert len(chunks) == 1
        assert "item descricao completa do produto" in chunks[0].text
        assert "taxa mensal referente ao serviço" in chunks[0].text

    def test_splits_large_table_into_multiple_chunks(self):
        from services.search.chunk_builder import MAX_TABLE_CHUNK_CHARS, build_chunks

        row = [
            "descricao detalhada do item de contrato para parcelamento bancario",
            "valor atualizado com juros e multa referente ao periodo processual",
        ]
        # Force split by exceeding MAX_TABLE_CHUNK_CHARS
        rows = [row for _ in range((MAX_TABLE_CHUNK_CHARS // 120) + 15)]
        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_t2",
                    type=BlockType.TABLE,
                    text=None,
                    rows=rows,
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=1.0,
                )
            ]
        )
        chunks = build_chunks(document=doc, document_id="doc-1")
        assert len(chunks) > 1
        assert chunks[0].chunk_id.endswith(":part1")
        assert chunks[1].chunk_id.endswith(":part2")

    def test_splits_large_paragraph_into_multiple_chunks(self):
        from services.search.chunk_builder import MAX_TEXT_CHUNK_CHARS, build_chunks

        long_text = (
            "clausula contratual relevante sobre dano moral e material " * 120
        ).strip()
        assert len(long_text) > MAX_TEXT_CHUNK_CHARS
        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_b9",
                    type=BlockType.PARAGRAPH,
                    text=long_text,
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=0.99,
                )
            ]
        )
        chunks = build_chunks(document=doc, document_id="doc-1")
        assert len(chunks) > 1
        assert chunks[0].chunk_id.endswith(":part1")
        assert chunks[1].chunk_id.endswith(":part2")

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
        chunks = build_chunks(document=doc, document_id="doc-1")
        assert not chunks

    def test_skips_short_chunks(self):
        from services.search.chunk_builder import MIN_CHUNK_CHARS, build_chunks

        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="fls. 24",
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=1.0,
                ),
                Block(
                    block_id="p1_b2",
                    type=BlockType.PARAGRAPH,
                    text="x" * (MIN_CHUNK_CHARS - 1),
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=1.0,
                ),
                Block(
                    block_id="p1_b3",
                    type=BlockType.PARAGRAPH,
                    text=(
                        "Este bloco tem conteúdo semântico "
                        "relevante e suficiente para indexação"
                    ),
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=0.95,
                ),
            ]
        )
        chunks = build_chunks(document=doc, document_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0].block_id == "p1_b3"

    def test_cleans_ocr_hash_prefix(self):
        from services.search.chunk_builder import build_chunks

        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text=(
                        ".dBmu9HEi Texto jurídico relevante "
                        "sobre inadimplemento contratual"
                    ),
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=1.0,
                ),
            ]
        )
        chunks = build_chunks(document=doc, document_id="doc-1")
        assert len(chunks) == 1
        assert not chunks[0].text.startswith(".dBmu9HEi")
        assert "Texto jurídico relevante" in chunks[0].text

    def test_skips_boilerplate_certidao_chunks(self):
        from services.search.chunk_builder import build_chunks

        doc = _doc_with_blocks(
            blocks=[
                Block(
                    block_id="p1_b2",
                    type=BlockType.PARAGRAPH,
                    text=(
                        "https://comunicaapi.pje.jus.br"
                        "/api/v1/comunicacao/abc/certidao "
                        "Código da certidão: abc "
                        "CERTIDÃO DE REMESSA DE RELAÇÃO"
                    ),
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=1.0,
                ),
            ]
        )
        chunks = build_chunks(document=doc, document_id="doc-1")
        assert not chunks
