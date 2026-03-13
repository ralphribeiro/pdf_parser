"""
Tests for semantic search service orchestration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingest_api.schemas import SearchResult
from services.search.service import SemanticSearchService


class _FakeEmbeddingClient:
    def embed_texts(self, texts):  # pragma: no cover - not used in these tests
        return [[0.1] for _ in texts]

    def embed_query(self, query):
        return [0.1, 0.2, 0.3]


class _FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def upsert_chunks(self, chunks, embeddings):  # pragma: no cover - not used
        _ = (chunks, embeddings)

    def query(
        self, query_embedding, n_results, *, document_id=None, min_similarity=None
    ):
        _ = (query_embedding, n_results, document_id, min_similarity)
        return self.results


class TestSemanticSearchService:
    def test_filters_results_without_keyword_overlap(self):
        service = SemanticSearchService(
            embedding_client=_FakeEmbeddingClient(),
            vector_store=_FakeVectorStore(
                [
                    SearchResult(
                        chunk_id="c1",
                        text="Saldo devedor e juros contratuais",
                        similarity=0.8,
                        document_id="j1",
                        source_file="a.pdf",
                        page_number=1,
                        block_id="b1",
                        block_type="paragraph",
                        confidence=1.0,
                    ),
                    SearchResult(
                        chunk_id="c2",
                        text="Pedido de dano moral por negativacao indevida",
                        similarity=0.7,
                        document_id="j1",
                        source_file="a.pdf",
                        page_number=2,
                        block_id="b2",
                        block_type="paragraph",
                        confidence=1.0,
                    ),
                ]
            ),
        )

        results = service.search("dano moral", n_results=10)
        assert len(results) == 1
        assert results[0].chunk_id == "c2"

    def test_returns_empty_when_no_overlap(self):
        service = SemanticSearchService(
            embedding_client=_FakeEmbeddingClient(),
            vector_store=_FakeVectorStore(
                [
                    SearchResult(
                        chunk_id="c1",
                        text="Tabela de parcelas e saldo atualizado",
                        similarity=0.9,
                        document_id="j1",
                        source_file="a.pdf",
                        page_number=1,
                        block_id="b1",
                        block_type="table",
                        confidence=1.0,
                    )
                ]
            ),
        )

        results = service.search("dano moral", n_results=10)
        assert not results

    def test_matches_plural_variants_via_prefix(self):
        service = SemanticSearchService(
            embedding_client=_FakeEmbeddingClient(),
            vector_store=_FakeVectorStore(
                [
                    SearchResult(
                        chunk_id="c1",
                        text=(
                            "pedido de danos morais e materiais por inscricao indevida"
                        ),
                        similarity=0.8,
                        document_id="j1",
                        source_file="a.pdf",
                        page_number=1,
                        block_id="b1",
                        block_type="paragraph",
                        confidence=1.0,
                    )
                ]
            ),
        )

        results = service.search("dano moral", n_results=10)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
