"""
High-level semantic indexing and query orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from services.ingest_api.schemas import SearchResult
from services.search.chunk_builder import build_chunks


class EmbeddingClientProtocol(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


class VectorStoreProtocol(Protocol):
    def upsert_chunks(self, chunks, embeddings) -> None: ...

    def query(
        self,
        query_embedding: list[float],
        n_results: int,
        *,
        job_id: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]: ...


@dataclass
class SemanticSearchService:
    """Orchestrates chunking, embedding and vector store operations."""

    embedding_client: EmbeddingClientProtocol
    vector_store: VectorStoreProtocol

    def index_document(self, job_id: str, document) -> int:
        """Index all semantic chunks for one processed document."""
        chunks = build_chunks(document=document, job_id=job_id)
        if not chunks:
            return 0
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_client.embed_texts(  # pylint: disable=assignment-from-no-return
            texts
        )
        self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        n_results: int,
        job_id: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """Execute semantic query against vector store."""
        query_embedding = self.embedding_client.embed_query(  # pylint: disable=assignment-from-no-return
            query
        )
        return self.vector_store.query(
            query_embedding=query_embedding,
            n_results=n_results,
            job_id=job_id,
            min_similarity=min_similarity,
        )
