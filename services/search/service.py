"""
High-level semantic indexing and query orchestration.
"""

from __future__ import annotations

import re
import unicodedata
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
        document_id: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def _keyword_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(_normalize(text)))


def _token_prefixes(tokens: set[str]) -> set[str]:
    return {token[:4] for token in tokens if len(token) >= 4}


def _keyword_filter(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Keep only results sharing at least one query keyword token."""
    query_tokens = _keyword_tokens(query)
    if not query_tokens:
        return results
    query_prefixes = _token_prefixes(query_tokens)
    required_prefix_matches = 1 if len(query_prefixes) == 1 else 2

    filtered: list[SearchResult] = []
    for item in results:
        text_tokens = _keyword_tokens(item.text)
        exact_overlap = text_tokens & query_tokens
        prefix_overlap = _token_prefixes(text_tokens) & query_prefixes
        if exact_overlap or len(prefix_overlap) >= required_prefix_matches:
            filtered.append(item)
    return filtered


@dataclass
class SemanticSearchService:
    """Orchestrates chunking, embedding and vector store operations."""

    embedding_client: EmbeddingClientProtocol
    vector_store: VectorStoreProtocol

    def index_document(self, document_id: str, document) -> int:
        """Index all semantic chunks for one processed document."""
        chunks = build_chunks(document=document, document_id=document_id)
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
        document_id: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """Execute semantic query against vector store."""
        query_embedding = self.embedding_client.embed_query(  # pylint: disable=assignment-from-no-return
            query
        )
        raw_results = self.vector_store.query(  # pylint: disable=assignment-from-no-return
            query_embedding=query_embedding,
            n_results=n_results,
            document_id=document_id,
            min_similarity=min_similarity,
        )
        return _keyword_filter(query=query, results=raw_results)
