"""
ChromaDB adapter for semantic chunk indexing and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from services.ingest_api.schemas import SearchResult
from services.search.chunk_builder import TextChunk


class ChromaCollectionProtocol(Protocol):
    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...

    def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...

    def query(self, **kwargs) -> dict: ...

    def delete(self, **kwargs) -> None: ...


@dataclass
class ChromaVectorStore:
    """Wrap a Chroma collection with app-specific operations."""

    collection: ChromaCollectionProtocol

    @classmethod
    def from_host(cls, chroma_host: str, collection_name: str) -> ChromaVectorStore:
        """Create adapter connected to a remote Chroma server."""
        import chromadb

        parsed = urlparse(chroma_host)
        host = parsed.hostname or parsed.path or "localhost"
        port = parsed.port or 8000
        client = chromadb.HttpClient(host=host, port=port)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return cls(collection=collection)

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks with matching embedding vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "job_id": chunk.job_id,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "block_id": chunk.block_id,
                "block_type": chunk.block_type,
                "confidence": chunk.confidence,
            }
            for chunk in chunks
        ]

        if hasattr(self.collection, "upsert"):
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        else:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    def query(
        self,
        query_embedding: list[float],
        n_results: int,
        *,
        job_id: str | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """Query nearest chunks and convert to API response shape."""
        where = {"job_id": job_id} if job_id else None
        raw = self.collection.query(  # pylint: disable=assignment-from-no-return
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for idx, chunk_id in enumerate(ids):
            text = docs[idx] if idx < len(docs) else ""
            metadata = metas[idx] if idx < len(metas) else {}
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            similarity = max(0.0, min(1.0, 1.0 - distance))

            if min_similarity is not None and similarity < min_similarity:
                continue

            results.append(
                SearchResult(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    similarity=similarity,
                    job_id=str(metadata.get("job_id", "")),
                    source_file=str(metadata.get("source_file", "")),
                    page_number=int(metadata.get("page_number", 1)),
                    block_id=str(metadata.get("block_id", "")),
                    block_type=str(metadata.get("block_type", "")),
                    confidence=float(metadata.get("confidence", 0.0)),
                )
            )
        return results

    def delete_job(self, job_id: str) -> None:
        """Delete all vectors belonging to one job."""
        self.collection.delete(where={"job_id": job_id})
