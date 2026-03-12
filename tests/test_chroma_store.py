"""
Tests for Chroma vector store adapter.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.search.chunk_builder import TextChunk


class _FakeCollection:
    def __init__(self):
        self.upsert_calls = []
        self.delete_calls = []
        self.query_kwargs = None
        self.query_response = {
            "ids": [["job-1:1:p1_b1"]],
            "documents": [["Conteudo"]],
            "metadatas": [
                [
                    {
                        "job_id": "job-1",
                        "source_file": "a.pdf",
                        "page_number": 1,
                        "block_id": "p1_b1",
                        "block_type": "paragraph",
                        "confidence": 0.9,
                    }
                ]
            ],
            "distances": [[0.08]],
        }

    def upsert(self, *, ids, embeddings, documents, metadatas):
        self.upsert_calls.append(
            {
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            }
        )

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return self.query_response

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)


def _sample_chunk():
    return TextChunk(
        chunk_id="job-1:1:p1_b1",
        job_id="job-1",
        source_file="a.pdf",
        page_number=1,
        block_id="p1_b1",
        block_type="paragraph",
        text="Conteudo",
        confidence=0.9,
    )


class TestChromaVectorStore:
    def test_upsert_chunks(self):
        from services.search.chroma_store import ChromaVectorStore

        collection = _FakeCollection()
        store = ChromaVectorStore(collection=collection)
        store.upsert_chunks(chunks=[_sample_chunk()], embeddings=[[0.1, 0.2]])

        assert len(collection.upsert_calls) == 1
        call = collection.upsert_calls[0]
        assert call["ids"] == ["job-1:1:p1_b1"]
        assert call["documents"] == ["Conteudo"]
        assert call["metadatas"][0]["job_id"] == "job-1"

    def test_upsert_validates_lengths(self):
        from services.search.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(collection=_FakeCollection())
        with pytest.raises(ValueError, match="same length"):
            store.upsert_chunks(chunks=[_sample_chunk()], embeddings=[])

    def test_query_returns_search_results(self):
        from services.search.chroma_store import ChromaVectorStore

        collection = _FakeCollection()
        store = ChromaVectorStore(collection=collection)
        results = store.query(
            query_embedding=[0.9, 0.8],
            n_results=5,
            job_id="job-1",
        )

        assert len(results) == 1
        assert results[0].chunk_id == "job-1:1:p1_b1"
        assert abs(results[0].similarity - 0.92) < 1e-9
        assert collection.query_kwargs["where"] == {"job_id": "job-1"}

    def test_query_applies_min_similarity(self):
        from services.search.chroma_store import ChromaVectorStore

        collection = _FakeCollection()
        store = ChromaVectorStore(collection=collection)
        results = store.query(
            query_embedding=[0.9, 0.8],
            n_results=5,
            min_similarity=0.95,
        )
        assert not results

    def test_delete_job(self):
        from services.search.chroma_store import ChromaVectorStore

        collection = _FakeCollection()
        store = ChromaVectorStore(collection=collection)
        store.delete_job("job-1")
        assert collection.delete_calls == [{"where": {"job_id": "job-1"}}]
