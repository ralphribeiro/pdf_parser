"""
Tests for POST /search endpoint on ingest API.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingest_api.app import create_app
from services.ingest_api.schemas import SearchResult
from services.ingest_api.store import JobStore


class _FakeSearchService:
    def __init__(self):
        self.calls = []

    def search(self, query, *, n_results, document_id=None, min_similarity=None):
        self.calls.append(
            {
                "query": query,
                "n_results": n_results,
                "document_id": document_id,
                "min_similarity": min_similarity,
            }
        )
        return [
            SearchResult(
                chunk_id="doc-1:1:p1_b1",
                text="Resultado",
                similarity=0.87,
                document_id="doc-1",
                source_file="doc.pdf",
                page_number=1,
                block_id="p1_b1",
                block_type="paragraph",
                confidence=0.95,
            )
        ]


class TestSearchEndpoint:
    def test_returns_503_when_search_not_configured(self, tmp_path):
        app = create_app(upload_dir=tmp_path, store=JobStore())
        client = TestClient(app)
        response = client.post("/search", json={"query": "teste"})
        assert response.status_code == 503

    def test_returns_search_results(self, tmp_path):
        service = _FakeSearchService()
        app = create_app(upload_dir=tmp_path, store=JobStore(), semantic_search=service)
        client = TestClient(app)
        response = client.post("/search", json={"query": "resultado"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["total_matches"] == 1
        assert payload["results"][0]["chunk_id"] == "doc-1:1:p1_b1"
        assert payload["processing_time_ms"] >= 0

    def test_passes_filters_to_service(self, tmp_path):
        service = _FakeSearchService()
        app = create_app(upload_dir=tmp_path, store=JobStore(), semantic_search=service)
        client = TestClient(app)
        response = client.post(
            "/search",
            json={
                "query": "resultado",
                "n_results": 3,
                "filters": {"document_id": "doc-1"},
                "min_similarity": 0.7,
            },
        )
        assert response.status_code == 200
        assert service.calls[0] == {
            "query": "resultado",
            "n_results": 3,
            "document_id": "doc-1",
            "min_similarity": 0.7,
        }


class _FakeDocStore:
    def __init__(self, docs):
        self._docs = docs

    def get_document(self, document_id):
        return self._docs.get(document_id)


class TestSearchEnrichesSourceFile:
    """Search results must show the original filename, not the hash."""

    def test_replaces_hash_with_real_filename(self, tmp_path):
        svc = _FakeSearchService()
        svc.search = lambda *a, **kw: [
            SearchResult(
                chunk_id="c1",
                text="texto",
                similarity=0.9,
                document_id="abc123",
                source_file="abc123.pdf",
                page_number=1,
                block_id="b1",
                block_type="text",
                confidence=0.9,
            )
        ]
        doc_store = _FakeDocStore({"abc123": {"filename": "contrato_social.pdf"}})
        app = create_app(
            upload_dir=tmp_path,
            store=JobStore(),
            semantic_search=svc,
            document_store=doc_store,
        )
        client = TestClient(app)
        resp = client.post("/search", json={"query": "texto"})
        assert resp.status_code == 200
        assert resp.json()["results"][0]["source_file"] == "contrato_social.pdf"

    def test_keeps_source_file_when_no_doc_store(self, tmp_path):
        svc = _FakeSearchService()
        app = create_app(
            upload_dir=tmp_path,
            store=JobStore(),
            semantic_search=svc,
        )
        client = TestClient(app)
        resp = client.post("/search", json={"query": "resultado"})
        assert resp.status_code == 200
        assert resp.json()["results"][0]["source_file"] == "doc.pdf"

    def test_keeps_source_file_when_doc_not_found(self, tmp_path):
        svc = _FakeSearchService()
        doc_store = _FakeDocStore({})
        app = create_app(
            upload_dir=tmp_path,
            store=JobStore(),
            semantic_search=svc,
            document_store=doc_store,
        )
        client = TestClient(app)
        resp = client.post("/search", json={"query": "resultado"})
        assert resp.status_code == 200
        assert resp.json()["results"][0]["source_file"] == "doc.pdf"
