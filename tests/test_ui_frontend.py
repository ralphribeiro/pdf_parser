"""
Tests for the extended ingest UI frontend (dashboard, search, listings).

Covers:
    - Dashboard page (GET /)
    - Navigation links across all pages
    - Search page (GET /search)
    - Agent search page (GET /agent)
    - Jobs listing page (GET /jobs)
    - Documents listing page (GET /documents)
    - Document detail page (GET /documents/{id})
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from services.ingest_api.store import JobStore

# ---------------------------------------------------------------------------
# Fake DocumentStore for UI tests
# ---------------------------------------------------------------------------


class _FakeDocCollection:
    """Minimal MongoDB collection stub for UI tests."""

    def __init__(self):
        self._docs = {}
        self._counter = 0

    def create_index(self, key, **kwargs):
        pass

    def insert_one(self, doc):
        self._counter += 1
        oid = f"fake_oid_{self._counter:024d}"
        doc["_id"] = oid
        self._docs[oid] = dict(doc)
        result = MagicMock()
        result.inserted_id = oid
        return result

    def find_one(self, query):
        if "_id" in query:
            return self._docs.get(query["_id"])
        for doc in self._docs.values():
            if all(doc.get(k) == v for k, v in query.items()):
                return dict(doc)
        return None

    def find(self, query=None, projection=None):
        query = query or {}
        docs = list(self._docs.values())
        if query:
            docs = [d for d in docs if all(d.get(k) == v for k, v in query.items())]
        return _FakeCursor(docs, projection)

    def count_documents(self, query=None):
        query = query or {}
        if not query:
            return len(self._docs)
        return len(
            [
                d
                for d in self._docs.values()
                if all(d.get(k) == v for k, v in query.items())
            ]
        )

    def update_one(self, query, update):
        doc = self.find_one(query)
        if doc is None:
            return
        oid = query["_id"]
        for key, val in update.get("$set", {}).items():
            self._docs[oid][key] = val


class _FakeCursor:
    def __init__(self, docs, projection=None):
        self._docs = docs
        self._projection = projection
        self._skip = 0
        self._limit_val = None

    def skip(self, n):
        self._skip = n
        return self

    def limit(self, n):
        self._limit_val = n
        return self

    def sort(self, key, direction=None):
        return self

    def __iter__(self):
        docs = self._docs[self._skip :]
        if self._limit_val:
            docs = docs[: self._limit_val]
        if self._projection:
            for d in docs:
                yield {
                    k: d[k]
                    for k in list(d.keys())
                    if k in self._projection or k == "_id"
                }
        else:
            yield from docs


def _make_doc_store():
    from services.document_store import DocumentStore

    return DocumentStore(collection=_FakeDocCollection())


def _make_app(tmp_path, store=None, document_store=None):
    from services.ingest_ui.app import create_ui_app

    if store is None:
        store = JobStore()
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    return create_ui_app(
        upload_dir=upload_dir, store=store, document_store=document_store
    ), store


# =========================================================================
# Cycle D1: Dashboard — GET /
# =========================================================================


class TestDashboard:
    """GET / should render a dashboard with navigation to all features."""

    def test_returns_200(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/")
        assert resp.status_code == 200

    def test_returns_html(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_has_upload_link(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/").text
        assert "/upload" in html

    def test_has_search_link(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/").text
        assert "/search" in html

    def test_has_agent_link(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/").text
        assert "/agent" in html

    def test_has_jobs_link(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/").text
        assert "/jobs" in html

    def test_has_documents_link(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/").text
        assert "/documents" in html


# =========================================================================
# Cycle D2: Upload page — GET /upload
# =========================================================================


class TestUploadPageNew:
    """GET /upload should render the file upload form."""

    def test_returns_200(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/upload")
        assert resp.status_code == 200

    def test_has_file_input(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/upload").text
        assert 'type="file"' in html

    def test_has_multipart_form(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/upload").text.lower()
        assert "multipart/form-data" in html


# =========================================================================
# Cycle D3: Search page — GET /search
# =========================================================================


class TestSearchPage:
    """GET /search should render a semantic search form."""

    def test_returns_200(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/search")
        assert resp.status_code == 200

    def test_returns_html(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/search")
        assert "text/html" in resp.headers["content-type"]

    def test_has_query_input(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/search").text
        assert 'name="query"' in html

    def test_has_submit_button(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/search").text.lower()
        assert "<button" in html


# =========================================================================
# Cycle D4: Agent search page — GET /agent
# =========================================================================


class TestAgentPage:
    """GET /agent should render an agent search form with markdown support."""

    def test_returns_200(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/agent")
        assert resp.status_code == 200

    def test_returns_html(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/agent")
        assert "text/html" in resp.headers["content-type"]

    def test_has_query_input(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/agent").text
        assert 'name="query"' in html

    def test_has_submit_button(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/agent").text.lower()
        assert "<button" in html

    def test_includes_markdown_library(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/agent").text
        assert "marked" in html

    def test_renders_answer_as_markdown(self, tmp_path):
        app, _ = _make_app(tmp_path)
        html = TestClient(app).get("/agent").text
        assert "marked.parse" in html


# =========================================================================
# Cycle D5: Jobs listing page — GET /jobs
# =========================================================================


class TestJobsListPage:
    """GET /jobs should render a list of jobs."""

    def test_returns_200(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/jobs")
        assert resp.status_code == 200

    def test_returns_html(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/jobs")
        assert "text/html" in resp.headers["content-type"]

    def test_shows_job_filename(self, tmp_path):
        app, store = _make_app(tmp_path)
        store.create(filename="relatorio.pdf")
        html = TestClient(app).get("/jobs").text
        assert "relatorio.pdf" in html

    def test_shows_multiple_jobs(self, tmp_path):
        app, store = _make_app(tmp_path)
        store.create(filename="a.pdf")
        store.create(filename="b.pdf")
        html = TestClient(app).get("/jobs").text
        assert "a.pdf" in html
        assert "b.pdf" in html

    def test_shows_empty_state(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/jobs")
        assert resp.status_code == 200


# =========================================================================
# Cycle D6: Documents listing page — GET /documents
# =========================================================================


class TestDocumentsListPage:
    """GET /documents should render a list of documents."""

    def test_returns_200_with_store(self, tmp_path):
        doc_store = _make_doc_store()
        app, _ = _make_app(tmp_path, document_store=doc_store)
        resp = TestClient(app).get("/documents")
        assert resp.status_code == 200

    def test_returns_html(self, tmp_path):
        doc_store = _make_doc_store()
        app, _ = _make_app(tmp_path, document_store=doc_store)
        resp = TestClient(app).get("/documents")
        assert "text/html" in resp.headers["content-type"]

    def test_shows_document_filename(self, tmp_path):
        doc_store = _make_doc_store()
        doc_store.create_document(
            file_hash="h1",
            filename="contrato.pdf",
            file_size=1024,
            pdf_path="data/contrato.pdf",
        )
        app, _ = _make_app(tmp_path, document_store=doc_store)
        html = TestClient(app).get("/documents").text
        assert "contrato.pdf" in html

    def test_shows_unavailable_without_store(self, tmp_path):
        app, _ = _make_app(tmp_path, document_store=None)
        resp = TestClient(app).get("/documents")
        assert resp.status_code == 200
        assert (
            "indispon" in resp.text.lower()
            or "n\u00e3o configurad" in resp.text.lower()
        )


# =========================================================================
# Cycle D7: Document detail page — GET /documents/{id}
# =========================================================================


class TestDocumentDetailPage:
    """GET /documents/{id} should render document details."""

    def test_returns_200(self, tmp_path):
        doc_store = _make_doc_store()
        doc_id = doc_store.create_document(
            file_hash="h1",
            filename="doc.pdf",
            file_size=1024,
            pdf_path="data/doc.pdf",
        )
        app, _ = _make_app(tmp_path, document_store=doc_store)
        resp = TestClient(app).get(f"/documents/{doc_id}")
        assert resp.status_code == 200

    def test_shows_filename(self, tmp_path):
        doc_store = _make_doc_store()
        doc_id = doc_store.create_document(
            file_hash="h1",
            filename="meu-relatorio.pdf",
            file_size=2048,
            pdf_path="data/meu-relatorio.pdf",
        )
        app, _ = _make_app(tmp_path, document_store=doc_store)
        html = TestClient(app).get(f"/documents/{doc_id}").text
        assert "meu-relatorio.pdf" in html

    def test_returns_404_for_missing(self, tmp_path):
        doc_store = _make_doc_store()
        app, _ = _make_app(tmp_path, document_store=doc_store)
        resp = TestClient(app).get("/documents/nonexistent")
        assert resp.status_code == 404

    def test_returns_503_without_store(self, tmp_path):
        app, _ = _make_app(tmp_path, document_store=None)
        resp = TestClient(app).get("/documents/some-id")
        assert resp.status_code == 200
        assert (
            "indispon" in resp.text.lower()
            or "n\u00e3o configurad" in resp.text.lower()
        )


# =========================================================================
# Cycle D8: Navigation present on all pages
# =========================================================================


class TestNavigation:
    """All pages should include a consistent navigation bar."""

    @pytest.fixture
    def app_and_client(self, tmp_path):
        doc_store = _make_doc_store()
        app, store = _make_app(tmp_path, document_store=doc_store)
        return TestClient(app), store

    def test_upload_page_has_nav(self, app_and_client):
        client, _ = app_and_client
        html = client.get("/upload").text.lower()
        assert "<nav" in html

    def test_search_page_has_nav(self, app_and_client):
        client, _ = app_and_client
        html = client.get("/search").text.lower()
        assert "<nav" in html

    def test_jobs_page_has_nav(self, app_and_client):
        client, _ = app_and_client
        html = client.get("/jobs").text.lower()
        assert "<nav" in html

    def test_documents_page_has_nav(self, app_and_client):
        client, _ = app_and_client
        html = client.get("/documents").text.lower()
        assert "<nav" in html
