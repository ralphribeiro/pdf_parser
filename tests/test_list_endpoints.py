"""
Tests for list endpoints (GET /api/jobs, GET /api/documents).

Covers:
    - JobListResponse / DocumentListResponse schemas
    - JobStore.list_all() method
    - GET /jobs  -> 200 with paginated job list
    - GET /documents -> 200 with paginated document list (503 when unconfigured)
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from services.ingest_api.schemas import Job, JobStatus
from services.ingest_api.store import JobStore

# =========================================================================
# Cycle L1: JobListResponse schema
# =========================================================================


class TestJobListResponseSchema:
    """JobListResponse should hold a paginated list of jobs."""

    def test_has_required_fields(self):
        from services.ingest_api.schemas import JobListResponse

        now = datetime.now()
        job = Job(
            job_id="j1",
            filename="a.pdf",
            status=JobStatus.QUEUED,
            created_at=now,
        )
        resp = JobListResponse(items=[job], total=1, limit=20, offset=0)
        assert resp.items == [job]
        assert resp.total == 1
        assert resp.limit == 20
        assert resp.offset == 0

    def test_empty_list(self):
        from services.ingest_api.schemas import JobListResponse

        resp = JobListResponse(items=[], total=0, limit=20, offset=0)
        assert resp.items == []
        assert resp.total == 0

    def test_serializes_to_json(self):
        from services.ingest_api.schemas import JobListResponse

        now = datetime.now()
        job = Job(
            job_id="j1",
            filename="a.pdf",
            status=JobStatus.QUEUED,
            created_at=now,
        )
        data = JobListResponse(items=[job], total=1, limit=20, offset=0).model_dump(
            mode="json"
        )
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["job_id"] == "j1"


# =========================================================================
# Cycle L2: DocumentListResponse schema
# =========================================================================


class TestDocumentListResponseSchema:
    """DocumentListResponse should hold a paginated list of document summaries."""

    def test_has_required_fields(self):
        from services.ingest_api.schemas import DocumentListResponse, DocumentSummary

        doc = DocumentSummary(
            document_id="d1",
            filename="a.pdf",
            status="pending",
            total_pages=None,
            created_at=datetime.now(),
            file_size=1024,
        )
        resp = DocumentListResponse(items=[doc], total=1, limit=20, offset=0)
        assert resp.items == [doc]
        assert resp.total == 1

    def test_empty_list(self):
        from services.ingest_api.schemas import DocumentListResponse

        resp = DocumentListResponse(items=[], total=0, limit=20, offset=0)
        assert resp.items == []
        assert resp.total == 0

    def test_document_summary_fields(self):
        from services.ingest_api.schemas import DocumentSummary

        now = datetime.now()
        doc = DocumentSummary(
            document_id="d1",
            filename="report.pdf",
            status="processed",
            total_pages=10,
            created_at=now,
            file_size=5000,
        )
        assert doc.document_id == "d1"
        assert doc.filename == "report.pdf"
        assert doc.status == "processed"
        assert doc.total_pages == 10
        assert doc.file_size == 5000


# =========================================================================
# Cycle L3: JobStore.list_all()
# =========================================================================


class TestJobStoreListAll:
    """JobStore should support listing all jobs with pagination."""

    def test_returns_empty_list_when_no_jobs(self):
        store = JobStore()
        result = store.list_all()
        assert not result

    def test_returns_all_jobs(self):
        store = JobStore()
        store.create(filename="a.pdf")
        store.create(filename="b.pdf")
        result = store.list_all()
        assert len(result) == 2

    def test_respects_limit(self):
        store = JobStore()
        for i in range(5):
            store.create(filename=f"{i}.pdf")
        result = store.list_all(limit=3)
        assert len(result) == 3

    def test_respects_offset(self):
        store = JobStore()
        for i in range(5):
            store.create(filename=f"{i}.pdf")
        result = store.list_all(offset=2)
        assert len(result) == 3

    def test_limit_and_offset_combined(self):
        store = JobStore()
        for i in range(10):
            store.create(filename=f"{i}.pdf")
        result = store.list_all(limit=3, offset=2)
        assert len(result) == 3

    def test_offset_beyond_end_returns_empty(self):
        store = JobStore()
        store.create(filename="a.pdf")
        result = store.list_all(offset=100)
        assert not result

    def test_count_returns_total_jobs(self):
        store = JobStore()
        store.create(filename="a.pdf")
        store.create(filename="b.pdf")
        assert store.count() == 2

    def test_count_returns_zero_when_empty(self):
        store = JobStore()
        assert store.count() == 0


# =========================================================================
# Cycle L4: GET /jobs (list all jobs)
# =========================================================================


class _FakeDocCollection:
    """Minimal MongoDB collection stub supporting find() for list tests."""

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
    """Minimal cursor stub chaining limit/skip/sort."""

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


@pytest.fixture
def api_client(tmp_path):
    """TestClient with an empty JobStore and no document store."""
    from services.ingest_api.app import create_app

    store = JobStore()
    app = create_app(upload_dir=tmp_path, store=store)
    return TestClient(app), store


@pytest.fixture
def api_client_with_docs(tmp_path):
    """TestClient with JobStore and a fake DocumentStore."""
    from services.ingest_api.app import create_app

    store = JobStore()
    doc_store = _make_doc_store()
    app = create_app(upload_dir=tmp_path, store=store, document_store=doc_store)
    return TestClient(app), store, doc_store


class TestListJobsEndpoint:
    """GET /jobs should return a paginated list of jobs."""

    def test_returns_200(self, api_client):
        client, _ = api_client
        resp = client.get("/jobs")
        assert resp.status_code == 200

    def test_returns_empty_list(self, api_client):
        client, _ = api_client
        data = client.get("/jobs").json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_returns_created_jobs(self, api_client):
        client, store = api_client
        store.create(filename="a.pdf")
        store.create(filename="b.pdf")
        data = client.get("/jobs").json()
        assert data["total"] == 2
        assert len(data["items"]) == 2

    def test_supports_limit(self, api_client):
        client, store = api_client
        for i in range(5):
            store.create(filename=f"{i}.pdf")
        data = client.get("/jobs", params={"limit": 2}).json()
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_supports_offset(self, api_client):
        client, store = api_client
        for i in range(5):
            store.create(filename=f"{i}.pdf")
        data = client.get("/jobs", params={"offset": 3}).json()
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_default_limit_is_20(self, api_client):
        client, _ = api_client
        data = client.get("/jobs").json()
        assert data["limit"] == 20

    def test_response_has_pagination_fields(self, api_client):
        client, _ = api_client
        data = client.get("/jobs").json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data


# =========================================================================
# Cycle L5: GET /documents (list all documents)
# =========================================================================


class TestListDocumentsEndpoint:
    """GET /documents should return a paginated list of documents."""

    def test_returns_503_without_document_store(self, api_client):
        client, _ = api_client
        resp = client.get("/documents")
        assert resp.status_code == 503

    def test_returns_200_with_document_store(self, api_client_with_docs):
        client, _, _ = api_client_with_docs
        resp = client.get("/documents")
        assert resp.status_code == 200

    def test_returns_empty_list(self, api_client_with_docs):
        client, _, _ = api_client_with_docs
        data = client.get("/documents").json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_returns_created_documents(self, api_client_with_docs):
        client, _, doc_store = api_client_with_docs
        doc_store.create_document(
            file_hash="h1", filename="a.pdf", file_size=100, pdf_path="data/a.pdf"
        )
        doc_store.create_document(
            file_hash="h2", filename="b.pdf", file_size=200, pdf_path="data/b.pdf"
        )
        data = client.get("/documents").json()
        assert data["total"] == 2
        assert len(data["items"]) == 2

    def test_supports_limit(self, api_client_with_docs):
        client, _, doc_store = api_client_with_docs
        for i in range(5):
            doc_store.create_document(
                file_hash=f"h{i}",
                filename=f"{i}.pdf",
                file_size=100,
                pdf_path=f"data/{i}.pdf",
            )
        data = client.get("/documents", params={"limit": 2}).json()
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_supports_offset(self, api_client_with_docs):
        client, _, doc_store = api_client_with_docs
        for i in range(5):
            doc_store.create_document(
                file_hash=f"h{i}",
                filename=f"{i}.pdf",
                file_size=100,
                pdf_path=f"data/{i}.pdf",
            )
        data = client.get("/documents", params={"offset": 3}).json()
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_response_has_pagination_fields(self, api_client_with_docs):
        client, _, _ = api_client_with_docs
        data = client.get("/documents").json()
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data

    def test_document_item_has_summary_fields(self, api_client_with_docs):
        client, _, doc_store = api_client_with_docs
        doc_store.create_document(
            file_hash="hx",
            filename="report.pdf",
            file_size=5000,
            pdf_path="data/report.pdf",
        )
        data = client.get("/documents").json()
        item = data["items"][0]
        assert "document_id" in item
        assert item["filename"] == "report.pdf"
        assert "status" in item
        assert "file_size" in item
