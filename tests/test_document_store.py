"""
Tests for DocumentStore (MongoDB document persistence layer).

Uses a dict-based stub so no real MongoDB connection is needed.
"""

from datetime import datetime

from services.document_store import DocumentStore


class _FakeCollection:
    """Minimal MongoDB collection stub for unit tests."""

    def __init__(self):
        self._docs = {}
        self._counter = 0

    def create_index(self, key, **kwargs):
        pass

    def insert_one(self, doc):
        from unittest.mock import MagicMock

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

    def update_one(self, query, update):
        doc = self.find_one(query)
        if doc is None:
            return
        oid = query["_id"]
        for key, val in update.get("$set", {}).items():
            self._docs[oid][key] = val


def _make_store() -> DocumentStore:
    return DocumentStore(collection=_FakeCollection())


class TestDocumentStoreCreate:
    def test_create_returns_document_id_string(self):
        store = _make_store()
        doc_id = store.create_document(
            file_hash="abc123",
            filename="test.pdf",
            file_size=1024,
            pdf_path="data/test.pdf",
        )
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_create_stores_all_fields(self):
        store = _make_store()
        doc_id = store.create_document(
            file_hash="sha256hash",
            filename="contract.pdf",
            file_size=5000,
            pdf_path="data/contract.pdf",
        )
        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc["file_hash"] == "sha256hash"
        assert doc["filename"] == "contract.pdf"
        assert doc["file_size"] == 5000
        assert doc["pdf_path"] == "data/contract.pdf"
        assert doc["status"] == "pending"
        assert isinstance(doc["created_at"], datetime)
        assert doc["processed_at"] is None
        assert doc["parsed_document"] is None
        assert doc["total_pages"] is None


class TestDocumentStoreFindByHash:
    def test_find_by_hash_returns_existing(self):
        store = _make_store()
        doc_id = store.create_document(
            file_hash="unique_hash",
            filename="doc.pdf",
            file_size=100,
            pdf_path="data/doc.pdf",
        )
        found = store.find_by_hash("unique_hash")
        assert found is not None
        assert str(found["_id"]) == doc_id

    def test_find_by_hash_returns_none_for_missing(self):
        store = _make_store()
        assert store.find_by_hash("nonexistent") is None


class TestDocumentStoreGetDocument:
    def test_get_existing_document(self):
        store = _make_store()
        doc_id = store.create_document(
            file_hash="h1",
            filename="a.pdf",
            file_size=10,
            pdf_path="data/a.pdf",
        )
        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc["filename"] == "a.pdf"

    def test_get_nonexistent_returns_none(self):
        store = _make_store()
        assert store.get_document("nonexistent_id") is None


class TestDocumentStoreSaveParsed:
    def test_save_parsed_updates_document(self):
        store = _make_store()
        doc_id = store.create_document(
            file_hash="h2",
            filename="b.pdf",
            file_size=200,
            pdf_path="data/b.pdf",
        )
        parsed = {"doc_id": "b", "pages": [{"page": 1, "blocks": []}]}
        store.save_parsed(doc_id, parsed_document=parsed, total_pages=5)

        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc["parsed_document"] == parsed
        assert doc["total_pages"] == 5
        assert doc["status"] == "processed"
        assert doc["processed_at"] is not None


class TestDocumentStoreUpdateStatus:
    def test_update_status_to_processing(self):
        store = _make_store()
        doc_id = store.create_document(
            file_hash="h3",
            filename="c.pdf",
            file_size=300,
            pdf_path="data/c.pdf",
        )
        store.update_status(doc_id, "processing")
        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc["status"] == "processing"

    def test_update_status_to_failed_with_error(self):
        store = _make_store()
        doc_id = store.create_document(
            file_hash="h4",
            filename="d.pdf",
            file_size=400,
            pdf_path="data/d.pdf",
        )
        store.update_status(doc_id, "failed", error_message="OCR crashed")
        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc["status"] == "failed"
        assert doc["error_message"] == "OCR crashed"
