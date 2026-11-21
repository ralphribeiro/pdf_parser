"""
MongoDB-backed document store for persistent document metadata and parsed output.

Wraps a pymongo Collection with app-specific operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _to_object_id(document_id: str) -> Any:
    """Convert a string document_id to ObjectId for MongoDB queries.

    Falls back to the raw string if bson is not available (e.g. in unit tests
    with dict-based stubs).
    """
    try:
        from bson import ObjectId

        return ObjectId(document_id)
    except Exception:
        return document_id


@dataclass
class DocumentStore:
    """Persistent document store backed by a MongoDB collection."""

    collection: Any

    def __post_init__(self) -> None:
        self.collection.create_index("file_hash", unique=True)

    def create_document(
        self,
        file_hash: str,
        filename: str,
        file_size: int,
        pdf_path: str = "",
        source_path: str | None = None,
        content_type: str | None = None,
    ) -> str:
        """Insert a new document record and return its document_id (str)."""
        stored_source_path = source_path if source_path is not None else pdf_path
        doc = {
            "file_hash": file_hash,
            "filename": filename,
            "file_size": file_size,
            "source_path": stored_source_path,
            "content_type": content_type,
            "pdf_path": pdf_path or stored_source_path,
            "status": "pending",
            "created_at": datetime.now(),
            "processed_at": None,
            "error_message": None,
            "total_pages": None,
            "parsed_document": None,
        }
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)

    def find_by_hash(self, file_hash: str) -> dict | None:
        """Return the document with the given SHA-256 hash, or None."""
        return self._with_legacy_paths(
            self.collection.find_one({"file_hash": file_hash})
        )

    def get_document(self, document_id: str) -> dict | None:
        """Return a document by its _id, or None."""
        return self._with_legacy_paths(
            self.collection.find_one({"_id": _to_object_id(document_id)})
        )

    def save_parsed(
        self,
        document_id: str,
        parsed_document: dict,
        total_pages: int,
    ) -> None:
        """Store the parsed document JSON and mark as processed."""
        self.collection.update_one(
            {"_id": _to_object_id(document_id)},
            {
                "$set": {
                    "parsed_document": parsed_document,
                    "total_pages": total_pages,
                    "status": "processed",
                    "processed_at": datetime.now(),
                    "error_message": None,
                }
            },
        )

    def update_status(
        self,
        document_id: str,
        status: str,
        *,
        error_message: str | None = None,
    ) -> None:
        """Update document status and optional error message."""
        update: dict[str, Any] = {"status": status}
        if error_message is not None:
            update["error_message"] = error_message
        self.collection.update_one(
            {"_id": _to_object_id(document_id)}, {"$set": update}
        )

    def reset_for_reprocessing(
        self,
        document_id: str,
        *,
        filename: str,
        source_path: str,
        content_type: str | None = None,
    ) -> bool:
        """Atomically reset a failed document back to pending for reprocessing.

        Returns True when the document was in ``failed`` state and updated,
        or False if its status was not ``failed`` (e.g. concurrent retry).
        """
        update: dict[str, Any] = {
            "filename": filename,
            "source_path": source_path,
            "pdf_path": source_path,
            "status": "pending",
            "error_message": None,
            "processed_at": None,
            "total_pages": None,
            "parsed_document": None,
        }
        if content_type is not None:
            update["content_type"] = content_type
        result = self.collection.update_one(
            {"_id": _to_object_id(document_id), "status": "failed"},
            {"$set": update},
        )
        return bool(getattr(result, "modified_count", 0))

    def list_documents(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """Return documents, optionally filtered by status, with pagination."""
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        projection = {
            "filename": 1,
            "status": 1,
            "total_pages": 1,
            "created_at": 1,
            "file_size": 1,
            "source_path": 1,
            "content_type": 1,
            "pdf_path": 1,
        }
        cursor = self.collection.find(query, projection).skip(offset).limit(limit)
        results: list[dict] = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(self._with_legacy_paths(doc))
        return results

    def count_documents(self, *, status: str | None = None) -> int:
        """Return total count of documents, optionally filtered by status."""
        query: dict[str, Any] = {}
        if status:
            query["status"] = status
        return self.collection.count_documents(query)

    def search_text(self, document_id: str, keyword: str) -> list[dict]:
        """Search for a keyword inside the parsed_document of a specific document."""
        doc = self.get_document(document_id)
        if not doc:
            return []
        parsed = doc.get("parsed_document") or {}
        pages = parsed.get("pages", [])
        hits: list[dict] = []
        kw_lower = keyword.lower()
        for page in pages:
            for block in page.get("blocks", []):
                text = block.get("text") or ""
                if text and kw_lower in text.lower():
                    hits.append(
                        {
                            "page": page.get("page") or page.get("page_number") or 0,
                            "block_id": block.get("block_id", ""),
                            "text": text,
                        }
                    )
        return hits

    def update_pdf_path(self, document_id: str, pdf_path: str) -> None:
        """Set the filesystem path of the stored PDF."""
        self.update_source_path(document_id, pdf_path, content_type="application/pdf")

    def update_source_path(
        self,
        document_id: str,
        source_path: str,
        *,
        content_type: str | None = None,
    ) -> None:
        """Set the filesystem path and content type of the stored source file."""
        update: dict[str, Any] = {
            "source_path": source_path,
            "pdf_path": source_path,
        }
        if content_type is not None:
            update["content_type"] = content_type
        self.collection.update_one(
            {"_id": _to_object_id(document_id)},
            {"$set": update},
        )

    @staticmethod
    def _with_legacy_paths(doc: dict | None) -> dict | None:
        """Normalize source_path/pdf_path fields for legacy records."""
        if doc is None:
            return None
        if not doc.get("source_path") and doc.get("pdf_path"):
            doc["source_path"] = doc["pdf_path"]
        if not doc.get("pdf_path") and doc.get("source_path"):
            doc["pdf_path"] = doc["source_path"]
        return doc

    @classmethod
    def from_url(cls, url: str, db_name: str) -> DocumentStore:
        """Create a DocumentStore connected to a remote MongoDB server."""
        from pymongo import MongoClient

        client: Any = MongoClient(url)
        db = client[db_name]
        collection = db["documents"]
        return cls(collection=collection)


def create_document_store() -> DocumentStore | None:
    """Build DocumentStore from config, or None if MONGO_URL is not set."""
    import config

    if not config.MONGO_URL:
        return None
    db_name: str = config.MONGO_DB or "doc_parser"
    return DocumentStore.from_url(config.MONGO_URL, db_name)
