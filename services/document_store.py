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
        pdf_path: str,
    ) -> str:
        """Insert a new document record and return its document_id (str)."""
        doc = {
            "file_hash": file_hash,
            "filename": filename,
            "file_size": file_size,
            "pdf_path": pdf_path,
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
        return self.collection.find_one({"file_hash": file_hash})

    def get_document(self, document_id: str) -> dict | None:
        """Return a document by its _id, or None."""
        return self.collection.find_one({"_id": _to_object_id(document_id)})

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

    def update_pdf_path(self, document_id: str, pdf_path: str) -> None:
        """Set the filesystem path of the stored PDF."""
        self.collection.update_one(
            {"_id": _to_object_id(document_id)},
            {"$set": {"pdf_path": pdf_path}},
        )

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
