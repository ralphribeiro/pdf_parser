"""
OCR worker: consumes queued jobs and runs the extraction pipeline.

Designed for dependency injection so the pipeline function
can be mocked in tests.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.document_store import DocumentStore
    from services.ingest_api.store import JobStore

logger = logging.getLogger(__name__)


class OcrWorker:
    """Processes queued jobs end-to-end via the OCR pipeline."""

    def __init__(
        self,
        store: JobStore,
        upload_dir: Path,
        output_dir: Path,
        artifact_fn: Callable[..., Any] | None = None,
        semantic_indexer: Any | None = None,
        document_store: DocumentStore | None = None,
    ) -> None:
        self.store = store
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.semantic_indexer = semantic_indexer
        self.document_store = document_store

        if artifact_fn is None:
            from src.pipeline import generate_artifacts

            self._artifact_fn = generate_artifacts
        else:
            self._artifact_fn = artifact_fn

    def process_job(self, job_id: str) -> None:
        """Drive a single job through queued -> processing -> uploaded/failed."""
        from services.ingest_api.schemas import JobStatus

        job = self.store.get(job_id)
        if job is None:
            logger.warning("Job %s not found, skipping", job_id)
            return

        document_id = job.document_id

        if document_id and self.document_store:
            mongo_doc = self.document_store.get_document(document_id)
            pdf_path = Path(mongo_doc["pdf_path"]) if mongo_doc else None
            if pdf_path is None or not pdf_path.exists():
                pdf_path = self.upload_dir / f"{document_id}.pdf"
            self.document_store.update_status(document_id, "processing")
        else:
            pdf_path = self.upload_dir / f"{job_id}.pdf"

        self.store.update_status(job_id, JobStatus.PROCESSING)

        try:
            artifacts = self._artifact_fn(pdf_path, self.output_dir)
        except Exception:
            logger.exception("Job %s failed", job_id)
            self.store.update_status(
                job_id,
                JobStatus.FAILED,
                error_message=traceback.format_exc(),
            )
            if document_id and self.document_store:
                self.document_store.update_status(
                    document_id, "failed", error_message=traceback.format_exc()
                )
            return

        if document_id and self.document_store:
            try:
                parsed = artifacts.document.model_dump()
                total_pages = len(artifacts.document.pages)
                self.document_store.save_parsed(document_id, parsed, total_pages)
            except Exception:
                logger.exception(
                    "Job %s: failed to save parsed document to MongoDB (non-fatal)",
                    job_id,
                )

        ref_id = document_id or job_id
        if self.semantic_indexer is not None:
            try:
                n = self.semantic_indexer.index_document(ref_id, artifacts.document)
                logger.info("Job %s: indexed %d chunks (ref=%s)", job_id, n, ref_id)
            except Exception:
                logger.exception("Job %s: semantic indexing failed (non-fatal)", job_id)

        self.store.update_status(job_id, JobStatus.UPLOADED)
        logger.info("Job %s completed (document_id=%s)", job_id, document_id)

    def process_next(self) -> bool:
        """Find and process the next queued job. Returns True if one was processed."""
        job = self.store.get_next_queued()
        if job is None:
            return False
        self.process_job(job.job_id)
        return True
