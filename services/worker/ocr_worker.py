"""
OCR worker: consumes queued jobs and runs the extraction pipeline.

Designed for dependency injection so the pipeline function
can be mocked in tests.
"""

from __future__ import annotations

import hashlib
import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.ingest_api.store import JobStore

logger = logging.getLogger(__name__)


def compute_file_hash(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file (for idempotency checks)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class OcrWorker:
    """Processes queued jobs end-to-end via the OCR pipeline."""

    def __init__(
        self,
        store: JobStore,
        upload_dir: Path,
        output_dir: Path,
        artifact_fn: Callable[..., Any] | None = None,
        semantic_indexer: Any | None = None,
    ) -> None:
        self.store = store
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.semantic_indexer = semantic_indexer

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

        pdf_path = self.upload_dir / f"{job_id}.pdf"
        file_hash = compute_file_hash(pdf_path)

        self.store.update_status(job_id, JobStatus.PROCESSING, file_hash=file_hash)

        try:
            artifacts = self._artifact_fn(pdf_path, self.output_dir)
        except Exception:
            logger.exception("Job %s failed", job_id)
            self.store.update_status(
                job_id,
                JobStatus.FAILED,
                error_message=traceback.format_exc(),
            )
            return

        if self.semantic_indexer is not None:
            try:
                n = self.semantic_indexer.index_document(job_id, artifacts.document)
                logger.info("Job %s: indexed %d chunks", job_id, n)
            except Exception:
                logger.exception("Job %s: semantic indexing failed (non-fatal)", job_id)

        self.store.update_status(job_id, JobStatus.UPLOADED)
        logger.info("Job %s completed", job_id)

    def process_next(self) -> bool:
        """Find and process the next queued job. Returns True if one was processed."""
        job = self.store.get_next_queued()
        if job is None:
            return False
        self.process_job(job.job_id)
        return True
