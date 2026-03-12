"""
In-memory job store for minimal persistence.

Thread-safe via a lock; suitable for single-process deployments.
Replace with Redis-backed store when scaling to multiple workers.
"""

import threading
import uuid
from datetime import datetime

from services.ingest_api.schemas import Job, JobStatus


class JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        """Create a new job with status QUEUED and return it."""
        job = Job(
            job_id=str(uuid.uuid4()),
            filename=filename,
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_next_queued(self) -> Job | None:
        """Return the oldest queued job, or None."""
        with self._lock:
            for job in self._jobs.values():
                if job.status == JobStatus.QUEUED:
                    return job
            return None

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_message: str | None = None,
        file_hash: str | None = None,
    ) -> Job | None:
        """Update job status and timestamps. Returns None if not found."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None

            updated = job.model_copy(
                update={
                    "status": status,
                    "error_message": error_message or job.error_message,
                    "started_at": (
                        datetime.now()
                        if status == JobStatus.PROCESSING and job.started_at is None
                        else job.started_at
                    ),
                    "completed_at": (
                        datetime.now()
                        if status in (JobStatus.UPLOADED, JobStatus.FAILED)
                        else job.completed_at
                    ),
                    "file_hash": (
                        file_hash if file_hash is not None else job.file_hash
                    ),
                }
            )
            self._jobs[job_id] = updated
            return updated
