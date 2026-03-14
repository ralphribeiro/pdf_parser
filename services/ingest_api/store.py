"""
In-memory job store for minimal persistence.

Thread-safe via a lock; suitable for single-process deployments.
Use create_store() to pick the right backend (memory vs Redis) at runtime.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import cast

from services.ingest_api.schemas import Job, JobStatus


def create_store() -> JobStore:
    """Return a JobStore or RedisJobStore depending on REDIS_URL config."""
    import config

    if config.REDIS_URL:
        from services.ingest_api.redis_store import RedisJobStore

        return cast(JobStore, RedisJobStore.from_url(config.REDIS_URL))
    return JobStore()


class JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(
        self,
        filename: str,
        document_id: str | None = None,
    ) -> Job:
        """Create a new job with status QUEUED and return it."""
        job = Job(
            job_id=str(uuid.uuid4()),
            filename=filename,
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            document_id=document_id,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Job]:
        """Return jobs with pagination (newest first)."""
        with self._lock:
            jobs = list(self._jobs.values())
        return jobs[offset : offset + limit]

    def count(self) -> int:
        """Return total number of stored jobs."""
        with self._lock:
            return len(self._jobs)

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
