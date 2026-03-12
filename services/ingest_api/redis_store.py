"""
Redis-backed job store for multi-process / multi-container deployments.

Same duck-typed interface as the in-memory JobStore so both are
interchangeable via the create_store() factory.

Storage layout:
  - ``job:{job_id}``  -> JSON-serialised Job (Redis string)
  - ``jobs:queued``   -> Sorted Set, score = epoch timestamp, member = job_id
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import cast

import redis

from services.ingest_api.schemas import Job, JobStatus

_KEY_PREFIX = "job:"
_QUEUE_KEY = "jobs:queued"


class RedisJobStore:
    """Redis-backed job store (same interface as JobStore)."""

    def __init__(self, client: redis.Redis) -> None:
        self._r = client

    @classmethod
    def from_url(cls, url: str) -> RedisJobStore:
        return cls(client=redis.Redis.from_url(url, decode_responses=True))

    # -- public interface (mirrors JobStore) --------------------------------

    def create(
        self,
        filename: str,
        document_id: str | None = None,
    ) -> Job:
        """Create a new job with status QUEUED and return it."""
        now = datetime.now()
        job = Job(
            job_id=str(uuid.uuid4()),
            filename=filename,
            status=JobStatus.QUEUED,
            created_at=now,
            document_id=document_id,
        )
        pipe = self._r.pipeline(transaction=True)
        pipe.set(f"{_KEY_PREFIX}{job.job_id}", job.model_dump_json())
        pipe.zadd(_QUEUE_KEY, {job.job_id: now.timestamp()})
        pipe.execute()
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job or None if not found."""
        raw = cast(
            str | bytes | bytearray | None,
            self._r.get(f"{_KEY_PREFIX}{job_id}"),
        )
        if raw is None:
            return None
        return Job.model_validate_json(raw)

    def get_next_queued(self) -> Job | None:
        """Atomically pop the oldest queued job from the sorted set."""
        result = cast(
            list[tuple[str | bytes, float]],
            self._r.zpopmin(_QUEUE_KEY, count=1),
        )
        if not result:
            return None
        job_id = result[0][0]
        if isinstance(job_id, bytes):
            job_id = job_id.decode()
        return self.get(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_message: str | None = None,
        file_hash: str | None = None,
    ) -> Job | None:
        """Update job status and timestamps. Returns None if not found."""
        key = f"{_KEY_PREFIX}{job_id}"
        raw = cast(str | bytes | bytearray | None, self._r.get(key))
        if raw is None:
            return None

        job = Job.model_validate_json(raw)
        now = datetime.now()

        updated = job.model_copy(
            update={
                "status": status,
                "error_message": error_message or job.error_message,
                "started_at": (
                    now
                    if status == JobStatus.PROCESSING and job.started_at is None
                    else job.started_at
                ),
                "completed_at": (
                    now
                    if status in (JobStatus.UPLOADED, JobStatus.FAILED)
                    else job.completed_at
                ),
                "file_hash": (file_hash if file_hash is not None else job.file_hash),
            }
        )

        pipe = self._r.pipeline(transaction=True)
        pipe.set(key, updated.model_dump_json())
        if status != JobStatus.QUEUED:
            pipe.zrem(_QUEUE_KEY, job_id)
        pipe.execute()

        return updated
