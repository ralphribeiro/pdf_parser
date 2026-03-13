"""
Tests for RedisJobStore — same interface as the in-memory JobStore.

Uses fakeredis so no real Redis server is needed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fakeredis
import pytest
from services.ingest_api.schemas import JobStatus


@pytest.fixture
def store():
    from services.ingest_api.redis_store import RedisJobStore

    return RedisJobStore(client=fakeredis.FakeRedis())


# =========================================================================
# create()
# =========================================================================


class TestRedisStoreCreate:
    def test_create_returns_job(self, store):
        job = store.create(filename="test.pdf")
        assert job.filename == "test.pdf"
        assert job.status == JobStatus.QUEUED

    def test_create_generates_unique_id(self, store):
        job1 = store.create(filename="a.pdf")
        job2 = store.create(filename="b.pdf")
        assert job1.job_id != job2.job_id

    def test_create_sets_created_at(self, store):
        job = store.create(filename="test.pdf")
        assert job.created_at is not None

    def test_create_adds_to_queue(self, store):
        job = store.create(filename="test.pdf")
        queued = store.get_next_queued()
        assert queued is not None
        assert queued.job_id == job.job_id


# =========================================================================
# get()
# =========================================================================


class TestRedisStoreGet:
    def test_get_existing_job(self, store):
        job = store.create(filename="test.pdf")
        retrieved = store.get(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent-id") is None

    def test_get_preserves_all_fields(self, store):
        job = store.create(filename="report.pdf")
        retrieved = store.get(job.job_id)
        assert retrieved.filename == "report.pdf"
        assert retrieved.status == JobStatus.QUEUED
        assert retrieved.started_at is None
        assert retrieved.completed_at is None
        assert retrieved.error_message is None
        assert retrieved.file_hash is None


# =========================================================================
# update_status()
# =========================================================================


class TestRedisStoreUpdateStatus:
    def test_update_status(self, store):
        job = store.create(filename="test.pdf")
        updated = store.update_status(job.job_id, JobStatus.PROCESSING)
        assert updated is not None
        assert updated.status == JobStatus.PROCESSING

    def test_update_status_with_error(self, store):
        job = store.create(filename="test.pdf")
        updated = store.update_status(
            job.job_id, JobStatus.FAILED, error_message="OCR failed"
        )
        assert updated is not None
        assert updated.status == JobStatus.FAILED
        assert updated.error_message == "OCR failed"

    def test_update_processing_records_started_at(self, store):
        job = store.create(filename="test.pdf")
        updated = store.update_status(job.job_id, JobStatus.PROCESSING)
        assert updated.started_at is not None

    def test_update_uploaded_records_completed_at(self, store):
        job = store.create(filename="test.pdf")
        store.update_status(job.job_id, JobStatus.PROCESSING)
        updated = store.update_status(job.job_id, JobStatus.UPLOADED)
        assert updated.completed_at is not None

    def test_update_nonexistent_returns_none(self, store):
        result = store.update_status("nonexistent", JobStatus.PROCESSING)
        assert result is None

    def test_update_with_file_hash(self, store):
        job = store.create(filename="test.pdf")
        updated = store.update_status(
            job.job_id, JobStatus.PROCESSING, file_hash="abc123"
        )
        assert updated.file_hash == "abc123"

    def test_update_preserves_existing_file_hash(self, store):
        job = store.create(filename="test.pdf")
        store.update_status(job.job_id, JobStatus.PROCESSING, file_hash="abc123")
        updated = store.update_status(job.job_id, JobStatus.UPLOADED)
        assert updated.file_hash == "abc123"

    def test_get_reflects_updated_status(self, store):
        job = store.create(filename="test.pdf")
        store.update_status(job.job_id, JobStatus.PROCESSING)
        retrieved = store.get(job.job_id)
        assert retrieved.status == JobStatus.PROCESSING


# =========================================================================
# get_next_queued() — ZPOPMIN semantics
# =========================================================================


class TestRedisStoreQueue:
    def test_returns_none_when_empty(self, store):
        assert store.get_next_queued() is None

    def test_returns_oldest_queued_job(self, store):
        j1 = store.create(filename="first.pdf")
        time.sleep(0.01)
        store.create(filename="second.pdf")
        next_job = store.get_next_queued()
        assert next_job.job_id == j1.job_id

    def test_removes_job_from_queue(self, store):
        store.create(filename="only.pdf")
        store.get_next_queued()
        assert store.get_next_queued() is None

    def test_skips_non_queued_jobs(self, store):
        j1 = store.create(filename="first.pdf")
        j2 = store.create(filename="second.pdf")
        store.update_status(j1.job_id, JobStatus.PROCESSING)
        next_job = store.get_next_queued()
        assert next_job.job_id == j2.job_id

    def test_fifo_order(self, store):
        j1 = store.create(filename="a.pdf")
        time.sleep(0.01)
        j2 = store.create(filename="b.pdf")
        time.sleep(0.01)
        j3 = store.create(filename="c.pdf")

        assert store.get_next_queued().job_id == j1.job_id
        assert store.get_next_queued().job_id == j2.job_id
        assert store.get_next_queued().job_id == j3.job_id
        assert store.get_next_queued() is None

    def test_job_data_still_accessible_after_dequeue(self, store):
        job = store.create(filename="test.pdf")
        store.get_next_queued()
        retrieved = store.get(job.job_id)
        assert retrieved is not None
        assert retrieved.filename == "test.pdf"


# =========================================================================
# from_url() class method
# =========================================================================


class TestRedisStoreFromUrl:
    def test_from_url_creates_instance(self):
        from services.ingest_api.redis_store import RedisJobStore

        store = RedisJobStore.from_url("redis://localhost:6379/0")
        assert isinstance(store, RedisJobStore)


# =========================================================================
# create_store() factory
# =========================================================================


class TestCreateStoreFactory:
    def test_returns_memory_store_when_no_redis_url(self, monkeypatch):
        from services.ingest_api.store import JobStore, create_store

        import config

        monkeypatch.setattr(config, "REDIS_URL", "")
        store = create_store()
        assert isinstance(store, JobStore)

    def test_returns_redis_store_when_redis_url_set(self, monkeypatch):
        from services.ingest_api.redis_store import RedisJobStore
        from services.ingest_api.store import create_store

        import config

        monkeypatch.setattr(config, "REDIS_URL", "redis://localhost:6379/0")
        store = create_store()
        assert isinstance(store, RedisJobStore)
