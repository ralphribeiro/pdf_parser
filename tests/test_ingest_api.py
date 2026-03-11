"""
Tests for the async ingest API (Milestone 2).

Covers: JobStatus enum, Job model, JobStore, and API endpoints.
    POST /jobs  -> 202 with job_id (queued)
    GET  /jobs/{job_id} -> 200 with full job status
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# =========================================================================
# Cycle J1: JobStatus enum
# =========================================================================


class TestJobStatus:
    """Tests for JobStatus enum with valid job states."""

    def test_has_queued_state(self):
        from services.ingest_api.schemas import JobStatus

        assert JobStatus.QUEUED == "queued"

    def test_has_processing_state(self):
        from services.ingest_api.schemas import JobStatus

        assert JobStatus.PROCESSING == "processing"

    def test_has_uploaded_state(self):
        from services.ingest_api.schemas import JobStatus

        assert JobStatus.UPLOADED == "uploaded"

    def test_has_failed_state(self):
        from services.ingest_api.schemas import JobStatus

        assert JobStatus.FAILED == "failed"


# =========================================================================
# Cycle J2: Job model
# =========================================================================


class TestJobModel:
    """Tests for the Job Pydantic model."""

    def test_job_has_required_fields(self):
        from datetime import datetime

        from services.ingest_api.schemas import Job, JobStatus

        job = Job(
            job_id="test-123",
            filename="document.pdf",
            status=JobStatus.QUEUED,
            created_at=datetime(2026, 3, 11, 12, 0, 0),
        )
        assert job.job_id == "test-123"
        assert job.filename == "document.pdf"
        assert job.status == JobStatus.QUEUED

    def test_job_optional_fields_default_to_none(self):
        from datetime import datetime

        from services.ingest_api.schemas import Job, JobStatus

        job = Job(
            job_id="test-123",
            filename="document.pdf",
            status=JobStatus.QUEUED,
            created_at=datetime(2026, 3, 11, 12, 0, 0),
        )
        assert job.started_at is None
        assert job.completed_at is None
        assert job.error_message is None

    def test_job_serializes_to_json(self):
        from datetime import datetime

        from services.ingest_api.schemas import Job, JobStatus

        job = Job(
            job_id="test-123",
            filename="document.pdf",
            status=JobStatus.QUEUED,
            created_at=datetime(2026, 3, 11, 12, 0, 0),
        )
        data = job.model_dump(mode="json")
        assert data["job_id"] == "test-123"
        assert data["status"] == "queued"
        assert "created_at" in data


# =========================================================================
# Cycle S1: JobStore (in-memory persistence)
# =========================================================================


class TestJobStore:
    """Tests for the in-memory JobStore."""

    def test_create_returns_job(self):
        from services.ingest_api.schemas import JobStatus
        from services.ingest_api.store import JobStore

        store = JobStore()
        job = store.create(filename="test.pdf")
        assert job.filename == "test.pdf"
        assert job.status == JobStatus.QUEUED

    def test_create_generates_unique_id(self):
        from services.ingest_api.store import JobStore

        store = JobStore()
        job1 = store.create(filename="a.pdf")
        job2 = store.create(filename="b.pdf")
        assert job1.job_id != job2.job_id

    def test_create_sets_created_at(self):
        from services.ingest_api.store import JobStore

        store = JobStore()
        job = store.create(filename="test.pdf")
        assert job.created_at is not None

    def test_get_existing_job(self):
        from services.ingest_api.store import JobStore

        store = JobStore()
        job = store.create(filename="test.pdf")
        retrieved = store.get(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_get_nonexistent_returns_none(self):
        from services.ingest_api.store import JobStore

        store = JobStore()
        assert store.get("nonexistent-id") is None

    def test_update_status(self):
        from services.ingest_api.schemas import JobStatus
        from services.ingest_api.store import JobStore

        store = JobStore()
        job = store.create(filename="test.pdf")
        updated = store.update_status(job.job_id, JobStatus.PROCESSING)
        assert updated is not None
        assert updated.status == JobStatus.PROCESSING

    def test_update_status_with_error(self):
        from services.ingest_api.schemas import JobStatus
        from services.ingest_api.store import JobStore

        store = JobStore()
        job = store.create(filename="test.pdf")
        updated = store.update_status(
            job.job_id, JobStatus.FAILED, error_message="OCR failed"
        )
        assert updated is not None
        assert updated.status == JobStatus.FAILED
        assert updated.error_message == "OCR failed"

    def test_update_processing_records_started_at(self):
        from services.ingest_api.schemas import JobStatus
        from services.ingest_api.store import JobStore

        store = JobStore()
        job = store.create(filename="test.pdf")
        updated = store.update_status(job.job_id, JobStatus.PROCESSING)
        assert updated.started_at is not None

    def test_update_uploaded_records_completed_at(self):
        from services.ingest_api.schemas import JobStatus
        from services.ingest_api.store import JobStore

        store = JobStore()
        job = store.create(filename="test.pdf")
        store.update_status(job.job_id, JobStatus.PROCESSING)
        updated = store.update_status(job.job_id, JobStatus.UPLOADED)
        assert updated.completed_at is not None

    def test_update_nonexistent_returns_none(self):
        from services.ingest_api.schemas import JobStatus
        from services.ingest_api.store import JobStore

        store = JobStore()
        result = store.update_status("nonexistent", JobStatus.PROCESSING)
        assert result is None


# =========================================================================
# Shared fixtures for API tests
# =========================================================================


@pytest.fixture
def valid_pdf_bytes():
    """Minimal valid PDF file content."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen.canvas import Canvas

    buf = io.BytesIO()
    c = Canvas(buf, pagesize=A4)
    c.drawString(100, 700, "Test content")
    c.showPage()
    c.save()
    return buf.getvalue()


# =========================================================================
# Cycle A1: POST /jobs endpoint
# =========================================================================


class TestPostJobs:
    """Tests for POST /jobs endpoint."""

    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient
        from services.ingest_api.app import create_app
        from services.ingest_api.store import JobStore

        app = create_app(upload_dir=tmp_path, store=JobStore())
        return TestClient(app)

    def test_valid_pdf_returns_202(self, client, valid_pdf_bytes):
        response = client.post(
            "/jobs",
            files={"file": ("document.pdf", valid_pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 202

    def test_response_has_job_id(self, client, valid_pdf_bytes):
        response = client.post(
            "/jobs",
            files={"file": ("document.pdf", valid_pdf_bytes, "application/pdf")},
        )
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0

    def test_response_has_status_queued(self, client, valid_pdf_bytes):
        response = client.post(
            "/jobs",
            files={"file": ("document.pdf", valid_pdf_bytes, "application/pdf")},
        )
        data = response.json()
        assert data["status"] == "queued"

    def test_response_has_filename(self, client, valid_pdf_bytes):
        response = client.post(
            "/jobs",
            files={"file": ("my-report.pdf", valid_pdf_bytes, "application/pdf")},
        )
        data = response.json()
        assert data["filename"] == "my-report.pdf"

    def test_no_file_returns_422(self, client):
        response = client.post("/jobs")
        assert response.status_code == 422

    def test_non_pdf_returns_400(self, client):
        response = client.post(
            "/jobs",
            files={"file": ("document.txt", b"plain text content", "text/plain")},
        )
        assert response.status_code == 400

    def test_non_pdf_error_has_detail(self, client):
        response = client.post(
            "/jobs",
            files={"file": ("document.txt", b"plain text content", "text/plain")},
        )
        data = response.json()
        assert "detail" in data

    def test_uploaded_file_is_saved(self, client, valid_pdf_bytes, tmp_path):
        response = client.post(
            "/jobs",
            files={"file": ("document.pdf", valid_pdf_bytes, "application/pdf")},
        )
        job_id = response.json()["job_id"]
        saved_files = list(tmp_path.glob(f"{job_id}*"))
        assert len(saved_files) == 1

    def test_saved_file_has_pdf_content(self, client, valid_pdf_bytes, tmp_path):
        response = client.post(
            "/jobs",
            files={"file": ("document.pdf", valid_pdf_bytes, "application/pdf")},
        )
        job_id = response.json()["job_id"]
        saved = next(tmp_path.glob(f"{job_id}*"))
        assert saved.read_bytes()[:5] == b"%PDF-"


# =========================================================================
# Cycle A2: GET /jobs/{job_id} endpoint
# =========================================================================


class TestGetJob:
    """Tests for GET /jobs/{job_id} endpoint."""

    @pytest.fixture
    def client_and_store(self, tmp_path):
        from fastapi.testclient import TestClient
        from services.ingest_api.app import create_app
        from services.ingest_api.store import JobStore

        store = JobStore()
        app = create_app(upload_dir=tmp_path, store=store)
        return TestClient(app), store

    def test_existing_job_returns_200(self, client_and_store):
        client, store = client_and_store
        job = store.create(filename="test.pdf")
        response = client.get(f"/jobs/{job.job_id}")
        assert response.status_code == 200

    def test_response_has_job_fields(self, client_and_store):
        client, store = client_and_store
        job = store.create(filename="test.pdf")
        response = client.get(f"/jobs/{job.job_id}")
        data = response.json()
        assert data["job_id"] == job.job_id
        assert data["filename"] == "test.pdf"
        assert data["status"] == "queued"
        assert "created_at" in data

    def test_nonexistent_job_returns_404(self, client_and_store):
        client, _ = client_and_store
        response = client.get("/jobs/nonexistent-id")
        assert response.status_code == 404

    def test_404_has_detail(self, client_and_store):
        client, _ = client_and_store
        response = client.get("/jobs/nonexistent-id")
        assert "detail" in response.json()

    def test_status_reflects_store_updates(self, client_and_store):
        from services.ingest_api.schemas import JobStatus

        client, store = client_and_store
        job = store.create(filename="test.pdf")
        store.update_status(job.job_id, JobStatus.PROCESSING)

        response = client.get(f"/jobs/{job.job_id}")
        assert response.json()["status"] == "processing"

    def test_failed_job_includes_error(self, client_and_store):
        from services.ingest_api.schemas import JobStatus

        client, store = client_and_store
        job = store.create(filename="test.pdf")
        store.update_status(
            job.job_id, JobStatus.FAILED, error_message="Pipeline crash"
        )

        response = client.get(f"/jobs/{job.job_id}")
        data = response.json()
        assert data["status"] == "failed"
        assert data["error_message"] == "Pipeline crash"
