"""
Tests for the ingest UI web interface (Milestone 4).

Covers: upload page (GET /), upload action (POST /upload),
status page (GET /jobs/{job_id}) with auto-refresh, and
error feedback.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from services.ingest_api.schemas import JobStatus
from services.ingest_api.store import JobStore

MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
    b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
)


def _make_app(tmp_path, store=None):
    from services.ingest_ui.app import create_ui_app

    if store is None:
        store = JobStore()
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    return create_ui_app(upload_dir=upload_dir, store=store), store


# =========================================================================
# Cycle U1: Upload page — GET /
# =========================================================================


class TestUploadPage:
    """GET /upload should render an HTML upload form."""

    def test_returns_200(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/upload")
        assert resp.status_code == 200

    def test_returns_html(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/upload")
        assert "text/html" in resp.headers["content-type"]

    def test_has_file_input(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/upload")
        assert 'type="file"' in resp.text

    def test_has_multipart_form(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/upload")
        html = resp.text.lower()
        assert "<form" in html
        assert 'method="post"' in html
        assert "multipart/form-data" in html


# =========================================================================
# Cycle U2: Upload action — POST /upload
# =========================================================================


class TestUploadAction:
    """POST /upload should validate, create job, and redirect."""

    def test_valid_pdf_redirects_to_status(self, tmp_path):
        app, _ = _make_app(tmp_path)
        client = TestClient(app, follow_redirects=False)
        resp = client.post(
            "/upload",
            files={"file": ("test.pdf", MINIMAL_PDF, "application/pdf")},
        )
        assert resp.status_code == 303
        assert "/jobs/" in resp.headers["location"]

    def test_creates_job_in_store(self, tmp_path):
        app, store = _make_app(tmp_path)
        client = TestClient(app, follow_redirects=False)
        resp = client.post(
            "/upload",
            files={"file": ("doc.pdf", MINIMAL_PDF, "application/pdf")},
        )
        job_id = resp.headers["location"].split("/jobs/")[-1]
        job = store.get(job_id)
        assert job is not None
        assert job.filename == "doc.pdf"

    def test_saves_file_to_upload_dir(self, tmp_path):
        app, _store = _make_app(tmp_path)
        upload_dir = tmp_path / "uploads"
        client = TestClient(app, follow_redirects=False)
        resp = client.post(
            "/upload",
            files={"file": ("x.pdf", MINIMAL_PDF, "application/pdf")},
        )
        job_id = resp.headers["location"].split("/jobs/")[-1]
        saved = upload_dir / f"{job_id}.pdf"
        assert saved.exists()
        assert saved.read_bytes()[:5] == b"%PDF-"

    def test_non_pdf_extension_shows_error(self, tmp_path):
        app, _ = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.post(
            "/upload",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert "text/html" in resp.headers["content-type"]
        html = resp.text.lower()
        assert "pdf" in html

    def test_invalid_pdf_content_shows_error(self, tmp_path):
        app, _ = _make_app(tmp_path)
        client = TestClient(app)
        resp = client.post(
            "/upload",
            files={"file": ("fake.pdf", b"not-a-pdf", "application/pdf")},
        )
        assert "text/html" in resp.headers["content-type"]
        html = resp.text.lower()
        assert "pdf" in html


# =========================================================================
# Cycle U3: Status page — GET /jobs/{job_id}
# =========================================================================


class TestStatusPage:
    """GET /jobs/{job_id} should render job status as HTML."""

    def test_returns_200(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        assert resp.status_code == 200

    def test_returns_html(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        assert "text/html" in resp.headers["content-type"]

    def test_shows_filename(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("relatorio-2026.pdf")
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        assert "relatorio-2026.pdf" in resp.text

    def test_shows_queued_status(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        assert "queued" in resp.text.lower()

    def test_shows_uploaded_status(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        store.update_status(job.job_id, JobStatus.UPLOADED)
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        assert "uploaded" in resp.text.lower()

    def test_auto_refresh_for_queued(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        html = resp.text.lower()
        has_refresh = (
            'http-equiv="refresh"' in html
            or "setinterval" in html
            or "settimeout" in html
        )
        assert has_refresh, "Pending jobs need auto-refresh"

    def test_auto_refresh_for_processing(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        store.update_status(job.job_id, JobStatus.PROCESSING)
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        html = resp.text.lower()
        has_refresh = (
            'http-equiv="refresh"' in html
            or "setinterval" in html
            or "settimeout" in html
        )
        assert has_refresh, "Processing jobs need auto-refresh"

    def test_no_auto_refresh_for_uploaded(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        store.update_status(job.job_id, JobStatus.UPLOADED)
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        assert 'http-equiv="refresh"' not in resp.text.lower()

    def test_no_auto_refresh_for_failed(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        store.update_status(job.job_id, JobStatus.FAILED)
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        assert 'http-equiv="refresh"' not in resp.text.lower()

    def test_shows_error_for_failed_job(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        store.update_status(job.job_id, JobStatus.FAILED, error_message="OCR crashed")
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        assert "OCR crashed" in resp.text

    def test_unknown_job_returns_404(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/jobs/nonexistent-id")
        assert resp.status_code == 404
