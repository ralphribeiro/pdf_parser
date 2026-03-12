"""
Tests for the deploy infrastructure (Milestone 5).

Covers: combined app (API under /api + UI at root), shared store
between API and UI, and the worker polling loop.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from services.ingest_api.store import JobStore

MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
    b"xref\n0 4\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
)


def _make_app(tmp_path, store=None):
    from services.app import create_combined_app

    if store is None:
        store = JobStore()
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    return create_combined_app(upload_dir=upload_dir, store=store), store


# =========================================================================
# Cycle D1: Combined app — API (/api) + UI (/)
# =========================================================================


class TestCombinedApp:
    """API (JSON at /api) and UI (HTML at /) share one store."""

    def test_ui_root_returns_upload_form(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert 'type="file"' in resp.text

    def test_api_create_job(self, tmp_path):
        app, _ = _make_app(tmp_path)
        resp = TestClient(app).post(
            "/api/jobs",
            files={"file": ("test.pdf", MINIMAL_PDF, "application/pdf")},
        )
        assert resp.status_code == 202
        assert "job_id" in resp.json()

    def test_api_get_job(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        resp = TestClient(app).get(f"/api/jobs/{job.job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job.job_id

    def test_ui_status_returns_html(self, tmp_path):
        app, store = _make_app(tmp_path)
        job = store.create("doc.pdf")
        resp = TestClient(app).get(f"/jobs/{job.job_id}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_shared_store_api_to_ui(self, tmp_path):
        """Job created via API is visible in UI."""
        app, _ = _make_app(tmp_path)
        client = TestClient(app)

        resp = client.post(
            "/api/jobs",
            files={"file": ("r.pdf", MINIMAL_PDF, "application/pdf")},
        )
        job_id = resp.json()["job_id"]

        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        assert "r.pdf" in resp.text

    def test_shared_store_ui_to_api(self, tmp_path):
        """Job created via UI upload is visible via API."""
        app, _ = _make_app(tmp_path)
        client = TestClient(app, follow_redirects=False)

        resp = client.post(
            "/upload",
            files={"file": ("x.pdf", MINIMAL_PDF, "application/pdf")},
        )
        job_id = resp.headers["location"].split("/jobs/")[-1]

        resp = TestClient(app).get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["filename"] == "x.pdf"


# =========================================================================
# Cycle D2: Worker runner — polling loop
# =========================================================================


class TestWorkerRunner:
    """run_loop should poll process_next and sleep when idle."""

    def test_processes_jobs(self):
        from services.worker.run import run_loop

        mock_worker = MagicMock()
        mock_worker.process_next.return_value = True

        run_loop(mock_worker, interval=0.01, max_iterations=3)

        assert mock_worker.process_next.call_count == 3

    def test_sleeps_when_idle(self):
        from services.worker.run import run_loop

        mock_worker = MagicMock()
        mock_worker.process_next.return_value = False

        start = time.monotonic()
        run_loop(mock_worker, interval=0.05, max_iterations=2)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.09

    def test_respects_max_iterations(self):
        from services.worker.run import run_loop

        mock_worker = MagicMock()
        mock_worker.process_next.return_value = True

        run_loop(mock_worker, interval=0.01, max_iterations=5)

        assert mock_worker.process_next.call_count == 5

    def test_no_sleep_when_busy(self):
        from services.worker.run import run_loop

        mock_worker = MagicMock()
        mock_worker.process_next.return_value = True

        start = time.monotonic()
        run_loop(mock_worker, interval=1.0, max_iterations=3)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5
