"""
Tests for the OCR worker and supporting extensions.

Covers: Job schema extensions (file_hash), JobStore.get_next_queued,
compute_file_hash, OcrWorker.process_job lifecycle, and
OcrWorker.process_next queue consumption.
"""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingest_api.schemas import Job, JobStatus
from services.ingest_api.store import JobStore

# =========================================================================
# Cycle W1: Job schema extensions
# =========================================================================


class TestJobWorkerFields:
    """The Job model must carry a file hash for idempotency."""

    def test_has_file_hash(self):
        job = Job(
            job_id="j1",
            filename="doc.pdf",
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            file_hash="abc123def456",
        )
        assert job.file_hash == "abc123def456"

    def test_file_hash_defaults_none(self):
        job = Job(
            job_id="j1",
            filename="doc.pdf",
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
        )
        assert job.file_hash is None

    def test_serializes_file_hash(self):
        job = Job(
            job_id="j1",
            filename="doc.pdf",
            status=JobStatus.UPLOADED,
            created_at=datetime.now(),
            file_hash="deadbeef",
        )
        data = job.model_dump()
        assert data["file_hash"] == "deadbeef"


# =========================================================================
# Cycle W2: JobStore.get_next_queued & update_status extensions
# =========================================================================


class TestJobStoreQueue:
    """JobStore must expose a queue-like interface for the worker."""

    def test_get_next_queued_returns_oldest(self):
        store = JobStore()
        j1 = store.create("first.pdf")
        store.create("second.pdf")

        nxt = store.get_next_queued()
        assert nxt is not None
        assert nxt.job_id == j1.job_id

    def test_get_next_queued_returns_none_when_empty(self):
        store = JobStore()
        assert store.get_next_queued() is None

    def test_get_next_queued_skips_non_queued(self):
        store = JobStore()
        j1 = store.create("a.pdf")
        j2 = store.create("b.pdf")

        store.update_status(j1.job_id, JobStatus.PROCESSING)

        nxt = store.get_next_queued()
        assert nxt is not None
        assert nxt.job_id == j2.job_id

    def test_update_status_sets_file_hash(self):
        store = JobStore()
        job = store.create("doc.pdf")

        updated = store.update_status(
            job.job_id,
            JobStatus.PROCESSING,
            file_hash="sha256hex",
        )
        assert updated is not None
        assert updated.file_hash == "sha256hex"


# =========================================================================
# Cycle W2b: compute_file_hash
# =========================================================================


class TestFileHash:
    """Tests for compute_file_hash (SHA-256 for idempotency)."""

    def test_returns_hex_string(self, tmp_path):
        from services.worker.ocr_worker import compute_file_hash

        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 content")

        h = compute_file_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_same_content_same_hash(self, tmp_path):
        from services.worker.ocr_worker import compute_file_hash

        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"%PDF-1.4 identical")
        b.write_bytes(b"%PDF-1.4 identical")

        assert compute_file_hash(a) == compute_file_hash(b)

    def test_different_content_different_hash(self, tmp_path):
        from services.worker.ocr_worker import compute_file_hash

        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"%PDF-1.4 content A")
        b.write_bytes(b"%PDF-1.4 content B")

        assert compute_file_hash(a) != compute_file_hash(b)


# =========================================================================
# Cycle W3: OcrWorker.process_job lifecycle
# =========================================================================


class TestOcrWorkerProcessJob:
    """Worker must drive the full job lifecycle: queued -> processing -> uploaded."""

    def _make_worker(self, store, tmp_path, *, artifact_fn=None, semantic_indexer=None):
        from services.worker.ocr_worker import OcrWorker

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(exist_ok=True)
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)

        if artifact_fn is None:
            from src.pipeline import ArtifactResult

            def _fake_artifacts(pdf_path, output_dir, **kwargs):
                json_p = Path(output_dir) / "test.json"
                pdf_p = Path(output_dir) / "test_searchable.pdf"
                json_p.write_text("{}")
                pdf_p.write_bytes(b"%PDF-1.4 fake searchable")
                return ArtifactResult(
                    json_path=json_p,
                    pdf_path=pdf_p,
                    document=MagicMock(),
                )

            artifact_fn = _fake_artifacts

        return OcrWorker(
            store=store,
            upload_dir=upload_dir,
            output_dir=output_dir,
            artifact_fn=artifact_fn,
            semantic_indexer=semantic_indexer,
        )

    def _seed_job(self, store, upload_dir):
        """Create a queued job and put a PDF in upload_dir."""
        job = store.create("sample.pdf")
        pdf = upload_dir / f"{job.job_id}.pdf"
        pdf.write_bytes(b"%PDF-1.4 sample content")
        return job

    def test_sets_status_to_processing(self, tmp_path):
        store = JobStore()
        worker = self._make_worker(store, tmp_path)
        job = self._seed_job(store, worker.upload_dir)

        worker.process_job(job.job_id)

        final = store.get(job.job_id)
        assert final is not None
        assert final.started_at is not None

    def test_calls_artifact_fn(self, tmp_path):
        store = JobStore()
        mock_fn = MagicMock()

        from src.pipeline import ArtifactResult

        mock_fn.return_value = ArtifactResult(
            json_path=tmp_path / "x.json",
            pdf_path=tmp_path / "x.pdf",
            document=MagicMock(),
        )
        (tmp_path / "x.json").write_text("{}")
        (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4")

        worker = self._make_worker(store, tmp_path, artifact_fn=mock_fn)
        job = self._seed_job(store, worker.upload_dir)

        worker.process_job(job.job_id)

        mock_fn.assert_called_once()
        call_args = mock_fn.call_args
        assert str(call_args[0][0]).endswith(".pdf")

    def test_sets_status_to_uploaded(self, tmp_path):
        store = JobStore()
        worker = self._make_worker(store, tmp_path)
        job = self._seed_job(store, worker.upload_dir)

        worker.process_job(job.job_id)

        final = store.get(job.job_id)
        assert final is not None
        assert final.status == JobStatus.UPLOADED
        assert final.completed_at is not None

    def test_stores_file_hash(self, tmp_path):
        store = JobStore()
        worker = self._make_worker(store, tmp_path)
        job = self._seed_job(store, worker.upload_dir)

        worker.process_job(job.job_id)

        final = store.get(job.job_id)
        assert final is not None
        assert final.file_hash is not None
        assert len(final.file_hash) == 64

    def test_pipeline_error_sets_failed(self, tmp_path):
        store = JobStore()

        def _boom(*args, **kwargs):
            raise RuntimeError("OCR engine exploded")

        worker = self._make_worker(store, tmp_path, artifact_fn=_boom)
        job = self._seed_job(store, worker.upload_dir)

        worker.process_job(job.job_id)

        final = store.get(job.job_id)
        assert final is not None
        assert final.status == JobStatus.FAILED
        assert "OCR engine exploded" in (final.error_message or "")

    def test_indexes_document_after_success(self, tmp_path):
        store = JobStore()
        indexer = MagicMock()
        worker = self._make_worker(store, tmp_path, semantic_indexer=indexer)
        job = self._seed_job(store, worker.upload_dir)

        worker.process_job(job.job_id)

        indexer.index_document.assert_called_once()
        args = indexer.index_document.call_args.args
        assert args[0] == job.job_id

    def test_index_error_sets_failed(self, tmp_path):
        store = JobStore()
        indexer = MagicMock()
        indexer.index_document.side_effect = RuntimeError("index backend down")
        worker = self._make_worker(store, tmp_path, semantic_indexer=indexer)
        job = self._seed_job(store, worker.upload_dir)

        worker.process_job(job.job_id)

        final = store.get(job.job_id)
        assert final is not None
        assert final.status == JobStatus.FAILED
        assert "index backend down" in (final.error_message or "")


# =========================================================================
# Cycle W4: OcrWorker.process_next
# =========================================================================


class TestOcrWorkerProcessNext:
    """Worker should consume the queue automatically."""

    def _make_worker(self, store, tmp_path):
        from services.worker.ocr_worker import OcrWorker

        from src.pipeline import ArtifactResult

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(exist_ok=True)
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)

        def _fake_artifacts(pdf_path, output_dir, **kwargs):
            json_p = Path(output_dir) / "test.json"
            pdf_p = Path(output_dir) / "test_searchable.pdf"
            json_p.write_text("{}")
            pdf_p.write_bytes(b"%PDF-1.4 fake")
            return ArtifactResult(
                json_path=json_p, pdf_path=pdf_p, document=MagicMock()
            )

        return OcrWorker(
            store=store,
            upload_dir=upload_dir,
            output_dir=output_dir,
            artifact_fn=_fake_artifacts,
        )

    def test_processes_next_queued_job(self, tmp_path):
        store = JobStore()
        worker = self._make_worker(store, tmp_path)

        job = store.create("doc.pdf")
        pdf = worker.upload_dir / f"{job.job_id}.pdf"
        pdf.write_bytes(b"%PDF-1.4 content")

        result = worker.process_next()
        assert result is True

        final = store.get(job.job_id)
        assert final is not None
        assert final.status == JobStatus.UPLOADED

    def test_returns_false_when_empty(self, tmp_path):
        store = JobStore()
        worker = self._make_worker(store, tmp_path)

        result = worker.process_next()
        assert result is False

    def test_skips_processing_jobs(self, tmp_path):
        store = JobStore()
        worker = self._make_worker(store, tmp_path)

        job = store.create("doc.pdf")
        store.update_status(job.job_id, JobStatus.PROCESSING)

        result = worker.process_next()
        assert result is False
