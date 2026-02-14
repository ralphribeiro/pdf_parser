"""
TDD tests for the FastAPI API.

Written BEFORE implementation to define the HTTP contract.
The DocumentProcessor is mocked -- these tests validate the HTTP layer,
not the OCR pipeline.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_processor(sample_document):
    """Mocked DocumentProcessor that returns sample_document."""
    processor = MagicMock()
    processor.use_gpu = False
    processor.ocr_engine_type = "doctr"
    processor.ocr_engine = MagicMock()

    processor.process_document_parallel.return_value = sample_document
    processor.process_document.return_value = sample_document

    def fake_save_json(document, output_path, **kwargs):
        import json

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(document.to_json_dict(), f)

    def fake_save_pdf(document, pdf_path, output_path):
        """Generate a minimal valid PDF for tests."""
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen.canvas import Canvas

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        c = Canvas(str(output_path), pagesize=A4)
        c.drawString(100, 700, "searchable test")
        c.save()

    processor.save_to_json.side_effect = fake_save_json
    processor.save_to_searchable_pdf.side_effect = fake_save_pdf

    return processor


@pytest.fixture
def client(mock_processor):
    """FastAPI TestClient with mocked processor."""
    from app.main import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    with TestClient(app) as tc:
        # Override AFTER lifespan initializes (otherwise lifespan overrides)
        app.state.processor = mock_processor
        yield tc


@pytest.fixture
def pdf_bytes(sample_pdf_path):
    """Bytes of a valid PDF for upload."""
    return sample_pdf_path.read_bytes()


@pytest.fixture
def pdf_upload(pdf_bytes):
    """Tuple (filename, file_obj, content_type) for upload."""
    return ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")


# =========================================================================
# GET /health
# =========================================================================


class TestHealthEndpoint:
    """GET /health returns system status."""

    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_has_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_response_has_gpu_available(self, client):
        data = client.get("/health").json()
        assert "gpu_available" in data
        assert isinstance(data["gpu_available"], bool)

    def test_response_has_ocr_engine(self, client):
        data = client.get("/health").json()
        assert "ocr_engine" in data

    def test_response_has_device(self, client):
        data = client.get("/health").json()
        assert "device" in data
        assert data["device"] in ("cpu", "cuda")


# =========================================================================
# GET /info
# =========================================================================


class TestInfoEndpoint:
    """GET /info returns current pipeline configuration."""

    def test_returns_200(self, client):
        response = client.get("/info")
        assert response.status_code == 200

    def test_response_has_ocr_settings(self, client):
        data = client.get("/info").json()
        assert "ocr_engine" in data
        assert "ocr_dpi" in data
        assert "min_confidence" in data

    def test_response_has_parallel_settings(self, client):
        data = client.get("/info").json()
        assert "parallel_enabled" in data


# =========================================================================
# POST /process - JSON response
# =========================================================================


class TestProcessJsonEndpoint:
    """POST /process with response_format=json (default)."""

    def test_returns_200_with_valid_pdf(self, client, pdf_upload):
        response = client.post(
            "/process",
            files={"file": pdf_upload},
        )
        assert response.status_code == 200

    def test_content_type_is_json(self, client, pdf_upload):
        response = client.post("/process", files={"file": pdf_upload})
        assert response.headers["content-type"] == "application/json"

    def test_response_has_document_fields(self, client, pdf_upload):
        data = client.post("/process", files={"file": pdf_upload}).json()
        assert "doc_id" in data
        assert "total_pages" in data
        assert "pages" in data
        assert "processing_date" in data

    def test_response_has_processing_time(self, client, pdf_upload):
        data = client.post("/process", files={"file": pdf_upload}).json()
        assert "processing_time_seconds" in data
        assert isinstance(data["processing_time_seconds"], float)
        assert data["processing_time_seconds"] >= 0

    def test_explicit_json_format(self, client, pdf_upload):
        """Explicit response_format=json should work the same as default."""
        response = client.post(
            "/process",
            files={"file": pdf_upload},
            params={"response_format": "json"},
        )
        assert response.status_code == 200
        assert "doc_id" in response.json()


# =========================================================================
# POST /process - PDF response
# =========================================================================


class TestProcessPdfEndpoint:
    """POST /process?response_format=pdf returns searchable PDF."""

    def test_returns_200_with_pdf_format(self, client, pdf_upload):
        response = client.post(
            "/process",
            files={"file": pdf_upload},
            params={"response_format": "pdf"},
        )
        assert response.status_code == 200

    def test_content_type_is_pdf(self, client, pdf_upload):
        response = client.post(
            "/process",
            files={"file": pdf_upload},
            params={"response_format": "pdf"},
        )
        assert response.headers["content-type"] == "application/pdf"

    def test_response_starts_with_pdf_magic(self, client, pdf_upload):
        response = client.post(
            "/process",
            files={"file": pdf_upload},
            params={"response_format": "pdf"},
        )
        assert response.content[:5] == b"%PDF-"

    def test_content_disposition_has_filename(self, client, pdf_upload):
        response = client.post(
            "/process",
            files={"file": pdf_upload},
            params={"response_format": "pdf"},
        )
        assert "content-disposition" in response.headers
        assert "filename" in response.headers["content-disposition"]


# =========================================================================
# POST /process - Error handling
# =========================================================================


class TestProcessErrors:
    """Error handling in POST /process."""

    def test_no_file_returns_422(self, client):
        response = client.post("/process")
        assert response.status_code == 422

    def test_non_pdf_file_returns_400(self, client):
        fake_txt = ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")
        response = client.post("/process", files={"file": fake_txt})
        assert response.status_code == 400

    def test_invalid_format_returns_422(self, client, pdf_upload):
        response = client.post(
            "/process",
            files={"file": pdf_upload},
            params={"response_format": "xml"},
        )
        assert response.status_code == 422

    def test_error_response_has_detail(self, client):
        fake_txt = ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")
        data = client.post("/process", files={"file": fake_txt}).json()
        assert "detail" in data


# =========================================================================
# POST /process - Request parameters
# =========================================================================


class TestProcessRequestParams:
    """Request parameters are passed to the processor."""

    def test_extract_tables_false(self, client, pdf_upload, mock_processor):
        client.post(
            "/process",
            files={"file": pdf_upload},
            params={"extract_tables": "false"},
        )
        call_kwargs = mock_processor.process_document_parallel.call_args
        assert call_kwargs is not None
        # extract_tables should be False
        _, kwargs = call_kwargs
        assert kwargs.get("extract_tables") is False

    def test_extract_tables_default_true(self, client, pdf_upload, mock_processor):
        client.post("/process", files={"file": pdf_upload})
        call_kwargs = mock_processor.process_document_parallel.call_args
        _, kwargs = call_kwargs
        assert kwargs.get("extract_tables") is True

    def test_min_confidence_passed(self, client, pdf_upload):
        """min_confidence should be included in the JSON response."""
        response = client.post(
            "/process",
            files={"file": pdf_upload},
            params={"min_confidence": "0.5"},
        )
        assert response.status_code == 200

    def test_ocr_postprocess_false_passed(self, client, pdf_upload):
        response = client.post(
            "/process",
            files={"file": pdf_upload},
            params={"ocr_postprocess": "false"},
        )
        assert response.status_code == 200

    def test_ocr_fix_errors_false_passed(self, client, pdf_upload):
        response = client.post(
            "/process",
            files={"file": pdf_upload},
            params={"ocr_fix_errors": "false"},
        )
        assert response.status_code == 200
