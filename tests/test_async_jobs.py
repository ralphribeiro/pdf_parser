"""
Testes para endpoints de jobs assíncronos.

TDD: Testes escritos antes da implementação.
"""

import os
import sys
import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True, scope="session")
def mock_mongodb():
    """Mock MongoDB para evitar conexão real durante testes."""

    # Mock pymongo
    mock_pymongo = MagicMock()
    mock_collection = MagicMock()
    mock_database = MagicMock()

    sys.modules["pymongo"] = mock_pymongo
    sys.modules["pymongo.collection"] = mock_collection
    sys.modules["pymongo.database"] = mock_database
    sys.modules["pymongo.results"] = MagicMock()
    sys.modules["pymongo.results.InsertOneResult"] = MagicMock()
    sys.modules["pymongo.results.UpdateResult"] = MagicMock()
    sys.modules["pymongo.results.DeleteResult"] = MagicMock()
    sys.modules["pymongo.engine"] = MagicMock()
    sys.modules["pymongo.cursor"] = MagicMock()
    sys.modules["pymongo.errors"] = MagicMock()

    yield

    # Clean up
    if "pymongo" in sys.modules:
        del sys.modules["pymongo"]
    if "pymongo.collection" in sys.modules:
        del sys.modules["pymongo.collection"]
    if "pymongo.database" in sys.modules:
        del sys.modules["pymongo.database"]
    if "pymongo.results" in sys.modules:
        del sys.modules["pymongo.results"]
    if "pymongo.engine" in sys.modules:
        del sys.modules["pymongo.engine"]
    if "pymongo.cursor" in sys.modules:
        del sys.modules["pymongo.cursor"]
    if "pymongo.errors" in sys.modules:
        del sys.modules["pymongo.errors"]


@pytest.fixture
def sample_pdf_bytes():
    """Bytes de PDF de exemplo."""
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"


@pytest.fixture
def sample_pdf_file():
    """Arquivo PDF temporário de exemplo."""
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"trailer\n<< /Root 1 0 R >>\n"
        b"startxref\n0\n"
        b"%%EOF\n"
    )

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        yield f.name
        os.unlink(f.name)


class TestCreateJobEndpoint:
    """Testes para POST /jobs."""

    def test_create_job_returns_202(self, client, sample_pdf_bytes):
        """Teste: Criar job retorna 202."""
        response = client.post(
            "/jobs/",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"generate_embeddings": "true"},
        )

        assert response.status_code == 202

    def test_create_job_returns_job_id(self, client, sample_pdf_bytes):
        """Teste: Criar job retorna job_id."""
        response = client.post(
            "/jobs/",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"generate_embeddings": "true"},
        )

        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0

    def test_create_job_returns_status_pending(self, client, sample_pdf_bytes):
        """Teste: Criar job retorna status pending."""
        response = client.post(
            "/jobs/",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"generate_embeddings": "false"},
        )

        data = response.json()
        assert data.get("status") == "pending"

    def test_create_job_returns_created_at(self, client, sample_pdf_bytes):
        """Teste: Criar job retorna created_at."""
        response = client.post(
            "/jobs/",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"generate_embeddings": "false"},
        )

        data = response.json()
        assert "created_at" in data

    def test_create_job_with_webhook_url(self, client, sample_pdf_bytes):
        """Teste: Criar job com webhook_url."""
        webhook_url = "https://example.com/webhook"

        response = client.post(
            "/jobs/",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"webhook_url": webhook_url, "generate_embeddings": "true"},
        )

        data = response.json()
        assert data.get("status") == "pending"

    def test_create_job_with_generate_embeddings_false(self, client, sample_pdf_bytes):
        """Teste: Criar job com generate_embeddings=false."""
        response = client.post(
            "/jobs/",
            files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"generate_embeddings": "false"},
        )

        data = response.json()
        assert data.get("status") == "pending"

    def test_create_job_file_too_large(self, client):
        """Teste: Criar job com arquivo muito grande."""
        # Criar PDF grande (> 50MB)
        large_pdf = b"%PDF-1.4\n" + b"X" * (50 * 1024 * 1024 + 100)

        response = client.post(
            "/jobs/", files={"file": ("test.pdf", large_pdf, "application/pdf")}
        )

        # Deve retornar erro (400 ou 413)
        assert response.status_code >= 400

    def test_create_job_invalid_file_type(self, client):
        """Teste: Criar job com arquivo não-PDF."""
        text_content = b"This is not a PDF file"

        response = client.post(
            "/jobs/", files={"file": ("test.txt", text_content, "text/plain")}
        )

        assert response.status_code >= 400

    def test_create_job_missing_file(self, client):
        """Teste: Criar job sem arquivo."""
        response = client.post("/jobs/", data={"generate_embeddings": "false"})

        assert response.status_code >= 400

    def test_create_job_file_not_pdf(self, client, sample_pdf_bytes):
        """Teste: Criar job com arquivo não-PDF."""
        # PDF válido mas com extensão .txt
        response = client.post(
            "/jobs/",
            files={"file": ("test.txt", sample_pdf_bytes, "application/octet-stream")},
        )

        # Deve aceitar o arquivo se for PDF válido
        assert response.status_code == 202


class TestGetJobEndpoint:
    """Testes para GET /jobs/{job_id}."""

    def test_get_job_returns_200_for_existing_job(self, client):
        """Teste: Obter job existente retorna 200."""
        # Criar job primeiro
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"

        response = client.post(
            "/jobs/", files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
        )

        job_id = response.json()["job_id"]

        # Obter job
        response = client.get(f"/jobs/{job_id}")

        assert response.status_code == 200

    def test_get_job_returns_job_status(self, client):
        """Teste: Obter job retorna status."""
        # Criar job primeiro
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"

        response = client.post(
            "/jobs/", files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
        )

        job_id = response.json()["job_id"]

        # Obter job
        response = client.get(f"/jobs/{job_id}")

        data = response.json()
        assert "status" in data

    def test_get_job_returns_result_when_completed(self, client):
        """Teste: Obter job retorna result quando completo."""
        # Criar job primeiro
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"

        response = client.post(
            "/jobs/", files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
        )

        job_id = response.json()["job_id"]

        # Simular job completo
        with patch("src.routers.async_jobs.get_job_document") as mock_get:
            mock_get.return_value = {
                "doc_id": "doc-uuid",
                "total_pages": 10,
                "pages": [],
            }

            response = client.get(f"/jobs/{job_id}")

            data = response.json()
            assert "result" in data

    def test_get_job_returns_error_when_failed(self, client):
        """Teste: Obter job retorna error quando falhou."""
        # Criar job primeiro
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"

        response = client.post(
            "/jobs/", files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
        )

        job_id = response.json()["job_id"]

        # Simular job falhou
        with patch("src.routers.async_jobs.get_job_document") as mock_get:
            mock_get.return_value = {
                "job_id": job_id,
                "status": "failed",
                "error": "Processamento falhou",
            }

            response = client.get(f"/jobs/{job_id}")

            data = response.json()
            assert "error" in data

    def test_get_job_returns_not_found_for_nonexistent_job(self, client):
        """Teste: Obter job inexistente retorna 404."""
        response = client.get("/jobs/nonexistent-job-id")

        assert response.status_code == 404

    def test_get_job_accepts_datetime_objects(self, client):
        """Teste: Endpoint aceita timestamps já em datetime."""
        job_id = "job-with-datetime-fields"
        created_at = datetime.now(UTC)
        updated_at = datetime.now(UTC)

        with patch("app.routers.async_jobs.get_job") as mock_get_job:
            mock_get_job.return_value = {
                "job_id": job_id,
                "status": "processing",
                "created_at": created_at,
                "updated_at": updated_at,
                "result": None,
                "error": None,
                "embeddings_generated": False,
            }

            response = client.get(f"/jobs/{job_id}")
            assert response.status_code == 200
            payload = response.json()
            assert payload["job_id"] == job_id
            assert payload["status"] == "processing"


class TestCancelJobEndpoint:
    """Testes para DELETE /jobs/{job_id}."""

    def test_cancel_job_returns_200(self, client):
        """Teste: Cancelar job retorna 200."""
        # Criar job primeiro
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"

        response = client.post(
            "/jobs/", files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
        )

        job_id = response.json()["job_id"]

        # Cancelar job
        response = client.delete(f"/jobs/{job_id}")

        assert response.status_code == 200

    def test_cancel_job_returns_not_found_for_nonexistent_job(self, client):
        """Teste: Cancelar job inexistente retorna 404."""
        response = client.delete("/jobs/nonexistent-job-id")

        assert response.status_code == 404


class TestSemanticSearchEndpoint:
    """Testes para POST /search/semantic."""

    def test_semantic_search_returns_200(self, client):
        """Teste: Busca semântica retorna 200."""
        response = client.post(
            "/search/semantic",
            json={
                "query": "Qual é o valor da cláusula de rescisão?",
                "top_k": 5,
                "min_score": 0.7,
            },
        )

        assert response.status_code == 200

    def test_semantic_search_returns_results(self, client):
        """Teste: Busca semântica retorna resultados."""
        response = client.post(
            "/search/semantic",
            json={"query": "Test query", "top_k": 10, "min_score": 0.5},
        )

        data = response.json()
        assert "results" in data
        assert "total_results" in data

    def test_semantic_search_with_filters(self, client):
        """Teste: Busca semântica com filtros."""
        response = client.post(
            "/search/semantic",
            json={
                "query": "Test query",
                "top_k": 5,
                "min_score": 0.7,
                "filters": {
                    "doc_id": "doc-uuid",
                    "created_after": "2026-01-01T00:00:00Z",
                },
            },
        )

        data = response.json()
        assert "results" in data

    def test_semantic_search_with_include_matches(self, client):
        """Teste: Busca semântica com include_matches."""
        response = client.post(
            "/search/semantic",
            json={
                "query": "Test query",
                "top_k": 5,
                "include_matches": True,
                "matches_limit": 3,
            },
        )

        data = response.json()
        assert "results" in data

    def test_semantic_search_empty_query(self, client):
        """Teste: Busca semântica com query vazia retorna erro."""
        response = client.post("/search/semantic", json={"query": "", "top_k": 5})

        # Pode retornar erro ou resultados vazios
        assert response.status_code in [200, 400]


class TestJobResponseSchema:
    """Testes para schema de resposta de job."""

    def test_job_response_has_required_fields(self):
        """Teste: Resposta de job tem campos obrigatórios."""
        from src.models.mongodb import JobResponse

        job_data = {
            "job_id": "test-job-123",
            "status": "completed",
            "created_at": "2026-03-04T12:00:00Z",
            "updated_at": "2026-03-04T12:05:00Z",
            "result": {"doc_id": "doc-uuid", "total_pages": 10, "pages": []},
            "embeddings_generated": True,
        }

        response = JobResponse(**job_data)

        assert response.job_id == "test-job-123"
        assert response.status == "completed"
        assert response.embeddings_generated == True

    def test_job_response_without_result(self):
        """Teste: Resposta de job sem result ainda é válida."""
        from src.models.mongodb import JobResponse

        job_data = {
            "job_id": "test-job-123",
            "status": "pending",
            "created_at": "2026-03-04T12:00:00Z",
        }

        response = JobResponse(**job_data)

        assert response.job_id == "test-job-123"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
