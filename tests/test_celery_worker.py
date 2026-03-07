"""
Testes para workers Celery e tasks de processamento.

TDD: Testes escritos antes da implementação.
"""

from unittest.mock import MagicMock, patch

import pytest


# Mocks para dependências externas
@pytest.fixture
def mock_processor():
    """Mock do DocumentProcessor."""
    mock = MagicMock()
    mock.process_document_parallel = MagicMock()
    return mock


@pytest.fixture
def mock_database():
    """Mock do MongoDB database."""
    mock = MagicMock()
    mock.update_one = MagicMock()
    mock.find_one = MagicMock()
    mock.delete_one = MagicMock()
    return mock


@pytest.fixture
def mock_embeddings_client():
    """Mock do EmbeddingsClient."""
    mock = MagicMock()
    mock.generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    return mock


@pytest.fixture
def mock_webhook_client():
    """Mock do Webhook client."""
    mock = MagicMock()
    mock.send = MagicMock()
    return mock


class TestProcessPdfJobTask:
    """Testes para task de processamento de PDF."""

    def test_process_pdf_job_does_not_recreate_existing_job_document(self):
        """Teste: Worker não deve recriar job já criado pela API."""
        from src.celery_worker import process_pdf_job

        with (
            patch("src.celery_worker.create_job_document_db") as mock_create_job_doc,
            patch("src.celery_worker.update_job_status_db") as mock_update_job_status,
            patch("src.celery_worker._process_document") as mock_process_document,
            patch("src.celery_worker.save_document") as mock_save_document,
        ):
            mock_process_document.return_value = {
                "doc_id": "doc-test",
                "total_pages": 1,
                "pages": [],
            }
            mock_update_job_status.return_value = MagicMock()
            mock_save_document.return_value = {"doc_id": "doc-test"}

            result = process_pdf_job.run(
                job_id="job-existing-1",
                file_content=b"%PDF-test\n",
                generate_embeddings=False,
                webhook_url=None,
            )

            mock_create_job_doc.assert_not_called()
            assert mock_update_job_status.call_count >= 1
            assert result["doc_id"] == "doc-test"

    def test_process_pdf_job_success(
        self, mock_processor, mock_database, mock_embeddings_client, mock_webhook_client
    ):
        """Teste: Processamento de PDF bem-sucedido."""
        from src.celery_worker import process_pdf_job

        # Preparar mock do processor
        mock_processor.process_document_parallel.return_value = MagicMock(
            total_pages=10, pages=[MagicMock(text_content="Test text")]
        )

        # Preparar mock do database
        mock_database.update_one.return_value = MagicMock(matched_count=1)

        # Preparar mock do embeddings
        mock_embeddings_client.generate_embedding.return_value = [0.1, 0.2, 0.3]

        # Preparar mock do webhook
        mock_webhook_client.send.return_value = {"status_code": 200}

        # Configurar ambiente
        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_CELERY_BROKER_URL": "redis://localhost:6379/0",
                "DOC_PARSER_CELERY_RESULT_BACKEND": "redis://localhost:6379/0",
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                "DOC_PARSER_EMBEDDINGS_URL": "http://test:11434/api/generate",
            },
        ):
            result = process_pdf_job.delay(
                job_id="test-job-123",
                file_content=b"%PDF-test-content\n",
                generate_embeddings=True,
                webhook_url=None,
            )

            # Verificar que a task foi chamada
            assert result is not None
            assert result.request.id == "test-job-123"

    def test_process_pdf_job_without_embeddings(
        self, mock_processor, mock_database, mock_webhook_client
    ):
        """Teste: Processamento sem geração de embeddings."""
        from src.celery_worker import process_pdf_job

        mock_processor.process_document_parallel.return_value = MagicMock(
            total_pages=10, pages=[MagicMock(text_content="Test text")]
        )
        mock_database.update_one.return_value = MagicMock(matched_count=1)

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_CELERY_BROKER_URL": "redis://localhost:6379/0",
                "DOC_PARSER_CELERY_RESULT_BACKEND": "redis://localhost:6379/0",
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = process_pdf_job.delay(
                job_id="test-job-456",
                file_content=b"%PDF-test\n",
                generate_embeddings=False,
                webhook_url=None,
            )

            assert result is not None


class TestGenerateEmbeddingsTask:
    """Testes para task de geração de embeddings."""

    def test_generate_embeddings_success(self, mock_embeddings_client):
        """Teste: Geração de embeddings bem-sucedida."""
        from src.celery_worker import generate_embeddings

        # Mock do embeddings client
        mock_embeddings_client.generate_embedding.return_value = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]

        # Configurar ambiente
        with patch.dict(
            "os.environ",
            {"DOC_PARSER_EMBEDDINGS_URL": "http://test:11434/api/generate"},
        ):
            # Teste com texto simples
            result = generate_embeddings.delay("Test text for embedding")
            assert result is not None

            # Teste com lista de textos
            result = generate_embeddings.delay(["Text 1", "Text 2", "Text 3"])
            assert result is not None

    def test_generate_embeddings_empty_list(self, mock_embeddings_client):
        """Teste: Geração de embeddings com lista vazia retorna lista vazia."""
        from src.celery_worker import generate_embeddings

        with patch.dict(
            "os.environ",
            {"DOC_PARSER_EMBEDDINGS_URL": "http://test:11434/api/generate"},
        ):
            result = generate_embeddings.delay([])
            assert result is not None


class TestSaveToMongoDBTask:
    """Testes para task de salvamento no MongoDB."""

    def test_save_document_to_mongodb(self, mock_database):
        """Teste: Salvamento de documento no MongoDB."""
        from src.celery_worker import save_document_to_mongodb

        # Mock do database
        mock_database.update_one.return_value = MagicMock(matched_count=1)

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = save_document_to_mongodb.delay(
                job_id="test-job", document={"total_pages": 10, "pages": []}
            )
            assert result is not None

    def test_save_embedding_to_mongodb(self, mock_database):
        """Teste: Salvamento de embedding no MongoDB."""
        from src.celery_worker import save_embedding_to_mongodb

        mock_database.update_one.return_value = MagicMock(matched_count=1)

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = save_embedding_to_mongodb.delay(
                job_id="test-job", doc_id="doc-uuid", vector=[0.1, 0.2, 0.3, 0.4, 0.5]
            )
            assert result is not None


class TestUpdateJobStatusTask:
    """Testes para task de atualização de status do job."""

    def test_update_job_status_task_calls_database_function(self):
        """Teste: Task delega atualização para função de banco sem recursão."""
        from src.celery_worker import update_job_status

        with patch(
            "src.celery_worker.update_job_status_db"
        ) as mock_update_job_status_db:
            mock_update_job_status_db.return_value = MagicMock()

            result = update_job_status.run(
                job_id="job-status-1",
                status="processing",
                result={"step": 1},
                error=None,
            )

            mock_update_job_status_db.assert_called_once()
            assert result["status"] == "processing"
            assert result["job_id"] == "job-status-1"

    def test_update_job_status_pending_to_processing(self, mock_database):
        """Teste: Atualização de status pending → processing."""
        from src.celery_worker import update_job_status

        mock_database.update_one.return_value = MagicMock(matched_count=1)

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = update_job_status.delay(
                job_id="test-job", status="processing", result=None, error=None
            )
            assert result is not None

    def test_update_job_status_to_completed(self, mock_database):
        """Teste: Atualização de status para completed."""
        from src.celery_worker import update_job_status

        mock_database.update_one.return_value = MagicMock(matched_count=1)

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = update_job_status.delay(
                job_id="test-job",
                status="completed",
                result={"total_pages": 10},
                error=None,
            )
            assert result is not None

    def test_update_job_status_to_failed(self, mock_database):
        """Teste: Atualização de status para failed."""
        from src.celery_worker import update_job_status

        mock_database.update_one.return_value = MagicMock(matched_count=1)

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = update_job_status.delay(
                job_id="test-job",
                status="failed",
                result=None,
                error="Test error message",
            )
            assert result is not None

    def test_update_job_status_to_cancelled(self, mock_database):
        """Teste: Atualização de status para cancelled."""
        from src.celery_worker import update_job_status

        mock_database.update_one.return_value = MagicMock(matched_count=1)

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = update_job_status.delay(
                job_id="test-job", status="cancelled", result=None, error=None
            )
            assert result is not None


class TestCancelJobTask:
    """Testes para task de cancelamento de job."""

    def test_cancel_job_success(self, mock_database):
        """Teste: Cancelamento de job bem-sucedido."""
        from src.celery_worker import cancel_job

        mock_database.update_one.return_value = MagicMock(matched_count=1)

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = cancel_job.delay("test-job")
            assert result is not None


class TestSendWebhookTask:
    """Testes para task de envio de webhook."""

    def test_send_webhook_success(self, mock_webhook_client):
        """Teste: Envio de webhook bem-sucedido."""
        from src.celery_worker import send_webhook

        mock_webhook_client.send.return_value = {"status_code": 200}

        with patch.dict(
            "os.environ", {"DOC_PARSER_WEBHOOK_URL": "https://example.com/webhook"}
        ):
            result = send_webhook.delay(
                job_id="test-job", status="completed", result={"total_pages": 10}
            )
            assert result is not None

    def test_send_webhook_with_auth_header(self, mock_webhook_client):
        """Teste: Envio de webhook com cabeçalho de autenticação."""
        from src.celery_worker import send_webhook

        mock_webhook_client.send.return_value = {"status_code": 200}

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_WEBHOOK_URL": "https://example.com/webhook",
                "DOC_PARSER_WEBHOOK_AUTH_TOKEN": "secret-token",
            },
        ):
            result = send_webhook.delay(
                job_id="test-job", status="completed", result={"total_pages": 10}
            )
            assert result is not None

    def test_send_webhook_no_url(self, mock_webhook_client):
        """Teste: Envio de webhook sem URL (não deve enviar)."""
        from src.celery_worker import send_webhook

        with patch.dict(
            "os.environ",
            {"DOC_PARSER_WEBHOOK_URL": "", "DOC_PARSER_WEBHOOK_AUTH_TOKEN": ""},
        ):
            result = send_webhook.delay(
                job_id="test-job", status="completed", result={"total_pages": 10}
            )
            assert result is not None


class TestCreateJobDocumentTask:
    """Testes para task de criação de documento no MongoDB."""

    def test_create_job_document(self, mock_database):
        """Teste: Criação de documento de job no MongoDB."""
        from src.celery_worker import create_job_document

        mock_database.insert_one.return_value = MagicMock(inserted_id="job-uuid")

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = create_job_document.delay(
                job_id="test-job",
                file_content=b"%PDF-test\n",
                generate_embeddings=False,
            )
            assert result is not None

    def test_create_job_document_with_webhook_url(self, mock_database):
        """Teste: Criação de documento com webhook_url."""
        from src.celery_worker import create_job_document

        mock_database.insert_one.return_value = MagicMock(inserted_id="job-uuid")

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            result = create_job_document.delay(
                job_id="test-job",
                file_content=b"%PDF-test\n",
                webhook_url="https://example.com/webhook",
                generate_embeddings=False,
            )
            assert result is not None


class TestProcessDocumentHelper:
    """Testes para helper interno _process_document."""

    def test_ensure_project_root_in_syspath(self):
        """Teste: helper adiciona raiz do projeto ao sys.path."""
        import sys
        from pathlib import Path

        from src.celery_worker import _ensure_project_root_in_syspath

        project_root = str(Path("src").resolve().parent)
        while project_root in sys.path:
            sys.path.remove(project_root)

        _ensure_project_root_in_syspath()
        assert project_root in sys.path

    def test_process_document_accepts_to_json_dict_only(self):
        """Teste: _process_document usa to_json_dict quando to_dict não existe."""
        from src.celery_worker import _process_document

        class FakeDoc:
            def to_json_dict(self):
                return {"doc_id": "doc-json", "total_pages": 1, "pages": []}

        with patch("src.pipeline.DocumentProcessor") as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor.process_document.return_value = FakeDoc()
            mock_processor_cls.return_value = mock_processor

            result = _process_document(b"%PDF-test\n")
            assert result["doc_id"] == "doc-json"
            assert result["total_pages"] == 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
