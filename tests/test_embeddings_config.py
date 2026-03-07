"""
Testes para configuração de embeddings (OpenAI-compatible API).

TDD: Testes escritos antes da implementação.
"""

from unittest.mock import MagicMock

import pytest

# Mock do requests para testes


@pytest.fixture
def sample_embedding_response():
    """Mock response de API de embeddings."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}
        ],
        "model": "nomic-embed-text",
    }


class TestEmbeddingsConfig:
    """Testes para configuração de embeddings."""

    def test_get_config_with_env_variables(self, monkeypatch):
        """Teste: Configuração carrega variáveis de ambiente corretamente."""
        # Redefinir variáveis de ambiente (Ollama v1/embeddings)
        monkeypatch.setenv(
            "DOC_PARSER_EMBEDDINGS_URL", "http://localhost:8080/v1/embeddings"
        )
        monkeypatch.setenv("DOC_PARSER_EMBEDDINGS_MODEL", "Qwen3-Embedding")
        monkeypatch.setenv("DOC_PARSER_EMBEDDINGS_API_KEY", "llama.cpp")

        # Importar após definir env vars
        from config import EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL, EMBEDDINGS_URL

        assert EMBEDDINGS_URL == "http://localhost:8080/v1/embeddings"
        assert EMBEDDINGS_MODEL == "Qwen3-Embedding"
        assert EMBEDDINGS_API_KEY == "llama.cpp"

    def test_get_config_with_defaults(self, monkeypatch):
        """Teste: Configuração usa valores padrão quando env não definido."""
        # Garantir que as envs não existem
        monkeypatch.delenv("DOC_PARSER_EMBEDDINGS_URL", raising=False)
        monkeypatch.delenv("DOC_PARSER_EMBEDDINGS_MODEL", raising=False)
        monkeypatch.delenv("DOC_PARSER_EMBEDDINGS_API_KEY", raising=False)

        # Importar após definir env vars
        from config import EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL, EMBEDDINGS_URL

        # Verificar que valores padrão existem (serão verificados na implementação)
        assert EMBEDDINGS_URL is not None
        assert EMBEDDINGS_MODEL is not None
        assert EMBEDDINGS_API_KEY is not None

    def test_embed_request_format(self, monkeypatch, sample_embedding_response):
        """Teste: Request format para API de embeddings é OpenAI-compatible."""
        from src.embeddings import build_request

        text = "Test text for embedding"
        request = build_request(text, model="Qwen3-Embedding")

        assert request is not None
        assert "model" in request
        assert "input" in request
        assert request["input"] == text
        assert request["model"] == "Qwen3-Embedding"

    def test_extract_embedding_from_response(
        self, monkeypatch, sample_embedding_response
    ):
        """Teste: Extração de embedding de resposta da API."""
        monkeypatch.setenv(
            "DOC_PARSER_EMBEDDINGS_URL", "http://localhost:8080/v1/embeddings"
        )
        monkeypatch.setenv("DOC_PARSER_EMBEDDINGS_MODEL", "Qwen3-Embedding")

        from src.embeddings import extract_embedding

        mock_response = MagicMock()
        mock_response.json.return_value = sample_embedding_response

        embedding = extract_embedding(mock_response)

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]


class TestCeleryConfig:
    """Testes para configuração de Celery."""

    def test_celery_config_with_env_variables(self, monkeypatch):
        """Teste: Configuração Celery carrega variáveis de ambiente."""
        monkeypatch.setenv("DOC_PARSER_CELERY_BROKER_URL", "redis://localhost:6379/0")
        monkeypatch.setenv(
            "DOC_PARSER_CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
        )
        monkeypatch.setenv("DOC_PARSER_CELERY_WORKERS", "2")

        from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, CELERY_WORKERS

        assert CELERY_BROKER_URL == "redis://localhost:6379/0"
        assert CELERY_RESULT_BACKEND == "redis://localhost:6379/0"
        assert CELERY_WORKERS == "2"

    def test_celery_config_with_defaults(self, monkeypatch):
        """Teste: Configuração Celery usa valores padrão."""
        monkeypatch.delenv("DOC_PARSER_CELERY_BROKER_URL", raising=False)
        monkeypatch.delenv("DOC_PARSER_CELERY_RESULT_BACKEND", raising=False)
        monkeypatch.delenv("DOC_PARSER_CELERY_WORKERS", raising=False)

        from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, CELERY_WORKERS

        assert CELERY_BROKER_URL is not None
        assert CELERY_RESULT_BACKEND is not None
        assert CELERY_WORKERS is not None

    def test_celery_task_serializer_default(self, monkeypatch):
        """Teste: Serializer padrão para tasks (pickle)."""
        monkeypatch.setenv("DOC_PARSER_CELERY_BROKER_URL", "redis://localhost:6379/0")

        from config import CELERY_RESULT_SERIALIZER, CELERY_TASK_SERIALIZER

        assert CELERY_TASK_SERIALIZER is not None
        assert CELERY_RESULT_SERIALIZER is not None


class TestMongoDBConfig:
    """Testes para configuração de MongoDB."""

    def test_mongodb_config_with_env_variables(self, monkeypatch):
        """Teste: Configuração MongoDB carrega variáveis de ambiente."""
        monkeypatch.setenv("DOC_PARSER_MONGODB_URI", "mongodb://localhost:27017")
        monkeypatch.setenv("DOC_PARSER_MONGODB_DB", "caseiro_docs")
        monkeypatch.setenv("DOC_PARSER_MONGODB_USE_VECTORS", "true")

        from config import MONGODB_DB, MONGODB_URI, MONGODB_USE_VECTORS

        assert MONGODB_URI == "mongodb://localhost:27017"
        assert MONGODB_DB == "caseiro_docs"
        assert MONGODB_USE_VECTORS == "true"

    def test_mongodb_config_with_defaults(self, monkeypatch):
        """Teste: Configuração MongoDB usa valores padrão."""
        monkeypatch.delenv("DOC_PARSER_MONGODB_URI", raising=False)
        monkeypatch.delenv("DOC_PARSER_MONGODB_DB", raising=False)
        monkeypatch.delenv("DOC_PARSER_MONGODB_USE_VECTORS", raising=False)

        from config import MONGODB_DB, MONGODB_URI, MONGODB_USE_VECTORS

        assert MONGODB_URI is not None
        assert MONGODB_DB is not None
        assert MONGODB_USE_VECTORS is not None


class TestJobSchemas:
    """Testes para schemas de jobs (Pydantic)."""

    def test_job_status_values(self):
        """Teste: Valores válidos para status de job."""
        from src.models.mongodb import JobStatus

        valid_statuses = ["pending", "processing", "completed", "failed", "cancelled"]

        for status in valid_statuses:
            assert status in JobStatus._value2member_map_.values()

    def test_job_create_schema(self):
        """Teste: Schema JobCreate valida campos obrigatórios."""
        from src.models.mongodb import JobCreate

        # Teste com campos mínimos
        job_data = {"file_content": b"%PDF-1.4\n", "generate_embeddings": False}

        job_create = JobCreate(**job_data)
        assert job_create.file_content == b"%PDF-1.4\n"
        assert job_create.generate_embeddings == False

    def test_job_create_with_webhook(self):
        """Teste: JobCreate aceita webhook_url opcional."""
        from src.models.mongodb import JobCreate

        job_data = {
            "file_content": b"%PDF-1.4\n",
            "webhook_url": "https://example.com/webhook",
            "generate_embeddings": True,
        }

        job_create = JobCreate(**job_data)
        assert job_create.webhook_url == "https://example.com/webhook"
        assert job_create.generate_embeddings == True

    def test_job_response_schema(self):
        """Teste: JobResponseSchema inclui todos os campos."""
        from src.models.mongodb import JobResponse

        job_data = {
            "job_id": "test-job-123",
            "status": "completed",
            "created_at": "2026-03-04T12:00:00Z",
            "updated_at": "2026-03-04T12:05:00Z",
            "result": {"doc_id": "doc-uuid", "total_pages": 10, "pages": []},
            "embeddings_generated": True,
        }

        job_response = JobResponse(**job_data)
        assert job_response.job_id == "test-job-123"
        assert job_response.status == "completed"
        assert job_response.embeddings_generated == True

    def test_job_response_invalid_status(self):
        """Teste: JobResponse rejeita status inválido."""
        from src.models.mongodb import JobResponse

        with pytest.raises(Exception) as excinfo:
            JobResponse(
                job_id="test-job-123",
                status="invalid_status",
                created_at="2026-03-04T12:00:00Z",
            )
        assert (
            "invalid_status" in str(excinfo.value)
            or "job status" in str(excinfo.value).lower()
        )


class TestDocumentSchemas:
    """Testes para schemas de documentos MongoDB."""

    def test_document_create_schema(self):
        """Teste: DocumentCreate valida campos obrigatórios."""
        from src.models.mongodb import DocumentCreate

        doc_data = {
            "doc_id": "doc-uuid",
            "source_file": "document.pdf",
            "total_pages": 10,
            "pages": [],
        }

        doc_create = DocumentCreate(**doc_data)
        assert doc_create.doc_id == "doc-uuid"
        assert doc_create.total_pages == 10

    def test_document_response_schema(self):
        """Teste: DocumentResponse inclui todos os campos."""
        from datetime import datetime

        from src.models.mongodb import DocumentResponse

        doc_data = {
            "id": "doc-uuid",
            "doc_id": "doc-uuid",
            "source_file": "document.pdf",
            "total_pages": 10,
            "status": "completed",
            "pages": [],
            "processing_date": datetime(2026, 3, 4, 12, 0, 0),
            "created_at": datetime(2026, 3, 4, 12, 0, 0),
            "completed_at": datetime(2026, 3, 4, 12, 5, 0),
        }

        doc_response = DocumentResponse(**doc_data)
        assert doc_response.doc_id == "doc-uuid"
        assert doc_response.status == "completed"


class TestEmbeddingSchemas:
    """Testes para schemas de embeddings."""

    def test_embedding_create_schema(self):
        """Teste: EmbeddingCreate valida campos obrigatórios."""
        from src.models.mongodb import EmbeddingCreate

        emb_data = {
            "doc_id": "doc-uuid",
            "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            "model": "nomic-embed-text",
        }

        emb_create = EmbeddingCreate(**emb_data)
        assert emb_create.doc_id == "doc-uuid"
        assert len(emb_create.vector) == 5

    def test_embedding_response_schema(self):
        """Teste: EmbeddingResponse inclui todos os campos."""
        from src.models.mongodb import EmbeddingResponse

        emb_data = {
            "_id": "emb-uuid",
            "doc_id": "doc-uuid",
            "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            "model": "nomic-embed-text",
            "chunk_size": 1024,
            "created_at": "2026-03-04T12:00:00Z",
        }

        emb_response = EmbeddingResponse(**emb_data)
        assert emb_response.doc_id == "doc-uuid"
        assert len(emb_response.vector) == 5


class TestChunkSchemas:
    """Testes para schemas de chunks."""

    def test_chunk_create_schema(self):
        """Teste: ChunkCreate valida campos obrigatórios."""
        from src.models.mongodb import ChunkCreate

        chunk_data = {
            "doc_id": "doc-uuid",
            "text": "Texto do chunk...",
            "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            "page": 1,
            "block_id": "p1_b1",
        }

        chunk_create = ChunkCreate(**chunk_data)
        assert chunk_create.doc_id == "doc-uuid"
        assert chunk_create.page == 1

    def test_chunk_response_schema(self):
        """Teste: ChunkResponse inclui todos os campos."""
        from src.models.mongodb import ChunkResponse

        chunk_data = {
            "_id": "chunk-uuid",
            "doc_id": "doc-uuid",
            "text": "Texto do chunk...",
            "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            "page": 1,
            "block_id": "p1_b1",
            "metadata": {},
        }

        chunk_response = ChunkResponse(**chunk_data)
        assert chunk_response.doc_id == "doc-uuid"
        assert chunk_response.page == 1

    def test_search_result_schema(self):
        """Teste: SearchResultSchema valida campos de busca."""
        from src.models.mongodb import SearchResult

        result_data = {
            "doc_id": "doc-uuid",
            "score": 0.92,
            "total_pages": 45,
            "created_at": "2026-03-04T10:00:00Z",
            "matches": [
                {
                    "page": 12,
                    "block_id": "p12_b3",
                    "text": "Texto do match...",
                    "bbox": [0.1, 0.2, 0.8, 0.25],
                }
            ],
        }

        search_result = SearchResult(**result_data)
        assert search_result.doc_id == "doc-uuid"
        assert search_result.score == 0.92
        assert len(search_result.matches) == 1

    def test_semantic_search_response_schema(self):
        """Teste: SemanticSearchResponseSchema valida resposta de busca."""
        from src.models.mongodb import SemanticSearchResponse

        search_data = {
            "query": "Texto da busca...",
            "total_results": 3,
            "results": [
                {
                    "doc_id": "doc-uuid",
                    "score": 0.92,
                    "total_pages": 45,
                    "created_at": "2026-03-04T10:00:00Z",
                    "matches": [
                        {
                            "page": 12,
                            "block_id": "p12_b3",
                            "text": "Texto do match...",
                            "bbox": [0.1, 0.2, 0.8, 0.25],
                        }
                    ],
                }
            ],
        }

        search_response = SemanticSearchResponse(**search_data)
        assert search_response.query == "Texto da busca..."
        assert search_response.total_results == 3
        assert len(search_response.results) == 1


class TestWebhookSchemas:
    """Testes para schemas de webhooks."""

    def test_webhook_payload_schema(self):
        """Teste: WebhookPayloadSchema valida payload de notificação."""
        from src.models.mongodb import WebhookPayload

        payload_data = {
            "job_id": "job-uuid",
            "status": "completed",
            "result": {"doc_id": "doc-uuid", "total_pages": 10, "pages": []},
        }

        webhook = WebhookPayload(**payload_data)
        assert webhook.job_id == "job-uuid"
        assert webhook.status == "completed"


class TestSearchFilters:
    """Testes para filtros de busca semântica."""

    def test_search_filters_validation(self):
        """Teste: Filtros de busca são válidos."""
        from src.models.mongodb import SearchFilters

        # Teste com filtros básicos
        filters = SearchFilters(doc_id="doc-uuid")
        assert filters.doc_id == "doc-uuid"

        # Teste com múltiplos filtros
        filters = SearchFilters(
            doc_id="doc-uuid", min_score=0.7, created_after="2026-01-01T00:00:00Z"
        )
        assert filters.min_score == 0.7

    def test_search_params_schema(self):
        """Teste: SearchParamsSchema valida parâmetros de busca."""
        from src.models.mongodb import SearchParams

        params = SearchParams(
            query="Texto da busca",
            top_k=10,
            min_score=0.5,
            filters={},
            include_matches=True,
            matches_limit=3,
        )

        assert params.query == "Texto da busca"
        assert params.top_k == 10
        assert params.min_score == 0.5
        assert params.filters == {}
        assert params.include_matches == True
        assert params.matches_limit == 3


if __name__ == "__main__":
    pytest.main(["-v", __file__])
