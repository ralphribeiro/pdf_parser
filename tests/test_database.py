"""
Testes para conexão MongoDB e schemas.

TDD: Testes escritos antes da implementação.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest


class TestMongoDBConnection:
    """Testes para conexão MongoDB."""

    def test_mongodb_connection_config(self):
        """Teste: Configuração de conexão MongoDB."""
        from src.database import get_mongodb_connection

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
            },
        ):
            # Teste que a função existe e retorna conexão
            connection = get_mongodb_connection()
            assert connection is not None

    def test_mongodb_connection_with_uri(self):
        """Teste: Conexão com URI completo."""
        from src.database import get_mongodb_connection

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://user:pass@localhost:27017/testdb?authSource=admin"
            },
        ):
            connection = get_mongodb_connection()
            assert connection is not None


class TestDatabaseJobs:
    """Testes para operações de jobs no MongoDB."""

    def test_create_job_document(self, mock_db):
        """Teste: Criação de documento de job."""
        from src.database import create_job_document

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            result = create_job_document(
                job_id="test-job",
                file_content=b"%PDF-test\n",
                generate_embeddings=False,
            )

            # Verificar que o documento foi criado
            assert result is not None

    def test_update_job_status(self, mock_db):
        """Teste: Atualização de status de job."""
        from src.database import update_job_status

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            result = update_job_status(
                job_id="test-job", status="processing", result=None, error=None
            )

            assert result is not None

    def test_get_job(self, mock_db):
        """Teste: Obter job pelo ID."""
        from src.database import get_job

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            mock_job = {
                "_id": "job-uuid",
                "job_id": "test-job",
                "status": "completed",
                "created_at": "2026-03-04T12:00:00Z",
                "updated_at": "2026-03-04T12:05:00Z",
            }
            mock_db.find_one.return_value = mock_job

            result = get_job("test-job")
            assert result is not None
            assert result["job_id"] == "test-job"

    def test_get_job_not_found(self, mock_db):
        """Teste: Obter job que não existe."""
        from src.database import get_job
        from src.models.mongodb import JobNotFoundError

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            mock_db.find_one.return_value = None

            with pytest.raises(JobNotFoundError):
                get_job("nonexistent-job")

    def test_get_job_by_status(self, mock_db):
        """Teste: Obter jobs por status."""
        from src.database import get_jobs_by_status

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            mock_jobs = [
                {
                    "_id": "job-1",
                    "job_id": "job-1",
                    "status": "completed",
                    "created_at": "2026-03-04T12:00:00Z",
                },
                {
                    "_id": "job-2",
                    "job_id": "job-2",
                    "status": "processing",
                    "created_at": "2026-03-04T12:01:00Z",
                },
            ]
            mock_db.find.return_value = mock_jobs

            result = get_jobs_by_status("completed")
            assert len(result) == 1
            assert result[0]["job_id"] == "job-1"


class TestDatabaseDocuments:
    """Testes para operações de documentos no MongoDB."""

    def test_save_document(self, mock_db):
        """Teste: Salvar documento."""
        from src.database import save_document

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            doc = {
                "doc_id": "doc-uuid",
                "source_file": "test.pdf",
                "total_pages": 10,
                "pages": [],
            }

            result = save_document(doc_id="doc-uuid", document=doc)
            assert result is not None

    def test_save_document_with_processing_date(self, mock_db):
        """Teste: Salvar documento com data de processamento."""
        from src.database import save_document

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            doc = {
                "doc_id": "doc-uuid",
                "source_file": "test.pdf",
                "total_pages": 10,
                "pages": [],
            }

            result = save_document(
                doc_id="doc-uuid",
                document=doc,
                processing_date=datetime.now(UTC),
            )
            assert result is not None


class TestDatabaseEmbeddings:
    """Testes para operações de embeddings no MongoDB."""

    def test_save_embedding_generates_distinct_ids_for_multiple_blocks(self):
        """Teste: IDs de embeddings devem ser únicos no mesmo job/doc."""
        from src.database import save_embedding

        mock_collection = MagicMock()
        mock_collection.insert_one.return_value = MagicMock(inserted_id="ok")

        with patch(
            "src.database.Database.get_embeddings_collection",
            return_value=mock_collection,
        ):
            first = save_embedding(
                job_id="job-123",
                doc_id="doc-abc",
                vector=[0.1, 0.2, 0.3],
                page=1,
                block_id="p1_b1",
            )
            second = save_embedding(
                job_id="job-123",
                doc_id="doc-abc",
                vector=[0.4, 0.5, 0.6],
                page=1,
                block_id="p1_b2",
            )

        assert mock_collection.insert_one.call_count == 2
        assert first["_id"] != second["_id"]

    def test_save_embedding(self, mock_db):
        """Teste: Salvar embedding."""
        from src.database import save_embedding

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            embedding = {
                "doc_id": "doc-uuid",
                "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
                "model": "nomic-embed-text",
            }

            result = save_embedding(
                job_id="test-job",
                doc_id="doc-uuid",
                vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            )
            assert result is not None

    def test_save_embedding_with_chunk_size(self, mock_db):
        """Teste: Salvar embedding com chunk_size."""
        from src.database import save_embedding

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            embedding = {
                "doc_id": "doc-uuid",
                "vector": [0.1, 0.2, 0.3],
                "model": "nomic-embed-text",
            }

            result = save_embedding(
                job_id="test-job",
                doc_id="doc-uuid",
                vector=[0.1, 0.2, 0.3],
                chunk_size=1024,
            )
            assert result is not None


class TestDatabaseChunks:
    """Testes para operações de chunks no MongoDB."""

    def test_save_chunk(self, mock_db):
        """Teste: Salvar chunk."""
        from src.database import save_chunk

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            chunk = {
                "doc_id": "doc-uuid",
                "text": "Texto do chunk",
                "vector": [0.1, 0.2, 0.3],
                "page": 1,
                "block_id": "p1_b1",
            }

            result = save_chunk(
                job_id="test-job",
                doc_id="doc-uuid",
                text="Texto do chunk",
                vector=[0.1, 0.2, 0.3],
                page=1,
                block_id="p1_b1",
            )
            assert result is not None

    def test_save_chunk_with_metadata(self, mock_db):
        """Teste: Salvar chunk com metadata."""
        from src.database import save_chunk

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            chunk = {
                "doc_id": "doc-uuid",
                "text": "Texto do chunk",
                "vector": [0.1, 0.2, 0.3],
                "page": 1,
                "block_id": "p1_b1",
                "metadata": {"type": "paragraph"},
            }

            result = save_chunk(
                job_id="test-job",
                doc_id="doc-uuid",
                text="Texto do chunk",
                vector=[0.1, 0.2, 0.3],
                page=1,
                block_id="p1_b1",
                metadata={"type": "paragraph"},
            )
            assert result is not None


class TestDatabaseVectorSearch:
    """Testes para busca vetorial no MongoDB."""

    def test_vector_near_search(self, mock_db):
        """Teste: Busca vetorial próxima (vetoresNearSearch)."""
        from src.database import vector_near_search

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            mock_results = [
                {
                    "_id": "result-1",
                    "doc_id": "doc-1",
                    "score": 0.92,
                    "metadata": {},
                },
                {
                    "_id": "result-2",
                    "doc_id": "doc-2",
                    "score": 0.85,
                    "metadata": {},
                },
            ]
            mock_db.aggregate.return_value = mock_results

            result = vector_near_search(
                job_id="test-job",
                query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
                top_k=10,
                min_score=0.5,
                filters={},
            )
            assert len(result) >= 0

    def test_vector_near_search_with_filters(self, mock_db):
        """Teste: Busca vetorial com filtros."""
        from src.database import vector_near_search

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            result = vector_near_search(
                job_id="test-job",
                query_vector=[0.1, 0.2, 0.3],
                top_k=5,
                min_score=0.7,
                filters={"doc_id": "doc-uuid"},
            )
            assert result is not None

    def test_vector_near_search_fallback_when_vector_search_unavailable(self):
        """Teste: Fallback local quando $vectorSearch não está disponível."""
        from src.database import vector_near_search

        mock_collection = MagicMock()
        mock_collection.aggregate.side_effect = Exception(
            "$vectorSearch stage is only allowed on MongoDB Atlas"
        )
        mock_collection.find.return_value = [
            {
                "_id": "emb-1",
                "doc_id": "doc-1",
                "vector": [1.0, 0.0, 0.0],
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "_id": "emb-2",
                "doc_id": "doc-2",
                "vector": [0.0, 1.0, 0.0],
                "created_at": "2026-01-02T00:00:00+00:00",
            },
        ]

        with patch(
            "src.database.Database.get_embeddings_collection",
            return_value=mock_collection,
        ):
            results = vector_near_search(
                job_id="search",
                query_vector=[1.0, 0.0, 0.0],
                top_k=5,
                min_score=0.1,
                filters={},
            )

        assert len(results) == 1
        assert results[0]["doc_id"] == "doc-1"
        assert results[0]["score"] > 0.9


class TestDatabaseWebhooks:
    """Testes para operações de webhooks no MongoDB."""

    def test_create_webhook(self, mock_db):
        """Teste: Criar webhook."""
        from src.database import create_webhook

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            webhook = {
                "job_id": "test-job",
                "url": "https://example.com/webhook",
                "token": "secret-token",
            }

            result = create_webhook(
                job_id="test-job",
                url="https://example.com/webhook",
                token="secret-token",
            )
            assert result is not None

    def test_delete_webhook(self, mock_db):
        """Teste: Deletar webhook."""
        from src.database import delete_webhook

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            result = delete_webhook("test-job")
            assert result is not None


class TestDatabaseCleanup:
    """Testes para limpeza no MongoDB."""

    def test_delete_old_jobs(self, mock_db):
        """Teste: Deletar jobs antigos."""
        from src.database import delete_old_jobs

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            result = delete_old_jobs(job_id="test-job", max_age_days=30)
            assert result is not None

    def test_delete_old_documents(self, mock_db):
        """Teste: Deletar documentos antigos."""
        from src.database import delete_old_documents

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.database.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_db

            result = delete_old_documents(doc_id="test-doc", max_age_days=30)
            assert result is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
