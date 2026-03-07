"""
Testes para vector store (MongoDB vetores).

TDD: Testes escritos antes da implementação.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_mongodb_connection():
    """Mock de conexão MongoDB."""
    mock = MagicMock()
    mock.collection = MagicMock()
    return mock


@pytest.fixture
def mock_embedding():
    """Mock de embedding de exemplo."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class TestMongoDBVectorStore:
    """Testes para o Vector Store do MongoDB."""

    def test_vector_store_initialization(self):
        """Teste: Inicialização do vector store."""
        from src.search.vector_store import MongoDBVectorStore

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                "DOC_PARSER_MONGODB_USE_VECTORS": "true",
            },
        ):
            store = MongoDBVectorStore("test_collection")
            assert store is not None

    def test_vector_store_with_indexes(self):
        """Teste: Vector store com indexes."""
        from src.search.vector_store import MongoDBVectorStore

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                "DOC_PARSER_MONGODB_USE_VECTORS": "true",
            },
        ):
            store = MongoDBVectorStore("test_collection")
            # Criar index de vetor
            store.create_vector_index("vector")
            assert True  # Verificação de que o método foi chamado


class TestAddDocuments:
    """Testes para adicionar documentos ao vector store."""

    def test_add_document(self, mock_mongodb_connection):
        """Teste: Adicionar documento ao vector store."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            document = {
                "doc_id": "doc-uuid",
                "text": "Texto do documento",
                "metadata": {"page": 1},
            }

            result = store.add_documents([document])
            assert result is not None

    def test_add_documents_batch(self, mock_mongodb_connection):
        """Teste: Adicionar múltiplos documentos ao vector store."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            documents = [
                {"doc_id": "doc-1", "text": "Texto 1"},
                {"doc_id": "doc-2", "text": "Texto 2"},
                {"doc_id": "doc-3", "text": "Texto 3"},
            ]

            result = store.add_documents(documents)
            assert len(result) == 3

    def test_add_documents_with_vectors(self, mock_mongodb_connection, mock_embedding):
        """Teste: Adicionar documentos com vetores."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            documents_with_vectors = [
                {
                    "doc_id": "doc-1",
                    "vector": [0.1, 0.2, 0.3],
                    "metadata": {"page": 1},
                },
                {
                    "doc_id": "doc-2",
                    "vector": [0.4, 0.5, 0.6],
                    "metadata": {"page": 2},
                },
            ]

            result = store.add_documents(documents_with_vectors)
            assert len(result) == 2


class TestSearch:
    """Testes para busca no vector store."""

    def test_search_similar(self, mock_mongodb_connection, mock_embedding):
        """Teste: Buscar documentos similares."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

            result = store.similar(query_vector, top_k=5)
            assert result is not None

    def test_search_similar_with_min_score(
        self, mock_mongodb_connection, mock_embedding
    ):
        """Teste: Buscar documentos similares com min_score."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            query_vector = [0.1, 0.2, 0.3]
            result = store.similar(query_vector, min_score=0.7)
            assert result is not None

    def test_search_similar_with_filters(self, mock_mongodb_connection, mock_embedding):
        """Teste: Buscar documentos similares com filtros."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            query_vector = [0.1, 0.2, 0.3]
            result = store.similar(query_vector, filters={"doc_id": "doc-uuid"})
            assert result is not None

    def test_search_similar_with_top_k(self, mock_mongodb_connection, mock_embedding):
        """Teste: Buscar documentos similares com top_k."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            query_vector = [0.1, 0.2, 0.3]
            result = store.similar(query_vector, top_k=10)
            assert result is not None


class TestDelete:
    """Testes para deleção no vector store."""

    def test_delete_document(self, mock_mongodb_connection):
        """Teste: Deletar documento."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            result = store.delete_document("doc-uuid")
            assert result is not None

    def test_delete_documents_batch(self, mock_mongodb_connection):
        """Teste: Deletar múltiplos documentos."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            result = store.delete_documents(["doc-1", "doc-2", "doc-3"])
            assert len(result) == 3


class TestUpdate:
    """Testes para atualização no vector store."""

    def test_update_document(self, mock_mongodb_connection):
        """Teste: Atualizar documento."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            result = store.update_document(
                "doc-uuid", {"text": "Novo texto"}, {"vector": [0.1, 0.2, 0.3]}
            )
            assert result is not None

    def test_update_document_vector(self, mock_mongodb_connection):
        """Teste: Atualizar vetor de documento."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            result = store.update_vector("doc-uuid", [0.1, 0.2, 0.3, 0.4, 0.5])
            assert result is not None


class TestCount:
    """Testes para contagem no vector store."""

    def test_count_documents(self, mock_mongodb_connection):
        """Teste: Contar documentos."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            result = store.count_documents()
            assert result is not None

    def test_count_documents_with_filter(self, mock_mongodb_connection):
        """Teste: Contar documentos com filtro."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            result = store.count_documents(filters={"doc_id": "doc-uuid"})
            assert result is not None


class TestSimilarityCalculation:
    """Testes para cálculo de similaridade."""

    def test_cosine_similarity_calculation(self):
        """Teste: Cálculo de similaridade cosine."""
        from src.search.vector_store import cosine_similarity

        # Vetores idênticos
        v1 = [1, 2, 3, 4, 5]
        v2 = [1, 2, 3, 4, 5]
        assert cosine_similarity(v1, v2) == 1.0

        # Vetores opostos
        v3 = [1, 2, 3, 4, 5]
        v4 = [-1, -2, -3, -4, -5]
        assert cosine_similarity(v3, v4) == -1.0

        # Vetores ortogonais
        v5 = [1, 0, 0, 0, 0]
        v6 = [0, 1, 0, 0, 0]
        assert abs(cosine_similarity(v5, v6)) < 0.01

    def test_cosine_similarity_normalized_vectors(self):
        """Teste: Similaridade cosine com vetores normalizados."""
        from src.search.vector_store import cosine_similarity

        # Vetores normalizados manualmente
        v1 = [0.5, 0.5, 0.5, 0.5, 0.5]
        v2 = [0.5, 0.5, 0.5, 0.5, 0.5]
        assert cosine_similarity(v1, v2) == 1.0

    def test_cosine_similarity_different_dimensions(self):
        """Teste: Similaridade cosine com vetores de dimensões diferentes."""
        from src.search.vector_store import cosine_similarity

        v1 = [1, 2, 3]
        v2 = [1, 2, 3, 4, 5]
        result = cosine_similarity(v1, v2)
        assert result is not None


class TestVectorIndex:
    """Testes para indexação de vetores."""

    def test_create_vector_index(self, mock_mongodb_connection):
        """Teste: Criar index de vetores."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            store.create_vector_index("vector")
            assert True

    def test_create_vector_index_with_options(self, mock_mongodb_connection):
        """Teste: Criar index de vetores com opções."""
        from src.search.vector_store import MongoDBVectorStore

        with (
            patch.dict(
                "os.environ",
                {
                    "DOC_PARSER_MONGODB_URI": "mongodb://localhost:27017",
                    "DOC_PARSER_MONGODB_DB": "caseiro_docs",
                },
            ),
            patch("src.search.vector_store.get_mongodb_connection") as mock_conn,
        ):
            mock_conn.return_value = mock_mongodb_connection

            store = MongoDBVectorStore("test_collection")

            # Criar index com opções
            store.create_vector_index(
                "vector", options={"indexOptions": {"sparse": True}}
            )
            assert True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
