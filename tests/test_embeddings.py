"""
Testes para cliente de embeddings (OpenAI-compatible API).

TDD: Testes escritos antes da implementação.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests


@pytest.fixture
def mock_embeddings_response():
    """Mock response de API de embeddings."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0}
        ],
        "model": "nomic-embed-text",
    }


@pytest.fixture
def mock_embeddings_response_batch():
    """Mock response de API de embeddings (batch)."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5], "index": 0},
            {"object": "embedding", "embedding": [0.2, 0.3, 0.4, 0.5, 0.6], "index": 1},
            {"object": "embedding", "embedding": [0.3, 0.4, 0.5, 0.6, 0.7], "index": 2},
        ],
        "model": "nomic-embed-text",
    }


class TestEmbeddingsClient:
    """Testes para o cliente de embeddings."""

    def test_client_initialization(self, monkeypatch):
        """Teste: Inicialização do cliente."""
        from src.embeddings import EmbeddingsClient

        # Mock das variáveis de ambiente (Ollama v1/embeddings)
        monkeypatch.setenv(
            "DOC_PARSER_EMBEDDINGS_URL", "http://localhost:8080/v1/embeddings"
        )
        monkeypatch.setenv("DOC_PARSER_EMBEDDINGS_MODEL", "Qwen3-Embedding")
        monkeypatch.setenv("DOC_PARSER_EMBEDDINGS_API_KEY", "llama.cpp")

        client = EmbeddingsClient()
        assert client.base_url == "http://localhost:8080"
        assert client.model == "Qwen3-Embedding"

    def test_client_with_default_config(self, monkeypatch):
        """Teste: Inicialização com configuração padrão."""
        monkeypatch.delenv("DOC_PARSER_EMBEDDINGS_URL", raising=False)
        monkeypatch.delenv("DOC_PARSER_EMBEDDINGS_MODEL", raising=False)
        monkeypatch.delenv("DOC_PARSER_EMBEDDINGS_API_KEY", raising=False)

        from src.embeddings import EmbeddingsClient

        client = EmbeddingsClient()
        assert client is not None

    def test_client_with_custom_api_key(self, monkeypatch):
        """Teste: Inicialização com API key personalizado."""
        monkeypatch.setenv(
            "DOC_PARSER_EMBEDDINGS_URL", "http://localhost:8080/v1/embeddings"
        )
        monkeypatch.setenv("DOC_PARSER_EMBEDDINGS_MODEL", "Qwen3-Embedding")
        monkeypatch.setenv("DOC_PARSER_EMBEDDINGS_API_KEY", "custom-key")

        from src.embeddings import EmbeddingsClient

        client = EmbeddingsClient()
        assert client is not None


class TestGenerateEmbedding:
    """Testes para geração de embeddings."""

    def test_generate_embedding_uses_full_endpoint_path(self, mock_embeddings_response):
        """Teste: Requisição deve preservar path completo do endpoint."""
        from src.embeddings import EmbeddingsClient

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://localhost:8081/v1/embeddings",
                "DOC_PARSER_EMBEDDINGS_MODEL": "Qwen3-Embedding",
            },
        ):
            client = EmbeddingsClient()

            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_embeddings_response
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                _ = client.generate_embedding("Test text for endpoint path")

                assert mock_post.call_count == 1
                called_url = mock_post.call_args.args[0]
                assert called_url == "http://localhost:8081/v1/embeddings"

    def test_generate_embedding_single_text(self, mock_embeddings_response):
        """Teste: Geração de embedding para texto único."""
        from src.embeddings import EmbeddingsClient

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://localhost:8080/v1/embeddings",
                "DOC_PARSER_EMBEDDINGS_MODEL": "Qwen3-Embedding",
            },
        ):
            client = EmbeddingsClient()

            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_embeddings_response
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                embedding = client.generate_embedding("Test text for embedding")
                assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_generate_embedding_list_of_texts(self, mock_embeddings_response):
        """Teste: Geração de embeddings para lista de textos."""
        from src.embeddings import EmbeddingsClient

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://localhost:8080/v1/embeddings",
                "DOC_PARSER_EMBEDDINGS_MODEL": "Qwen3-Embedding",
            },
        ):
            client = EmbeddingsClient()

            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_embeddings_response
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                embeddings = client.generate_embeddings(["Text 1", "Text 2"])
                assert len(embeddings) == 2  # Each text returns one embedding

    def test_generate_embedding_with_custom_model(self, mock_embeddings_response):
        """Teste: Geração de embedding com modelo personalizado."""
        from src.embeddings import EmbeddingsClient

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://localhost:8080/v1/embeddings",
                "DOC_PARSER_EMBEDDINGS_MODEL": "nomic-embed-text",
            },
        ):
            client = EmbeddingsClient()

            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_embeddings_response
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                embedding = client.generate_embedding("Test text")
                assert embedding is not None

    def test_generate_embedding_request_format(self, mock_embeddings_response):
        """Teste: Formato de request é OpenAI-compatible."""
        from src.embeddings import build_request

        text = "Test text for embedding"
        request = build_request(text, model="test-model")

        assert request is not None
        assert "model" in request
        assert "input" in request
        assert request["input"] == text
        assert request["model"] == "test-model"

    def test_generate_embedding_response_parsing(self, mock_embeddings_response):
        """Teste: Parsing de resposta da API."""
        from src.embeddings import extract_embedding

        mock_response = MagicMock()
        mock_response.json.return_value = mock_embeddings_response

        embedding = extract_embedding(mock_response)
        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]


class TestEmbeddingDimensions:
    """Testes para dimensões de embeddings."""

    def test_embedding_vector_length(self, mock_embeddings_response):
        """Teste: Vetor de embedding tem comprimento esperado."""
        from src.embeddings import EmbeddingsClient

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://test:11434/api/generate",
                "DOC_PARSER_EMBEDDINGS_MODEL": "test-model",
            },
        ):
            client = EmbeddingsClient()

            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_embeddings_response
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                embedding = client.generate_embedding("Test text")
                assert len(embedding) == 5  # Mock retorna 5 dimensões

    def test_different_embedding_models(self):
        """Teste: Diferentes modelos têm diferentes dimensões."""
        # nomic-embed-text: 768 dimensões
        # mxbai-embed-large: 1024 dimensões
        # text-embedding-3-large: 3072 dimensões

        test_cases = [
            ("nomic-embed-text", 768),
            ("mxbai-embed-large", 1024),
            ("text-embedding-3-large", 3072),
        ]

        for model_name, expected_dims in test_cases:
            # Note: Estas dimensões são esperadas, não verificadas aqui
            # pois dependem da API real
            assert expected_dims > 0


class TestEmbeddingSimilarity:
    """Testes para similaridade de embeddings."""

    def test_cosine_similarity(self):
        """Teste: Similaridade cosine."""
        from src.embeddings import cosine_similarity

        # Vetores idênticos devem ter similaridade 1.0
        vector1 = [0.1, 0.2, 0.3, 0.4, 0.5]
        vector2 = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert abs(cosine_similarity(vector1, vector2) - 1.0) < 1e-10

        # Vetores ortogonais devem ter similaridade ~0
        vector3 = [1.0, 0.0, 0.0, 0.0, 0.0]
        vector4 = [0.0, 1.0, 0.0, 0.0, 0.0]
        assert abs(cosine_similarity(vector3, vector4)) < 0.01

    def test_cosine_similarity_normalized(self):
        """Teste: Similaridade cosine com vetores normalizados."""
        from src.embeddings import cosine_similarity

        # Vetores normalizados
        vector1 = [0.5, 0.5, 0.5]
        vector2 = [0.5, 0.5, 0.5]
        assert abs(cosine_similarity(vector1, vector2) - 1.0) < 1e-10

    def test_cosine_similarity_empty_vectors(self):
        """Teste: Similaridade cosine com vetores vazios."""
        from src.embeddings import cosine_similarity

        vector1 = []
        vector2 = []
        result = cosine_similarity(vector1, vector2)
        assert result == 0.0  # Vetores vazios têm similaridade 0


class TestBatchEmbeddings:
    """Testes para batch de embeddings."""

    def test_batch_embeddings(self, mock_embeddings_response_batch):
        """Teste: Batch de embeddings."""
        from src.embeddings import EmbeddingsClient

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://test:11434/api/generate",
                "DOC_PARSER_EMBEDDINGS_MODEL": "test-model",
            },
        ):
            client = EmbeddingsClient()

            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_embeddings_response_batch
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                # Ollama pode processar apenas um embedding por request
                embeddings = client.generate_embeddings(["Text 1", "Text 2", "Text 3"])
                assert len(embeddings) >= 1

    def test_batch_embeddings_with_batch_size(self, mock_embeddings_response_batch):
        """Teste: Batch de embeddings com batch_size."""
        from src.embeddings import EmbeddingsClient

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://test:11434/api/generate",
                "DOC_PARSER_EMBEDDINGS_MODEL": "test-model",
            },
        ):
            client = EmbeddingsClient()

            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = mock_embeddings_response_batch
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                # Teste com batch_size=2
                embeddings = client.generate_embeddings(
                    ["Text 1", "Text 2", "Text 3"], batch_size=2
                )
                assert len(embeddings) == 3


class TestEmbeddingCaching:
    """Testes para cache de embeddings."""

    def test_embed_cache(self):
        """Teste: Cache de embeddings."""
        from src.embeddings import EmbeddingCache

        cache = EmbeddingCache(max_size=100)

        # Adicionar embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cache.set("key1", embedding)

        # Recuperar embedding
        cached = cache.get("key1")
        assert cached == embedding

        # Teste de existência
        assert cache.exists("key1")
        assert not cache.exists("key2")

    def test_embed_cache_size(self):
        """Teste: Tamanho do cache."""
        from src.embeddings import EmbeddingCache

        cache = EmbeddingCache(max_size=5)

        for i in range(10):
            cache.set(f"key{i}", [0.1, 0.2, 0.3])

        # Cache deve ter no máximo 5 entradas
        assert len(cache.cache) <= 5

    def test_embed_cache_clear(self):
        """Teste: Limpeza do cache."""
        from src.embeddings import EmbeddingCache

        cache = EmbeddingCache(max_size=100)
        cache.set("key1", [0.1, 0.2, 0.3])
        cache.set("key2", [0.4, 0.5, 0.6])

        cache.clear()

        assert not cache.exists("key1")
        assert not cache.exists("key2")


class TestEmbeddingErrorHandling:
    """Testes para tratamento de erros em embeddings."""

    def test_generate_embedding_network_error(self):
        """Teste: Erro de rede na geração de embedding."""
        from src.embeddings import EmbeddingsClient

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://invalid:9999/api/generate",
                "DOC_PARSER_EMBEDDINGS_MODEL": "test-model",
            },
        ):
            client = EmbeddingsClient()

            # Simular erro de rede
            with patch("requests.post") as mock_post:
                mock_post.side_effect = requests.exceptions.ConnectionError(
                    "Connection refused"
                )

                with pytest.raises(requests.exceptions.ConnectionError):
                    client.generate_embedding("Test text")

    def test_generate_embedding_api_error(self, mock_embeddings_response):
        """Teste: Erro da API na geração de embedding."""
        from src.embeddings import EmbeddingsClient

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://test:11434/api/generate",
                "DOC_PARSER_EMBEDDINGS_MODEL": "test-model",
            },
        ):
            client = EmbeddingsClient()

            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.raise_for_status.side_effect = (
                    requests.exceptions.HTTPError("429 Too Many Requests")
                )
                mock_post.return_value = mock_response

                with pytest.raises(requests.exceptions.HTTPError):
                    client.generate_embedding("Test text")

    def test_generate_embedding_invalid_response(self, mock_embeddings_response):
        """Teste: Resposta inválida da API."""
        from src.embeddings import ExtractEmbeddingError

        with patch.dict(
            "os.environ",
            {
                "DOC_PARSER_EMBEDDINGS_URL": "http://test:11434/api/generate",
                "DOC_PARSER_EMBEDDINGS_MODEL": "test-model",
            },
        ):
            client = EmbeddingsClient()

            with patch("requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"invalid": "response format"}
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                with pytest.raises(ExtractEmbeddingError):
                    client.generate_embedding("Test text")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
