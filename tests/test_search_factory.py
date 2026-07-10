"""
Tests for semantic search factory backend selection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class FakeVectorStore:
    pass


def _patch_common_config(monkeypatch, *, backend, api_url="", model="model"):
    import config

    monkeypatch.setattr(config, "CHROMA_HOST", "http://chromadb:8000")
    monkeypatch.setattr(config, "CHROMA_COLLECTION", "test_collection")
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", backend)
    monkeypatch.setattr(config, "EMBEDDING_API_URL", api_url)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", model)
    monkeypatch.setattr(config, "EMBEDDING_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 8)
    monkeypatch.setattr(config, "EMBEDDING_API_KEY", "")
    monkeypatch.setattr(config, "EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(config, "EMBEDDING_TRUST_REMOTE_CODE", False)
    monkeypatch.setattr(config, "EMBEDDING_MAX_CHARS", 100)


class TestCreateSemanticSearchService:
    def test_local_backend_does_not_require_api_url(self, monkeypatch):
        from services.search import factory

        _patch_common_config(
            monkeypatch,
            backend="local",
            api_url="",
            model="sentence-transformers/test-model",
        )

        created = {}

        class FakeLocalClient:
            def __init__(self, **kwargs):
                created["local"] = kwargs

        monkeypatch.setattr(factory, "LocalEmbeddingClient", FakeLocalClient)
        monkeypatch.setattr(
            factory.ChromaVectorStore,
            "from_host",
            lambda **_kwargs: FakeVectorStore(),
        )

        service = factory.create_semantic_search_service()

        assert service is not None
        assert isinstance(service.embedding_client, FakeLocalClient)
        assert created["local"] == {
            "model": "sentence-transformers/test-model",
            "device": "cpu",
            "trust_remote_code": False,
            "batch_size": 8,
            "max_chars_per_text": 100,
        }

    def test_api_backend_keeps_remote_client_requirements(self, monkeypatch):
        from services.search import factory

        _patch_common_config(
            monkeypatch,
            backend="api",
            api_url="http://embed.local",
            model="Qwen3-Embedding",
        )

        created = {}

        class FakeRemoteClient:
            def __init__(self, **kwargs):
                created["remote"] = kwargs

        monkeypatch.setattr(factory, "EmbeddingClient", FakeRemoteClient)
        monkeypatch.setattr(
            factory.ChromaVectorStore,
            "from_host",
            lambda **_kwargs: FakeVectorStore(),
        )

        service = factory.create_semantic_search_service()

        assert service is not None
        assert isinstance(service.embedding_client, FakeRemoteClient)
        assert created["remote"]["base_url"] == "http://embed.local"
        assert created["remote"]["model"] == "Qwen3-Embedding"

    def test_api_backend_without_url_disables_search(self, monkeypatch):
        from services.search import factory

        _patch_common_config(monkeypatch, backend="api", api_url="")

        assert factory.create_semantic_search_service() is None

    def test_local_backend_without_model_disables_search(self, monkeypatch):
        from services.search import factory

        _patch_common_config(monkeypatch, backend="local", api_url="", model="")

        assert factory.create_semantic_search_service() is None
