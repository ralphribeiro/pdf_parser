"""
Tests for local sentence-transformers embedding client.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class FakeSentenceTransformer:
    def __init__(self):
        self.max_seq_length = 3
        self.batches: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.batches.append(list(texts))
        assert kwargs["convert_to_numpy"] is False
        assert kwargs["normalize_embeddings"] is True
        return [[float(len(text)), 1.0] for text in texts]


class TestLocalEmbeddingClient:
    def setup_method(self):
        from services.search.local_embedding_client import LocalEmbeddingClient

        LocalEmbeddingClient.clear_model_cache()

    def test_embed_texts_loads_model_lazily_and_batches(self, monkeypatch):
        from services.search import local_embedding_client
        from services.search.local_embedding_client import LocalEmbeddingClient

        loaded = []
        fake_model = FakeSentenceTransformer()

        def fake_loader(model_name, *, device, trust_remote_code):
            loaded.append(
                {
                    "model_name": model_name,
                    "device": device,
                    "trust_remote_code": trust_remote_code,
                }
            )
            return fake_model

        monkeypatch.setattr(local_embedding_client, "_load_model", fake_loader)

        client = LocalEmbeddingClient(
            model="sentence-transformers/test-model",
            device="cpu",
            trust_remote_code=True,
            batch_size=2,
            max_chars_per_text=50,
        )

        vectors = client.embed_texts(["abcdef", "xy", "123456"])

        assert loaded == [
            {
                "model_name": "sentence-transformers/test-model",
                "device": "cpu",
                "trust_remote_code": True,
            }
        ]
        assert fake_model.batches == [["abcdef", "xy"], ["123456"]]
        assert vectors == [[6.0, 1.0], [2.0, 1.0], [6.0, 1.0]]

    def test_embed_query_returns_first_vector(self, monkeypatch):
        from services.search import local_embedding_client
        from services.search.local_embedding_client import LocalEmbeddingClient

        monkeypatch.setattr(
            local_embedding_client,
            "_load_model",
            lambda *_args, **_kwargs: FakeSentenceTransformer(),
        )

        client = LocalEmbeddingClient(model="local/path", device="cpu")

        assert client.embed_query("consulta") == [8.0, 1.0]

    def test_requires_model_name(self):
        from services.search.local_embedding_client import LocalEmbeddingClient

        with pytest.raises(ValueError, match="EMBEDDING_MODEL is required"):
            LocalEmbeddingClient(model="")

    def test_reuses_cached_model_for_same_settings(self, monkeypatch):
        from services.search import local_embedding_client
        from services.search.local_embedding_client import LocalEmbeddingClient

        load_count = 0

        def fake_loader(*_args, **_kwargs):
            nonlocal load_count
            load_count += 1
            return FakeSentenceTransformer()

        monkeypatch.setattr(local_embedding_client, "_load_model", fake_loader)

        first = LocalEmbeddingClient(model="same-model", device="cpu")
        second = LocalEmbeddingClient(model="same-model", device="cpu")

        first.embed_query("a")
        second.embed_query("b")

        assert load_count == 1
