"""
Tests for remote embedding client (OpenAI-compatible endpoint).
"""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _build_client(handler):
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://embed.local")
    from services.search.embedding_client import EmbeddingClient

    return EmbeddingClient(
        base_url="http://embed.local",
        model="Qwen3-Embedding",
        timeout_seconds=2.0,
        http_client=http,
    )


class TestEmbeddingClient:
    def test_embed_texts_sends_openai_payload(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["json"] = request.read().decode()
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2]},
                        {"index": 1, "embedding": [0.3, 0.4]},
                    ]
                },
            )

        client = _build_client(handler)
        embeddings = client.embed_texts(["a", "b"])

        assert captured["url"].endswith("/v1/embeddings")
        assert '"model":"Qwen3-Embedding"' in captured["json"]
        assert '"input":["a","b"]' in captured["json"]
        assert embeddings == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_query_returns_first_vector(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.9, 0.8, 0.7]}]},
            )

        client = _build_client(handler)
        vec = client.embed_query("consulta")
        assert vec == [0.9, 0.8, 0.7]

    def test_raises_on_http_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal"})

        client = _build_client(handler)
        with pytest.raises(RuntimeError, match="Embedding request failed"):
            client.embed_texts(["x"])

    def test_raises_on_malformed_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"index": 0}]})

        client = _build_client(handler)
        with pytest.raises(RuntimeError, match="Malformed embedding response"):
            client.embed_texts(["x"])

    def test_batching_splits_large_input(self):
        """When texts exceed batch_size, multiple API calls are made."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            import json

            body = json.loads(request.read())
            n = len(body["input"])
            data = [{"index": i, "embedding": [float(i)]} for i in range(n)]
            return httpx.Response(200, json={"data": data})

        transport = httpx.MockTransport(handler)
        http = httpx.Client(transport=transport, base_url="http://embed.local")
        from services.search.embedding_client import EmbeddingClient

        client = EmbeddingClient(
            base_url="http://embed.local",
            model="test",
            batch_size=3,
            http_client=http,
        )

        texts = [f"text_{i}" for i in range(7)]
        vectors = client.embed_texts(texts)

        assert len(vectors) == 7
        assert call_count == 3  # ceil(7/3) = 3 batches

    def test_healthcheck_true_on_200(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(404)

        client = _build_client(handler)
        assert client.healthcheck() is True

    def test_healthcheck_false_on_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        client = _build_client(handler)
        assert client.healthcheck() is False
