"""
Remote embedding client using an OpenAI-compatible API.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class EmbeddingClient:
    """HTTP client for embedding generation."""

    base_url: str
    model: str
    timeout_seconds: float = 10.0
    http_client: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        if self.http_client is not None:
            return self.http_client
        return httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        client = self._client()
        owns_client = self.http_client is None
        try:
            response = client.post(
                "/v1/embeddings",
                json={"input": texts, "model": self.model},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        finally:
            if owns_client:
                client.close()

        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Malformed embedding response: missing 'data' list")

        vectors: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list):
                raise RuntimeError("Malformed embedding response: missing 'embedding'")
            vectors.append([float(v) for v in embedding])

        if len(vectors) != len(texts):
            raise RuntimeError("Malformed embedding response: vector count mismatch")
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Generate one embedding vector for a query string."""
        vectors = self.embed_texts([query])
        return vectors[0]

    def healthcheck(self) -> bool:
        """Probe remote embedding service health endpoint."""
        client = self._client()
        owns_client = self.http_client is None
        try:
            response = client.get("/health")
            return response.status_code < 400
        except Exception:
            return False
        finally:
            if owns_client:
                client.close()
