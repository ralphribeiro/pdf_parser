"""
Remote embedding client using an OpenAI-compatible API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_MAX_CHARS = 6000  # ~3000-4000 tokens, safely under 8192-token context


@dataclass
class EmbeddingClient:
    """HTTP client for embedding generation."""

    base_url: str
    model: str
    timeout_seconds: float = 120.0
    batch_size: int = 32
    max_chars_per_text: int = _DEFAULT_MAX_CHARS
    http_client: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        if self.http_client is not None:
            return self.http_client
        return httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds)

    def _embed_batch(self, client: httpx.Client, texts: list[str]) -> list[list[float]]:
        """Send a single batch to the embedding API and parse the response."""
        response = client.post(
            "/v1/embeddings",
            json={"input": texts, "model": self.model},
        )
        if response.status_code >= 400:
            logger.error(
                "Embedding API error %d (batch=%d texts): %s",
                response.status_code,
                len(texts),
                response.text[:500],
            )
            response.raise_for_status()
        payload = response.json()

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

    def _truncate(self, texts: list[str]) -> list[str]:
        limit = self.max_chars_per_text
        return [t[:limit] if len(t) > limit else t for t in texts]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, batching automatically."""
        texts = self._truncate(texts)
        client = self._client()
        owns_client = self.http_client is None
        try:
            all_vectors: list[list[float]] = []
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                logger.info(
                    "Embedding batch %d/%d (%d texts)",
                    start // self.batch_size + 1,
                    -(-len(texts) // self.batch_size),
                    len(batch),
                )
                all_vectors.extend(self._embed_batch(client, batch))
            return all_vectors
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        finally:
            if owns_client:
                client.close()

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
