"""
Local embedding client backed by sentence-transformers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 6000


def _resolve_device(device: str) -> str:
    normalized = (device or "auto").lower()
    if normalized == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("EMBEDDING_DEVICE must be one of: auto, cpu, cuda")
    return normalized


def _load_model(model_name: str, *, device: str, trust_remote_code: bool):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )


def _as_vectors(encoded) -> list[list[float]]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [[float(value) for value in vector] for vector in encoded]


@dataclass
class LocalEmbeddingClient:
    """Embedding client that runs a Hugging Face model in-process."""

    model: str
    device: str = "auto"
    trust_remote_code: bool = False
    batch_size: int = 32
    max_chars_per_text: int = _DEFAULT_MAX_CHARS
    _resolved_device: str = field(init=False, repr=False)

    _MODEL_CACHE: ClassVar[dict[tuple[str, str, bool], object]] = {}

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("EMBEDDING_MODEL is required when EMBEDDING_BACKEND=local")
        self._resolved_device = _resolve_device(self.device)

    @classmethod
    def clear_model_cache(cls) -> None:
        """Clear cached models for tests and controlled reloads."""
        cls._MODEL_CACHE.clear()

    def _model(self):
        cache_key = (self.model, self._resolved_device, self.trust_remote_code)
        loaded = self._MODEL_CACHE.get(cache_key)
        if loaded is None:
            logger.info(
                "Loading local embedding model '%s' on %s",
                self.model,
                self._resolved_device,
            )
            loaded = _load_model(
                self.model,
                device=self._resolved_device,
                trust_remote_code=self.trust_remote_code,
            )
            self._MODEL_CACHE[cache_key] = loaded
        return loaded

    def _effective_max_chars(self, model) -> int:
        model_limit = getattr(model, "max_seq_length", None)
        if isinstance(model_limit, int) and model_limit > 0:
            return min(self.max_chars_per_text, model_limit * 4)
        return self.max_chars_per_text

    @staticmethod
    def _truncate(texts: list[str], limit: int) -> list[str]:
        return [text[:limit] if len(text) > limit else text for text in texts]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, batching automatically."""
        if not texts:
            return []

        model = self._model()
        texts = self._truncate(texts, self._effective_max_chars(model))

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = model.encode(
                batch,
                convert_to_numpy=False,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors.extend(_as_vectors(encoded))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Generate one embedding vector for a query string."""
        return self.embed_texts([query])[0]

    def healthcheck(self) -> bool:
        """Return true when the configured model can be loaded."""
        try:
            self._model()
            return True
        except Exception:
            logger.exception("Local embedding model healthcheck failed")
            return False
