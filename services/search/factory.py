"""
Factory for semantic search service wiring.
"""

from __future__ import annotations

import logging
from typing import cast

from services.search.chroma_store import ChromaVectorStore
from services.search.embedding_client import EmbeddingClient
from services.search.local_embedding_client import LocalEmbeddingClient
from services.search.service import SemanticSearchService

logger = logging.getLogger(__name__)


def create_semantic_search_service() -> SemanticSearchService | None:
    """Build semantic search service if all required config exists."""
    import config

    if not config.CHROMA_HOST:
        return None

    backend = cast(str, config.EMBEDDING_BACKEND).lower()
    if backend == "api":
        if not config.EMBEDDING_API_URL:
            logger.warning("Semantic search disabled: EMBEDDING_API_URL is required")
            return None
        embedding_client = EmbeddingClient(
            base_url=config.EMBEDDING_API_URL,
            model=cast(str, config.EMBEDDING_MODEL),
            timeout_seconds=config.EMBEDDING_TIMEOUT_SECONDS,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            api_key=config.EMBEDDING_API_KEY or "",
        )
    elif backend == "local":
        if not config.EMBEDDING_MODEL:
            logger.warning(
                "Semantic search disabled: EMBEDDING_MODEL is required "
                "when EMBEDDING_BACKEND=local"
            )
            return None
        embedding_client = LocalEmbeddingClient(
            model=cast(str, config.EMBEDDING_MODEL),
            device=cast(str, config.EMBEDDING_DEVICE),
            trust_remote_code=bool(config.EMBEDDING_TRUST_REMOTE_CODE),
            batch_size=config.EMBEDDING_BATCH_SIZE,
            max_chars_per_text=config.EMBEDDING_MAX_CHARS,
        )
    else:
        logger.warning(
            "Semantic search disabled: EMBEDDING_BACKEND must be 'api' or 'local'"
        )
        return None

    try:
        vector_store = ChromaVectorStore.from_host(
            chroma_host=config.CHROMA_HOST,
            collection_name=cast(str, config.CHROMA_COLLECTION),
        )
        return SemanticSearchService(
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
    except Exception as exc:
        logger.warning("Semantic search disabled: %s", exc)
        return None
