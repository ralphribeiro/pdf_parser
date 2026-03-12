"""
Factory for semantic search service wiring.
"""

from __future__ import annotations

import logging
from typing import cast

from services.search.chroma_store import ChromaVectorStore
from services.search.embedding_client import EmbeddingClient
from services.search.service import SemanticSearchService

logger = logging.getLogger(__name__)


def create_semantic_search_service() -> SemanticSearchService | None:
    """Build semantic search service if all required config exists."""
    import config

    if not config.CHROMA_HOST or not config.EMBEDDING_API_URL:
        return None

    embedding_client = EmbeddingClient(
        base_url=config.EMBEDDING_API_URL,
        model=cast(str, config.EMBEDDING_MODEL),
        timeout_seconds=config.EMBEDDING_TIMEOUT_SECONDS,
    )
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
