"""
Combined FastAPI application: API (JSON at /api) + UI (HTML at /).

Both sub-apps share the same store (memory or Redis) so jobs created
via the API are visible in the UI and vice versa.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI

import config
from services.document_store import create_document_store
from services.ingest_api.app import create_app as create_api_app
from services.ingest_api.store import JobStore, create_store
from services.ingest_ui.app import create_ui_app
from services.search.factory import create_semantic_search_service

config.setup_logging()

logger = logging.getLogger(__name__)


def _create_agent_runner(semantic_search: Any, document_store: Any) -> Any | None:
    """Build the AgentRunner if LLM_API_URL is configured."""
    import config

    if not config.LLM_API_URL:
        logger.info("LLM_API_URL not set — agent search disabled")
        return None

    from services.agent.agent import AgentRunner
    from services.agent.llm_client import LlmClient
    from services.agent.tools import ToolRegistry

    model: str = config.LLM_MODEL or "Qwen3.5-9B-Q4_K_M"
    llm = LlmClient(
        base_url=config.LLM_API_URL,
        model=model,
        max_tokens=config.LLM_MAX_TOKENS,
        api_key=config.LLM_API_KEY or "",
    )
    tool_registry = ToolRegistry(
        search_service=semantic_search,
        document_store=document_store,
    )
    return AgentRunner(
        llm=llm,
        tool_registry=tool_registry,
        max_iterations=config.AGENT_MAX_ITERATIONS,
        context_budget=config.AGENT_CONTEXT_BUDGET,
    )


_UNSET = object()


def create_combined_app(
    upload_dir: Path | None = None,
    store: JobStore | None = None,
    document_store: Any = _UNSET,
) -> FastAPI:
    """Build the combined application with shared state."""
    if store is None:
        store = create_store()
    semantic_search = create_semantic_search_service()
    if document_store is _UNSET:
        document_store = create_document_store()
    agent_runner = _create_agent_runner(semantic_search, document_store)
    if upload_dir is None:
        upload_dir = Path("data")
    upload_dir.mkdir(parents=True, exist_ok=True)

    application = FastAPI(
        title="Doc Parser Services",
        version="0.1.0",
    )

    api = create_api_app(
        upload_dir=upload_dir,
        store=store,
        semantic_search=semantic_search,
        document_store=document_store,
        agent_runner=agent_runner,
    )
    ui = create_ui_app(upload_dir=upload_dir, store=store)

    application.mount("/api", api)
    application.mount("/", ui)

    return application


app = create_combined_app()
