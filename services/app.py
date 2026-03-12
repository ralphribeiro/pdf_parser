"""
Combined FastAPI application: API (JSON at /api) + UI (HTML at /).

Both sub-apps share the same store (memory or Redis) so jobs created
via the API are visible in the UI and vice versa.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from services.ingest_api.app import create_app as create_api_app
from services.ingest_api.store import JobStore, create_store
from services.ingest_ui.app import create_ui_app


def create_combined_app(
    upload_dir: Path | None = None,
    store: JobStore | None = None,
) -> FastAPI:
    """Build the combined application with shared state."""
    if store is None:
        store = create_store()
    if upload_dir is None:
        upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    application = FastAPI(
        title="Doc Parser Services",
        version="0.1.0",
    )

    api = create_api_app(upload_dir=upload_dir, store=store)
    ui = create_ui_app(upload_dir=upload_dir, store=store)

    application.mount("/api", api)
    application.mount("/", ui)

    return application


app = create_combined_app()
