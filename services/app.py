"""
Combined FastAPI application: API (JSON at /api) + UI (HTML at /).

Both sub-apps share the same JobStore so jobs created via the API
are visible in the UI and vice versa.  A background worker thread
can be enabled to process queued jobs within the same process
(needed while using the in-memory store).
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from services.ingest_api.app import create_app as create_api_app
from services.ingest_api.store import JobStore
from services.ingest_ui.app import create_ui_app


def create_combined_app(
    upload_dir: Path | None = None,
    store: JobStore | None = None,
    *,
    enable_worker: bool = False,
) -> FastAPI:
    """Build the combined application with shared state.

    When *enable_worker* is True a daemon thread polls the store
    for queued jobs and feeds them through the OCR pipeline.
    """
    if store is None:
        store = JobStore()
    if upload_dir is None:
        upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        if enable_worker:
            from services.worker.ocr_worker import OcrWorker
            from services.worker.run import run_loop

            worker = OcrWorker(
                store=store,
                upload_dir=upload_dir,
                output_dir=output_dir,
            )
            thread = threading.Thread(target=run_loop, args=(worker,), daemon=True)
            thread.start()
        yield

    application = FastAPI(
        title="Doc Parser Services",
        version="0.1.0",
        lifespan=lifespan,
    )

    api = create_api_app(upload_dir=upload_dir, store=store)
    ui = create_ui_app(upload_dir=upload_dir, store=store)

    application.mount("/api", api)
    application.mount("/", ui)

    return application


_enable = os.getenv("ENABLE_WORKER", "").lower() in ("true", "1")
app = create_combined_app(enable_worker=_enable)
