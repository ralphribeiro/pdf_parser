"""
Worker runner: polls for queued jobs and processes them.

Run standalone::

    python -m services.worker.run
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from services.worker.ocr_worker import OcrWorker

logger = logging.getLogger(__name__)


def run_loop(
    worker: OcrWorker | Any,
    interval: float = 2.0,
    max_iterations: int | None = None,
) -> None:
    """Poll *worker.process_next()* in a loop.

    Sleeps *interval* seconds when no job was found.
    Set *max_iterations* for testing; ``None`` means run forever.
    """
    count = 0
    while max_iterations is None or count < max_iterations:
        processed = worker.process_next()
        if not processed:
            time.sleep(interval)
        count += 1


def main() -> None:
    """Bootstrap a standalone worker process using the store factory."""
    from services.document_store import create_document_store
    from services.ingest_api.store import create_store
    from services.search.factory import create_semantic_search_service
    from services.worker.ocr_worker import OcrWorker

    store = create_store()
    semantic_search = create_semantic_search_service()
    document_store = create_document_store()

    upload_dir = Path(os.getenv("DOC_PARSER_UPLOAD_DIR", "data"))
    output_dir = Path(os.getenv("DOC_PARSER_OUTPUT_DIR", "output"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    worker = OcrWorker(
        store=store,
        upload_dir=upload_dir,
        output_dir=output_dir,
        semantic_indexer=semantic_search,
        document_store=document_store,
    )

    logger.info("Worker started (store=%s)", type(store).__name__)
    run_loop(worker)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
