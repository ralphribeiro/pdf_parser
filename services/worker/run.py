"""
Worker runner: polls for queued jobs and processes them.

Can be used standalone (``python -m services.worker.run``) or
called from the combined app as a background thread.
"""

from __future__ import annotations

import logging
import time
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
