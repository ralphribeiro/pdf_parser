#!/usr/bin/env python3
"""
Backfill MongoDB documents collection from existing jobs.

Jobs uploaded before the MongoDB integration have document_id=None.
This script creates MongoDB entries for those jobs and saves the
parsed output (JSON) when available.

Usage (inside container or with correct env):
    python scripts/backfill_documents.py [--dry-run]
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

config.setup_logging()

from services.document_store import create_document_store  # noqa: E402
from services.ingest_api.store import create_store  # noqa: E402

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(*, dry_run: bool = False) -> int:
    store = create_store()
    doc_store = create_document_store()
    if doc_store is None:
        logger.error("MONGO_URL not set — cannot backfill")
        return 1

    upload_dir = Path(os.getenv("DOC_PARSER_UPLOAD_DIR", "data"))
    output_dir = Path(config.OUTPUT_DIR)

    jobs = store.list_all(limit=1000)
    backfilled = 0

    for job in jobs:
        if job.document_id is not None:
            logger.debug("Skipping %s — already has document_id", job.job_id)
            continue

        pdf_path = upload_dir / f"{job.job_id}.pdf"
        if not pdf_path.exists():
            logger.warning("PDF not found for job %s at %s", job.job_id, pdf_path)
            continue

        file_hash = _sha256(pdf_path)
        existing = doc_store.find_by_hash(file_hash)
        if existing:
            doc_id = str(existing["_id"])
            logger.info(
                "Job %s: hash already in MongoDB as %s — skipping",
                job.job_id,
                doc_id,
            )
            continue

        if dry_run:
            logger.info(
                "[DRY-RUN] Would create document for job %s (%s)",
                job.job_id,
                job.filename,
            )
            backfilled += 1
            continue

        doc_id = doc_store.create_document(
            file_hash=file_hash,
            filename=job.filename,
            file_size=pdf_path.stat().st_size,
            pdf_path=str(pdf_path),
        )
        logger.info(
            "Job %s: created MongoDB document %s (%s)",
            job.job_id,
            doc_id,
            job.filename,
        )

        json_path = output_dir / f"{job.job_id}.json"
        if json_path.exists():
            try:
                parsed = json.loads(json_path.read_text())
                total_pages = len(parsed.get("pages", []))
                doc_store.save_parsed(doc_id, parsed, total_pages)
                logger.info(
                    "Job %s: saved parsed document (%d pages)",
                    job.job_id,
                    total_pages,
                )
            except Exception:
                logger.exception("Job %s: failed to save parsed doc", job.job_id)
        else:
            logger.info("Job %s: no parsed JSON at %s", job.job_id, json_path)

        backfilled += 1

    action = "would backfill" if dry_run else "backfilled"
    logger.info("Done: %s %d document(s)", action, backfilled)
    return 0


if __name__ == "__main__":
    dry_run_flag = "--dry-run" in sys.argv
    sys.exit(main(dry_run=dry_run_flag))
