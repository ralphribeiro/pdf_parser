"""
E2E smoke test: async job processing + semantic search.

Flow:
1) Submit PDF to async jobs endpoint.
2) Poll job status until completed/failed.
3) Execute semantic search query based on reference JSON facts.
4) Validate that the expected document appears in results.

Usage:
    python scripts/test_e2e_async_semantic.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

API_BASE_URL = os.getenv("DOC_PARSER_E2E_API_URL", "http://127.0.0.1:8000")
POLL_INTERVAL_SECONDS = int(os.getenv("DOC_PARSER_E2E_POLL_INTERVAL", "10"))
POLL_TIMEOUT_SECONDS = int(os.getenv("DOC_PARSER_E2E_POLL_TIMEOUT", "2400"))

PDF_PATH = Path("resource/1008086-69.2016.8.26.0005.pdf")
REFERENCE_JSON_PATH = Path("output/1008086-69.2016.8.26.0005.json")


def load_reference() -> tuple[str, str]:
    """Load reference doc_id and build semantic query from reference JSON."""
    with REFERENCE_JSON_PATH.open("r", encoding="utf-8") as f:
        reference = json.load(f)

    doc_id = reference.get("doc_id")
    if not doc_id:
        raise RuntimeError("Reference JSON missing 'doc_id'.")

    # Question derived from the reference document context.
    # This process repeatedly mentions execution action and parties.
    query = (
        f"Processo {doc_id} de execução de título extrajudicial "
        "Banco Santander Brasil S/A contra Ana Paula Morelato Teixeira"
    )
    return doc_id, query


def submit_async_job(pdf_path: Path) -> str:
    """Submit PDF for async processing and return job_id."""
    with pdf_path.open("rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data = {"generate_embeddings": "true"}
        response = requests.post(
            f"{API_BASE_URL}/jobs/", files=files, data=data, timeout=180
        )

    response.raise_for_status()
    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"Missing job_id in response: {payload}")
    return job_id


def wait_for_job_completion(job_id: str) -> dict:
    """Poll job status endpoint until completion, failure, or timeout."""
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    last_payload: dict = {}

    while time.time() < deadline:
        response = requests.get(f"{API_BASE_URL}/jobs/{job_id}", timeout=60)
        response.raise_for_status()
        payload = response.json()
        last_payload = payload
        status = payload.get("status", "")

        print(f"[poll] job_id={job_id} status={status}")

        if status == "completed":
            return payload
        if status == "failed":
            raise RuntimeError(f"Job failed: {payload}")

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Timed out waiting for job completion. Last payload: {last_payload}"
    )


def run_semantic_search(query: str) -> dict:
    """Run semantic search and return JSON payload."""
    body = {
        "query": query,
        "top_k": 10,
        "min_score": 0.0,
        "include_matches": False,
    }
    response = requests.post(f"{API_BASE_URL}/search/semantic", json=body, timeout=180)
    response.raise_for_status()
    return response.json()


def validate_semantic_result(search_payload: dict, expected_doc_id: str) -> None:
    """Ensure semantic search includes the expected processed document."""
    results = search_payload.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError(f"Invalid semantic search response shape: {search_payload}")

    if not results:
        raise RuntimeError("Semantic search returned zero results.")

    doc_ids = [r.get("doc_id") for r in results]
    if expected_doc_id not in doc_ids:
        raise RuntimeError(
            f"Expected doc_id '{expected_doc_id}' not found in semantic results: {doc_ids}"
        )


def main() -> int:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
    if not REFERENCE_JSON_PATH.exists():
        raise FileNotFoundError(f"Reference JSON not found: {REFERENCE_JSON_PATH}")

    reference_doc_id, query = load_reference()
    print(f"[info] reference_doc_id={reference_doc_id}")
    print(f"[info] semantic_query={query}")

    job_id = submit_async_job(PDF_PATH)
    print(f"[info] submitted job_id={job_id}")

    job_payload = wait_for_job_completion(job_id)
    print(f"[info] job completed payload keys={list(job_payload.keys())}")

    # In async pipeline the processed document id can differ from source filename
    # (e.g., temp file stem). Validate semantic search against the actual produced doc_id.
    expected_doc_id = job_payload.get("result", {}).get("doc_id") or reference_doc_id
    print(f"[info] semantic_expected_doc_id={expected_doc_id}")

    search_payload = run_semantic_search(query)
    print(f"[info] semantic total_results={search_payload.get('total_results')}")

    validate_semantic_result(search_payload, expected_doc_id)
    print("[ok] E2E async job + semantic search passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
