"""
FastAPI application for asynchronous document ingest.

Endpoints:
    POST /jobs              - Upload PDF and create a processing job (202 / 409)
    GET  /jobs/{job_id}     - Query job status (200 / 404)
    POST /search            - Semantic search over indexed chunks
    GET  /documents/{id}    - Retrieve parsed document from MongoDB
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from services.ingest_api.schemas import Job, SearchRequest, SearchResponse
from services.ingest_api.store import JobStore


def create_app(
    upload_dir: Path | None = None,
    store: JobStore | None = None,
    semantic_search: Any = None,
    document_store: Any = None,
) -> FastAPI:
    """Factory with dependency injection for testing."""
    if store is None:
        store = JobStore()
    if upload_dir is None:
        upload_dir = Path("data")

    upload_dir.mkdir(parents=True, exist_ok=True)

    application = FastAPI(
        title="Ingest API",
        description="Async document ingest — upload PDFs and track processing status",
        version="0.1.0",
    )

    application.state.store = store
    application.state.upload_dir = upload_dir
    application.state.semantic_search = semantic_search
    application.state.document_store = document_store

    _register_routes(application)

    return application


def _compute_hash(content: bytes) -> str:
    """Return SHA-256 hex digest of in-memory content."""
    return hashlib.sha256(content).hexdigest()


def _register_routes(app: FastAPI) -> None:
    @app.get("/jobs/healthcheck", summary="Healthcheck")
    async def jobs_healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/jobs",
        response_model=Job,
        status_code=202,
        summary="Upload PDF and create processing job",
        responses={
            400: {"description": "Invalid file"},
            409: {"description": "Duplicate document"},
        },
    )
    async def create_job(
        request: Request,
        file: UploadFile = File(..., description="PDF file to process"),
    ) -> Job:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        content = await file.read()

        if not content or content[:5] != b"%PDF-":
            raise HTTPException(status_code=400, detail="File is not a valid PDF")

        store: JobStore = request.app.state.store
        doc_store = request.app.state.document_store
        file_hash = _compute_hash(content)
        document_id: str | None = None

        if doc_store is not None:
            existing = doc_store.find_by_hash(file_hash)
            if existing is not None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Document already exists",
                        "document_id": str(existing["_id"]),
                    },
                )
            document_id = doc_store.create_document(
                file_hash=file_hash,
                filename=file.filename,
                file_size=len(content),
                pdf_path="",
            )
            doc_store.update_pdf_path(document_id, f"data/{document_id}.pdf")

        job = store.create(filename=file.filename, document_id=document_id)

        pdf_name = document_id if document_id else job.job_id
        dest: Path = request.app.state.upload_dir / f"{pdf_name}.pdf"
        dest.write_bytes(content)

        if doc_store is not None and document_id is not None:
            job = job.model_copy(update={"file_hash": file_hash})

        return job

    @app.get(
        "/jobs/{job_id}",
        response_model=Job,
        summary="Get job status",
        responses={404: {"description": "Job not found"}},
    )
    async def get_job(job_id: str, request: Request) -> Job:
        store: JobStore = request.app.state.store
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post(
        "/search",
        response_model=SearchResponse,
        summary="Semantic search over indexed chunks",
    )
    async def search(request: Request, payload: SearchRequest) -> SearchResponse:
        service = request.app.state.semantic_search
        if service is None:
            raise HTTPException(
                status_code=503,
                detail="Semantic search service is not configured",
            )

        started = time.monotonic()
        results = service.search(
            payload.query,
            n_results=payload.n_results,
            document_id=(payload.filters.document_id if payload.filters else None),
            min_similarity=payload.min_similarity,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return SearchResponse(
            results=results,
            total_matches=len(results),
            processing_time_ms=elapsed_ms,
        )

    @app.get(
        "/documents/{document_id}",
        summary="Retrieve parsed document from MongoDB",
        responses={
            404: {"description": "Document not found"},
            503: {"description": "Document store not configured"},
        },
    )
    async def get_document(document_id: str, request: Request) -> dict:
        doc_store = request.app.state.document_store
        if doc_store is None:
            raise HTTPException(
                status_code=503,
                detail="Document store is not configured",
            )
        doc = doc_store.get_document(document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        doc["_id"] = str(doc["_id"])
        return doc
