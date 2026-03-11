"""
FastAPI application for asynchronous document ingest.

Endpoints:
    POST /jobs          - Upload PDF and create a processing job (202)
    GET  /jobs/{job_id} - Query job status (200 / 404)
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from services.ingest_api.schemas import Job
from services.ingest_api.store import JobStore


def create_app(
    upload_dir: Path | None = None,
    store: JobStore | None = None,
) -> FastAPI:
    """Factory with dependency injection for testing."""
    if store is None:
        store = JobStore()
    if upload_dir is None:
        upload_dir = Path("uploads")

    upload_dir.mkdir(parents=True, exist_ok=True)

    application = FastAPI(
        title="Ingest API",
        description="Async document ingest — upload PDFs and track processing status",
        version="0.1.0",
    )

    application.state.store = store
    application.state.upload_dir = upload_dir

    _register_routes(application)

    return application


def _register_routes(app: FastAPI) -> None:

    @app.post(
        "/jobs",
        response_model=Job,
        status_code=202,
        summary="Upload PDF and create processing job",
        responses={400: {"description": "Invalid file"}},
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
        job = store.create(filename=file.filename)

        dest: Path = request.app.state.upload_dir / f"{job.job_id}.pdf"
        dest.write_bytes(content)

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
