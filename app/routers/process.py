"""
Main API router for document processing.

Endpoints:
    POST /process  - Process a PDF and return JSON or searchable PDF
    GET  /health   - Service healthcheck
    GET  /info     - Current pipeline configuration
"""

import asyncio
import gc
import tempfile
import time
from enum import StrEnum
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

import config
from app.dependencies import get_processor, get_semaphore
from app.schemas import ErrorResponse, HealthResponse, InfoResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResponseFormat(StrEnum):
    JSON = "json"
    PDF = "pdf"


# ---------------------------------------------------------------------------
# POST /process
# ---------------------------------------------------------------------------


@router.post(
    "/process",
    summary="Process PDF document",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        422: {"description": "Invalid parameters"},
    },
)
async def process_document(
    file: UploadFile = File(..., description="PDF file to process"),
    response_format: ResponseFormat = Query(
        ResponseFormat.JSON,
        description="Response format: json or pdf",
    ),
    extract_tables: bool = Query(True, description="Extract tables from PDF"),
    min_confidence: float | None = Query(
        None, ge=0.0, le=1.0, description="Minimum OCR confidence (overrides config)"
    ),
    ocr_postprocess: bool | None = Query(
        None, description="OCR post-processing (overrides config)"
    ),
    ocr_fix_errors: bool | None = Query(
        None, description="Fix common OCR errors (overrides config)"
    ),
    processor=Depends(get_processor),
    semaphore: asyncio.Semaphore = Depends(get_semaphore),
):
    """
    Receives a PDF via upload, processes it with OCR/digital extraction
    and returns the result as JSON or as a searchable PDF.
    """
    # 1. Validate file
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    content = await file.read()

    if not content or content[:5] != b"%PDF-":
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    # 2. Save to temporary file
    tmp_path = None
    tmp_output_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="doc_parser_")
        tmp_path = Path(tmp_dir) / file.filename
        tmp_path.write_bytes(content)
        del content  # free memory

        # 3. Apply config overrides for this request
        original_confidence = config.MIN_CONFIDENCE
        original_postprocess = config.OCR_POSTPROCESS
        original_fix_errors = config.OCR_FIX_ERRORS

        if min_confidence is not None:
            config.MIN_CONFIDENCE = min_confidence
        if ocr_postprocess is not None:
            config.OCR_POSTPROCESS = ocr_postprocess
        if ocr_fix_errors is not None:
            config.OCR_FIX_ERRORS = ocr_fix_errors

        # 4. Process with semaphore (serializes GPU access)
        start_time = time.monotonic()

        try:
            async with semaphore:
                loop = asyncio.get_event_loop()
                document = await loop.run_in_executor(
                    None,
                    lambda: processor.process_document_parallel(
                        str(tmp_path),
                        extract_tables=extract_tables,
                        show_progress=False,
                    ),
                )
        finally:
            # Restore original config
            config.MIN_CONFIDENCE = original_confidence
            config.OCR_POSTPROCESS = original_postprocess
            config.OCR_FIX_ERRORS = original_fix_errors

        elapsed = time.monotonic() - start_time

        # 5. Return result
        if response_format == ResponseFormat.JSON:
            result = document.to_json_dict()
            result["processing_time_seconds"] = round(elapsed, 3)
            return JSONResponse(content=result)

        # response_format == PDF
        tmp_output_dir = tempfile.mkdtemp(prefix="doc_parser_out_")
        pdf_output = Path(tmp_output_dir) / f"{document.doc_id}_searchable.pdf"

        await loop.run_in_executor(
            None,
            lambda: processor.save_to_searchable_pdf(
                document, str(tmp_path), str(pdf_output)
            ),
        )

        filename = f"{document.doc_id}_searchable.pdf"

        # Clean input temp now; output temp cleaned AFTER response is sent
        _cleanup_dir(tmp_path.parent)
        tmp_path = None  # Avoid double cleanup in finally

        return FileResponse(
            path=str(pdf_output),
            media_type="application/pdf",
            filename=filename,
            background=BackgroundTask(_cleanup_dir, Path(tmp_output_dir)),
        )

    finally:
        # Clean temporary files (JSON path; PDF cleaned via BackgroundTask)
        if tmp_path and tmp_path.exists():
            _cleanup_dir(tmp_path.parent)
        gc.collect()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service healthcheck",
)
async def health(processor=Depends(get_processor)):
    """Returns service status, GPU and loaded OCR engine."""
    return HealthResponse(
        status="ok",
        gpu_available=processor.use_gpu,
        ocr_engine=processor.ocr_engine_type,
        device=config.DEVICE,
    )


# ---------------------------------------------------------------------------
# GET /info
# ---------------------------------------------------------------------------


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="Current pipeline configuration",
)
async def info():
    """Returns the current configuration (read-only)."""
    return InfoResponse(
        ocr_engine=config.OCR_ENGINE,
        ocr_dpi=config.OCR_DPI,
        ocr_batch_size=config.OCR_BATCH_SIZE,
        min_confidence=config.MIN_CONFIDENCE,
        ocr_postprocess=config.OCR_POSTPROCESS,
        ocr_fix_errors=config.OCR_FIX_ERRORS,
        ocr_lang=config.OCR_LANG,
        assume_straight_pages=config.ASSUME_STRAIGHT_PAGES,
        detect_orientation=config.DETECT_ORIENTATION,
        parallel_enabled=config.PARALLEL_ENABLED,
        parallel_workers=config.PARALLEL_WORKERS,
        searchable_pdf=config.SEARCHABLE_PDF,
        device=config.DEVICE,
        use_gpu=config.USE_GPU,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _cleanup_dir(path: Path):
    """Remove temporary directory and its contents."""
    import shutil

    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: S110
        pass
