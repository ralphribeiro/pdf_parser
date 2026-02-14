"""
Router principal da API de processamento de documentos.

Endpoints:
    POST /process  - Processa um PDF e retorna JSON ou PDF pesquisável
    GET  /health   - Healthcheck do serviço
    GET  /info     - Configuração atual do pipeline
"""
import asyncio
import gc
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Optional

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


class ResponseFormat(str, Enum):
    JSON = "json"
    PDF = "pdf"


# ---------------------------------------------------------------------------
# POST /process
# ---------------------------------------------------------------------------


@router.post(
    "/process",
    summary="Processar documento PDF",
    responses={
        400: {"model": ErrorResponse, "description": "Arquivo inválido"},
        422: {"description": "Parâmetros inválidos"},
    },
)
async def process_document(
    file: UploadFile = File(..., description="Arquivo PDF para processar"),
    response_format: ResponseFormat = Query(
        ResponseFormat.JSON,
        description="Formato da resposta: json ou pdf",
    ),
    extract_tables: bool = Query(True, description="Extrair tabelas do PDF"),
    min_confidence: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Confiança mínima OCR (sobrescreve config)"
    ),
    ocr_postprocess: Optional[bool] = Query(
        None, description="Pós-processamento OCR (sobrescreve config)"
    ),
    ocr_fix_errors: Optional[bool] = Query(
        None, description="Corrigir erros comuns OCR (sobrescreve config)"
    ),
    processor=Depends(get_processor),
    semaphore: asyncio.Semaphore = Depends(get_semaphore),
):
    """
    Recebe um PDF via upload, processa com OCR/extração digital
    e retorna o resultado em JSON ou como PDF pesquisável.
    """
    # 1. Validar arquivo
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF")

    content = await file.read()

    if not content or content[:5] != b"%PDF-":
        raise HTTPException(
            status_code=400, detail="Arquivo não é um PDF válido"
        )

    # 2. Salvar em arquivo temporário
    tmp_path = None
    tmp_output_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="doc_parser_")
        tmp_path = Path(tmp_dir) / file.filename
        tmp_path.write_bytes(content)
        del content  # libera memória

        # 3. Aplicar overrides de config para este request
        original_confidence = config.MIN_CONFIDENCE
        original_postprocess = config.OCR_POSTPROCESS
        original_fix_errors = config.OCR_FIX_ERRORS

        if min_confidence is not None:
            config.MIN_CONFIDENCE = min_confidence
        if ocr_postprocess is not None:
            config.OCR_POSTPROCESS = ocr_postprocess
        if ocr_fix_errors is not None:
            config.OCR_FIX_ERRORS = ocr_fix_errors

        # 4. Processar com semáforo (serializa acesso à GPU)
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
            # Restaura config original
            config.MIN_CONFIDENCE = original_confidence
            config.OCR_POSTPROCESS = original_postprocess
            config.OCR_FIX_ERRORS = original_fix_errors

        elapsed = time.monotonic() - start_time

        # 5. Retornar resultado
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

        # Limpa input temp agora; output temp limpa APÓS response ser enviada
        _cleanup_dir(tmp_path.parent)
        tmp_path = None  # Evita dupla limpeza no finally

        return FileResponse(
            path=str(pdf_output),
            media_type="application/pdf",
            filename=filename,
            background=BackgroundTask(
                _cleanup_dir, Path(tmp_output_dir)
            ),
        )

    finally:
        # Limpa arquivos temporários (JSON path; PDF limpa via BackgroundTask)
        if tmp_path and tmp_path.exists():
            _cleanup_dir(tmp_path.parent)
        gc.collect()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Healthcheck do serviço",
)
async def health(processor=Depends(get_processor)):
    """Retorna status do serviço, GPU e engine OCR carregado."""
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
    summary="Configuração atual do pipeline",
)
async def info():
    """Retorna a configuração atual (somente leitura)."""
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
# Utilitários
# ---------------------------------------------------------------------------


def _cleanup_dir(path: Path):
    """Remove diretório temporário e seu conteúdo."""
    import shutil

    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
