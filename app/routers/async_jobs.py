"""
Endpoints de jobs assíncronos.

Testes: tests/test_async_jobs.py
"""

import logging
import mimetypes
import os
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.celery_worker import (
    cancel_job,
    process_pdf_job,
)
from src.database import (
    create_job_document,
    get_job,
    get_jobs_by_status,
)
from src.models.mongodb import JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Async Jobs"])


# Schemas de resposta


class JobCreateResponse(BaseModel):
    """Resposta de criação de job."""

    job_id: str = Field(..., description="ID do job")
    status: str = Field(..., description="Status do job")
    created_at: datetime = Field(..., description="Timestamp de criação")
    file_size: int = Field(..., description="Tamanho do arquivo em bytes")


class JobStatusResponse(BaseModel):
    """Resposta de status do job."""

    job_id: str = Field(..., description="ID do job")
    status: str = Field(..., description="Status do job")
    created_at: datetime = Field(..., description="Timestamp de criação")
    updated_at: datetime | None = Field(
        default=None, description="Timestamp de atualização"
    )
    result: dict | None = Field(default=None, description="Resultado do job")
    error: str | None = Field(default=None, description="Erro do job")
    embeddings_generated: bool = Field(default=False, description="Embeddings gerados")


class WebhookPayload(BaseModel):
    """Payload de webhook."""

    job_id: str
    status: str
    result: dict | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Funções auxiliares


def _coerce_datetime(value: datetime | str | None) -> datetime | None:
    """Aceita datetime nativo ou string ISO e retorna datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value)
    return None


def validate_pdf_file(file: UploadFile) -> tuple[bytes, str]:
    """
    Validar arquivo PDF.

    Args:
        file: Arquivo upload

    Returns:
        Tuple com (conteúdo, nome do arquivo)

    Raises:
        HTTPException: Se arquivo não for PDF
    """
    # Verificar extensão
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Apenas arquivos PDF são permitidos"
        )

    # Verificar mimetype
    mimetype, _ = mimetypes.guess_type(file.filename)
    if mimetype and not mimetype.startswith("application/pdf"):
        raise HTTPException(status_code=400, detail="Tipo de arquivo não é PDF")

    # Ler conteúdo
    contents = file.file.read()

    # Verificar cabeçalho PDF
    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Arquivo não é um PDF válido")

    # Verificar tamanho
    file_size = len(contents)
    max_file_size = (
        int(os.getenv("DOC_PARSER_MAX_FILE_SIZE", "50")) * 1024 * 1024
    )  # 50MB

    if file_size > max_file_size:
        max_size_mb = max_file_size // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande. Máximo permitido: {max_size_mb}MB",
        )

    return contents, file.filename


def create_job_id(file_content: bytes) -> str:
    """
    Criar ID único para job.

    Args:
        file_content: Conteúdo do arquivo

    Returns:
        ID único
    """
    # Usar hash do arquivo + timestamp para ID único
    import hashlib

    content_hash = hashlib.sha256(file_content).hexdigest()[:16]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"job_{content_hash}_{timestamp}"


@router.post(
    "/", include_in_schema=False, response_model=JobCreateResponse, status_code=202
)
async def create_job(
    file: UploadFile = File(..., description="Arquivo PDF a processar"),
    response_format: str = Form("json", description="Formato de resposta"),
    extract_tables: bool = Form(True, description="Extraer tabelas"),
    min_confidence: float = Form(0.3, ge=0, le=1, description="Confiança mínima OCR"),
    ocr_postprocess: bool = Form(True, description="Pós-processamento OCR"),
    ocr_fix_errors: bool = Form(True, description="Corrigir erros OCR"),
    generate_embeddings: bool = Form(True, description="Gerar embeddings"),
    webhook_url: str | None = Form(None, description="URL para webhook"),
    detect_orientation: bool = Form(True, description="Detectar orientação"),
    straighten_pages: bool = Form(True, description="Corrigir inclinação"),
    assume_straight_pages: bool = Form(False, description="Assumir páginas retas"),
) -> JobCreateResponse:
    """
    Criar novo job de processamento.

    O job é processado em background e o cliente recebe o job_id para consultar
    o status.
    """
    # Validar arquivo
    file_content, _file_name = validate_pdf_file(file)

    # Criar job_id único
    job_id = create_job_id(file_content)

    try:
        # Criar documento do job no MongoDB
        create_job_document(
            job_id=job_id,
            file_content=file_content,
            generate_embeddings=generate_embeddings,
            webhook_url=webhook_url,
        )

        # Disparar task de processamento
        process_pdf_job.delay(
            job_id=job_id,
            file_content=file_content,
            generate_embeddings=generate_embeddings,
            webhook_url=webhook_url,
        )

        logger.info(f"Job criado: {job_id}")

        return JobCreateResponse(
            job_id=job_id,
            status="pending",
            created_at=datetime.utcnow(),
            file_size=len(file_content),
        )

    except Exception as e:
        logger.error(f"Erro ao criar job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao criar job: {e!s}")


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    """
    Obter status de job.

    O cliente pode consultar o status do job usando o job_id retornado na criação.
    """
    try:
        job = get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job não encontrado: {job_id}")

        # Converter timestamp para datetime aceitando string ou datetime.
        created_at = _coerce_datetime(job.get("created_at"))
        updated_at = _coerce_datetime(job.get("updated_at"))

        if created_at is None:
            raise HTTPException(
                status_code=500, detail=f"Job inválido sem created_at: {job_id}"
            )

        return JobStatusResponse(
            job_id=job.get("job_id", job_id),
            status=job.get("status", "pending"),
            created_at=created_at,
            updated_at=updated_at,
            result=job.get("result"),
            error=job.get("error"),
            embeddings_generated=job.get("embeddings_generated", False),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter job: {e!s}")


@router.delete("/{job_id}")
async def cancel_job_endpoint(job_id: str, request: Request) -> JobStatusResponse:
    """
    Cancelar job pendente.

    Apenas jobs com status 'pending' ou 'processing' podem ser cancelados.
    """
    # Verificar se job existe
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job não encontrado: {job_id}")

    # Verificar status
    current_status = job.get("status")
    if current_status not in ["pending", "processing"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job não pode ser cancelado. Status atual: {current_status}",
        )

    try:
        # Cancelar job
        cancel_job.delay(job_id)

        return JobStatusResponse(
            job_id=job_id,
            status="cancelled",
            created_at=datetime.fromisoformat(job.get("created_at", "")),
            updated_at=datetime.utcnow(),
            error="Job cancelado",
        )

    except Exception as e:
        logger.error(f"Erro ao cancelar job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao cancelar job: {e!s}")


@router.get("")
async def list_jobs(
    status: str | None = None, limit: int = 50
) -> list[JobStatusResponse]:
    """
    Listar jobs (para debug).
    """
    try:
        if status:
            jobs = get_jobs_by_status(JobStatus(status))
        else:
            jobs = get_jobs_by_status(JobStatus.COMPLETED)

        results = []
        for job in jobs[:limit]:
            created_at = _coerce_datetime(job.get("created_at"))
            updated_at = _coerce_datetime(job.get("updated_at"))

            if created_at is None:
                continue

            results.append(
                JobStatusResponse(
                    job_id=job.get("job_id"),
                    status=job.get("status"),
                    created_at=created_at,
                    updated_at=updated_at,
                    result=job.get("result"),
                    error=job.get("error"),
                )
            )

        return results

    except Exception as e:
        logger.error(f"Erro ao listar jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao listar jobs: {e!s}")


@router.post("/{job_id}/webhook")
async def webhook_endpoint(
    job_id: str, payload: WebhookPayload, request: Request
) -> dict:
    """
    Endpoint para receber webhooks.

    Este endpoint pode ser usado para configurar webhooks externos.
    """
    # Verificar autenticação (opcional)
    auth_header = request.headers.get("Authorization")
    expected_token = os.getenv("DOC_PARSER_WEBHOOK_AUTH_TOKEN")

    if expected_token and not auth_header:
        raise HTTPException(status_code=401, detail="Token de autenticação necessário")

    # Salvar webhook no MongoDB
    from src.database import create_webhook

    create_webhook(
        job_id=job_id,
        url=payload.url if hasattr(payload, "url") else None,
        token=payload.token if hasattr(payload, "token") else None,
    )

    return {"success": True, "job_id": job_id}
