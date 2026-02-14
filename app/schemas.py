"""
Schemas Pydantic para request/response da API.

Separados dos schemas do pipeline (src/models/schemas.py) para manter
a camada HTTP desacoplada da lógica de negócio.
"""
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass
class ProcessingOptions:
    """
    Opções de processamento por request.

    Valores None significam "usar default do config".
    """

    extract_tables: bool = True
    min_confidence: Optional[float] = None
    ocr_postprocess: Optional[bool] = None
    ocr_fix_errors: Optional[bool] = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Resposta do endpoint /health."""

    status: str = Field(description="Status do serviço")
    gpu_available: bool = Field(description="GPU disponível para OCR")
    ocr_engine: str = Field(description="Engine OCR carregado")
    device: str = Field(description="Device em uso (cpu/cuda)")


class InfoResponse(BaseModel):
    """Resposta do endpoint /info com configuração atual."""

    ocr_engine: str
    ocr_dpi: int
    ocr_batch_size: int
    min_confidence: float
    ocr_postprocess: bool
    ocr_fix_errors: bool
    ocr_lang: str
    assume_straight_pages: bool
    detect_orientation: bool
    parallel_enabled: bool
    parallel_workers: Optional[int]
    searchable_pdf: bool
    device: str
    use_gpu: bool


class ErrorResponse(BaseModel):
    """Resposta de erro padronizada."""

    detail: str = Field(description="Mensagem de erro")
