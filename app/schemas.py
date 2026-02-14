"""
Pydantic schemas for API request/response.

Separated from pipeline schemas (src/models/schemas.py) to keep
the HTTP layer decoupled from business logic.
"""

from dataclasses import dataclass

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass
class ProcessingOptions:
    """
    Per-request processing options.

    None values mean "use config default".
    """

    extract_tables: bool = True
    min_confidence: float | None = None
    ocr_postprocess: bool | None = None
    ocr_fix_errors: bool | None = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str = Field(description="Service status")
    gpu_available: bool = Field(description="GPU available for OCR")
    ocr_engine: str = Field(description="Loaded OCR engine")
    device: str = Field(description="Device in use (cpu/cuda)")


class InfoResponse(BaseModel):
    """Response for the /info endpoint with current configuration."""

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
    parallel_workers: int | None
    searchable_pdf: bool
    device: str
    use_gpu: bool


class ErrorResponse(BaseModel):
    """Standardized error response."""

    detail: str = Field(description="Error message")
