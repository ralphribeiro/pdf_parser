"""
Document extraction pipeline configuration.

All settings can be overridden via environment variables
with the DOC_PARSER_ prefix. Example: DOC_PARSER_OCR_ENGINE=tesseract

Default values are kept for backward compatibility with direct usage:
    import config
    config.VERBOSE  # True (or the env var value)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Helpers for env var type conversion
# ---------------------------------------------------------------------------


def _env_bool(key: str, default: bool) -> bool:
    """Read env var as bool. Accepts true/1/yes (case-insensitive)."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    """Read env var as int."""
    val = os.getenv(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Read env var as float."""
    val = os.getenv(key)
    return float(val) if val is not None else default


def _env_int_or_none(key: str, default: int | None) -> int | None:
    """Read env var as int, preserving None as default."""
    val = os.getenv(key)
    if val is None:
        return default
    return int(val) if val else default


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
RESOURCE_DIR = BASE_DIR / "resource"

_output_env = os.getenv("DOC_PARSER_OUTPUT_DIR")
OUTPUT_DIR = Path(_output_env) if _output_env else BASE_DIR / "output"

# ---------------------------------------------------------------------------
# GPU / Device
# ---------------------------------------------------------------------------

_gpu_env = os.getenv("DOC_PARSER_USE_GPU")
if _gpu_env is not None:
    USE_GPU = _gpu_env.lower() in ("true", "1", "yes")
else:
    # Auto-detect: imports torch only when needed
    try:
        import torch

        USE_GPU = torch.cuda.is_available()
    except ImportError:
        USE_GPU = False

DEVICE = "cuda" if USE_GPU else "cpu"

# ---------------------------------------------------------------------------
# OCR - General Settings
# ---------------------------------------------------------------------------

OCR_ENGINE = os.getenv("DOC_PARSER_OCR_ENGINE", "doctr")
OCR_DPI = _env_int("DOC_PARSER_OCR_DPI", 350)
IMAGE_DPI = OCR_DPI  # Alias for compatibility
MIN_CONFIDENCE = _env_float("DOC_PARSER_MIN_CONFIDENCE", 0.3)
OCR_BATCH_SIZE = _env_int("DOC_PARSER_OCR_BATCH_SIZE", 20)

# ---------------------------------------------------------------------------
# OCR - Page Orientation (docTR)
# ---------------------------------------------------------------------------

ASSUME_STRAIGHT_PAGES = _env_bool("DOC_PARSER_ASSUME_STRAIGHT_PAGES", True)
DETECT_ORIENTATION = _env_bool("DOC_PARSER_DETECT_ORIENTATION", False)
STRAIGHTEN_PAGES = _env_bool("DOC_PARSER_STRAIGHTEN_PAGES", False)

# ---------------------------------------------------------------------------
# OCR - Tesseract (if OCR_ENGINE = 'tesseract')
# ---------------------------------------------------------------------------

OCR_LANG = os.getenv("DOC_PARSER_OCR_LANG", "por")
TESSERACT_CONFIG = os.getenv("DOC_PARSER_TESSERACT_CONFIG", "--oem 1 --psm 3")

# ---------------------------------------------------------------------------
# OCR - Post-processing
# ---------------------------------------------------------------------------

OCR_POSTPROCESS = _env_bool("DOC_PARSER_OCR_POSTPROCESS", True)
OCR_FIX_ERRORS = _env_bool("DOC_PARSER_OCR_FIX_ERRORS", True)
OCR_MIN_LINE_LENGTH = _env_int("DOC_PARSER_OCR_MIN_LINE_LENGTH", 3)

# ---------------------------------------------------------------------------
# Image preprocessing (DISABLED - degrades OCR quality)
# docTR and Tesseract handle their own preprocessing internally
# ---------------------------------------------------------------------------

OCR_PREPROCESS = False
BINARIZATION_METHOD = "adaptive"
DENOISE_KERNEL_SIZE = 3
DESKEW_ANGLE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

TABLE_DETECTION_CONFIDENCE = _env_float("DOC_PARSER_TABLE_DETECTION_CONFIDENCE", 0.7)
CAMELOT_FLAVOR = os.getenv("DOC_PARSER_CAMELOT_FLAVOR", "lattice")

# ---------------------------------------------------------------------------
# Page type detection (digital vs scan)
# ---------------------------------------------------------------------------

IMAGE_AREA_THRESHOLD = _env_float("DOC_PARSER_IMAGE_AREA_THRESHOLD", 0.3)
TEXT_COVERAGE_THRESHOLD = _env_float("DOC_PARSER_TEXT_COVERAGE_THRESHOLD", 0.05)

# ---------------------------------------------------------------------------
# Searchable PDF
# ---------------------------------------------------------------------------

SEARCHABLE_PDF = _env_bool("DOC_PARSER_SEARCHABLE_PDF", True)

# ---------------------------------------------------------------------------
# Parallelization
# ---------------------------------------------------------------------------

PARALLEL_ENABLED = _env_bool("DOC_PARSER_PARALLEL_ENABLED", True)
PARALLEL_WORKERS = _env_int_or_none("DOC_PARSER_PARALLEL_WORKERS", None)
PARALLEL_MIN_PAGES = _env_int("DOC_PARSER_PARALLEL_MIN_PAGES", 4)

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

SAVE_PREPROCESSED_IMAGES = _env_bool("DOC_PARSER_SAVE_PREPROCESSED_IMAGES", False)
VERBOSE = _env_bool("DOC_PARSER_VERBOSE", True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("DOC_PARSER_LOG_LEVEL", "").upper()
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_log_level() -> int:
    """
    Determine the effective log level.

    Priority:
    1. DOC_PARSER_LOG_LEVEL (if set): DEBUG, INFO, WARNING, ERROR
    2. VERBOSE=True  -> INFO
    3. VERBOSE=False -> WARNING
    """
    if LOG_LEVEL and hasattr(logging, LOG_LEVEL):
        return getattr(logging, LOG_LEVEL)
    return logging.INFO if VERBOSE else logging.WARNING


def setup_logging() -> None:
    """
    Configure project-wide logging and align uvicorn loggers.

    Ensures that all output (application + uvicorn.error + uvicorn.access)
    uses the same format and level.
    """
    level = _resolve_log_level()
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Root logger (application)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        force=True,
    )

    # Uvicorn loggers — use their own handlers with different format.
    # Override formatter and level to align with the project.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.setLevel(level)
        for handler in uv_logger.handlers:
            handler.setFormatter(formatter)
