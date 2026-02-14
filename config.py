"""
Configurações do pipeline de extração de documentos.

Todas as configurações podem ser sobrescritas via variáveis de ambiente
com prefixo DOC_PARSER_. Exemplo: DOC_PARSER_OCR_ENGINE=tesseract

Valores default são mantidos para backward-compat com uso direto:
    import config
    config.VERBOSE  # True (ou o valor da env var)
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Helpers para conversão de tipos de env vars
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    """Lê env var como bool. Aceita true/1/yes (case-insensitive)."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    """Lê env var como int."""
    val = os.getenv(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Lê env var como float."""
    val = os.getenv(key)
    return float(val) if val is not None else default


def _env_int_or_none(key: str, default: int | None) -> int | None:
    """Lê env var como int, preservando None como default."""
    val = os.getenv(key)
    if val is None:
        return default
    return int(val) if val else default


# ---------------------------------------------------------------------------
# Diretórios
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
    # Auto-detect: importa torch apenas quando necessário
    try:
        import torch
        USE_GPU = torch.cuda.is_available()
    except ImportError:
        USE_GPU = False

DEVICE = "cuda" if USE_GPU else "cpu"

# ---------------------------------------------------------------------------
# OCR - Configurações Gerais
# ---------------------------------------------------------------------------

OCR_ENGINE = os.getenv("DOC_PARSER_OCR_ENGINE", "doctr")
OCR_DPI = _env_int("DOC_PARSER_OCR_DPI", 350)
IMAGE_DPI = OCR_DPI  # Alias para compatibilidade
MIN_CONFIDENCE = _env_float("DOC_PARSER_MIN_CONFIDENCE", 0.3)
OCR_BATCH_SIZE = _env_int("DOC_PARSER_OCR_BATCH_SIZE", 20)

# ---------------------------------------------------------------------------
# OCR - Orientação de página (docTR)
# ---------------------------------------------------------------------------

ASSUME_STRAIGHT_PAGES = _env_bool("DOC_PARSER_ASSUME_STRAIGHT_PAGES", True)
DETECT_ORIENTATION = _env_bool("DOC_PARSER_DETECT_ORIENTATION", False)
STRAIGHTEN_PAGES = _env_bool("DOC_PARSER_STRAIGHTEN_PAGES", False)

# ---------------------------------------------------------------------------
# OCR - Tesseract (se OCR_ENGINE = 'tesseract')
# ---------------------------------------------------------------------------

OCR_LANG = os.getenv("DOC_PARSER_OCR_LANG", "por")
TESSERACT_CONFIG = os.getenv(
    "DOC_PARSER_TESSERACT_CONFIG", "--oem 1 --psm 3"
)

# ---------------------------------------------------------------------------
# OCR - Pós-processamento
# ---------------------------------------------------------------------------

OCR_POSTPROCESS = _env_bool("DOC_PARSER_OCR_POSTPROCESS", True)
OCR_FIX_ERRORS = _env_bool("DOC_PARSER_OCR_FIX_ERRORS", True)
OCR_MIN_LINE_LENGTH = _env_int("DOC_PARSER_OCR_MIN_LINE_LENGTH", 3)

# ---------------------------------------------------------------------------
# Pré-processamento de imagem (DESATIVADO - degrada qualidade do OCR)
# O docTR e Tesseract fazem seu próprio pré-processamento internamente
# ---------------------------------------------------------------------------

OCR_PREPROCESS = False
BINARIZATION_METHOD = "adaptive"
DENOISE_KERNEL_SIZE = 3
DESKEW_ANGLE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

TABLE_DETECTION_CONFIDENCE = _env_float(
    "DOC_PARSER_TABLE_DETECTION_CONFIDENCE", 0.7
)
CAMELOT_FLAVOR = os.getenv("DOC_PARSER_CAMELOT_FLAVOR", "lattice")

# ---------------------------------------------------------------------------
# Detecção de tipo de página (digital vs scan)
# ---------------------------------------------------------------------------

IMAGE_AREA_THRESHOLD = _env_float("DOC_PARSER_IMAGE_AREA_THRESHOLD", 0.3)
TEXT_COVERAGE_THRESHOLD = _env_float(
    "DOC_PARSER_TEXT_COVERAGE_THRESHOLD", 0.05
)

# ---------------------------------------------------------------------------
# PDF pesquisável (searchable PDF)
# ---------------------------------------------------------------------------

SEARCHABLE_PDF = _env_bool("DOC_PARSER_SEARCHABLE_PDF", True)

# ---------------------------------------------------------------------------
# Paralelização
# ---------------------------------------------------------------------------

PARALLEL_ENABLED = _env_bool("DOC_PARSER_PARALLEL_ENABLED", True)
PARALLEL_WORKERS = _env_int_or_none("DOC_PARSER_PARALLEL_WORKERS", None)
PARALLEL_MIN_PAGES = _env_int("DOC_PARSER_PARALLEL_MIN_PAGES", 4)

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

SAVE_PREPROCESSED_IMAGES = _env_bool(
    "DOC_PARSER_SAVE_PREPROCESSED_IMAGES", False
)
VERBOSE = _env_bool("DOC_PARSER_VERBOSE", True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("DOC_PARSER_LOG_LEVEL", "").upper()
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """
    Configura logging padrão do projeto.

    Nível é determinado por:
    1. DOC_PARSER_LOG_LEVEL (se definido): DEBUG, INFO, WARNING, ERROR
    2. VERBOSE=True  → INFO
    3. VERBOSE=False → WARNING
    """
    if LOG_LEVEL and hasattr(logging, LOG_LEVEL):
        level = getattr(logging, LOG_LEVEL)
    elif VERBOSE:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        force=True,
    )
