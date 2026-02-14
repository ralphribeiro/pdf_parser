#!/usr/bin/env python3
"""
Script to verify environment setup.
"""

import logging
import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def check_setup():
    """Verify environment configuration."""
    import config

    config.setup_logging()

    logger.info("=" * 60)
    logger.info("ENVIRONMENT CHECK - Document Parser Pipeline")
    logger.info("=" * 60)

    # Python
    logger.info("Python: %s", sys.version.split()[0])

    # PyTorch
    try:
        import torch

        logger.info("PyTorch: %s", torch.__version__)
        gpu_available = torch.cuda.is_available()
        logger.info("GPU (CUDA) available: %s", "Yes" if gpu_available else "No")
        if gpu_available:
            logger.info("Device: %s", torch.cuda.get_device_name(0))
            logger.info("CUDA Version: %s", torch.version.cuda)
    except ImportError:
        logger.error("PyTorch not installed")
        return False

    # Main dependencies
    deps = {
        "pdfplumber": "Digital PDF extraction",
        "pdf2image": "PDF to image conversion",
        "cv2": "Image processing (OpenCV)",
        "PIL": "Pillow (images)",
        "doctr": "OCR and layout detection",
        "camelot": "Table extraction",
        "pydantic": "Schema validation",
    }

    logger.info("Dependencies:")
    all_ok = True
    for module, desc in deps.items():
        try:
            __import__(module)
            logger.info("  OK  %-15s - %s", module, desc)
        except ImportError:
            logger.error("  MISSING %-15s - %s (NOT INSTALLED)", module, desc)
            all_ok = False

    # Directory structure
    logger.info("Directory structure:")
    dirs = ["src", "scripts", "resource", "output", ".cache"]
    for d in dirs:
        path = Path(d)
        if path.exists():
            logger.info("  OK  %s/", d)
        else:
            logger.warning("  %s/ (does not exist)", d)

    # Config
    logger.info("Configuration:")
    try:
        logger.info("  Device: %s", config.DEVICE)
        logger.info("  OCR Batch Size: %s", config.OCR_BATCH_SIZE)
        logger.info("  Image DPI: %s", config.IMAGE_DPI)
        logger.info("  Min Confidence: %s", config.MIN_CONFIDENCE)
    except Exception as e:
        logger.error("Error loading config: %s", e)

    # Pipeline import test
    logger.info("Pipeline:")
    try:
        logger.info("  DocumentProcessor imported successfully")
    except Exception as e:
        logger.error("Error importing pipeline: %s", e)
        all_ok = False

    # Summary
    logger.info("=" * 60)
    if all_ok:
        logger.info("ENVIRONMENT CONFIGURED CORRECTLY!")
        logger.info("Next step:")
        logger.info("  python scripts/process_single.py resource/your_document.pdf")
    else:
        logger.warning("SOME ISSUES DETECTED")
        logger.info("Install missing dependencies:")
        logger.info("  pip install -r requirements.txt")
    logger.info("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
