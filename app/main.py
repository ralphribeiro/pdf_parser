"""
FastAPI application factory and lifespan.

The DocumentProcessor is created once at startup (singleton)
and shared across all requests via app.state.
The gpu_semaphore serializes GPU access between concurrent requests.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
import gc
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
from app.routers.process import router as process_router

# Configure centralized project logging before any module emits logs
config.setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    Startup:  loads DocumentProcessor with OCR model on GPU.
    Shutdown: releases resources and cleans up memory.
    """
    # --- Startup ---
    # Re-apply logging after uvicorn has created its handlers
    config.setup_logging()

    logger.info("Initializing DocumentProcessor (device=%s)...", config.DEVICE)

    from src.pipeline import DocumentProcessor

    processor = DocumentProcessor(use_gpu=config.USE_GPU)
    app.state.processor = processor
    app.state.gpu_semaphore = asyncio.Semaphore(1)

    logger.info(
        "Processor ready: engine=%s, device=%s",
        processor.ocr_engine_type,
        config.DEVICE,
    )

    yield

    # --- Shutdown ---
    logger.info("Shutting down: releasing resources...")
    app.state.processor.ocr_engine = None
    app.state.processor.tesseract_engine = None
    app.state.processor = None
    gc.collect()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """
    Factory function to create the FastAPI application.

    Separated from module-level `app` to facilitate testing
    (TestClient can create instances with controlled lifespan).
    """
    application = FastAPI(
        title="Doc Parser API",
        description="API for text and table extraction from PDFs via OCR",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.include_router(process_router)

    return application


# Global instance for uvicorn
app = create_app()
