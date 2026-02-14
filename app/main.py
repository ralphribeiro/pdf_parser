"""
FastAPI application factory e lifespan.

O DocumentProcessor é criado uma única vez no startup (singleton)
e compartilhado entre todas as requests via app.state.
O gpu_semaphore serializa acesso à GPU entre requests concorrentes.

Uso:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""
import asyncio
import gc
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
from app.routers.process import router as process_router

# Configura logging centralizado do projeto antes de qualquer módulo emitir log
config.setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia ciclo de vida da aplicação.

    Startup:  carrega DocumentProcessor com modelo OCR na GPU.
    Shutdown: libera recursos e limpa memória.
    """
    # --- Startup ---
    # Re-aplica logging após uvicorn ter criado seus handlers
    config.setup_logging()

    logger.info("Inicializando DocumentProcessor (device=%s)...", config.DEVICE)

    from src.pipeline import DocumentProcessor

    processor = DocumentProcessor(use_gpu=config.USE_GPU)
    app.state.processor = processor
    app.state.gpu_semaphore = asyncio.Semaphore(1)

    logger.info(
        "Processor pronto: engine=%s, device=%s",
        processor.ocr_engine_type,
        config.DEVICE,
    )

    yield

    # --- Shutdown ---
    logger.info("Encerrando: liberando recursos...")
    app.state.processor.ocr_engine = None
    app.state.processor.tesseract_engine = None
    app.state.processor = None
    gc.collect()
    logger.info("Shutdown completo.")


def create_app() -> FastAPI:
    """
    Factory function para criar a aplicação FastAPI.

    Separada do módulo-level `app` para facilitar testes
    (TestClient pode criar instâncias com lifespan controlado).
    """
    application = FastAPI(
        title="Doc Parser API",
        description="API para extração de texto e tabelas de PDFs via OCR",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.include_router(process_router)

    return application


# Instância global para uvicorn
app = create_app()
