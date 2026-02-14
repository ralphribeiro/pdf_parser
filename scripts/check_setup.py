#!/usr/bin/env python3
"""
Script para verificar a configuração do ambiente
"""
import logging
import sys
from pathlib import Path

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def check_setup():
    """Verifica configuração do ambiente."""
    import config

    config.setup_logging()

    logger.info("=" * 60)
    logger.info("VERIFICAÇÃO DO AMBIENTE - Document Parser Pipeline")
    logger.info("=" * 60)

    # Python
    logger.info("Python: %s", sys.version.split()[0])

    # PyTorch
    try:
        import torch

        logger.info("PyTorch: %s", torch.__version__)
        gpu_available = torch.cuda.is_available()
        logger.info("GPU (CUDA) disponível: %s", "Sim" if gpu_available else "Não")
        if gpu_available:
            logger.info("Device: %s", torch.cuda.get_device_name(0))
            logger.info("CUDA Version: %s", torch.version.cuda)
    except ImportError:
        logger.error("PyTorch não instalado")
        return False

    # Dependências principais
    deps = {
        "pdfplumber": "Extração de PDFs digitais",
        "pdf2image": "Conversão PDF para imagem",
        "cv2": "Processamento de imagem (OpenCV)",
        "PIL": "Pillow (imagens)",
        "doctr": "OCR e layout detection",
        "camelot": "Extração de tabelas",
        "pydantic": "Validação de schemas",
    }

    logger.info("Dependências:")
    all_ok = True
    for module, desc in deps.items():
        try:
            __import__(module)
            logger.info("  OK  %-15s - %s", module, desc)
        except ImportError:
            logger.error("  FALTA %-15s - %s (NÃO INSTALADO)", module, desc)
            all_ok = False

    # Estrutura de diretórios
    logger.info("Estrutura de diretórios:")
    dirs = ["src", "scripts", "resource", "output", ".cache"]
    for d in dirs:
        path = Path(d)
        if path.exists():
            logger.info("  OK  %s/", d)
        else:
            logger.warning("  %s/ (não existe)", d)

    # Config
    logger.info("Configuração:")
    try:
        logger.info("  Device: %s", config.DEVICE)
        logger.info("  OCR Batch Size: %s", config.OCR_BATCH_SIZE)
        logger.info("  Image DPI: %s", config.IMAGE_DPI)
        logger.info("  Min Confidence: %s", config.MIN_CONFIDENCE)
    except Exception as e:
        logger.error("Erro ao carregar config: %s", e)

    # Teste de import do pipeline
    logger.info("Pipeline:")
    try:
        from src.pipeline import DocumentProcessor

        logger.info("  DocumentProcessor importado com sucesso")
    except Exception as e:
        logger.error("Erro ao importar pipeline: %s", e)
        all_ok = False

    # Resumo
    logger.info("=" * 60)
    if all_ok:
        logger.info("AMBIENTE CONFIGURADO CORRETAMENTE!")
        logger.info("Próximo passo:")
        logger.info("  python scripts/process_single.py resource/seu_documento.pdf")
    else:
        logger.warning("ALGUNS PROBLEMAS DETECTADOS")
        logger.info("Instale as dependências faltantes:")
        logger.info("  pip install -r requirements.txt")
    logger.info("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
