# =============================================================================
# Doc Parser API — Docker image com suporte ROCm (AMD GPU)
#
# Base: PyTorch oficial da AMD com ROCm 7.2 (suporta gfx1100 / RX 7900 XT)
# Workers = 1 porque o modelo OCR é carregado uma vez na GPU;
# múltiplos workers duplicariam o modelo e desperdiçariam VRAM.
# =============================================================================

FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1

WORKDIR /app

# Dependências de sistema
# poppler-utils: pdf2image (pdftopm)
# tesseract-ocr + tesseract-ocr-por: OCR Tesseract (opcional)
# ghostscript: dependência do camelot-py
# libgl1-mesa-glx + libglib2.0-0: OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-por \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dependências Python (torch já está na imagem base — NÃO reinstalar)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código da aplicação
COPY config.py .
COPY app/ app/
COPY src/ src/

# Porta da API
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

# Uvicorn com 1 worker (GPU compartilhada via semáforo interno)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
