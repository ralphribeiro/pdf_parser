# =============================================================================
# Doc Parser API — Docker image with ROCm support (AMD GPU)
#
# Base: Official AMD PyTorch with ROCm 7.2 (supports gfx1100 / RX 7900 XT)
# Workers = 1 because the OCR model is loaded once on the GPU;
# multiple workers would duplicate the model and waste VRAM.
# =============================================================================

FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1

WORKDIR /app

# System dependencies
# poppler-utils: pdf2image (pdftopm)
# tesseract-ocr + tesseract-ocr-por: Tesseract OCR (optional)
# ghostscript: camelot-py dependency
# libgl1-mesa-glx + libglib2.0-0: OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-por \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (torch is already in the base image — do NOT reinstall)
# Deps are declared in pyproject.toml [project.dependencies]; extracted here
# to preserve Docker layer caching (this layer rebuilds only when deps change).
COPY pyproject.toml .
RUN python3 -c "\
import tomllib, subprocess, sys; \
deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; \
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + deps)"

# Application code
COPY config.py .
COPY app/ app/
COPY src/ src/

# API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

# Uvicorn with 1 worker (GPU shared via internal semaphore)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
