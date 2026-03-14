# =============================================================================
# Doc Parser — single image for API, UI, and worker
#
# Base: Official AMD PyTorch with ROCm 7.2 (gfx1100 / RX 7900 XT)
#
# This image is used by both the `api` and `worker` services defined in
# docker-compose.yml — the CMD is overridden per service.
#
# Build:  docker compose -f docker-compose.yml build
# =============================================================================

FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1

WORKDIR /app

# System dependencies (poppler, tesseract, ghostscript, opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-por \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (torch already in base image — do NOT reinstall)
COPY pyproject.toml .
RUN python3 -c "\
import tomllib, subprocess, sys; \
deps = tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']; \
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + deps)"

# Application code
COPY config.py .
COPY src/ src/
COPY services/ services/

RUN mkdir -p /app/data /app/output
VOLUME ["/app/data", "/app/output"]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/api/jobs/healthcheck'); r.raise_for_status(); print('ok')" || exit 1

CMD ["uvicorn", "services.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
