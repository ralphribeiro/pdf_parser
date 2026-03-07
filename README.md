# Document Parser Pipeline

Local pipeline for text and structure extraction from mixed PDFs (digital and scanned), with JSON output and searchable PDF generation. Supports GPU-accelerated OCR via ROCm (AMD) or CUDA (NVIDIA).

## Features

- **Digital PDFs**: Direct text extraction with `pdfplumber`
- **Scanned PDFs**: OCR with `docTR` (PyTorch + GPU) or `Tesseract` (LSTM, CPU)
- **Tables**: Detection and extraction with `camelot-py`
- **Searchable PDFs**: Invisible text overlay on scanned pages for search/selection
- **Parallel processing**: Concurrent page processing with `ProcessPoolExecutor`
- **REST API**: FastAPI endpoints for HTTP-based processing
- **Docker**: Ready-to-deploy container with ROCm support
- **Positioning**: Normalized bounding box coordinates (0-1) for each block
- **OCR post-processing**: Noise removal, error correction, broken word merging
- **Environment-based configuration**: All settings via `DOC_PARSER_*` env vars

## Quick Start

### Docker (recommended)

```bash
docker compose up --build

# Process a PDF
curl -X POST http://localhost:8000/process \
  -F "file=@document.pdf" \
  -o result.json

# Get a searchable PDF back
curl -X POST "http://localhost:8000/process?response_format=pdf" \
  -F "file=@document.pdf" \
  -o searchable.pdf

# Health check
curl http://localhost:8000/health
```

### Local Installation

#### Prerequisites

- Python 3.10+
- PyTorch with ROCm (AMD GPU) or CUDA (NVIDIA GPU)
- Ghostscript and Poppler (for camelot and pdf2image)
- Tesseract (optional, for Tesseract OCR engine)

```bash
# Ubuntu/Debian
sudo apt-get install ghostscript poppler-utils tesseract-ocr tesseract-ocr-por

# macOS
brew install ghostscript poppler tesseract tesseract-lang
```

#### Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with ROCm support (skip if using Docker)
pip install --index-url https://download.pytorch.org/whl/rocm7.0 ".[torch-rocm]"

# Install project dependencies
pip install .

# Install with dev tools (linting, testing)
pip install ".[dev]"
```

## Usage

### REST API

Start the API server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note:** Use `--workers 1` because the OCR model is loaded once on the GPU; multiple workers would duplicate the model and waste VRAM.

#### Endpoints

| Method | Endpoint   | Description                               |
|--------|------------|-------------------------------------------|
| POST   | `/process` | Process a PDF and return JSON or searchable PDF |
| GET    | `/health`  | Service health check (status, GPU, OCR engine) |
| GET    | `/info`    | Current pipeline configuration             |

#### `POST /process` Parameters

| Parameter        | Type   | Default | Description                          |
|------------------|--------|---------|--------------------------------------|
| `file`           | file   | required | PDF file to process                 |
| `response_format`| string | `json`  | `json` or `pdf` (searchable PDF)    |
| `extract_tables` | bool   | `true`  | Enable table extraction              |
| `min_confidence` | float  | —       | Minimum OCR confidence (0.0–1.0)    |
| `ocr_postprocess`| bool   | —       | Enable OCR text post-processing     |
| `ocr_fix_errors` | bool   | —       | Fix common OCR errors               |

### CLI

```bash
# Process a single PDF
python scripts/process_single.py document.pdf

# Specify output directory
python scripts/process_single.py document.pdf -o /path/to/output

# Disable table extraction
python scripts/process_single.py document.pdf --no-tables

# Force CPU usage
python scripts/process_single.py document.pdf --no-gpu

# Use Tesseract instead of docTR
python scripts/process_single.py document.pdf --ocr-engine tesseract

# Generate searchable PDF
python scripts/process_single.py document.pdf --searchable-pdf

# Quiet mode
python scripts/process_single.py document.pdf --quiet
```

### Programmatic Usage

```python
from src.pipeline import DocumentProcessor

processor = DocumentProcessor(use_gpu=True)
doc = processor.process_document("document.pdf")

print(f"Total pages: {doc.total_pages}")
print(f"Total blocks: {sum(len(p.blocks) for p in doc.pages)}")

# Save JSON output
processor.save_to_json(doc, "output.json", indent=4)

# Save searchable PDF
processor.save_to_searchable_pdf(doc, "document.pdf", "searchable_output.pdf")
```

## JSON Output Structure

```json
{
  "doc_id": "document-name",
  "source_file": "document.pdf",
  "total_pages": 10,
  "processing_date": "2026-02-14T12:00:00Z",
  "pages": [
    {
      "page": 1,
      "source": "digital",
      "blocks": [
        {
          "block_id": "p1_b1",
          "type": "paragraph",
          "text": "Paragraph content...",
          "bbox": [0.1, 0.2, 0.9, 0.3],
          "confidence": 1.0
        }
      ]
    }
  ]
}
```

### Block Types

| Type        | Description                              |
|-------------|------------------------------------------|
| `paragraph` | Text paragraphs                          |
| `table`     | Tables (includes `rows` field with data) |
| `header`    | Headers                                  |
| `footer`    | Footers                                  |
| `list`      | Lists                                    |
| `image`     | Images (placeholder for future use)      |

### BBox Coordinates

All coordinates are normalized (0–1) in the format `[x1, y1, x2, y2]`:
- `x1, y1`: Top-left corner
- `x2, y2`: Bottom-right corner

## Configuration

All settings can be overridden via environment variables with the `DOC_PARSER_` prefix. Create a `.env` file from the example:

```bash
cp .env.example .env
```

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DOC_PARSER_USE_GPU` | auto-detected | Use GPU for OCR |
| `DOC_PARSER_OCR_ENGINE` | `doctr` | OCR engine: `doctr` or `tesseract` |
| `DOC_PARSER_OCR_DPI` | `350` | Resolution for PDF-to-image conversion |
| `DOC_PARSER_OCR_BATCH_SIZE` | `20` | Batch size for docTR (adjust per VRAM) |
| `DOC_PARSER_MIN_CONFIDENCE` | `0.3` | Minimum confidence to accept OCR result |
| `DOC_PARSER_OCR_LANG` | `por` | Tesseract language code |
| `DOC_PARSER_CAMELOT_FLAVOR` | `lattice` | Table detection flavor: `lattice` or `stream` |
| `DOC_PARSER_SEARCHABLE_PDF` | `true` | Generate searchable PDF by default |
| `DOC_PARSER_PARALLEL_ENABLED` | `true` | Enable parallel page processing |
| `DOC_PARSER_PARALLEL_WORKERS` | auto | Number of parallel workers |
| `DOC_PARSER_OCR_POSTPROCESS` | `true` | Enable OCR text post-processing |
| `DOC_PARSER_OCR_FIX_ERRORS` | `true` | Fix common OCR errors |
| `DOC_PARSER_VERBOSE` | `true` | Verbose logging output |
| `DOC_PARSER_OUTPUT_DIR` | `./output` | Default output directory |
| `DOC_PARSER_CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker URL |
| `DOC_PARSER_CELERY_RESULT_BACKEND` | `redis://localhost:6379/0` | Celery result backend |
| `DOC_PARSER_CELERY_WORKERS` | `2` | Number of Celery workers |
| `DOC_PARSER_MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection URI |
| `DOC_PARSER_MONGODB_DB` | `caseiro_docs` | MongoDB database name |
| `DOC_PARSER_MONGODB_USE_VECTORS` | `true` | Enable MongoDB vector search |
| `DOC_PARSER_EMBEDDINGS_URL` | `http://localhost:11434/api/generate` | Embeddings API URL |
| `DOC_PARSER_EMBEDDINGS_MODEL` | `nomic-embed-text` | Embeddings model name |
| `DOC_PARSER_WEBHOOK_URL` | `` | Webhook URL for job notifications |

See `config.py` for the full list of available settings.

## Async Job Processing

Processamento assíncrono usando Celery, Redis e MongoDB para jobs longos (PDFs com OCR).

### Endpoints de Jobs

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/jobs` | Criar novo job de processamento |
| GET | `/jobs/{job_id}` | Obter status e resultado |
| DELETE | `/jobs/{job_id}` | Cancelar job pendente |
| GET | `/jobs` | Listar jobs (debug) |

### Busca Semântica

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/search/semantic` | Buscar documentos por semelhança vetorial |
| GET | `/search/semantic` | Buscar com query string |

### Exemplo de Uso

#### Criar Job

```bash
curl -X POST http://localhost:8000/jobs \
  -F "file=@document.pdf" \
  -F "generate_embeddings=true"
```

**Resposta:**
```json
{
  "job_id": "job_abc123",
  "status": "pending",
  "created_at": "2026-03-04T12:00:00Z",
  "file_size": 12345678
}
```

#### Consultar Status

```bash
curl http://localhost:8000/jobs/job_abc123
```

**Resposta (pendente):**
```json
{
  "job_id": "job_abc123",
  "status": "processing",
  "created_at": "2026-03-04T12:00:00Z",
  "updated_at": "2026-03-04T12:00:01Z"
}
```

**Resposta (completo):**
```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "created_at": "2026-03-04T12:00:00Z",
  "updated_at": "2026-03-04T12:05:00Z",
  "result": {
    "doc_id": "doc-uuid",
    "total_pages": 10,
    "processing_time_seconds": 300.5
  },
  "embeddings_generated": true,
  "error": null
}
```

#### Busca Semântica

```bash
curl -X POST http://localhost:8000/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Qual é o valor da cláusula de rescisão?",
    "top_k": 5,
    "min_score": 0.7,
    "include_matches": true,
    "matches_limit": 3
  }'
```

**Resposta:**
```json
{
  "query": "Qual é o valor da cláusula de rescisão?",
  "total_results": 2,
  "results": [
    {
      "doc_id": "contrato-2024",
      "score": 0.92,
      "total_pages": 45,
      "created_at": "2026-03-04T10:00:00Z",
      "matches": [
        {
          "page": 12,
          "block_id": "p12_b3",
          "text": "A cláusula de rescisão contratual tem o valor de R$ 50.000,00.",
          "bbox": [0.1, 0.2, 0.8, 0.25]
        }
      ]
    }
  ]
}
```

### Iniciar Celery Workers

```bash
# Iniciar workers (modo solo para debug)
python scripts/celery_worker.py worker --workers 2

# Iniciar com monitoramento Flower
python scripts/celery_worker.py worker --workers 4 --with-flower

# Acessar Flower: http://localhost:5555
```

### Iniciar com Docker

```bash
docker compose up --build

# Iniciar workers separadamente
docker compose up redis mongodb

# Iniciar workers em background
celery -A src.celery_worker worker --loglevel=info --concurrency=2 -Q default
```

## Architecture

```
doc_parser/
├── app/                          # FastAPI application
│   ├── main.py                   # App factory and lifespan
│   ├── dependencies.py           # Dependency injection
│   ├── schemas.py                # API request/response schemas
│   └── routers/
│       └── process.py            # API endpoints
├── src/                          # Core processing pipeline
│   ├── pipeline.py               # Main orchestrator (sequential + parallel)
│   ├── detector.py               # Page type detection (digital/scan/hybrid)
│   ├── extractors/
│   │   ├── digital.py            # Digital PDF extraction (pdfplumber)
│   │   ├── ocr.py                # OCR with docTR (PyTorch)
│   │   ├── ocr_tesseract.py      # OCR with Tesseract (LSTM)
│   │   └── tables.py             # Table extraction (camelot)
│   ├── preprocessing/
│   │   └── image_enhancer.py     # Image preprocessing for OCR
│   ├── models/
│   │   └── schemas.py            # Pydantic data models
│   ├── exporters/
│   │   └── searchable_pdf.py     # Searchable PDF generation
│   └── utils/
│       ├── bbox.py               # Bounding box manipulation
│       ├── text_normalizer.py    # Text cleanup and normalization
│       └── ocr_postprocess.py    # OCR text post-processing
├── scripts/
│   ├── process_single.py         # CLI for single PDF processing
│   └── check_setup.py            # Environment verification
├── tests/
│   ├── conftest.py               # Shared test fixtures
│   ├── test_api.py               # API endpoint tests
│   ├── test_config.py            # Configuration tests
│   └── test_searchable_pdf.py    # Searchable PDF tests
├── config.py                     # Global configuration
├── pyproject.toml                # Dependencies, build config, tool settings
├── Dockerfile                    # Docker image (ROCm)
└── docker-compose.yml            # Docker Compose setup
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test module
pytest tests/test_api.py
```

## Docker

The Docker image is based on the official AMD ROCm PyTorch image and includes all system dependencies.

```bash
# Build and start
docker compose up --build

# Run in background
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

**Host requirements for GPU support:**
- AMD GPU with ROCm support (e.g., RX 7900 XT)
- `amdgpu` driver + ROCm installed
- User in `video` and `render` groups

## Performance

Tested with a 353-page PDF (27 MB) (248 digital pages, 105 OCR pages, JSON output: 591 blocks, 27 tables):
- **Time**: ~5.5 minutes (AMD Ryzen CPU + RX 7900 XT GPU)
- **Average speed**: ~1.2 pages/second
- **JSON output**: 754 KB (591 blocks, 27 tables)

## Current Limitations

- Signatures: not yet handled
- Advanced layout detection (columns, sections)
- Form recognition

## Roadmap

1. Improved table detection in scanned PDFs
2. Signature detection and classification
3. Smart chunking for embeddings
4. Vectorization pipeline (embeddings)
5. Batch processing of multiple PDFs

## License

Internal project for legal document analysis.
