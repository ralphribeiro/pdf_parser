# Document Parser Pipeline

Local pipeline for text and structure extraction from mixed PDFs (digital and scanned), with JSON output and searchable PDF generation. Supports GPU-accelerated OCR via ROCm (AMD) or CUDA (NVIDIA).

## Features

- **Digital PDFs**: Direct text extraction with `pdfplumber`
- **Scanned PDFs**: OCR with `docTR` (PyTorch + GPU) or `Tesseract` (LSTM, CPU)
- **Tables**: Detection and extraction with `camelot-py`
- **Searchable PDFs**: Invisible text overlay on scanned pages for search/selection
- **Parallel processing**: Concurrent page processing with `ProcessPoolExecutor`
- **REST API**: FastAPI endpoints for HTTP-based processing
- **Async ingest API**: job queue/status API (`/api/jobs`) with worker processing
- **Redis-backed job store**: shared state between API and worker containers
- **MongoDB document store**: parsed documents via `GET /api/documents`
- **Semantic search**: chunk indexing in ChromaDB + `POST /api/search`
- **Agent search**: enriched Q&A via `POST /api/agent/search` (requires `LLM_API_URL`)
- **Docker**: Ready-to-deploy container with ROCm support
- **Positioning**: Normalized bounding box coordinates (0-1) for each block
- **OCR post-processing**: Noise removal, error correction, broken word merging
- **Environment-based configuration**: All settings via `DOC_PARSER_*` env vars

## Quick Start

### Docker (Async services with Redis + ChromaDB) — recommended

```bash
docker compose up --build -d

# Health check
curl http://localhost:8090/api/jobs/healthcheck

# 1) Create async job
JOB_ID=$(
  curl -sS -X POST http://localhost:8090/api/jobs \
    -F "file=@document.pdf;type=application/pdf" | jq -r '.job_id'
)

# 2) Poll job status (includes document_id when MongoDB is configured)
curl -sS "http://localhost:8090/api/jobs/$JOB_ID" | jq
DOC_ID=$(curl -sS "http://localhost:8090/api/jobs/$JOB_ID" | jq -r '.document_id')

# 3) Semantic search (optionally scoped by processed document_id)
curl -sS -X POST http://localhost:8090/api/search \
  -H "content-type: application/json" \
  -d "{\"query\":\"contrato de locacao\",\"n_results\":5,\"document_id\":\"$DOC_ID\"}" | jq

# 4) Fetch parsed document (after job status is uploaded)
curl -sS "http://localhost:8090/api/documents/$DOC_ID" | jq

# 5) Agent search scoped to the same processed document
curl -sS -X POST http://localhost:8090/api/agent/search \
  -H "content-type: application/json" \
  -d "{\"query\":\"resuma as obrigacoes principais\",\"document_id\":\"$DOC_ID\"}" | jq
```

> **Ports:** Docker maps container port 8080 to host **8090** (`localhost:8090`). Local `uvicorn` uses **8080** directly.

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

Start the combined app:

```bash
uvicorn services.app:app --host 0.0.0.0 --port 8080 --workers 1
```

> Note: use `--workers 1` when OCR model is GPU-loaded to avoid duplicated VRAM usage.

#### REST API endpoints

Interactive OpenAPI docs: `/api/docs` and `/api/redoc` (same host/port as the app).

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/jobs/healthcheck` | Ingest API healthcheck |
| GET | `/api/jobs` | List jobs (`limit`, `offset`; default 20/0) |
| POST | `/api/jobs` | Upload PDF and create job (**202**; **409** if duplicate) |
| GET | `/api/jobs/{job_id}` | Get job status |
| GET | `/api/documents` | List documents (**503** without MongoDB) |
| GET | `/api/documents/{document_id}` | Parsed document from MongoDB |
| POST | `/api/search` | Semantic search over indexed chunks; optional processed `document_id` scope |
| POST | `/api/agent/search` | Agent-based enriched search; optional processed `document_id` scope (**503** without `LLM_API_URL`) |

#### Web UI routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/`, `/upload`, `/jobs`, `/documents`, `/search`, `/agent` | HTML pages |
| POST | `/upload` | Form upload → redirect to `/jobs/{job_id}` |
| GET | `/jobs/{job_id}`, `/documents/{document_id}` | Detail pages |

#### `POST /api/search` payload

When `document_id` is provided, it must identify an existing document with
`status="processed"`. The API returns **404** for unknown documents and **409**
for documents that are still pending, processing, or failed.

```bash
curl -X POST http://localhost:8080/api/search \
  -H "content-type: application/json" \
  -d '{
    "query": "texto da consulta",
    "n_results": 10,
    "document_id": "optional-processed-document-id",
    "min_similarity": 0.7
  }'
```

#### `POST /api/agent/search` payload

The same `document_id` rule applies here. When set, the backend enforces that
all agent tools stay scoped to that document, even if the model omits or changes
the document id in a tool call.

```bash
curl -X POST http://localhost:8080/api/agent/search \
  -H "content-type: application/json" \
  -d '{
    "query": "What are the main contract terms?",
    "document_id": "optional-processed-document-id",
    "max_iterations": 5
  }'
```

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
| `DOC_PARSER_REDIS_URL` | empty | Redis URL for async job store |

**Async services** (no `DOC_PARSER_` prefix; see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | empty | Redis URL (Docker: `redis://redis:6379/0`) |
| `MONGO_URL` | empty | MongoDB URI (Docker: `mongodb://mongodb:27017`) |
| `MONGO_DB` | `doc_parser` | MongoDB database name |
| `CHROMA_HOST` / `DOC_PARSER_CHROMA_HOST` | empty | ChromaDB host URL (e.g. `http://chromadb:8000`) |
| `CHROMA_COLLECTION` / `DOC_PARSER_CHROMA_COLLECTION` | `document_embeddings` | Chroma collection name |
| `EMBEDDING_API_URL` / `DOC_PARSER_EMBEDDING_API_URL` | empty | Remote embedding API base URL |
| `EMBEDDING_MODEL` / `DOC_PARSER_EMBEDDING_MODEL` | `Qwen3-Embedding` | Embedding model name |
| `EMBEDDING_API_KEY` | empty | Embedding API key |
| `EMBEDDING_TIMEOUT_SECONDS` / `DOC_PARSER_EMBEDDING_TIMEOUT_SECONDS` | `10` | Embedding API timeout |
| `LLM_API_URL` | empty | LLM API URL (enables agent search when set) |
| `LLM_MODEL` | `Qwen3.5-9B-Q4_K_M` | LLM model name |
| `LLM_API_KEY` | empty | LLM API key |

See `config.py` for the full list of pipeline settings.

## Architecture

```
doc_parser/
├── services/                     # Async API, UI, worker, search, agent
│   ├── app.py                    # Combined FastAPI app factory (API at /api)
│   ├── document_store.py         # MongoDB document persistence
│   ├── ingest_api/               # REST API (/api/jobs, /api/search, /api/agent)
│   ├── ingest_ui/                # HTML UI (/, /jobs, /documents, /search, /agent)
│   ├── worker/                   # OCR worker polling Redis
│   ├── search/                   # Semantic search (ChromaDB + embeddings)
│   └── agent/                    # ReAct agent for enriched search
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
├── tests/                        # Test suite
├── config.py                     # Global configuration
├── pyproject.toml                # Dependencies, build config, tool settings
├── Dockerfile                    # Docker image (ROCm)
└── docker-compose.yml   # Docker Compose (Redis + ChromaDB + API + Worker)
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v
```

## Docker

The Docker image is based on the official AMD ROCm PyTorch image and includes all system dependencies.

```bash
docker compose up --build

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

**Done:**

- Vectorization pipeline (embeddings via ChromaDB)
- Async job processing (Redis + worker)
- MongoDB document persistence
- Semantic search over indexed chunks
- AI agent for enriched search (ReAct loop)

**Planned:**

1. Improved table detection in scanned PDFs
2. Signature detection and classification
3. Batch processing of multiple PDFs

## License

Internal project for legal document analysis.
