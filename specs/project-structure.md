# CLAUDE.md - Doc Parser Pipeline

## Visão Geral

Este é um pipeline de processamento de documentos PDF para extração de texto e tabelas de PDFs mistos (digitais e digitalizados/escaneados), com saída em JSON e geração de PDFs pesquisáveis.

## Arquitetura

```
doc_parser/
├── services/                     # API assíncrona, UI, worker, search, agent
│   ├── app.py                    # App factory combinada (API + UI + Agent)
│   ├── document_store.py         # MongoDB: persistência de documentos
│   ├── ingest_api/               # API de jobs (/api/jobs, /api/search, /api/agent)
│   ├── ingest_ui/                # UI de upload e status (/, /jobs/{id})
│   ├── worker/                   # Worker OCR polling Redis
│   ├── search/                   # Busca semântica (ChromaDB + embeddings)
│   └── agent/                    # Agente AI de busca enriquecida
│       ├── agent.py              # Loop ReAct com budget tracking
│       ├── llm_client.py         # Client HTTP para /v1/chat/completions
│       ├── tools.py              # Ferramentas do agente (search, get_doc, etc.)
│       └── prompts.py            # System prompt e constantes
├── src/                          # Pipeline de processamento principal
│   ├── pipeline.py               # Orquestrador principal (sequencial + paralela)
│   ├── detector.py               # Detecção de tipo de página (digital/scan/hybrid)
│   ├── extractors/
│   │   ├── digital.py            # Extração de PDF digital (pdfplumber)
│   │   ├── ocr.py                # OCR com docTR (PyTorch)
│   │   ├── ocr_tesseract.py      # OCR com Tesseract (LSTM, CPU)
│   │   └── tables.py             # Extração de tabelas (camelot)
│   ├── exporters/
│   │   └── searchable_pdf.py     # Geração de PDF pesquisável
│   ├── models/
│   │   └── schemas.py            # Modelos Pydantic
│   ├── preprocessing/
│   │   └── image_enhancer.py     # Pré-processamento de imagem
│   └── utils/
│       ├── bbox.py               # Manipulação de bounding boxes
│       ├── text_normalizer.py    # Limpeza e normalização de texto
│       └── ocr_postprocess.py    # Pós-processamento de texto OCR
├── scripts/
│   ├── process_single.py         # CLI para processamento de PDF único
│   └── llm_postprocess.py        # Pós-processamento de OCR via LLM (Ollama)
├── config.py                     # Configuração global
├── pyproject.toml                # Dependências e configurações
├── docker-compose.yml   # Docker Compose (Redis + MongoDB + ChromaDB + API + Worker)
└── tests/                        # Suite de testes
```

## Funcionalidades Principais

### Extração de PDF
- **PDFs Digitais**: Extração direta de texto com `pdfplumber`
- **PDFs Digitalizados**: OCR com `docTR` (PyTorch + GPU) ou `Tesseract` (LSTM, CPU)
- **Tabelas**: Detecção e extração com `camelot-py`
- **PDFs Pesquisáveis**: Sobreposição de texto invisível em páginas escaneadas
- **Processamento Paralelo**: Processamento concorrente de páginas com `ProcessPoolExecutor`

### OCR
- **docTR**: Deep learning com PyTorch, rápido com GPU (ROCm/CUDA)
- **Tesseract**: Tradicional LSTM, bom para português
- **Configuração de orientação**: Detecção automática de rotação e correção

### Pós-processamento de OCR
- Remoção de ruído e caracteres inválidos
- Correção de pontuação e espaçamento
- Correção de erros comuns (RN, Nº, n2, etc.)
- Fusão de palavras quebradas
- Remoção de linhas curtas (ruído)

### Ordenação por Leitura
- Algoritmo baseado em "bands" (linhas horizontais)
- Detecção de layouts multicolumna
- Normalização de coordenadas (0-1)

## Configuração

### Variáveis de Ambiente

Todas as configurações são controladas via variáveis de ambiente com prefixo `DOC_PARSER_`:

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `DOC_PARSER_USE_GPU` | auto-detected | Usar GPU para OCR |
| `DOC_PARSER_OCR_ENGINE` | `doctr` | Motor OCR: `doctr` ou `tesseract` |
| `DOC_PARSER_OCR_DPI` | `350` | Resolução PDF-to-image |
| `DOC_PARSER_OCR_BATCH_SIZE` | `20` | Batch size para docTR |
| `DOC_PARSER_MIN_CONFIDENCE` | `0.3` | Confiança mínima OCR |
| `DOC_PARSER_OCR_LANG` | `por` | Código de idioma Tesseract |
| `DOC_PARSER_CAMELOT_FLAVOR` | `lattice` | Flavor de detecção de tabelas |
| `DOC_PARSER_SEARCHABLE_PDF` | `true` | Gerar PDF pesquisável |
| `DOC_PARSER_PARALLEL_ENABLED` | `true` | Habilitar processamento paralelo |
| `DOC_PARSER_PARALLEL_WORKERS` | auto | Número de workers paralelos |
| `DOC_PARSER_OCR_POSTPROCESS` | `true` | Habilitar pós-processamento OCR |
| `DOC_PARSER_OCR_FIX_ERRORS` | `true` | Corrigir erros comuns OCR |
| `DOC_PARSER_VERBOSE` | `true` | Logs verbosos |
| `DOC_PARSER_OUTPUT_DIR` | `./output` | Diretório de saída padrão |
| `DOC_PARSER_ASSUME_STRAIGHT_PAGES` | `false` | Assumir páginas retas |
| `DOC_PARSER_DETECT_ORIENTATION` | `true` | Detectar orientação |
| `DOC_PARSER_STRAIGHTEN_PAGES` | `true` | Corrigir inclinação |

**Serviços externos e persistência:**

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `MONGO_URL` | `mongodb://mongodb:27017` | URI do MongoDB |
| `MONGO_DB` | `doc_parser` | Nome do banco MongoDB |
| `EMBEDDING_API_URL` | (obrigatória) | URL da API de embeddings (llama.cpp) |
| `EMBEDDING_MODEL` | `Qwen3-Embedding` | Modelo de embeddings |
| `EMBEDDING_API_KEY` | (vazio) | API key para serviço de embeddings externo |
| `LLM_API_URL` | (vazio) | URL da API LLM (habilita agente AI) |
| `LLM_MODEL` | `Qwen3.5-9B-Q4_K_M` | Modelo LLM para o agente |
| `LLM_API_KEY` | (vazio) | API key para serviço LLM externo |
| `CHROMADB_HOST` | `http://chromadb:8000` | URL do ChromaDB |
| `REDIS_URL` | `redis://redis:6379/0` | URL do Redis |

### Criando o arquivo `.env`

```bash
cp .env.example .env
```

## Instalação

### Docker (recomendado)

```bash
docker compose up --build -d

# Health check
curl http://localhost:8090/api/jobs/healthcheck

# Criar job assíncrono
curl -sS -X POST http://localhost:8090/api/jobs \
  -F "file=@document.pdf;type=application/pdf" | jq

# Busca semântica
curl -sS -X POST http://localhost:8090/api/search \
  -H "content-type: application/json" \
  -d '{"query":"contrato de locacao","n_results":5}' | jq
```

### Instalação Local

```bash
# Pré-requisitos
sudo apt-get install ghostscript poppler-utils tesseract-ocr tesseract-ocr-por

# Python
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm7.0 ".[torch-rocm]"
pip install .
pip install ".[dev]"  # Dev tools (linting, testing)
```

## Uso

### API REST

```bash
# Iniciar servidor
uvicorn services.app:app --host 0.0.0.0 --port 8080 --workers 1

# Nota: Use --workers 1 porque o modelo OCR é carregado uma vez na GPU
```

#### Endpoints

| Método | Endpoint                   | Descrição |
|--------|----------------------------|-----------|
| POST   | `/api/jobs`                | Upload de PDF e criação de job |
| GET    | `/api/jobs/{job_id}`       | Status do job |
| GET    | `/api/jobs/healthcheck`    | Health check da API |
| POST   | `/api/search`              | Busca semântica sobre chunks indexados |
| GET    | `/api/documents/{id}`      | Documento parseado (MongoDB) |
| POST   | `/api/agent/search`        | Busca enriquecida via agente AI |
| GET    | `/`                        | UI de upload |
| GET    | `/jobs/{job_id}`           | UI de status do job |

### CLI

```bash
# Processar PDF único
python scripts/process_single.py document.pdf

# Especificar diretório de saída
python scripts/process_single.py document.pdf -o /path/to/output

# Desabilitar extração de tabelas
python scripts/process_single.py document.pdf --no-tables

# Forçar CPU
python scripts/process_single.py document.pdf --no-gpu

# Usar Tesseract
python scripts/process_single.py document.pdf --ocr-engine tesseract

# Gerar PDF pesquisável
python scripts/process_single.py document.pdf --searchable-pdf

# Modo quiet
python scripts/process_single.py document.pdf --quiet
```

### Uso Programático

```python
from src.pipeline import DocumentProcessor

processor = DocumentProcessor(use_gpu=True)
doc = processor.process_document("document.pdf")

print(f"Total pages: {doc.total_pages}")
print(f"Total blocks: {sum(len(p.blocks) for p in doc.pages)}")

# Salvar JSON
processor.save_to_json(doc, "output.json", indent=4)

# Salvar PDF pesquisável
processor.save_to_searchable_pdf(doc, "document.pdf", "searchable_output.pdf")
```

## Estrutura de Saída JSON

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
          "text": "Texto do parágrafo...",
          "bbox": [0.1, 0.2, 0.9, 0.3],
          "confidence": 1.0,
          "lines": [
            {"text": "Line 1", "bbox": [0.1, 0.2, 0.9, 0.25]}
          ]
        }
      ],
      "width": 595,
      "height": 842
    }
  ]
}
```

### Tipos de Blocos

| Tipo | Descrição |
|------|-----------|
| `paragraph` | Parágrafos de texto |
| `table` | Tabelas (inclui campo `rows`) |
| `header` | Cabeçalhos |
| `footer` | Rodapés |
| `list` | Listas |
| `image` | Imagens (placeholder) |

### Coordenadas BBox

Todas as coordenadas são normalizadas (0–1) no formato `[x1, y1, x2, y2]`:
- `x1, y1`: Canto superior-esquerdo
- `x2, y2`: Canto inferior-direito

## Scripts

### process_single.py
Script CLI para processamento de PDF único com opções de configuração.

### llm_postprocess.py
Pós-processamento contextual de texto OCR via LLM (Ollama).

Uso:
```bash
# Dry run (apenas contar blocos)
python scripts/llm_postprocess.py output/documento.json --dry-run

# Processar e salvar
python scripts/llm_postprocess.py output/documento.json -o output/documento_llm.json
```

## Testes

```bash
# Rodar todos os testes
pytest

# Rodar com saída verbosa
pytest -v
```

## Limitações Atuais

- Assinaturas: ainda não tratadas
- Detecção de layout avançado (colunas, seções)
- Reconhecimento de formulários

## Roadmap

- [x] Pipeline de vetorização (embeddings via ChromaDB)
- [x] Processamento assíncrono de jobs (Redis + Worker)
- [x] Persistência de documentos (MongoDB)
- [x] Busca semântica sobre chunks indexados
- [x] Agente AI para busca enriquecida (ReAct loop)
- [ ] Melhor detecção de tabelas em PDFs digitalizados
- [ ] Detecção e classificação de assinaturas
- [ ] Processamento em lote de múltiplos PDFs

## Performance

Testado com PDF de 353 páginas (27 MB) (248 páginas digitais, 105 páginas OCR):
- **Tempo**: ~5.5 minutos (AMD Ryzen + RX 7900 XT)
- **Velocidade média**: ~1.2 páginas/segundo
- **Saída JSON**: 754 KB (591 blocos, 27 tabelas)

## Licença

Projeto interno para análise de documentos jurídicos.

## Development Commands

### AI Behavior Rules: Strict TDD Enforcement

You are an expert Senior Backend Engineer strictly following Test-Driven Development (TDD). From now on, you are strictly forbidden from writing production code (features or implementation details) before writing the tests.

You must obey the following rules unconditionally:
1. **Tests First, Always:** Whenever I request a new feature, logic, or module, your FIRST response must contain ONLY the unit/integration tests for that request. Use mocks where necessary.
2. **Refuse Direct Code Requests:** If I explicitly ask you to write the implementation of a feature without a test, you must politely refuse, remind me of our TDD rule, and provide the tests instead.
3. **Wait for Execution:** Do not write the implementation code until I confirm that I have run the test and it failed as expected (Red phase).
4. **Green Phase:** Only after the test is written and confirmed to be failing, you will provide the exact minimum implementation code required to make the test pass.
5. **Business Logic:** The code should be written in a way that is easy to understand, maintain, extend and attend the business requirements.
6. **Refactor:** After the test is passing, you will refactor the code to make it more readable, maintainable, and efficient, following the best practices of the Python community.
7. **No Hallucinated Coverage:** Never assume a test exists. If we are touching an existing file, check if it has tests. If not, write the tests before refactoring or adding features.

Reply with "TDD MODE ACTIVATED" if you understand and agree to these terms. Let's begin.

### Build Command

```bash
# Docker build
docker compose build

# Local install (with ROCm GPU support)
pip install --index-url https://download.pytorch.org/whl/rocm7.0 ".[torch-rocm]"

# Local install (CPU only)
pip install .
```

### Lint Command

```bash
# Run all linters
ruff check src services tests scripts
pylint src services tests scripts
mypy src services

# Pre-commit hooks
pre-commit run --all-files
```

### Test Command

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## Git Hooks

The `.git/hooks/` directory contains sample hook files (`.sample` extension):
- `pre-commit.sample` - Runs ruff, pylint, and mypy before committing
- `commit-msg.sample` - Validates commit message format
- `post-commit.sample` - Post-commit hook (placeholder)
