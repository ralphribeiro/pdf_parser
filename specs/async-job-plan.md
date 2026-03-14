# Plano de Implementação: Processamento Assíncrono com Vetorização

## 1. Visão Geral do Sistema

### 1.1 Arquitetura Proposta

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Cliente   │────▶│    API       │────▶│    Fila      │────▶│   Worker     │
│             │◀────│  (FastAPI)   │     │   (Redis)    │     │  OCR + Index │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                          │                                         │
                          ▼                                         ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Agente    │────▶│   MongoDB    │     │  ChromaDB    │◀────│  Embedding   │
│  AI (ReAct) │     │  (Documentos)│     │  (Vetores)   │     │  (llama.cpp) │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                    │
                                                              ┌──────────────┐
                                                              │  LLM Chat    │
                                                              │  (llama.cpp) │
                                                              └──────────────┘
```

---

## 2. Componentes Principais

### 2.1 Job Processing System

#### Estrutura de Dados

```python
@dataclass
class ProcessingJob:
    job_id: str                    # UUID único
    pdf_path: Path                 # Caminho do arquivo
    client_id: str                 # Identificador do cliente
    status: JobStatus              # PENDING | PROCESSING | COMPLETED | FAILED
    created_at: datetime           # Timestamp de criação
    started_at: datetime | None    # Início do processamento
    completed_at: datetime | None  # Conclusão
    error_message: str | None      # Mensagem de erro se falhar
    metadata: dict                 # Metadados adicionais
    webhook_url: str | None        # URL para notificação
```

#### Estados do Job

| Estado | Descrição | Ação |
|--------|-----------|------|
| `PENDING` | Aguardando na fila | Enfileirado para worker |
| `PROCESSING` | Em processamento | Worker atualizando progresso |
| `COMPLETED` | Finalizado com sucesso | Notificação enviada |
| `FAILED` | Erro durante processamento | Log de erro, possível retry |
| `CANCELLED` | Cancelado pelo cliente | Limpeza de recursos |

---

## 3. Pipeline de Processamento

### 3.1 Fluxo Completo

```
1. Upload PDF → Salvo em /data, registrado no MongoDB → Recebe job_id + document_id
2. Job adicionado à fila (Redis)
3. Worker extrai texto (digital + OCR via docTR/Tesseract)
4. Documento parseado salvo no MongoDB (parsed_document)
5. Texto segmentado em chunks por bloco
6. Chunks enviados ao serviço de embedding (llama.cpp)
7. Embeddings + metadados salvos no ChromaDB
8. Status atualizado para COMPLETED no Redis e MongoDB
9. Cliente pode buscar via /api/search (semântica) ou /api/agent/search (agente AI)
```

### 3.2 Configuração de Batching

| Parâmetro | Valor Sugerido | Justificativa |
|-----------|----------------|---------------|
| Batch Size | 500-1000 tokens | Balance custo/contexto |
| Overlap | 50-100 tokens | Preserva contexto entre chunks |
| Max Chunks | 100 por documento | Evita limites de API |
| Chunk Size | 300-500 chars | Tamanho ideal para embedding |

### 3.3 Estrutura de Chunk

```python
@dataclass
class TextChunk:
    chunk_id: str                    # UUID único
    job_id: str                      # ID do job pai
    page_number: int                 # Página original
    chunk_index: int                 # Índice dentro da página
    text: str                        # Conteúdo do texto
    metadata: dict                   # {source_file, page, position}
    embedding: list[float] | None    # Embedding gerado
    created_at: datetime             # Timestamp
```

---

## 4. Serviço de Embedding (externo)

O serviço de embedding roda em um servidor dedicado separado (llama.cpp servindo API compatível com OpenAI).
A aplicação consome esse serviço via `EMBEDDING_API_URL`.

### 4.1 Endpoints consumidos

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/v1/embeddings` | POST | Gera embeddings (padrão OpenAI) |
| `/health` | GET | Health check |

### 4.2 Request/Response Schema

**Request (padrão OpenAI /v1/embeddings):**
```json
{
    "input": ["string"],
    "model": "model-name"
}
```

**Response:**
```json
{
    "data": [
        {
            "embedding": [float],
            "index": 0
        }
    ],
    "usage": {
        "prompt_tokens": 123,
        "total_tokens": 123
    },
    "model": "model-name"
}
```

### 4.3 Modelos Recomendados

| Modelo | Dimensões | Custo | Latência | Uso |
|--------|-----------|-------|----------|-----|
| GGUF local (llama.cpp) | Variável | Gratuito | ~50ms | Self-hosted, principal |
| OpenAI Ada-002 | 1536 | Pago | ~100ms | Alternativa cloud |
| Cohere Embed | 1024 | Pago | ~80ms | Alternativa multilíngue |

---

## 5. Banco de Dados Vetorial (ChromaDB)

### 5.1 Estrutura de Coleções

```python
# Coleção principal
collection_name = "document_embeddings"

# Índices compostos
indexes = [
    ("job_id", "eq"),
    ("client_id", "eq"),
    ("page_number", "gte"),
    ("created_at", "gte")
]
```

### 5.2 Metadados Armazenados

```json
{
    "job_id": "uuid-string",
    "client_id": "uuid-string",
    "page_number": 5,
    "chunk_index": 12,
    "source_file": "original.pdf",
    "created_at": "2024-01-15T10:30:00Z",
    "text_preview": "primeiras 50 chars...",
    "word_count": 250,
    "language": "pt-BR"
}
```

### 5.3 Operações Principais

```python
# Inserção
chromadb_client.get_or_create_collection(name="document_embeddings")
collection.add(
    embeddings=embeddings,
    metadatas=metadata_list,
    ids=chunk_ids
)

# Consulta semântica
results = collection.query(
    query_embeddings=query_embedding,
    n_results=10,
    where={"client_id": client_id},
    include=["metadatas", "documents"]
)

# Exclusão por job
collection.delete(where={"job_id": job_id})
```

### 5.4 Persistência

Configurado via variáveis de ambiente no container Docker (`IS_PERSISTENT=TRUE`, `ANONYMIZED_TELEMETRY=FALSE`).
Dados persistidos no volume `chroma_data:/chroma/chroma`.

```yaml
chroma_config:
  tenant: default_tenant
  database: default_database
```

---

## 6. Sistema de Notificações

### 6.1 Métodos Disponíveis

| Método | Vantagens | Desvantagens | Recomendação |
|--------|-----------|--------------|--------------|
| Webhook | Push imediato | Requer endpoint externo | ✅ Preferencial |
| Polling | Simples | Latência variável | ⚠️ Alternativo |
| WebSocket | Real-time | Complexidade adicional | 🔄 Opcional |

### 6.2 Payload de Notificação

```json
{
    "event": "job.completed",
    "job_id": "uuid-string",
    "status": "completed",
    "result_url": "/api/v1/jobs/{job_id}/result",
    "chunks_processed": 150,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### 6.3 Retry Logic

```python
retry_config = {
    "max_retries": 3,
    "retry_delay": 60,  # segundos
    "backoff_multiplier": 2
}
```

---

## 7. API de Consulta Semântica

### 7.1 Endpoints

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/jobs` | POST | Upload de PDF e criação de job |
| `/api/jobs/{job_id}` | GET | Status do job |
| `/api/jobs/healthcheck` | GET | Health check |
| `/api/search` | POST | Busca semântica sobre chunks |
| `/api/documents/{id}` | GET | Documento parseado (MongoDB) |
| `/api/agent/search` | POST | Busca enriquecida via agente AI |

### 7.2 Endpoint de Busca

**POST /api/v1/search**

**Request:**
```json
{
    "query": "texto da consulta",
    "n_results": 10,
    "filters": {
        "client_id": "uuid",
        "date_range": {"start": "...", "end": "..."},
        "page_numbers": [1, 2, 3]
    },
    "min_similarity": 0.7,
    "include_metadata": true
}
```

**Response:**
```json
{
    "results": [
        {
            "chunk_id": "uuid",
            "text": "conteúdo recuperado",
            "similarity": 0.92,
            "metadata": {
                "job_id": "uuid",
                "page_number": 5,
                "source_file": "doc.pdf"
            }
        }
    ],
    "query_embedding": [...],
    "total_matches": 47,
    "processing_time_ms": 125
}
```

### 7.3 Endpoint de Status

**GET /api/v1/jobs/{job_id}**

**Response:**
```json
{
    "job_id": "uuid",
    "status": "completed",
    "progress": 100,
    "chunks_processed": 150,
    "total_chunks": 150,
    "created_at": "2024-01-15T10:00:00Z",
    "started_at": "2024-01-15T10:00:05Z",
    "completed_at": "2024-01-15T10:05:30Z",
    "error_message": null
}
```

---

## 8. Dependências Necessárias

### 8.1 Python Packages

```toml
[dependencies]
# Job Queue
redis = "^5.0.0"

# Databases
chromadb = "^0.4.0"
pymongo = "^4.0.0"

# HTTP Clients (embedding + LLM via API OpenAI-compatible)
httpx = "^0.26.0"

# Utilities
pydantic-settings = "^2.0.0"
python-multipart = "^0.0.9"
```

### 8.2 Docker Services

```yaml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: ["redis_data:/data"]

  mongodb:
    image: mongo:7
    ports: ["27017:27017"]
    volumes: ["mongo_data:/data/db"]

  chromadb:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes: ["chroma_data:/chroma/chroma"]
    environment:
      - ANONYMIZED_TELEMETRY=FALSE
      - IS_PERSISTENT=TRUE

  api:
    build: .
    ports: ["8090:8080"]
    depends_on: [redis, mongodb, chromadb]
    environment:
      - REDIS_URL, MONGO_URL, CHROMADB_HOST
      - EMBEDDING_API_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY
      - LLM_API_URL, LLM_MODEL, LLM_API_KEY

  worker:
    build: .
    command: python -m services.worker.run
    depends_on: [redis, mongodb, chromadb]
    environment:
      - REDIS_URL, MONGO_URL, CHROMADB_HOST
      - EMBEDDING_API_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY
```

---

## 9. Cronograma Estimado

| Fase | Duração | Entregáveis |
|------|---------|-------------|
| **Fase 1: Infraestrutura** | | |
| Setup Redis + ChromaDB | 1 dia | Docker Compose funcional |
| Config Celery + Workers | 1 dia | Jobs básicos funcionando |
| **Fase 2: Core Processing** | | |
| Pipeline de extração | 2 dias | Texto extraído corretamente |
| Chunking inteligente | 1 dia | Batches otimizados |
| Integração Embedding | 2 dias | Embeddings gerados |
| **Fase 3: Storage & Query** | | |
| ChromaDB integration | 2 dias | Coleções operacionais |
| API de busca | 2 dias | Endpoints funcionais |
| **Fase 4: Notifications** | | |
| Webhook system | 1 dia | Notificações enviadas |
| Polling fallback | 1 dia | Status endpoints |
| **Fase 5: Testing** | | |
| Unit tests | 2 dias | Cobertura >80% |
| Integration tests | 2 dias | Fluxos completos |
| Load testing | 1 dia | Performance validada |
| **Total** | **~18 dias** | **Sistema completo** |

---

## 10. Considerações de Segurança

### 10.1 Autenticação e Autorização

- [ ] JWT tokens para todos os endpoints
- [ ] Rate limiting por cliente (ex: 100 req/min)
- [ ] Sanitização de uploads de PDF
- [ ] Validação de tamanho máximo (ex: 50MB)
- [ ] Isolamento de dados por cliente (tenant isolation)

### 10.2 Proteção de Dados

- [ ] Criptografia em trânsito (TLS/HTTPS)
- [ ] Criptografia em repouso (ChromaDB)
- [ ] Logs auditáveis de todas as operações
- [ ] Backup automático do ChromaDB
- [ ] Retenção automática de arquivos temporários (24h)

### 10.3 Monitoramento

- [ ] Health checks em todos os serviços
- [ ] Métricas de performance (latência, throughput)
- [ ] Alertas de falha (email/slack)
- [ ] Dashboard de uso (jobs, tokens, custos)

---

## 11. Próximos Passos Imediatos

### Semana 1: Preparação

1. [ ] Validar conectividade com serviço externo de embedding (llama.cpp)
2. [ ] Configurar Docker Compose com Redis e ChromaDB
3. [ ] Definir schema de metadados do ChromaDB
4. [ ] Criar testes unitários para cada componente

### Semana 2: Core

5. [ ] Implementar estrutura básica do worker Celery
6. [ ] Integrar pipeline de extração existente
7. [ ] Criar sistema de chunking inteligente
8. [ ] Conectar serviço de embedding

### Semana 3: Integração

9. [ ] Implementar CRUD do ChromaDB
10. [ ] Criar API de busca semântica
11. [ ] Implementar sistema de notificações
12. [ ] Testes de integração completos

### Semana 4: Refinamento

13. [ ] Load testing e otimização
14. [ ] Documentação da API
15. [ ] Deploy em ambiente de staging
16. [ ] Go-live e monitoramento inicial

---

## 12. Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Falha no serviço de embedding | Alto | Retry logic, alerta ao admin do servidor externo, circuit breaker |
| ChromaDB indisponível | Médio | Retry logic, fila de espera |
| Volume excessivo de jobs | Médio | Rate limiting, filas prioritárias |
| Indisponibilidade do servidor externo de embedding | Médio | Health check periódico, caching de embeddings, retry com backoff |
| Tempo de processamento longo | Médio | Progress tracking, timeout configurável |

---

## 13. Métricas de Sucesso

| Métrica | Objetivo | Medição |
|---------|----------|---------|
| Tempo médio de processamento | < 5 min/job | Logs do worker |
| Taxa de sucesso | > 99% | Status dos jobs |
| Latência de busca | < 200ms | Timing da API |
| Uptime do sistema | > 99.5% | Monitoring |
| Custo por documento | Self-hosted | Monitoramento de recursos |

---

## 14. Glossário

| Termo | Definição |
|-------|-----------|
| **Chunk** | Fragmento de texto para embedding |
| **Embedding** | Representação vetorial do texto |
| **Tenant Isolation** | Separação lógica de dados por cliente |
| **Webhook** | Notificação push via HTTP |
| **Polling** | Verificação periódica de status |
| **Batch Processing** | Processamento em lotes para eficiência |

---

*Documento criado: 2024-01-15*
*Última atualização: 2026-03-13*
*Versão: 2.0*
*Status: Implementado*
