# Plano de Implementação: Processamento Assíncrono com Vetorização

## 1. Visão Geral do Sistema

### 1.1 Arquitetura Proposta

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Cliente   │────▶│    API       │────▶│    Fila      │────▶│   Worker     │
│             │◀────│  Principal   │     │  (Redis/Celery)│     │  de Process  │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                          │
                                                          ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Cliente   │◀────│    API       │     │  ChromaDB    │◀────│  Serviço     │
│  (Consulta) │     │  de Consulta │     │  (Vetores)   │     │  de Embedding│
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
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
1. Upload PDF → Recebe job_id + receipt
2. Arquivo salvo em storage temporário
3. Job adicionado à fila (Celery/Redis)
4. Worker extrai texto (digital + OCR)
5. Texto segmentado em chunks ideais
6. Chunks enviados ao serviço de embedding
7. Embeddings salvos no ChromaDB
8. Status atualizado para COMPLETED
9. Notificação enviada ao cliente
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

## 4. Serviço de Embedding

### 4.1 Endpoints

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/embed` | POST | Gera embedding para texto único |
| `/batch-embed` | POST | Gera embeddings para múltiplos textos |
| `/health` | GET | Health check |
| `/models` | GET | Lista modelos disponíveis |

### 4.2 Request/Response Schema

**Request:**
```json
{
    "texts": ["string"],
    "model": "text-embedding-ada-002",
    "dimensions": 1536
}
```

**Response:**
```json
{
    "embeddings": [[float]],
    "usage": {
        "prompt_tokens": 123,
        "total_tokens": 123
    },
    "model": "text-embedding-ada-002"
}
```

### 4.3 Modelos Recomendados

| Modelo | Dimensões | Custo | Latência | Uso |
|--------|-----------|-------|----------|-----|
| OpenAI Ada-002 | 1536 | Pago | ~100ms | Alta qualidade |
| MiniLM-L6-v2 | 384 | Gratuito | ~50ms | Self-hosted |
| Cohere Embed | 1024 | Pago | ~80ms | Multilíngue |

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

```yaml
chroma_config:
  persist_directory: "./chroma_db"
  anonymized_telemetry: false
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
| `/search` | POST | Busca semântica |
| `/jobs/{job_id}` | GET | Status do job |
| `/jobs/{job_id}/chunks` | GET | Lista chunks processados |
| `/health` | GET | Health check |

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
# Async Task Queue
celery = "^5.3.0"
redis = "^5.0.0"
flower = "^2.0.0"

# ChromaDB
chromadb = "^0.4.0"

# Embeddings
openai = "^1.0.0"
sentence-transformers = "^2.2.0"

# HTTP Clients
httpx = "^0.26.0"
websockets = "^12.0"

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
  
  chromadb:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes: ["chroma_data:/chroma/chroma"]
  
  embedding-service:
    build: ./services/embedding
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MODEL_NAME=text-embedding-ada-002
  
  worker:
    build: ./app
    command: celery -A app.celery worker -Q processing --loglevel=info
    depends_on:
      - redis
      - chromadb
    environment:
      - REDIS_URL=redis://redis:6379
      - CHROMA_HOST=http://chromadb:8000
  
  api:
    build: ./app
    ports: ["8000:8000"]
    depends_on:
      - redis
      - chromadb
      - embedding-service
  
  flower:
    build: ./app
    command: celery -A app.celery flower --port=5555
    ports: ["5555:5555"]
    depends_on:
      - redis
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

1. [ ] Criar repositório para serviço de embedding
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
| Falha no serviço de embedding | Alto | Cache local, fallback para modelo gratuito |
| ChromaDB indisponível | Médio | Retry logic, fila de espera |
| Volume excessivo de jobs | Médio | Rate limiting, filas prioritárias |
| Custos de API de embedding | Alto | Caching de embeddings, batch processing |
| Tempo de processamento longo | Médio | Progress tracking, timeout configurável |

---

## 13. Métricas de Sucesso

| Métrica | Objetivo | Medição |
|---------|----------|---------|
| Tempo médio de processamento | < 5 min/job | Logs do worker |
| Taxa de sucesso | > 99% | Status dos jobs |
| Latência de busca | < 200ms | Timing da API |
| Uptime do sistema | > 99.5% | Monitoring |
| Custo por documento | < $0.10 | Billing reports |

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
*Versão: 1.0*  
*Status: Planejamento*
