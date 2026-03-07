"""
Schemas MongoDB para jobs, documentos e embeddings.

Testes: tests/test_embeddings_config.py::TestJobSchemas, TestDocumentSchemas, TestEmbeddingSchemas, TestChunkSchemas, TestWebhookSchemas, TestSearchFilters
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Statuses válidos para jobs."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreate(BaseModel):
    """Schema para criação de job."""

    file_content: bytes = Field(..., description="Conteúdo do arquivo PDF")
    generate_embeddings: bool = Field(
        default=True, description="Gerar embeddings do documento"
    )
    webhook_url: str | None = Field(
        default=None, description="URL para webhook de notificação"
    )


class JobResponse(BaseModel):
    """Schema para resposta de job."""

    job_id: str = Field(..., description="ID único do job")
    status: JobStatus = Field(..., description="Status atual do job")
    created_at: datetime = Field(..., description="Timestamp de criação")
    updated_at: datetime | None = Field(
        default=None, description="Timestamp de atualização"
    )
    result: dict[str, Any] | None = Field(default=None, description="Resultado do job")
    error: str | None = Field(default=None, description="Erro do job")
    embeddings_generated: bool = Field(
        default=False, description="Embeddings foram gerados"
    )


class JobNotFoundError(Exception):
    """Exceção para job não encontrado."""

    pass


class DocumentCreate(BaseModel):
    """Schema para criação de documento."""

    doc_id: str = Field(..., description="ID único do documento")
    source_file: str = Field(..., description="Arquivo fonte")
    total_pages: int = Field(..., description="Número total de páginas")
    pages: list[dict[str, Any]] = Field(
        default_factory=list, description="Detalhes das páginas"
    )


class DocumentResponse(BaseModel):
    """Schema para resposta de documento."""

    id: str = Field(..., description="ID do documento no MongoDB")
    doc_id: str = Field(..., description="ID único do documento")
    source_file: str = Field(..., description="Arquivo fonte")
    total_pages: int = Field(..., description="Número total de páginas")
    status: JobStatus = Field(..., description="Status do documento")
    pages: list[dict[str, Any]] = Field(
        default_factory=list, description="Detalhes das páginas"
    )
    processing_date: datetime | None = Field(
        default=None, description="Data de processamento"
    )
    created_at: datetime = Field(..., description="Timestamp de criação")
    completed_at: datetime | None = Field(
        default=None, description="Timestamp de conclusão"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Metadados adicionais"
    )


class EmbeddingCreate(BaseModel):
    """Schema para criação de embedding."""

    doc_id: str = Field(..., description="ID do documento")
    vector: list[float] = Field(..., description="Vetor de embedding")
    model: str = Field(default="nomic-embed-text", description="Modelo usado")


class EmbeddingResponse(BaseModel):
    """Schema para resposta de embedding."""

    id: str = Field(..., description="ID do embedding no MongoDB")
    doc_id: str = Field(..., description="ID do documento")
    vector: list[float] = Field(..., description="Vetor de embedding")
    model: str = Field(default="nomic-embed-text", description="Modelo usado")
    chunk_size: int | None = Field(default=None, description="Tamanho do chunk")
    created_at: datetime = Field(..., description="Timestamp de criação")


class ChunkCreate(BaseModel):
    """Schema para criação de chunk."""

    doc_id: str = Field(..., description="ID do documento")
    text: str = Field(..., description="Texto do chunk")
    vector: list[float] | None = Field(default=None, description="Vetor do chunk")
    page: int = Field(..., description="Número da página")
    block_id: str | None = Field(default=None, description="ID do bloco")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Metadados do chunk"
    )


class ChunkResponse(BaseModel):
    """Schema para resposta de chunk."""

    id: str = Field(..., description="ID do chunk no MongoDB")
    doc_id: str = Field(..., description="ID do documento")
    text: str = Field(..., description="Texto do chunk")
    vector: list[float] | None = Field(default=None, description="Vetor do chunk")
    page: int = Field(..., description="Número da página")
    block_id: str | None = Field(default=None, description="ID do bloco")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Metadados do chunk"
    )


class SearchResult(BaseModel):
    """Schema para resultado de busca semântica."""

    doc_id: str = Field(..., description="ID do documento")
    score: float = Field(..., ge=0, le=1, description="Score de similaridade (0-1)")
    total_pages: int = Field(..., description="Número total de páginas")
    created_at: datetime = Field(..., description="Timestamp de criação")
    matches: list[dict[str, Any]] | None = Field(
        default_factory=list, description="Blocos de texto correspondentes"
    )


class SemanticSearchResponse(BaseModel):
    """Schema para resposta de busca semântica."""

    query: str = Field(..., description="Query de busca")
    total_results: int = Field(..., description="Número total de resultados")
    results: list[SearchResult] = Field(..., description="Lista de resultados")


class SearchFilters(BaseModel):
    """Schema para filtros de busca semântica."""

    doc_id: str | None = Field(default=None, description="Filtrar por doc_id")
    min_score: float | None = Field(
        default=None, ge=0, le=1, description="Score mínimo"
    )
    created_after: datetime | None = Field(default=None, description="Criado após")
    created_before: datetime | None = Field(default=None, description="Criado antes")
    include_chunks: bool | None = Field(default=True, description="Incluir chunks")


class SearchParams(BaseModel):
    """Schema para parâmetros de busca semântica."""

    query: str = Field(..., min_length=1, description="Texto da busca")
    top_k: int = Field(
        default=10, ge=1, le=100, description="Número máximo de resultados"
    )
    min_score: float = Field(
        default=0.5, ge=0, le=1, description="Score mínimo de similaridade"
    )
    filters: dict[str, Any] | None = Field(
        default_factory=dict, description="Filtros adicionais"
    )
    include_matches: bool = Field(
        default=True, description="Incluir blocos de texto correspondentes"
    )
    matches_limit: int = Field(
        default=3, ge=0, le=10, description="Máximo de matches por documento"
    )


class WebhookPayload(BaseModel):
    """Schema para payload de webhook."""

    job_id: str = Field(..., description="ID do job")
    status: JobStatus = Field(..., description="Status do job")
    result: dict[str, Any] | None = Field(default=None, description="Resultado do job")


class WebhookConfig(BaseModel):
    """Schema para configuração de webhook."""

    url: str = Field(..., description="URL do webhook")
    token: str | None = Field(default=None, description="Token de autenticação")
    enabled: bool = Field(default=True, description="Webhook habilitado")


class VectorMetadata(BaseModel):
    """Schema para metadados de vetor."""

    doc_id: str = Field(..., description="ID do documento")
    source_file: str | None = Field(default=None, description="Arquivo fonte")
    page: int | None = Field(default=None, description="Número da página")
    block_id: str | None = Field(default=None, description="ID do bloco")
    type: str | None = Field(default=None, description="Tipo (document/chunk)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp de criação",
    )


class VectorSearchResult(BaseModel):
    """Schema para resultado de busca vetorial."""

    id: str = Field(..., description="ID do documento")
    doc_id: str = Field(..., description="ID do documento")
    score: float = Field(..., ge=0, le=1, description="Score de similaridade")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadados")
    text: str | None = Field(default=None, description="Texto do documento")
    page: int | None = Field(default=None, description="Número da página")
    block_id: str | None = Field(default=None, description="ID do bloco")


class BatchResponse(BaseModel):
    """Schema para resposta de batch."""

    success_count: int = Field(..., description="Número de operações bem-sucedidas")
    failed_count: int = Field(..., description="Número de operações falhas")
    results: list[dict[str, Any]] = Field(
        default_factory=list, description="Resultados individuais"
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list, description="Erros individuais"
    )


# Funções utilitárias para validação de timestamps


def validate_timestamp(value: Any) -> datetime | None:
    """Valida e converte timestamp para datetime."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        # Tenta vários formatos
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(f"Timestamp inválido: {value}")

    raise ValueError(f"Tipo inválido para timestamp: {type(value)}")
