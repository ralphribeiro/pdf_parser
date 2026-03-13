"""
Job schemas for the async ingest API.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    UPLOADED = "uploaded"
    FAILED = "failed"


class Job(BaseModel):
    """Represents a processing job and its current state."""

    job_id: str = Field(description="Unique job identifier")
    filename: str = Field(description="Original uploaded filename")
    status: JobStatus = Field(description="Current job status")
    created_at: datetime = Field(description="Job creation timestamp")
    started_at: datetime | None = Field(
        default=None, description="Processing start timestamp"
    )
    completed_at: datetime | None = Field(
        default=None, description="Processing completion timestamp"
    )
    error_message: str | None = Field(
        default=None, description="Error details if status is failed"
    )
    file_hash: str | None = Field(
        default=None, description="SHA-256 hash of the uploaded file"
    )
    document_id: str | None = Field(
        default=None, description="MongoDB document identifier"
    )


class SearchFilters(BaseModel):
    """Optional filters for semantic search."""

    document_id: str | None = Field(
        default=None, description="Restrict results to one document"
    )


class SearchRequest(BaseModel):
    """Semantic search request payload."""

    query: str = Field(min_length=1, description="Natural-language query text")
    n_results: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results to return"
    )
    filters: SearchFilters | None = Field(default=None, description="Optional filters")
    min_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum similarity threshold",
    )


class SearchResult(BaseModel):
    """Single semantic search match."""

    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Retrieved chunk text")
    similarity: float = Field(ge=0.0, le=1.0, description="Similarity score")
    document_id: str = Field(description="Origin document identifier")
    source_file: str = Field(description="Origin filename")
    page_number: int = Field(ge=1, description="Origin page number")
    block_id: str = Field(description="Origin block id")
    block_type: str = Field(description="Origin block type")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")


class SearchResponse(BaseModel):
    """Semantic search response payload."""

    results: list[SearchResult] = Field(description="Ordered search results")
    total_matches: int = Field(ge=0, description="Number of results returned")
    processing_time_ms: int = Field(ge=0, description="Server processing time in ms")


# ---------------------------------------------------------------------------
# Agent search
# ---------------------------------------------------------------------------


class AgentSearchRequest(BaseModel):
    """Request payload for agent-based enriched search."""

    query: str = Field(min_length=1, description="Natural-language query text")
    document_id: str | None = Field(
        default=None, description="Restrict to a specific document"
    )
    max_iterations: int | None = Field(
        default=None, ge=1, le=20, description="Override default max iterations"
    )


class AgentSource(BaseModel):
    """A single source cited by the agent."""

    document_id: str = Field(description="Origin document identifier")
    filename: str = Field(default="", description="Original document filename")
    chunk_id: str = Field(default="", description="Origin chunk identifier")
    page: int = Field(description="Page number")
    text: str = Field(default="", description="Source text excerpt")


class AgentSearchResponse(BaseModel):
    """Response from agent-based enriched search."""

    answer: str = Field(description="Synthesized answer from the agent")
    sources: list[AgentSource] = Field(description="Sources cited by the agent")
    iterations: int = Field(ge=1, description="Number of agent iterations")
    processing_time_ms: int = Field(ge=0, description="Server processing time in ms")
