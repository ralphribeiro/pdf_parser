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


class SearchFilters(BaseModel):
    """Optional filters for semantic search."""

    job_id: str | None = Field(default=None, description="Restrict results to one job")


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
    job_id: str = Field(description="Origin job identifier")
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
