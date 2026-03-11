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
