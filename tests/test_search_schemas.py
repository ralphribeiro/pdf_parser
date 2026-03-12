"""
Tests for semantic search API schemas.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSearchRequest:
    def test_defaults(self):
        from services.ingest_api.schemas import SearchRequest

        req = SearchRequest(query="fatura vencida")
        assert req.query == "fatura vencida"
        assert req.n_results == 10
        assert req.filters is None
        assert req.min_similarity is None

    def test_rejects_empty_query(self):
        from services.ingest_api.schemas import SearchRequest

        with pytest.raises(Exception, match="at least 1 character"):
            SearchRequest(query="")

    def test_rejects_invalid_n_results(self):
        from services.ingest_api.schemas import SearchRequest

        with pytest.raises(ValidationError):
            SearchRequest(query="abc", n_results=0)

    def test_accepts_job_filter(self):
        from services.ingest_api.schemas import SearchFilters, SearchRequest

        req = SearchRequest(
            query="contrato",
            filters=SearchFilters(job_id="job-123"),
        )
        assert req.filters is not None
        assert req.filters.job_id == "job-123"


class TestSearchResponse:
    def test_result_shape(self):
        from services.ingest_api.schemas import SearchResponse, SearchResult

        result = SearchResult(
            chunk_id="job-1:3:p3_b1",
            text="Texto relevante",
            similarity=0.92,
            job_id="job-1",
            source_file="doc.pdf",
            page_number=3,
            block_id="p3_b1",
            block_type="paragraph",
            confidence=0.98,
        )
        response = SearchResponse(
            results=[result],
            total_matches=1,
            processing_time_ms=12,
        )
        assert response.total_matches == 1
        assert response.results[0].similarity == 0.92

    def test_similarity_bounds(self):
        from services.ingest_api.schemas import SearchResult

        with pytest.raises(ValidationError):
            SearchResult(
                chunk_id="x",
                text="x",
                similarity=1.5,
                job_id="j",
                source_file="f.pdf",
                page_number=1,
                block_id="b",
                block_type="paragraph",
                confidence=0.5,
            )
