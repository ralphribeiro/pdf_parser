"""Tests for the POST /agent/search endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient
from services.agent.agent import AgentResult
from services.ingest_api.app import create_app


@dataclass
class _FakeAgentRunner:
    """Stub that returns a pre-configured AgentResult."""

    answer: str = "Resposta sintetizada."
    sources: list[dict[str, Any]] | None = None
    iterations: int = 2
    max_iterations: int = 8

    def run(self, query: str) -> AgentResult:
        return AgentResult(
            answer=self.answer,
            sources=self.sources
            or [
                {
                    "document_id": "abc",
                    "filename": "contrato.pdf",
                    "chunk_id": "abc:3:b1",
                    "page": 3,
                    "text": "trecho relevante",
                }
            ],
            iterations=self.iterations,
        )


@pytest.fixture()
def client() -> TestClient:
    app = create_app(agent_runner=_FakeAgentRunner())
    return TestClient(app)


@pytest.fixture()
def client_no_agent() -> TestClient:
    app = create_app(agent_runner=None)
    return TestClient(app)


class TestAgentSearchEndpoint:
    def test_success(self, client: TestClient) -> None:
        resp = client.post(
            "/agent/search",
            json={"query": "clausulas de multa"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Resposta sintetizada."
        assert len(body["sources"]) == 1
        assert body["sources"][0]["document_id"] == "abc"
        assert body["sources"][0]["filename"] == "contrato.pdf"
        assert body["iterations"] == 2
        assert body["processing_time_ms"] >= 0

    def test_agent_not_configured(self, client_no_agent: TestClient) -> None:
        resp = client_no_agent.post(
            "/agent/search",
            json={"query": "test"},
        )
        assert resp.status_code == 503

    def test_empty_query_rejected(self, client: TestClient) -> None:
        resp = client.post("/agent/search", json={"query": ""})
        assert resp.status_code == 422

    def test_max_iterations_override(self, client: TestClient) -> None:
        resp = client.post(
            "/agent/search",
            json={"query": "test", "max_iterations": 3},
        )
        assert resp.status_code == 200
