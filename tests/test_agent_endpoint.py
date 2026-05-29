"""Tests for the POST /agent/search endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    calls: list[dict[str, Any]] = field(default_factory=list)
    last_query: str | None = None
    last_document_id: str | None = None

    def run(self, query: str, document_id: str | None = None) -> AgentResult:
        self.last_query = query
        self.last_document_id = document_id
        self.calls.append({"query": query, "document_id": document_id})
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


class _FakeDocStore:
    def __init__(self, docs: dict[str, dict]) -> None:
        self._docs = docs

    def get_document(self, document_id: str) -> dict | None:
        return self._docs.get(document_id)


@pytest.fixture()
def client() -> TestClient:
    app = create_app(agent_runner=_FakeAgentRunner())
    return TestClient(app)


@pytest.fixture()
def client_no_agent() -> TestClient:
    app = create_app(agent_runner=None)
    return TestClient(app)


@pytest.fixture()
def client_with_processed_doc() -> tuple[TestClient, _FakeAgentRunner]:
    runner = _FakeAgentRunner()
    doc_store = _FakeDocStore(
        {"doc-1": {"_id": "doc-1", "filename": "contrato.pdf", "status": "processed"}}
    )
    app = create_app(agent_runner=runner, document_store=doc_store)
    return TestClient(app), runner


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

    def test_success_with_document_id_passes_scope_to_runner(
        self, client_with_processed_doc: tuple[TestClient, _FakeAgentRunner]
    ) -> None:
        client, runner = client_with_processed_doc
        resp = client.post(
            "/agent/search",
            json={"query": "clausulas de multa", "document_id": "doc-1"},
        )
        assert resp.status_code == 200
        assert runner.calls[0]["document_id"] == "doc-1"
        assert runner.last_query == "clausulas de multa"

    def test_max_iterations_override_with_document_id(self) -> None:
        runner = _FakeAgentRunner()
        doc_store = _FakeDocStore(
            {
                "doc-1": {
                    "_id": "doc-1",
                    "filename": "contrato.pdf",
                    "status": "processed",
                }
            }
        )
        app = create_app(agent_runner=runner, document_store=doc_store)
        resp = TestClient(app).post(
            "/agent/search",
            json={"query": "test", "document_id": "doc-1", "max_iterations": 3},
        )
        assert resp.status_code == 200
        assert runner.calls[0]["document_id"] == "doc-1"

    def test_document_not_found(self) -> None:
        runner = _FakeAgentRunner()
        app = create_app(agent_runner=runner, document_store=_FakeDocStore({}))
        resp = TestClient(app).post(
            "/agent/search",
            json={"query": "test", "document_id": "missing"},
        )
        assert resp.status_code == 404
        assert not runner.calls

    @pytest.mark.parametrize("status", ["pending", "processing", "failed"])
    def test_document_not_ready(self, status: str) -> None:
        runner = _FakeAgentRunner()
        doc_store = _FakeDocStore(
            {"doc-1": {"_id": "doc-1", "filename": "contrato.pdf", "status": status}}
        )
        app = create_app(agent_runner=runner, document_store=doc_store)
        resp = TestClient(app).post(
            "/agent/search",
            json={"query": "test", "document_id": "doc-1"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["document_id"] == "doc-1"
        assert resp.json()["detail"]["status"] == status
        assert not runner.calls

    def test_document_store_required_when_document_id_is_provided(self) -> None:
        runner = _FakeAgentRunner()
        app = create_app(agent_runner=runner)
        resp = TestClient(app).post(
            "/agent/search",
            json={"query": "test", "document_id": "doc-1"},
        )
        assert resp.status_code == 503
        assert not runner.calls
