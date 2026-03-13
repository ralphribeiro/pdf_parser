"""Tests for agent tool functions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from services.agent.tools import (
    ToolRegistry,
    get_document,
    list_documents,
    search_chunks,
    search_document_text,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeSearchService:
    """Stub for SemanticSearchService."""

    results: list

    def search(self, query, *, n_results=5, document_id=None, min_similarity=None):
        return self.results


class _FakeDocStore:
    """Stub for DocumentStore."""

    def __init__(self, docs: list[dict] | None = None):
        self._docs = {str(d["_id"]): d for d in (docs or [])}

    def get_document(self, document_id: str):
        return self._docs.get(document_id)

    def list_documents(self, *, status: str | None = None, limit: int = 20):
        docs = list(self._docs.values())
        if status:
            docs = [d for d in docs if d.get("status") == status]
        return docs[:limit]

    def search_text(self, document_id: str, keyword: str):
        doc = self._docs.get(document_id)
        if not doc:
            return []
        parsed = doc.get("parsed_document") or {}
        pages = parsed.get("pages", [])
        hits = []
        for page in pages:
            for block in page.get("blocks", []):
                text = block.get("text", "")
                if keyword.lower() in text.lower():
                    hits.append(
                        {
                            "page": page.get("page_number", 0),
                            "block_id": block.get("block_id", ""),
                            "text": text,
                        }
                    )
        return hits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def search_service():
    from services.ingest_api.schemas import SearchResult

    results = [
        SearchResult(
            chunk_id="doc1:1:b1",
            text="Clausula de multa de 10%",
            similarity=0.92,
            document_id="doc1",
            source_file="contrato.pdf",
            page_number=1,
            block_id="b1",
            block_type="text",
            confidence=0.95,
        ),
    ]
    return _FakeSearchService(results=results)


@pytest.fixture()
def doc_store():
    return _FakeDocStore(
        docs=[
            {
                "_id": "doc1",
                "filename": "contrato.pdf",
                "status": "processed",
                "total_pages": 5,
                "created_at": "2025-01-01",
                "parsed_document": {
                    "pages": [
                        {
                            "page_number": 1,
                            "blocks": [
                                {"block_id": "b1", "text": "Clausula de multa de 10%"},
                                {"block_id": "b2", "text": "Prazo de 30 dias"},
                            ],
                        }
                    ]
                },
            },
            {
                "_id": "doc2",
                "filename": "aditivo.pdf",
                "status": "pending",
                "total_pages": None,
                "created_at": "2025-01-02",
                "parsed_document": None,
            },
        ]
    )


# ---------------------------------------------------------------------------
# Tests: search_chunks
# ---------------------------------------------------------------------------


class TestSearchChunks:
    def test_returns_formatted_results(self, search_service, doc_store):
        result = search_chunks(search_service, doc_store, query="multa", n_results=5)
        assert "Clausula de multa" in result
        assert "doc1" in result
        assert "0.92" in result
        assert '"filename": "contrato.pdf"' in result

    def test_no_results(self, doc_store):
        empty_svc = _FakeSearchService(results=[])
        result = search_chunks(empty_svc, doc_store, query="inexistente")
        assert "Nenhum resultado" in result


# ---------------------------------------------------------------------------
# Tests: get_document
# ---------------------------------------------------------------------------


class TestGetDocument:
    def test_found(self, search_service, doc_store):
        result = get_document(search_service, doc_store, document_id="doc1")
        assert "contrato.pdf" in result
        assert "processed" in result

    def test_not_found(self, search_service, doc_store):
        result = get_document(search_service, doc_store, document_id="nope")
        assert "nao encontrado" in result.lower() or "não encontrado" in result.lower()


# ---------------------------------------------------------------------------
# Tests: list_documents
# ---------------------------------------------------------------------------


class TestListDocuments:
    def test_lists_all(self, search_service, doc_store):
        result = list_documents(search_service, doc_store)
        assert "contrato.pdf" in result
        assert "aditivo.pdf" in result

    def test_filter_by_status(self, search_service, doc_store):
        result = list_documents(search_service, doc_store, status="pending")
        assert "aditivo.pdf" in result
        assert "contrato.pdf" not in result


# ---------------------------------------------------------------------------
# Tests: search_document_text
# ---------------------------------------------------------------------------


class TestSearchDocumentText:
    def test_keyword_match(self, search_service, doc_store):
        result = search_document_text(
            search_service, doc_store, document_id="doc1", keyword="multa"
        )
        assert "multa" in result.lower()
        assert '"document_id": "doc1"' in result
        assert '"filename": "contrato.pdf"' in result
        assert '"chunk_id": "b1"' in result

    def test_no_match(self, search_service, doc_store):
        result = search_document_text(
            search_service, doc_store, document_id="doc1", keyword="xyz999"
        )
        assert "nenhum" in result.lower()

    def test_document_not_found(self, search_service, doc_store):
        result = search_document_text(
            search_service, doc_store, document_id="nope", keyword="x"
        )
        assert "nao encontrado" in result.lower() or "não encontrado" in result.lower()


# ---------------------------------------------------------------------------
# Tests: ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_schemas_list(self):
        registry = ToolRegistry(
            search_service=_FakeSearchService([]),
            document_store=_FakeDocStore(),
        )
        schemas = registry.tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "search_chunks" in names
        assert "get_document" in names
        assert "list_documents" in names
        assert "search_document_text" in names

    def test_execute_dispatches(self, search_service, doc_store):
        registry = ToolRegistry(
            search_service=search_service,
            document_store=doc_store,
        )
        result = registry.execute("get_document", {"document_id": "doc1"})
        assert "contrato.pdf" in result

    def test_execute_unknown_tool(self, search_service, doc_store):
        registry = ToolRegistry(
            search_service=search_service,
            document_store=doc_store,
        )
        result = registry.execute("nonexistent", {})
        assert "desconhecida" in result.lower()

    def test_truncation(self, search_service, doc_store):
        registry = ToolRegistry(
            search_service=search_service,
            document_store=doc_store,
        )
        result = registry.execute("get_document", {"document_id": "doc1"}, max_chars=30)
        assert len(result) <= 30 + len("... [truncado]")
