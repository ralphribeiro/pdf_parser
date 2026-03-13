"""
Agent tools — thin wrappers around SemanticSearchService and DocumentStore.

Each tool function accepts the shared services as the first two positional args,
followed by tool-specific kwargs parsed from the LLM's function call arguments.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": (
                "Busca semantica por query nos documentos indexados. "
                "Retorna os trechos mais relevantes com similaridade."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Texto da busca semantica",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Numero maximo de resultados (default 5)",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Opcional: restringir a um documento",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": (
                "Retorna metadados de um documento"
                " (filename, status, total_pages, created_at)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID do documento no MongoDB",
                    },
                },
                "required": ["document_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "Lista documentos disponiveis, opcionalmente filtrados por status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": (
                            "Filtrar por status (pending, processed, failed)"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Numero maximo de documentos (default 20)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_document_text",
            "description": (
                "Busca por palavra-chave dentro do texto"
                " parseado de um documento especifico."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID do documento",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Palavra-chave para buscar",
                    },
                },
                "required": ["document_id", "keyword"],
            },
        },
    },
]


def _truncate(text: str, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "... [truncado]"
    return text


def search_chunks(
    search_service: Any,
    document_store: Any,
    *,
    query: str,
    n_results: int = 5,
    document_id: str | None = None,
) -> str:
    results = search_service.search(
        query,
        n_results=n_results,
        document_id=document_id,
        min_similarity=None,
    )
    if not results:
        return "Nenhum resultado encontrado."

    filename_cache: dict[str, str] = {}
    lines: list[str] = []
    for r in results:
        doc_id = r.document_id or ""
        if doc_id and doc_id not in filename_cache:
            doc = document_store.get_document(doc_id) if document_store else None
            filename_cache[doc_id] = (doc.get("filename") or "") if doc else ""
        entry = {
            "chunk_id": r.chunk_id,
            "document_id": doc_id,
            "filename": filename_cache.get(doc_id, ""),
            "page": r.page_number,
            "similarity": r.similarity,
            "text": r.text,
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(lines)


def get_document(
    search_service: Any,
    document_store: Any,
    *,
    document_id: str,
) -> str:
    doc = document_store.get_document(document_id)
    if doc is None:
        return f"Documento nao encontrado: {document_id}"

    meta = {
        "document_id": str(doc.get("_id", document_id)),
        "filename": doc.get("filename"),
        "status": doc.get("status"),
        "total_pages": doc.get("total_pages"),
        "created_at": str(doc.get("created_at", "")),
    }
    return json.dumps(meta, ensure_ascii=False)


def list_documents(
    search_service: Any,
    document_store: Any,
    *,
    status: str | None = None,
    limit: int = 20,
) -> str:
    docs = document_store.list_documents(status=status, limit=limit)
    if not docs:
        return "Nenhum documento encontrado."

    lines: list[str] = []
    for doc in docs:
        entry = {
            "document_id": str(doc.get("_id", "")),
            "filename": doc.get("filename"),
            "status": doc.get("status"),
            "total_pages": doc.get("total_pages"),
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(lines)


def search_document_text(
    search_service: Any,
    document_store: Any,
    *,
    document_id: str,
    keyword: str = "",
    query: str = "",
) -> str:
    keyword = keyword or query
    if not keyword:
        return "Erro: parametro 'keyword' é obrigatorio."

    doc = document_store.get_document(document_id)
    if doc is None:
        return f"Documento nao encontrado: {document_id}"

    filename = doc.get("filename") or ""
    hits = document_store.search_text(document_id, keyword)
    if not hits:
        return f"Nenhum trecho encontrado com '{keyword}'."

    lines: list[str] = []
    for h in hits:
        h["document_id"] = document_id
        h["filename"] = filename
        h["chunk_id"] = h.pop("block_id", "")
        lines.append(json.dumps(h, ensure_ascii=False))
    return "\n".join(lines)


_TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "search_chunks": search_chunks,
    "get_document": get_document,
    "list_documents": list_documents,
    "search_document_text": search_document_text,
}


@dataclass
class ToolRegistry:
    """Manages tool schemas and dispatch for the agent loop."""

    search_service: Any
    document_store: Any

    def tool_schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def execute(
        self, tool_name: str, arguments: dict[str, Any], max_chars: int = 0
    ) -> str:
        func = _TOOL_FUNCTIONS.get(tool_name)
        if func is None:
            return f"Ferramenta desconhecida: {tool_name}"

        logger.info("Executing tool %s with args %s", tool_name, arguments)
        try:
            result = func(self.search_service, self.document_store, **arguments)
        except TypeError as exc:
            logger.warning("Tool %s call failed: %s", tool_name, exc)
            return f"Erro ao chamar {tool_name}: {exc}"
        return _truncate(result, max_chars)
