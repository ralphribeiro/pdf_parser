"""
Endpoint de busca semântica.

Testes: tests/test_async_jobs.py (testes de busca semântica)
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.database import vector_near_search
from src.embeddings import EmbeddingsClient, cosine_similarity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Semantic Search"])


# Schema de busca


class SemanticSearchQuery(BaseModel):
    """Query de busca semântica."""

    query: str = Field(..., min_length=1, description="Texto da busca")
    top_k: int = Field(
        default=10, ge=1, le=100, description="Número máximo de documentos"
    )
    min_score: float = Field(
        default=0.5, ge=0, le=1, description="Score mínimo de similaridade"
    )
    filters: dict | None = Field(default=None, description="Filtros adicionais")
    include_matches: bool = Field(default=True, description="Incluir blocos de texto")
    matches_limit: int = Field(
        default=3, ge=0, le=10, description="Máximo de matches por documento"
    )


class SemanticSearchMatch(BaseModel):
    """Match de texto em documento."""

    page: int
    block_id: str
    text: str
    bbox: list | None = None


class SemanticSearchResult(BaseModel):
    """Resultado de busca semântica."""

    doc_id: str
    score: float
    total_pages: int
    created_at: datetime
    matches: list | None = Field(default=None)


class SemanticSearchResponse(BaseModel):
    """Resposta de busca semântica."""

    query: str
    total_results: int
    results: list[SemanticSearchResult]


# Funções auxiliares


def generate_embedding_for_query(query: str) -> list[float]:
    """
    Gerar embedding para query de busca.

    Args:
        query: Texto da query

    Returns:
        Embedding como lista de floats
    """
    try:
        client = EmbeddingsClient()
        return client.generate_embedding(query)
    except Exception as e:
        logger.error(f"Erro ao gerar embedding para query: {e}")
        return [0.0] * 768  # Retornar zeros como fallback


@router.post("/semantic")
async def semantic_search(
    search_query: SemanticSearchQuery,
) -> SemanticSearchResponse:
    """
    Buscar documentos por semelhança semântica.

    Args:
        search_query: Query de busca com texto e parâmetros

    Returns:
        Resultados ordenados por similaridade
    """
    logger.info(f"Busca semântica: {search_query.query[:50]}...")

    try:
        # 1. Gerar embedding da query
        query_vector = generate_embedding_for_query(search_query.query)

        if not query_vector or all(v == 0 for v in query_vector):
            logger.warning("Query vector é zero - embedding não gerou resultado")
            return SemanticSearchResponse(
                query=search_query.query, total_results=0, results=[]
            )

        # 2. Buscar documentos similares no MongoDB
        filters = search_query.filters or {}
        min_score = search_query.min_score
        top_k = search_query.top_k

        results = vector_near_search(
            job_id="search",  # Job fictício para busca
            query_vector=query_vector,
            top_k=top_k,
            min_score=min_score,
            filters=filters,
        )

        logger.info(f"Encontrados {len(results)} resultados")

        # 3. Formatar resultados
        search_results = []

        for result in results:
            # Extrair campos do resultado
            doc_id = result.get("doc_id", "unknown")
            score = result.get("_score", result.get("score", 0))
            total_pages = result.get("total_pages", 0)
            created_at = result.get("created_at", datetime.utcnow())

            # Converter timestamp se for string
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except (ValueError, TypeError):
                    created_at = datetime.utcnow()

            # Obter documento completo para buscar matches
            matches = []
            if search_query.include_matches:
                # Buscar blocos de texto relacionados
                matches = await get_document_matches(doc_id, search_query.matches_limit)

            search_result = SemanticSearchResult(
                doc_id=doc_id,
                score=score,
                total_pages=total_pages,
                created_at=created_at,
                matches=matches,
            )

            search_results.append(search_result)

        # Ordenar por score decrescente
        search_results.sort(key=lambda x: x.score, reverse=True)

        return SemanticSearchResponse(
            query=search_query.query,
            total_results=len(search_results),
            results=search_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na busca semântica: {e}")
        raise HTTPException(status_code=500, detail=f"Erro na busca semântica: {e!s}")


@router.post("/semantic/batch")
async def semantic_search_batch(
    search_queries: list[SemanticSearchQuery],
) -> SemanticSearchResponse:
    """
    Buscar múltiplas queries de uma vez (batch).

    Args:
        search_queries: Lista de queries de busca

    Returns:
        Lista de respostas de busca
    """
    results = []

    for i, query in enumerate(search_queries):
        try:
            result = await semantic_search(query)
            result.results_index = i
            results.append(result)
        except Exception as e:
            logger.error(f"Erro ao processar query {i}: {e}")
            results.append(
                SemanticSearchResponse(
                    query=query.query, total_results=0, results=[], error=str(e)
                )
            )

    return SemanticSearchResponse(
        query=search_queries[0].query if search_queries else "",
        total_results=len(results),
        results=results,
    )


@router.get("/semantic")
async def semantic_search_get(
    query: str,
    top_k: int = 10,
    min_score: float = 0.5,
    include_matches: bool = True,
    matches_limit: int = 3,
) -> SemanticSearchResponse:
    """
    Buscar documentos por semelhança semântica (query string).

    Versão alternativa usando query string em vez de JSON body.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query é obrigatória")

    search_query = SemanticSearchQuery(
        query=query,
        top_k=top_k,
        min_score=min_score,
        include_matches=include_matches,
        matches_limit=matches_limit,
    )

    return await semantic_search(search_query)


async def get_document_matches(
    doc_id: str, limit: int = 3
) -> list[SemanticSearchMatch]:
    """
    Obter matches de texto em documento.

    Args:
        doc_id: ID do documento
        limit: Limite de matches

    Returns:
        Lista de matches
    """
    # Buscar documento no MongoDB
    from src.database import get_documents_collection

    docs_collection = get_documents_collection()
    doc = docs_collection.find_one({"doc_id": doc_id})

    if not doc:
        return []

    matches = []

    # Buscar blocos de texto em cada página
    for page in doc.get("pages", []):
        page_num = page.get("page", 0)
        blocks = page.get("blocks", [])

        for block in blocks:
            # Verificar se o bloco é relevante para a query
            block_text = block.get("text", "")

            # Calcular similaridade com a query
            query_vector = generate_embedding_for_query(doc.get("query", ""))
            block_vector = generate_embedding_for_query(block_text)

            if query_vector and block_vector and len(query_vector) == len(block_vector):
                similarity = cosine_similarity(query_vector, block_vector)
            else:
                # Fallback para similaridade simples
                similarity = 0

            if similarity > 0.5 and len(matches) < limit:
                matches.append(
                    SemanticSearchMatch(
                        page=page_num,
                        block_id=block.get("block_id", f"p{page_num}_b0"),
                        text=block_text[:200],  # Limite de texto
                        bbox=block.get("bbox"),
                    )
                )

    return matches[:limit]
