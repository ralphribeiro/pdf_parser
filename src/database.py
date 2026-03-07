"""
Conexão e operações MongoDB com suporte a vetores.

Testes: tests/test_database.py
"""

import logging
import math
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.results import DeleteResult, UpdateResult

from src.models.mongodb import (
    JobStatus,
)

logger = logging.getLogger(__name__)


class MongoDBConnection:
    """Gerenciador de conexão MongoDB."""

    def __init__(self):
        """Inicializa o gerenciador de conexão."""
        uri = os.getenv("DOC_PARSER_MONGODB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("DOC_PARSER_MONGODB_DB", "caseiro_docs")

        self.uri = uri
        self.db_name = db_name
        self.client: MongoClient | None = None
        self.db: Any | None = None

    def connect(self) -> MongoClient:
        """
        Conectar ao MongoDB.

        Returns:
            Instância de MongoClient
        """
        if self.client:
            logger.info("Já existe conexão MongoDB ativa")
            return self.client

        try:
            self.client = MongoClient(
                self.uri, serverSelectionTimeoutMS=5000, socketTimeoutMS=5000
            )

            # Testar conexão
            self.client.admin.command("ping")

            # Selecionar banco de dados
            self.db = self.client[self.db_name]

            logger.info(f"Conectado ao MongoDB: {self.uri}")

            # Criar índices para vetores se necessário
            self._create_vector_index()

            return self.client

        except ConnectionFailure as e:
            logger.error(f"Falha ao conectar ao MongoDB: {e}")
            raise
        except OperationFailure as e:
            logger.error(f"Erro de operação ao conectar ao MongoDB: {e}")
            raise

    def disconnect(self):
        """Desconectar do MongoDB."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("Desconectado do MongoDB")

    def __enter__(self):
        """Context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.disconnect()

    def _create_vector_index(self):
        """Criar índice de vetores se necessário."""
        try:
            # MongoDB 7.0+ suporta vetores nativos
            if hasattr(self.db, "embeddings"):
                self.db.embeddings.create_index(
                    "vector",
                    name="vector_vector_2dsphere",
                    background=True,
                    sparse=True,
                )
                logger.info("Índice de vetor criado")
        except Exception as e:
            logger.warning(f"Falha ao criar índice de vetor: {e}")


class VectorIndexManager:
    """Gerenciador de índices de vetores."""

    @staticmethod
    def create_vector_index(collection, vector_field: str = "vector"):
        """
        Criar índice de vetores para busca aproximada.

        Args:
            collection: Coleção MongoDB
            vector_field: Nome do campo de vetor
        """
        try:
            # MongoDB 7.0+ suporta vetores nativos
            collection.create_index(
                vector_field,
                name=f"vector_{vector_field}_2dsphere",
                background=True,
                sparse=True,
            )
            logger.info(f"Índice de vetor criado para {vector_field}")
        except Exception as e:
            logger.warning(f"Falha ao criar índice de vetor: {e}")

    @staticmethod
    def drop_vector_index(collection, vector_field: str = "vector"):
        """
        Remover índice de vetores.

        Args:
            collection: Coleção MongoDB
            vector_field: Nome do campo de vetor
        """
        try:
            index_names = collection.index_information()
            for name, idx_info in index_names.items():
                if vector_field in name:
                    collection.drop_index(name)
                    logger.info(f"Índice de vetor removido: {name}")
        except Exception as e:
            logger.warning(f"Falha ao remover índice de vetor: {e}")


class Database:
    """Operações de banco de dados."""

    def __init__(self):
        """Inicializa o banco de dados."""
        self.conn = MongoDBConnection()

    def get_collection(self, name: str) -> Any:
        """
        Obter coleção do banco de dados.

        Args:
            name: Nome da coleção

        Returns:
            Instância de coleção
        """
        self.conn.connect()
        return self.conn.db[name]

    def get_jobs_collection(self):
        """Obter coleção de jobs."""
        return self.get_collection("jobs")

    def get_documents_collection(self):
        """Obter coleção de documentos."""
        return self.get_collection("documents")

    def get_embeddings_collection(self):
        """Obter coleção de embeddings."""
        return self.get_collection("embeddings")

    def get_chunks_collection(self):
        """Obter coleção de chunks."""
        return self.get_collection("chunks")

    def get_webhooks_collection(self):
        """Obter coleção de webhooks."""
        return self.get_collection("webhooks")


# Funções de negócios


def create_job_document(
    job_id: str,
    file_content: bytes,
    generate_embeddings: bool = True,
    webhook_url: str | None = None,
) -> str:
    """
    Criar documento de job no MongoDB.

    Args:
        job_id: ID do job
        file_content: Conteúdo do arquivo
        generate_embeddings: Gerar embeddings
        webhook_url: URL do webhook

    Returns:
        ID do documento inserido
    """
    db = Database()
    jobs = db.get_jobs_collection()

    document = {
        "_id": job_id,
        "job_id": job_id,
        "file_size": len(file_content),
        "file_content_sha256": None,
        "status": JobStatus.PENDING,
        "file_content_base64": None,
        "generated_at": datetime.now(UTC).isoformat(),
        "generate_embeddings": generate_embeddings,
        "webhook_url": webhook_url,
    }

    result = jobs.insert_one(document)
    logger.info(f"Documento de job criado: {job_id}")

    return str(result.inserted_id)


def update_job_status(
    job_id: str,
    status: JobStatus,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> UpdateResult:
    """
    Atualizar status de job no MongoDB.

    Args:
        job_id: ID do job
        status: Status atualizado
        result: Resultado do job
        error: Erro do job

    Returns:
        Resultado da atualização
    """
    db = Database()
    jobs = db.get_jobs_collection()

    update_data = {
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
    }

    if result:
        update_data["result"] = result

    if error:
        update_data["error"] = error

    # Marcar embeddings como gerados
    if status == JobStatus.COMPLETED and update_data.get("result"):
        update_data["embeddings_generated"] = True

    jobs.update_one({"_id": job_id}, {"$set": update_data})

    logger.info(f"Status de job atualizado: {job_id} -> {status}")

    return jobs.update_one({"_id": job_id}, {"$set": update_data})


def get_job(job_id: str) -> dict[str, Any] | None:
    """
    Obter job pelo ID.

    Args:
        job_id: ID do job

    Returns:
        Documento do job ou None
    """
    db = Database()
    jobs = db.get_jobs_collection()

    job = jobs.find_one({"_id": job_id})

    if job:
        # Converter timestamp para datetime
        if job.get("generated_at"):
            job["created_at"] = datetime.fromisoformat(job["generated_at"])
        if job.get("updated_at"):
            job["updated_at"] = datetime.fromisoformat(job["updated_at"])

        # Mapear _id para id para compatibilidade com schemas
        job["id"] = job.pop("_id")

        return job

    return None


def get_jobs_by_status(status: JobStatus) -> list[dict[str, Any]]:
    """
    Obter jobs por status.

    Args:
        status: Status dos jobs

    Returns:
        Lista de documentos de jobs
    """
    db = Database()
    jobs = db.get_jobs_collection()

    jobs_list = list(jobs.find({"status": status.value}))

    for job in jobs_list:
        if job.get("generated_at"):
            job["created_at"] = datetime.fromisoformat(job["generated_at"])
        if job.get("updated_at"):
            job["updated_at"] = datetime.fromisoformat(job["updated_at"])
        # Mapear _id para id
        job["id"] = job.pop("_id")

    return jobs_list


def save_document(
    doc_id: str, document: dict[str, Any], processing_date: datetime | None = None
) -> dict[str, Any]:
    """
    Salvar documento processado no MongoDB.

    Args:
        doc_id: ID do documento
        document: Documento processado
        processing_date: Data de processamento

    Returns:
        Documento salvo
    """
    db = Database()
    docs = db.get_documents_collection()

    doc_data = {
        "_id": doc_id,
        "doc_id": doc_id,
        "source_file": document.get("source_file", "unknown"),
        "total_pages": document.get("total_pages", 0),
        "pages": document.get("pages", []),
        "status": JobStatus.COMPLETED,
        "processing_date": processing_date.isoformat() if processing_date else None,
        "created_at": datetime.now(UTC).isoformat(),
    }

    result = docs.insert_one(doc_data)
    logger.info(f"Documento salvo: {doc_id}")

    return doc_data


def save_embedding(
    job_id: str,
    doc_id: str,
    vector: list[float],
    chunk_size: int | None = None,
    page: int | None = None,
    block_id: str | None = None,
) -> dict[str, Any]:
    """
    Salvar embedding no MongoDB.

    Args:
        job_id: ID do job
        doc_id: ID do documento
        vector: Vetor de embedding
        chunk_size: Tamanho do chunk
        page: Número da página
        block_id: ID do bloco

    Returns:
        Documento do embedding
    """
    db = Database()
    embeddings = db.get_embeddings_collection()

    # Garante _id único para múltiplos embeddings do mesmo job/doc.
    suffix = block_id or uuid4().hex
    page_part = f"_p{page}" if page is not None else ""
    embedding_id = f"{job_id}_{doc_id}{page_part}_{suffix}"

    embedding_data = {
        "_id": embedding_id,
        "job_id": job_id,
        "doc_id": doc_id,
        "vector": vector,
        "model": "nomic-embed-text",
        "chunk_size": chunk_size,
        "page": page,
        "block_id": block_id,
        "created_at": datetime.now(UTC).isoformat(),
    }

    result = embeddings.insert_one(embedding_data)
    logger.info(f"Embedding salvo: {job_id}_{doc_id}")

    return embedding_data


def save_chunk(
    job_id: str,
    doc_id: str,
    text: str,
    vector: list[float],
    page: int,
    block_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Salvar chunk no MongoDB.

    Args:
        job_id: ID do job
        doc_id: ID do documento
        text: Texto do chunk
        vector: Vetor do chunk
        page: Número da página
        block_id: ID do bloco
        metadata: Metadados do chunk

    Returns:
        Documento do chunk
    """
    db = Database()
    chunks = db.get_chunks_collection()

    chunk_data = {
        "_id": f"{job_id}_{doc_id}_{page}",
        "job_id": job_id,
        "doc_id": doc_id,
        "text": text,
        "vector": vector,
        "page": page,
        "block_id": block_id,
        "metadata": metadata or {},
        "created_at": datetime.now(UTC).isoformat(),
    }

    result = chunks.insert_one(chunk_data)
    logger.info(f"Chunk salvo: {job_id}_{doc_id}_{page}")

    return chunk_data


def vector_near_search(
    job_id: str,
    query_vector: list[float],
    top_k: int = 10,
    min_score: float = 0.5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Buscar documentos similares por vetor.

    Args:
        job_id: ID do job
        query_vector: Vetor de busca
        top_k: Número máximo de resultados
        min_score: Score mínimo
        filters: Filtros adicionais

    Returns:
        Lista de resultados
    """
    db = Database()
    embeddings = db.get_embeddings_collection()

    # Construir pipeline de agregação
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_vector_2dsphere",
                "queryVector": query_vector,
                "topK": top_k,
                "numCandidates": min(top_k * 4, 1000),
                "path": "vector",
                "filter": filters,
            }
        },
        {
            "$project": {
                "_id": 1,
                "doc_id": 1,
                "score": {"$meta": "vectorSearchScore"},
                "model": 1,
                "chunk_size": 1,
                "page": 1,
                "block_id": 1,
                "created_at": 1,
            }
        },
    ]

    try:
        results = list(embeddings.aggregate(pipeline))
    except Exception as exc:
        if "$vectorSearch stage is only allowed on MongoDB Atlas" not in str(exc):
            raise

        def _cosine_similarity(a: list[float], b: list[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        fallback_results: list[dict[str, Any]] = []
        mongo_filter = filters or {}
        for item in embeddings.find(mongo_filter):
            score = _cosine_similarity(query_vector, item.get("vector", []))
            fallback_results.append(
                {
                    "_id": item.get("_id"),
                    "doc_id": item.get("doc_id"),
                    "score": score,
                    "model": item.get("model"),
                    "chunk_size": item.get("chunk_size"),
                    "page": item.get("page"),
                    "block_id": item.get("block_id"),
                    "created_at": item.get("created_at"),
                }
            )

        fallback_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        results = fallback_results[:top_k]

    # Filtrar por min_score
    filtered_results = [r for r in results if r.get("score", 0) >= min_score]

    # Converter timestamps
    for result in filtered_results:
        if result.get("created_at"):
            try:
                result["created_at"] = datetime.fromisoformat(result["created_at"])
            except (ValueError, TypeError):
                pass

    return filtered_results


def create_webhook(job_id: str, url: str, token: str | None = None) -> dict[str, Any]:
    """
    Criar webhook para notificação.

    Args:
        job_id: ID do job
        url: URL do webhook
        token: Token de autenticação

    Returns:
        Documento do webhook
    """
    db = Database()
    webhooks = db.get_webhooks_collection()

    webhook_data = {
        "_id": job_id,
        "job_id": job_id,
        "url": url,
        "token": token,
        "enabled": True,
        "created_at": datetime.now(UTC).isoformat(),
    }

    result = webhooks.insert_one(webhook_data)
    logger.info(f"Webhook criado: {job_id}")

    return webhook_data


def delete_webhook(job_id: str) -> DeleteResult:
    """
    Deletar webhook.

    Args:
        job_id: ID do job

    Returns:
        Resultado da deleção
    """
    db = Database()
    webhooks = db.get_webhooks_collection()

    return webhooks.delete_one({"_id": job_id})


def delete_old_jobs(job_id: str, max_age_days: int = 30) -> int:
    """
    Deletar jobs antigos.

    Args:
        job_id: ID do job
        max_age_days: Dias máximos de retenção

    Returns:
        Número de jobs deletados
    """
    db = Database()
    jobs = db.get_jobs_collection()

    # Só deletar jobs COMPLETADOS ou FALHADOS
    cutoff_date = datetime.now(UTC)
    cutoff_date = cutoff_date.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_date = cutoff_date - __import__("datetime").timedelta(days=max_age_days)

    deleted_count = jobs.delete_many(
        {
            "_id": job_id,
            "status": {"$in": ["completed", "failed"]},
            "generated_at": {"$lt": cutoff_date.isoformat()},
        }
    )

    logger.info(f"Jobs deletados: {deleted_count.deleted_count}")

    return deleted_count.deleted_count


def delete_old_documents(doc_id: str, max_age_days: int = 30) -> int:
    """
    Deletar documentos antigos.

    Args:
        doc_id: ID do documento
        max_age_days: Dias máximos de retenção

    Returns:
        Número de documentos deletados
    """
    db = Database()
    docs = db.get_documents_collection()

    cutoff_date = datetime.now(UTC)
    cutoff_date = cutoff_date.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_date = cutoff_date - __import__("datetime").timedelta(days=max_age_days)

    deleted_count = docs.delete_many(
        {
            "_id": doc_id,
            "status": "completed",
            "created_at": {"$lt": cutoff_date.isoformat()},
        }
    )

    logger.info(f"Documentos deletados: {deleted_count.deleted_count}")

    return deleted_count.deleted_count


# Singleton para acesso ao banco de dados
db_instance: Database | None = None


def get_db() -> Database:
    """Obter instância do banco de dados (singleton)."""
    global db_instance
    if db_instance is None:
        db_instance = Database()
    return db_instance


def get_collection(collection_name: str) -> Any:
    """Obter coleção (singleton)."""
    return get_db().get_collection(collection_name)


def get_mongodb_connection() -> MongoDBConnection:
    """
    Obter instância da conexão MongoDB (singleton).

    Returns:
        Instância de MongoDBConnection
    """
    global _mongo_connection_instance
    if _mongo_connection_instance is None:
        _mongo_connection_instance = MongoDBConnection()
    return _mongo_connection_instance


# Singleton instances - must be defined after all class definitions
_db_instance: Database | None = None
_mongo_connection_instance: MongoDBConnection | None = None
