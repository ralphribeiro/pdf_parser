"""
Vector Store para MongoDB com vetores nativos.

Testes: tests/test_vector_store.py
"""

import os
from datetime import datetime
from typing import Any

from pymongo import MongoClient
from pymongo.results import DeleteResult, UpdateResult


def get_mongodb_connection() -> MongoClient:
    """
    Obter conexão MongoDB.

    Returns:
        Instância de MongoClient
    """
    uri = os.getenv("DOC_PARSER_MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("DOC_PARSER_MONGODB_DB", "caseiro_docs")

    client = MongoClient(uri, serverSelectionTimeoutMS=5000, socketTimeoutMS=5000)

    # Testar conexão
    client.admin.command("ping")

    return client


class MongoDBVectorStore:
    """Vector Store usando MongoDB com vetores nativos."""

    def __init__(self, collection_name: str, use_vectors: bool = True):
        """
        Inicializa o Vector Store.

        Args:
            collection_name: Nome da coleção
            use_vectors: Usar índices de vetores
        """
        self.collection_name = collection_name
        self.use_vectors = use_vectors
        self.client = get_mongodb_connection()
        self.db = self.client[os.getenv("DOC_PARSER_MONGODB_DB", "caseiro_docs")]
        self.collection = self.db[collection_name]

        # Criar índice de vetores se necessário
        if use_vectors:
            self._ensure_vector_index()

    def _ensure_vector_index(self):
        """Garantir que o índice de vetor existe."""
        try:
            # Verificar se já existe índice
            indexes = self.collection.index_information()
            has_vector_index = any(
                "vector" in idx_name.lower() for idx_name in indexes.keys()
            )

            if not has_vector_index:
                # Criar índice 2dsphere para vetores
                self.collection.create_index(
                    "vector",
                    name="vector_vector_2dsphere",
                    background=True,
                    sparse=True,
                    weights={"vector": 1},
                )
                print("Índice de vetor criado: vector_vector_2dsphere")
        except Exception as e:
            print(f"Aviso: Falha ao criar índice de vetor: {e}")

    def add_documents(
        self, documents: list[dict[str, Any]], embedding_fn=None
    ) -> list[dict[str, Any]]:
        """
        Adicionar documentos ao vector store.

        Args:
            documents: Lista de documentos com campos: doc_id, vector (opcional), metadata
            embedding_fn: Função para gerar embedding (opcional)

        Returns:
            Lista de documentos inseridos
        """
        inserted = []

        for doc in documents:
            try:
                # Se não tiver vector e houver embedding_fn, gerar
                if "vector" not in doc and embedding_fn:
                    try:
                        doc["vector"] = embedding_fn(doc.get("text", ""))
                    except Exception as e:
                        print(f"Aviso: Falha ao gerar embedding: {e}")
                        # Continuar sem vector
                        pass

                # Construir documento
                insert_doc = {
                    "doc_id": doc.get("doc_id"),
                    "source_file": doc.get("source_file"),
                    "page": doc.get("page"),
                    "block_id": doc.get("block_id"),
                    "text": doc.get("text"),
                    "vector": doc.get("vector"),
                    "metadata": doc.get("metadata", {}),
                    "created_at": datetime.utcnow(),
                }

                # Só inserir se tiver vector ou não for obrigatório
                if self.use_vectors and "vector" in doc:
                    result = self.collection.insert_one(insert_doc)
                    inserted.append(
                        {
                            "_id": str(result.inserted_id),
                            "doc_id": doc.get("doc_id"),
                            "text": doc.get("text", "")[:100],  # Preview
                        }
                    )
                else:
                    # Sem vector, usar índice normal
                    result = self.collection.insert_one(insert_doc)
                    inserted.append(
                        {
                            "_id": str(result.inserted_id),
                            "doc_id": doc.get("doc_id"),
                            "text": doc.get("text", "")[:100],
                        }
                    )

            except Exception as e:
                print(f"Erro ao inserir documento: {e}")

        return inserted

    def similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Buscar documentos similares por vetor.

        Args:
            query_vector: Vetor de busca
            top_k: Número máximo de resultados
            min_score: Score mínimo de similaridade
            filters: Filtros adicionais

        Returns:
            Lista de resultados ordenados por similaridade
        """
        results = []

        if self.use_vectors and query_vector and "vector" in self.collection.schema:
            # Usar vetoresNearSearch (MongoDB 6.0+)
            try:
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
                            "text": 1,
                            "metadata": 1,
                            "page": 1,
                            "block_id": 1,
                            "created_at": 1,
                        }
                    },
                ]

                results = list(self.collection.aggregate(pipeline))

                # Converter timestamps e scores
                for result in results:
                    # MongoDB retorna score como -cosine_similarity
                    score = result.get("score", 0)
                    # Inverter para ficar 0-1
                    if isinstance(score, (int, float)):
                        result["_score"] = 1 - score
                        result["score"] = result["_score"]

                    # Converter timestamp
                    if isinstance(result.get("created_at"), datetime):
                        result["created_at"] = result["created_at"].isoformat()

                    # Remover timestamp original
                    result.pop("_id", None)

            except Exception as e:
                print(f"Aviso: Erro ao usar vetoresNearSearch: {e}")
                # Fallback para busca normal
                results = self._fallback_search(query_vector, top_k, min_score, filters)
        else:
            # Fallback para busca normal
            results = self._fallback_search(query_vector, top_k, min_score, filters)

        return results

    def _fallback_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Busca fallback sem vetores (ex: texto).

        Args:
            query_vector: Vetor de busca (não será usado)
            top_k: Número máximo de resultados
            min_score: Score mínimo
            filters: Filtros

        Returns:
            Lista de resultados
        """
        # Se não houver vector, retornar documentos recentes
        query = {"text": {"$exists": True}}

        if filters:
            for key, value in filters.items():
                if key != "_score":
                    query[key] = {"$eq": value}

        docs = list(self.collection.find(query).limit(top_k).sort("created_at", -1))

        # Converter timestamps
        for doc in docs:
            if isinstance(doc.get("created_at"), datetime):
                doc["created_at"] = doc["created_at"].isoformat()

        return docs

    def delete_document(self, doc_id: str) -> DeleteResult:
        """
        Deletar documento.

        Args:
            doc_id: ID do documento

        Returns:
            Resultado da deleção
        """
        return self.collection.delete_one({"doc_id": doc_id})

    def delete_documents(self, doc_ids: list[str]) -> int:
        """
        Deletar múltiplos documentos.

        Args:
            doc_ids: Lista de IDs de documentos

        Returns:
            Número de documentos deletados
        """
        if not doc_ids:
            return 0

        query = {"doc_id": {"$in": doc_ids}}
        result = self.collection.delete_many(query)

        return result.deleted_count

    def update_document(
        self,
        doc_id: str,
        updates: dict[str, Any],
        vector_updates: dict[str, Any] | None = None,
    ) -> UpdateResult:
        """
        Atualizar documento.

        Args:
            doc_id: ID do documento
            updates: Atualizações de campos normais
            vector_updates: Atualizações do vetor

        Returns:
            Resultado da atualização
        """
        updates["_updated_at"] = datetime.utcnow()

        if vector_updates:
            updates["$set"] = {
                **self.collection.find_one({"doc_id": doc_id}, {"$set": {}}).get(
                    "$set", {}
                ),
                "vector": vector_updates.get("vector"),
                **{k: v for k, v in vector_updates.items() if k != "vector"},
            }
        else:
            self.collection.update_one(
                {"doc_id": doc_id},
                {"$set": updates, "$currentDate": {"_updated_at": True}},
            )

        return self.collection.update_one({"doc_id": doc_id}, {"$set": updates})

    def update_vector(self, doc_id: str, vector: list[float]) -> UpdateResult:
        """
        Atualizar vetor de documento.

        Args:
            doc_id: ID do documento
            vector: Novo vetor

        Returns:
            Resultado da atualização
        """
        self.collection.update_one(
            {"doc_id": doc_id},
            {"$set": {"vector": vector, "_updated_at": datetime.utcnow()}},
        )

        return self.collection.update_one(
            {"doc_id": doc_id}, {"$set": {"vector": vector}}
        )

    def count_documents(self, filters: dict[str, Any] | None = None) -> int:
        """
        Contar documentos.

        Args:
            filters: Filtros opcionais

        Returns:
            Número de documentos
        """
        if filters:
            return self.collection.count_documents(filters)
        return self.collection.count_documents({})

    def find_by_id(self, doc_id: str) -> dict[str, Any] | None:
        """
        Buscar documento por ID.

        Args:
            doc_id: ID do documento

        Returns:
            Documento ou None
        """
        return self.collection.find_one({"doc_id": doc_id})

    def create_vector_index(
        self, vector_field: str = "vector", options: dict[str, Any] | None = None
    ):
        """
        Criar índice de vetores.

        Args:
            vector_field: Campo de vetor
            options: Opções do índice
        """
        try:
            self.collection.create_index(
                vector_field,
                name=f"vector_{vector_field}_2dsphere",
                background=True,
                sparse=True,
                **options or {},
            )
            print(f"Índice de vetor criado: {vector_field}")
        except Exception as e:
            print(f"Erro ao criar índice: {e}")

    def drop_vector_index(self, vector_field: str = "vector"):
        """
        Remover índice de vetores.

        Args:
            vector_field: Campo de vetor
        """
        try:
            indexes = self.collection.index_information()
            for idx_name in indexes:
                if vector_field in idx_name.lower():
                    self.collection.drop_index(idx_name)
                    print(f"Índice removido: {idx_name}")
        except Exception as e:
            print(f"Erro ao remover índice: {e}")

    def __del__(self):
        """Limpagem."""
        try:
            self.client.close()
        except:
            pass


def cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
    """
    Calcular similaridade cosine.

    Args:
        vector1: Primeiro vetor
        vector2: Segundo vetor

    Returns:
        Similaridade cosine (entre -1 e 1)
    """
    if len(vector1) != len(vector2):
        raise ValueError(f"Dimensões diferentes: {len(vector1)} vs {len(vector2)}")

    if not vector1 or not vector2:
        return 0.0

    # Calcular produto escalar
    dot_product = sum(a * b for a, b in zip(vector1, vector2))

    # Calcular normas
    norm1 = sum(a * a for a in vector1) ** 0.5
    norm2 = sum(b * b for b in vector2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def normalize_vector(vector: list[float]) -> list[float]:
    """
    Normalizar vetor para unidade.

    Args:
        vector: Vetor a normalizar

    Returns:
        Vetor normalizado
    """
    norm = sum(x * x for x in vector) ** 0.5

    if norm == 0:
        return [0.0] * len(vector)

    return [x / norm for x in vector]


def vector_search_pipeline(
    query_vector: list[float],
    collection,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Pipeline de busca vetorial.

    Args:
        query_vector: Vetor de busca
        collection: Coleção MongoDB
        top_k: Número máximo de resultados
        filters: Filtros

    Returns:
        Lista de resultados
    """
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
                "text": 1,
                "metadata": 1,
                "page": 1,
                "block_id": 1,
                "created_at": 1,
            }
        },
    ]

    return list(collection.aggregate(pipeline))
