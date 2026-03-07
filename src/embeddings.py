"""
Cliente para API de embeddings (OpenAI-compatible).

Testes: tests/test_embeddings_config.py, tests/test_embeddings.py
"""

import os
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
)
from requests.exceptions import (
    HTTPError,
    RequestException,
)


class EmbeddingError(Exception):
    """Exceção para erros de embedding."""

    pass


class ExtractEmbeddingError(EmbeddingError):
    """Exceção para erro na extração de embedding."""

    pass


class NetworkError(EmbeddingError):
    """Exceção para erro de rede."""

    pass


class EmbeddingsClient:
    """Cliente para API de embeddings OpenAI-compatible."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 1,
    ):
        """
        Inicializa o cliente de embeddings.

        Args:
            base_url: URL da API de embeddings
            model: Modelo de embeddings a usar
            api_key: Chave de API (opcional)
            batch_size: Tamanho do batch para processamento
        """
        self.model = model or os.getenv(
            "DOC_PARSER_EMBEDDINGS_MODEL", "nomic-embed-text"
        )
        self.batch_size = batch_size

        # Configurar URL completa do endpoint e base URL (host)
        if base_url:
            full_url = base_url
        else:
            full_url = os.getenv(
                "DOC_PARSER_EMBEDDINGS_URL", "http://localhost:11434/api/generate"
            )

        parsed = urlparse(full_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.endpoint_url = full_url

        # Configurar API key
        self.api_key = api_key or os.getenv("DOC_PARSER_EMBEDDINGS_API_KEY", "ollama")

        # Validar URL
        try:
            if not self.base_url or not self.endpoint_url:
                raise EmbeddingError("URL inválida")
        except Exception as e:
            raise EmbeddingError(f"URL inválida: {self.endpoint_url} - {e}")

    def generate_embedding(self, text: str, model: str | None = None) -> list[float]:
        """
        Gerar embedding para um texto.

        Args:
            text: Texto a ser embbedado
            model: Modelo específico (ignorado se model do cliente é definido)

        Returns:
            Lista de floats representando o embedding
        """
        model = model or self.model

        request = build_request(text, model=model)

        try:
            response = requests.post(
                self.endpoint_url,
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=60,  # Timeout de 60 segundos
            )

            response.raise_for_status()
            embedding = extract_embedding(response)

            return embedding if embedding else []

        except HTTPError as e:
            raise EmbeddingError(f"Erro HTTP ao gerar embedding: {e}")
        except RequestsConnectionError as e:
            raise NetworkError(f"Erro de conexão com API de embeddings: {e}")
        except RequestException as e:
            raise EmbeddingError(f"Erro ao gerar embedding: {e}")

    def generate_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        Gerar embeddings para múltiplos textos.

        Args:
            texts: Lista de textos a serem embbedados
            model: Modelo específico (ignorado se model do cliente é definido)
            batch_size: Tamanho do batch para processamento

        Returns:
            Lista de listas de floats representando os embeddings
        """
        batch_size = batch_size or self.batch_size
        model = model or self.model

        if not texts:
            return []

        # Processar em batches (para Ollama, podemos fazer um por um)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                batch_embeddings = self._generate_batch_embeddings(batch_texts, model)
                all_embeddings.extend(batch_embeddings)
            except EmbeddingError as e:
                # Continuar com o próximo batch
                print(f"Aviso: Erro ao processar batch {i // batch_size + 1}: {e}")

        return all_embeddings

    def _generate_batch_embeddings(
        self, texts: list[str], model: str
    ) -> list[list[float]]:
        """
        Gerar embeddings para um batch de textos.

        Para Ollama, cada request retorna um embedding por vez.
        """
        embeddings = []

        for text in texts:
            try:
                embedding = self.generate_embedding(text, model)
                embeddings.append(embedding)
            except EmbeddingError:
                # Retornar zeros para falhas
                embeddings.append([0.0] * len(texts))

        return embeddings


def build_request(
    text: str, model: str = "nomic-embed-text", encoding_format: str = "float"
) -> dict[str, Any]:
    """
    Construir request para API de embeddings OpenAI-compatible.

    Args:
        text: Texto a ser embbedado
        model: Modelo de embeddings
        encoding_format: Formato de codificação ("float" ou "base64")

    Returns:
        Dicionário com o request
    """
    return {"model": model, "input": text, "encoding_format": encoding_format}


def extract_embedding(response: requests.Response) -> list[float]:
    """
    Extrair embedding de resposta da API.

    Args:
        response: Resposta da API

    Returns:
        Lista de floats representando o embedding
    """
    try:
        data = response.json()

        # Tentar formatos diferentes de resposta
        # Ollama: {"object": "list", "data": [{"embedding": [...]}]}
        if "data" in data and data["data"] and len(data["data"]) > 0:
            embedding = data["data"][0].get("embedding", [])
            return embedding if embedding else []

        # OpenAI: {"object": "list", "data": [{"embedding": [...]}]}
        if isinstance(data, list) and data:
            embedding = data[0].get("embedding", [])
            return embedding if embedding else []

        # Formato alternativo
        if "embedding" in data:
            return data["embedding"]

        raise ExtractEmbeddingError(f"Formato de resposta inválido: {data}")

    except ValueError as e:
        raise ExtractEmbeddingError(f"Erro ao parsing resposta: {e}")


def cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
    """
    Calcular similaridade cosine entre dois vetores.

    Args:
        vector1: Primeiro vetor
        vector2: Segundo vetor

    Returns:
        Similaridade cosine (entre -1 e 1)
    """
    if len(vector1) != len(vector2):
        raise ValueError(f"Dimenções diferentes: {len(vector1)} vs {len(vector2)}")

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


class EmbeddingCache:
    """Cache para embeddings para evitar regeneração."""

    def __init__(self, max_size: int = 1000):
        """
        Inicializa o cache de embeddings.

        Args:
            max_size: Tamanho máximo do cache
        """
        self.max_size = max_size
        self.cache: dict[str, list[float]] = {}
        self._access_times: dict[str, float] = {}

    def set(self, key: str, embedding: list[float]):
        """Adicionar embedding ao cache."""
        self.cache[key] = embedding
        self._access_times[key] = datetime.now().timestamp()

        # Manter tamanho do cache
        if len(self.cache) > self.max_size:
            self._cleanup()

    def get(self, key: str, default: list[float] | None = None) -> list[float] | None:
        """
        Recuperar embedding do cache.

        Args:
            key: Chave do cache
            default: Valor default se não encontrado

        Returns:
            Embedding ou default
        """
        if key in self.cache:
            # Atualizar tempo de acesso
            self._access_times[key] = datetime.now().timestamp()
            return self.cache[key]
        return default

    def exists(self, key: str) -> bool:
        """Verificar se embedding existe no cache."""
        return key in self.cache

    def clear(self):
        """Limpar cache."""
        self.cache.clear()
        self._access_times.clear()

    def _cleanup(self):
        """Limpar embeddings menos acessados."""
        # Ordenar por tempo de acesso
        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])

        # Remover os menos acessados
        to_remove = len(self.cache) - self.max_size
        for key, _ in sorted_keys[:to_remove]:
            del self.cache[key]
            del self._access_times[key]
