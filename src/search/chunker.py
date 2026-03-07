"""
Chunking de texto para busca granular.

Testes: tests/test_chunker.py
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """Representa um chunk de texto."""

    text: str
    page: int | None = None
    doc_id: str | None = None
    block_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Divide texto em chunks para busca semântica."""

    def __init__(
        self,
        chunk_size_chars: int = 1024,
        chunk_overlap_chars: int = 256,
        chunk_size_words: int | None = None,
        separator: str = "\n\n",
    ):
        """
        Inicializa o chunker de texto.

        Args:
            chunk_size_chars: Tamanho máximo de chunk em caracteres
            chunk_overlap_chars: Sobreposição entre chunks
            chunk_size_words: Tamanho máximo de chunk em palavras (opcional)
            separator: Separador entre chunks
        """
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.chunk_size_words = chunk_size_words
        self.separator = separator

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """
        Dividir texto em chunks.

        Args:
            text: Texto a ser dividido
            metadata: Metadados para todos os chunks

        Returns:
            Lista de chunks (strings)
        """
        if not text or len(text.strip()) == 0:
            return []

        chunks = []
        text = text.strip()

        # Tentar dividir por palavras primeiro
        if self.chunk_size_words:
            chunks = self._chunk_by_words(text)
        else:
            chunks = self._chunk_by_chars(text)

        return chunks

    def _chunk_by_chars(self, text: str) -> list[str]:
        """
        Dividir texto por caracteres.

        Args:
            text: Texto a ser dividido

        Returns:
            Lista de chunks
        """
        chunks = []
        text = text.strip()

        if len(text) <= self.chunk_size_chars:
            return [text]

        current_chunk = []
        current_length = 0
        overlap_buffer = ""

        for char in text:
            if current_length + 1 > self.chunk_size_chars:
                # Checar se o buffer tem sobreposição
                if overlap_buffer and len(overlap_buffer) >= self.chunk_overlap_chars:
                    # Criar chunk com buffer
                    chunk = overlap_buffer
                    if chunk.strip():
                        chunks.append(chunk)
                    overlap_buffer = ""
                else:
                    # Criar chunk normal
                    chunk = "".join(current_chunk)
                    if chunk.strip():
                        chunks.append(chunk)

                current_chunk = [char]
                current_length = 1
                continue

            current_chunk.append(char)
            current_length += 1

        # Adicionar último chunk
        chunk = "".join(current_chunk)
        if chunk.strip():
            chunks.append(chunk)

        return chunks

    def _chunk_by_words(self, text: str) -> list[str]:
        """
        Dividir texto por palavras.

        Args:
            text: Texto a ser dividido

        Returns:
            Lista de chunks
        """
        # Remover caracteres especiais e manter estrutura
        words = text.split()
        if not words:
            return []

        chunks = []
        current_chunk = []
        current_words = 0
        overlap_chars = self.chunk_overlap_chars

        for word in words:
            if current_words + len(word) + 1 > self.chunk_size_words:
                # Criar chunk
                chunk = self.separator.join(current_chunk)
                if chunk.strip():
                    chunks.append(chunk)

                # Adicionar overlap por caracteres
                overlap_chars_count = 0
                while current_chunk and overlap_chars_count < overlap_chars:
                    last_word = current_chunk.pop()
                    overlap_chars_count += len(last_word) + 1  # +1 para espaço
                current_chunk = current_chunk[::-1]  # Restaurar ordem

                current_words = (
                    sum(len(w) + 1 for w in current_chunk) if current_chunk else 0
                )
                continue

            current_chunk.append(word)
            current_words += len(word) + 1

        # Adicionar último chunk
        chunk = self.separator.join(current_chunk)
        if chunk.strip():
            chunks.append(chunk)

        return chunks


class PageChunker:
    """Divide documentos por páginas."""

    def __init__(
        self,
        chunk_size_chars: int = 1024,
        chunk_overlap_chars: int = 256,
        preserve_page_boundaries: bool = True,
    ):
        """
        Inicializa o page chunker.

        Args:
            chunk_size_chars: Tamanho máximo de chunk em caracteres
            chunk_overlap_chars: Sobreposição entre chunks
            preserve_page_boundaries: Preservar quebras de página
        """
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.preserve_page_boundaries = preserve_page_boundaries

    def chunk(
        self, pages: list[dict[str, Any]], metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Dividir páginas em chunks.

        Args:
            pages: Lista de páginas com texto
            metadata: Metadados para todos os chunks

        Returns:
            Lista de chunks com página
        """
        chunks = []

        for page in pages:
            page_num = page.get("page", 0)
            text = page.get("text", "")

            # Chunkar texto da página
            page_chunks = TextChunker(
                chunk_size_chars=self.chunk_size_chars,
                chunk_overlap_chars=self.chunk_overlap_chars,
            ).chunk(text)

            for chunk_text in page_chunks:
                chunk_data = {
                    "text": chunk_text,
                    "page": page_num,
                    "doc_id": page.get("doc_id"),
                    "block_id": page.get("block_id"),
                    "metadata": metadata or {},
                }
                chunks.append(chunk_data)

        return chunks


class DocumentChunker:
    """Divide documentos inteiros em chunks."""

    def __init__(
        self,
        strategy: str = "hybrid",
        chunk_size_chars: int = 1024,
        chunk_overlap_chars: int = 256,
        chunk_size_words: int | None = None,
    ):
        """
        Inicializa o document chunker.

        Args:
            strategy: Estratégia de chunking ("hybrid", "document", "page")
            chunk_size_chars: Tamanho máximo de chunk em caracteres
            chunk_overlap_chars: Sobreposição entre chunks
            chunk_size_words: Tamanho máximo de chunk em palavras
        """
        self.strategy = strategy
        self.chunk_size_chars = chunk_size_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.chunk_size_words = chunk_size_words

        self.text_chunker = TextChunker(
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
            chunk_size_words=chunk_size_words,
        )
        self.page_chunker = PageChunker(
            chunk_size_chars=chunk_size_chars, chunk_overlap_chars=chunk_overlap_chars
        )

    def chunk(
        self,
        pages: list[dict[str, Any]],
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dividir documento em chunks.

        Args:
            pages: Lista de páginas com texto
            doc_id: ID do documento
            metadata: Metadados para todos os chunks

        Returns:
            Lista de chunks
        """
        chunks = []

        if self.strategy == "document":
            # Concatenar todo o documento
            full_text = self._concatenate_documents(pages)
            doc_chunks = self.text_chunker.chunk(full_text, metadata or {})

            for i, chunk_text in enumerate(doc_chunks):
                chunks.append(
                    {
                        "text": chunk_text,
                        "doc_id": doc_id,
                        "page": None,
                        "block_id": None,
                        "metadata": metadata or {},
                        "type": "document",
                    }
                )

        elif self.strategy == "page":
            # Chunkar por página
            page_chunks = self.page_chunker.chunk(pages, metadata)

            for chunk in page_chunks:
                chunks.append(
                    {
                        "text": chunk["text"],
                        "page": chunk["page"],
                        "doc_id": doc_id,
                        "block_id": chunk.get("block_id"),
                        "metadata": chunk.get("metadata", {}),
                        "type": "page",
                    }
                )

        elif self.strategy == "hybrid":
            # Estratégia híbrida: documento + chunks de página
            doc_chunks = self.text_chunker.chunk(
                self._concatenate_documents(pages), metadata or {}
            )

            page_chunks = self.page_chunker.chunk(pages, metadata)

            # Combinar chunks
            all_chunks = []

            for chunk_text in doc_chunks:
                all_chunks.append(
                    {
                        "text": chunk_text,
                        "doc_id": doc_id,
                        "page": None,
                        "block_id": None,
                        "metadata": metadata or {},
                        "type": "document",
                    }
                )

            for chunk in page_chunks:
                all_chunks.append(
                    {
                        "text": chunk["text"],
                        "page": chunk["page"],
                        "doc_id": doc_id,
                        "block_id": chunk.get("block_id"),
                        "metadata": chunk.get("metadata", {}),
                        "type": "page",
                    }
                )

            chunks = all_chunks

        return chunks

    def _concatenate_documents(self, pages: list[dict[str, Any]]) -> str:
        """
        Concatenar páginas em um documento único.

        Args:
            pages: Lista de páginas com texto

        Returns:
            Texto concatenado
        """
        texts = []
        for page in pages:
            text = page.get("text", "")
            page_num = page.get("page", 0)
            if text:
                texts.append(f"\n\n--- Page {page_num} ---\n\n{text}\n\n")

        return "".join(texts)


class ChunkEmbedding:
    """Representa um chunk com embedding."""

    def __init__(
        self,
        text: str,
        doc_id: str | None = None,
        page: int | None = None,
        block_id: str | None = None,
        vector: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Inicializa um chunk com embedding.

        Args:
            text: Texto do chunk
            doc_id: ID do documento
            page: Número da página
            block_id: ID do bloco
            vector: Vetor do chunk (se já calculado)
            metadata: Metadados do chunk
        """
        self.text = text
        self.doc_id = doc_id
        self.page = page
        self.block_id = block_id
        self.vector = vector
        self.metadata = metadata or {}

    @classmethod
    def batch(cls, chunks: list[dict[str, Any]]) -> list["ChunkEmbedding"]:
        """
        Criar batch de ChunkEmbedding a partir de dicts.

        Args:
            chunks: Lista de dicts com chunks

        Returns:
            Lista de ChunkEmbedding
        """
        return [
            cls(
                text=chunk.get("text", ""),
                doc_id=chunk.get("doc_id"),
                page=chunk.get("page"),
                block_id=chunk.get("block_id"),
                vector=chunk.get("vector"),
                metadata=chunk.get("metadata", {}),
            )
            for chunk in chunks
        ]


def merge_chunks_by_doc(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Agrupar chunks por documento.

    Args:
        chunks: Lista de chunks

    Returns:
        Dict com chunks agrupados por doc_id
    """
    grouped = {}

    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if not doc_id:
            continue

        if doc_id not in grouped:
            grouped[doc_id] = []

        grouped[doc_id].append(chunk)

    return grouped


def merge_chunks_by_page(
    chunks: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """
    Agrupar chunks por página.

    Args:
        chunks: Lista de chunks

    Returns:
        Dict com chunks agrupados por página
    """
    grouped = {}

    for chunk in chunks:
        page = chunk.get("page")
        if page is None:
            continue

        if page not in grouped:
            grouped[page] = []

        grouped[page].append(chunk)

    return grouped


def clean_chunk(chunk_text: str) -> str:
    """
    Limpar texto do chunk.

    Args:
        chunk_text: Texto do chunk

    Returns:
        Texto limpo
    """
    # Remover espaços extras
    text = re.sub(r"\s+", " ", chunk_text).strip()

    # Remover quebras de linha excessivas
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def calculate_chunk_stats(chunks: list[str]) -> dict[str, Any]:
    """
    Calcular estatísticas de chunks.

    Args:
        chunks: Lista de chunks

    Returns:
        Dicionário com estatísticas
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_size": 0,
            "min_size": 0,
            "max_size": 0,
            "total_chars": 0,
        }

    sizes = [len(chunk) for chunk in chunks]

    return {
        "total_chunks": len(chunks),
        "avg_size": sum(sizes) / len(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "total_chars": sum(sizes),
    }


def get_chunk_id(
    doc_id: str,
    page: int | None = None,
    block_id: str | None = None,
    index: int | None = None,
) -> str:
    """
    Gerar ID único para chunk.

    Args:
        doc_id: ID do documento
        page: Número da página
        block_id: ID do bloco
        index: Índice do chunk

    Returns:
        ID do chunk
    """
    parts = [doc_id]

    if page is not None:
        parts.append(f"p{page}")
    if block_id:
        parts.append(block_id)
    if index is not None:
        parts.append(f"c{index}")

    return "_".join(parts)


def chunk_text_for_search(
    text: str,
    chunk_size_chars: int = 1024,
    chunk_overlap_chars: int = 256,
    min_chunk_size: int = 100,
) -> list[str]:
    """
    Chunkar texto otimizado para busca semântica.

    Args:
        text: Texto a ser chunkado
        chunk_size_chars: Tamanho do chunk
        chunk_overlap_chars: Sobreposição
        min_chunk_size: Tamanho mínimo do chunk

    Returns:
        Lista de chunks otimizados para busca
    """
    if not text or len(text.strip()) < min_chunk_size:
        return []

    # Normalizar texto
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= chunk_size_chars:
        return [text] if text else []

    chunks = []
    current = ""
    overlap = ""

    for char in text:
        if len(current) + 1 > chunk_size_chars:
            # Adicionar overlap
            if len(current) >= chunk_overlap_chars:
                overlap = current[-chunk_overlap_chars:]
            else:
                overlap = ""

            # Criar chunk
            chunk = current
            if chunk.strip() and len(chunk) >= min_chunk_size:
                chunks.append(chunk)

            current = overlap
            continue

        current += char

    # Último chunk
    if current.strip() and len(current) >= min_chunk_size:
        chunks.append(current)

    return chunks
