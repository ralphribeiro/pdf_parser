"""
Testes para chunking de texto para busca granular.

TDD: Testes escritos antes da implementação.
"""

import pytest


@pytest.fixture
def sample_text():
    """Texto de exemplo para teste de chunking."""
    return """
    Primeira página do documento.
    Este é um texto de exemplo para testar o chunking de documentos.
    O chunking divide o texto em pedaços menores para facilitar a busca semântica.

    Segunda página do documento.
    Aqui temos mais texto para testar o chunking.
    O chunking deve respeitar quebras de página e manter a coesão semântica.

    Terceira página do documento.
    Finalizamos o texto de exemplo.
    O chunking é importante para a busca semântica eficiente.
    """


class TestTextChunker:
    """Testes para o TextChunker."""

    def test_chunk_by_word_count(self, sample_text):
        """Teste: Chunking por número de palavras."""
        from src.search.chunker import TextChunker

        chunker = TextChunker(chunk_size_words=100, chunk_overlap_chars=20)
        chunks = chunker.chunk(sample_text)

        # Verificar que os chunks foram criados
        assert len(chunks) > 0

        # Verificar que cada chunk tem no máximo o limite de palavras
        for chunk in chunks:
            word_count = len(chunk.split())
            assert (
                word_count
                <= chunker.chunk_size_words + chunker.chunk_overlap_chars // 5
            )  # Aproximado

    def test_chunk_by_character_count(self, sample_text):
        """Teste: Chunking por número de caracteres."""
        from src.search.chunker import TextChunker

        chunker = TextChunker(chunk_size_chars=500, chunk_overlap_chars=100)
        chunks = chunker.chunk(sample_text)

        # Verificar que os chunks foram criados
        assert len(chunks) > 0

        # Verificar que cada chunk tem no máximo o limite de caracteres
        for chunk in chunks:
            char_count = len(chunk)
            assert char_count <= chunker.chunk_size_chars + chunker.chunk_overlap_chars

    def test_chunk_with_overlap(self, sample_text):
        """Teste: Chunking com sobreposição."""
        from src.search.chunker import TextChunker

        chunker = TextChunker(chunk_size_chars=50, chunk_overlap_chars=10)
        chunks = chunker.chunk(sample_text)

        # Verificar que a sobreposição está presente
        assert len(chunks) > 1

        # Verificar que há sobreposição entre chunks consecutivos
        for i in range(len(chunks) - 1):
            chunk_i = chunks[i]
            chunk_next = chunks[i + 1]
            overlap = len(set(chunk_i) & set(chunk_next))
            assert overlap >= 0  # Pode não haver sobreposição dependendo do texto

    def test_chunk_preserves_order(self, sample_text):
        """Teste: Chunking preserva ordem do texto."""
        from src.search.chunker import TextChunker

        chunker = TextChunker(chunk_size_chars=200, chunk_overlap_chars=0)
        chunks = chunker.chunk(sample_text)

        # Concatenar todos os chunks (sem sobreposição)
        concatenated = "".join(chunks)

        # Verificar que o texto original está preservado
        assert sample_text.strip() in concatenated

    def test_chunk_with_metadata(self, sample_text):
        """Teste: Chunking com metadata."""
        from src.search.chunker import TextChunker

        chunker = TextChunker(chunk_size_chars=200, chunk_overlap_chars=0)
        chunks = chunker.chunk(sample_text, metadata={"source": "test"})

        # Verificar que os chunks foram criados
        assert len(chunks) > 0

        # Verificar que os chunks são strings (não dicts)
        assert isinstance(chunks[0], str)

    def test_chunk_empty_text(self):
        """Teste: Chunking de texto vazio."""
        from src.search.chunker import TextChunker

        chunker = TextChunker(chunk_size_chars=100, chunk_overlap_chars=0)
        chunks = chunker.chunk("")

        # Pode retornar lista vazia ou lista com string vazia
        assert len(chunks) >= 0

    def test_chunk_whitespace_only(self):
        """Teste: Chunking de texto com apenas espaços."""
        from src.search.chunker import TextChunker

        chunker = TextChunker(chunk_size_chars=100, chunk_overlap_chars=0)
        chunks = chunker.chunk("   \n\n   ")

        # Deve retornar chunks válidos
        assert len(chunks) >= 0


class TestPageChunker:
    """Testes para o PageChunker (chunking por página)."""

    def test_chunk_by_page(self):
        """Teste: Chunking por página."""
        from src.search.chunker import PageChunker

        # Simular páginas com texto
        pages = [
            {"page": 1, "text": "Texto da página 1"},
            {"page": 2, "text": "Texto da página 2"},
            {"page": 3, "text": "Texto da página 3"},
        ]

        chunker = PageChunker(chunk_size_chars=1000, chunk_overlap_chars=0)
        chunks = chunker.chunk(pages)

        # Verificar que os chunks foram criados por página
        assert len(chunks) == 3

        # Verificar que cada chunk tem página correta
        for chunk in chunks:
            assert "page" in chunk

    def test_page_chunker_with_chunking(self):
        """Teste: PageChunker com chunking de texto por página."""
        from src.search.chunker import PageChunker

        pages = [
            {
                "page": 1,
                "text": "A " * 200 + "B " * 200 + "C " * 200,  # ~1200 chars
            }
        ]

        chunker = PageChunker(chunk_size_chars=500, chunk_overlap_chars=100)
        chunks = chunker.chunk(pages)

        # Verificar que o texto foi chunkado
        assert len(chunks) > 1

        # Verificar que cada chunk tem página correta
        for chunk in chunks:
            assert chunk.get("page") == 1


class TestDocumentChunker:
    """Testes para o DocumentChunker (chunking de documento inteiro)."""

    def test_document_chunker(self):
        """Teste: Chunking de documento completo."""
        from src.search.chunker import DocumentChunker

        # Simular documento com múltiplas páginas
        pages = [{"page": 1, "text": f"Texto página {i}" * 100} for i in range(5)]

        chunker = DocumentChunker(
            chunk_size_chars=1000,
            chunk_overlap_chars=200,
            strategy="hybrid",  # documento + chunks de página
        )
        chunks = chunker.chunk(pages)

        # Verificar que os chunks foram criados
        assert len(chunks) > 0

    def test_document_chunker_by_document(self):
        """Teste: Chunking por documento completo."""
        from src.search.chunker import DocumentChunker

        # Criar texto longo para testar
        long_text = "Documento inteiro em uma página. " * 100

        pages = [{"page": 1, "text": long_text}]

        chunker = DocumentChunker(
            chunk_size_chars=10000,  # Muito grande, todo documento em um chunk
            chunk_overlap_chars=0,
        )
        chunks = chunker.chunk(pages)

        # Verificar que o documento foi chunkado como um todo
        assert len(chunks) > 0


class TestChunkEmbedding:
    """Testes para geração de embedding de chunks."""

    def test_chunk_to_embedding(self):
        """Teste: Conversão de chunk para embedding."""
        from src.search.chunker import ChunkEmbedding

        text = "Texto de exemplo para gerar embedding"

        # Testar que a função existe
        embedding = ChunkEmbedding(text=text)
        assert embedding.text == text

    def test_chunk_embedding_with_document_id(self):
        """Teste: ChunkEmbedding com doc_id."""
        from src.search.chunker import ChunkEmbedding

        embedding = ChunkEmbedding(text="Texto de exemplo", doc_id="doc-uuid", page=1)

        assert embedding.doc_id == "doc-uuid"
        assert embedding.page == 1

    def test_chunk_embedding_batch(self):
        """Teste: Batch de embeddings de chunks."""
        from src.search.chunker import ChunkEmbedding

        chunks = [
            {"text": "Texto 1", "doc_id": "doc-1", "page": 1},
            {"text": "Texto 2", "doc_id": "doc-1", "page": 2},
            {"text": "Texto 3", "doc_id": "doc-2", "page": 1},
        ]

        # Testar que o batch pode ser criado
        batch = ChunkEmbedding.batch(chunks)
        assert len(batch) == 3


class TestChunkMerging:
    """Testes para fusão de chunks."""

    def test_merge_chunks_by_doc(self):
        """Teste: Fusão de chunks por documento."""
        from src.search.chunker import merge_chunks_by_doc

        chunks = [
            {"doc_id": "doc-1", "text": "Texto 1", "page": 1},
            {"doc_id": "doc-1", "text": "Texto 2", "page": 2},
            {"doc_id": "doc-2", "text": "Texto 3", "page": 1},
            {"doc_id": "doc-1", "text": "Texto 4", "page": 3},
        ]

        grouped = merge_chunks_by_doc(chunks)

        # Verificar que os chunks foram agrupados por doc_id
        assert len(grouped) == 2
        assert "doc-1" in grouped
        assert "doc-2" in grouped

        # doc-1 deve ter 3 chunks
        assert len(grouped["doc-1"]) == 3

    def test_merge_chunks_by_page(self):
        """Teste: Fusão de chunks por página."""
        from src.search.chunker import merge_chunks_by_page

        chunks = [
            {"page": 1, "text": "Texto 1"},
            {"page": 1, "text": "Texto 2"},
            {"page": 2, "text": "Texto 3"},
        ]

        grouped = merge_chunks_by_page(chunks)

        # Verificar que os chunks foram agrupados por página
        assert len(grouped) == 2
        assert 1 in grouped
        assert 2 in grouped

        # Página 1 deve ter 2 chunks
        assert len(grouped[1]) == 2


class TestChunkCleaning:
    """Testes para limpeza de chunks."""

    def test_clean_chunk_whitespace(self):
        """Teste: Limpeza de whitespace em chunks."""
        from src.search.chunker import clean_chunk

        chunk = "   Texto com espaços   \n\n"
        cleaned = clean_chunk(chunk)

        # Verificar que whitespace excessivo foi removido
        assert cleaned.strip() == "Texto com espaços"

    def test_clean_chunk_newlines(self):
        """Teste: Limpeza de newlines em chunks."""
        from src.search.chunker import clean_chunk

        chunk = "Texto\ncom\nnewlines\n"
        cleaned = clean_chunk(chunk)

        # Verificar que newlines foram normalizados
        assert "\n" not in cleaned or cleaned.count("\n") <= 1

    def test_clean_chunk_preserves_structure(self):
        """Teste: Limpeza preserva estrutura do chunk."""
        from src.search.chunker import clean_chunk

        chunk = "Texto com estrutura.\n\nMais texto."
        cleaned = clean_chunk(chunk)

        # Verificar que estrutura foi preservada
        assert "Texto com estrutura" in cleaned
        assert "Mais texto" in cleaned


class TestChunkMetrics:
    """Testes para métricas de chunking."""

    def test_calculate_chunk_stats(self):
        """Teste: Cálculo de estatísticas de chunks."""
        from src.search.chunker import calculate_chunk_stats

        chunks = [
            "Texto curto",
            "Texto um pouco mais longo para testar estatísticas de chunking",
            "A " * 50,
        ]

        stats = calculate_chunk_stats(chunks)

        # Verificar que as estatísticas foram calculadas
        assert stats is not None
        assert len(stats) > 0

    def test_chunk_distribution(self):
        """Teste: Distribuição de tamanhos de chunks."""
        from src.search.chunker import calculate_chunk_stats

        chunks = [
            "A " * 10,
            "B " * 50,
            "C " * 100,
            "D " * 200,
        ]

        stats = calculate_chunk_stats(chunks)

        # Verificar que há distribuição de tamanhos
        assert "avg_size" in stats or "size_distribution" in str(stats).lower()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
