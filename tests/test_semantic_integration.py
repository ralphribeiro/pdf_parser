"""
Integration test for semantic indexing + search endpoint.
"""

import sys
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.ingest_api.app import create_app
from services.ingest_api.store import JobStore
from services.search.service import SemanticSearchService
from services.worker.ocr_worker import OcrWorker

from src.models.schemas import Block, BlockType, Document, Page
from src.pipeline import ArtifactResult


class _FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query))]


class _InMemoryVectorStore:
    def __init__(self):
        self._rows = []

    def upsert_chunks(self, chunks, embeddings):
        self._rows = [
            {
                "chunk": chunk,
                "embedding": emb,
            }
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]

    def query(self, query_embedding, n_results, *, job_id=None, min_similarity=None):
        from services.ingest_api.schemas import SearchResult

        q = query_embedding[0]
        results = []
        for row in self._rows:
            chunk = row["chunk"]
            emb = row["embedding"][0]
            similarity = max(0.0, 1.0 - abs(q - emb) / max(q, emb, 1.0))
            if job_id and chunk.job_id != job_id:
                continue
            if min_similarity is not None and similarity < min_similarity:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity=similarity,
                    job_id=chunk.job_id,
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    block_id=chunk.block_id,
                    block_type=chunk.block_type,
                    confidence=chunk.confidence,
                )
            )
        return results[:n_results]


def _artifact_with_document(pdf_path, output_dir, **kwargs):
    document = Document(
        doc_id="doc-1",
        source_file=Path(pdf_path).name,
        total_pages=1,
        processing_date=datetime.now(),
        pages=[
            Page(
                page=1,
                source="digital",
                blocks=[
                    Block(
                        block_id="p1_b1",
                        type=BlockType.PARAGRAPH,
                        text=(
                            "contrato de locacao com vencimento e clausulas "
                            "de multa por inadimplemento contratual"
                        ),
                        bbox=[0.0, 0.0, 1.0, 1.0],
                        confidence=0.99,
                    )
                ],
            )
        ],
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "doc-1.json"
    pdf_out = out_dir / "doc-1_searchable.pdf"
    json_path.write_text("{}")
    pdf_out.write_bytes(b"%PDF-1.4 searchable")
    return ArtifactResult(json_path=json_path, pdf_path=pdf_out, document=document)


class TestSemanticFlow:
    def test_worker_indexes_and_api_searches(self, tmp_path):
        vector_store = _InMemoryVectorStore()
        service = SemanticSearchService(
            embedding_client=_FakeEmbeddingClient(),
            vector_store=vector_store,
        )

        job_store = JobStore()
        upload_dir = tmp_path / "uploads"
        output_dir = tmp_path / "output"
        upload_dir.mkdir()
        output_dir.mkdir()

        worker = OcrWorker(
            store=job_store,
            upload_dir=upload_dir,
            output_dir=output_dir,
            artifact_fn=_artifact_with_document,
            semantic_indexer=service,
        )

        job = job_store.create("invoice.pdf")
        (upload_dir / f"{job.job_id}.pdf").write_bytes(b"%PDF-1.4 fake")
        worker.process_job(job.job_id)

        app = create_app(upload_dir=tmp_path, store=job_store, semantic_search=service)
        response = TestClient(app).post(
            "/search",
            json={"query": "contrato de locacao", "filters": {"job_id": job.job_id}},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["total_matches"] == 1
        assert "contrato de locacao" in payload["results"][0]["text"]
