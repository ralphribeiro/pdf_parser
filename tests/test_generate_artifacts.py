"""
Tests for the generate_artifacts orchestration function.

Phase 1 / Milestone 1: defines the high-level contract
    input PDF -> output (searchable PDF + JSON + ArtifactResult)

This is the "contract" function that workers will use in later phases.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pypdf
import pytest

# =========================================================================
# Cycle G1: ArtifactResult data class
# =========================================================================


class TestArtifactResultSchema:
    """Tests for the ArtifactResult data class."""

    def test_artifact_result_importable(self):
        """ArtifactResult should be importable from src.pipeline."""
        from src.pipeline import ArtifactResult

        assert ArtifactResult is not None

    def test_artifact_result_has_json_path(self, tmp_path):
        """ArtifactResult should store the JSON output path."""
        from unittest.mock import MagicMock

        from src.pipeline import ArtifactResult

        json_path = tmp_path / "test.json"
        result = ArtifactResult(
            json_path=json_path,
            pdf_path=tmp_path / "test.pdf",
            document=MagicMock(),
        )
        assert result.json_path == json_path

    def test_artifact_result_has_pdf_path(self, tmp_path):
        """ArtifactResult should store the searchable PDF output path."""
        from unittest.mock import MagicMock

        from src.pipeline import ArtifactResult

        pdf_path = tmp_path / "test.pdf"
        result = ArtifactResult(
            json_path=tmp_path / "test.json",
            pdf_path=pdf_path,
            document=MagicMock(),
        )
        assert result.pdf_path == pdf_path

    def test_artifact_result_has_document(self, tmp_path):
        """ArtifactResult should hold the processed Document."""
        from unittest.mock import MagicMock

        from src.pipeline import ArtifactResult

        doc = MagicMock()
        result = ArtifactResult(
            json_path=tmp_path / "test.json",
            pdf_path=tmp_path / "test.pdf",
            document=doc,
        )
        assert result.document is doc


# =========================================================================
# Cycle G2: generate_artifacts function contract
# =========================================================================


class TestGenerateArtifacts:
    """Tests for the generate_artifacts orchestration function.

    Contract: raw PDF -> (searchable PDF + JSON) always generated.
    """

    def test_function_exists(self):
        """generate_artifacts should be importable from src.pipeline."""
        from src.pipeline import generate_artifacts

        assert callable(generate_artifacts)

    def test_returns_artifact_result(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """generate_artifacts should return an ArtifactResult."""
        from src import pipeline
        from src.pipeline import ArtifactResult, generate_artifacts

        self._patch_processor(monkeypatch, pipeline, sample_document)

        result = generate_artifacts(str(sample_pdf_path), str(output_dir))

        assert isinstance(result, ArtifactResult)

    def test_creates_json_file(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """generate_artifacts should create a JSON file."""
        from src import pipeline
        from src.pipeline import generate_artifacts

        self._patch_processor(monkeypatch, pipeline, sample_document)

        result = generate_artifacts(str(sample_pdf_path), str(output_dir))

        assert result.json_path.exists()
        assert result.json_path.suffix == ".json"

    def test_creates_searchable_pdf_file(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """generate_artifacts should always create a searchable PDF."""
        from src import pipeline
        from src.pipeline import generate_artifacts

        self._patch_processor(monkeypatch, pipeline, sample_document)

        result = generate_artifacts(str(sample_pdf_path), str(output_dir))

        assert result.pdf_path.exists()
        assert result.pdf_path.suffix == ".pdf"

    def test_json_is_valid_document_structure(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """JSON should contain valid Document structure."""
        from src import pipeline
        from src.pipeline import generate_artifacts

        self._patch_processor(monkeypatch, pipeline, sample_document)

        result = generate_artifacts(str(sample_pdf_path), str(output_dir))

        with open(result.json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "doc_id" in data
        assert "pages" in data
        assert "total_pages" in data
        assert data["total_pages"] == 3

    def test_pdf_is_valid(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """Searchable PDF should be a valid PDF file."""
        from src import pipeline
        from src.pipeline import generate_artifacts

        self._patch_processor(monkeypatch, pipeline, sample_document)

        result = generate_artifacts(str(sample_pdf_path), str(output_dir))

        with open(result.pdf_path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_pdf_has_correct_page_count(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """Searchable PDF should have the same page count as the source."""
        from src import pipeline
        from src.pipeline import generate_artifacts

        self._patch_processor(monkeypatch, pipeline, sample_document)

        result = generate_artifacts(str(sample_pdf_path), str(output_dir))

        reader = pypdf.PdfReader(str(result.pdf_path))
        assert len(reader.pages) == 3

    def test_pdf_has_searchable_text(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """Searchable PDF should have extractable text on OCR pages."""
        import pdfplumber

        from src import pipeline
        from src.pipeline import generate_artifacts

        self._patch_processor(monkeypatch, pipeline, sample_document)

        result = generate_artifacts(str(sample_pdf_path), str(output_dir))

        with pdfplumber.open(str(result.pdf_path)) as pdf:
            text_p3 = pdf.pages[2].extract_text() or ""

        assert "OCR test text sample" in text_p3

    def test_document_in_result_matches_json(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """The Document in ArtifactResult should match the JSON output."""
        from src import pipeline
        from src.pipeline import generate_artifacts

        self._patch_processor(monkeypatch, pipeline, sample_document)

        result = generate_artifacts(str(sample_pdf_path), str(output_dir))

        assert result.document.doc_id == sample_document.doc_id
        assert result.document.total_pages == sample_document.total_pages

    def test_output_dir_created_if_not_exists(
        self, sample_pdf_path, sample_document, tmp_path, monkeypatch
    ):
        """generate_artifacts should create the output dir if needed."""
        from src import pipeline
        from src.pipeline import generate_artifacts

        self._patch_processor(monkeypatch, pipeline, sample_document)

        new_dir = tmp_path / "new_output_dir"
        result = generate_artifacts(str(sample_pdf_path), str(new_dir))

        assert new_dir.exists()
        assert result.json_path.exists()
        assert result.pdf_path.exists()

    def test_file_not_found_raises(self, output_dir):
        """generate_artifacts with non-existent PDF should raise FileNotFoundError."""
        from src.pipeline import generate_artifacts

        with pytest.raises(FileNotFoundError):
            generate_artifacts("/nonexistent/file.pdf", str(output_dir))

    @staticmethod
    def _patch_processor(monkeypatch, pipeline_module, sample_document):
        """Patch DocumentProcessor to avoid loading OCR engines."""

        class FakeProcessor:
            def __init__(self, **kwargs):
                self.ocr_engine = None
                self.tesseract_engine = None

            def process_document_parallel(self, *args, **kwargs):
                return sample_document

            def process_document(self, *args, **kwargs):
                return sample_document

            def save_to_json(self, document, output_path, **kwargs):
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(document.to_json_dict(), f)

            def save_to_searchable_pdf(self, document, pdf_path, output_path):
                from src.exporters.searchable_pdf import create_searchable_pdf

                create_searchable_pdf(pdf_path, document, output_path)

        monkeypatch.setattr(pipeline_module, "DocumentProcessor", FakeProcessor)
