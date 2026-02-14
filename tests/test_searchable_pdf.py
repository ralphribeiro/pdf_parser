"""
Tests for the searchable PDF exporter.

Follows TDD: each test is written before the corresponding implementation.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pypdf
import pytest

from src.models.schemas import Block, BlockType, Page

# =========================================================================
# Cycle 1: _create_text_overlay
# =========================================================================


class TestCreateTextOverlay:
    """Tests for _create_text_overlay: generates 1-page PDF with invisible text."""

    def _make_ocr_page(self):
        return Page(
            page=3,
            source="ocr",
            blocks=[
                Block(
                    block_id="p3_b1",
                    type=BlockType.PARAGRAPH,
                    text="OCR test text sample",
                    bbox=[0.1, 0.05, 0.9, 0.15],
                    confidence=0.92,
                ),
                Block(
                    block_id="p3_b2",
                    type=BlockType.PARAGRAPH,
                    text="Second line of the scanned document",
                    bbox=[0.1, 0.20, 0.85, 0.30],
                    confidence=0.88,
                ),
            ],
            width=595.0,
            height=842.0,
        )

    def test_returns_non_empty_bytes(self):
        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_ocr_page()
        result = _create_text_overlay(page, 595.0, 842.0)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_returns_valid_pdf(self):
        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_ocr_page()
        result = _create_text_overlay(page, 595.0, 842.0)

        # PDF magic bytes
        assert result[:5] == b"%PDF-"

    def test_pdf_has_one_page(self):
        import io

        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_ocr_page()
        result = _create_text_overlay(page, 595.0, 842.0)

        reader = pypdf.PdfReader(io.BytesIO(result))
        assert len(reader.pages) == 1

    def test_empty_blocks_returns_valid_pdf(self):
        """Page without blocks should still generate valid PDF."""
        from src.exporters.searchable_pdf import _create_text_overlay

        page = Page(
            page=1,
            source="ocr",
            blocks=[],
            width=595.0,
            height=842.0,
        )
        result = _create_text_overlay(page, 595.0, 842.0)

        assert result[:5] == b"%PDF-"


# =========================================================================
# Cycle 2: create_searchable_pdf
# =========================================================================


class TestCreateSearchablePdf:
    """Tests for create_searchable_pdf: generates multi-page searchable PDF."""

    def test_output_file_is_created(self, sample_pdf_path, sample_document, output_dir):
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_output_is_valid_pdf(self, sample_pdf_path, sample_document, output_dir):
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        with open(output_path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-"

    def test_output_has_correct_page_count(
        self, sample_pdf_path, sample_document, output_dir
    ):
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        reader = pypdf.PdfReader(str(output_path))
        assert len(reader.pages) == 3

    def test_digital_pages_preserved(
        self, sample_pdf_path, sample_document, output_dir
    ):
        """Digital pages should keep the original content intact."""
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        reader = pypdf.PdfReader(str(output_path))
        # Page 1 is digital - should have extractable text from the original
        text_p1 = reader.pages[0].extract_text()
        assert "digital" in text_p1.lower() or len(text_p1) > 0


# =========================================================================
# Cycle 3: extractable (searchable) text
# =========================================================================


class TestSearchableTextExtraction:
    """Tests that verify the invisible text is extractable/searchable."""

    def test_ocr_text_extractable_with_pdfplumber(
        self, sample_pdf_path, sample_document, output_dir
    ):
        """The invisible OCR text should be extractable via pdfplumber."""
        import pdfplumber

        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        with pdfplumber.open(str(output_path)) as pdf:
            # Page 3 (index 2) is OCR with known text
            text_p3 = pdf.pages[2].extract_text() or ""

        assert "OCR test text sample" in text_p3

    def test_ocr_text_extractable_with_pypdf(
        self, sample_pdf_path, sample_document, output_dir
    ):
        """The invisible OCR text should be extractable via pypdf."""
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        reader = pypdf.PdfReader(str(output_path))
        text_p3 = reader.pages[2].extract_text()

        assert "OCR test text sample" in text_p3

    def test_multiple_blocks_extractable(
        self, sample_pdf_path, sample_document, output_dir
    ):
        """All OCR blocks should be extractable."""
        import pdfplumber

        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        with pdfplumber.open(str(output_path)) as pdf:
            text_p3 = pdf.pages[2].extract_text() or ""

        assert "OCR test text sample" in text_p3
        assert "Second line of the scanned document" in text_p3

    def test_digital_page_text_unchanged(
        self, sample_pdf_path, sample_document, output_dir
    ):
        """Text of digital pages should not be affected."""
        import pdfplumber

        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        with pdfplumber.open(str(output_path)) as pdf:
            text_p1 = pdf.pages[0].extract_text() or ""

        # Original text from page 1 of the PDF generated by the fixture
        assert "digital" in text_p1.lower()


# =========================================================================
# Cycle 4: _calculate_font_size
# =========================================================================


class TestCalculateFontSize:
    """Tests for _calculate_font_size: text should fit within bbox width."""

    def test_text_fits_within_bbox_width(self):
        from reportlab.pdfbase.pdfmetrics import stringWidth

        from src.exporters.searchable_pdf import _calculate_font_size

        text = "OCR test text sample"
        bbox_width = 476.0  # ~80% of 595 pts

        font_size = _calculate_font_size(text, bbox_width)
        rendered_width = stringWidth(text, "Helvetica", font_size)

        assert rendered_width <= bbox_width

    def test_returns_min_size_for_very_long_text(self):
        from src.exporters.searchable_pdf import _calculate_font_size

        # Very long text — should return minimum size
        text = "A" * 500
        bbox_width = 100.0

        font_size = _calculate_font_size(text, bbox_width)
        assert font_size == 4.0  # default min_size

    def test_returns_max_size_for_short_text(self):
        from src.exporters.searchable_pdf import _calculate_font_size

        text = "OK"
        bbox_width = 500.0  # plenty of space

        font_size = _calculate_font_size(text, bbox_width)
        assert font_size == 16.0  # default max_size

    def test_empty_text_returns_min(self):
        from src.exporters.searchable_pdf import _calculate_font_size

        font_size = _calculate_font_size("", 200.0)
        assert font_size == 4.0

    def test_zero_width_returns_min(self):
        from src.exporters.searchable_pdf import _calculate_font_size

        font_size = _calculate_font_size("Text", 0.0)
        assert font_size == 4.0

    def test_custom_min_max(self):
        from src.exporters.searchable_pdf import _calculate_font_size

        text = "OK"
        font_size = _calculate_font_size(text, 500.0, min_size=6.0, max_size=20.0)
        assert font_size == 20.0


# =========================================================================
# Cycle 5: Pipeline integration
# =========================================================================


class TestPipelineIntegration:
    """Integration tests with DocumentProcessor and process_pdf."""

    def test_save_to_searchable_pdf_method_exists(self):
        """DocumentProcessor should have the save_to_searchable_pdf method."""
        from src.pipeline import DocumentProcessor

        assert hasattr(DocumentProcessor, "save_to_searchable_pdf")

    def test_save_to_searchable_pdf_creates_file(
        self, sample_pdf_path, sample_document, output_dir
    ):
        """save_to_searchable_pdf should create a valid PDF."""
        from src.pipeline import DocumentProcessor

        processor = DocumentProcessor.__new__(DocumentProcessor)
        # Minimal initialization — no OCR engine needed to save PDF
        processor.use_gpu = False
        processor.ocr_engine_type = "doctr"
        processor.ocr_engine = None
        processor.tesseract_engine = None

        output_path = output_dir / "searchable_out.pdf"
        processor.save_to_searchable_pdf(
            sample_document, str(sample_pdf_path), str(output_path)
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify it's a valid PDF with 3 pages
        reader = pypdf.PdfReader(str(output_path))
        assert len(reader.pages) == 3

    def test_process_pdf_save_pdf_param_creates_both_files(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """
        process_pdf with save_pdf=True should create both .json AND _searchable.pdf.

        Uses monkeypatch to avoid initializing real OCR engines.
        """
        from src import pipeline

        # Mock DocumentProcessor to return our sample_document
        class FakeProcessor:
            def __init__(self, **kwargs):
                self.ocr_engine = None
                self.tesseract_engine = None

            def process_document_parallel(self, *args, **kwargs):
                return sample_document

            def process_document(self, *args, **kwargs):
                return sample_document

            def save_to_json(self, document, output_path, **kwargs):
                import json

                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(document.to_json_dict(), f)

            def save_to_searchable_pdf(self, document, pdf_path, output_path):
                from src.exporters.searchable_pdf import create_searchable_pdf

                create_searchable_pdf(pdf_path, document, output_path)

        monkeypatch.setattr(pipeline, "DocumentProcessor", FakeProcessor)

        document = pipeline.process_pdf(
            str(sample_pdf_path),
            output_dir=str(output_dir),
            save_pdf=True,
        )

        json_path = output_dir / f"{document.doc_id}.json"
        pdf_path = output_dir / f"{document.doc_id}_searchable.pdf"

        assert json_path.exists(), "JSON file should be created"
        assert pdf_path.exists(), "Searchable PDF file should be created"

    def test_process_pdf_save_pdf_false_no_pdf(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """process_pdf with save_pdf=False should NOT create _searchable.pdf."""
        from src import pipeline

        class FakeProcessor:
            def __init__(self, **kwargs):
                self.ocr_engine = None
                self.tesseract_engine = None

            def process_document_parallel(self, *args, **kwargs):
                return sample_document

            def process_document(self, *args, **kwargs):
                return sample_document

            def save_to_json(self, document, output_path, **kwargs):
                import json

                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(document.to_json_dict(), f)

            def save_to_searchable_pdf(self, document, pdf_path, output_path):
                from src.exporters.searchable_pdf import create_searchable_pdf

                create_searchable_pdf(pdf_path, document, output_path)

        monkeypatch.setattr(pipeline, "DocumentProcessor", FakeProcessor)

        document = pipeline.process_pdf(
            str(sample_pdf_path),
            output_dir=str(output_dir),
            save_pdf=False,
        )

        pdf_path = output_dir / f"{document.doc_id}_searchable.pdf"
        assert not pdf_path.exists(), "Searchable PDF should NOT be created"


# =========================================================================
# Cycle 6: CLI --pdf / --no-pdf flag
# =========================================================================


class TestCLIPdfFlag:
    """Tests for the --pdf / --no-pdf CLI flag."""

    def _get_parser(self):
        """Import and return the argparse parser from the script."""
        from scripts.process_single import _build_parser

        return _build_parser()

    def test_pdf_flag_sets_true(self):
        parser = self._get_parser()
        args = parser.parse_args(["dummy.pdf", "--pdf"])
        assert args.pdf is True

    def test_no_pdf_flag_sets_false(self):
        parser = self._get_parser()
        args = parser.parse_args(["dummy.pdf", "--no-pdf"])
        assert args.pdf is False

    def test_default_is_none(self):
        """Without explicit flag, should be None (uses config)."""
        parser = self._get_parser()
        args = parser.parse_args(["dummy.pdf"])
        assert args.pdf is None

    def test_pdf_and_no_pdf_mutually_exclusive(self):
        """--pdf and --no-pdf are mutually exclusive."""
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["dummy.pdf", "--pdf", "--no-pdf"])


# =========================================================================
# Cycle 7: config.SEARCHABLE_PDF
# =========================================================================


class TestConfigSearchablePdf:
    """Tests for the SEARCHABLE_PDF configuration option."""

    def test_config_has_searchable_pdf_attribute(self):
        import config

        assert hasattr(config, "SEARCHABLE_PDF")

    def test_config_searchable_pdf_default_is_true(self):
        import config

        assert config.SEARCHABLE_PDF is True

    def test_config_searchable_pdf_is_boolean(self):
        import config

        assert isinstance(config.SEARCHABLE_PDF, bool)


# =========================================================================
# Cycle A1: Block.lines field (schema)
# =========================================================================


class TestBlockLinesField:
    """Tests for the lines field in the Block schema."""

    def test_block_without_lines_still_valid(self):
        """Backward compat: Block without lines should work."""
        block = Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="Some text",
            bbox=[0.1, 0.1, 0.9, 0.3],
            confidence=0.95,
        )
        assert block.lines is None

    def test_block_with_lines_data(self):
        """Block with lines should store per-line data."""
        lines_data = [
            {"text": "First line", "bbox": [0.1, 0.1, 0.9, 0.15]},
            {"text": "Second line", "bbox": [0.1, 0.16, 0.85, 0.21]},
        ]
        block = Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="First line\nSecond line",
            bbox=[0.1, 0.1, 0.9, 0.21],
            confidence=0.95,
            lines=lines_data,
        )
        assert block.lines is not None
        assert len(block.lines) == 2
        assert block.lines[0]["text"] == "First line"
        assert block.lines[1]["bbox"] == [0.1, 0.16, 0.85, 0.21]

    def test_block_lines_serializes_to_json(self):
        """lines should appear in the JSON output."""
        lines_data = [
            {"text": "Line 1", "bbox": [0.1, 0.1, 0.9, 0.15]},
        ]
        block = Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="Line 1",
            bbox=[0.1, 0.1, 0.9, 0.15],
            lines=lines_data,
        )
        d = block.model_dump()
        assert "lines" in d
        assert d["lines"][0]["text"] == "Line 1"

    def test_block_without_lines_json_has_null(self):
        """Block without lines should serialize lines as null."""
        block = Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="Text",
            bbox=[0.1, 0.1, 0.9, 0.3],
        )
        d = block.model_dump()
        assert d["lines"] is None


# =========================================================================
# Cycle A2: OCR parser line data storage
# =========================================================================


class TestOcrParserLineData:
    """Tests to verify that OCR parsers preserve per-line data."""

    def _make_mock_doctr_result(self):
        """Create a mock docTR result with 2 lines in 1 block."""
        from unittest.mock import MagicMock

        word1 = MagicMock()
        word1.value = "First"
        word1.confidence = 0.95

        word2 = MagicMock()
        word2.value = "line"
        word2.confidence = 0.93

        word3 = MagicMock()
        word3.value = "Second"
        word3.confidence = 0.90

        word4 = MagicMock()
        word4.value = "line"
        word4.confidence = 0.92

        line1 = MagicMock()
        line1.words = [word1, word2]
        line1.geometry = ((0.1, 0.1), (0.8, 0.15))

        line2 = MagicMock()
        line2.words = [word3, word4]
        line2.geometry = ((0.1, 0.2), (0.75, 0.25))

        block_data = MagicMock()
        block_data.lines = [line1, line2]

        page = MagicMock()
        page.blocks = [block_data]

        result = MagicMock()
        result.pages = [page]

        return result

    def test_parse_doctr_result_includes_lines(self):
        """_parse_doctr_result should populate block.lines with per-line data."""
        from src.extractors.ocr import _parse_doctr_result

        result = self._make_mock_doctr_result()
        blocks = _parse_doctr_result(
            result, page_number=1, page_width=2894, page_height=4093
        )

        assert len(blocks) >= 1
        block = blocks[0]
        assert block.lines is not None
        assert len(block.lines) == 2
        assert block.lines[0]["text"] == "First line"
        assert block.lines[1]["text"] == "Second line"

    def test_parse_doctr_result_lines_have_bboxes(self):
        """Each line should have its own normalized bbox."""
        from src.extractors.ocr import _parse_doctr_result

        result = self._make_mock_doctr_result()
        blocks = _parse_doctr_result(
            result, page_number=1, page_width=2894, page_height=4093
        )

        block = blocks[0]
        # Line 1: geometry ((0.1, 0.1), (0.8, 0.15))
        assert block.lines[0]["bbox"] == [0.1, 0.1, 0.8, 0.15]
        # Line 2: geometry ((0.1, 0.2), (0.75, 0.25))
        assert block.lines[1]["bbox"] == [0.1, 0.2, 0.75, 0.25]

    def test_parse_doctr_page_result_includes_lines(self):
        """_parse_doctr_page_result (pipeline) should populate block.lines."""
        from src.pipeline import DocumentProcessor

        processor = DocumentProcessor.__new__(DocumentProcessor)
        processor.use_gpu = False
        processor.ocr_engine_type = "doctr"
        processor.ocr_engine = None
        processor.tesseract_engine = None

        # Get the page_result (without pages wrapper)
        result = self._make_mock_doctr_result()
        page_result = result.pages[0]

        blocks = processor._parse_doctr_page_result(
            page_result, page_number=1, page_width=2894, page_height=4093
        )

        assert len(blocks) >= 1
        block = blocks[0]
        assert block.lines is not None
        assert len(block.lines) == 2
        assert block.lines[0]["text"] == "First line"
        assert block.lines[1]["bbox"] == [0.1, 0.2, 0.75, 0.25]


# =========================================================================
# Cycle B1: Per-line rendering in _create_text_overlay
# =========================================================================


class TestPerLineOverlay:
    """Tests for per-line rendering using block.lines."""

    def _make_ocr_page_with_lines(self):
        """Create OCR page with 1 block of 3 lines, each with its own bbox."""
        return Page(
            page=1,
            source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="Line one of block\nLine two of block\nLine three of block",
                    bbox=[0.1, 0.1, 0.9, 0.35],
                    confidence=0.95,
                    lines=[
                        {"text": "Line one of block", "bbox": [0.1, 0.10, 0.9, 0.18]},
                        {"text": "Line two of block", "bbox": [0.1, 0.19, 0.85, 0.27]},
                        {
                            "text": "Line three of block",
                            "bbox": [0.1, 0.28, 0.80, 0.35],
                        },
                    ],
                ),
            ],
            width=595.0,
            height=842.0,
        )

    def test_overlay_with_lines_produces_valid_pdf(self):
        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_ocr_page_with_lines()
        result = _create_text_overlay(page, 595.0, 842.0)

        assert result[:5] == b"%PDF-"

    def test_overlay_with_lines_all_text_extractable(self):
        """All 3 lines should be extractable from the overlay."""
        import pdfplumber

        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_ocr_page_with_lines()
        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            text = pdf.pages[0].extract_text() or ""

        assert "Line one of block" in text
        assert "Line two of block" in text
        assert "Line three of block" in text

    def test_overlay_lines_positioned_top_to_bottom(self):
        """Lines should be positioned from top to bottom (y decreases in PDF)."""
        import pdfplumber

        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_ocr_page_with_lines()
        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            chars = pdf.pages[0].chars

        if chars:
            # Group chars by proximity of y (top)
            line_ys = {}
            for ch in chars:
                # Round to group chars on the same line
                y_key = round(ch["top"], 0)
                if y_key not in line_ys:
                    line_ys[y_key] = []
                line_ys[y_key].append(ch["text"])

            y_positions = sorted(line_ys.keys())
            # There should be at least 3 distinct Y positions
            assert len(y_positions) >= 3, (
                f"Expected >= 3 distinct Y positions, got {len(y_positions)}"
            )

    def test_per_line_x_position_matches_line_bbox(self):
        """Each line should be positioned at its OWN bbox x1, not the block's."""
        import pdfplumber

        from src.exporters.searchable_pdf import _create_text_overlay

        page = Page(
            page=1,
            source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="Left line\nIndented line",
                    bbox=[0.05, 0.1, 0.9, 0.25],
                    confidence=0.9,
                    lines=[
                        # Line 1 starts at x=0.05
                        {"text": "Left line", "bbox": [0.05, 0.10, 0.80, 0.17]},
                        # Line 2 starts at x=0.25 (indented)
                        {"text": "Indented line", "bbox": [0.25, 0.18, 0.90, 0.25]},
                    ],
                ),
            ],
            width=595.0,
            height=842.0,
        )

        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            chars = pdf.pages[0].chars

        # Group chars by Y position (top)
        line_groups = {}
        for ch in chars:
            y_key = round(ch["top"], 0)
            if y_key not in line_groups:
                line_groups[y_key] = []
            line_groups[y_key].append(ch)

        y_positions = sorted(line_groups.keys())
        assert len(y_positions) >= 2, f"Expected 2 lines, got {len(y_positions)}"

        # First char of 1st line should be near x=0.05*595 ~ 29.75
        first_line_x = line_groups[y_positions[0]][0]["x0"]
        # First char of 2nd line should be near x=0.25*595 ~ 148.75
        second_line_x = line_groups[y_positions[1]][0]["x0"]

        # 2nd line should start SIGNIFICANTLY more to the right
        assert second_line_x > first_line_x + 50, (
            f"Second line x ({second_line_x:.1f}) should be >50 pts right of "
            f"first line x ({first_line_x:.1f}) to reflect indent"
        )

    def test_overlay_multiblock_with_lines(self):
        """Multiple blocks with lines should be rendered."""
        import pdfplumber

        from src.exporters.searchable_pdf import _create_text_overlay

        page = Page(
            page=1,
            source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="Block A line 1\nBlock A line 2",
                    bbox=[0.1, 0.05, 0.9, 0.15],
                    confidence=0.9,
                    lines=[
                        {"text": "Block A line 1", "bbox": [0.1, 0.05, 0.9, 0.10]},
                        {"text": "Block A line 2", "bbox": [0.1, 0.11, 0.9, 0.15]},
                    ],
                ),
                Block(
                    block_id="p1_b2",
                    type=BlockType.PARAGRAPH,
                    text="Block B single line",
                    bbox=[0.1, 0.50, 0.85, 0.55],
                    confidence=0.9,
                    lines=[
                        {
                            "text": "Block B single line",
                            "bbox": [0.1, 0.50, 0.85, 0.55],
                        },
                    ],
                ),
            ],
            width=595.0,
            height=842.0,
        )
        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            text = pdf.pages[0].extract_text() or ""

        assert "Block A line 1" in text
        assert "Block A line 2" in text
        assert "Block B single line" in text


# =========================================================================
# Cycle B2: Fallback rendering (blocks without lines data)
# =========================================================================


class TestFallbackOverlay:
    """Tests for fallback: blocks without lines should use uniform distribution."""

    def _make_page_without_lines(self):
        """Page with blocks WITHOUT lines data (backward compat)."""
        return Page(
            page=1,
            source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="Line A in block\nLine B in block",
                    bbox=[0.1, 0.1, 0.9, 0.25],
                    confidence=0.9,
                    lines=None,  # no line data!
                ),
            ],
            width=595.0,
            height=842.0,
        )

    def test_fallback_produces_valid_pdf(self):
        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_page_without_lines()
        result = _create_text_overlay(page, 595.0, 842.0)
        assert result[:5] == b"%PDF-"

    def test_fallback_text_extractable(self):
        """Text should be extractable even without lines data."""
        import pdfplumber

        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_page_without_lines()
        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            text = pdf.pages[0].extract_text() or ""

        assert "Line A in block" in text
        assert "Line B in block" in text

    def test_fallback_multiple_lines_distinct_positions(self):
        """Lines in fallback should be at distinct Y positions."""
        import pdfplumber

        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_page_without_lines()
        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            chars = pdf.pages[0].chars

        if chars:
            y_positions = set()
            for ch in chars:
                y_positions.add(round(ch["top"], 0))
            assert len(y_positions) >= 2, (
                f"Expected >= 2 Y positions, got {len(y_positions)}"
            )


# =========================================================================
# Cycle B3: Font sizing from line height + horizontal scaling
# =========================================================================


class TestFontSizingFromLineHeight:
    """Tests that font_size derives from line height (not width)."""

    def test_font_size_proportional_to_line_height(self):
        """Font size should be proportional to the line bbox height."""
        import pdfplumber

        from src.exporters.searchable_pdf import _create_text_overlay

        # Block with 2 lines of very different heights
        page = Page(
            page=1,
            source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="Small line\nBIG line",
                    bbox=[0.1, 0.1, 0.9, 0.5],
                    confidence=0.9,
                    lines=[
                        # Thin line: 5% of page (~42 pts for A4)
                        {"text": "Small line", "bbox": [0.1, 0.10, 0.9, 0.15]},
                        # Thick line: 25% of page (~210 pts)
                        {"text": "BIG line", "bbox": [0.1, 0.25, 0.9, 0.50]},
                    ],
                ),
            ],
            width=595.0,
            height=842.0,
        )

        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            chars = pdf.pages[0].chars

        if chars:
            # Chars from "Small line" should have smaller font size than "BIG line"
            small_chars = [c for c in chars if c["top"] < 200]  # upper part
            big_chars = [c for c in chars if c["top"] > 200]  # lower part

            if small_chars and big_chars:
                small_size = small_chars[0].get("size", small_chars[0].get("height", 0))
                big_size = big_chars[0].get("size", big_chars[0].get("height", 0))
                assert small_size < big_size, (
                    f"Small line size ({small_size}) should be "
                    f"< big line size ({big_size})"
                )

    def test_horizontal_text_selection_not_vertical(self):
        """Text with lines data should be selectable horizontally."""
        import pdfplumber

        from src.exporters.searchable_pdf import _create_text_overlay

        page = Page(
            page=1,
            source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="This text should be horizontal",
                    bbox=[0.1, 0.1, 0.9, 0.15],
                    confidence=0.9,
                    lines=[
                        {
                            "text": "This text should be horizontal",
                            "bbox": [0.1, 0.1, 0.9, 0.15],
                        },
                    ],
                ),
            ],
            width=595.0,
            height=842.0,
        )

        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            chars = pdf.pages[0].chars

        if chars:
            # All chars should have approximately the same top (horizontal)
            tops = [round(c["top"], 0) for c in chars]
            unique_tops = set(tops)
            assert len(unique_tops) <= 2, (
                f"All chars should be on ~1 line (horizontal), "
                f"got {len(unique_tops)} distinct tops"
            )


# =========================================================================
# Cycle C1: Config for page orientation
# =========================================================================


class TestConfigOrientation:
    """Tests for page orientation configuration options."""

    def test_config_has_assume_straight_pages(self):
        import config

        assert hasattr(config, "ASSUME_STRAIGHT_PAGES")

    def test_config_assume_straight_pages_is_bool(self):
        import config

        assert isinstance(config.ASSUME_STRAIGHT_PAGES, bool)

    def test_config_assume_straight_pages_default_false(self):
        """Default should be False to detect rotated text."""
        import config

        assert config.ASSUME_STRAIGHT_PAGES is False

    def test_config_has_detect_orientation(self):
        import config

        assert hasattr(config, "DETECT_ORIENTATION")

    def test_config_detect_orientation_is_bool(self):
        import config

        assert isinstance(config.DETECT_ORIENTATION, bool)

    def test_config_detect_orientation_default_true(self):
        """Default should be True to automatically correct orientation."""
        import config

        assert config.DETECT_ORIENTATION is True

    def test_config_has_straighten_pages(self):
        import config

        assert hasattr(config, "STRAIGHTEN_PAGES")

    def test_config_straighten_pages_is_bool(self):
        import config

        assert isinstance(config.STRAIGHTEN_PAGES, bool)

    def test_config_straighten_pages_default_true(self):
        import config

        assert config.STRAIGHTEN_PAGES is True


# =========================================================================
# Cycle C2: DocTREngine uses orientation config
# =========================================================================


class TestDocTREngineOrientation:
    """Tests for DocTREngine accepting orientation configuration."""

    def test_engine_passes_assume_straight_pages_to_predictor(self):
        """DocTREngine should use config.ASSUME_STRAIGHT_PAGES in ocr_predictor."""
        from unittest.mock import MagicMock, patch

        mock_predictor = MagicMock()
        mock_predictor_func = MagicMock(return_value=mock_predictor)
        mock_predictor.to = MagicMock(return_value=mock_predictor)

        with patch.dict(
            "sys.modules", {"doctr": MagicMock(), "doctr.models": MagicMock()}
        ):
            import sys

            sys.modules["doctr.models"].ocr_predictor = mock_predictor_func

            # Force reimport to pick up mock
            import importlib

            import src.extractors.ocr as ocr_mod

            importlib.reload(ocr_mod)

            ocr_mod.DocTREngine(device="cpu")

            # Verify ocr_predictor was called with assume_straight_pages
            call_kwargs = mock_predictor_func.call_args
            assert (
                "assume_straight_pages" in call_kwargs.kwargs
                or len(call_kwargs.args) > 3
            ), "ocr_predictor should receive assume_straight_pages"

    def test_engine_passes_detect_orientation_to_predictor(self):
        """DocTREngine should use config.DETECT_ORIENTATION in ocr_predictor."""
        from unittest.mock import MagicMock, patch

        mock_predictor = MagicMock()
        mock_predictor_func = MagicMock(return_value=mock_predictor)
        mock_predictor.to = MagicMock(return_value=mock_predictor)

        with patch.dict(
            "sys.modules", {"doctr": MagicMock(), "doctr.models": MagicMock()}
        ):
            import sys

            sys.modules["doctr.models"].ocr_predictor = mock_predictor_func

            import importlib

            import src.extractors.ocr as ocr_mod

            importlib.reload(ocr_mod)

            ocr_mod.DocTREngine(device="cpu")

            call_kwargs = mock_predictor_func.call_args
            assert "detect_orientation" in call_kwargs.kwargs, (
                "ocr_predictor should receive detect_orientation"
            )
