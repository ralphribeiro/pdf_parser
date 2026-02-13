"""
Testes para o exportador de PDF pesquisável (searchable PDF).

Segue TDD: cada teste é escrito antes da implementação correspondente.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pypdf

from src.models.schemas import Page, Block, BlockType


# =========================================================================
# Cycle 1: _create_text_overlay
# =========================================================================

class TestCreateTextOverlay:
    """Testes para _create_text_overlay: gera PDF de 1 página com texto invisível."""

    def _make_ocr_page(self):
        return Page(
            page=3,
            source="ocr",
            blocks=[
                Block(
                    block_id="p3_b1",
                    type=BlockType.PARAGRAPH,
                    text="Texto de teste para OCR",
                    bbox=[0.1, 0.05, 0.9, 0.15],
                    confidence=0.92,
                ),
                Block(
                    block_id="p3_b2",
                    type=BlockType.PARAGRAPH,
                    text="Segunda linha do documento escaneado",
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
        from src.exporters.searchable_pdf import _create_text_overlay
        import io

        page = self._make_ocr_page()
        result = _create_text_overlay(page, 595.0, 842.0)

        reader = pypdf.PdfReader(io.BytesIO(result))
        assert len(reader.pages) == 1

    def test_empty_blocks_returns_valid_pdf(self):
        """Página sem blocos ainda deve gerar PDF válido."""
        from src.exporters.searchable_pdf import _create_text_overlay

        page = Page(
            page=1, source="ocr", blocks=[],
            width=595.0, height=842.0,
        )
        result = _create_text_overlay(page, 595.0, 842.0)

        assert result[:5] == b"%PDF-"


# =========================================================================
# Cycle 2: create_searchable_pdf
# =========================================================================

class TestCreateSearchablePdf:
    """Testes para create_searchable_pdf: gera PDF pesquisável multi-página."""

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

    def test_output_has_correct_page_count(self, sample_pdf_path, sample_document, output_dir):
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        reader = pypdf.PdfReader(str(output_path))
        assert len(reader.pages) == 3

    def test_digital_pages_preserved(self, sample_pdf_path, sample_document, output_dir):
        """Páginas digitais devem manter o conteúdo original intacto."""
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        reader = pypdf.PdfReader(str(output_path))
        # Página 1 é digital - deve ter texto extraível do original
        text_p1 = reader.pages[0].extract_text()
        assert "digital original" in text_p1.lower() or len(text_p1) > 0


# =========================================================================
# Cycle 3: texto extraível (searchable)
# =========================================================================

class TestSearchableTextExtraction:
    """Testes que verificam que o texto invisível é extraível/pesquisável."""

    def test_ocr_text_extractable_with_pdfplumber(self, sample_pdf_path, sample_document, output_dir):
        """O texto OCR invisível deve ser extraível via pdfplumber."""
        import pdfplumber
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        with pdfplumber.open(str(output_path)) as pdf:
            # Página 3 (index 2) é OCR com texto conhecido
            text_p3 = pdf.pages[2].extract_text() or ""

        assert "Texto de teste para OCR" in text_p3

    def test_ocr_text_extractable_with_pypdf(self, sample_pdf_path, sample_document, output_dir):
        """O texto OCR invisível deve ser extraível via pypdf."""
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        reader = pypdf.PdfReader(str(output_path))
        text_p3 = reader.pages[2].extract_text()

        assert "Texto de teste para OCR" in text_p3

    def test_multiple_blocks_extractable(self, sample_pdf_path, sample_document, output_dir):
        """Todos os blocos OCR devem ser extraíveis."""
        import pdfplumber
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        with pdfplumber.open(str(output_path)) as pdf:
            text_p3 = pdf.pages[2].extract_text() or ""

        assert "Texto de teste para OCR" in text_p3
        assert "Segunda linha do documento escaneado" in text_p3

    def test_digital_page_text_unchanged(self, sample_pdf_path, sample_document, output_dir):
        """Texto de páginas digitais não deve ser afetado."""
        import pdfplumber
        from src.exporters.searchable_pdf import create_searchable_pdf

        output_path = output_dir / "searchable.pdf"
        create_searchable_pdf(str(sample_pdf_path), sample_document, str(output_path))

        with pdfplumber.open(str(output_path)) as pdf:
            text_p1 = pdf.pages[0].extract_text() or ""

        # Texto original da página 1 do PDF gerado pela fixture
        assert "digital original" in text_p1.lower()


# =========================================================================
# Cycle 4: _calculate_font_size
# =========================================================================

class TestCalculateFontSize:
    """Testes para _calculate_font_size: texto deve caber na largura do bbox."""

    def test_text_fits_within_bbox_width(self):
        from src.exporters.searchable_pdf import _calculate_font_size
        from reportlab.pdfbase.pdfmetrics import stringWidth

        text = "Texto de teste para OCR"
        bbox_width = 476.0  # ~80% de 595 pts

        font_size = _calculate_font_size(text, bbox_width)
        rendered_width = stringWidth(text, "Helvetica", font_size)

        assert rendered_width <= bbox_width

    def test_returns_min_size_for_very_long_text(self):
        from src.exporters.searchable_pdf import _calculate_font_size

        # Texto muito longo — deve retornar tamanho mínimo
        text = "A" * 500
        bbox_width = 100.0

        font_size = _calculate_font_size(text, bbox_width)
        assert font_size == 4.0  # min_size padrão

    def test_returns_max_size_for_short_text(self):
        from src.exporters.searchable_pdf import _calculate_font_size

        text = "OK"
        bbox_width = 500.0  # muito espaço

        font_size = _calculate_font_size(text, bbox_width)
        assert font_size == 16.0  # max_size padrão

    def test_empty_text_returns_min(self):
        from src.exporters.searchable_pdf import _calculate_font_size

        font_size = _calculate_font_size("", 200.0)
        assert font_size == 4.0

    def test_zero_width_returns_min(self):
        from src.exporters.searchable_pdf import _calculate_font_size

        font_size = _calculate_font_size("Texto", 0.0)
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
    """Testes de integração com DocumentProcessor e process_pdf."""

    def test_save_to_searchable_pdf_method_exists(self):
        """DocumentProcessor deve ter o método save_to_searchable_pdf."""
        from src.pipeline import DocumentProcessor
        assert hasattr(DocumentProcessor, "save_to_searchable_pdf")

    def test_save_to_searchable_pdf_creates_file(
        self, sample_pdf_path, sample_document, output_dir
    ):
        """save_to_searchable_pdf deve criar um PDF válido."""
        from src.pipeline import DocumentProcessor

        processor = DocumentProcessor.__new__(DocumentProcessor)
        # Inicialização mínima — não precisa de OCR engine para salvar PDF
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

        # Verifica que é PDF válido com 3 páginas
        reader = pypdf.PdfReader(str(output_path))
        assert len(reader.pages) == 3

    def test_process_pdf_save_pdf_param_creates_both_files(
        self, sample_pdf_path, sample_document, output_dir, monkeypatch
    ):
        """
        process_pdf com save_pdf=True deve criar .json E _searchable.pdf.

        Usa monkeypatch para evitar inicializar OCR engines reais.
        """
        from src import pipeline

        # Mock do DocumentProcessor para retornar nosso sample_document
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
        """process_pdf com save_pdf=False NÃO deve criar _searchable.pdf."""
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
    """Testes para o flag --pdf / --no-pdf no CLI."""

    def _get_parser(self):
        """Importa e retorna o argparse parser do script."""
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
        """Sem flag explícito, deve ser None (usa config)."""
        parser = self._get_parser()
        args = parser.parse_args(["dummy.pdf"])
        assert args.pdf is None

    def test_pdf_and_no_pdf_mutually_exclusive(self):
        """--pdf e --no-pdf são mutuamente exclusivos."""
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["dummy.pdf", "--pdf", "--no-pdf"])


# =========================================================================
# Cycle 7: config.SEARCHABLE_PDF
# =========================================================================

class TestConfigSearchablePdf:
    """Testes para a opção de configuração SEARCHABLE_PDF."""

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
    """Testes para o campo lines no schema Block."""

    def test_block_without_lines_still_valid(self):
        """Backward compat: Block sem lines deve funcionar."""
        block = Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="Texto qualquer",
            bbox=[0.1, 0.1, 0.9, 0.3],
            confidence=0.95,
        )
        assert block.lines is None

    def test_block_with_lines_data(self):
        """Block com lines deve armazenar dados por linha."""
        lines_data = [
            {"text": "Primeira linha", "bbox": [0.1, 0.1, 0.9, 0.15]},
            {"text": "Segunda linha", "bbox": [0.1, 0.16, 0.85, 0.21]},
        ]
        block = Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="Primeira linha\nSegunda linha",
            bbox=[0.1, 0.1, 0.9, 0.21],
            confidence=0.95,
            lines=lines_data,
        )
        assert block.lines is not None
        assert len(block.lines) == 2
        assert block.lines[0]["text"] == "Primeira linha"
        assert block.lines[1]["bbox"] == [0.1, 0.16, 0.85, 0.21]

    def test_block_lines_serializes_to_json(self):
        """lines deve aparecer no JSON de saída."""
        lines_data = [
            {"text": "Linha 1", "bbox": [0.1, 0.1, 0.9, 0.15]},
        ]
        block = Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="Linha 1",
            bbox=[0.1, 0.1, 0.9, 0.15],
            lines=lines_data,
        )
        d = block.model_dump()
        assert "lines" in d
        assert d["lines"][0]["text"] == "Linha 1"

    def test_block_without_lines_json_has_null(self):
        """Block sem lines deve serializar lines como null."""
        block = Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="Texto",
            bbox=[0.1, 0.1, 0.9, 0.3],
        )
        d = block.model_dump()
        assert d["lines"] is None


# =========================================================================
# Cycle A2: OCR parser line data storage
# =========================================================================

class TestOcrParserLineData:
    """Testes para verificar que os parsers OCR preservam dados por linha."""

    def _make_mock_doctr_result(self):
        """Cria um mock de resultado docTR com 2 linhas em 1 bloco."""
        from unittest.mock import MagicMock

        word1 = MagicMock()
        word1.value = "Primeira"
        word1.confidence = 0.95

        word2 = MagicMock()
        word2.value = "linha"
        word2.confidence = 0.93

        word3 = MagicMock()
        word3.value = "Segunda"
        word3.confidence = 0.90

        word4 = MagicMock()
        word4.value = "linha"
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
        """_parse_doctr_result deve popular block.lines com dados por linha."""
        from src.extractors.ocr import _parse_doctr_result

        result = self._make_mock_doctr_result()
        blocks = _parse_doctr_result(result, page_number=1,
                                     page_width=2894, page_height=4093)

        assert len(blocks) >= 1
        block = blocks[0]
        assert block.lines is not None
        assert len(block.lines) == 2
        assert block.lines[0]["text"] == "Primeira linha"
        assert block.lines[1]["text"] == "Segunda linha"

    def test_parse_doctr_result_lines_have_bboxes(self):
        """Cada linha deve ter seu próprio bbox normalizado."""
        from src.extractors.ocr import _parse_doctr_result

        result = self._make_mock_doctr_result()
        blocks = _parse_doctr_result(result, page_number=1,
                                     page_width=2894, page_height=4093)

        block = blocks[0]
        # Linha 1: geometry ((0.1, 0.1), (0.8, 0.15))
        assert block.lines[0]["bbox"] == [0.1, 0.1, 0.8, 0.15]
        # Linha 2: geometry ((0.1, 0.2), (0.75, 0.25))
        assert block.lines[1]["bbox"] == [0.1, 0.2, 0.75, 0.25]

    def test_parse_doctr_page_result_includes_lines(self):
        """_parse_doctr_page_result (pipeline) deve popular block.lines."""
        from src.pipeline import DocumentProcessor

        processor = DocumentProcessor.__new__(DocumentProcessor)
        processor.use_gpu = False
        processor.ocr_engine_type = "doctr"
        processor.ocr_engine = None
        processor.tesseract_engine = None

        # Pega o page_result (sem wrapper de pages)
        result = self._make_mock_doctr_result()
        page_result = result.pages[0]

        blocks = processor._parse_doctr_page_result(
            page_result, page_number=1,
            page_width=2894, page_height=4093
        )

        assert len(blocks) >= 1
        block = blocks[0]
        assert block.lines is not None
        assert len(block.lines) == 2
        assert block.lines[0]["text"] == "Primeira linha"
        assert block.lines[1]["bbox"] == [0.1, 0.2, 0.75, 0.25]


# =========================================================================
# Cycle B1: Per-line rendering in _create_text_overlay
# =========================================================================

class TestPerLineOverlay:
    """Testes para rendering per-line usando block.lines."""

    def _make_ocr_page_with_lines(self):
        """Cria página OCR com 1 bloco de 3 linhas, cada uma com bbox próprio."""
        return Page(
            page=1,
            source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="Linha um do bloco\nLinha dois do bloco\nLinha tres do bloco",
                    bbox=[0.1, 0.1, 0.9, 0.35],
                    confidence=0.95,
                    lines=[
                        {"text": "Linha um do bloco", "bbox": [0.1, 0.10, 0.9, 0.18]},
                        {"text": "Linha dois do bloco", "bbox": [0.1, 0.19, 0.85, 0.27]},
                        {"text": "Linha tres do bloco", "bbox": [0.1, 0.28, 0.80, 0.35]},
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
        """Todas as 3 linhas devem ser extraíveis do overlay."""
        import pdfplumber
        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_ocr_page_with_lines()
        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            text = pdf.pages[0].extract_text() or ""

        assert "Linha um do bloco" in text
        assert "Linha dois do bloco" in text
        assert "Linha tres do bloco" in text

    def test_overlay_lines_positioned_top_to_bottom(self):
        """Linhas devem estar posicionadas de cima para baixo (y decresce no PDF)."""
        from src.exporters.searchable_pdf import _create_text_overlay
        import pdfplumber

        page = self._make_ocr_page_with_lines()
        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            chars = pdf.pages[0].chars

        if chars:
            # Agrupa chars por proximidade de y (top)
            line_ys = {}
            for ch in chars:
                # Arredonda para agrupar chars da mesma linha
                y_key = round(ch["top"], 0)
                if y_key not in line_ys:
                    line_ys[y_key] = []
                line_ys[y_key].append(ch["text"])

            y_positions = sorted(line_ys.keys())
            # Deve haver pelo menos 3 posições Y distintas
            assert len(y_positions) >= 3, f"Expected >= 3 distinct Y positions, got {len(y_positions)}"

    def test_per_line_x_position_matches_line_bbox(self):
        """Cada linha deve estar posicionada no x1 do SEU próprio bbox, não do bloco."""
        import pdfplumber
        from src.exporters.searchable_pdf import _create_text_overlay

        page = Page(
            page=1, source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1", type=BlockType.PARAGRAPH,
                    text="Linha esquerda\nLinha indentada",
                    bbox=[0.05, 0.1, 0.9, 0.25],
                    confidence=0.9,
                    lines=[
                        # Linha 1 começa em x=0.05
                        {"text": "Linha esquerda", "bbox": [0.05, 0.10, 0.80, 0.17]},
                        # Linha 2 começa em x=0.25 (indentada)
                        {"text": "Linha indentada", "bbox": [0.25, 0.18, 0.90, 0.25]},
                    ],
                ),
            ],
            width=595.0, height=842.0,
        )

        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            chars = pdf.pages[0].chars

        # Agrupa chars por posição Y (top)
        line_groups = {}
        for ch in chars:
            y_key = round(ch["top"], 0)
            if y_key not in line_groups:
                line_groups[y_key] = []
            line_groups[y_key].append(ch)

        y_positions = sorted(line_groups.keys())
        assert len(y_positions) >= 2, f"Expected 2 lines, got {len(y_positions)}"

        # Primeiro char da 1a linha deve estar perto de x=0.05*595 ≈ 29.75
        first_line_x = line_groups[y_positions[0]][0]["x0"]
        # Primeiro char da 2a linha deve estar perto de x=0.25*595 ≈ 148.75
        second_line_x = line_groups[y_positions[1]][0]["x0"]

        # A 2a linha deve começar SIGNIFICATIVAMENTE mais à direita
        assert second_line_x > first_line_x + 50, (
            f"Second line x ({second_line_x:.1f}) should be >50 pts right of "
            f"first line x ({first_line_x:.1f}) to reflect indent"
        )

    def test_overlay_multiblock_with_lines(self):
        """Múltiplos blocos com lines devem ser renderizados."""
        import pdfplumber
        from src.exporters.searchable_pdf import _create_text_overlay

        page = Page(
            page=1, source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1", type=BlockType.PARAGRAPH,
                    text="Bloco A linha 1\nBloco A linha 2",
                    bbox=[0.1, 0.05, 0.9, 0.15],
                    confidence=0.9,
                    lines=[
                        {"text": "Bloco A linha 1", "bbox": [0.1, 0.05, 0.9, 0.10]},
                        {"text": "Bloco A linha 2", "bbox": [0.1, 0.11, 0.9, 0.15]},
                    ],
                ),
                Block(
                    block_id="p1_b2", type=BlockType.PARAGRAPH,
                    text="Bloco B unica linha",
                    bbox=[0.1, 0.50, 0.85, 0.55],
                    confidence=0.9,
                    lines=[
                        {"text": "Bloco B unica linha", "bbox": [0.1, 0.50, 0.85, 0.55]},
                    ],
                ),
            ],
            width=595.0, height=842.0,
        )
        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            text = pdf.pages[0].extract_text() or ""

        assert "Bloco A linha 1" in text
        assert "Bloco A linha 2" in text
        assert "Bloco B unica linha" in text


# =========================================================================
# Cycle B2: Fallback rendering (blocks without lines data)
# =========================================================================

class TestFallbackOverlay:
    """Testes para fallback: blocks sem lines devem usar distribuição uniforme."""

    def _make_page_without_lines(self):
        """Página com blocos SEM dados de lines (backward compat)."""
        return Page(
            page=1,
            source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1",
                    type=BlockType.PARAGRAPH,
                    text="Linha A no bloco\nLinha B no bloco",
                    bbox=[0.1, 0.1, 0.9, 0.25],
                    confidence=0.9,
                    lines=None,  # sem dados de line!
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
        """Texto deve ser extraível mesmo sem dados de lines."""
        import pdfplumber
        from src.exporters.searchable_pdf import _create_text_overlay

        page = self._make_page_without_lines()
        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            text = pdf.pages[0].extract_text() or ""

        assert "Linha A no bloco" in text
        assert "Linha B no bloco" in text

    def test_fallback_multiple_lines_distinct_positions(self):
        """Linhas no fallback devem estar em posições Y distintas."""
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
            assert len(y_positions) >= 2, f"Expected >= 2 Y positions, got {len(y_positions)}"


# =========================================================================
# Cycle B3: Font sizing from line height + horizontal scaling
# =========================================================================

class TestFontSizingFromLineHeight:
    """Testes para que font_size derive da altura da linha (não largura)."""

    def test_font_size_proportional_to_line_height(self):
        """Font size deve ser proporcional à altura do bbox da linha."""
        import pdfplumber
        from src.exporters.searchable_pdf import _create_text_overlay

        # Bloco com 2 linhas de alturas bem diferentes
        page = Page(
            page=1, source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1", type=BlockType.PARAGRAPH,
                    text="Linha pequena\nLinha GRANDE",
                    bbox=[0.1, 0.1, 0.9, 0.5],
                    confidence=0.9,
                    lines=[
                        # Linha fina: 5% da página (~42 pts para A4)
                        {"text": "Linha pequena", "bbox": [0.1, 0.10, 0.9, 0.15]},
                        # Linha grossa: 25% da página (~210 pts)
                        {"text": "Linha GRANDE", "bbox": [0.1, 0.25, 0.9, 0.50]},
                    ],
                ),
            ],
            width=595.0, height=842.0,
        )

        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            chars = pdf.pages[0].chars

        if chars:
            # Chars da "Linha pequena" devem ter font size menor que "Linha GRANDE"
            small_chars = [c for c in chars if c["top"] < 200]  # parte superior
            big_chars = [c for c in chars if c["top"] > 200]    # parte inferior

            if small_chars and big_chars:
                small_size = small_chars[0].get("size", small_chars[0].get("height", 0))
                big_size = big_chars[0].get("size", big_chars[0].get("height", 0))
                assert small_size < big_size, (
                    f"Small line size ({small_size}) should be < big line size ({big_size})"
                )

    def test_horizontal_text_selection_not_vertical(self):
        """Texto com lines data deve ser selecionável horizontalmente."""
        import pdfplumber
        from src.exporters.searchable_pdf import _create_text_overlay

        page = Page(
            page=1, source="ocr",
            blocks=[
                Block(
                    block_id="p1_b1", type=BlockType.PARAGRAPH,
                    text="Este texto deve ser horizontal",
                    bbox=[0.1, 0.1, 0.9, 0.15],
                    confidence=0.9,
                    lines=[
                        {"text": "Este texto deve ser horizontal", "bbox": [0.1, 0.1, 0.9, 0.15]},
                    ],
                ),
            ],
            width=595.0, height=842.0,
        )

        result = _create_text_overlay(page, 595.0, 842.0)

        with pdfplumber.open(io.BytesIO(result)) as pdf:
            chars = pdf.pages[0].chars

        if chars:
            # Todos os chars devem ter aproximadamente o mesmo top (horizontal)
            tops = [round(c["top"], 0) for c in chars]
            unique_tops = set(tops)
            assert len(unique_tops) <= 2, (
                f"All chars should be on ~1 line (horizontal), got {len(unique_tops)} distinct tops"
            )


# =========================================================================
# Cycle C1: Config para orientação de página
# =========================================================================

class TestConfigOrientation:
    """Testes para opções de configuração de orientação de página."""

    def test_config_has_assume_straight_pages(self):
        import config
        assert hasattr(config, "ASSUME_STRAIGHT_PAGES")

    def test_config_assume_straight_pages_is_bool(self):
        import config
        assert isinstance(config.ASSUME_STRAIGHT_PAGES, bool)

    def test_config_assume_straight_pages_default_false(self):
        """Padrão deve ser False para detectar texto rotacionado."""
        import config
        assert config.ASSUME_STRAIGHT_PAGES is False

    def test_config_has_detect_orientation(self):
        import config
        assert hasattr(config, "DETECT_ORIENTATION")

    def test_config_detect_orientation_is_bool(self):
        import config
        assert isinstance(config.DETECT_ORIENTATION, bool)

    def test_config_detect_orientation_default_true(self):
        """Padrão deve ser True para corrigir orientação automaticamente."""
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
# Cycle C2: DocTREngine usa config de orientação
# =========================================================================

class TestDocTREngineOrientation:
    """Testes para DocTREngine aceitar configuração de orientação."""

    def test_engine_passes_assume_straight_pages_to_predictor(self):
        """DocTREngine deve usar config.ASSUME_STRAIGHT_PAGES no ocr_predictor."""
        from unittest.mock import patch, MagicMock

        mock_predictor = MagicMock()
        mock_predictor_func = MagicMock(return_value=mock_predictor)
        mock_predictor.to = MagicMock(return_value=mock_predictor)

        with patch.dict('sys.modules', {'doctr': MagicMock(), 'doctr.models': MagicMock()}):
            import sys
            sys.modules['doctr.models'].ocr_predictor = mock_predictor_func

            from src.extractors.ocr import DocTREngine
            # Force reimport to pick up mock
            import importlib
            import src.extractors.ocr as ocr_mod
            importlib.reload(ocr_mod)

            engine = ocr_mod.DocTREngine(device='cpu')

            # Verifica que ocr_predictor foi chamado com assume_straight_pages do config
            call_kwargs = mock_predictor_func.call_args
            assert 'assume_straight_pages' in call_kwargs.kwargs or \
                   len(call_kwargs.args) > 3, \
                "ocr_predictor deve receber assume_straight_pages"

    def test_engine_passes_detect_orientation_to_predictor(self):
        """DocTREngine deve usar config.DETECT_ORIENTATION no ocr_predictor."""
        from unittest.mock import patch, MagicMock

        mock_predictor = MagicMock()
        mock_predictor_func = MagicMock(return_value=mock_predictor)
        mock_predictor.to = MagicMock(return_value=mock_predictor)

        with patch.dict('sys.modules', {'doctr': MagicMock(), 'doctr.models': MagicMock()}):
            import sys
            sys.modules['doctr.models'].ocr_predictor = mock_predictor_func

            import importlib
            import src.extractors.ocr as ocr_mod
            importlib.reload(ocr_mod)

            engine = ocr_mod.DocTREngine(device='cpu')

            call_kwargs = mock_predictor_func.call_args
            assert 'detect_orientation' in call_kwargs.kwargs, \
                "ocr_predictor deve receber detect_orientation"
