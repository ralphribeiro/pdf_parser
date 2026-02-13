"""
Fixtures compartilhadas para testes do doc_parser.

Todas as fixtures que produzem arquivos usam tmp_path para evitar
efeitos colaterais no filesystem do projeto.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Adiciona raiz do projeto ao path (mesmo padrão de scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Mock de dependências pesadas que não são necessárias nos testes unitários.
# Isso permite importar src.pipeline e config sem ter torch/doctr instalados.
# ---------------------------------------------------------------------------
_HEAVY_DEPS = [
    "torch", "torch.cuda",
    "numpy", "numpy.ndarray",
    "cv2",
    "skimage", "skimage.filters", "skimage.transform",
    "doctr", "doctr.io", "doctr.models", "doctr.models.predictor",
    "doctr.models.detection", "doctr.models.recognition",
    "transformers",
    "pdf2image", "pdf2image.convert_from_path",
    "camelot",
    "pytesseract",
    "ghostscript",
]
for _mod in _HEAVY_DEPS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# torch.cuda.is_available() deve retornar False nos testes
sys.modules["torch"].cuda.is_available.return_value = False

import pytest
from datetime import datetime
from src.models.schemas import Document, Page, Block, BlockType


# ---------------------------------------------------------------------------
# Fixtures de modelo (Document / Page / Block)
# ---------------------------------------------------------------------------

@pytest.fixture
def ocr_blocks():
    """Blocos OCR realistas (baseados no benchmark), com dados por linha."""
    return [
        Block(
            block_id="p3_b1",
            type=BlockType.PARAGRAPH,
            text="Texto de teste para OCR",
            bbox=[0.1, 0.05, 0.9, 0.15],
            confidence=0.92,
            lines=[
                {"text": "Texto de teste para OCR", "bbox": [0.1, 0.05, 0.9, 0.15]},
            ],
        ),
        Block(
            block_id="p3_b2",
            type=BlockType.PARAGRAPH,
            text="Segunda linha do documento escaneado",
            bbox=[0.1, 0.20, 0.85, 0.30],
            confidence=0.88,
            lines=[
                {"text": "Segunda linha do documento escaneado", "bbox": [0.1, 0.20, 0.85, 0.30]},
            ],
        ),
    ]


@pytest.fixture
def digital_blocks():
    """Blocos de página digital."""
    return [
        Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="Conteúdo digital original",
            bbox=[0.14, 0.10, 0.99, 0.50],
            confidence=1.0,
        ),
    ]


@pytest.fixture
def sample_document(digital_blocks, ocr_blocks):
    """
    Documento com 3 páginas: 2 digitais + 1 OCR.

    Dimensões padrão A4 em pontos (595 x 842).
    """
    page1 = Page(
        page=1, source="digital", blocks=digital_blocks,
        width=595.0, height=842.0,
    )
    page2 = Page(
        page=2, source="digital", blocks=[
            Block(
                block_id="p2_b1",
                type=BlockType.PARAGRAPH,
                text="Página dois digital",
                bbox=[0.1, 0.1, 0.9, 0.5],
                confidence=1.0,
            ),
        ],
        width=595.0, height=842.0,
    )
    page3 = Page(
        page=3, source="ocr", blocks=ocr_blocks,
        width=595.0, height=842.0,
    )

    return Document(
        doc_id="test-doc-001",
        source_file="test-doc-001.pdf",
        total_pages=3,
        processing_date=datetime(2026, 1, 1, 12, 0, 0),
        pages=[page1, page2, page3],
    )


# ---------------------------------------------------------------------------
# Fixtures de arquivo PDF (gerado via reportlab, sem dependência externa)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pdf_path(tmp_path):
    """
    Gera um PDF de 3 páginas no diretório temporário.

    - Página 1: texto digital
    - Página 2: texto digital
    - Página 3: página em branco (simula scan)
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen.canvas import Canvas

    pdf_path = tmp_path / "test-doc-001.pdf"
    c = Canvas(str(pdf_path), pagesize=A4)
    width, height = A4  # 595.27, 841.89

    # Página 1
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, "Conteúdo digital original")
    c.showPage()

    # Página 2
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, "Página dois digital")
    c.showPage()

    # Página 3 (em branco — simula scan)
    c.showPage()

    c.save()
    return pdf_path


@pytest.fixture
def output_dir(tmp_path):
    """Diretório de saída temporário."""
    out = tmp_path / "output"
    out.mkdir()
    return out
