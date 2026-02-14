"""
Shared fixtures for doc_parser tests.

All fixtures that produce files use tmp_path to avoid
side effects on the project filesystem.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path (same pattern as scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Mock heavy dependencies that are not needed in unit tests.
# This allows importing src.pipeline and config without torch/doctr installed.
# ---------------------------------------------------------------------------
_HEAVY_DEPS = [
    "torch",
    "torch.cuda",
    "numpy",
    "numpy.ndarray",
    "cv2",
    "skimage",
    "skimage.filters",
    "skimage.transform",
    "doctr",
    "doctr.io",
    "doctr.models",
    "doctr.models.predictor",
    "doctr.models.detection",
    "doctr.models.recognition",
    "transformers",
    "pdf2image",
    "pdf2image.convert_from_path",
    "camelot",
    "pytesseract",
    "ghostscript",
]
for _mod in _HEAVY_DEPS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# torch.cuda.is_available() should return False in tests
sys.modules["torch"].cuda.is_available.return_value = False

from datetime import datetime

import pytest

from src.models.schemas import Block, BlockType, Document, Page

# ---------------------------------------------------------------------------
# Model fixtures (Document / Page / Block)
# ---------------------------------------------------------------------------


@pytest.fixture
def ocr_blocks():
    """Realistic OCR blocks (based on benchmark), with per-line data."""
    return [
        Block(
            block_id="p3_b1",
            type=BlockType.PARAGRAPH,
            text="OCR test text sample",
            bbox=[0.1, 0.05, 0.9, 0.15],
            confidence=0.92,
            lines=[
                {"text": "OCR test text sample", "bbox": [0.1, 0.05, 0.9, 0.15]},
            ],
        ),
        Block(
            block_id="p3_b2",
            type=BlockType.PARAGRAPH,
            text="Second line of the scanned document",
            bbox=[0.1, 0.20, 0.85, 0.30],
            confidence=0.88,
            lines=[
                {
                    "text": "Second line of the scanned document",
                    "bbox": [0.1, 0.20, 0.85, 0.30],
                },
            ],
        ),
    ]


@pytest.fixture
def digital_blocks():
    """Digital page blocks."""
    return [
        Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="Original digital content",
            bbox=[0.14, 0.10, 0.99, 0.50],
            confidence=1.0,
        ),
    ]


@pytest.fixture
def sample_document(digital_blocks, ocr_blocks):
    """
    Document with 3 pages: 2 digital + 1 OCR.

    Standard A4 dimensions in points (595 x 842).
    """
    page1 = Page(
        page=1,
        source="digital",
        blocks=digital_blocks,
        width=595.0,
        height=842.0,
    )
    page2 = Page(
        page=2,
        source="digital",
        blocks=[
            Block(
                block_id="p2_b1",
                type=BlockType.PARAGRAPH,
                text="Page two digital",
                bbox=[0.1, 0.1, 0.9, 0.5],
                confidence=1.0,
            ),
        ],
        width=595.0,
        height=842.0,
    )
    page3 = Page(
        page=3,
        source="ocr",
        blocks=ocr_blocks,
        width=595.0,
        height=842.0,
    )

    return Document(
        doc_id="test-doc-001",
        source_file="test-doc-001.pdf",
        total_pages=3,
        processing_date=datetime(2026, 1, 1, 12, 0, 0),
        pages=[page1, page2, page3],
    )


# ---------------------------------------------------------------------------
# PDF file fixtures (generated via reportlab, no external dependencies)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pdf_path(tmp_path):
    """
    Generate a 3-page PDF in the temporary directory.

    - Page 1: digital text
    - Page 2: digital text
    - Page 3: blank page (simulates scan)
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen.canvas import Canvas

    pdf_path = tmp_path / "test-doc-001.pdf"
    c = Canvas(str(pdf_path), pagesize=A4)
    _width, height = A4  # 595.27, 841.89

    # Page 1
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, "Original digital content")
    c.showPage()

    # Page 2
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, "Page two digital")
    c.showPage()

    # Page 3 (blank — simulates scan)
    c.showPage()

    c.save()
    return pdf_path


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
