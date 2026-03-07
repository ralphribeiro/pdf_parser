"""
Shared fixtures for doc_parser tests.

All fixtures that produce files use tmp_path to avoid
side effects on the project filesystem.
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path (same pattern as scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# Mock other heavy dependencies
# =============================================================================

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

# Load test environment variables before importing app modules
_test_env_file = Path(__file__).parent.parent / ".env.test"
if _test_env_file.exists():
    # Override MongoDB URI with test container
    os.environ["DOC_PARSER_MONGODB_URI"] = "mongodb://test:test123@mongodb-test:27017"
    os.environ["DOC_PARSER_MONGODB_DB"] = "test_caseiro_docs"

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


# ---------------------------------------------------------------------------
# API fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_processor(sample_document):
    """Mocked DocumentProcessor that returns sample_document."""
    processor = MagicMock()
    processor.use_gpu = False
    processor.ocr_engine_type = "doctr"
    processor.ocr_engine = MagicMock()

    processor.process_document_parallel.return_value = sample_document
    processor.process_document.return_value = sample_document

    def fake_save_json(document, output_path, **kwargs):
        import json

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(document.to_json_dict(), f)

    def fake_save_pdf(document, pdf_path, output_path):
        """Generate a minimal valid PDF for tests."""
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen.canvas import Canvas

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        c = Canvas(str(output_path), pagesize=A4)
        c.drawString(100, 700, "searchable test")
        c.save()

    processor.save_to_json.side_effect = fake_save_json
    processor.save_to_searchable_pdf.side_effect = fake_save_pdf

    return processor


@pytest.fixture
def client(mock_processor):
    """FastAPI TestClient with mocked processor."""
    from app.main import create_app

    app = create_app()

    from fastapi.testclient import TestClient

    with TestClient(app) as tc:
        # Override AFTER lifespan initializes (otherwise lifespan overrides)
        app.state.processor = mock_processor
        yield tc


@pytest.fixture
def pdf_bytes(sample_pdf_path):
    """Bytes of a valid PDF for upload."""
    return sample_pdf_path.read_bytes()


@pytest.fixture
def pdf_upload(pdf_bytes):
    """Tuple (filename, file_obj, content_type) for upload."""
    return ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")


@pytest.fixture
def mock_db():
    """Mock MongoDB database for tests."""
    mock_db = MagicMock()
    mock_db.create_collection = MagicMock()
    mock_db.list_collection_names = MagicMock(return_value=[])
    mock_db.get_collection = MagicMock()
    mock_db.command = MagicMock()
    mock_db.admin.command = MagicMock()
    return mock_db
