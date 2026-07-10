"""Image extractor using the existing OCR stack."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import config
from src.models.schemas import Block, Document, Page


def extract_image(
    path: str | Path,
    *,
    doc_id: str | None = None,
    ocr_engine_type: str | None = None,
) -> Document:
    """Run OCR on a single image and return the standard Document shape."""
    source_path = Path(path)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        image_array = np.array(image)

    engine_type = ocr_engine_type or getattr(config, "OCR_ENGINE", "doctr")
    if engine_type == "tesseract":
        blocks = _extract_blocks_with_tesseract(source_path)
    else:
        blocks = _extract_blocks_with_doctr(image_array, width, height)

    page = Page(
        page=1,
        source="ocr",
        blocks=blocks,
        width=float(width),
        height=float(height),
    )
    return Document(
        doc_id=doc_id or source_path.stem,
        source_file=source_path.name,
        total_pages=1,
        processing_date=datetime.now(),
        pages=[page],
    )


def _extract_blocks_with_doctr(image_array, width: int, height: int) -> list[Block]:
    from src.extractors.ocr import DocTREngine, _parse_doctr_result

    engine = DocTREngine()
    result = engine.extract_from_image(image_array)
    return _parse_doctr_result(result, 1, float(width), float(height))


def _extract_blocks_with_tesseract(path: Path) -> list[Block]:
    from src.extractors.ocr_tesseract import TesseractEngine, _parse_tesseract_result

    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        engine = TesseractEngine()
        data = engine.extract_from_image(image)
    return _parse_tesseract_result(data, 1, float(width), float(height))
