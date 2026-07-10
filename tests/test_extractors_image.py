from pathlib import Path

from PIL import Image


def test_image_extractor_uses_ocr_stack(monkeypatch, tmp_path: Path):
    from src.formats import image as image_module
    from src.models.schemas import Block, BlockType

    image_path = tmp_path / "scan.png"
    Image.new("RGB", (120, 60), color="white").save(image_path)

    expected_blocks = [
        Block(
            block_id="p1_b1",
            type=BlockType.PARAGRAPH,
            text="OCR text",
            bbox=[0.1, 0.1, 0.9, 0.2],
            confidence=0.9,
        )
    ]

    monkeypatch.setattr(
        image_module,
        "_extract_blocks_with_doctr",
        lambda image_array, width, height: expected_blocks,
    )

    doc = image_module.extract_image(image_path, ocr_engine_type="doctr")

    assert doc.doc_id == "scan"
    assert doc.source_file == "scan.png"
    assert doc.total_pages == 1
    assert doc.pages[0].source == "ocr"
    assert doc.pages[0].width == 120
    assert doc.pages[0].height == 60
    assert doc.pages[0].blocks == expected_blocks
