from pathlib import Path


def test_plain_text_extractor_builds_document_with_paragraph_blocks(tmp_path: Path):
    from src.formats.plain_text import extract_plain_text

    path = tmp_path / "notes.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    doc = extract_plain_text(path)

    assert doc.doc_id == "notes"
    assert doc.source_file == "notes.txt"
    assert doc.total_pages == 1
    assert doc.pages[0].source == "digital"
    assert [b.text for b in doc.pages[0].blocks] == [
        "First paragraph.",
        "Second paragraph.",
    ]
    assert doc.pages[0].blocks[0].bbox[1] < doc.pages[0].blocks[1].bbox[1]


def test_markdown_extractor_preserves_ordered_blocks(tmp_path: Path):
    from src.formats.plain_text import extract_plain_text

    path = tmp_path / "notes.md"
    path.write_text("# Heading\n\n- item one\n- item two", encoding="utf-8")

    doc = extract_plain_text(path)

    assert [b.text for b in doc.pages[0].blocks] == [
        "# Heading",
        "- item one\n- item two",
    ]
