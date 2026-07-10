from pathlib import Path


def test_docx_extractor_builds_digital_blocks(monkeypatch, tmp_path: Path):
    from src.formats import docx as docx_module

    class Paragraph:
        def __init__(self, text: str):
            self.text = text

    class FakeDocx:
        paragraphs = [
            Paragraph("First paragraph"),
            Paragraph(""),
            Paragraph("Second paragraph"),
        ]

    monkeypatch.setattr(docx_module, "DocxDocument", lambda _path: FakeDocx())

    path = tmp_path / "sample.docx"
    path.write_bytes(b"PK\x03\x04fake")

    doc = docx_module.extract_docx(path)

    assert doc.doc_id == "sample"
    assert doc.source_file == "sample.docx"
    assert doc.pages[0].source == "digital"
    assert [b.text for b in doc.pages[0].blocks] == [
        "First paragraph",
        "Second paragraph",
    ]
