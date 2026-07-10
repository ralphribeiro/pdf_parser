from pathlib import Path


def test_odt_extractor_builds_digital_blocks(monkeypatch, tmp_path: Path):
    from src.formats import odt as odt_module

    class FakeParagraph:
        pass

    class FakeOdt:
        def getElementsByType(self, _node_type):
            return [FakeParagraph(), FakeParagraph()]

    texts = iter(["First ODT paragraph", "Second ODT paragraph"])
    monkeypatch.setattr(odt_module, "load", lambda _path: FakeOdt())
    monkeypatch.setattr(odt_module, "extractText", lambda _paragraph: next(texts))

    path = tmp_path / "sample.odt"
    path.write_bytes(b"PK\x03\x04fake")

    doc = odt_module.extract_odt(path)

    assert doc.doc_id == "sample"
    assert doc.source_file == "sample.odt"
    assert doc.pages[0].source == "digital"
    assert [b.text for b in doc.pages[0].blocks] == [
        "First ODT paragraph",
        "Second ODT paragraph",
    ]
