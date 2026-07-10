from pathlib import Path


def test_dispatcher_routes_text_files(tmp_path: Path):
    from src.formats.dispatcher import process_file

    path = tmp_path / "notes.txt"
    path.write_text("Hello dispatcher", encoding="utf-8")

    doc = process_file(path)

    assert doc.doc_id == "notes"
    assert doc.pages[0].blocks[0].text == "Hello dispatcher"
