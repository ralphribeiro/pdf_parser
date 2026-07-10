from pathlib import Path


def test_cli_accepts_supported_text_file(monkeypatch, tmp_path: Path):
    import scripts.process_single as cli

    from src.models.schemas import Document, Page

    source = tmp_path / "notes.txt"
    source.write_text("hello", encoding="utf-8")
    called = {}

    def fake_process_file(path, **kwargs):
        called["path"] = path
        return Document(
            doc_id="notes",
            source_file="notes.txt",
            total_pages=1,
            pages=[Page(page=1, source="digital", blocks=[])],
        )

    monkeypatch.setattr(cli, "process_file", fake_process_file)
    monkeypatch.setattr(cli.sys, "argv", ["process_single.py", str(source), "--quiet"])

    assert cli.main() == 0
    assert called["path"] == str(source)


def test_cli_rejects_unsupported_extension(monkeypatch, tmp_path: Path):
    import scripts.process_single as cli

    source = tmp_path / "notes.exe"
    source.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(cli.sys, "argv", ["process_single.py", str(source), "--quiet"])

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 1
