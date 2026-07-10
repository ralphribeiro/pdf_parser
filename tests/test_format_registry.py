from pathlib import Path

import pytest


def test_supported_extensions_include_requested_formats():
    from src.formats.registry import SUPPORTED_EXTENSIONS

    assert {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".tiff",
        ".tif",
        ".docx",
        ".odt",
        ".txt",
        ".md",
    }.issubset(SUPPORTED_EXTENSIONS)


@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        ("doc.pdf", b"%PDF-1.7\n", "pdf"),
        ("scan.jpg", b"\xff\xd8\xff\xe0data", "image"),
        ("scan.png", b"\x89PNG\r\n\x1a\nrest", "image"),
        ("scan.tif", b"II*\x00rest", "image"),
        ("notes.txt", b"plain text", "text"),
        ("notes.md", b"# Heading\n\nText", "text"),
    ],
)
def test_detect_format_from_upload_bytes(filename, content, expected):
    from src.formats.registry import detect_format_from_upload

    assert detect_format_from_upload(filename, content).value == expected


def test_rejects_extension_content_mismatch():
    from src.formats.registry import UnsupportedFormatError, detect_format_from_upload

    with pytest.raises(UnsupportedFormatError):
        detect_format_from_upload("not-a-pdf.pdf", b"plain text")


def test_detect_format_from_path(tmp_path: Path):
    from src.formats.registry import DocumentFormat, detect_format

    p = tmp_path / "notes.md"
    p.write_text("# Title", encoding="utf-8")

    assert detect_format(p) is DocumentFormat.TEXT
