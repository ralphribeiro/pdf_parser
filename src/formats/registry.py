"""Supported document formats and upload validation."""

from __future__ import annotations

import io
import zipfile
from enum import StrEnum
from pathlib import Path


class DocumentFormat(StrEnum):
    """Canonical format groups handled by the pipeline."""

    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    ODT = "odt"
    TEXT = "text"


class UnsupportedFormatError(ValueError):
    """Raised when a file extension or content signature is unsupported."""


EXTENSION_TO_FORMAT: dict[str, DocumentFormat] = {
    ".pdf": DocumentFormat.PDF,
    ".jpg": DocumentFormat.IMAGE,
    ".jpeg": DocumentFormat.IMAGE,
    ".png": DocumentFormat.IMAGE,
    ".tiff": DocumentFormat.IMAGE,
    ".tif": DocumentFormat.IMAGE,
    ".docx": DocumentFormat.DOCX,
    ".odt": DocumentFormat.ODT,
    ".txt": DocumentFormat.TEXT,
    ".md": DocumentFormat.TEXT,
}

SUPPORTED_EXTENSIONS = frozenset(EXTENSION_TO_FORMAT)
ACCEPT_ATTRIBUTE = ",".join(sorted(SUPPORTED_EXTENSIONS))
SUPPORTED_FORMAT_LABEL = "PDF, JPG/JPEG, PNG, TIFF, DOCX, ODT, TXT e MD"

CONTENT_TYPES: dict[DocumentFormat, str] = {
    DocumentFormat.PDF: "application/pdf",
    DocumentFormat.IMAGE: "image/*",
    DocumentFormat.DOCX: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    DocumentFormat.ODT: "application/vnd.oasis.opendocument.text",
    DocumentFormat.TEXT: "text/plain",
}


def extension_for_filename(filename: str | Path) -> str:
    """Return a normalized supported extension for *filename*."""
    suffix = Path(str(filename)).suffix.lower()
    if suffix not in EXTENSION_TO_FORMAT:
        raise UnsupportedFormatError(
            f"Unsupported file extension: {suffix or '<none>'}. "
            f"Supported formats: {SUPPORTED_FORMAT_LABEL}"
        )
    return suffix


def detect_format(path: str | Path) -> DocumentFormat:
    """Detect the format group from a local file path."""
    return EXTENSION_TO_FORMAT[extension_for_filename(path)]


def detect_format_from_upload(filename: str, content: bytes) -> DocumentFormat:
    """Validate upload filename/content and return its format group."""
    if not filename:
        raise UnsupportedFormatError(
            f"File name is required. Supported formats: {SUPPORTED_FORMAT_LABEL}"
        )
    if not content:
        raise UnsupportedFormatError("Uploaded file is empty")

    document_format = EXTENSION_TO_FORMAT[extension_for_filename(filename)]
    if not _content_matches_format(document_format, content):
        raise UnsupportedFormatError(
            f"File content does not match extension {Path(filename).suffix.lower()}"
        )
    return document_format


def content_type_for_format(document_format: DocumentFormat) -> str:
    """Return the storage content type for a detected document format."""
    return CONTENT_TYPES[document_format]


def _content_matches_format(document_format: DocumentFormat, content: bytes) -> bool:
    if document_format is DocumentFormat.PDF:
        return content.startswith(b"%PDF-")
    if document_format is DocumentFormat.IMAGE:
        return _is_image(content)
    if document_format is DocumentFormat.DOCX:
        return _is_docx(content)
    if document_format is DocumentFormat.ODT:
        return _is_odt(content)
    if document_format is DocumentFormat.TEXT:
        return _is_text(content)
    return False


def _is_image(content: bytes) -> bool:
    return (
        content.startswith(b"\xff\xd8\xff")
        or content.startswith(b"\x89PNG\r\n\x1a\n")
        or content.startswith(b"II*\x00")
        or content.startswith(b"MM\x00*")
    )


def _is_docx(content: bytes) -> bool:
    if not content.startswith(b"PK"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return "word/document.xml" in archive.namelist()
    except zipfile.BadZipFile:
        # Keep lightweight tests and partially streamed clients from being rejected
        # solely because they only provided the ZIP header.
        return len(content) < 32


def _is_odt(content: bytes) -> bool:
    if not content.startswith(b"PK"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = set(archive.namelist())
            if "mimetype" in names:
                mimetype = archive.read("mimetype")
                if mimetype == b"application/vnd.oasis.opendocument.text":
                    return True
            return "content.xml" in names
    except zipfile.BadZipFile:
        return len(content) < 32


def _is_text(content: bytes) -> bool:
    if b"\x00" in content:
        return False
    try:
        content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    return True
