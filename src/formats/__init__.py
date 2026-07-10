"""Multi-format document ingestion helpers."""

from src.formats.dispatcher import process_file
from src.formats.registry import (
    ACCEPT_ATTRIBUTE,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMAT_LABEL,
    DocumentFormat,
    UnsupportedFormatError,
    detect_format,
    detect_format_from_upload,
    extension_for_filename,
)

__all__ = [
    "ACCEPT_ATTRIBUTE",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_FORMAT_LABEL",
    "DocumentFormat",
    "UnsupportedFormatError",
    "detect_format",
    "detect_format_from_upload",
    "extension_for_filename",
    "process_file",
]
