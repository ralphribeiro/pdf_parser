"""
Functions for normalizing and cleaning extracted text
"""
import re
from typing import List


def normalize_text(text: str, remove_extra_whitespace: bool = True) -> str:
    """
    Normalize extracted text.

    Args:
        text: raw text
        remove_extra_whitespace: remove extra spaces

    Returns:
        normalized text
    """
    if not text:
        return ""

    # Remove control characters (except line breaks and tabs)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

    if remove_extra_whitespace:
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)

        # Remove spaces at the beginning and end of lines
        lines = [line.strip() for line in text.split('\n')]

        # Remove multiple empty lines
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')
                prev_empty = True

        text = '\n'.join(cleaned_lines)

    return text.strip()


def merge_hyphenated_words(text: str) -> str:
    """
    Merge hyphenated words at end of line.

    Example:
        "This is an exam-\nple" -> "This is an example"
    """
    # Pattern: word + hyphen + line break + word
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text


def remove_repeated_headers_footers(lines: List[str], min_repetitions: int = 3) -> List[str]:
    """
    Remove repeated headers and footers.

    Args:
        lines: document lines
        min_repetitions: minimum repetitions to consider header/footer

    Returns:
        filtered lines
    """
    if len(lines) < min_repetitions * 2:
        return lines

    # Detect repeated patterns at the beginning (headers)
    first_lines = lines[:5]
    header_candidates = []

    for line in first_lines:
        if line.strip():
            count = sum(1 for l in lines if l.strip() == line.strip())
            if count >= min_repetitions:
                header_candidates.append(line.strip())

    # Detect repeated patterns at the end (footers)
    last_lines = lines[-5:]
    footer_candidates = []

    for line in last_lines:
        if line.strip():
            count = sum(1 for l in lines if l.strip() == line.strip())
            if count >= min_repetitions:
                footer_candidates.append(line.strip())

    # Remove headers and footers
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped not in header_candidates and stripped not in footer_candidates:
            filtered.append(line)

    return filtered


def clean_ocr_artifacts(text: str) -> str:
    """
    Remove common OCR artifacts.
    """
    # Remove isolated strange characters
    text = re.sub(r'\s[•·∙■□▪▫]\s', ' ', text)

    # Fix spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Fix multiple dots
    text = re.sub(r'\.{3,}', '...', text)

    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences (useful for later chunking).
    """
    # Simple sentence-end detection pattern
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ú])', text)
    return [s.strip() for s in sentences if s.strip()]
