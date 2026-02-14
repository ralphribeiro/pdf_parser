"""
OCR text post-processing

Functions to clean and improve OCR-extracted text:
- Noise and invalid character removal
- Spacing correction
- Short line removal
- Broken word merging
"""
import re
from typing import List, Set, Optional


def clean_ocr_text(text: str) -> str:
    """
    Clean OCR-extracted text by removing common noise.

    Args:
        text: raw OCR text

    Returns:
        cleaned text
    """
    if not text:
        return ""

    # Remove common OCR noise characters
    # Keep letters, numbers, basic punctuation and accents
    noise_chars = r'[|\\{}\[\]<>©®™°§¶†‡•◦▪▫●○◆◇★☆♦♠♣♥]'
    text = re.sub(noise_chars, '', text)

    # Remove repeated character sequences (e.g., "====", "----", "____")
    text = re.sub(r'([=\-_*#~])\1{3,}', '', text)

    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)

    # Add space after punctuation if missing
    text = re.sub(r'([.,;:!?])([A-ZÀ-Úa-zà-ú])', r'\1 \2', text)

    # Remove multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove lines with only numbers/symbols (usually noise)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Keep line if it has at least 2 alphabetic characters
        if len(re.findall(r'[A-Za-zÀ-ú]', line)) >= 2:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def remove_short_lines(text: str, min_length: int = 3) -> str:
    """
    Remove very short lines that are usually noise.

    Args:
        text: text
        min_length: minimum length to keep a line

    Returns:
        text without short lines
    """
    lines = text.split('\n')
    filtered = [line for line in lines if len(line.strip()) >= min_length]
    return '\n'.join(filtered)


def fix_common_ocr_errors(text: str) -> str:
    """
    Fix common OCR errors in Portuguese text.

    Args:
        text: text with possible errors

    Returns:
        corrected text
    """
    corrections = {
        # Confused letters
        r'\bRN\b': 'RN',  # Keep RN (document)
        r'l<': 'k',
        r'\bl\b(?=[A-Z])': 'I',  # lone l before uppercase -> I
        r'(?<=[a-z])O(?=[a-z])': 'o',  # O between lowercase -> o
        r'(?<=[A-Z])o(?=[A-Z])': 'O',  # o between uppercase -> O

        # Numbers confused with letters
        r'(?<=[A-Za-z])0(?=[A-Za-z])': 'O',  # 0 between letters -> O
        r'(?<=[0-9])O(?=[0-9])': '0',  # O between numbers -> 0
        r'(?<=[A-Za-z])1(?=[A-Za-z])': 'l',  # 1 between letters -> l
        r'(?<=[0-9])l(?=[0-9])': '1',  # l between numbers -> 1

        # Commonly misrecognized words (Portuguese legal)
        r'\bDl<\b': 'DK',
        r'\bNQ\b': 'Nº',
        r'\bn2\b': 'nº',
        r'\bNR\b': 'NR',  # Keep
    }

    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)

    return text


def merge_broken_words(text: str, min_word_length: int = 4) -> str:
    """
    Attempt to merge words that were incorrectly split.

    Example: "TA BELIÃO" -> may remain if we don't have a dictionary

    Args:
        text: text
        min_word_length: minimum fragment length to attempt merging

    Returns:
        text with merged words
    """
    # Pattern: short word + space + short word at the beginning of a larger word
    # E.g., "COM ARCA" where "COMARCA" would be expected

    # This is a simple heuristic — ideally we would use a dictionary
    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        words = line.split()
        if len(words) < 2:
            fixed_lines.append(line)
            continue

        # Attempt to merge very short fragments
        merged = []
        i = 0
        while i < len(words):
            word = words[i]

            # If word is very short and the next one is too
            if (len(word) <= 2 and
                i + 1 < len(words) and
                len(words[i + 1]) >= 2 and
                word.isupper() == words[i + 1].isupper()):
                # Merge the words
                merged.append(word + words[i + 1])
                i += 2
            else:
                merged.append(word)
                i += 1

        fixed_lines.append(' '.join(merged))

    return '\n'.join(fixed_lines)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace.

    Args:
        text: text

    Returns:
        text with normalized whitespace
    """
    # Remove spaces at the beginning/end of lines
    lines = [line.strip() for line in text.split('\n')]

    # Remove consecutive empty lines
    cleaned = []
    prev_empty = False
    for line in lines:
        if line:
            cleaned.append(line)
            prev_empty = False
        elif not prev_empty:
            cleaned.append('')
            prev_empty = True

    return '\n'.join(cleaned).strip()


def postprocess_ocr_text(text: str,
                        clean: bool = True,
                        fix_errors: bool = True,
                        merge_words: bool = False,
                        min_line_length: int = 3) -> str:
    """
    Complete OCR text post-processing pipeline.

    Args:
        text: raw OCR text
        clean: apply noise cleaning
        fix_errors: fix common errors
        merge_words: attempt to merge broken words
        min_line_length: minimum line length

    Returns:
        processed text
    """
    if not text:
        return ""

    if clean:
        text = clean_ocr_text(text)

    if fix_errors:
        text = fix_common_ocr_errors(text)

    if merge_words:
        text = merge_broken_words(text)

    if min_line_length > 0:
        text = remove_short_lines(text, min_line_length)

    text = normalize_whitespace(text)

    return text
