"""
Extractor for digital PDFs (with selectable text)
"""
import pdfplumber
from typing import List, Tuple
from src.models.schemas import Block, BlockType
from src.utils.bbox import normalize_bbox, sort_blocks_by_position
from src.utils.text_normalizer import normalize_text


def extract_digital_page(pdf_path: str, page_number: int) -> Tuple[List[Block], float, float]:
    """
    Extract content from a digital page (with selectable text).

    Args:
        pdf_path: path to the PDF
        page_number: page number (1-indexed)

    Returns:
        (blocks, width, height)
    """
    blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        page_width = page.width
        page_height = page.height

        # Extract full page text
        full_text = page.extract_text()

        if not full_text:
            return blocks, page_width, page_height

        # Extract words with position
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False
        )

        if not words:
            # If couldn't extract words, create a single block with all text
            block = Block(
                block_id=f"p{page_number}_b1",
                type=BlockType.PARAGRAPH,
                text=normalize_text(full_text),
                bbox=[0.0, 0.0, 1.0, 1.0],
                confidence=1.0
            )
            blocks.append(block)
            return blocks, page_width, page_height

        # Group words into lines (based on Y)
        lines = _group_words_into_lines(words, page_width, page_height)

        # Group lines into paragraphs
        paragraphs = _group_lines_into_paragraphs(lines)

        # Create blocks for each paragraph
        for idx, paragraph in enumerate(paragraphs):
            text = " ".join([line['text'] for line in paragraph])

            # Calculate paragraph bbox (union of all lines)
            all_bboxes = [line['bbox'] for line in paragraph]
            merged_bbox = _merge_bboxes(all_bboxes)

            block = Block(
                block_id=f"p{page_number}_b{idx + 1}",
                type=BlockType.PARAGRAPH,
                text=normalize_text(text),
                bbox=merged_bbox,
                confidence=1.0
            )
            blocks.append(block)

        # Sort blocks by reading position
        blocks = sort_blocks_by_position(blocks)

    return blocks, page_width, page_height


def _group_words_into_lines(words: List[dict], page_width: float, page_height: float,
                            y_tolerance: float = 3) -> List[dict]:
    """
    Group words into lines based on Y coordinate.
    """
    if not words:
        return []

    # Sort words by Y, then X
    sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))

    lines = []
    current_line = [sorted_words[0]]
    current_y = sorted_words[0]['top']

    for word in sorted_words[1:]:
        # If Y is close to current line, add to line
        if abs(word['top'] - current_y) <= y_tolerance:
            current_line.append(word)
        else:
            # Create new line
            lines.append(_words_to_line(current_line, page_width, page_height))
            current_line = [word]
            current_y = word['top']

    # Add last line
    if current_line:
        lines.append(_words_to_line(current_line, page_width, page_height))

    return lines


def _words_to_line(words: List[dict], page_width: float, page_height: float) -> dict:
    """
    Convert a list of words into a line.
    """
    text = " ".join([w['text'] for w in words])

    # Line bbox (union of all words)
    x0 = min(w['x0'] for w in words)
    top = min(w['top'] for w in words)
    x1 = max(w['x1'] for w in words)
    bottom = max(w['bottom'] for w in words)

    bbox = normalize_bbox([x0, top, x1, bottom], page_width, page_height)

    return {
        'text': text,
        'bbox': bbox,
        'y': top  # keep absolute Y for grouping
    }


def _group_lines_into_paragraphs(lines: List[dict],
                                 gap_threshold: float = 0.03) -> List[List[dict]]:
    """
    Group lines into paragraphs based on vertical gaps.
    """
    if not lines:
        return []

    paragraphs = []
    current_paragraph = [lines[0]]

    for i in range(1, len(lines)):
        prev_line = lines[i - 1]
        curr_line = lines[i]

        # Calculate vertical gap between lines (normalized)
        gap = curr_line['bbox'][1] - prev_line['bbox'][3]

        # If gap is large, start new paragraph
        if gap > gap_threshold:
            paragraphs.append(current_paragraph)
            current_paragraph = [curr_line]
        else:
            current_paragraph.append(curr_line)

    # Add last paragraph
    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs


def _merge_bboxes(bboxes: List[List[float]]) -> List[float]:
    """
    Merge multiple bboxes into a single enclosing bbox.
    """
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]

    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)

    return [x1, y1, x2, y2]
