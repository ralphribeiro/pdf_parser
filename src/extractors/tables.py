"""
Table extractor for PDFs (digital and scanned)
"""

import logging

import camelot

import config
from src.models.schemas import Block, BlockType
from src.utils.bbox import normalize_bbox

logger = logging.getLogger(__name__)


def extract_tables_digital(
    pdf_path: str, page_number: int, flavor: str | None = None
) -> list[Block]:
    """
    Extract tables from a digital page using camelot.

    Args:
        pdf_path: path to the PDF
        page_number: page number (1-indexed)
        flavor: 'lattice' (tables with borders) or 'stream' (without borders)

    Returns:
        list of table blocks
    """
    if flavor is None:
        flavor = config.CAMELOT_FLAVOR

    blocks = []

    try:
        # Extract tables from the page
        tables = camelot.read_pdf(
            pdf_path, pages=str(page_number), flavor=flavor, suppress_stdout=True
        )

        if not tables:
            return blocks

        # Process each detected table
        for idx, table in enumerate(tables):
            # Check detection confidence
            confidence = table.parsing_report.get("accuracy", 0.0) / 100.0

            if confidence < config.TABLE_DETECTION_CONFIDENCE:
                continue

            # Convert table to list of lists
            rows = table.df.values.tolist()

            # Remove completely empty rows
            rows = [row for row in rows if any(str(cell).strip() for cell in row)]

            if not rows:
                continue

            # Get table bbox (camelot returns absolute coordinates)
            # Camelot uses coordinates with origin at bottom-left corner
            x1, y1, x2, y2 = table._bbox

            # Convert to normalized coordinates
            # Need to get page dimensions
            from src.detector import get_page_dimensions

            page_width, page_height = get_page_dimensions(pdf_path, page_number)

            # Camelot uses coordinates with inverted Y (origin at bottom)
            # Need to convert to Y with origin at top
            bbox = normalize_bbox(
                [x1, page_height - y2, x2, page_height - y1], page_width, page_height
            )

            block = Block(
                block_id=f"p{page_number}_t{idx + 1}",
                type=BlockType.TABLE,
                text=None,  # Tables don't have a text field
                bbox=bbox,
                confidence=confidence,
                rows=rows,
            )

            blocks.append(block)

    except Exception as e:
        logger.warning("Error extracting tables from page %d: %s", page_number, e)

    return blocks


def extract_tables_from_blocks(
    blocks: list[Block], min_rows: int = 2, min_cols: int = 2
) -> list[Block]:
    """
    Detect tables in OCR text blocks using heuristics.

    This function attempts to identify tabular patterns in text blocks
    based on alignment and regular structure.

    Args:
        blocks: text blocks extracted by OCR
        min_rows: minimum number of rows to consider a table
        min_cols: minimum number of columns to consider a table

    Returns:
        list of detected table blocks
    """
    # TODO: Implement table detection in OCR
    # This is an advanced feature that can be added later
    # For now, returns empty list
    return []


def merge_table_cells(rows: list[list[str]]) -> list[list[str]]:
    """
    Merge table cells that were incorrectly split.

    Args:
        rows: table rows

    Returns:
        rows with merged cells
    """
    if not rows:
        return rows

    cleaned_rows = []

    for row in rows:
        # Remove empty cells at beginning and end
        cleaned_row = []
        for cell in row:
            cell_str = str(cell).strip()
            cleaned_row.append(cell_str)

        cleaned_rows.append(cleaned_row)

    return cleaned_rows


def validate_table_structure(rows: list[list[str]]) -> bool:
    """
    Validate if the table structure is consistent.

    Args:
        rows: table rows

    Returns:
        True if the structure is valid
    """
    if not rows:
        return False

    # Check if all rows have the same number of columns
    num_cols = len(rows[0])

    if num_cols == 0:
        return False

    for row in rows:
        # Allow small variations (+/-1 column)
        if len(row) != num_cols and abs(len(row) - num_cols) > 1:
            return False

    return True


def normalize_table_data(rows: list[list[str]]) -> list[list[str]]:
    """
    Normalize table data (text cleanup, formatting).

    Args:
        rows: raw table rows

    Returns:
        normalized rows
    """
    normalized = []

    for row in rows:
        normalized_row = []
        for cell in row:
            # Convert to string and clean
            cell_str = str(cell).strip()

            # Remove line breaks within cells
            cell_str = cell_str.replace("\n", " ")

            # Remove multiple spaces
            cell_str = " ".join(cell_str.split())

            normalized_row.append(cell_str)

        normalized.append(normalized_row)

    return normalized
