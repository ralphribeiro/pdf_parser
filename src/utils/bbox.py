"""
Bounding box manipulation and reading-order sorting functions.

The reading-order algorithm groups text blocks into horizontal bands (rows)
based on vertical overlap, then sorts left-to-right within each band and
top-to-bottom across bands.  This prevents common OCR layout issues such as
text from the beginning of a page appearing at the end of the extracted output.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic bbox helpers
# ---------------------------------------------------------------------------


def normalize_bbox(
    bbox: list[float], page_width: float, page_height: float
) -> list[float]:
    """
    Normalize bounding box from absolute coordinates to relative (0-1).

    Args:
        bbox: [x1, y1, x2, y2] in absolute coordinates
        page_width: page width
        page_height: page height

    Returns:
        [x1, y1, x2, y2] normalized (0-1)
    """
    return [
        bbox[0] / page_width,
        bbox[1] / page_height,
        bbox[2] / page_width,
        bbox[3] / page_height,
    ]


def denormalize_bbox(
    bbox: list[float], page_width: float, page_height: float
) -> list[float]:
    """
    Convert normalized bbox (0-1) to absolute coordinates.
    """
    return [
        bbox[0] * page_width,
        bbox[1] * page_height,
        bbox[2] * page_width,
        bbox[3] * page_height,
    ]


def bbox_area(bbox: list[float]) -> float:
    """Calculate area of a bounding box."""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_overlap(bbox1: list[float], bbox2: list[float]) -> float:
    """
    Calculate the overlap area between two bounding boxes.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


# ---------------------------------------------------------------------------
# Reading-order sorting  (band-based algorithm)
# ---------------------------------------------------------------------------

_DEFAULT_Y_TOLERANCE: float = 0.008
_OVERLAP_BAND_THRESHOLD: float = 0.3
_COLUMN_GAP_THRESHOLD: float = 0.08


def sort_blocks_by_position(
    blocks: list[Any],
    reading_order: str = "top-to-bottom",
    y_tolerance: float | None = None,
) -> list[Any]:
    """
    Sort blocks by reading position using a band-based algorithm.

    Blocks are grouped into horizontal **bands** (rows) based on vertical
    overlap or proximity.  Within each band blocks are sorted left-to-right;
    bands themselves are ordered top-to-bottom.

    When multiple columns are detected (significant horizontal gap that
    splits most bands), the algorithm reads each column top-to-bottom
    before moving to the next one.

    Args:
        blocks: list of blocks with ``bbox`` attribute ``[x1, y1, x2, y2]``
            (normalised 0-1).
        reading_order: ``'top-to-bottom'`` (default) or ``'left-to-right'``.
        y_tolerance: vertical tolerance for grouping blocks into the same
            band (normalised 0-1).  ``None`` → use config value or 0.008.

    Returns:
        Blocks sorted in reading order.
    """
    if not blocks or len(blocks) <= 1:
        return blocks

    if reading_order == "left-to-right":
        return sorted(blocks, key=lambda b: (b.bbox[0], b.bbox[1]))

    if reading_order != "top-to-bottom":
        return blocks

    if y_tolerance is None:
        try:
            import config as _cfg

            y_tolerance = getattr(
                _cfg, "READING_ORDER_Y_TOLERANCE", _DEFAULT_Y_TOLERANCE
            )
        except ImportError:
            y_tolerance = _DEFAULT_Y_TOLERANCE

    # Detect columns and sort accordingly
    columns = _detect_columns(blocks)
    if columns is not None:
        logger.debug("Multi-column layout detected (%d columns)", len(columns))
        return _sort_multicolumn(blocks, columns, y_tolerance)

    return _sort_by_reading_bands(blocks, y_tolerance)


# ---------------------------------------------------------------------------
# Band-based sorting (single column)
# ---------------------------------------------------------------------------


def _sort_by_reading_bands(
    blocks: list[Any], y_tolerance: float = _DEFAULT_Y_TOLERANCE
) -> list[Any]:
    """
    Sort blocks into reading order using horizontal band grouping.

    Algorithm
    ---------
    1. Sort blocks by their top-Y coordinate.
    2. Greedily group consecutive blocks into the same **band** when their
       vertical extents overlap by ≥30 % of the smaller block height, *or*
       the gap between the band bottom and the block top is less than
       *y_tolerance*.
    3. Sort blocks inside each band by their left-X coordinate.
    4. Concatenate bands top-to-bottom.
    """
    sorted_blocks = sorted(blocks, key=lambda b: b.bbox[1])

    bands: list[list[Any]] = []
    current_band: list[Any] = [sorted_blocks[0]]
    band_y_top: float = sorted_blocks[0].bbox[1]
    band_y_bottom: float = sorted_blocks[0].bbox[3]

    for block in sorted_blocks[1:]:
        b_y_top = block.bbox[1]
        b_y_bottom = block.bbox[3]
        b_height = max(b_y_bottom - b_y_top, 0.001)
        band_height = max(band_y_bottom - band_y_top, 0.001)

        # Vertical overlap between block and current band
        overlap = max(0.0, min(band_y_bottom, b_y_bottom) - max(band_y_top, b_y_top))

        # Use the *smaller* of the two heights as reference
        min_height = min(b_height, band_height)
        overlap_ratio = overlap / min_height if min_height > 0 else 0.0

        # Gap between band bottom and block top (negative = overlap)
        y_gap = b_y_top - band_y_bottom

        if overlap_ratio >= _OVERLAP_BAND_THRESHOLD or (0 <= y_gap < y_tolerance):
            # Same band
            current_band.append(block)
            band_y_bottom = max(band_y_bottom, b_y_bottom)
        else:
            # New band
            bands.append(current_band)
            current_band = [block]
            band_y_top = b_y_top
            band_y_bottom = b_y_bottom

    if current_band:
        bands.append(current_band)

    # Sort each band left→right, then concatenate
    result: list[Any] = []
    for band in bands:
        band.sort(key=lambda b: b.bbox[0])
        result.extend(band)

    return result


# ---------------------------------------------------------------------------
# Multi-column detection and sorting
# ---------------------------------------------------------------------------


def _detect_columns(
    blocks: list[Any],
    min_blocks: int = 4,
    gap_threshold: float = _COLUMN_GAP_THRESHOLD,
) -> list[tuple[float, float]] | None:
    """
    Detect whether blocks form a multi-column layout.

    Returns a list of ``(x_start, x_end)`` tuples for each column, or
    ``None`` if the layout appears to be single-column.

    Heuristic
    ---------
    1. Collect the horizontal centre of every block.
    2. Sort centres and look for gaps wider than *gap_threshold*.
    3. If a gap splits *at least 40 %* of the blocks into two groups,
       declare a two-column layout.
    """
    if len(blocks) < min_blocks:
        return None

    x_centres = sorted((b.bbox[0] + b.bbox[2]) / 2 for b in blocks)

    # Find largest gap
    best_gap = 0.0
    best_idx = -1
    for i in range(1, len(x_centres)):
        gap = x_centres[i] - x_centres[i - 1]
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    if best_gap < gap_threshold:
        return None

    # Check that both sides have a reasonable number of blocks
    left_count = best_idx
    right_count = len(x_centres) - best_idx
    min_side = min(left_count, right_count)
    if min_side / len(x_centres) < 0.2:
        # One side has too few blocks — likely not a real column
        return None

    # Determine column boundaries
    col_boundary = (x_centres[best_idx - 1] + x_centres[best_idx]) / 2
    left_col = (0.0, col_boundary)
    right_col = (col_boundary, 1.0)

    return [left_col, right_col]


def _sort_multicolumn(
    blocks: list[Any],
    columns: list[tuple[float, float]],
    y_tolerance: float,
) -> list[Any]:
    """
    Sort blocks column-by-column, reading each column top-to-bottom before
    moving to the next.

    Full-width blocks (spanning multiple columns) are inserted at their
    vertical position relative to the column content around them.
    """
    col_mid = columns[0][1]  # boundary between columns

    left_blocks: list[Any] = []
    right_blocks: list[Any] = []
    full_width: list[Any] = []

    for block in blocks:
        block_x_centre = (block.bbox[0] + block.bbox[2]) / 2
        block_width = block.bbox[2] - block.bbox[0]

        # Full-width block: spans ≥70 % of page width
        if block_width >= 0.60:
            full_width.append(block)
        elif block_x_centre < col_mid:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    # Sort each column with band algorithm
    sorted_left = (
        _sort_by_reading_bands(left_blocks, y_tolerance) if left_blocks else []
    )
    sorted_right = (
        _sort_by_reading_bands(right_blocks, y_tolerance) if right_blocks else []
    )

    # Interleave full-width blocks at the correct vertical position
    if not full_width:
        return sorted_left + sorted_right

    # Full-width blocks sorted by Y
    full_width.sort(key=lambda b: b.bbox[1])

    result: list[Any] = []
    fw_idx = 0
    col_blocks = sorted_left + sorted_right

    for block in col_blocks:
        # Insert any full-width blocks that come before this block vertically
        while fw_idx < len(full_width) and full_width[fw_idx].bbox[1] <= block.bbox[1]:
            result.append(full_width[fw_idx])
            fw_idx += 1
        result.append(block)

    # Append remaining full-width blocks
    while fw_idx < len(full_width):
        result.append(full_width[fw_idx])
        fw_idx += 1

    return result


# ---------------------------------------------------------------------------
# Box merging helper
# ---------------------------------------------------------------------------


def merge_nearby_boxes(
    boxes: list[list[float]], threshold: float = 0.01
) -> list[list[float]]:
    """
    Merge nearby bounding boxes (useful for joining words into lines).

    Args:
        boxes: list of bboxes
        threshold: maximum distance to consider "nearby" (normalized)

    Returns:
        list of merged bboxes
    """
    if not boxes:
        return []

    # Sort boxes by position
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    merged = [sorted_boxes[0]]

    for current in sorted_boxes[1:]:
        last = merged[-1]

        # Check if they are nearby (same line, approximately)
        y_distance = abs(current[1] - last[1])
        x_distance = current[0] - last[2]  # horizontal distance

        if y_distance < threshold and 0 <= x_distance < threshold:
            # Merge: expand the last box
            merged[-1] = [
                min(last[0], current[0]),
                min(last[1], current[1]),
                max(last[2], current[2]),
                max(last[3], current[3]),
            ]
        else:
            merged.append(current)

    return merged
