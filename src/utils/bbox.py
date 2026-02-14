"""
Bounding box manipulation functions
"""

from typing import Any


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


def sort_blocks_by_position(
    blocks: list[Any], reading_order: str = "top-to-bottom"
) -> list[Any]:
    """
    Sort blocks by reading position.

    Args:
        blocks: list of blocks with 'bbox' attribute
        reading_order: 'top-to-bottom' or 'left-to-right'

    Returns:
        sorted blocks
    """
    if reading_order == "top-to-bottom":
        # Sort by Y (top), then by X (left)
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    elif reading_order == "left-to-right":
        # Sort by X (left), then by Y (top)
        return sorted(blocks, key=lambda b: (b.bbox[0], b.bbox[1]))
    else:
        return blocks


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
