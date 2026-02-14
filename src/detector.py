"""
Page type detector (digital vs scan)

Detects whether a page is:
- digital: native selectable text (digitally generated PDF)
- scan: scanned image that needs OCR
- hybrid: image with text overlay (e.g., court stamp over scanned document)
"""

from typing import Literal

import pdfplumber

import config


def detect_page_type(
    pdf_path: str, page_number: int
) -> Literal["digital", "scan", "hybrid"]:
    """
    Detect page type using image area analysis and text coverage.

    Logic:
    - If has large image (>30% of page) AND little text (<5% coverage) -> scan/hybrid
    - If has text covering significant area (>5%) -> digital
    - Otherwise -> scan

    Args:
        pdf_path: path to the PDF file
        page_number: page number (1-indexed)

    Returns:
        "digital": native extractable text
        "scan": image that needs OCR
        "hybrid": image with text overlay (will be treated as scan)
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]

        # Check for large images on the page
        has_large_img, image_coverage = _has_large_images(page)

        # Calculate selectable text coverage
        text_coverage = _get_text_coverage(page)

        # Decision logic
        if has_large_img:
            # Page has large image
            if text_coverage < config.TEXT_COVERAGE_THRESHOLD:
                # Little text -> probably scan with possible stamp
                return "hybrid" if text_coverage > 0 else "scan"
            else:
                # Image + significant text -> could be PDF with embedded images
                # But if image covers >80% of the page, it's probably a scan
                if image_coverage > 0.8:
                    return "hybrid"
                return "digital"
        else:
            # No large image
            if text_coverage > 0.01:  # Has some text
                return "digital"
            else:
                # No image and no text -> probably poorly detected scan
                return "scan"


def _has_large_images(page) -> tuple[bool, float]:
    """
    Check if the page has images covering significant area.

    Args:
        page: pdfplumber page

    Returns:
        (has_large_image, coverage_percentage)
    """
    images = page.images
    if not images:
        return False, 0.0

    page_area = page.width * page.height
    if page_area == 0:
        return False, 0.0

    # Calculate total image area
    total_image_area = 0
    for img in images:
        # Image coordinates
        x0 = img.get("x0", 0)
        x1 = img.get("x1", 0)
        top = img.get("top", 0)
        bottom = img.get("bottom", 0)

        width = abs(x1 - x0)
        height = abs(bottom - top)
        total_image_area += width * height

    coverage = total_image_area / page_area
    has_large = coverage > config.IMAGE_AREA_THRESHOLD

    return has_large, coverage


def _get_text_coverage(page) -> float:
    """
    Calculate the percentage of the page covered by selectable text.

    Args:
        page: pdfplumber page

    Returns:
        coverage (0.0 to 1.0)
    """
    words = page.extract_words()
    if not words:
        return 0.0

    page_area = page.width * page.height
    if page_area == 0:
        return 0.0

    # Calculate total area covered by words
    total_text_area = 0
    for w in words:
        x0 = w.get("x0", 0)
        x1 = w.get("x1", 0)
        top = w.get("top", 0)
        bottom = w.get("bottom", 0)

        width = abs(x1 - x0)
        height = abs(bottom - top)
        total_text_area += width * height

    return total_text_area / page_area


def has_extractable_text(pdf_path: str, page_number: int, min_chars: int = 10) -> bool:
    """
    Check if a page has extractable text.

    Args:
        pdf_path: path to the PDF
        page_number: page number (1-indexed)
        min_chars: minimum characters to consider "has text"

    Returns:
        True if has extractable text
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            text = page.extract_text()
            return text is not None and len(text.strip()) >= min_chars
    except Exception:
        return False


def get_page_dimensions(pdf_path: str, page_number: int) -> tuple[float, float]:
    """
    Return page dimensions (width, height) in points.

    Args:
        pdf_path: path to the PDF
        page_number: page number (1-indexed)

    Returns:
        (width, height) in points
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        return (page.width, page.height)


def get_page_info(pdf_path: str, page_number: int) -> dict:
    """
    Return detailed information about a page (useful for debugging).

    Args:
        pdf_path: path to the PDF
        page_number: page number (1-indexed)

    Returns:
        dict with page information
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]

        has_large_img, image_coverage = _has_large_images(page)
        text_coverage = _get_text_coverage(page)

        text = page.extract_text()
        char_count = len(text.strip()) if text else 0

        return {
            "page_number": page_number,
            "width": page.width,
            "height": page.height,
            "num_images": len(page.images) if page.images else 0,
            "image_coverage": round(image_coverage * 100, 2),
            "has_large_images": has_large_img,
            "text_coverage": round(text_coverage * 100, 2),
            "char_count": char_count,
            "detected_type": detect_page_type(pdf_path, page_number),
        }
