"""
Searchable PDF exporter.

Overlays invisible text (render mode 3) on top of original PDF pages,
allowing text search and selection in scanned documents.
"""
import io
from pathlib import Path
from typing import List, Union

from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth

from src.models.schemas import Page, Document
from src.utils.bbox import denormalize_bbox


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FONT_NAME = "Helvetica"
_FONT_SIZE_FROM_HEIGHT_FACTOR = 0.80  # font_size = line_height * factor
_MIN_FONT_SIZE = 4.0
_MAX_FONT_SIZE = 72.0


# ---------------------------------------------------------------------------
# Text overlay (per-page)
# ---------------------------------------------------------------------------

def _create_text_overlay(page: Page, page_width_pts: float,
                         page_height_pts: float) -> bytes:
    """
    Generate a single-page PDF containing invisible text positioned
    according to individual line bounding boxes (when available)
    or uniformly distributed within the block bbox (fallback).

    Args:
        page: Page object with blocks and normalized bboxes (0-1).
        page_width_pts: page width in PDF points.
        page_height_pts: page height in PDF points.

    Returns:
        Overlay PDF bytes (1 page).
    """
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=(page_width_pts, page_height_pts))

    for block in page.blocks:
        if not block.text:
            continue

        if block.lines:
            _render_block_with_lines(c, block.lines,
                                     page_width_pts, page_height_pts)
        else:
            _render_block_fallback(c, block,
                                   page_width_pts, page_height_pts)

    c.showPage()
    c.save()

    return buf.getvalue()


def _render_line(canvas: Canvas, text: str, line_bbox: List[float],
                 page_width_pts: float, page_height_pts: float) -> None:
    """
    Render ONE line of invisible text on the canvas, positioned and
    scaled according to the line bbox.

    Args:
        canvas: reportlab Canvas.
        text: line text.
        line_bbox: [x1, y1, x2, y2] normalized (0-1).
        page_width_pts: page width in points.
        page_height_pts: page height in points.
    """
    if not text or not text.strip():
        return

    # Denormalize bbox
    abs_bbox = denormalize_bbox(line_bbox, page_width_pts, page_height_pts)
    lx1, ly1, lx2, ly2 = abs_bbox
    line_width = lx2 - lx1
    line_height = ly2 - ly1

    if line_width <= 0 or line_height <= 0:
        return

    # Font size derived from line HEIGHT (not width)
    font_size = max(_MIN_FONT_SIZE,
                    min(line_height * _FONT_SIZE_FROM_HEIGHT_FACTOR, _MAX_FONT_SIZE))

    # Horizontal scale: adjust text width to fit within bbox
    natural_width = stringWidth(text, _FONT_NAME, font_size)
    if natural_width > 0:
        horiz_scale = (line_width / natural_width) * 100.0
    else:
        horiz_scale = 100.0

    # PDF coordinates (bottom-up): text starts at bottom-left corner
    pdf_x = lx1
    pdf_y = page_height_pts - ly2 + (line_height - font_size) * 0.5

    text_obj = canvas.beginText(pdf_x, pdf_y)
    text_obj.setFont(_FONT_NAME, font_size)
    text_obj.setHorizScale(horiz_scale)
    text_obj.setTextRenderMode(3)  # Invisible
    text_obj.textLine(text)
    canvas.drawText(text_obj)


def _render_block_with_lines(canvas: Canvas, lines: List[dict],
                              page_width_pts: float,
                              page_height_pts: float) -> None:
    """
    Render a block using per-line data (text + bbox).
    """
    for line_data in lines:
        text = line_data.get("text", "")
        bbox = line_data.get("bbox", [])
        if text and len(bbox) == 4:
            _render_line(canvas, text, bbox, page_width_pts, page_height_pts)


def _render_block_fallback(canvas: Canvas, block,
                            page_width_pts: float,
                            page_height_pts: float) -> None:
    """
    Fallback for blocks without `lines` data: distributes text lines
    uniformly within the block bbox.
    """
    text_lines = [l for l in block.text.split("\n") if l.strip()]
    if not text_lines:
        return

    abs_bbox = denormalize_bbox(block.bbox, page_width_pts, page_height_pts)
    bx1, by1, bx2, by2 = abs_bbox
    block_width = bx2 - bx1
    block_height = by2 - by1

    if block_width <= 0 or block_height <= 0:
        return

    n_lines = len(text_lines)
    line_height = block_height / n_lines

    for i, text in enumerate(text_lines):
        # Distribute uniformly: each line occupies 1/n of the block height
        line_y1 = by1 + i * line_height
        line_y2 = line_y1 + line_height

        # Create normalized bbox for the distributed line
        line_bbox = [
            bx1 / page_width_pts,
            line_y1 / page_height_pts,
            bx2 / page_width_pts,
            line_y2 / page_height_pts,
        ]

        _render_line(canvas, text, line_bbox, page_width_pts, page_height_pts)


# ---------------------------------------------------------------------------
# Font size calculation (legacy, used by old tests)
# ---------------------------------------------------------------------------

def _calculate_font_size(text_line: str, bbox_width_pts: float,
                         font_name: str = "Helvetica",
                         min_size: float = 4.0,
                         max_size: float = 16.0) -> float:
    """
    Calculate font size so that text fits within the bbox width.

    Note: function kept for backward compat. Per-line rendering
    now uses font_size derived from line height (see _render_line).

    Args:
        text_line: text line (the longest or first).
        bbox_width_pts: available width in points.
        font_name: font name.
        min_size: minimum size.
        max_size: maximum size.

    Returns:
        Font size in points.
    """
    if not text_line or bbox_width_pts <= 0:
        return min_size

    # Start at maximum size and decrease until it fits
    for size in [max_size, 14, 12, 10, 8, 6, min_size]:
        width = stringWidth(text_line, font_name, size)
        if width <= bbox_width_pts:
            return size

    return min_size


def create_searchable_pdf(original_pdf_path: Union[str, Path],
                          document: Document,
                          output_path: Union[str, Path]) -> None:
    """
    Create a searchable PDF by combining the original PDF with invisible
    text overlays on OCR pages.

    - Digital pages (source="digital"): copied without changes.
    - OCR pages (source="ocr"): receive invisible text overlay.

    Args:
        original_pdf_path: path to the original PDF.
        document: Document with processed pages and blocks.
        output_path: output PDF path.
    """
    import pypdf

    original_pdf_path = Path(original_pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = pypdf.PdfReader(str(original_pdf_path))
    writer = pypdf.PdfWriter()

    for page_data in document.pages:
        page_idx = page_data.page - 1  # 0-indexed

        if page_idx >= len(reader.pages):
            # More pages in Document than in PDF — skip
            continue

        original_page = reader.pages[page_idx]

        if page_data.source == "ocr" and page_data.blocks:
            # Determine original PDF page dimensions (in points)
            media_box = original_page.mediabox
            page_width_pts = float(media_box.width)
            page_height_pts = float(media_box.height)

            # Generate overlay with invisible text
            overlay_bytes = _create_text_overlay(
                page_data, page_width_pts, page_height_pts
            )

            # Merge overlay on top of the original page
            overlay_reader = pypdf.PdfReader(io.BytesIO(overlay_bytes))
            original_page.merge_page(overlay_reader.pages[0])

        writer.add_page(original_page)

    with open(output_path, "wb") as f:
        writer.write(f)
