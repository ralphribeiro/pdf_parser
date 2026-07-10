"""Generate simple searchable PDFs from extracted document text."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from src.models.schemas import Document


def create_text_pdf(document: Document, output_path: str | Path) -> None:
    """Create a readable/searchable PDF from a parsed Document."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = Canvas(str(output_path), pagesize=A4)
    _width, height = A4
    margin_x = 54
    y = height - 54
    line_height = 14

    for page in document.pages:
        if page.page > 1:
            canvas.showPage()
            y = height - 54
        for block in page.blocks:
            text = block.text or ""
            for raw_line in text.splitlines() or [""]:
                for line in _wrap_line(raw_line, max_chars=95):
                    if y < 54:
                        canvas.showPage()
                        y = height - 54
                    canvas.drawString(margin_x, y, line)
                    y -= line_height
            if y < 54:
                canvas.showPage()
                y = height - 54
            else:
                y -= line_height
    canvas.save()


def _wrap_line(line: str, *, max_chars: int) -> list[str]:
    words = line.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
