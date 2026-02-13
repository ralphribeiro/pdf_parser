"""
Exportador de PDF pesquisável (searchable PDF).

Sobrepõe texto invisível (render mode 3) sobre as páginas originais do PDF,
permitindo busca e seleção de texto em documentos escaneados.
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
# Constantes
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
    Gera um PDF de uma única página contendo texto invisível posicionado
    de acordo com os bounding boxes das linhas individuais (quando disponíveis)
    ou distribuído uniformemente no bbox do bloco (fallback).

    Args:
        page: objeto Page com blocos e bboxes normalizados (0-1).
        page_width_pts: largura da página em pontos PDF.
        page_height_pts: altura da página em pontos PDF.

    Returns:
        Bytes do PDF de overlay (1 página).
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
    Renderiza UMA linha de texto invisível no canvas, posicionada e
    escalada de acordo com o bbox da linha.

    Args:
        canvas: reportlab Canvas.
        text: texto da linha.
        line_bbox: [x1, y1, x2, y2] normalizado (0-1).
        page_width_pts: largura da página em pontos.
        page_height_pts: altura da página em pontos.
    """
    if not text or not text.strip():
        return

    # Denormaliza bbox
    abs_bbox = denormalize_bbox(line_bbox, page_width_pts, page_height_pts)
    lx1, ly1, lx2, ly2 = abs_bbox
    line_width = lx2 - lx1
    line_height = ly2 - ly1

    if line_width <= 0 or line_height <= 0:
        return

    # Font size derivado da ALTURA da linha (não da largura)
    font_size = max(_MIN_FONT_SIZE,
                    min(line_height * _FONT_SIZE_FROM_HEIGHT_FACTOR, _MAX_FONT_SIZE))

    # Escala horizontal: ajusta largura do texto para caber no bbox
    natural_width = stringWidth(text, _FONT_NAME, font_size)
    if natural_width > 0:
        horiz_scale = (line_width / natural_width) * 100.0
    else:
        horiz_scale = 100.0

    # Coordenadas PDF (bottom-up): texto inicia no canto inferior-esquerdo
    pdf_x = lx1
    pdf_y = page_height_pts - ly2 + (line_height - font_size) * 0.5

    text_obj = canvas.beginText(pdf_x, pdf_y)
    text_obj.setFont(_FONT_NAME, font_size)
    text_obj.setHorizScale(horiz_scale)
    text_obj.setTextRenderMode(3)  # Invisível
    text_obj.textLine(text)
    canvas.drawText(text_obj)


def _render_block_with_lines(canvas: Canvas, lines: List[dict],
                              page_width_pts: float,
                              page_height_pts: float) -> None:
    """
    Renderiza um bloco usando dados per-line (text + bbox).
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
    Fallback para blocos sem dados de `lines`: distribui linhas do texto
    uniformemente dentro do bbox do bloco.
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
        # Distribui uniformemente: cada linha ocupa 1/n da altura do bloco
        line_y1 = by1 + i * line_height
        line_y2 = line_y1 + line_height

        # Cria bbox normalizado para a linha distribuída
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
    Calcula o tamanho da fonte para que o texto caiba na largura do bbox.

    Nota: função mantida para backward compat. A renderização per-line
    agora usa font_size derivado da altura da linha (ver _render_line).

    Args:
        text_line: linha de texto (a mais longa ou a primeira).
        bbox_width_pts: largura disponível em pontos.
        font_name: nome da fonte.
        min_size: tamanho mínimo.
        max_size: tamanho máximo.

    Returns:
        Tamanho da fonte em pontos.
    """
    if not text_line or bbox_width_pts <= 0:
        return min_size

    # Começa no tamanho máximo e diminui até caber
    for size in [max_size, 14, 12, 10, 8, 6, min_size]:
        width = stringWidth(text_line, font_name, size)
        if width <= bbox_width_pts:
            return size

    return min_size


def create_searchable_pdf(original_pdf_path: Union[str, Path],
                          document: Document,
                          output_path: Union[str, Path]) -> None:
    """
    Cria um PDF pesquisável combinando o PDF original com overlays de texto
    invisível nas páginas OCR.

    - Páginas digitais (source="digital"): copiadas sem alteração.
    - Páginas OCR (source="ocr"): recebem overlay de texto invisível.

    Args:
        original_pdf_path: caminho do PDF original.
        document: Document com páginas e blocos processados.
        output_path: caminho do PDF de saída.
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
            # Mais páginas no Document do que no PDF — pula
            continue

        original_page = reader.pages[page_idx]

        if page_data.source == "ocr" and page_data.blocks:
            # Determina dimensões da página do PDF original (em pontos)
            media_box = original_page.mediabox
            page_width_pts = float(media_box.width)
            page_height_pts = float(media_box.height)

            # Gera overlay com texto invisível
            overlay_bytes = _create_text_overlay(
                page_data, page_width_pts, page_height_pts
            )

            # Merge overlay sobre a página original
            overlay_reader = pypdf.PdfReader(io.BytesIO(overlay_bytes))
            original_page.merge_page(overlay_reader.pages[0])

        writer.add_page(original_page)

    with open(output_path, "wb") as f:
        writer.write(f)
