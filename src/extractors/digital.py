"""
Extrator para PDFs digitais (com texto selecionável)
"""
import pdfplumber
from typing import List, Tuple
from src.models.schemas import Block, BlockType
from src.utils.bbox import normalize_bbox, sort_blocks_by_position
from src.utils.text_normalizer import normalize_text


def extract_digital_page(pdf_path: str, page_number: int) -> Tuple[List[Block], float, float]:
    """
    Extrai conteúdo de uma página digital (com texto selecionável)

    Args:
        pdf_path: caminho para o PDF
        page_number: número da página (1-indexed)

    Returns:
        (blocos, largura, altura)
    """
    blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        page_width = page.width
        page_height = page.height

        # Extrai texto completo da página
        full_text = page.extract_text()

        if not full_text:
            return blocks, page_width, page_height

        # Extrai palavras com posição
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False
        )

        if not words:
            # Se não conseguiu extrair palavras, cria um bloco com todo o texto
            block = Block(
                block_id=f"p{page_number}_b1",
                type=BlockType.PARAGRAPH,
                text=normalize_text(full_text),
                bbox=[0.0, 0.0, 1.0, 1.0],
                confidence=1.0
            )
            blocks.append(block)
            return blocks, page_width, page_height

        # Agrupa palavras em linhas (baseado em Y)
        lines = _group_words_into_lines(words, page_width, page_height)

        # Agrupa linhas em parágrafos
        paragraphs = _group_lines_into_paragraphs(lines)

        # Cria blocos para cada parágrafo
        for idx, paragraph in enumerate(paragraphs):
            text = " ".join([line['text'] for line in paragraph])

            # Calcula bbox do parágrafo (união de todas as linhas)
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

        # Ordena blocos por posição de leitura
        blocks = sort_blocks_by_position(blocks)

    return blocks, page_width, page_height


def _group_words_into_lines(words: List[dict], page_width: float, page_height: float,
                            y_tolerance: float = 3) -> List[dict]:
    """
    Agrupa palavras em linhas baseado em coordenada Y
    """
    if not words:
        return []

    # Ordena palavras por Y, depois X
    sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))

    lines = []
    current_line = [sorted_words[0]]
    current_y = sorted_words[0]['top']

    for word in sorted_words[1:]:
        # Se Y está próximo da linha atual, adiciona à linha
        if abs(word['top'] - current_y) <= y_tolerance:
            current_line.append(word)
        else:
            # Cria nova linha
            lines.append(_words_to_line(current_line, page_width, page_height))
            current_line = [word]
            current_y = word['top']

    # Adiciona última linha
    if current_line:
        lines.append(_words_to_line(current_line, page_width, page_height))

    return lines


def _words_to_line(words: List[dict], page_width: float, page_height: float) -> dict:
    """
    Converte lista de palavras em uma linha
    """
    text = " ".join([w['text'] for w in words])

    # Bbox da linha (união de todas as palavras)
    x0 = min(w['x0'] for w in words)
    top = min(w['top'] for w in words)
    x1 = max(w['x1'] for w in words)
    bottom = max(w['bottom'] for w in words)

    bbox = normalize_bbox([x0, top, x1, bottom], page_width, page_height)

    return {
        'text': text,
        'bbox': bbox,
        'y': top  # mantém Y absoluto para agrupamento
    }


def _group_lines_into_paragraphs(lines: List[dict],
                                 gap_threshold: float = 0.03) -> List[List[dict]]:
    """
    Agrupa linhas em parágrafos baseado em gaps verticais
    """
    if not lines:
        return []

    paragraphs = []
    current_paragraph = [lines[0]]

    for i in range(1, len(lines)):
        prev_line = lines[i - 1]
        curr_line = lines[i]

        # Calcula gap vertical entre linhas (normalizado)
        gap = curr_line['bbox'][1] - prev_line['bbox'][3]

        # Se gap é grande, inicia novo parágrafo
        if gap > gap_threshold:
            paragraphs.append(current_paragraph)
            current_paragraph = [curr_line]
        else:
            current_paragraph.append(curr_line)

    # Adiciona último parágrafo
    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs


def _merge_bboxes(bboxes: List[List[float]]) -> List[float]:
    """
    Mescla múltiplos bboxes em um único bbox envolvente
    """
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]

    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)

    return [x1, y1, x2, y2]
