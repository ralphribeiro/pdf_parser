"""
Detector de tipo de página (digital vs scan)

Detecta se uma página é:
- digital: texto nativo selecionável (PDF gerado digitalmente)
- scan: imagem escaneada que precisa de OCR
- hybrid: imagem com overlay de texto (ex: carimbo do TJSP sobre documento escaneado)
"""
import pdfplumber
from typing import Literal, Tuple
import config


def detect_page_type(pdf_path: str, page_number: int) -> Literal["digital", "scan", "hybrid"]:
    """
    Detecta o tipo de página usando análise de área de imagem e cobertura de texto.

    Lógica:
    - Se tem imagem grande (>30% da página) E pouco texto (<5% de cobertura) -> scan/hybrid
    - Se tem texto cobrindo área significativa (>5%) -> digital
    - Caso contrário -> scan

    Args:
        pdf_path: caminho para o arquivo PDF
        page_number: número da página (1-indexed)

    Returns:
        "digital": texto nativo extraível
        "scan": imagem que precisa de OCR
        "hybrid": imagem com overlay de texto (será tratado como scan)
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]

        # Verifica se há imagens grandes na página
        has_large_img, image_coverage = _has_large_images(page)

        # Calcula cobertura de texto selecionável
        text_coverage = _get_text_coverage(page)

        # Lógica de decisão
        if has_large_img:
            # Página tem imagem grande
            if text_coverage < config.TEXT_COVERAGE_THRESHOLD:
                # Pouco texto -> provavelmente scan com possível carimbo
                return "hybrid" if text_coverage > 0 else "scan"
            else:
                # Imagem + texto significativo -> pode ser PDF com imagens embutidas
                # Mas se a imagem cobre >80% da página, provavelmente é scan
                if image_coverage > 0.8:
                    return "hybrid"
                return "digital"
        else:
            # Sem imagem grande
            if text_coverage > 0.01:  # Tem algum texto
                return "digital"
            else:
                # Sem imagem e sem texto -> provavelmente scan mal detectado
                return "scan"


def _has_large_images(page) -> Tuple[bool, float]:
    """
    Verifica se a página tem imagens que cobrem área significativa.

    Args:
        page: página do pdfplumber

    Returns:
        (tem_imagem_grande, cobertura_percentual)
    """
    images = page.images
    if not images:
        return False, 0.0

    page_area = page.width * page.height
    if page_area == 0:
        return False, 0.0

    # Calcula área total de imagens
    total_image_area = 0
    for img in images:
        # Coordenadas da imagem
        x0 = img.get('x0', 0)
        x1 = img.get('x1', 0)
        top = img.get('top', 0)
        bottom = img.get('bottom', 0)

        width = abs(x1 - x0)
        height = abs(bottom - top)
        total_image_area += width * height

    coverage = total_image_area / page_area
    has_large = coverage > config.IMAGE_AREA_THRESHOLD

    return has_large, coverage


def _get_text_coverage(page) -> float:
    """
    Calcula a porcentagem da página coberta por texto selecionável.

    Args:
        page: página do pdfplumber

    Returns:
        cobertura (0.0 a 1.0)
    """
    words = page.extract_words()
    if not words:
        return 0.0

    page_area = page.width * page.height
    if page_area == 0:
        return 0.0

    # Calcula área total coberta por palavras
    total_text_area = 0
    for w in words:
        x0 = w.get('x0', 0)
        x1 = w.get('x1', 0)
        top = w.get('top', 0)
        bottom = w.get('bottom', 0)

        width = abs(x1 - x0)
        height = abs(bottom - top)
        total_text_area += width * height

    return total_text_area / page_area


def has_extractable_text(pdf_path: str, page_number: int, min_chars: int = 10) -> bool:
    """
    Verifica se uma página tem texto extraível

    Args:
        pdf_path: caminho para o PDF
        page_number: número da página (1-indexed)
        min_chars: mínimo de caracteres para considerar "tem texto"

    Returns:
        True se tem texto extraível
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
    Retorna dimensões da página (largura, altura) em pontos

    Args:
        pdf_path: caminho para o PDF
        page_number: número da página (1-indexed)

    Returns:
        (width, height) em pontos
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        return (page.width, page.height)


def get_page_info(pdf_path: str, page_number: int) -> dict:
    """
    Retorna informações detalhadas sobre uma página (útil para debug).

    Args:
        pdf_path: caminho para o PDF
        page_number: número da página (1-indexed)

    Returns:
        dict com informações da página
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]

        has_large_img, image_coverage = _has_large_images(page)
        text_coverage = _get_text_coverage(page)

        text = page.extract_text()
        char_count = len(text.strip()) if text else 0

        return {
            'page_number': page_number,
            'width': page.width,
            'height': page.height,
            'num_images': len(page.images) if page.images else 0,
            'image_coverage': round(image_coverage * 100, 2),
            'has_large_images': has_large_img,
            'text_coverage': round(text_coverage * 100, 2),
            'char_count': char_count,
            'detected_type': detect_page_type(pdf_path, page_number)
        }
