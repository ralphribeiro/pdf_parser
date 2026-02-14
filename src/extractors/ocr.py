"""
Extrator OCR usando docTR (PyTorch)

IMPORTANTE: O docTR faz seu próprio pré-processamento internamente.
Passar imagem binarizada/grayscale DEGRADA a qualidade do OCR.
Sempre passar imagem RGB original em alta resolução.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
from pdf2image import convert_from_path
from PIL import Image

import config
from src.models.schemas import Block, BlockType
from src.utils.bbox import normalize_bbox, sort_blocks_by_position
from src.utils.text_normalizer import normalize_text

logger = logging.getLogger(__name__)


class DocTREngine:
    """
    Engine OCR usando docTR (mindee/doctr)

    Melhores práticas:
    - Passar imagem RGB original (NÃO binarizada)
    - DPI alto (300-400) melhora qualidade

    Configuração de orientação (via config.py):
    - ASSUME_STRAIGHT_PAGES=False: detecta texto rotacionado (mais preciso, mais lento)
    - DETECT_ORIENTATION=True: detecta e corrige orientação da página (0/90/180/270)
    - STRAIGHTEN_PAGES=True: corrige páginas inclinadas automaticamente
    """
    def __init__(self, device: str = None):
        """
        Inicializa o engine OCR

        Args:
            device: 'cuda' ou 'cpu' (se None, usa config.DEVICE)
        """
        from doctr.models import ocr_predictor

        self.device = device or config.DEVICE

        # Configuração de orientação de página
        assume_straight = getattr(config, 'ASSUME_STRAIGHT_PAGES', False)
        detect_orient = getattr(config, 'DETECT_ORIENTATION', True)
        straighten = getattr(config, 'STRAIGHTEN_PAGES', True)

        # Carrega modelo docTR com configuração otimizada
        # det_arch: arquitetura de detecção de texto
        # reco_arch: arquitetura de reconhecimento de texto
        self.model = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True,
            assume_straight_pages=assume_straight,
            detect_orientation=detect_orient,
            straighten_pages=straighten,
        ).to(self.device)

        orient_info = []
        if not assume_straight:
            orient_info.append("rotação detectada")
        if detect_orient:
            orient_info.append("orientação auto")
        if straighten:
            orient_info.append("endireitamento auto")
        orient_str = f" ({', '.join(orient_info)})" if orient_info else ""
        logger.info("DocTR Engine inicializado no device: %s%s", self.device, orient_str)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepara uma imagem para processamento pelo docTR.

        Args:
            image: imagem como numpy array

        Returns:
            imagem preparada (RGB, uint8)
        """
        # docTR espera imagem RGB
        if len(image.shape) == 2:
            # Converte grayscale para RGB (não recomendado, mas suporta)
            image = np.stack([image, image, image], axis=-1)

        # Remove canal alpha se presente
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # docTR espera valores 0-255 uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return image

    def extract_from_image(self, image: np.ndarray) -> dict:
        """
        Extrai texto de uma imagem

        Args:
            image: imagem como numpy array RGB (NÃO passar grayscale/binarizada!)

        Returns:
            resultado do docTR
        """
        prepared = self._prepare_image(image)
        result = self.model([prepared])
        return result

    def extract_from_images_batch(self, images: List[np.ndarray]) -> dict:
        """
        Extrai texto de múltiplas imagens em batch.

        Mais eficiente que processar uma imagem por vez, especialmente com GPU.
        O docTR internamente faz batching para maximizar throughput.

        Args:
            images: lista de imagens como numpy arrays RGB

        Returns:
            resultado do docTR com múltiplas páginas
        """
        if not images:
            return None

        # Prepara todas as imagens
        prepared_images = [self._prepare_image(img) for img in images]

        # Processa batch
        result = self.model(prepared_images)

        return result


# Alias para compatibilidade
OCREngine = DocTREngine


def extract_ocr_page(pdf_path: str, page_number: int,
                    preprocess: bool = False,  # DESATIVADO por padrão - degrada qualidade
                    ocr_engine: Optional[DocTREngine] = None) -> Tuple[List[Block], float, float]:
    """
    Extrai conteúdo de uma página usando OCR

    Args:
        pdf_path: caminho para o PDF
        page_number: número da página (1-indexed)
        preprocess: NÃO USAR - mantido para compatibilidade
        ocr_engine: engine OCR (se None, cria um novo)

    Returns:
        (blocos, largura, altura)
    """
    # Converte página para imagem em alta resolução
    # DPI alto é crucial para qualidade do OCR
    dpi = getattr(config, 'OCR_DPI', config.IMAGE_DPI)

    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        dpi=dpi
    )

    if not images:
        return [], 0, 0

    image = images[0]
    page_width, page_height = image.size

    # IMPORTANTE: Passar imagem RGB original SEM pré-processamento
    # O docTR faz seu próprio pré-processamento otimizado internamente
    # Binarização/grayscale DEGRADA a qualidade do OCR
    image_array = np.array(image)

    # Fecha imagem PIL para evitar memory leak
    # (pdf2image/Poppler mantém referências internas)
    image.close()
    del images

    # Garante que é RGB (não RGBA)
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Cria engine se necessário
    if ocr_engine is None:
        ocr_engine = DocTREngine()

    # Executa OCR
    result = ocr_engine.extract_from_image(image_array)

    # Processa resultado do docTR
    blocks = _parse_doctr_result(result, page_number, page_width, page_height)

    # Ordena blocos por posição
    blocks = sort_blocks_by_position(blocks)

    return blocks, page_width, page_height


def _parse_doctr_result(result, page_number: int, page_width: float,
                       page_height: float) -> List[Block]:
    """
    Parseia resultado do docTR e converte em blocos
    """
    blocks = []
    block_counter = 1

    # docTR retorna estrutura: pages -> blocks -> lines -> words
    for page in result.pages:
        for block_data in page.blocks:
            # Extrai texto do bloco
            block_text = []

            # Calcula bbox do bloco (união de todas as linhas)
            all_line_bboxes = []
            total_confidence = 0
            word_count = 0

            for line in block_data.lines:
                line_text = []
                for word in line.words:
                    line_text.append(word.value)
                    total_confidence += word.confidence
                    word_count += 1

                block_text.append(" ".join(line_text))

                # Bbox da linha (docTR retorna coordenadas normalizadas)
                line_bbox = line.geometry
                # line_bbox é ((x1, y1), (x2, y2)) normalizado
                all_line_bboxes.append([
                    line_bbox[0][0],  # x1
                    line_bbox[0][1],  # y1
                    line_bbox[1][0],  # x2
                    line_bbox[1][1]   # y2
                ])

            if not block_text:
                continue

            # Junta texto do bloco
            text = "\n".join(block_text)
            text = normalize_text(text)

            if not text:
                continue

            # Calcula bbox do bloco (união de todas as linhas)
            if all_line_bboxes:
                bbox = [
                    min(b[0] for b in all_line_bboxes),
                    min(b[1] for b in all_line_bboxes),
                    max(b[2] for b in all_line_bboxes),
                    max(b[3] for b in all_line_bboxes)
                ]
            else:
                bbox = [0.0, 0.0, 1.0, 1.0]

            # Calcula confiança média
            confidence = total_confidence / word_count if word_count > 0 else 0.0

            # Filtra blocos com confiança muito baixa
            if confidence < config.MIN_CONFIDENCE:
                continue

            # Preserva dados por linha (texto + bbox) para overlay preciso
            lines_data = [
                {"text": line_text, "bbox": line_bbox}
                for line_text, line_bbox in zip(block_text, all_line_bboxes)
            ]

            block = Block(
                block_id=f"p{page_number}_b{block_counter}",
                type=BlockType.PARAGRAPH,
                text=text,
                bbox=bbox,
                confidence=confidence,
                lines=lines_data
            )

            blocks.append(block)
            block_counter += 1

    return blocks
