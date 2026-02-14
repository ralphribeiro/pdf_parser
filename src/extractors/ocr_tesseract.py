"""
Extrator OCR usando Tesseract (pytesseract)

Tesseract 5.x com motor LSTM oferece:
- Suporte nativo a português (por, por_BR)
- Modelos treinados especificamente para cada idioma
- Boa precisão em documentos escaneados
- Rápido e leve

Requisitos:
- Sistema: tesseract-ocr, tesseract-ocr-por
- Python: pytesseract
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pdf2image import convert_from_path
from PIL import Image

import config
from src.models.schemas import Block, BlockType
from src.utils.bbox import normalize_bbox, sort_blocks_by_position
from src.utils.text_normalizer import normalize_text

logger = logging.getLogger(__name__)


class TesseractEngine:
    """
    Engine OCR usando Tesseract via pytesseract
    """
    def __init__(self, lang: str = None, config_str: str = None):
        """
        Inicializa o engine Tesseract
        
        Args:
            lang: idioma(s) para OCR (ex: 'por', 'por+eng')
            config_str: configuração do Tesseract (ex: '--oem 1 --psm 3')
        """
        import pytesseract
        
        self.pytesseract = pytesseract
        self.lang = lang or getattr(config, 'OCR_LANG', 'por')
        
        # Configuração padrão otimizada para documentos
        # --oem 1: LSTM engine (melhor qualidade)
        # --psm 3: Automatic page segmentation (padrão)
        # --psm 6: Assume uniform block of text (mais rápido)
        default_config = '--oem 1 --psm 3'
        self.config = config_str or getattr(config, 'TESSERACT_CONFIG', default_config)
        
        # Verifica se Tesseract está instalado
        try:
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract Engine inicializado: v%s, lang=%s", version, self.lang)
        except Exception as e:
            raise RuntimeError(
                f"Tesseract não encontrado. Instale com:\n"
                f"  Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-por\n"
                f"  macOS: brew install tesseract tesseract-lang\n"
                f"Erro: {e}"
            )
    
    def extract_from_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extrai texto de uma imagem com dados de posição
        
        Args:
            image: imagem PIL
        
        Returns:
            dicionário com dados do OCR (text, conf, left, top, width, height, etc)
        """
        # Extrai dados completos com posição de cada palavra
        data = self.pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=self.config,
            output_type=self.pytesseract.Output.DICT
        )
        return data
    
    def extract_text_only(self, image: Image.Image) -> str:
        """
        Extrai apenas o texto (mais rápido, sem posição)
        
        Args:
            image: imagem PIL
        
        Returns:
            texto extraído
        """
        return self.pytesseract.image_to_string(
            image,
            lang=self.lang,
            config=self.config
        )


def extract_ocr_page_tesseract(pdf_path: str, page_number: int,
                               ocr_engine: Optional[TesseractEngine] = None) -> Tuple[List[Block], float, float]:
    """
    Extrai conteúdo de uma página usando Tesseract OCR
    
    Args:
        pdf_path: caminho para o PDF
        page_number: número da página (1-indexed)
        ocr_engine: engine OCR (se None, cria um novo)
    
    Returns:
        (blocos, largura, altura)
    """
    # Converte página para imagem em alta resolução
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
    
    # Cria engine se necessário
    if ocr_engine is None:
        ocr_engine = TesseractEngine()
    
    # Executa OCR
    data = ocr_engine.extract_from_image(image)
    
    # Fecha imagem PIL para evitar memory leak
    # (pdf2image/Poppler mantém referências internas)
    image.close()
    del images
    
    # Processa resultado do Tesseract
    blocks = _parse_tesseract_result(data, page_number, page_width, page_height)
    
    # Ordena blocos por posição
    blocks = sort_blocks_by_position(blocks)
    
    return blocks, page_width, page_height


def _parse_tesseract_result(data: Dict[str, Any], page_number: int,
                           page_width: float, page_height: float) -> List[Block]:
    """
    Parseia resultado do Tesseract e converte em blocos
    
    Tesseract retorna níveis:
    - 1: page
    - 2: block
    - 3: paragraph
    - 4: line
    - 5: word
    """
    blocks = []
    
    # Agrupa palavras por bloco (level 2)
    current_block_num = -1
    current_block_words = []
    current_block_boxes = []
    current_block_confs = []
    
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        block_num = data['block_num'][i]
        
        # Ignora entradas vazias ou com confiança muito baixa
        if not text or conf < 0:
            continue
        
        # Novo bloco detectado
        if block_num != current_block_num:
            # Salva bloco anterior se existir
            if current_block_words:
                block = _create_block_from_words(
                    current_block_words,
                    current_block_boxes,
                    current_block_confs,
                    page_number,
                    len(blocks) + 1,
                    page_width,
                    page_height
                )
                if block:
                    blocks.append(block)
            
            # Inicia novo bloco
            current_block_num = block_num
            current_block_words = []
            current_block_boxes = []
            current_block_confs = []
        
        # Adiciona palavra ao bloco atual
        current_block_words.append(text)
        current_block_boxes.append({
            'left': data['left'][i],
            'top': data['top'][i],
            'width': data['width'][i],
            'height': data['height'][i]
        })
        current_block_confs.append(conf)
    
    # Salva último bloco
    if current_block_words:
        block = _create_block_from_words(
            current_block_words,
            current_block_boxes,
            current_block_confs,
            page_number,
            len(blocks) + 1,
            page_width,
            page_height
        )
        if block:
            blocks.append(block)
    
    return blocks


def _create_block_from_words(words: List[str], boxes: List[dict], 
                            confs: List[int], page_number: int,
                            block_counter: int, page_width: float,
                            page_height: float) -> Optional[Block]:
    """
    Cria um Block a partir de uma lista de palavras
    """
    if not words:
        return None
    
    # Junta palavras em texto
    text = ' '.join(words)
    text = normalize_text(text)
    
    if not text or len(text.strip()) < 2:
        return None
    
    # Calcula bbox do bloco (união de todas as palavras)
    x1 = min(b['left'] for b in boxes)
    y1 = min(b['top'] for b in boxes)
    x2 = max(b['left'] + b['width'] for b in boxes)
    y2 = max(b['top'] + b['height'] for b in boxes)
    
    # Normaliza bbox
    bbox = normalize_bbox([x1, y1, x2, y2], page_width, page_height)
    
    # Calcula confiança média (Tesseract usa 0-100)
    confidence = sum(confs) / len(confs) / 100.0
    
    # Filtra blocos com confiança muito baixa
    min_conf = getattr(config, 'MIN_CONFIDENCE', 0.3)
    if confidence < min_conf:
        return None
    
    return Block(
        block_id=f"p{page_number}_b{block_counter}",
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=bbox,
        confidence=confidence
    )
