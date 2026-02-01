"""
Extrator OCR usando docTR (PyTorch)
"""
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from pdf2image import convert_from_path
import config
from src.models.schemas import Block, BlockType
from src.utils.bbox import normalize_bbox, sort_blocks_by_position
from src.utils.text_normalizer import normalize_text
from src.preprocessing import preprocess_image


class OCREngine:
    """
    Engine OCR usando docTR
    """
    def __init__(self, device: str = None):
        """
        Inicializa o engine OCR
        
        Args:
            device: 'cuda' ou 'cpu' (se None, usa config.DEVICE)
        """
        from doctr.models import ocr_predictor
        
        self.device = device or config.DEVICE
        
        # Carrega modelo docTR
        # det_arch: arquitetura de detecção de texto
        # reco_arch: arquitetura de reconhecimento de texto
        self.model = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True,
            assume_straight_pages=False  # suporta páginas rotacionadas
        ).to(self.device)
        
        if config.VERBOSE:
            print(f"OCR Engine inicializado no device: {self.device}")
    
    def extract_from_image(self, image: np.ndarray) -> dict:
        """
        Extrai texto de uma imagem
        
        Args:
            image: imagem como numpy array (RGB ou grayscale)
        
        Returns:
            resultado do docTR
        """
        # docTR espera imagem RGB
        if len(image.shape) == 2:
            # Converte grayscale para RGB
            image = np.stack([image, image, image], axis=-1)
        
        # docTR espera valores 0-255 uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Executa OCR
        result = self.model([image])
        
        return result


def extract_ocr_page(pdf_path: str, page_number: int, 
                    preprocess: bool = True,
                    ocr_engine: Optional[OCREngine] = None) -> Tuple[List[Block], float, float]:
    """
    Extrai conteúdo de uma página usando OCR
    
    Args:
        pdf_path: caminho para o PDF
        page_number: número da página (1-indexed)
        preprocess: aplicar pré-processamento na imagem
        ocr_engine: engine OCR (se None, cria um novo)
    
    Returns:
        (blocos, largura, altura)
    """
    # Converte página para imagem
    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        dpi=config.IMAGE_DPI
    )
    
    if not images:
        return [], 0, 0
    
    image = images[0]
    page_width, page_height = image.size
    
    # Pré-processamento
    if preprocess:
        image_array = preprocess_image(image)
        # Converte de volta para PIL para compatibilidade
        image_pil = Image.fromarray(image_array)
    else:
        image_pil = image
    
    # Converte para numpy array RGB
    image_array = np.array(image_pil)
    
    # Cria engine se necessário
    if ocr_engine is None:
        ocr_engine = OCREngine()
    
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
            
            block = Block(
                block_id=f"p{page_number}_b{block_counter}",
                type=BlockType.PARAGRAPH,
                text=text,
                bbox=bbox,
                confidence=confidence
            )
            
            blocks.append(block)
            block_counter += 1
    
    return blocks
