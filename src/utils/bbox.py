"""
Funções para manipulação de bounding boxes
"""
from typing import List, Tuple, Any


def normalize_bbox(bbox: List[float], page_width: float, page_height: float) -> List[float]:
    """
    Normaliza bounding box de coordenadas absolutas para relativas (0-1)
    
    Args:
        bbox: [x1, y1, x2, y2] em coordenadas absolutas
        page_width: largura da página
        page_height: altura da página
    
    Returns:
        [x1, y1, x2, y2] normalizado (0-1)
    """
    return [
        bbox[0] / page_width,
        bbox[1] / page_height,
        bbox[2] / page_width,
        bbox[3] / page_height
    ]


def denormalize_bbox(bbox: List[float], page_width: float, page_height: float) -> List[float]:
    """
    Converte bbox normalizado (0-1) para coordenadas absolutas
    """
    return [
        bbox[0] * page_width,
        bbox[1] * page_height,
        bbox[2] * page_width,
        bbox[3] * page_height
    ]


def bbox_area(bbox: List[float]) -> float:
    """Calcula área de um bounding box"""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calcula a área de overlap entre dois bounding boxes
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    return (x2 - x1) * (y2 - y1)


def sort_blocks_by_position(blocks: List[Any], reading_order: str = 'top-to-bottom') -> List[Any]:
    """
    Ordena blocos por posição de leitura
    
    Args:
        blocks: lista de blocos com atributo 'bbox'
        reading_order: 'top-to-bottom' ou 'left-to-right'
    
    Returns:
        blocos ordenados
    """
    if reading_order == 'top-to-bottom':
        # Ordena por Y (topo), depois por X (esquerda)
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    elif reading_order == 'left-to-right':
        # Ordena por X (esquerda), depois por Y (topo)
        return sorted(blocks, key=lambda b: (b.bbox[0], b.bbox[1]))
    else:
        return blocks


def merge_nearby_boxes(boxes: List[List[float]], threshold: float = 0.01) -> List[List[float]]:
    """
    Mescla bounding boxes próximos (útil para juntar palavras em linhas)
    
    Args:
        boxes: lista de bboxes
        threshold: distância máxima para considerar "próximo" (normalizado)
    
    Returns:
        lista de bboxes mesclados
    """
    if not boxes:
        return []
    
    # Ordena boxes por posição
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    
    merged = [sorted_boxes[0]]
    
    for current in sorted_boxes[1:]:
        last = merged[-1]
        
        # Verifica se estão próximos (mesma linha, aproximadamente)
        y_distance = abs(current[1] - last[1])
        x_distance = current[0] - last[2]  # distância horizontal
        
        if y_distance < threshold and 0 <= x_distance < threshold:
            # Mescla: expande o último box
            merged[-1] = [
                min(last[0], current[0]),
                min(last[1], current[1]),
                max(last[2], current[2]),
                max(last[3], current[3])
            ]
        else:
            merged.append(current)
    
    return merged
