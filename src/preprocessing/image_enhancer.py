"""
Pré-processamento de imagens para melhorar qualidade do OCR
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple
import config


def preprocess_image(image: Image.Image, dpi: int = None) -> np.ndarray:
    """
    Pipeline completo de pré-processamento
    
    Args:
        image: imagem PIL
        dpi: DPI da imagem (se None, usa config)
    
    Returns:
        imagem processada como array numpy
    """
    if dpi is None:
        dpi = config.IMAGE_DPI
    
    # Converte para numpy array
    img_array = np.array(image)
    
    # Converte para escala de cinza se necessário
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # 1. Deskew (correção de rotação)
    img_deskewed = deskew_image(img_gray)
    
    # 2. Melhora contraste
    img_enhanced = enhance_contrast(img_deskewed)
    
    # 3. Remove ruído
    img_denoised = remove_noise(img_enhanced)
    
    # 4. Binarização
    img_binary = binarize_image(img_denoised, method=config.BINARIZATION_METHOD)
    
    return img_binary


def deskew_image(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Corrige rotação da imagem (deskew)
    
    Args:
        image: imagem em escala de cinza
        max_angle: ângulo máximo de correção (graus)
    
    Returns:
        imagem corrigida
    """
    # Detecta ângulo de rotação usando transformada de Hough
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is None:
        return image
    
    # Calcula ângulo médio das linhas detectadas
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) <= max_angle:
            angles.append(angle)
    
    if not angles:
        return image
    
    # Usa mediana dos ângulos para evitar outliers
    rotation_angle = np.median(angles)
    
    # Aplica rotação apenas se significativa
    if abs(rotation_angle) < config.DESKEW_ANGLE_THRESHOLD:
        return image
    
    # Rotaciona imagem
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def binarize_image(image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    """
    Binariza imagem (preto e branco)
    
    Args:
        image: imagem em escala de cinza
        method: 'otsu' ou 'adaptive'
    
    Returns:
        imagem binarizada
    """
    if method == 'otsu':
        # Método de Otsu: automático
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        # Binarização adaptativa: melhor para iluminação irregular
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    return binary


def remove_noise(image: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """
    Remove ruído da imagem
    
    Args:
        image: imagem em escala de cinza
        kernel_size: tamanho do kernel para morfologia
    
    Returns:
        imagem sem ruído
    """
    if kernel_size is None:
        kernel_size = config.DENOISE_KERNEL_SIZE
    
    # Filtro mediano para remover sal e pimenta
    denoised = cv2.medianBlur(image, 3)
    
    # Operação morfológica de abertura (erosão + dilatação)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    
    return opened


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Melhora contraste da imagem usando CLAHE
    
    Args:
        image: imagem em escala de cinza
    
    Returns:
        imagem com contraste melhorado
    """
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    return enhanced


def remove_borders(image: np.ndarray, border_size: int = 10) -> np.ndarray:
    """
    Remove bordas da imagem (útil para scans com bordas pretas)
    
    Args:
        image: imagem
        border_size: pixels a remover de cada lado
    
    Returns:
        imagem sem bordas
    """
    h, w = image.shape[:2]
    return image[border_size:h-border_size, border_size:w-border_size]


def resize_to_dpi(image: Image.Image, target_dpi: int = 300) -> Image.Image:
    """
    Redimensiona imagem para DPI desejado
    
    Args:
        image: imagem PIL
        target_dpi: DPI alvo
    
    Returns:
        imagem redimensionada
    """
    # Obtém DPI atual (se disponível)
    current_dpi = image.info.get('dpi', (72, 72))[0]
    
    if current_dpi == target_dpi:
        return image
    
    # Calcula novo tamanho
    scale = target_dpi / current_dpi
    new_size = (int(image.width * scale), int(image.height * scale))
    
    # Redimensiona com alta qualidade
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return resized
