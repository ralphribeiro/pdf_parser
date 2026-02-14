"""
Image preprocessing to improve OCR quality
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple
import config


def preprocess_image(image: Image.Image, dpi: int = None) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    Args:
        image: PIL image
        dpi: image DPI (if None, uses config)

    Returns:
        processed image as numpy array
    """
    if dpi is None:
        dpi = config.IMAGE_DPI

    # Convert to numpy array
    img_array = np.array(image)

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    # 1. Deskew (rotation correction)
    img_deskewed = deskew_image(img_gray)

    # 2. Enhance contrast
    img_enhanced = enhance_contrast(img_deskewed)

    # 3. Remove noise
    img_denoised = remove_noise(img_enhanced)

    # 4. Binarization
    img_binary = binarize_image(img_denoised, method=config.BINARIZATION_METHOD)

    return img_binary


def deskew_image(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Correct image rotation (deskew).

    Args:
        image: grayscale image
        max_angle: maximum correction angle (degrees)

    Returns:
        corrected image
    """
    # Detect rotation angle using Hough transform
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        return image

    # Calculate average angle of detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return image

    # Use median of angles to avoid outliers
    rotation_angle = np.median(angles)

    # Apply rotation only if significant
    if abs(rotation_angle) < config.DESKEW_ANGLE_THRESHOLD:
        return image

    # Rotate image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)

    return rotated


def binarize_image(image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    """
    Binarize image (black and white).

    Args:
        image: grayscale image
        method: 'otsu' or 'adaptive'

    Returns:
        binarized image
    """
    if method == 'otsu':
        # Otsu's method: automatic
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        # Adaptive binarization: better for uneven lighting
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return binary


def remove_noise(image: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """
    Remove noise from the image.

    Args:
        image: grayscale image
        kernel_size: kernel size for morphology

    Returns:
        denoised image
    """
    if kernel_size is None:
        kernel_size = config.DENOISE_KERNEL_SIZE

    # Median filter to remove salt and pepper noise
    denoised = cv2.medianBlur(image, 3)

    # Morphological opening operation (erosion + dilation)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    return opened


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.

    Args:
        image: grayscale image

    Returns:
        contrast-enhanced image
    """
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    return enhanced


def remove_borders(image: np.ndarray, border_size: int = 10) -> np.ndarray:
    """
    Remove image borders (useful for scans with black borders).

    Args:
        image: image
        border_size: pixels to remove from each side

    Returns:
        image without borders
    """
    h, w = image.shape[:2]
    return image[border_size:h-border_size, border_size:w-border_size]


def resize_to_dpi(image: Image.Image, target_dpi: int = 300) -> Image.Image:
    """
    Resize image to desired DPI.

    Args:
        image: PIL image
        target_dpi: target DPI

    Returns:
        resized image
    """
    # Get current DPI (if available)
    current_dpi = image.info.get('dpi', (72, 72))[0]

    if current_dpi == target_dpi:
        return image

    # Calculate new size
    scale = target_dpi / current_dpi
    new_size = (int(image.width * scale), int(image.height * scale))

    # Resize with high quality
    resized = image.resize(new_size, Image.Resampling.LANCZOS)

    return resized
