"""
OCR extractor using docTR (PyTorch)

IMPORTANT: docTR handles its own preprocessing internally.
Passing binarized/grayscale images DEGRADES OCR quality.
Always pass the original RGB image in high resolution.
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
    OCR engine using docTR (mindee/doctr)

    Best practices:
    - Pass original RGB image (NOT binarized)
    - High DPI (300-400) improves quality

    Orientation settings (via config.py):
    - ASSUME_STRAIGHT_PAGES=False: detects rotated text (more accurate, slower)
    - DETECT_ORIENTATION=True: detects and corrects page orientation (0/90/180/270)
    - STRAIGHTEN_PAGES=True: automatically corrects skewed pages
    """
    def __init__(self, device: str = None):
        """
        Initialize the OCR engine.

        Args:
            device: 'cuda' or 'cpu' (if None, uses config.DEVICE)
        """
        from doctr.models import ocr_predictor

        self.device = device or config.DEVICE

        # Page orientation settings
        assume_straight = getattr(config, 'ASSUME_STRAIGHT_PAGES', False)
        detect_orient = getattr(config, 'DETECT_ORIENTATION', True)
        straighten = getattr(config, 'STRAIGHTEN_PAGES', True)

        # Load docTR model with optimized configuration
        # det_arch: text detection architecture
        # reco_arch: text recognition architecture
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
            orient_info.append("rotation detected")
        if detect_orient:
            orient_info.append("auto orientation")
        if straighten:
            orient_info.append("auto straightening")
        orient_str = f" ({', '.join(orient_info)})" if orient_info else ""
        logger.info("DocTR Engine initialized on device: %s%s", self.device, orient_str)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare an image for docTR processing.

        Args:
            image: image as numpy array

        Returns:
            prepared image (RGB, uint8)
        """
        # docTR expects RGB image
        if len(image.shape) == 2:
            # Convert grayscale to RGB (not recommended, but supported)
            image = np.stack([image, image, image], axis=-1)

        # Remove alpha channel if present
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        # docTR expects 0-255 uint8 values
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return image

    def extract_from_image(self, image: np.ndarray) -> dict:
        """
        Extract text from an image.

        Args:
            image: image as RGB numpy array (do NOT pass grayscale/binarized!)

        Returns:
            docTR result
        """
        prepared = self._prepare_image(image)
        result = self.model([prepared])
        return result

    def extract_from_images_batch(self, images: List[np.ndarray]) -> dict:
        """
        Extract text from multiple images in batch.

        More efficient than processing one image at a time, especially with GPU.
        docTR internally handles batching to maximize throughput.

        Args:
            images: list of images as RGB numpy arrays

        Returns:
            docTR result with multiple pages
        """
        if not images:
            return None

        # Prepare all images
        prepared_images = [self._prepare_image(img) for img in images]

        # Process batch
        result = self.model(prepared_images)

        return result


# Alias for compatibility
OCREngine = DocTREngine


def extract_ocr_page(pdf_path: str, page_number: int,
                    preprocess: bool = False,  # DISABLED by default - degrades quality
                    ocr_engine: Optional[DocTREngine] = None) -> Tuple[List[Block], float, float]:
    """
    Extract content from a page using OCR.

    Args:
        pdf_path: path to the PDF
        page_number: page number (1-indexed)
        preprocess: DO NOT USE - kept for compatibility
        ocr_engine: OCR engine (if None, creates a new one)

    Returns:
        (blocks, width, height)
    """
    # Convert page to high-resolution image
    # High DPI is crucial for OCR quality
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

    # IMPORTANT: Pass original RGB image WITHOUT preprocessing
    # docTR handles its own optimized preprocessing internally
    # Binarization/grayscale DEGRADES OCR quality
    image_array = np.array(image)

    # Close PIL image to avoid memory leak
    # (pdf2image/Poppler keeps internal references)
    image.close()
    del images

    # Ensure it's RGB (not RGBA)
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Create engine if needed
    if ocr_engine is None:
        ocr_engine = DocTREngine()

    # Run OCR
    result = ocr_engine.extract_from_image(image_array)

    # Process docTR result
    blocks = _parse_doctr_result(result, page_number, page_width, page_height)

    # Sort blocks by position
    blocks = sort_blocks_by_position(blocks)

    return blocks, page_width, page_height


def _parse_doctr_result(result, page_number: int, page_width: float,
                       page_height: float) -> List[Block]:
    """
    Parse docTR result and convert to blocks.
    """
    blocks = []
    block_counter = 1

    # docTR returns structure: pages -> blocks -> lines -> words
    for page in result.pages:
        for block_data in page.blocks:
            # Extract block text
            block_text = []

            # Calculate block bbox (union of all lines)
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

                # Line bbox (docTR returns normalized coordinates)
                line_bbox = line.geometry
                # line_bbox is ((x1, y1), (x2, y2)) normalized
                all_line_bboxes.append([
                    line_bbox[0][0],  # x1
                    line_bbox[0][1],  # y1
                    line_bbox[1][0],  # x2
                    line_bbox[1][1]   # y2
                ])

            if not block_text:
                continue

            # Join block text
            text = "\n".join(block_text)
            text = normalize_text(text)

            if not text:
                continue

            # Calculate block bbox (union of all lines)
            if all_line_bboxes:
                bbox = [
                    min(b[0] for b in all_line_bboxes),
                    min(b[1] for b in all_line_bboxes),
                    max(b[2] for b in all_line_bboxes),
                    max(b[3] for b in all_line_bboxes)
                ]
            else:
                bbox = [0.0, 0.0, 1.0, 1.0]

            # Calculate average confidence
            confidence = total_confidence / word_count if word_count > 0 else 0.0

            # Filter blocks with very low confidence
            if confidence < config.MIN_CONFIDENCE:
                continue

            # Preserve per-line data (text + bbox) for precise overlay
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
