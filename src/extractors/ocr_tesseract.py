"""
OCR extractor using Tesseract (pytesseract)

Tesseract 5.x with LSTM engine offers:
- Native Portuguese support (por, por_BR)
- Models specifically trained for each language
- Good accuracy on scanned documents
- Fast and lightweight

Requirements:
- System: tesseract-ocr, tesseract-ocr-por
- Python: pytesseract
"""

import logging
from typing import Any

from pdf2image import convert_from_path
from PIL import Image

import config
from src.models.schemas import Block, BlockType
from src.utils.bbox import normalize_bbox, sort_blocks_by_position
from src.utils.text_normalizer import normalize_text

logger = logging.getLogger(__name__)


class TesseractEngine:
    """
    OCR engine using Tesseract via pytesseract.
    """

    def __init__(self, lang: str | None = None, config_str: str | None = None):
        """
        Initialize the Tesseract engine.

        Args:
            lang: language(s) for OCR (e.g., 'por', 'por+eng')
            config_str: Tesseract configuration (e.g., '--oem 1 --psm 3')
        """
        import pytesseract

        self.pytesseract = pytesseract
        self.lang = lang or getattr(config, "OCR_LANG", "por")

        # Default configuration optimized for documents
        # --oem 1: LSTM engine (best quality)
        # --psm 3: Automatic page segmentation (default)
        # --psm 6: Assume uniform block of text (faster)
        default_config = "--oem 1 --psm 3"
        self.config = config_str or getattr(config, "TESSERACT_CONFIG", default_config)

        # Check if Tesseract is installed
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(
                "Tesseract Engine initialized: v%s, lang=%s", version, self.lang
            )
        except Exception as e:
            raise RuntimeError(
                f"Tesseract not found. Install with:\n"
                f"  Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-por\n"
                f"  macOS: brew install tesseract tesseract-lang\n"
                f"Error: {e}"
            ) from e

    def extract_from_image(self, image: Image.Image) -> dict[str, Any]:
        """
        Extract text from an image with position data.

        Args:
            image: PIL image

        Returns:
            dictionary with OCR data (text, conf, left, top, width, height, etc.)
        """
        # Extract complete data with position of each word
        data = self.pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=self.config,
            output_type=self.pytesseract.Output.DICT,
        )
        return data

    def extract_text_only(self, image: Image.Image) -> str:
        """
        Extract only the text (faster, no position data).

        Args:
            image: PIL image

        Returns:
            extracted text
        """
        return self.pytesseract.image_to_string(
            image, lang=self.lang, config=self.config
        )


def extract_ocr_page_tesseract(
    pdf_path: str, page_number: int, ocr_engine: TesseractEngine | None = None
) -> tuple[list[Block], float, float]:
    """
    Extract content from a page using Tesseract OCR.

    Args:
        pdf_path: path to the PDF
        page_number: page number (1-indexed)
        ocr_engine: OCR engine (if None, creates a new one)

    Returns:
        (blocks, width, height)
    """
    # Convert page to high-resolution image
    dpi = getattr(config, "OCR_DPI", config.IMAGE_DPI)

    images = convert_from_path(
        pdf_path, first_page=page_number, last_page=page_number, dpi=dpi
    )

    if not images:
        return [], 0, 0

    image = images[0]
    page_width, page_height = image.size

    # Create engine if needed
    if ocr_engine is None:
        ocr_engine = TesseractEngine()

    # Run OCR
    data = ocr_engine.extract_from_image(image)

    # Close PIL image to avoid memory leak
    # (pdf2image/Poppler keeps internal references)
    image.close()
    del images

    # Process Tesseract result
    blocks = _parse_tesseract_result(data, page_number, page_width, page_height)

    # Sort blocks by position
    blocks = sort_blocks_by_position(blocks)

    return blocks, page_width, page_height


def _parse_tesseract_result(
    data: dict[str, Any], page_number: int, page_width: float, page_height: float
) -> list[Block]:
    """
    Parse Tesseract result and convert to blocks.

    Tesseract returns levels:
    - 1: page
    - 2: block
    - 3: paragraph
    - 4: line
    - 5: word
    """
    blocks = []

    # Group words by block (level 2)
    current_block_num = -1
    current_block_words = []
    current_block_boxes = []
    current_block_confs = []

    n_boxes = len(data["text"])

    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        block_num = data["block_num"][i]

        # Ignore empty entries or those with very low confidence
        if not text or conf < 0:
            continue

        # New block detected
        if block_num != current_block_num:
            # Save previous block if exists
            if current_block_words:
                block = _create_block_from_words(
                    current_block_words,
                    current_block_boxes,
                    current_block_confs,
                    page_number,
                    len(blocks) + 1,
                    page_width,
                    page_height,
                )
                if block:
                    blocks.append(block)

            # Start new block
            current_block_num = block_num
            current_block_words = []
            current_block_boxes = []
            current_block_confs = []

        # Add word to current block
        current_block_words.append(text)
        current_block_boxes.append(
            {
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            }
        )
        current_block_confs.append(conf)

    # Save last block
    if current_block_words:
        block = _create_block_from_words(
            current_block_words,
            current_block_boxes,
            current_block_confs,
            page_number,
            len(blocks) + 1,
            page_width,
            page_height,
        )
        if block:
            blocks.append(block)

    return blocks


def _create_block_from_words(
    words: list[str],
    boxes: list[dict],
    confs: list[int],
    page_number: int,
    block_counter: int,
    page_width: float,
    page_height: float,
) -> Block | None:
    """
    Create a Block from a list of words.
    """
    if not words:
        return None

    # Join words into text
    text = " ".join(words)
    text = normalize_text(text)

    if not text or len(text.strip()) < 2:
        return None

    # Calculate block bbox (union of all words)
    x1 = min(b["left"] for b in boxes)
    y1 = min(b["top"] for b in boxes)
    x2 = max(b["left"] + b["width"] for b in boxes)
    y2 = max(b["top"] + b["height"] for b in boxes)

    # Normalize bbox
    bbox = normalize_bbox([x1, y1, x2, y2], page_width, page_height)

    # Calculate average confidence (Tesseract uses 0-100)
    confidence = sum(confs) / len(confs) / 100.0

    # Filter blocks with very low confidence
    min_conf = getattr(config, "MIN_CONFIDENCE", 0.3)
    if confidence < min_conf:
        return None

    return Block(
        block_id=f"p{page_number}_b{block_counter}",
        type=BlockType.PARAGRAPH,
        text=text,
        bbox=bbox,
        confidence=confidence,
    )
