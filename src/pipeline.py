"""
Main pipeline for PDF processing.

Supports multiple OCR engines:
- doctr: Deep learning (PyTorch), fast with GPU
- tesseract: Traditional LSTM, good for Portuguese

Supports parallel processing:
- Digital pages: ProcessPoolExecutor (CPU-bound)
- OCR pages with docTR: Batch processing (GPU)
- OCR pages with Tesseract: ProcessPoolExecutor (CPU)
"""
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pdfplumber

import config
from src.detector import detect_page_type, get_page_dimensions
from src.extractors.digital import extract_digital_page
from src.extractors.ocr import DocTREngine, OCREngine, extract_ocr_page
from src.extractors.tables import extract_tables_digital
from src.models.schemas import Block, Document, Page
from src.utils.ocr_postprocess import postprocess_ocr_text

logger = logging.getLogger(__name__)

# Conditional Tesseract import
try:
    from src.extractors.ocr_tesseract import TesseractEngine, extract_ocr_page_tesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    TesseractEngine = None
    extract_ocr_page_tesseract = None


# =============================================================================
# Helper functions for parallel processing (must be top-level for pickle)
# =============================================================================

def _process_digital_page_worker(args: Tuple[str, int, bool]) -> Tuple[int, Page]:
    """
    Worker to process a digital page in parallel.

    Args:
        args: tuple (pdf_path, page_number, extract_tables)

    Returns:
        tuple (page_number, Page)
    """
    pdf_path, page_number, extract_tables = args

    try:
        blocks, width, height = extract_digital_page(pdf_path, page_number)

        # Extract tables if requested
        if extract_tables:
            try:
                table_blocks = extract_tables_digital(pdf_path, page_number)
                if table_blocks:
                    from src.utils.bbox import bbox_area, bbox_overlap, sort_blocks_by_position

                    filtered_blocks = []
                    for text_block in blocks:
                        should_keep = True
                        for table_block in table_blocks:
                            overlap = bbox_overlap(text_block.bbox, table_block.bbox)
                            text_area = bbox_area(text_block.bbox)
                            if text_area > 0 and (overlap / text_area) > 0.5:
                                should_keep = False
                                break
                        if should_keep:
                            filtered_blocks.append(text_block)

                    blocks = filtered_blocks + table_blocks
                    blocks = sort_blocks_by_position(blocks)
            except Exception:
                pass  # Silently ignore table errors

        page = Page(
            page=page_number,
            source="digital",
            blocks=blocks,
            width=width,
            height=height,
        )
        return (page_number, page)

    except Exception:
        return (page_number, Page(page=page_number, source="digital", blocks=[]))


def _process_tesseract_page_worker(args: Tuple[str, int, str, str]) -> Tuple[int, Page]:
    """
    Worker to process a page with Tesseract in parallel.

    Args:
        args: tuple (pdf_path, page_number, lang, tesseract_config)

    Returns:
        tuple (page_number, Page)
    """
    pdf_path, page_number, lang, tesseract_config = args

    try:
        from src.extractors.ocr_tesseract import TesseractEngine, extract_ocr_page_tesseract

        engine = TesseractEngine(lang=lang, tesseract_config=tesseract_config)
        blocks, width, height = extract_ocr_page_tesseract(
            pdf_path, page_number, ocr_engine=engine
        )

        if getattr(config, "OCR_POSTPROCESS", True):
            blocks = _postprocess_blocks(blocks)

        page = Page(
            page=page_number,
            source="ocr",
            blocks=blocks,
            width=width,
            height=height,
        )
        return (page_number, page)

    except Exception:
        return (page_number, Page(page=page_number, source="ocr", blocks=[]))


def _postprocess_blocks(blocks: list) -> list:
    """Apply post-processing to OCR blocks (standalone version for workers)."""
    fix_errors = getattr(config, "OCR_FIX_ERRORS", True)
    min_line = getattr(config, "OCR_MIN_LINE_LENGTH", 3)

    processed_blocks = []
    for block in blocks:
        if block.text:
            cleaned_text = postprocess_ocr_text(
                block.text,
                clean=True,
                fix_errors=fix_errors,
                merge_words=False,
                min_line_length=min_line,
            )

            if cleaned_text and len(cleaned_text.strip()) >= 2:
                processed_blocks.append(
                    Block(
                        block_id=block.block_id,
                        type=block.type,
                        text=cleaned_text,
                        bbox=block.bbox,
                        confidence=block.confidence,
                        rows=block.rows,
                    )
                )
        else:
            processed_blocks.append(block)

    return processed_blocks


def _classify_page_worker(args: Tuple[str, int]) -> Tuple[int, str]:
    """
    Worker to classify a page type in parallel.

    Args:
        args: tuple (pdf_path, page_number)

    Returns:
        tuple (page_number, page_type)
    """
    pdf_path, page_number = args

    try:
        page_type = detect_page_type(pdf_path, page_number)
        return (page_number, page_type)
    except Exception:
        return (page_number, "scan")


class DocumentProcessor:
    """
    Main extraction pipeline orchestrator.

    Supports multiple OCR engines configurable via config.OCR_ENGINE.
    """

    def __init__(self, use_gpu: bool = None, ocr_engine: str = None):
        self.use_gpu = use_gpu if use_gpu is not None else config.USE_GPU
        self.ocr_engine_type = ocr_engine or getattr(config, "OCR_ENGINE", "doctr")

        self.ocr_engine = None
        self.tesseract_engine = None

        if self.ocr_engine_type == "tesseract":
            if TESSERACT_AVAILABLE:
                try:
                    self.tesseract_engine = TesseractEngine()
                    logger.info(
                        "Tesseract Engine initialized: lang=%s",
                        self.tesseract_engine.lang,
                    )
                except Exception as e:
                    logger.warning("Error initializing Tesseract, falling back to docTR: %s", e)
                    self._init_doctr_engine()
            else:
                logger.warning("Tesseract not available, falling back to docTR")
                self._init_doctr_engine()
        else:
            self._init_doctr_engine()

    def _init_doctr_engine(self):
        """Initialize docTR engine."""
        try:
            device = "cuda" if self.use_gpu else "cpu"
            self.ocr_engine = DocTREngine(device=device)
            self.ocr_engine_type = "doctr"
        except Exception as e:
            logger.error("Error initializing docTR with GPU: %s", e)
            if self.use_gpu:
                try:
                    self.ocr_engine = DocTREngine(device="cpu")
                except Exception as e2:
                    logger.critical("Critical error initializing OCR: %s", e2)

    def process_document(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
        extract_tables: bool = True,
        show_progress: bool = True,
    ) -> Document:
        """
        Process a complete PDF document.

        Args:
            pdf_path: path to the PDF file
            doc_id: document ID (if None, uses filename)
            extract_tables: attempt to extract tables
            show_progress: (ignored, kept for compat — uses logging)

        Returns:
            Document with all processed pages
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        if doc_id is None:
            doc_id = pdf_path.stem

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        logger.info("Processing: %s (%d pages, tables=%s)",
                     pdf_path.name, total_pages, extract_tables)

        document = Document(
            doc_id=doc_id,
            source_file=pdf_path.name,
            total_pages=total_pages,
            processing_date=datetime.now(),
        )

        for page_num in range(1, total_pages + 1):
            try:
                page = self.process_page(
                    str(pdf_path), page_num, extract_tables=extract_tables
                )
                document.pages.append(page)
                logger.debug("Page %d/%d processed", page_num, total_pages)
            except Exception as e:
                logger.error("Error processing page %d: %s", page_num, e)
                page = Page(page=page_num, source="digital", blocks=[])
                document.pages.append(page)

        total_blocks = sum(len(p.blocks) for p in document.pages)
        total_tables = sum(
            len([b for b in p.blocks if b.type == "table"]) for p in document.pages
        )
        logger.info(
            "Processing completed: %d blocks, %d tables",
            total_blocks,
            total_tables,
        )

        return document

    def process_page(
        self, pdf_path: str, page_number: int, extract_tables: bool = True
    ) -> Page:
        """Process a single page."""
        page_type = detect_page_type(pdf_path, page_number)
        logger.debug("Page %d: type=%s", page_number, page_type)

        if page_type == "digital":
            blocks, width, height = extract_digital_page(pdf_path, page_number)
            source = "digital"

            if extract_tables:
                try:
                    table_blocks = extract_tables_digital(pdf_path, page_number)
                    if table_blocks:
                        blocks = self._remove_overlapping_text_blocks(
                            blocks, table_blocks
                        )
                        blocks.extend(table_blocks)
                        from src.utils.bbox import sort_blocks_by_position

                        blocks = sort_blocks_by_position(blocks)
                except Exception as e:
                    logger.warning(
                        "Error extracting tables from page %d: %s", page_number, e
                    )

        else:
            blocks, width, height = self._extract_with_ocr(pdf_path, page_number)
            source = "ocr"

            if getattr(config, "OCR_POSTPROCESS", True):
                blocks = self._postprocess_ocr_blocks(blocks)

        return Page(
            page=page_number,
            source=source,
            blocks=blocks,
            width=width,
            height=height,
        )

    def _extract_with_ocr(self, pdf_path: str, page_number: int):
        """Extract content using the configured OCR engine."""
        if self.ocr_engine_type == "tesseract" and self.tesseract_engine is not None:
            return extract_ocr_page_tesseract(
                pdf_path, page_number, ocr_engine=self.tesseract_engine
            )
        return extract_ocr_page(
            pdf_path, page_number, preprocess=False, ocr_engine=self.ocr_engine
        )

    def _postprocess_ocr_blocks(self, blocks: list) -> list:
        """Apply post-processing to OCR-extracted blocks."""
        fix_errors = getattr(config, "OCR_FIX_ERRORS", True)
        min_line = getattr(config, "OCR_MIN_LINE_LENGTH", 3)

        processed_blocks = []
        for block in blocks:
            if block.text:
                cleaned_text = postprocess_ocr_text(
                    block.text,
                    clean=True,
                    fix_errors=fix_errors,
                    merge_words=False,
                    min_line_length=min_line,
                )
                if cleaned_text and len(cleaned_text.strip()) >= 2:
                    processed_blocks.append(
                        Block(
                            block_id=block.block_id,
                            type=block.type,
                            text=cleaned_text,
                            bbox=block.bbox,
                            confidence=block.confidence,
                            rows=block.rows,
                        )
                    )
            else:
                processed_blocks.append(block)

        return processed_blocks

    def _remove_overlapping_text_blocks(
        self,
        text_blocks: list[Block],
        table_blocks: list[Block],
        overlap_threshold: float = 0.5,
    ) -> list[Block]:
        """Remove text blocks that significantly overlap with tables."""
        from src.utils.bbox import bbox_area, bbox_overlap

        filtered_blocks = []
        for text_block in text_blocks:
            should_keep = True
            for table_block in table_blocks:
                overlap = bbox_overlap(text_block.bbox, table_block.bbox)
                text_area = bbox_area(text_block.bbox)
                if text_area > 0 and (overlap / text_area) > overlap_threshold:
                    should_keep = False
                    break
            if should_keep:
                filtered_blocks.append(text_block)

        return filtered_blocks

    def save_to_json(
        self,
        document: Document,
        output_path: str,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> None:
        """Save document to JSON file."""
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = document.to_json_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        logger.info(
            "JSON saved to: %s (%.1f KB)",
            output_path,
            output_path.stat().st_size / 1024,
        )

    def save_to_searchable_pdf(
        self, document: Document, pdf_path: str, output_path: str
    ) -> None:
        """Save searchable PDF with invisible text overlaid on OCR pages."""
        from src.exporters.searchable_pdf import create_searchable_pdf

        create_searchable_pdf(pdf_path, document, output_path)

        output_path_obj = Path(output_path)
        logger.info(
            "Searchable PDF saved to: %s (%.1f KB)",
            output_path,
            output_path_obj.stat().st_size / 1024,
        )

    # =========================================================================
    # Parallel processing methods
    # =========================================================================

    def _classify_all_pages(
        self, pdf_path: str, total_pages: int, show_progress: bool = True
    ) -> Dict[int, str]:
        """Classify all document pages before processing."""
        page_types = {}
        min_pages_parallel = 10

        if total_pages < min_pages_parallel:
            for page_num in range(1, total_pages + 1):
                try:
                    page_type = detect_page_type(pdf_path, page_num)
                    page_types[page_num] = page_type
                except Exception:
                    page_types[page_num] = "scan"
        else:
            num_workers = getattr(config, "PARALLEL_WORKERS", None)
            if num_workers is None:
                num_workers = min(multiprocessing.cpu_count(), total_pages, 8)

            worker_args = [
                (pdf_path, page_num) for page_num in range(1, total_pages + 1)
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_classify_page_worker, args): args[1]
                    for args in worker_args
                }

                for future in as_completed(futures):
                    try:
                        page_num, page_type = future.result()
                        page_types[page_num] = page_type
                    except Exception:
                        page_num = futures[future]
                        page_types[page_num] = "scan"

        logger.info("Classification completed: %d pages", total_pages)
        return page_types

    def _process_ocr_batch_doctr(
        self, pdf_path: str, page_numbers: List[int], show_progress: bool = True
    ) -> Dict[int, Page]:
        """Process multiple OCR pages in batch using docTR."""
        from pdf2image import convert_from_path
        import numpy as np

        if not page_numbers:
            return {}

        results = {}
        batch_size = getattr(config, "OCR_BATCH_SIZE", 8)
        dpi = getattr(config, "OCR_DPI", config.IMAGE_DPI)

        batches = [
            page_numbers[i : i + batch_size]
            for i in range(0, len(page_numbers), batch_size)
        ]

        for batch_idx, batch_pages in enumerate(batches, 1):
            logger.info(
                "OCR batch %d/%d (%d pages)", batch_idx, len(batches), len(batch_pages)
            )
            try:
                images = [None] * len(batch_pages)
                page_dimensions = [(0, 0)] * len(batch_pages)
                page_to_idx = {
                    page_num: idx for idx, page_num in enumerate(batch_pages)
                }

                sorted_pages = sorted(batch_pages)
                ranges = []
                start = prev = sorted_pages[0]
                for page_num in sorted_pages[1:]:
                    if page_num == prev + 1:
                        prev = page_num
                    else:
                        ranges.append((start, prev))
                        start = prev = page_num
                ranges.append((start, prev))

                for first_page, last_page in ranges:
                    page_images = convert_from_path(
                        pdf_path,
                        first_page=first_page,
                        last_page=last_page,
                        dpi=dpi,
                    )

                    expected = last_page - first_page + 1
                    for offset in range(expected):
                        page_num = first_page + offset
                        b_idx = page_to_idx.get(page_num)
                        if b_idx is None or offset >= len(page_images):
                            continue

                        img = page_images[offset]
                        page_dimensions[b_idx] = (img.size[0], img.size[1])

                        img_array = np.array(img)
                        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]
                        images[b_idx] = img_array

                    for img in page_images:
                        img.close()
                    del page_images

                valid_indices = [i for i, img in enumerate(images) if img is not None]
                valid_images = [images[i] for i in valid_indices]

                if valid_images and self.ocr_engine is not None:
                    batch_result = self.ocr_engine.extract_from_images_batch(
                        valid_images
                    )

                    for result_idx, b_idx in enumerate(valid_indices):
                        page_num = batch_pages[b_idx]
                        width, height = page_dimensions[b_idx]

                        if (
                            batch_result is not None
                            and result_idx < len(batch_result.pages)
                        ):
                            page_result = batch_result.pages[result_idx]
                            blocks = self._parse_doctr_page_result(
                                page_result, page_num, width, height
                            )

                            if getattr(config, "OCR_POSTPROCESS", True):
                                blocks = self._postprocess_ocr_blocks(blocks)

                            results[page_num] = Page(
                                page=page_num,
                                source="ocr",
                                blocks=blocks,
                                width=width,
                                height=height,
                            )
                        else:
                            results[page_num] = Page(
                                page=page_num, source="ocr", blocks=[]
                            )

                    invalid_indices = set(range(len(batch_pages))) - set(valid_indices)
                    for idx in invalid_indices:
                        page_num = batch_pages[idx]
                        results[page_num] = Page(
                            page=page_num, source="ocr", blocks=[]
                        )
                else:
                    for page_num in batch_pages:
                        results[page_num] = Page(
                            page=page_num, source="ocr", blocks=[]
                        )

            except Exception as e:
                logger.error("Error in OCR batch %d: %s", batch_idx, e)
                for page_num in batch_pages:
                    if page_num not in results:
                        results[page_num] = Page(
                            page=page_num, source="ocr", blocks=[]
                        )

        return results

    def _parse_doctr_page_result(
        self, page_result, page_number: int, page_width: float, page_height: float
    ) -> List[Block]:
        """Parse a docTR page result."""
        from src.utils.text_normalizer import normalize_text

        blocks = []
        block_counter = 1

        for block_data in page_result.blocks:
            block_text = []
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

                line_bbox = line.geometry
                all_line_bboxes.append(
                    [
                        line_bbox[0][0],
                        line_bbox[0][1],
                        line_bbox[1][0],
                        line_bbox[1][1],
                    ]
                )

            if not block_text:
                continue

            text = "\n".join(block_text)
            text = normalize_text(text)

            if not text:
                continue

            if all_line_bboxes:
                bbox = [
                    min(b[0] for b in all_line_bboxes),
                    min(b[1] for b in all_line_bboxes),
                    max(b[2] for b in all_line_bboxes),
                    max(b[3] for b in all_line_bboxes),
                ]
            else:
                bbox = [0.0, 0.0, 1.0, 1.0]

            confidence = total_confidence / word_count if word_count > 0 else 0.0

            if confidence < config.MIN_CONFIDENCE:
                continue

            lines_data = [
                {"text": lt, "bbox": lb}
                for lt, lb in zip(block_text, all_line_bboxes)
            ]

            from src.models.schemas import BlockType

            block = Block(
                block_id=f"p{page_number}_b{block_counter}",
                type=BlockType.PARAGRAPH,
                text=text,
                bbox=bbox,
                confidence=confidence,
                lines=lines_data,
            )

            blocks.append(block)
            block_counter += 1

        from src.utils.bbox import sort_blocks_by_position

        blocks = sort_blocks_by_position(blocks)

        return blocks

    def process_document_parallel(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
        extract_tables: bool = True,
        show_progress: bool = True,
    ) -> Document:
        """
        Process a PDF document with parallelization.

        Strategy:
        1. Classify all pages (digital vs OCR)
        2. Process digital pages in parallel (ProcessPoolExecutor)
        3. Process OCR pages in batch (docTR) or parallel (Tesseract)
        4. Ordered merge of results
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        if doc_id is None:
            doc_id = pdf_path.stem

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        parallel_enabled = getattr(config, "PARALLEL_ENABLED", True)
        min_pages = getattr(config, "PARALLEL_MIN_PAGES", 4)

        if not parallel_enabled or total_pages < min_pages:
            logger.info("Sequential processing (%d pages)", total_pages)
            return self.process_document(
                pdf_path, doc_id, extract_tables, show_progress
            )

        logger.info("Processing (parallel): %s (%d pages)", pdf_path.name, total_pages)

        # Phase 1: Classify all pages
        page_types = self._classify_all_pages(
            str(pdf_path), total_pages, show_progress
        )

        digital_pages = [p for p, t in page_types.items() if t == "digital"]
        ocr_pages = [p for p, t in page_types.items() if t in ("scan", "hybrid")]

        logger.info(
            "Digital pages: %d, OCR pages: %d",
            len(digital_pages),
            len(ocr_pages),
        )

        results: Dict[int, Page] = {}

        # Phase 2a: Process digital pages in parallel
        if digital_pages:
            num_workers = getattr(config, "PARALLEL_WORKERS", None)
            if num_workers is None:
                num_workers = min(
                    multiprocessing.cpu_count(), len(digital_pages), 8
                )

            logger.info(
                "Processing %d digital pages (%d workers)...",
                len(digital_pages),
                num_workers,
            )

            worker_args = [
                (str(pdf_path), page_num, extract_tables)
                for page_num in digital_pages
            ]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_process_digital_page_worker, args): args[1]
                    for args in worker_args
                }

                for future in as_completed(futures):
                    try:
                        page_num, page = future.result()
                        results[page_num] = page
                    except Exception as e:
                        page_num = futures[future]
                        logger.error("Error on digital page %d: %s", page_num, e)
                        results[page_num] = Page(
                            page=page_num, source="digital", blocks=[]
                        )

        # Phase 2b: Process OCR pages
        if ocr_pages:
            if self.ocr_engine_type == "tesseract" and TESSERACT_AVAILABLE:
                num_workers = getattr(config, "PARALLEL_WORKERS", None)
                if num_workers is None:
                    num_workers = min(
                        multiprocessing.cpu_count(), len(ocr_pages), 4
                    )

                logger.info(
                    "Processing %d OCR Tesseract pages (%d workers)...",
                    len(ocr_pages),
                    num_workers,
                )

                lang = getattr(config, "OCR_LANG", "por")
                tess_config = getattr(
                    config, "TESSERACT_CONFIG", "--oem 1 --psm 3"
                )

                worker_args = [
                    (str(pdf_path), page_num, lang, tess_config)
                    for page_num in ocr_pages
                ]

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(_process_tesseract_page_worker, args): args[1]
                        for args in worker_args
                    }

                    for future in as_completed(futures):
                        try:
                            page_num, page = future.result()
                            results[page_num] = page
                        except Exception as e:
                            page_num = futures[future]
                            logger.error("OCR error on page %d: %s", page_num, e)
                            results[page_num] = Page(
                                page=page_num, source="ocr", blocks=[]
                            )
            else:
                logger.info(
                    "Processing %d OCR docTR pages (batch)...", len(ocr_pages)
                )
                ocr_results = self._process_ocr_batch_doctr(
                    str(pdf_path), ocr_pages, show_progress
                )
                results.update(ocr_results)

        # Phase 3: Build ordered document
        document = Document(
            doc_id=doc_id,
            source_file=pdf_path.name,
            total_pages=total_pages,
            processing_date=datetime.now(),
        )

        for page_num in range(1, total_pages + 1):
            if page_num in results:
                document.pages.append(results[page_num])
            else:
                document.pages.append(
                    Page(page=page_num, source="digital", blocks=[])
                )

        total_blocks = sum(len(p.blocks) for p in document.pages)
        total_tables = sum(
            len([b for b in p.blocks if b.type == "table"]) for p in document.pages
        )
        logger.info(
            "Parallel processing completed: %d blocks, %d tables",
            total_blocks,
            total_tables,
        )

        return document


def process_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    extract_tables: bool = True,
    use_gpu: bool = None,
    parallel: bool = None,
    save_pdf: bool = None,
) -> Document:
    """Helper function to process a PDF and save the result."""
    import gc

    processor = DocumentProcessor(use_gpu=use_gpu)

    try:
        use_parallel = (
            parallel
            if parallel is not None
            else getattr(config, "PARALLEL_ENABLED", True)
        )

        if use_parallel:
            document = processor.process_document_parallel(
                pdf_path, extract_tables=extract_tables, show_progress=True
            )
        else:
            document = processor.process_document(
                pdf_path, extract_tables=extract_tables, show_progress=True
            )

        if output_dir is None:
            output_dir = config.OUTPUT_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{document.doc_id}.json"
        processor.save_to_json(document, str(output_path))

        should_save_pdf = (
            save_pdf
            if save_pdf is not None
            else getattr(config, "SEARCHABLE_PDF", False)
        )
        if should_save_pdf:
            pdf_output_path = output_dir / f"{document.doc_id}_searchable.pdf"
            processor.save_to_searchable_pdf(
                document, pdf_path, str(pdf_output_path)
            )

        return document

    finally:
        processor.ocr_engine = None
        processor.tesseract_engine = None
        gc.collect()
