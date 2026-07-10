#!/usr/bin/env python3
"""
Script to process a single supported document.
Usage: python scripts/process_single.py <source_path>
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.formats.registry import SUPPORTED_FORMAT_LABEL, detect_format
from src.pipeline import process_file

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Separated from main() to allow unit testing of argparse.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Process a supported document and extract content in JSON format "
            f"({SUPPORTED_FORMAT_LABEL})"
        )
    )

    parser.add_argument("source_path", help="Path to the supported document file")

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"Output directory (default: {config.OUTPUT_DIR})",
    )

    parser.add_argument(
        "--no-tables", action="store_true", help="Do not extract tables"
    )

    parser.add_argument(
        "--no-gpu", action="store_true", help="Do not use GPU (force CPU)"
    )

    parser.add_argument("--quiet", action="store_true", help="Quiet mode (no logs)")

    # Flag to generate searchable PDF
    pdf_group = parser.add_mutually_exclusive_group()
    pdf_group.add_argument(
        "--pdf",
        action="store_true",
        default=None,
        dest="pdf",
        help="Generate searchable PDF (with invisible text)",
    )
    pdf_group.add_argument(
        "--no-pdf",
        action="store_false",
        dest="pdf",
        help="Do not generate searchable PDF",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Configure verbosity
    if args.quiet:
        config.VERBOSE = False

    # Configure logging
    config.setup_logging()

    source_path = Path(args.source_path)
    if not source_path.exists():
        logger.error("File not found: %s", source_path)
        sys.exit(1)

    try:
        detect_format(source_path)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Process document
    try:
        document = process_file(
            str(source_path),
            output_dir=args.output,
            extract_tables=not args.no_tables,
            use_gpu=not args.no_gpu,
            save_pdf=args.pdf,
        )

        logger.info(
            "Processing completed: doc=%s, pages=%d, blocks=%d",
            document.doc_id,
            document.total_pages,
            sum(len(p.blocks) for p in document.pages),
        )

        # Clean references to avoid Poppler/pdf2image warnings
        del document
        gc.collect()

        return 0

    except Exception as e:
        logger.error("Error processing document: %s", e, exc_info=config.VERBOSE)
        return 1

    finally:
        gc.collect()


if __name__ == "__main__":
    sys.exit(main())
