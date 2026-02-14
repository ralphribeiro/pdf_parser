#!/usr/bin/env python3
"""
Script para processar um único PDF
Uso: python scripts/process_single.py <caminho_do_pdf>
"""
import gc
import logging
import sys
import argparse
from pathlib import Path

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.pipeline import process_pdf

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """
    Constrói o parser de argumentos do CLI.

    Separado de main() para permitir testes unitários do argparse.
    """
    parser = argparse.ArgumentParser(
        description="Processa um PDF e extrai conteúdo em formato JSON"
    )

    parser.add_argument("pdf_path", help="Caminho para o arquivo PDF")

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"Diretório de saída (padrão: {config.OUTPUT_DIR})",
    )

    parser.add_argument(
        "--no-tables", action="store_true", help="Não extrair tabelas"
    )

    parser.add_argument(
        "--no-gpu", action="store_true", help="Não usar GPU (forçar CPU)"
    )

    parser.add_argument(
        "--quiet", action="store_true", help="Modo silencioso (sem logs)"
    )

    # Flag para gerar PDF pesquisável
    pdf_group = parser.add_mutually_exclusive_group()
    pdf_group.add_argument(
        "--pdf",
        action="store_true",
        default=None,
        dest="pdf",
        help="Gerar PDF pesquisável (com texto invisível)",
    )
    pdf_group.add_argument(
        "--no-pdf",
        action="store_false",
        dest="pdf",
        help="Não gerar PDF pesquisável",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Configura verbosidade
    if args.quiet:
        config.VERBOSE = False

    # Configura logging
    config.setup_logging()

    # Verifica se arquivo existe
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error("Arquivo não encontrado: %s", pdf_path)
        sys.exit(1)

    if not pdf_path.suffix.lower() == ".pdf":
        logger.error("Arquivo não é um PDF: %s", pdf_path)
        sys.exit(1)

    # Processa PDF
    try:
        document = process_pdf(
            str(pdf_path),
            output_dir=args.output,
            extract_tables=not args.no_tables,
            use_gpu=not args.no_gpu,
            save_pdf=args.pdf,
        )

        logger.info(
            "Processamento concluído: doc=%s, páginas=%d, blocos=%d",
            document.doc_id,
            document.total_pages,
            sum(len(p.blocks) for p in document.pages),
        )

        # Limpa referências para evitar warning do Poppler/pdf2image
        del document
        gc.collect()

        return 0

    except Exception as e:
        logger.error("Erro ao processar PDF: %s", e, exc_info=config.VERBOSE)
        return 1

    finally:
        gc.collect()


if __name__ == "__main__":
    sys.exit(main())
