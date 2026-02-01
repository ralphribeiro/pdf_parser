#!/usr/bin/env python3
"""
Script para processar um único PDF
Uso: python scripts/process_single.py <caminho_do_pdf>
"""
import sys
import gc
import argparse
from pathlib import Path

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.pipeline import process_pdf


def main():
    parser = argparse.ArgumentParser(
        description='Processa um PDF e extrai conteúdo em formato JSON'
    )
    
    parser.add_argument(
        'pdf_path',
        help='Caminho para o arquivo PDF'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help=f'Diretório de saída (padrão: {config.OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--no-tables',
        action='store_true',
        help='Não extrair tabelas'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Não usar GPU (forçar CPU)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso (sem logs)'
    )
    
    args = parser.parse_args()
    
    # Configura verbosidade
    if args.quiet:
        config.VERBOSE = False
    
    # Verifica se arquivo existe
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Erro: arquivo não encontrado: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == '.pdf':
        print(f"Erro: arquivo não é um PDF: {pdf_path}")
        sys.exit(1)
    
    # Processa PDF
    try:
        document = process_pdf(
            str(pdf_path),
            output_dir=args.output,
            extract_tables=not args.no_tables,
            use_gpu=not args.no_gpu
        )
        
        print(f"\n✅ Processamento concluído com sucesso!")
        print(f"   - Documento: {document.doc_id}")
        print(f"   - Páginas: {document.total_pages}")
        print(f"   - Blocos totais: {sum(len(p.blocks) for p in document.pages)}")
        
        # Limpa referências para evitar warning do Poppler/pdf2image
        del document
        gc.collect()
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Erro ao processar PDF: {e}")
        
        if config.VERBOSE:
            import traceback
            traceback.print_exc()
        
        return 1
    
    finally:
        # Força garbage collection para limpar objetos do Poppler
        gc.collect()


if __name__ == '__main__':
    sys.exit(main())
