"""
Pipeline principal para processamento de PDFs
"""
import pdfplumber
from pathlib import Path
from typing import Optional
from datetime import datetime
from tqdm import tqdm

import config
from src.models.schemas import Document, Page, Block
from src.detector import detect_page_type, get_page_dimensions
from src.extractors.digital import extract_digital_page
from src.extractors.ocr import extract_ocr_page, OCREngine
from src.extractors.tables import extract_tables_digital


class DocumentProcessor:
    """
    Orquestrador principal do pipeline de extração
    """
    
    def __init__(self, use_gpu: bool = None):
        """
        Inicializa o processador
        
        Args:
            use_gpu: usar GPU para OCR (se None, usa config.USE_GPU)
        """
        self.use_gpu = use_gpu if use_gpu is not None else config.USE_GPU
        
        # Inicializa engine OCR (carrega modelo na GPU)
        self.ocr_engine = None
        if self.use_gpu:
            try:
                self.ocr_engine = OCREngine()
                if config.VERBOSE:
                    print(f"OCR Engine inicializado com GPU: {config.DEVICE}")
            except Exception as e:
                if config.VERBOSE:
                    print(f"Erro ao inicializar GPU, usando CPU: {e}")
                self.ocr_engine = OCREngine(device='cpu')
    
    def process_document(self, pdf_path: str, doc_id: Optional[str] = None,
                        extract_tables: bool = True,
                        show_progress: bool = True) -> Document:
        """
        Processa um documento PDF completo
        
        Args:
            pdf_path: caminho para o arquivo PDF
            doc_id: ID do documento (se None, usa nome do arquivo)
            extract_tables: tentar extrair tabelas
            show_progress: mostrar barra de progresso
        
        Returns:
            Document com todas as páginas processadas
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
        
        # Gera doc_id se não fornecido
        if doc_id is None:
            doc_id = pdf_path.stem
        
        # Obtém número total de páginas
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        
        if config.VERBOSE:
            print(f"\nProcessando: {pdf_path.name}")
            print(f"Total de páginas: {total_pages}")
            print(f"Extração de tabelas: {'Sim' if extract_tables else 'Não'}")
        
        # Cria documento
        document = Document(
            doc_id=doc_id,
            source_file=pdf_path.name,
            total_pages=total_pages,
            processing_date=datetime.now()
        )
        
        # Processa cada página
        iterator = range(1, total_pages + 1)
        if show_progress:
            iterator = tqdm(iterator, desc="Processando páginas", unit="pág")
        
        for page_num in iterator:
            try:
                page = self.process_page(
                    str(pdf_path),
                    page_num,
                    extract_tables=extract_tables
                )
                document.pages.append(page)
            except Exception as e:
                if config.VERBOSE:
                    print(f"\nErro ao processar página {page_num}: {e}")
                # Adiciona página vazia em caso de erro
                page = Page(
                    page=page_num,
                    source="digital",
                    blocks=[]
                )
                document.pages.append(page)
        
        if config.VERBOSE:
            total_blocks = sum(len(p.blocks) for p in document.pages)
            total_tables = sum(
                len([b for b in p.blocks if b.type == "table"])
                for p in document.pages
            )
            print(f"\n✓ Processamento concluído!")
            print(f"  - Total de blocos: {total_blocks}")
            print(f"  - Total de tabelas: {total_tables}")
        
        return document
    
    def process_page(self, pdf_path: str, page_number: int,
                    extract_tables: bool = True) -> Page:
        """
        Processa uma única página
        
        Args:
            pdf_path: caminho para o PDF
            page_number: número da página (1-indexed)
            extract_tables: tentar extrair tabelas
        
        Returns:
            Page com blocos extraídos
        """
        # Detecta tipo da página (digital, scan, ou hybrid)
        page_type = detect_page_type(pdf_path, page_number)
        
        if config.VERBOSE and config.VERBOSE == "debug":
            print(f"\nPágina {page_number}: {page_type}")
        
        # Extrai conteúdo baseado no tipo
        # "digital" -> extração direta de texto
        # "scan" ou "hybrid" -> OCR (hybrid = imagem com overlay de texto, ex: carimbo TJSP)
        if page_type == "digital":
            blocks, width, height = extract_digital_page(pdf_path, page_number)
            source = "digital"
            
            # Tenta extrair tabelas se solicitado
            if extract_tables:
                try:
                    table_blocks = extract_tables_digital(pdf_path, page_number)
                    
                    # Remove blocos de texto que se sobrepõem com tabelas
                    if table_blocks:
                        blocks = self._remove_overlapping_text_blocks(
                            blocks, table_blocks
                        )
                        blocks.extend(table_blocks)
                        
                        # Reordena todos os blocos
                        from src.utils.bbox import sort_blocks_by_position
                        blocks = sort_blocks_by_position(blocks)
                except Exception as e:
                    if config.VERBOSE:
                        print(f"Aviso: erro ao extrair tabelas da página {page_number}: {e}")
        
        else:  # scan ou hybrid -> usar OCR
            blocks, width, height = extract_ocr_page(
                pdf_path,
                page_number,
                preprocess=True,
                ocr_engine=self.ocr_engine
            )
            source = "ocr"
        
        # Cria objeto Page
        page = Page(
            page=page_number,
            source=source,
            blocks=blocks,
            width=width,
            height=height
        )
        
        return page
    
    def _remove_overlapping_text_blocks(self, text_blocks: list[Block],
                                       table_blocks: list[Block],
                                       overlap_threshold: float = 0.5) -> list[Block]:
        """
        Remove blocos de texto que se sobrepõem significativamente com tabelas
        
        Args:
            text_blocks: blocos de texto
            table_blocks: blocos de tabela
            overlap_threshold: % mínima de overlap para remover bloco
        
        Returns:
            blocos de texto filtrados
        """
        from src.utils.bbox import bbox_overlap, bbox_area
        
        filtered_blocks = []
        
        for text_block in text_blocks:
            # Verifica overlap com cada tabela
            should_keep = True
            
            for table_block in table_blocks:
                overlap = bbox_overlap(text_block.bbox, table_block.bbox)
                text_area = bbox_area(text_block.bbox)
                
                if text_area > 0:
                    overlap_ratio = overlap / text_area
                    
                    if overlap_ratio > overlap_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                filtered_blocks.append(text_block)
        
        return filtered_blocks
    
    def save_to_json(self, document: Document, output_path: str,
                    indent: int = 2, ensure_ascii: bool = False) -> None:
        """
        Salva documento em arquivo JSON
        
        Args:
            document: documento processado
            output_path: caminho do arquivo de saída
            indent: indentação do JSON
            ensure_ascii: se True, escapa caracteres não-ASCII
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Converte para dict
        data = document.to_json_dict()
        
        # Salva JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        if config.VERBOSE:
            print(f"\n✓ JSON salvo em: {output_path}")
            print(f"  Tamanho: {output_path.stat().st_size / 1024:.1f} KB")


def process_pdf(pdf_path: str, output_dir: Optional[str] = None,
               extract_tables: bool = True,
               use_gpu: bool = None) -> Document:
    """
    Função auxiliar para processar um PDF e salvar o resultado
    
    Args:
        pdf_path: caminho para o PDF
        output_dir: diretório de saída (se None, usa config.OUTPUT_DIR)
        extract_tables: extrair tabelas
        use_gpu: usar GPU
    
    Returns:
        Document processado
    """
    processor = DocumentProcessor(use_gpu=use_gpu)
    
    # Processa documento
    document = processor.process_document(
        pdf_path,
        extract_tables=extract_tables,
        show_progress=True
    )
    
    # Salva JSON
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{document.doc_id}.json"
    processor.save_to_json(document, str(output_path))
    
    return document
