"""
Pipeline principal para processamento de PDFs

Suporta múltiplos engines de OCR:
- doctr: Deep learning (PyTorch), rápido com GPU
- tesseract: LSTM tradicional, bom para português

Suporta processamento paralelo:
- Páginas digitais: ProcessPoolExecutor (CPU-bound)
- Páginas OCR com docTR: Batch processing (GPU)
- Páginas OCR com Tesseract: ProcessPoolExecutor (CPU)
"""
import pdfplumber
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

import config
from src.models.schemas import Document, Page, Block
from src.detector import detect_page_type, get_page_dimensions
from src.extractors.digital import extract_digital_page
from src.extractors.ocr import extract_ocr_page, OCREngine, DocTREngine
from src.extractors.tables import extract_tables_digital
from src.utils.ocr_postprocess import postprocess_ocr_text

# Importação condicional do Tesseract
try:
    from src.extractors.ocr_tesseract import extract_ocr_page_tesseract, TesseractEngine
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    TesseractEngine = None
    extract_ocr_page_tesseract = None


# =============================================================================
# Funções auxiliares para processamento paralelo (devem ser top-level para pickle)
# =============================================================================

def _process_digital_page_worker(args: Tuple[str, int, bool]) -> Tuple[int, Page]:
    """
    Worker para processar uma página digital em paralelo.
    
    Args:
        args: tupla (pdf_path, page_number, extract_tables)
    
    Returns:
        tupla (page_number, Page)
    """
    pdf_path, page_number, extract_tables = args
    
    try:
        blocks, width, height = extract_digital_page(pdf_path, page_number)
        
        # Extrai tabelas se solicitado
        if extract_tables:
            try:
                table_blocks = extract_tables_digital(pdf_path, page_number)
                if table_blocks:
                    # Remove blocos sobrepostos
                    from src.utils.bbox import bbox_overlap, bbox_area, sort_blocks_by_position
                    
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
                pass  # Ignora erros de tabela silenciosamente
        
        page = Page(
            page=page_number,
            source="digital",
            blocks=blocks,
            width=width,
            height=height
        )
        return (page_number, page)
    
    except Exception as e:
        # Retorna página vazia em caso de erro
        return (page_number, Page(page=page_number, source="digital", blocks=[]))


def _process_tesseract_page_worker(args: Tuple[str, int, str, str]) -> Tuple[int, Page]:
    """
    Worker para processar uma página com Tesseract em paralelo.
    
    Args:
        args: tupla (pdf_path, page_number, lang, tesseract_config)
    
    Returns:
        tupla (page_number, Page)
    """
    pdf_path, page_number, lang, tesseract_config = args
    
    try:
        # Inicializa engine Tesseract no worker (cada processo precisa do seu)
        from src.extractors.ocr_tesseract import TesseractEngine, extract_ocr_page_tesseract
        
        engine = TesseractEngine(lang=lang, tesseract_config=tesseract_config)
        blocks, width, height = extract_ocr_page_tesseract(
            pdf_path, page_number, ocr_engine=engine
        )
        
        # Aplica pós-processamento
        if getattr(config, 'OCR_POSTPROCESS', True):
            blocks = _postprocess_blocks(blocks)
        
        page = Page(
            page=page_number,
            source="ocr",
            blocks=blocks,
            width=width,
            height=height
        )
        return (page_number, page)
    
    except Exception as e:
        return (page_number, Page(page=page_number, source="ocr", blocks=[]))


def _postprocess_blocks(blocks: list) -> list:
    """
    Aplica pós-processamento em blocos OCR (versão standalone para workers).
    """
    fix_errors = getattr(config, 'OCR_FIX_ERRORS', True)
    min_line = getattr(config, 'OCR_MIN_LINE_LENGTH', 3)
    
    processed_blocks = []
    for block in blocks:
        if block.text:
            cleaned_text = postprocess_ocr_text(
                block.text,
                clean=True,
                fix_errors=fix_errors,
                merge_words=False,
                min_line_length=min_line
            )
            
            if cleaned_text and len(cleaned_text.strip()) >= 2:
                processed_blocks.append(Block(
                    block_id=block.block_id,
                    type=block.type,
                    text=cleaned_text,
                    bbox=block.bbox,
                    confidence=block.confidence,
                    rows=block.rows
                ))
        else:
            processed_blocks.append(block)
    
    return processed_blocks


def _classify_page_worker(args: Tuple[str, int]) -> Tuple[int, str]:
    """
    Worker para classificar o tipo de uma página em paralelo.
    
    Args:
        args: tupla (pdf_path, page_number)
    
    Returns:
        tupla (page_number, page_type)
    """
    pdf_path, page_number = args
    
    try:
        page_type = detect_page_type(pdf_path, page_number)
        return (page_number, page_type)
    except Exception:
        # Em caso de erro, assume scan (mais seguro, usa OCR)
        return (page_number, "scan")


class DocumentProcessor:
    """
    Orquestrador principal do pipeline de extração
    
    Suporta múltiplos engines de OCR configuráveis via config.OCR_ENGINE
    """
    
    def __init__(self, use_gpu: bool = None, ocr_engine: str = None):
        """
        Inicializa o processador
        
        Args:
            use_gpu: usar GPU para OCR (se None, usa config.USE_GPU)
            ocr_engine: 'doctr' ou 'tesseract' (se None, usa config.OCR_ENGINE)
        """
        self.use_gpu = use_gpu if use_gpu is not None else config.USE_GPU
        self.ocr_engine_type = ocr_engine or getattr(config, 'OCR_ENGINE', 'doctr')
        
        # Inicializa engine OCR baseado na configuração
        self.ocr_engine = None
        self.tesseract_engine = None
        
        if self.ocr_engine_type == 'tesseract':
            if TESSERACT_AVAILABLE:
                try:
                    self.tesseract_engine = TesseractEngine()
                    if config.VERBOSE:
                        print(f"Tesseract Engine inicializado: lang={self.tesseract_engine.lang}")
                except Exception as e:
                    if config.VERBOSE:
                        print(f"Erro ao inicializar Tesseract, usando docTR: {e}")
                    self._init_doctr_engine()
            else:
                if config.VERBOSE:
                    print("Tesseract não disponível, usando docTR")
                self._init_doctr_engine()
        else:
            self._init_doctr_engine()
    
    def _init_doctr_engine(self):
        """Inicializa engine docTR"""
        try:
            device = 'cuda' if self.use_gpu else 'cpu'
            self.ocr_engine = DocTREngine(device=device)
            self.ocr_engine_type = 'doctr'
        except Exception as e:
            if config.VERBOSE:
                print(f"Erro ao inicializar docTR com GPU: {e}")
            # Tenta CPU se GPU falhou
            if self.use_gpu:
                try:
                    self.ocr_engine = DocTREngine(device='cpu')
                except Exception as e2:
                    if config.VERBOSE:
                        print(f"Erro crítico ao inicializar OCR: {e2}")
    
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
            blocks, width, height = self._extract_with_ocr(pdf_path, page_number)
            source = "ocr"
            
            # Aplica pós-processamento se configurado
            if getattr(config, 'OCR_POSTPROCESS', True):
                blocks = self._postprocess_ocr_blocks(blocks)
        
        # Cria objeto Page
        page = Page(
            page=page_number,
            source=source,
            blocks=blocks,
            width=width,
            height=height
        )
        
        return page
    
    def _extract_with_ocr(self, pdf_path: str, page_number: int):
        """
        Extrai conteúdo usando o engine OCR configurado
        """
        if self.ocr_engine_type == 'tesseract' and self.tesseract_engine is not None:
            return extract_ocr_page_tesseract(
                pdf_path,
                page_number,
                ocr_engine=self.tesseract_engine
            )
        else:
            return extract_ocr_page(
                pdf_path,
                page_number,
                preprocess=False,  # NÃO usar pré-processamento - degrada qualidade
                ocr_engine=self.ocr_engine
            )
    
    def _postprocess_ocr_blocks(self, blocks: list) -> list:
        """
        Aplica pós-processamento em blocos extraídos por OCR
        """
        fix_errors = getattr(config, 'OCR_FIX_ERRORS', True)
        min_line = getattr(config, 'OCR_MIN_LINE_LENGTH', 3)
        
        processed_blocks = []
        for block in blocks:
            if block.text:
                # Aplica pós-processamento no texto
                cleaned_text = postprocess_ocr_text(
                    block.text,
                    clean=True,
                    fix_errors=fix_errors,
                    merge_words=False,  # Pode causar problemas, desativado
                    min_line_length=min_line
                )
                
                # Só mantém bloco se ainda tiver texto após limpeza
                if cleaned_text and len(cleaned_text.strip()) >= 2:
                    # Cria novo bloco com texto limpo
                    processed_blocks.append(Block(
                        block_id=block.block_id,
                        type=block.type,
                        text=cleaned_text,
                        bbox=block.bbox,
                        confidence=block.confidence,
                        rows=block.rows
                    ))
            else:
                # Blocos sem texto (tabelas) passam direto
                processed_blocks.append(block)
        
        return processed_blocks
    
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
    
    # =========================================================================
    # Métodos para processamento paralelo
    # =========================================================================
    
    def _classify_all_pages(self, pdf_path: str, total_pages: int,
                             show_progress: bool = True) -> Dict[int, str]:
        """
        Classifica todas as páginas do documento antes do processamento.
        
        Usa ProcessPoolExecutor para paralelizar a classificação em documentos
        grandes (>= 10 páginas). Para documentos menores, o overhead de
        paralelização não compensa.
        
        Args:
            pdf_path: caminho para o PDF
            total_pages: número total de páginas
            show_progress: mostrar barra de progresso
        
        Returns:
            dict mapeando page_number -> tipo ('digital', 'scan', 'hybrid')
        """
        page_types = {}
        
        # Threshold: só paraleliza se tiver páginas suficientes
        min_pages_parallel = 10
        
        if total_pages < min_pages_parallel:
            # Processamento sequencial (overhead de paralelização não compensa)
            iterator = range(1, total_pages + 1)
            if show_progress:
                iterator = tqdm(iterator, desc="Classificando", unit="pág")
            
            for page_num in iterator:
                try:
                    page_type = detect_page_type(pdf_path, page_num)
                    page_types[page_num] = page_type
                except Exception:
                    page_types[page_num] = "scan"
        else:
            # Processamento paralelo
            num_workers = getattr(config, 'PARALLEL_WORKERS', None)
            if num_workers is None:
                num_workers = min(multiprocessing.cpu_count(), total_pages, 8)
            
            worker_args = [(pdf_path, page_num) for page_num in range(1, total_pages + 1)]
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_classify_page_worker, args): args[1]
                    for args in worker_args
                }
                
                if show_progress:
                    pbar = tqdm(total=total_pages, desc="Classificando", unit="pág")
                
                for future in as_completed(futures):
                    try:
                        page_num, page_type = future.result()
                        page_types[page_num] = page_type
                    except Exception:
                        page_num = futures[future]
                        page_types[page_num] = "scan"
                    
                    if show_progress:
                        pbar.update(1)
                
                if show_progress:
                    pbar.close()
        
        return page_types
    
    def _process_ocr_batch_doctr(self, pdf_path: str, page_numbers: List[int],
                                  show_progress: bool = True) -> Dict[int, Page]:
        """
        Processa múltiplas páginas OCR em batch usando docTR.
        
        O docTR aceita lista de imagens e processa em batch, maximizando
        uso da GPU e reduzindo overhead de transferência CPU<->GPU.
        
        Args:
            pdf_path: caminho para o PDF
            page_numbers: lista de números de páginas para processar
            show_progress: mostrar barra de progresso
        
        Returns:
            dict mapeando page_number -> Page
        """
        from pdf2image import convert_from_path
        import numpy as np
        
        if not page_numbers:
            return {}
        
        results = {}
        batch_size = getattr(config, 'OCR_BATCH_SIZE', 8)
        dpi = getattr(config, 'OCR_DPI', config.IMAGE_DPI)
        
        # Processa em batches
        batches = [page_numbers[i:i + batch_size] 
                   for i in range(0, len(page_numbers), batch_size)]
        
        iterator = batches
        if show_progress:
            iterator = tqdm(batches, desc="OCR (batches)", unit="batch")
        
        for batch_pages in iterator:
            try:
                # Converte páginas do batch para imagens
                images = []
                page_dimensions = []
                
                for page_num in batch_pages:
                    page_images = convert_from_path(
                        pdf_path,
                        first_page=page_num,
                        last_page=page_num,
                        dpi=dpi
                    )
                    if page_images:
                        img = page_images[0]
                        page_dimensions.append((img.size[0], img.size[1]))
                        
                        # Converte para numpy array RGB
                        img_array = np.array(img)
                        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]  # Remove alpha
                        images.append(img_array)
                        
                        img.close()
                        del page_images
                    else:
                        images.append(None)
                        page_dimensions.append((0, 0))
                
                # Filtra imagens válidas para batch OCR
                valid_images = [img for img in images if img is not None]
                
                if valid_images and self.ocr_engine is not None:
                    # Processa batch com docTR
                    batch_result = self.ocr_engine.model(valid_images)
                    
                    # Parseia resultados
                    valid_idx = 0
                    for i, page_num in enumerate(batch_pages):
                        if images[i] is not None:
                            width, height = page_dimensions[i]
                            
                            # Extrai resultado da página específica
                            if valid_idx < len(batch_result.pages):
                                page_result = batch_result.pages[valid_idx]
                                blocks = self._parse_doctr_page_result(
                                    page_result, page_num, width, height
                                )
                                
                                # Aplica pós-processamento
                                if getattr(config, 'OCR_POSTPROCESS', True):
                                    blocks = self._postprocess_ocr_blocks(blocks)
                                
                                results[page_num] = Page(
                                    page=page_num,
                                    source="ocr",
                                    blocks=blocks,
                                    width=width,
                                    height=height
                                )
                            else:
                                results[page_num] = Page(
                                    page=page_num, source="ocr", blocks=[]
                                )
                            
                            valid_idx += 1
                        else:
                            results[page_num] = Page(
                                page=page_num, source="ocr", blocks=[]
                            )
                else:
                    # Fallback: páginas vazias
                    for page_num in batch_pages:
                        results[page_num] = Page(
                            page=page_num, source="ocr", blocks=[]
                        )
                
            except Exception as e:
                if config.VERBOSE:
                    print(f"\nErro no batch OCR: {e}")
                # Marca páginas do batch como vazias
                for page_num in batch_pages:
                    if page_num not in results:
                        results[page_num] = Page(
                            page=page_num, source="ocr", blocks=[]
                        )
        
        return results
    
    def _parse_doctr_page_result(self, page_result, page_number: int,
                                  page_width: float, page_height: float) -> List[Block]:
        """
        Parseia resultado de uma página do docTR.
        
        Similar a _parse_doctr_result em ocr.py, mas para uma única página.
        """
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
                all_line_bboxes.append([
                    line_bbox[0][0], line_bbox[0][1],
                    line_bbox[1][0], line_bbox[1][1]
                ])
            
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
                    max(b[3] for b in all_line_bboxes)
                ]
            else:
                bbox = [0.0, 0.0, 1.0, 1.0]
            
            confidence = total_confidence / word_count if word_count > 0 else 0.0
            
            if confidence < config.MIN_CONFIDENCE:
                continue
            
            from src.models.schemas import BlockType
            block = Block(
                block_id=f"p{page_number}_b{block_counter}",
                type=BlockType.PARAGRAPH,
                text=text,
                bbox=bbox,
                confidence=confidence
            )
            
            blocks.append(block)
            block_counter += 1
        
        # Ordena blocos por posição
        from src.utils.bbox import sort_blocks_by_position
        blocks = sort_blocks_by_position(blocks)
        
        return blocks
    
    def process_document_parallel(self, pdf_path: str, doc_id: Optional[str] = None,
                                   extract_tables: bool = True,
                                   show_progress: bool = True) -> Document:
        """
        Processa um documento PDF com paralelização.
        
        Estratégia:
        1. Classifica todas as páginas (digital vs OCR)
        2. Processa páginas digitais em paralelo (ProcessPoolExecutor)
        3. Processa páginas OCR em batch (docTR) ou paralelo (Tesseract)
        4. Merge ordenado dos resultados
        
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
        
        if doc_id is None:
            doc_id = pdf_path.stem
        
        # Obtém número total de páginas
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
        
        # Verifica se vale a pena paralelizar
        parallel_enabled = getattr(config, 'PARALLEL_ENABLED', True)
        min_pages = getattr(config, 'PARALLEL_MIN_PAGES', 4)
        
        if not parallel_enabled or total_pages < min_pages:
            # Usa processamento sequencial
            if config.VERBOSE:
                print(f"Usando processamento sequencial ({total_pages} páginas)")
            return self.process_document(pdf_path, doc_id, extract_tables, show_progress)
        
        if config.VERBOSE:
            print(f"\nProcessando (paralelo): {pdf_path.name}")
            print(f"Total de páginas: {total_pages}")
        
        # Fase 1: Classificar todas as páginas
        page_types = self._classify_all_pages(str(pdf_path), total_pages, show_progress)
        
        digital_pages = [p for p, t in page_types.items() if t == "digital"]
        ocr_pages = [p for p, t in page_types.items() if t in ("scan", "hybrid")]
        
        if config.VERBOSE:
            print(f"  - Páginas digitais: {len(digital_pages)}")
            print(f"  - Páginas OCR: {len(ocr_pages)}")
        
        # Dicionário para armazenar resultados
        results: Dict[int, Page] = {}
        
        # Fase 2a: Processa páginas digitais em paralelo
        if digital_pages:
            num_workers = getattr(config, 'PARALLEL_WORKERS', None)
            if num_workers is None:
                num_workers = min(multiprocessing.cpu_count(), len(digital_pages), 8)
            
            if config.VERBOSE:
                print(f"Processando {len(digital_pages)} páginas digitais ({num_workers} workers)...")
            
            # Prepara argumentos para workers
            worker_args = [
                (str(pdf_path), page_num, extract_tables)
                for page_num in digital_pages
            ]
            
            # Processa em paralelo
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_process_digital_page_worker, args): args[1]
                    for args in worker_args
                }
                
                # Coleta resultados com progress bar
                if show_progress:
                    pbar = tqdm(total=len(digital_pages), desc="Digital", unit="pág")
                
                for future in as_completed(futures):
                    try:
                        page_num, page = future.result()
                        results[page_num] = page
                    except Exception as e:
                        page_num = futures[future]
                        if config.VERBOSE:
                            print(f"\nErro página digital {page_num}: {e}")
                        results[page_num] = Page(
                            page=page_num, source="digital", blocks=[]
                        )
                    
                    if show_progress:
                        pbar.update(1)
                
                if show_progress:
                    pbar.close()
        
        # Fase 2b: Processa páginas OCR
        if ocr_pages:
            if self.ocr_engine_type == 'tesseract' and TESSERACT_AVAILABLE:
                # Tesseract: processa em paralelo (cada worker tem seu engine)
                num_workers = getattr(config, 'PARALLEL_WORKERS', None)
                if num_workers is None:
                    num_workers = min(multiprocessing.cpu_count(), len(ocr_pages), 4)
                
                if config.VERBOSE:
                    print(f"Processando {len(ocr_pages)} páginas OCR Tesseract ({num_workers} workers)...")
                
                lang = getattr(config, 'OCR_LANG', 'por')
                tess_config = getattr(config, 'TESSERACT_CONFIG', '--oem 1 --psm 3')
                
                worker_args = [
                    (str(pdf_path), page_num, lang, tess_config)
                    for page_num in ocr_pages
                ]
                
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(_process_tesseract_page_worker, args): args[1]
                        for args in worker_args
                    }
                    
                    if show_progress:
                        pbar = tqdm(total=len(ocr_pages), desc="OCR Tesseract", unit="pág")
                    
                    for future in as_completed(futures):
                        try:
                            page_num, page = future.result()
                            results[page_num] = page
                        except Exception as e:
                            page_num = futures[future]
                            if config.VERBOSE:
                                print(f"\nErro OCR página {page_num}: {e}")
                            results[page_num] = Page(
                                page=page_num, source="ocr", blocks=[]
                            )
                        
                        if show_progress:
                            pbar.update(1)
                    
                    if show_progress:
                        pbar.close()
            else:
                # docTR: processa em batch (mais eficiente para GPU)
                if config.VERBOSE:
                    print(f"Processando {len(ocr_pages)} páginas OCR docTR (batch)...")
                
                ocr_results = self._process_ocr_batch_doctr(
                    str(pdf_path), ocr_pages, show_progress
                )
                results.update(ocr_results)
        
        # Fase 3: Monta documento ordenado
        document = Document(
            doc_id=doc_id,
            source_file=pdf_path.name,
            total_pages=total_pages,
            processing_date=datetime.now()
        )
        
        # Ordena páginas por número
        for page_num in range(1, total_pages + 1):
            if page_num in results:
                document.pages.append(results[page_num])
            else:
                # Página não processada (não deveria acontecer)
                document.pages.append(Page(
                    page=page_num, source="digital", blocks=[]
                ))
        
        if config.VERBOSE:
            total_blocks = sum(len(p.blocks) for p in document.pages)
            total_tables = sum(
                len([b for b in p.blocks if b.type == "table"])
                for p in document.pages
            )
            print(f"\n✓ Processamento paralelo concluído!")
            print(f"  - Total de blocos: {total_blocks}")
            print(f"  - Total de tabelas: {total_tables}")
        
        return document


def process_pdf(pdf_path: str, output_dir: Optional[str] = None,
               extract_tables: bool = True,
               use_gpu: bool = None,
               parallel: bool = None) -> Document:
    """
    Função auxiliar para processar um PDF e salvar o resultado
    
    Args:
        pdf_path: caminho para o PDF
        output_dir: diretório de saída (se None, usa config.OUTPUT_DIR)
        extract_tables: extrair tabelas
        use_gpu: usar GPU
        parallel: usar processamento paralelo (se None, usa config.PARALLEL_ENABLED)
    
    Returns:
        Document processado
    """
    import gc
    
    processor = DocumentProcessor(use_gpu=use_gpu)
    
    try:
        # Decide se usa paralelo ou sequencial
        use_parallel = parallel if parallel is not None else getattr(config, 'PARALLEL_ENABLED', True)
        
        if use_parallel:
            document = processor.process_document_parallel(
                pdf_path,
                extract_tables=extract_tables,
                show_progress=True
            )
        else:
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
    
    finally:
        # Limpa recursos para evitar warning do Poppler/pdf2image
        processor.ocr_engine = None
        processor.tesseract_engine = None
        gc.collect()
