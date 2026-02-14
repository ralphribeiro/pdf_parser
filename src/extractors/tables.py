"""
Extrator de tabelas de PDFs (digitais e scaneados)
"""
import logging
from typing import List, Optional, Tuple

import camelot
import numpy as np

import config
from src.models.schemas import Block, BlockType
from src.utils.bbox import normalize_bbox

logger = logging.getLogger(__name__)


def extract_tables_digital(pdf_path: str, page_number: int,
                          flavor: str = None) -> List[Block]:
    """
    Extrai tabelas de página digital usando camelot

    Args:
        pdf_path: caminho para o PDF
        page_number: número da página (1-indexed)
        flavor: 'lattice' (tabelas com bordas) ou 'stream' (sem bordas)

    Returns:
        lista de blocos de tabela
    """
    if flavor is None:
        flavor = config.CAMELOT_FLAVOR

    blocks = []

    try:
        # Extrai tabelas da página
        tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_number),
            flavor=flavor,
            suppress_stdout=True
        )

        if not tables:
            return blocks

        # Processa cada tabela detectada
        for idx, table in enumerate(tables):
            # Verifica confiança da detecção
            confidence = table.parsing_report.get('accuracy', 0.0) / 100.0

            if confidence < config.TABLE_DETECTION_CONFIDENCE:
                continue

            # Converte tabela para lista de listas
            rows = table.df.values.tolist()

            # Remove linhas completamente vazias
            rows = [row for row in rows if any(str(cell).strip() for cell in row)]

            if not rows:
                continue

            # Obtém bbox da tabela (camelot retorna coordenadas absolutas)
            # Camelot usa coordenadas com origem no canto inferior esquerdo
            x1, y1, x2, y2 = table._bbox

            # Converte para coordenadas normalizadas
            # Precisa obter dimensões da página
            from src.detector import get_page_dimensions
            page_width, page_height = get_page_dimensions(pdf_path, page_number)

            # Camelot usa coordenadas com Y invertido (origem embaixo)
            # Precisamos converter para Y com origem em cima
            bbox = normalize_bbox(
                [x1, page_height - y2, x2, page_height - y1],
                page_width,
                page_height
            )

            block = Block(
                block_id=f"p{page_number}_t{idx + 1}",
                type=BlockType.TABLE,
                text=None,  # Tabelas não têm campo texto
                bbox=bbox,
                confidence=confidence,
                rows=rows
            )

            blocks.append(block)

    except Exception as e:
        logger.warning("Erro ao extrair tabelas da página %d: %s", page_number, e)

    return blocks


def extract_tables_from_blocks(blocks: List[Block],
                               min_rows: int = 2,
                               min_cols: int = 2) -> List[Block]:
    """
    Detecta tabelas em blocos de texto OCR usando heurísticas

    Esta função tenta identificar padrões tabulares em blocos de texto
    baseando-se em alinhamento e estrutura regular.

    Args:
        blocks: blocos de texto extraídos por OCR
        min_rows: número mínimo de linhas para considerar tabela
        min_cols: número mínimo de colunas para considerar tabela

    Returns:
        lista de blocos de tabela detectados
    """
    # TODO: Implementar detecção de tabelas em OCR
    # Esta é uma funcionalidade avançada que pode ser adicionada depois
    # Por enquanto, retorna lista vazia
    return []


def merge_table_cells(rows: List[List[str]]) -> List[List[str]]:
    """
    Mescla células de tabela que foram quebradas incorretamente

    Args:
        rows: linhas da tabela

    Returns:
        linhas com células mescladas
    """
    if not rows:
        return rows

    cleaned_rows = []

    for row in rows:
        # Remove células vazias no início e fim
        cleaned_row = []
        for cell in row:
            cell_str = str(cell).strip()
            cleaned_row.append(cell_str)

        cleaned_rows.append(cleaned_row)

    return cleaned_rows


def validate_table_structure(rows: List[List[str]]) -> bool:
    """
    Valida se a estrutura da tabela é consistente

    Args:
        rows: linhas da tabela

    Returns:
        True se a estrutura é válida
    """
    if not rows:
        return False

    # Verifica se todas as linhas têm o mesmo número de colunas
    num_cols = len(rows[0])

    if num_cols == 0:
        return False

    for row in rows:
        if len(row) != num_cols:
            # Permite pequenas variações (±1 coluna)
            if abs(len(row) - num_cols) > 1:
                return False

    return True


def normalize_table_data(rows: List[List[str]]) -> List[List[str]]:
    """
    Normaliza dados da tabela (limpeza de texto, formatação)

    Args:
        rows: linhas brutas da tabela

    Returns:
        linhas normalizadas
    """
    normalized = []

    for row in rows:
        normalized_row = []
        for cell in row:
            # Converte para string e limpa
            cell_str = str(cell).strip()

            # Remove quebras de linha dentro de células
            cell_str = cell_str.replace('\n', ' ')

            # Remove espaços múltiplos
            cell_str = ' '.join(cell_str.split())

            normalized_row.append(cell_str)

        normalized.append(normalized_row)

    return normalized
