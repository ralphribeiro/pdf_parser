"""
Funções para normalização e limpeza de texto extraído
"""
import re
from typing import List


def normalize_text(text: str, remove_extra_whitespace: bool = True) -> str:
    """
    Normaliza texto extraído

    Args:
        text: texto bruto
        remove_extra_whitespace: remover espaços extras

    Returns:
        texto normalizado
    """
    if not text:
        return ""

    # Remove caracteres de controle (exceto quebras de linha e tabs)
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

    if remove_extra_whitespace:
        # Remove espaços múltiplos
        text = re.sub(r' +', ' ', text)

        # Remove espaços no início e fim de linhas
        lines = [line.strip() for line in text.split('\n')]

        # Remove linhas vazias múltiplas
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')
                prev_empty = True

        text = '\n'.join(cleaned_lines)

    return text.strip()


def merge_hyphenated_words(text: str) -> str:
    """
    Mescla palavras hifenizadas no fim de linha

    Exemplo:
        "Isso é um exem-\nplo" -> "Isso é um exemplo"
    """
    # Padrão: palavra + hífen + quebra de linha + palavra
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text


def remove_repeated_headers_footers(lines: List[str], min_repetitions: int = 3) -> List[str]:
    """
    Remove cabeçalhos e rodapés repetidos

    Args:
        lines: linhas do documento
        min_repetitions: número mínimo de repetições para considerar header/footer

    Returns:
        linhas filtradas
    """
    if len(lines) < min_repetitions * 2:
        return lines

    # Detecta padrões repetidos no início (headers)
    first_lines = lines[:5]
    header_candidates = []

    for line in first_lines:
        if line.strip():
            count = sum(1 for l in lines if l.strip() == line.strip())
            if count >= min_repetitions:
                header_candidates.append(line.strip())

    # Detecta padrões repetidos no fim (footers)
    last_lines = lines[-5:]
    footer_candidates = []

    for line in last_lines:
        if line.strip():
            count = sum(1 for l in lines if l.strip() == line.strip())
            if count >= min_repetitions:
                footer_candidates.append(line.strip())

    # Remove headers e footers
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped not in header_candidates and stripped not in footer_candidates:
            filtered.append(line)

    return filtered


def clean_ocr_artifacts(text: str) -> str:
    """
    Remove artefatos comuns de OCR
    """
    # Remove caracteres isolados estranhos
    text = re.sub(r'\s[•·∙■□▪▫]\s', ' ', text)

    # Corrige espaços antes de pontuação
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Corrige múltiplos pontos
    text = re.sub(r'\.{3,}', '...', text)

    return text


def split_into_sentences(text: str) -> List[str]:
    """
    Divide texto em sentenças (útil para chunking posterior)
    """
    # Padrão simples de detecção de fim de sentença
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ú])', text)
    return [s.strip() for s in sentences if s.strip()]
