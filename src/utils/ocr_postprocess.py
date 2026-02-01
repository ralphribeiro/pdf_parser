"""
Pós-processamento de texto OCR

Funções para limpar e melhorar texto extraído por OCR:
- Remoção de ruído e caracteres inválidos
- Correção de espaçamento
- Remoção de linhas muito curtas
- União de palavras quebradas
"""
import re
from typing import List, Set, Optional


def clean_ocr_text(text: str) -> str:
    """
    Limpa texto extraído por OCR removendo ruído comum
    
    Args:
        text: texto bruto do OCR
    
    Returns:
        texto limpo
    """
    if not text:
        return ""
    
    # Remove caracteres de ruído comum do OCR
    # Mantém letras, números, pontuação básica e acentos
    noise_chars = r'[|\\{}\[\]<>©®™°§¶†‡•◦▪▫●○◆◇★☆♦♠♣♥]'
    text = re.sub(noise_chars, '', text)
    
    # Remove sequências de caracteres repetidos (ex: "====", "----", "____")
    text = re.sub(r'([=\-_*#~])\1{3,}', '', text)
    
    # Remove espaços antes de pontuação
    text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
    
    # Adiciona espaço após pontuação se não houver
    text = re.sub(r'([.,;:!?])([A-ZÀ-Úa-zà-ú])', r'\1 \2', text)
    
    # Remove espaços múltiplos
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove linhas com apenas números/símbolos (geralmente ruído)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Mantém linha se tiver pelo menos 2 caracteres alfabéticos
        if len(re.findall(r'[A-Za-zÀ-ú]', line)) >= 2:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def remove_short_lines(text: str, min_length: int = 3) -> str:
    """
    Remove linhas muito curtas que geralmente são ruído
    
    Args:
        text: texto
        min_length: comprimento mínimo para manter linha
    
    Returns:
        texto sem linhas curtas
    """
    lines = text.split('\n')
    filtered = [line for line in lines if len(line.strip()) >= min_length]
    return '\n'.join(filtered)


def fix_common_ocr_errors(text: str) -> str:
    """
    Corrige erros comuns de OCR em português
    
    Args:
        text: texto com possíveis erros
    
    Returns:
        texto corrigido
    """
    corrections = {
        # Letras confundidas
        r'\bRN\b': 'RN',  # Mantém RN (documento)
        r'l<': 'k',
        r'\bl\b(?=[A-Z])': 'I',  # l sozinho antes de maiúscula -> I
        r'(?<=[a-z])O(?=[a-z])': 'o',  # O entre minúsculas -> o
        r'(?<=[A-Z])o(?=[A-Z])': 'O',  # o entre maiúsculas -> O
        
        # Números confundidos com letras
        r'(?<=[A-Za-z])0(?=[A-Za-z])': 'O',  # 0 entre letras -> O
        r'(?<=[0-9])O(?=[0-9])': '0',  # O entre números -> 0
        r'(?<=[A-Za-z])1(?=[A-Za-z])': 'l',  # 1 entre letras -> l
        r'(?<=[0-9])l(?=[0-9])': '1',  # l entre números -> 1
        
        # Palavras comuns mal reconhecidas (português jurídico)
        r'\bDl<\b': 'DK',
        r'\bNQ\b': 'Nº',
        r'\bn2\b': 'nº',
        r'\bNR\b': 'NR',  # Mantém
    }
    
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    
    return text


def merge_broken_words(text: str, min_word_length: int = 4) -> str:
    """
    Tenta unir palavras que foram quebradas incorretamente
    
    Exemplo: "TA BELIÃO" -> pode permanecer se não tivermos dicionário
    
    Args:
        text: texto
        min_word_length: comprimento mínimo de fragmento para tentar unir
    
    Returns:
        texto com palavras unidas
    """
    # Padrão: palavra curta + espaço + palavra curta no início de palavra maior
    # Ex: "COM ARCA" onde "COMARCA" seria esperado
    
    # Esta é uma heurística simples - idealmente usaríamos um dicionário
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        words = line.split()
        if len(words) < 2:
            fixed_lines.append(line)
            continue
        
        # Tenta unir fragmentos muito curtos
        merged = []
        i = 0
        while i < len(words):
            word = words[i]
            
            # Se palavra é muito curta e próxima também é
            if (len(word) <= 2 and 
                i + 1 < len(words) and 
                len(words[i + 1]) >= 2 and
                word.isupper() == words[i + 1].isupper()):
                # Une as palavras
                merged.append(word + words[i + 1])
                i += 2
            else:
                merged.append(word)
                i += 1
        
        fixed_lines.append(' '.join(merged))
    
    return '\n'.join(fixed_lines)


def normalize_whitespace(text: str) -> str:
    """
    Normaliza espaços em branco
    
    Args:
        text: texto
    
    Returns:
        texto com espaços normalizados
    """
    # Remove espaços no início/fim de linhas
    lines = [line.strip() for line in text.split('\n')]
    
    # Remove linhas vazias consecutivas
    cleaned = []
    prev_empty = False
    for line in lines:
        if line:
            cleaned.append(line)
            prev_empty = False
        elif not prev_empty:
            cleaned.append('')
            prev_empty = True
    
    return '\n'.join(cleaned).strip()


def postprocess_ocr_text(text: str, 
                        clean: bool = True,
                        fix_errors: bool = True,
                        merge_words: bool = False,
                        min_line_length: int = 3) -> str:
    """
    Pipeline completo de pós-processamento de texto OCR
    
    Args:
        text: texto bruto do OCR
        clean: aplicar limpeza de ruído
        fix_errors: corrigir erros comuns
        merge_words: tentar unir palavras quebradas
        min_line_length: comprimento mínimo de linha
    
    Returns:
        texto processado
    """
    if not text:
        return ""
    
    if clean:
        text = clean_ocr_text(text)
    
    if fix_errors:
        text = fix_common_ocr_errors(text)
    
    if merge_words:
        text = merge_broken_words(text)
    
    if min_line_length > 0:
        text = remove_short_lines(text, min_line_length)
    
    text = normalize_whitespace(text)
    
    return text
