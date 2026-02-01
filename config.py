"""
Configurações do pipeline de extração de documentos
"""
import torch
from pathlib import Path

# Diretórios
BASE_DIR = Path(__file__).parent
RESOURCE_DIR = BASE_DIR / "resource"
OUTPUT_DIR = BASE_DIR / "output"

# GPU / Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_GPU = torch.cuda.is_available()

# OCR - Configurações Gerais
OCR_ENGINE = 'doctr'  # 'doctr' ou 'tesseract'
OCR_DPI = 350  # DPI para conversão de PDF (maior = melhor qualidade, mais lento)
IMAGE_DPI = 350  # Alias para compatibilidade
MIN_CONFIDENCE = 0.3  # Confiança mínima para aceitar resultado OCR (0-1)
OCR_BATCH_SIZE = 12  # Ajustar conforme VRAM disponível
# OCR_BATCH_SIZE = 8  # Conservador, seguro
# OCR_BATCH_SIZE = 12 # Balanceado
# OCR_BATCH_SIZE = 16 # Agressivo, máximo throughput


# OCR - Tesseract (se OCR_ENGINE = 'tesseract')
OCR_LANG = 'por'  # Idioma(s): 'por', 'por+eng', etc
TESSERACT_CONFIG = '--oem 1 --psm 3'  # --oem 1: LSTM, --psm 3: auto page segmentation

# OCR - Pós-processamento
OCR_POSTPROCESS = True  # Aplicar limpeza de texto após OCR
OCR_FIX_ERRORS = True  # Corrigir erros comuns de OCR
OCR_MIN_LINE_LENGTH = 3  # Remover linhas menores que N caracteres

# Pré-processamento de imagem (DESATIVADO - degrada qualidade do OCR)
# O docTR e Tesseract fazem seu próprio pré-processamento internamente
OCR_PREPROCESS = False  # NÃO ativar - mantido para compatibilidade
BINARIZATION_METHOD = 'adaptive'  # 'otsu' ou 'adaptive' (não usado se OCR_PREPROCESS=False)
DENOISE_KERNEL_SIZE = 3
DESKEW_ANGLE_THRESHOLD = 0.5  # graus

# Tabelas
TABLE_DETECTION_CONFIDENCE = 0.7
CAMELOT_FLAVOR = 'lattice'  # 'lattice' ou 'stream'

# Detecção de tipo de página (digital vs scan)
IMAGE_AREA_THRESHOLD = 0.3  # % mínimo de área de imagem para considerar possível scan
TEXT_COVERAGE_THRESHOLD = 0.05  # % mínimo de cobertura de texto para considerar digital puro

# Paralelização
PARALLEL_ENABLED = True  # Habilitar processamento paralelo
PARALLEL_WORKERS = None  # Número de workers para páginas digitais (None = auto, usa cpu_count)
PARALLEL_MIN_PAGES = 4  # Mínimo de páginas para ativar paralelização (overhead não compensa para poucos)

# Debug
SAVE_PREPROCESSED_IMAGES = False  # Salvar imagens pré-processadas para debug
VERBOSE = True
