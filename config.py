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

# OCR
OCR_BATCH_SIZE = 4  # Ajustar conforme VRAM disponível
IMAGE_DPI = 300  # Resolução para conversão de PDF para imagem
MIN_CONFIDENCE = 0.5  # Confiança mínima para aceitar OCR

# Pré-processamento
BINARIZATION_METHOD = 'adaptive'  # 'otsu' ou 'adaptive'
DENOISE_KERNEL_SIZE = 3
DESKEW_ANGLE_THRESHOLD = 0.5  # graus

# Tabelas
TABLE_DETECTION_CONFIDENCE = 0.7
CAMELOT_FLAVOR = 'lattice'  # 'lattice' ou 'stream'

# Detecção de tipo de página (digital vs scan)
IMAGE_AREA_THRESHOLD = 0.3  # % mínimo de área de imagem para considerar possível scan
TEXT_COVERAGE_THRESHOLD = 0.05  # % mínimo de cobertura de texto para considerar digital puro

# Debug
SAVE_PREPROCESSED_IMAGES = False  # Salvar imagens pré-processadas para debug
VERBOSE = True
