# Document Parser Pipeline (otimizado para ROCm)

Pipeline local para extração de texto e estrutura de PDFs mistos (digitais e escaneados), com saída em formato JSON. Otimizado para ROCm.

## Características

- ✅ **PDFs Digitais**: Extração direta de texto com `pdfplumber`
- ✅ **PDFs Escaneados**: OCR com `docTR` (PyTorch + ROCm/CUDA)
- ✅ **Tabelas**: Detecção e extração com `camelot-py`
- ✅ **Pré-processamento**: Deskew, binarização, melhoria de contraste
- ✅ **Posicionamento**: Coordenadas normalizadas (0-1) para cada bloco
- ✅ **Estruturação**: Blocos organizados por tipo (parágrafo, tabela, etc)

## Estrutura de Saída JSON

```json
{
  "doc_id": "1008086-69.2016.8.26.0005",
  "source_file": "1008086-69.2016.8.26.0005.pdf",
  "total_pages": 353,
  "processing_date": "2026-01-31T22:40:00Z",
  "pages": [
    {
      "page": 1,
      "source": "digital",
      "blocks": [
        {
          "block_id": "p1_b1",
          "type": "paragraph",
          "text": "Conteúdo do parágrafo...",
          "bbox": [0.1, 0.2, 0.9, 0.3],
          "confidence": 1.0
        }
      ]
    }
  ]
}
```

### Tipos de Blocos

- `paragraph`: Parágrafos de texto
- `table`: Tabelas (inclui campo `rows` com dados)
- `header`: Cabeçalhos
- `footer`: Rodapés
- `list`: Listas
- `image`: Imagens (placeholder para implementação futura)

### Coordenadas BBox

Todas as coordenadas são normalizadas (0-1) no formato `[x1, y1, x2, y2]`:
- `x1, y1`: Canto superior esquerdo
- `x2, y2`: Canto inferior direito

## Instalação

### Pré-requisitos

- Python 3.10+
- PyTorch com suporte ROCm (GPU AMD) ou CUDA (GPU NVIDIA)
- Ghostscript (para camelot)

```bash
# Ubuntu/Debian
sudo apt-get install ghostscript poppler-utils

# macOS
brew install ghostscript poppler
```

### Dependências Python

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Uso

### Processar um PDF

```bash
python scripts/process_single.py resource/documento.pdf
```

### Opções

```bash
# Especificar diretório de saída
python scripts/process_single.py documento.pdf -o /caminho/saida

# Desativar extração de tabelas
python scripts/process_single.py documento.pdf --no-tables

# Forçar uso de CPU (sem GPU)
python scripts/process_single.py documento.pdf --no-gpu

# Modo silencioso
python scripts/process_single.py documento.pdf --quiet
```

### Uso Programático

```python
from src.pipeline import process_pdf

# Processa PDF e salva JSON
document = process_pdf(
    'caminho/para/documento.pdf',
    output_dir='output',
    extract_tables=True,
    use_gpu=True
)

# Acessa dados do documento
print(f"Total de páginas: {document.total_pages}")
print(f"Total de blocos: {sum(len(p.blocks) for p in document.pages)}")

# Salva JSON customizado
from src.pipeline import DocumentProcessor

processor = DocumentProcessor(use_gpu=True)
doc = processor.process_document('documento.pdf')
processor.save_to_json(doc, 'saida.json', indent=4)
```

## Configuração

Edite `config.py` para ajustar parâmetros:

```python
# GPU / Device
DEVICE = 'cuda'  # ou 'cpu'
OCR_BATCH_SIZE = 4  # Ajustar conforme VRAM

# OCR
IMAGE_DPI = 300  # Resolução para OCR
MIN_CONFIDENCE = 0.5  # Confiança mínima para aceitar resultado

# Pré-processamento
BINARIZATION_METHOD = 'adaptive'  # ou 'otsu'
DESKEW_ANGLE_THRESHOLD = 0.5  # graus

# Tabelas
TABLE_DETECTION_CONFIDENCE = 0.7
CAMELOT_FLAVOR = 'lattice'  # ou 'stream'
```

## Performance

Testado com PDF de 353 páginas (27 MB):
- **Tempo**: ~5.5 minutos (CPU AMD Ryzen)
- **GPU**: Suporta AMD RX 7900 XT via ROCm
- **Velocidade média**: ~1.2 páginas/segundo
- **Saída JSON**: 754 KB (591 blocos, 27 tabelas)

## Arquitetura

```
src/
├── pipeline.py              # Orquestrador principal
├── detector.py              # Detecta tipo de página (digital/scan)
├── extractors/
│   ├── digital.py          # Extração de PDF digital
│   ├── ocr.py              # OCR com docTR
│   └── tables.py           # Extração de tabelas
├── preprocessing/
│   └── image_enhancer.py   # Pré-processamento de imagem
├── models/
│   └── schemas.py          # Schemas Pydantic
└── utils/
    ├── bbox.py             # Funções de bounding box
    └── text_normalizer.py # Limpeza de texto
```

## Exemplo de Resultado

```
✅ Processamento concluído com sucesso!
   - Documento: 1008086-69.2016.8.26.0005
   - Páginas: 353
   - Blocos totais: 591
   - Tabelas detectadas: 27

✓ JSON salvo em: output/1008086-69.2016.8.26.0005.json
  Tamanho: 753.6 KB
```

## Limitações Atuais

- ✅ Extração de texto e tabelas
- ✅ Posicionamento aproximado (bbox normalizado)
- ⏳ Assinaturas: tratamento futuro
- ⏳ Detecção de layout avançado (colunas, seções)
- ⏳ Reconhecimento de formulários

## Roadmap

1. ✅ Pipeline básico funcional
2. ⏳ Melhoria de detecção de tabelas em PDFs escaneados
3. ⏳ Detecção e classificação de assinaturas
4. ⏳ Chunking inteligente para embeddings
5. ⏳ Pipeline de vetorização (embeddings)
6. ⏳ Processamento batch de múltiplos PDFs

## Licença

Projeto interno para uso em análise de documentos jurídicos.
