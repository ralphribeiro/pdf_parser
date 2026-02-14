# Document Parser Pipeline (optimized for ROCm)

Local pipeline for text and structure extraction from mixed PDFs (digital and scanned), with JSON output. Optimized for ROCm.

## Features

- ✅ **Digital PDFs**: Direct text extraction with `pdfplumber`
- ✅ **Scanned PDFs**: OCR with `docTR` (PyTorch + ROCm/CUDA)
- ✅ **Tables**: Detection and extraction with `camelot-py`
- ✅ **Preprocessing**: Deskew, binarization, contrast enhancement
- ✅ **Positioning**: Normalized coordinates (0-1) for each block
- ✅ **Structuring**: Blocks organized by type (paragraph, table, etc.)

## JSON Output Structure

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
          "text": "Paragraph content...",
          "bbox": [0.1, 0.2, 0.9, 0.3],
          "confidence": 1.0
        }
      ]
    }
  ]
}
```

### Block Types

- `paragraph`: Text paragraphs
- `table`: Tables (includes `rows` field with data)
- `header`: Headers
- `footer`: Footers
- `list`: Lists
- `image`: Images (placeholder for future implementation)

### BBox Coordinates

All coordinates are normalized (0-1) in the format `[x1, y1, x2, y2]`:
- `x1, y1`: Top-left corner
- `x2, y2`: Bottom-right corner

## Installation

### Prerequisites

- Python 3.10+
- PyTorch with ROCm support (AMD GPU) or CUDA (NVIDIA GPU)
- Ghostscript (for camelot)

```bash
# Ubuntu/Debian
sudo apt-get install ghostscript poppler-utils

# macOS
brew install ghostscript poppler
```

### Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Usage

### Process a PDF

```bash
python scripts/process_single.py resource/document.pdf
```

### Options

```bash
# Specify output directory
python scripts/process_single.py document.pdf -o /path/to/output

# Disable table extraction
python scripts/process_single.py document.pdf --no-tables

# Force CPU usage (no GPU)
python scripts/process_single.py document.pdf --no-gpu

# Quiet mode
python scripts/process_single.py document.pdf --quiet
```

### Programmatic Usage

```python
from src.pipeline import process_pdf

# Process PDF and save JSON
document = process_pdf(
    'path/to/document.pdf',
    output_dir='output',
    extract_tables=True,
    use_gpu=True
)

# Access document data
print(f"Total pages: {document.total_pages}")
print(f"Total blocks: {sum(len(p.blocks) for p in document.pages)}")

# Save custom JSON
from src.pipeline import DocumentProcessor

processor = DocumentProcessor(use_gpu=True)
doc = processor.process_document('document.pdf')
processor.save_to_json(doc, 'output.json', indent=4)
```

## Configuration

Edit `config.py` to adjust parameters:

```python
# GPU / Device
DEVICE = 'cuda'  # or 'cpu'
OCR_BATCH_SIZE = 4  # Adjust according to VRAM

# OCR
IMAGE_DPI = 300  # Resolution for OCR
MIN_CONFIDENCE = 0.5  # Minimum confidence to accept result

# Preprocessing
BINARIZATION_METHOD = 'adaptive'  # or 'otsu'
DESKEW_ANGLE_THRESHOLD = 0.5  # degrees

# Tables
TABLE_DETECTION_CONFIDENCE = 0.7
CAMELOT_FLAVOR = 'lattice'  # or 'stream'
```

## Performance

Tested with a 353-page PDF (27 MB):
- **Time**: ~5.5 minutes (AMD Ryzen CPU)
- **GPU**: Supports AMD RX 7900 XT via ROCm
- **Average speed**: ~1.2 pages/second
- **JSON output**: 754 KB (591 blocks, 27 tables)

## Architecture

```
src/
├── pipeline.py              # Main orchestrator
├── detector.py              # Detects page type (digital/scan)
├── extractors/
│   ├── digital.py          # Digital PDF extraction
│   ├── ocr.py              # OCR with docTR
│   └── tables.py           # Table extraction
├── preprocessing/
│   └── image_enhancer.py   # Image preprocessing
├── models/
│   └── schemas.py          # Pydantic schemas
└── utils/
    ├── bbox.py             # Bounding box functions
    └── text_normalizer.py # Text cleanup
```

## Example Output

```
✅ Processing completed successfully!
   - Document: 1008086-69.2016.8.26.0005
   - Pages: 353
   - Total blocks: 591
   - Tables detected: 27

✓ JSON saved to: output/1008086-69.2016.8.26.0005.json
  Size: 753.6 KB
```

## Current Limitations

- ✅ Text and table extraction
- ✅ Approximate positioning (normalized bbox)
- ⏳ Signatures: future handling
- ⏳ Advanced layout detection (columns, sections)
- ⏳ Form recognition

## Roadmap

1. ✅ Basic functional pipeline
2. ⏳ Improved table detection in scanned PDFs
3. ⏳ Signature detection and classification
4. ⏳ Smart chunking for embeddings
5. ⏳ Vectorization pipeline (embeddings)
6. ⏳ Batch processing of multiple PDFs

## License

Internal project for legal document analysis.
