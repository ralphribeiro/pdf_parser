#!/usr/bin/env python3
"""
Script para verificar a configuração do ambiente
"""
import sys
from pathlib import Path

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_setup():
    """Verifica configuração do ambiente"""
    print("=" * 60)
    print("VERIFICAÇÃO DO AMBIENTE - Document Parser Pipeline")
    print("=" * 60)
    
    # Python
    print(f"\n📦 Python: {sys.version.split()[0]}")
    
    # PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   GPU (CUDA) disponível: {'Sim' if torch.cuda.is_available() else 'Não'}")
        if torch.cuda.is_available():
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
    except ImportError:
        print("❌ PyTorch não instalado")
        return False
    
    # Dependências principais
    deps = {
        'pdfplumber': 'Extração de PDFs digitais',
        'pdf2image': 'Conversão PDF para imagem',
        'cv2': 'Processamento de imagem (OpenCV)',
        'PIL': 'Pillow (imagens)',
        'doctr': 'OCR e layout detection',
        'camelot': 'Extração de tabelas',
        'pydantic': 'Validação de schemas',
        'tqdm': 'Barras de progresso'
    }
    
    print("\n📚 Dependências:")
    all_ok = True
    for module, desc in deps.items():
        try:
            if module == 'cv2':
                __import__('cv2')
            elif module == 'PIL':
                __import__('PIL')
            elif module == 'doctr':
                __import__('doctr')
            else:
                __import__(module)
            print(f"   ✅ {module:15s} - {desc}")
        except ImportError:
            print(f"   ❌ {module:15s} - {desc} (NÃO INSTALADO)")
            all_ok = False
    
    # Estrutura de diretórios
    print("\n📁 Estrutura de diretórios:")
    dirs = ['src', 'scripts', 'resource', 'output', '.cache']
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f"   ✅ {d}/")
        else:
            print(f"   ⚠️  {d}/ (não existe)")
    
    # Config
    print("\n⚙️  Configuração:")
    try:
        import config
        print(f"   Device: {config.DEVICE}")
        print(f"   OCR Batch Size: {config.OCR_BATCH_SIZE}")
        print(f"   Image DPI: {config.IMAGE_DPI}")
        print(f"   Min Confidence: {config.MIN_CONFIDENCE}")
    except Exception as e:
        print(f"   ❌ Erro ao carregar config: {e}")
    
    # Teste de import do pipeline
    print("\n🔧 Pipeline:")
    try:
        from src.pipeline import DocumentProcessor
        print("   ✅ DocumentProcessor importado com sucesso")
    except Exception as e:
        print(f"   ❌ Erro ao importar pipeline: {e}")
        all_ok = False
    
    # Resumo
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ AMBIENTE CONFIGURADO CORRETAMENTE!")
        print("\nPróximo passo:")
        print("  python scripts/process_single.py resource/seu_documento.pdf")
    else:
        print("⚠️  ALGUNS PROBLEMAS DETECTADOS")
        print("\nInstale as dependências faltantes:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return all_ok


if __name__ == '__main__':
    success = check_setup()
    sys.exit(0 if success else 1)
