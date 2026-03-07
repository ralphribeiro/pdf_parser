#!/usr/bin/env python3
"""
Pós-correção contextual de blocos OCR via LLM (Ollama).

Percorre o JSON de output do pipeline, envia cada bloco de páginas
com source="ocr" para o modelo e gera um novo arquivo corrigido.

Uso:
    python scripts/llm_postprocess.py output/documento.json
    python scripts/llm_postprocess.py output/documento.json -o output/documento_corrigido.json
    python scripts/llm_postprocess.py output/documento.json --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://192.168.0.25:11434/api/generate"
MODEL = "gpt-oss-safeguard:latest"

SYSTEM_PROMPT = (
    "Você é um revisor especializado em documentos jurídicos brasileiros. "
    "Receba o texto extraído por OCR de um documento judicial/legal e devolva "
    "APENAS o texto corrigido, sem explicações, comentários ou formatação extra. "
    "Corrija erros de OCR: letras trocadas, acentuação, cedilha, palavras "
    "quebradas ou grudadas, pontuação e capitalização. "
    "Preserve fielmente o conteúdo original — não resuma, não omita, não "
    "reescreva. Mantenha nomes próprios, números de processo, datas, valores "
    "monetários e siglas exatamente como aparecem (corrigindo apenas erros "
    "evidentes de OCR). Se o texto já estiver correto, devolva-o inalterado."
)


def call_ollama(text: str, timeout: int = 120) -> str:
    """Envia texto para o Ollama e retorna a resposta completa."""
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def process_file(input_path: str, output_path: str, dry_run: bool = False) -> None:
    with open(input_path, encoding="utf-8") as f:
        doc = json.load(f)

    ocr_pages = [p for p in doc["pages"] if p["source"] == "ocr"]
    total_blocks = sum(len(p["blocks"]) for p in ocr_pages)

    print(f"Páginas OCR: {len(ocr_pages)}  |  Blocos a corrigir: {total_blocks}")

    if dry_run:
        print("(dry-run) Nenhuma chamada será feita.")
        return

    done = 0
    t0 = time.time()

    for page in ocr_pages:
        for block in page["blocks"]:
            text = block.get("text")
            if not text or len(text.strip()) < 5:
                done += 1
                continue

            done += 1
            try:
                corrected = call_ollama(text)
                if corrected:
                    block["text"] = corrected
                print(
                    f"  [{done}/{total_blocks}] p{page['page']} "
                    f"{block['block_id']} ok ({len(text)}→{len(corrected)} chars)"
                )
            except requests.RequestException as exc:
                print(
                    f"  [{done}/{total_blocks}] p{page['page']} "
                    f"{block['block_id']} ERRO: {exc}"
                )

    elapsed = time.time() - t0
    print(f"\nConcluído em {elapsed:.1f}s  ({done} blocos)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"Salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pós-correção OCR via LLM (Ollama)")
    parser.add_argument("input", help="JSON de entrada (output do pipeline)")
    parser.add_argument(
        "-o",
        "--output",
        help="JSON de saída (default: <input>_llm.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas conta blocos, sem chamar a LLM",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Arquivo não encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(input_path.with_stem(input_path.stem + "_llm"))

    process_file(str(input_path), output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
