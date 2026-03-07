#!/usr/bin/env python3
"""
CLI para iniciar Celery worker.

Uso:
    python scripts/celery_worker.py worker
    python scripts/celery_worker.py worker --workers 4
    python scripts/celery_worker.py worker --with-flower
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_script_dir():
    """Obter diretório deste script."""
    return str(Path(__file__).parent)


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="CLI para Celery worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python scripts/celery_worker.py worker
    python scripts/celery_worker.py worker --workers 4
    python scripts/celery_worker.py worker --with-flower
    python scripts/celery_worker.py worker --concurrency 2
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: worker
    worker_parser = subparsers.add_parser("worker", help="Iniciar worker")
    worker_parser.add_argument(
        "--workers", "-w", type=int, default=2, help="Número de workers (default: 2)"
    )
    worker_parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Concorrência do worker (default: workers)",
    )
    worker_parser.add_argument(
        "--with-flower",
        action="store_true",
        help="Iniciar também Flower (monitoramento)",
    )
    worker_parser.add_argument(
        "--port", type=int, default=5555, help="Porta do Flower (default: 5555)"
    )
    worker_parser.add_argument(
        "--loglevel",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Nível de logging (default: info)",
    )
    worker_parser.add_argument(
        "--pool",
        type=str,
        default="solo",
        choices=["solo", "thread", "process", "eventlet", "gevent"],
        help="Pool do worker (default: solo para debug)",
    )

    # Command: beat
    beat_parser = subparsers.add_parser("beat", help="Iniciar beat scheduler")
    beat_parser.add_argument(
        "--interval", type=int, default=300, help="Intervalo em segundos (default: 300)"
    )

    # Command: inspect
    inspect_parser = subparsers.add_parser("inspect", help="Iniciar inspect")

    return parser.parse_args()


def setup_celery():
    """Configurar Celery."""
    from src.celery_worker import CeleryConfig, celery_app

    celery_app.conf.update(
        task_serializer=CeleryConfig.TASK_SERIALIZER,
        result_serializer=CeleryConfig.RESULT_SERIALIZER,
        accept_content=["pickle", "json"],
        task_track_started=True,
        time_limit=int(os.getenv("DOC_PARSER_CELERY_TASK_TIME_LIMIT", "3600")),
        soft_time_limit=int(
            os.getenv("DOC_PARSER_CELERY_TASK_SOFT_TIME_LIMIT", "3000")
        ),
    )


def start_worker(
    num_workers: int,
    concurrency: int = None,
    loglevel: str = "info",
    pool: str = "solo",
):
    """
    Iniciar Celery worker.

    Args:
        num_workers: Número de workers
        concurrency: Concorrência
        loglevel: Nível de logging
        pool: Pool do worker
    """
    setup_celery()

    concurrency = concurrency or num_workers

    worker_cmd = [
        "celery",
        "-A",
        "src.celery_worker",
        "worker",
        f"--loglevel={loglevel}",
        f"--concurrency={concurrency}",
        f"--pool={pool}",
        "--without-gossip",
        "--without-mingle",
        "--without-heartbeat",
    ]

    print("=" * 60)
    print("Celery Worker")
    print("=" * 60)
    print(f"Broker: {CeleryConfig.BROKER_URL}")
    print(f"Backend: {CeleryConfig.RESULT_BACKEND}")
    print(f"Workers: {concurrency}")
    print(f"Pool: {pool}")
    print(f"Time limit: {CeleryConfig.TASK_TIME_LIMIT}s")
    print(f"Soft time limit: {CeleryConfig.TASK_SOFT_TIME_LIMIT}s")
    print()
    print("Comando:")
    print("  " + " ".join(worker_cmd))
    print()
    print("Iniciando worker...")
    print()

    # Executar worker
    subprocess.run(worker_cmd, cwd=get_script_dir())


def start_beat(interval: int = 300):
    """
    Iniciar Celery beat.

    Args:
        interval: Intervalo em segundos
    """
    setup_celery()

    beat_cmd = [
        "celery",
        "-A",
        "src.celery_worker",
        "beat",
        f"--interval={interval}",
    ]

    print("=" * 60)
    print("Celery Beat")
    print("=" * 60)
    print(f"Intervalo: {interval}s")
    print()
    print("Comando:")
    print("  " + " ".join(beat_cmd))
    print()
    print("Iniciando beat...")
    print()

    subprocess.run(beat_cmd, cwd=get_script_dir())


def start_flower(port: int = 5555):
    """
    Iniciar Flower (monitoramento).

    Args:
        port: Porta
    """
    setup_celery()

    flower_cmd = [
        "celery",
        "-A",
        "src.celery_worker",
        "flower",
        f"--port={port}",
        f"--broker_url={CeleryConfig.BROKER_URL}",
        f"--backend_url={CeleryConfig.RESULT_BACKEND}",
    ]

    print("=" * 60)
    print("Flower (Monitoramento)")
    print("=" * 60)
    print(f"Porta: {port}")
    print(f"Broker: {CeleryConfig.BROKER_URL}")
    print(f"Backend: {CeleryConfig.RESULT_BACKEND}")
    print()
    print("Comando:")
    print("  " + " ".join(flower_cmd))
    print()
    print("Iniciando Flower...")
    print()

    subprocess.run(flower_cmd, cwd=get_script_dir())


def start_all(num_workers: int = 2, port: int = 5555):
    """
    Iniciar worker, beat e flower.

    Args:
        num_workers: Número de workers
        port: Porta do Flower
    """
    setup_celery()

    print("=" * 60)
    print("Iniciando todos os componentes Celery")
    print("=" * 60)
    print()

    # Iniciar worker em background
    worker_cmd = [
        "celery",
        "-A",
        "src.celery_worker",
        "worker",
        "--loglevel=info",
        f"--concurrency={num_workers}",
        "--pool=solo",
        "--without-gossip",
        "--without-mingle",
        "--without-heartbeat",
    ]

    print(f"Iniciando worker com {num_workers} workers...")
    subprocess.Popen(
        worker_cmd,
        cwd=get_script_dir(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Iniciar beat em background
    beat_cmd = [
        "celery",
        "-A",
        "src.celery_worker",
        "beat",
        "--interval=300",
    ]

    print("Iniciando beat...")
    subprocess.Popen(
        beat_cmd,
        cwd=get_script_dir(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Iniciar flower
    flower_cmd = [
        "celery",
        "-A",
        "src.celery_worker",
        "flower",
        f"--port={port}",
        f"--broker_url={CeleryConfig.BROKER_URL}",
        f"--backend_url={CeleryConfig.RESULT_BACKEND}",
    ]

    print(f"Iniciando Flower na porta {port}...")
    subprocess.Popen(
        flower_cmd,
        cwd=get_script_dir(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print()
    print("=" * 60)
    print("Todos os componentes iniciados!")
    print("=" * 60)
    print()
    print(f"Acessar Flower: http://localhost:{port}")
    print()
    print("Pressione Ctrl+C para parar todos os processos")


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "worker":
        start_worker(
            num_workers=args.workers,
            concurrency=args.concurrency,
            loglevel=args.loglevel,
            pool=args.pool,
        )
    elif args.command == "beat":
        start_beat(interval=args.interval)
    elif args.command == "flower":
        start_flower(port=args.port)
    elif args.command == "all":
        start_all(num_workers=args.workers, port=args.port)
    elif args.command == "inspect":
        # Celery inspect
        subprocess.run(["celery", "-A", "src.celery_worker", "inspect"])
    else:
        parser = argparse.ArgumentParser(description="CLI para Celery worker")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
