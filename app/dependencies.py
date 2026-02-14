"""
Dependências FastAPI para injeção via Depends().

Centraliza acesso ao processor singleton e ao semáforo de GPU,
facilitando override em testes.
"""
import asyncio

from fastapi import Request


def get_processor(request: Request):
    """Retorna o DocumentProcessor singleton armazenado em app.state."""
    return request.app.state.processor


def get_semaphore(request: Request) -> asyncio.Semaphore:
    """Retorna o semáforo de GPU armazenado em app.state."""
    return request.app.state.gpu_semaphore
