"""
FastAPI dependencies for injection via Depends().

Centralizes access to the processor singleton and the GPU semaphore,
facilitating overrides in tests.
"""
import asyncio

from fastapi import Request


def get_processor(request: Request):
    """Returns the singleton DocumentProcessor stored in app.state."""
    return request.app.state.processor


def get_semaphore(request: Request) -> asyncio.Semaphore:
    """Returns the GPU semaphore stored in app.state."""
    return request.app.state.gpu_semaphore
