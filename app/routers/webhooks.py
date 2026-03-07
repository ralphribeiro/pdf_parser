"""
Webhooks para notificações assíncronas.

Testes: tests/test_async_jobs.py (testes de webhook)
"""

import hashlib
import hmac
import logging
import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.models.mongodb import WebhookPayload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# Schema para verificação de webhooks
class WebhookVerification(BaseModel):
    """Schema para verificação de webhook."""

    signature: str | None = Field(None, description="Assinatura do webhook")
    timestamp: str | None = Field(None, description="Timestamp do webhook")


# Funções auxiliares


def verify_webhook_signature(
    body: bytes, signature: str | None, token: str | None
) -> bool:
    """
    Verificar assinatura do webhook.

    Args:
        body: Corpo do request
        signature: Assinatura fornecida
        token: Token de autenticação

    Returns:
        True se verificado, False senão
    """
    if not token:
        return False

    if not signature:
        return False

    # Calcular hash esperado
    expected_signature = hmac.new(token.encode(), body, hashlib.sha256).hexdigest()

    # Comparar usando timing-safe compare
    return hmac.compare_digest(signature, expected_signature)


def get_webhook_token() -> str | None:
    """
    Obter token de webhook.

    Returns:
        Token ou None
    """
    return os.getenv("DOC_PARSER_WEBHOOK_AUTH_TOKEN")


async def verify_webhook(
    request: Request, token: str | None = Depends(get_webhook_token)
) -> dict:
    """
    Depêndência para verificar webhook.

    Args:
        request: Request FastAPI
        token: Token de autenticação

    Returns:
        Dicionário de verificação
    """
    body = await request.body()
    signature = request.headers.get("X-Signature") or request.headers.get(
        "X-Webhook-Signature"
    )

    is_valid = verify_webhook_signature(body, signature, token)

    logger.info(f"Webhook verification: {is_valid}")

    return {"verified": is_valid, "timestamp": datetime.utcnow().isoformat()}


@router.post("/verify")
async def verify_webhook_endpoint(
    request: Request, token: str | None = Depends(get_webhook_token)
) -> dict:
    """
    Verificar webhook manualmente.

    Endpoint para verificar assinatura de webhook recebido.
    """
    body = await request.body()
    signature = request.headers.get("X-Signature") or request.headers.get(
        "X-Webhook-Signature"
    )

    if not signature:
        raise HTTPException(status_code=400, detail="X-Signature header required")

    is_valid = verify_webhook_signature(body, signature, token)

    if not is_valid:
        raise HTTPException(
            status_code=401, detail="Webhook signature verification failed"
        )

    return {"verified": True, "timestamp": datetime.utcnow().isoformat()}


@router.post("/{job_id}")
async def receive_webhook(
    job_id: str,
    request: Request,
    payload: WebhookPayload,
    token: str | None = Depends(get_webhook_token),
) -> dict:
    """
    Receber webhook de notificação.

    Este endpoint recebe notificações de jobs assíncronos.
    """
    try:
        # Salvar webhook no MongoDB
        from src.database import create_webhook

        create_webhook(
            job_id=job_id,
            url=payload.url if hasattr(payload, "url") else None,
            token=payload.token if hasattr(payload, "token") else None,
        )

        logger.info(f"Webhook recebido: {job_id}")

        return {
            "success": True,
            "job_id": job_id,
            "status": payload.status,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao processar webhook {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar webhook: {e!s}")


@router.get("/{job_id}")
async def get_webhook(
    job_id: str, token: str | None = Depends(get_webhook_token)
) -> dict:
    """
    Obter webhook pelo job_id.

    Args:
        job_id: ID do job
        token: Token de autenticação

    Returns:
        Configuração do webhook
    """
    from src.database import get_webhook

    try:
        webhook = get_webhook(job_id)

        if not webhook:
            raise HTTPException(
                status_code=404, detail=f"Webhook não encontrado para job: {job_id}"
            )

        return {
            "job_id": webhook.job_id,
            "url": webhook.url,
            "token": webhook.token,
            "enabled": webhook.enabled,
            "created_at": webhook.created_at,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter webhook {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter webhook: {e!s}")


@router.delete("/{job_id}")
async def delete_webhook(
    job_id: str, token: str | None = Depends(get_webhook_token)
) -> dict:
    """
    Deletar webhook.

    Args:
        job_id: ID do job
        token: Token de autenticação

    Returns:
        Resultado da deleção
    """
    from src.database import delete_webhook

    try:
        result = delete_webhook(job_id)
        logger.info(f"Webhook deletado: {job_id}")

        return {"success": True, "job_id": job_id, "deleted": result.deleted_count > 0}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao deletar webhook {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao deletar webhook: {e!s}")


@router.get("/config")
async def get_webhook_config(token: str | None = Depends(get_webhook_token)) -> dict:
    """
    Obter configuração de webhook.

    Args:
        token: Token de autenticação

    Returns:
        Configuração de webhook
    """
    token_value = os.getenv("DOC_PARSER_WEBHOOK_AUTH_TOKEN")

    return {
        "enabled": bool(token_value),
        "has_token": bool(token_value),
        "max_retries": int(os.getenv("DOC_PARSER_WEBHOOK_MAX_RETRIES", "3")),
        "timeout_seconds": int(os.getenv("DOC_PARSER_WEBHOOK_TIMEOUT", "30")),
    }


@router.post("/test")
async def test_webhook(
    request: Request, token: str | None = Depends(get_webhook_token)
) -> dict:
    """
    Testar webhook.

    Endpoint de teste para verificar se webhook está funcionando.
    """
    body = await request.body()

    return {
        "success": True,
        "received_body": len(body),
        "timestamp": datetime.utcnow().isoformat(),
    }
