"""
Celery worker tasks para processamento assíncrono de PDFs.

Testes: tests/test_celery_worker.py
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from celery import Celery, Task

# Import local modules
from src.database import (
    create_job_document as create_job_document_db,
)
from src.database import (
    delete_old_jobs,
    delete_webhook,
    get_job,
    save_document,
    save_embedding,
)
from src.database import (
    update_job_status as update_job_status_db,
)
from src.embeddings import EmbeddingError, EmbeddingsClient
from src.models.mongodb import JobStatus

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configurar Celery
class CeleryConfig:
    """Configurações do Celery."""

    BROKER_URL = os.getenv("DOC_PARSER_CELERY_BROKER_URL", "redis://localhost:6379/0")
    RESULT_BACKEND = os.getenv(
        "DOC_PARSER_CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
    )
    WORKERS = int(os.getenv("DOC_PARSER_CELERY_WORKERS", "2"))

    # Serializers
    TASK_SERIALIZER = os.getenv("DOC_PARSER_CELERY_TASK_SERIALIZER", "pickle")
    RESULT_SERIALIZER = os.getenv("DOC_PARSER_CELERY_RESULT_SERIALIZER", "pickle")

    # Timeouts
    TASK_TIME_LIMIT = int(
        os.getenv("DOC_PARSER_CELERY_TASK_TIME_LIMIT", "3600")
    )  # 1 hora
    TASK_SOFT_TIME_LIMIT = int(
        os.getenv("DOC_PARSER_CELERY_TASK_SOFT_TIME_LIMIT", "3000")
    )  # 50 minutos

    # Embeddings
    EMBEDDINGS_URL = os.getenv(
        "DOC_PARSER_EMBEDDINGS_URL", "http://localhost:11434/api/generate"
    )


celery_app = Celery(
    "doc_parser",
    broker=CeleryConfig.BROKER_URL,
    backend=CeleryConfig.RESULT_BACKEND,
    task_serializer=CeleryConfig.TASK_SERIALIZER,
    result_serializer=CeleryConfig.RESULT_SERIALIZER,
    accept_content=["pickle", "json"],
    task_track_started=True,
    time_limit=CeleryConfig.TASK_TIME_LIMIT,
    soft_time_limit=CeleryConfig.TASK_SOFT_TIME_LIMIT,
)


@celery_app.task(bind=True, name="src.celery_worker.process_pdf_job")
def process_pdf_job(
    self: Task,
    job_id: str,
    file_content: bytes,
    generate_embeddings: bool = True,
    webhook_url: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Processar PDF em background.

    Args:
        self: Task instance
        job_id: ID do job
        file_content: Conteúdo do arquivo PDF
        generate_embeddings: Gerar embeddings
        webhook_url: URL do webhook
        **kwargs: Outros parâmetros

    Returns:
        Documento processado
    """
    logger.info(f"Iniciando processamento do job: {job_id}")

    try:
        # O job já é criado pela API ao receber o upload.
        doc_id = job_id

        # 2. Atualizar status para processing
        update_job_status_db(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            result={"doc_id": doc_id},
            error=None,
        )

        # 3. Processar PDF
        logger.info(f"Processando PDF {len(file_content)} bytes")
        document = _process_document(file_content)

        logger.info(
            f"Processamento concluído: {document.get('total_pages', 0)} páginas"
        )

        # 4. Gerar embeddings se solicitado
        embeddings_generated = False
        if generate_embeddings:
            logger.info("Gerando embeddings...")
            embeddings = _generate_document_embeddings(document)
            embeddings_generated = len(embeddings) > 0
            logger.info(f"Embeddings gerados: {embeddings_generated}")

        # 5. Atualizar status para completed
        result = {
            "doc_id": document.get("doc_id", doc_id),
            "total_pages": document.get("total_pages", 0),
            "processing_time_seconds": kwargs.get("processing_time_seconds", 0),
        }

        update_job_status_db(
            job_id=job_id, status=JobStatus.COMPLETED, result=result, error=None
        )

        # 6. Salvar embeddings no MongoDB
        if embeddings_generated:
            logger.info("Salvando embeddings no MongoDB...")
            _save_embeddings_to_mongodb(job_id, document, embeddings)

        # 7. Enviar webhook se necessário
        if webhook_url:
            logger.info(f"Enviando webhook: {webhook_url}")
            _send_webhook(job_id, JobStatus.COMPLETED, result)

        # 8. Salvar documento no MongoDB
        logger.info("Salvando documento no MongoDB...")
        save_document(
            doc_id=document.get("doc_id", doc_id),
            document=document,
            processing_date=datetime.now(),
        )

        logger.info(f"Job concluído com sucesso: {job_id}")

        return {
            "doc_id": document.get("doc_id", doc_id),
            "total_pages": document.get("total_pages", 0),
            "embeddings_generated": embeddings_generated,
        }

    except Exception as e:
        logger.error(f"Erro ao processar job {job_id}: {e}")

        # Atualizar status para failed
        update_job_status_db(
            job_id=job_id, status=JobStatus.FAILED, result=None, error=str(e)
        )

        # Enviar webhook de erro se necessário
        if webhook_url:
            _send_webhook(job_id, JobStatus.FAILED, {"error": str(e)})

        raise


@celery_app.task(bind=True, name="src.celery_worker.generate_embeddings")
def generate_embeddings(
    self: Task,
    job_id: str,
    texts: list[str],
    doc_id: str,
    page: int | None = None,
    block_id: str | None = None,
    **kwargs,
) -> list[list[float]]:
    """
    Gerar embeddings para textos.

    Args:
        self: Task instance
        job_id: ID do job
        texts: Lista de textos
        doc_id: ID do documento
        page: Página (opcional)
        block_id: ID do bloco (opcional)
        **kwargs: Outros parâmetros

    Returns:
        Lista de embeddings
    """
    logger.info(f"Gerando embeddings para {len(texts)} textos: {job_id}")

    try:
        # Criar cliente de embeddings
        embeddings_client = EmbeddingsClient()

        # Gerar embeddings
        embeddings = embeddings_client.generate_embeddings(texts)

        logger.info(f"Embeddings gerados: {len(embeddings)}")

        # Salvar embeddings no MongoDB
        for i, embedding in enumerate(embeddings):
            page_num = page or 0
            block = block_id or f"p{page_num}_b{i}"

            save_embedding(
                job_id=job_id,
                doc_id=doc_id,
                vector=embedding,
                chunk_size=len(embedding),
            )

        return embeddings

    except EmbeddingError as e:
        logger.error(f"Erro ao gerar embeddings: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao gerar embeddings: {e}")
        raise


@celery_app.task(bind=True, name="src.celery_worker.save_to_mongodb")
def save_to_mongodb(
    self: Task,
    job_id: str,
    doc_id: str,
    document: dict[str, Any],
    embeddings: list[list[float]] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Salvar documento e embeddings no MongoDB.

    Args:
        self: Task instance
        job_id: ID do job
        doc_id: ID do documento
        document: Documento processado
        embeddings: Lista de embeddings (opcional)
        **kwargs: Outros parâmetros

    Returns:
        Resultado do salvamento
    """
    logger.info(f" Salvando documento no MongoDB: {doc_id}")

    try:
        # Salvar documento
        save_document(
            doc_id=doc_id,
            document=document,
            processing_date=kwargs.get("processing_date"),
        )

        # Salvar embeddings
        if embeddings:
            logger.info(f" Salvando {len(embeddings)} embeddings...")
            for i, embedding in enumerate(embeddings):
                page_num = kwargs.get("page", 0)
                block = kwargs.get("block_id", f"p{page_num}_b{i}")

                save_embedding(
                    job_id=job_id,
                    doc_id=doc_id,
                    vector=embedding,
                    chunk_size=len(embedding),
                )

        logger.info(f"Documento salvo com sucesso: {doc_id}")

        return {"success": True, "doc_id": doc_id}

    except Exception as e:
        logger.error(f"Erro ao salvar no MongoDB: {e}")
        raise


@celery_app.task(bind=True, name="src.celery_worker.update_job_status")
def update_job_status(
    self: Task,
    job_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Atualizar status do job.

    Args:
        self: Task instance
        job_id: ID do job
        status: Status atualizado
        result: Resultado do job
        error: Erro do job
        **kwargs: Outros parâmetros

    Returns:
        Status atualizado
    """
    try:
        # Converter status para JobStatus se necessário
        from src.models.mongodb import JobStatus

        status_enum = JobStatus(status)

        # Atualizar status no banco
        update_job_status_db(
            job_id=job_id, status=status_enum, result=result, error=error
        )

        logger.info(f"Status atualizado: {job_id} -> {status}")

        return {"status": status, "job_id": job_id}

    except Exception as e:
        logger.error(f"Erro ao atualizar status: {e}")
        raise


@celery_app.task(bind=True, name="src.celery_worker.cancel_job")
def cancel_job(self: Task, job_id: str, **kwargs) -> dict[str, Any]:
    """
    Cancelar job pendente.

    Args:
        self: Task instance
        job_id: ID do job
        **kwargs: Outros parâmetros

    Returns:
        Resultado do cancelamento
    """
    logger.info(f"Cancelando job: {job_id}")

    try:
        # Atualizar status para cancelled
        update_job_status_db(
            job_id=job_id,
            status=JobStatus.CANCELLED,
            result=None,
            error="Job cancelado pelo usuário",
        )

        # Deletar webhook se existir
        delete_webhook(job_id)

        # Remover job antigo se já estiver completo
        if get_job(job_id):
            delete_old_jobs(job_id, max_age_days=1)

        logger.info(f"Job cancelado: {job_id}")

        return {"success": True, "job_id": job_id}

    except Exception as e:
        logger.error(f"Erro ao cancelar job: {e}")
        raise


@celery_app.task(bind=True, name="src.celery_worker.send_webhook")
def send_webhook(
    self: Task,
    job_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Enviar webhook de notificação.

    Args:
        self: Task instance
        job_id: ID do job
        status: Status do job
        result: Resultado do job
        **kwargs: Outros parâmetros

    Returns:
        Resultado do envio
    """
    # Obter URL do webhook
    webhook_url = os.getenv("DOC_PARSER_WEBHOOK_URL", "")
    webhook_token = os.getenv("DOC_PARSER_WEBHOOK_AUTH_TOKEN", "")

    if not webhook_url:
        logger.warning(f"Nenhuma URL de webhook configurada para job: {job_id}")
        return {"sent": False, "reason": "No webhook URL configured"}

    logger.info(f"Enviando webhook para {webhook_url}: {job_id}")

    try:
        payload = {
            "job_id": job_id,
            "status": status,
            "result": result or {},
            "timestamp": datetime.now().isoformat(),
        }

        import requests

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Authorization": f"Bearer {webhook_token}"},
            timeout=30,
        )

        response.raise_for_status()

        logger.info(f"Webhook enviado com sucesso: {job_id}")

        return {"success": True, "status_code": response.status_code, "job_id": job_id}

    except Exception as e:
        logger.error(f"Erro ao enviar webhook: {e}")
        return {"success": False, "error": str(e), "job_id": job_id}


@celery_app.task(bind=True, name="src.celery_worker.create_job_document")
def create_job_document_task(
    self: Task,
    job_id: str,
    file_content: bytes,
    generate_embeddings: bool = True,
    webhook_url: str | None = None,
    **kwargs,
) -> str:
    """
    Criar documento de job no MongoDB.

    Args:
        self: Task instance
        job_id: ID do job
        file_content: Conteúdo do arquivo
        generate_embeddings: Gerar embeddings
        webhook_url: URL do webhook
        **kwargs: Outros parâmetros

    Returns:
        ID do documento criado
    """

    doc_id = create_job_document_db(
        job_id=job_id,
        file_content=file_content,
        generate_embeddings=generate_embeddings,
        webhook_url=webhook_url,
    )

    logger.info(f"Documento criado: {doc_id}")

    return doc_id


# Funções internas (não são tasks)


def _ensure_project_root_in_syspath() -> None:
    """Garante que a raiz do projeto esteja no sys.path."""
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _process_document(file_content: bytes) -> dict[str, Any]:
    """
    Processar arquivo PDF.

    Args:
        file_content: Conteúdo do arquivo PDF

    Returns:
        Documento processado
    """
    import tempfile

    _ensure_project_root_in_syspath()
    from src.pipeline import DocumentProcessor

    # Criar arquivo temporário
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
        f.write(file_content)
        temp_path = f.name

    try:
        # Criar processor
        processor = DocumentProcessor(use_gpu=True)

        # Processar documento
        document = processor.process_document(temp_path)

        if hasattr(document, "to_dict"):
            return document.to_dict()
        if hasattr(document, "to_json_dict"):
            return document.to_json_dict()
        if isinstance(document, dict):
            return document

        raise TypeError(f"Tipo de documento não suportado: {type(document)}")

    finally:
        # Limpar arquivo temporário
        try:
            os.unlink(temp_path)
        except:
            pass


def _generate_document_embeddings(document: dict[str, Any]) -> list[list[float]]:
    """
    Gerar embeddings para documento.

    Args:
        document: Documento processado

    Returns:
        Lista de embeddings
    """
    from src.embeddings import EmbeddingsClient

    embeddings_client = EmbeddingsClient()

    # Extrair textos para embedding
    texts = []

    # Embedding do documento completo
    doc_text = " ".join(
        page.get("text_content", "") for page in document.get("pages", [])
    )
    if doc_text:
        texts.append(doc_text)

    # Embeddings por página
    for page in document.get("pages", []):
        page_text = page.get("text_content", "")
        if page_text:
            texts.append(page_text)

    # Gerar embeddings
    embeddings = embeddings_client.generate_embeddings(texts)

    return embeddings


def _save_embeddings_to_mongodb(
    job_id: str, document: dict[str, Any], embeddings: list[list[float]]
) -> None:
    """
    Salvar embeddings no MongoDB.

    Args:
        job_id: ID do job
        document: Documento processado
        embeddings: Lista de embeddings
    """
    from src.database import save_embedding

    doc_id = document.get("doc_id", job_id)

    for i, embedding in enumerate(embeddings):
        page_num = 0
        block_id = f"doc_b{i}"

        # Tentar obter página do embedding
        if i < len(document.get("pages", [])):
            page_num = document["pages"][i].get("page", 0)

        save_embedding(
            job_id=job_id,
            doc_id=doc_id,
            vector=embedding,
            chunk_size=len(embedding),
            page=page_num,
            block_id=block_id,
        )


def _send_webhook(job_id: str, status: JobStatus, result: dict[str, Any]) -> None:
    """
    Enviar webhook de notificação.

    Args:
        job_id: ID do job
        status: Status do job
        result: Resultado do job
    """
    from src.celery_worker import send_webhook

    send_webhook.delay(job_id=job_id, status=status.value, result=result)


# Funções de inicialização


def configure_celery():
    """Configurar Celery."""
    celery_app.conf.update(
        task_serializer=CeleryConfig.TASK_SERIALIZER,
        result_serializer=CeleryConfig.RESULT_SERIALIZER,
        accept_content=["pickle", "json"],
        task_track_started=True,
        time_limit=CeleryConfig.TASK_TIME_LIMIT,
        soft_time_limit=CeleryConfig.TASK_SOFT_TIME_LIMIT,
    )


def start_workers(num_workers: int = None):
    """
    Iniciar workers do Celery.

    Args:
        num_workers: Número de workers
    """
    if num_workers is None:
        num_workers = int(os.getenv("DOC_PARSER_CELERY_WORKERS", "2"))

    # Configurar Celery
    configure_celery()

    # Comandos para iniciar workers
    worker_cmd = [
        "celery",
        "-A",
        "src.celery_worker",
        "worker",
        "--loglevel=info",
        "--concurrency",
        str(num_workers),
        "--pool=solo",  # Para garantir que tasks sejam executadas em sequência
        "--without-gossip",
        "--without-mingle",
        "--without-heartbeat",
    ]

    print(" ".join(worker_cmd))
    print("\nInicie os workers com:")
    print("  celery -A src.celery_worker worker --loglevel=info")
    print("  celery -A src.celery_worker flower --port=5555")


if __name__ == "__main__":
    # Modo CLI
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        # Iniciar como worker
        start_workers()
    else:
        # Testar task
        print("Testando task de processamento...")
        # process_pdf_job.delay(
        #     job_id="test-job",
        #     file_content=b"%PDF-test",
        #     generate_embeddings=False
        # )
        print("Task configurada. Use: celery -A src.celery_worker worker")
