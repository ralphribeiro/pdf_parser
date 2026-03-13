"""
FastAPI application for the ingest UI.

Serves HTML pages for uploading PDFs and tracking job status.
Shares a JobStore with the ingest API for direct state access.
"""

from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from services.ingest_api.store import JobStore

_STATUS_LABELS = {
    "queued": ("Na fila", "info"),
    "processing": ("Processando…", "warning"),
    "uploaded": ("Concluído", "success"),
    "failed": ("Falhou", "danger"),
}


def create_ui_app(
    upload_dir: Path | None = None,
    store: JobStore | None = None,
) -> FastAPI:
    """Factory with dependency injection for testing."""
    if store is None:
        store = JobStore()
    if upload_dir is None:
        upload_dir = Path("data")

    upload_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Ingest UI", version="0.1.0")
    app.state.store = store
    app.state.upload_dir = upload_dir

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:

    @app.get("/", response_class=HTMLResponse)
    async def upload_page(request: Request) -> HTMLResponse:
        return HTMLResponse(_render_upload_page())

    @app.post("/upload", response_model=None)
    async def handle_upload(
        request: Request,
        file: UploadFile = File(...),
    ) -> HTMLResponse | RedirectResponse:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            return HTMLResponse(
                _render_upload_page(error="O arquivo deve ser um PDF."),
                status_code=400,
            )

        content = await file.read()
        if not content or content[:5] != b"%PDF-":
            return HTMLResponse(
                _render_upload_page(
                    error="Conteúdo inválido — o arquivo não é um PDF."
                ),
                status_code=400,
            )

        store: JobStore = request.app.state.store
        job = store.create(filename=file.filename)

        dest: Path = request.app.state.upload_dir / f"{job.job_id}.pdf"
        dest.write_bytes(content)

        return RedirectResponse(url=f"/jobs/{job.job_id}", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def status_page(job_id: str, request: Request) -> HTMLResponse:
        store: JobStore = request.app.state.store
        job = store.get(job_id)
        if job is None:
            return HTMLResponse(
                _render_error_page("Job não encontrado."), status_code=404
            )
        return HTMLResponse(_render_status_page(job))


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

_STYLE = """
<style>
  :root { --bg: #f8f9fa; --fg: #212529; --accent: #0d6efd;
          --card: #fff; --border: #dee2e6; --radius: .5rem; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif;
         background: var(--bg); color: var(--fg);
         display: flex; justify-content: center; padding: 2rem; }
  .container { max-width: 540px; width: 100%; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: var(--radius); padding: 2rem;
          box-shadow: 0 .125rem .25rem rgba(0,0,0,.08); }
  h1 { font-size: 1.4rem; margin-bottom: 1.2rem; }
  label { display: block; margin-bottom: .4rem; font-weight: 500; }
  input[type=file] { display: block; width: 100%; margin-bottom: 1rem;
                     padding: .5rem; border: 1px solid var(--border);
                     border-radius: var(--radius); }
  button { background: var(--accent); color: #fff; border: none;
           padding: .6rem 1.4rem; border-radius: var(--radius);
           font-size: 1rem; cursor: pointer; }
  button:hover { opacity: .9; }
  .error { background: #f8d7da; color: #842029; padding: .8rem;
           border-radius: var(--radius); margin-bottom: 1rem; }
  .badge { display: inline-block; padding: .3rem .7rem;
           border-radius: var(--radius); font-weight: 600;
           font-size: .9rem; }
  .badge.info { background: #cff4fc; color: #055160; }
  .badge.warning { background: #fff3cd; color: #664d03; }
  .badge.success { background: #d1e7dd; color: #0f5132; }
  .badge.danger  { background: #f8d7da; color: #842029; }
  .detail { margin-top: 1rem; font-size: .9rem; color: #6c757d; }
  .error-box { margin-top: 1rem; background: #f8d7da; color: #842029;
               padding: .8rem; border-radius: var(--radius);
               white-space: pre-wrap; font-size: .85rem; }
  a { color: var(--accent); }
</style>
"""


def _page(title: str, body: str, *, head_extra: str = "") -> str:
    return (
        "<!DOCTYPE html>"
        f'<html lang="pt-BR"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title>{_STYLE}{head_extra}</head>"
        f"<body><div class='container'>{body}</div></body></html>"
    )


def _render_upload_page(*, error: str | None = None) -> str:
    error_html = f'<div class="error">{error}</div>' if error else ""
    form = (
        f"{error_html}"
        '<form action="/upload" method="post" enctype="multipart/form-data">'
        '<label for="file">Selecione um PDF</label>'
        '<input type="file" id="file" name="file" accept=".pdf">'
        '<button type="submit">Enviar</button>'
        "</form>"
    )
    return _page("Upload", f'<div class="card"><h1>Enviar documento</h1>{form}</div>')


def _render_status_page(job) -> str:
    label, badge_cls = _STATUS_LABELS.get(job.status, (job.status, "info"))
    is_pending = job.status in ("queued", "processing")
    head_extra = '<meta http-equiv="refresh" content="3">' if is_pending else ""

    error_html = ""
    if job.error_message:
        error_html = f'<div class="error-box">{job.error_message}</div>'

    body = (
        f'<div class="card">'
        f"<h1>{job.filename}</h1>"
        f'<p>Status: <span class="badge {badge_cls}"'
        f' data-status="{job.status}">{label}</span></p>'
        f'<p class="detail">Job: <code>{job.job_id}</code></p>'
        f"{error_html}"
        f'<p class="detail" style="margin-top:1.5rem">'
        f'<a href="/">← Enviar outro documento</a></p>'
        f"</div>"
    )
    return _page(f"Status — {job.filename}", body, head_extra=head_extra)


def _render_error_page(message: str) -> str:
    body = (
        f'<div class="card">'
        f"<h1>Erro</h1><p>{message}</p>"
        f'<p class="detail" style="margin-top:1rem">'
        f'<a href="/">← Voltar</a></p></div>'
    )
    return _page("Erro", body)
