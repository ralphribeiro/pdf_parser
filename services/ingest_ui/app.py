"""
FastAPI application for the ingest UI.

Serves HTML pages for: dashboard, uploading PDFs, tracking job status,
semantic search, agent search, and document listing/detail.
Shares a JobStore with the ingest API for direct state access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from services.ingest_api.store import JobStore

_STATUS_LABELS = {
    "queued": ("Na fila", "info"),
    "processing": ("Processando\u2026", "warning"),
    "uploaded": ("Conclu\u00eddo", "success"),
    "failed": ("Falhou", "danger"),
}

_DOC_STATUS_LABELS = {
    "pending": ("Pendente", "info"),
    "processing": ("Processando\u2026", "warning"),
    "processed": ("Processado", "success"),
    "failed": ("Falhou", "danger"),
}


def create_ui_app(
    upload_dir: Path | None = None,
    store: JobStore | None = None,
    document_store: Any = None,
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
    app.state.document_store = document_store

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        return HTMLResponse(_render_dashboard())

    @app.get("/upload", response_class=HTMLResponse)
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
            msg = "Conte\u00fado inv\u00e1lido \u2014 o arquivo n\u00e3o \u00e9 um PDF."
            return HTMLResponse(
                _render_upload_page(error=msg),
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
                _render_error_page("Job n\u00e3o encontrado."), status_code=404
            )
        return HTMLResponse(_render_status_page(job))

    @app.get("/search", response_class=HTMLResponse)
    async def search_page(request: Request) -> HTMLResponse:
        return HTMLResponse(_render_search_page())

    @app.get("/agent", response_class=HTMLResponse)
    async def agent_page(request: Request) -> HTMLResponse:
        return HTMLResponse(_render_agent_page())

    @app.get("/jobs", response_class=HTMLResponse)
    async def jobs_list_page(request: Request) -> HTMLResponse:
        store: JobStore = request.app.state.store
        jobs = store.list_all(limit=50)
        return HTMLResponse(_render_jobs_list_page(jobs))

    @app.get("/documents", response_class=HTMLResponse)
    async def documents_list_page(request: Request) -> HTMLResponse:
        doc_store = request.app.state.document_store
        if doc_store is None:
            msg = (
                "Document store n\u00e3o configurado"
                " \u2014 servi\u00e7o indispon\u00edvel."
            )
            return HTMLResponse(_render_info_page("Documentos", msg))
        docs = doc_store.list_documents(limit=50)
        return HTMLResponse(_render_documents_list_page(docs))

    @app.get("/documents/{document_id}", response_class=HTMLResponse)
    async def document_detail_page(document_id: str, request: Request) -> HTMLResponse:
        doc_store = request.app.state.document_store
        if doc_store is None:
            msg = (
                "Document store n\u00e3o configurado"
                " \u2014 servi\u00e7o indispon\u00edvel."
            )
            return HTMLResponse(_render_info_page("Documento", msg))
        doc = doc_store.get_document(document_id)
        if doc is None:
            return HTMLResponse(
                _render_error_page("Documento n\u00e3o encontrado."), status_code=404
            )
        doc["_id"] = str(doc["_id"])
        return HTMLResponse(_render_document_detail_page(doc))


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

_STYLE = """
<style>
  :root { --bg: #f8f9fa; --fg: #212529; --accent: #0d6efd;
          --card: #fff; --border: #dee2e6; --radius: .5rem;
          --nav-bg: #212529; --nav-fg: #f8f9fa; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif;
         background: var(--bg); color: var(--fg); }
  nav { background: var(--nav-bg); color: var(--nav-fg);
        padding: .8rem 1.5rem; display: flex; align-items: center;
        gap: 1.5rem; flex-wrap: wrap; }
  nav a { color: var(--nav-fg); text-decoration: none; font-size: .95rem;
          opacity: .85; transition: opacity .15s; }
  nav a:hover { opacity: 1; }
  nav .brand { font-weight: 700; font-size: 1.1rem; opacity: 1;
               margin-right: 1rem; }
  .main { max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: var(--radius); padding: 2rem;
          box-shadow: 0 .125rem .25rem rgba(0,0,0,.08);
          margin-bottom: 1.5rem; }
  h1 { font-size: 1.4rem; margin-bottom: 1.2rem; }
  h2 { font-size: 1.15rem; margin-bottom: .8rem; color: #495057; }
  label { display: block; margin-bottom: .4rem; font-weight: 500; }
  input[type=file], input[type=text], textarea {
    display: block; width: 100%; margin-bottom: 1rem;
    padding: .5rem; border: 1px solid var(--border);
    border-radius: var(--radius); font-size: .95rem; }
  textarea { min-height: 4rem; resize: vertical; }
  button, .btn { background: var(--accent); color: #fff; border: none;
           padding: .6rem 1.4rem; border-radius: var(--radius);
           font-size: 1rem; cursor: pointer; text-decoration: none;
           display: inline-block; }
  button:hover, .btn:hover { opacity: .9; }
  .error { background: #f8d7da; color: #842029; padding: .8rem;
           border-radius: var(--radius); margin-bottom: 1rem; }
  .info-box { background: #cff4fc; color: #055160; padding: .8rem;
              border-radius: var(--radius); margin-bottom: 1rem; }
  .badge { display: inline-block; padding: .3rem .7rem;
           border-radius: var(--radius); font-weight: 600;
           font-size: .85rem; }
  .badge.info { background: #cff4fc; color: #055160; }
  .badge.warning { background: #fff3cd; color: #664d03; }
  .badge.success { background: #d1e7dd; color: #0f5132; }
  .badge.danger  { background: #f8d7da; color: #842029; }
  .detail { margin-top: 1rem; font-size: .9rem; color: #6c757d; }
  .error-box { margin-top: 1rem; background: #f8d7da; color: #842029;
               padding: .8rem; border-radius: var(--radius);
               white-space: pre-wrap; font-size: .85rem; }
  a { color: var(--accent); }
  table { width: 100%; border-collapse: collapse; margin-top: .5rem; }
  th, td { text-align: left; padding: .6rem .8rem;
           border-bottom: 1px solid var(--border); font-size: .9rem; }
  th { background: #f1f3f5; font-weight: 600; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1rem; }
  .grid .card { text-align: center; padding: 1.5rem; }
  .grid .card h2 { margin-bottom: .5rem; }
  .empty { color: #6c757d; font-style: italic; padding: 1rem 0; }
  pre.json { background: #f1f3f5; padding: 1rem; border-radius: var(--radius);
             overflow-x: auto; font-size: .8rem; max-height: 500px; }
  .md-body h1,.md-body h2,.md-body h3 {
    margin: 1rem 0 .5rem; }
  .md-body p { margin: .5rem 0; line-height: 1.6; }
  .md-body ul,.md-body ol { margin: .5rem 0 .5rem 1.5rem; }
  .md-body li { margin: .25rem 0; }
  .md-body code { background: #f1f3f5; padding: .15rem .4rem;
                  border-radius: 3px; font-size: .88em; }
  .md-body pre { background: #f1f3f5; padding: 1rem;
                 border-radius: var(--radius); overflow-x: auto;
                 margin: .5rem 0; }
  .md-body pre code { background: none; padding: 0; }
  .md-body blockquote { border-left: 3px solid var(--accent);
                        margin: .5rem 0; padding: .5rem 1rem;
                        color: #6c757d; }
  .meta { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem;
          font-size: .9rem; color: #6c757d; }
</style>
"""

_NAV = (
    "<nav>"
    '<a class="brand" href="/">Doc Parser</a>'
    '<a href="/upload">Upload</a>'
    '<a href="/jobs">Jobs</a>'
    '<a href="/documents">Documentos</a>'
    '<a href="/search">Busca</a>'
    '<a href="/agent">Agente</a>'
    "</nav>"
)


def _page(title: str, body: str, *, head_extra: str = "") -> str:
    return (
        "<!DOCTYPE html>"
        f'<html lang="pt-BR"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title>{_STYLE}{head_extra}</head>"
        f"<body>{_NAV}<div class='main'>{body}</div></body></html>"
    )


# -- Dashboard -------------------------------------------------------------


def _render_dashboard() -> str:
    cards = (
        '<div class="grid">'
        '<a href="/upload" style="text-decoration:none;color:inherit">'
        '<div class="card"><h2>Upload</h2><p>Enviar PDF</p></div></a>'
        '<a href="/jobs" style="text-decoration:none;color:inherit">'
        '<div class="card"><h2>Jobs</h2><p>Acompanhar processamento</p></div></a>'
        '<a href="/documents" style="text-decoration:none;color:inherit">'
        '<div class="card"><h2>Documentos</h2><p>Visualizar documentos</p></div></a>'
        '<a href="/search" style="text-decoration:none;color:inherit">'
        '<div class="card"><h2>Busca</h2><p>Busca sem\u00e2ntica</p></div></a>'
        '<a href="/agent" style="text-decoration:none;color:inherit">'
        '<div class="card"><h2>Agente</h2><p>Busca enriquecida</p></div></a>'
        "</div>"
    )
    return _page("Dashboard", f"<h1>Doc Parser</h1>{cards}")


# -- Upload ----------------------------------------------------------------


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
    return _page(
        "Upload",
        f'<div class="card"><h1>Enviar documento</h1>{form}</div>',
    )


# -- Job status ------------------------------------------------------------


def _render_status_page(job: Any) -> str:
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
        f'<a href="/jobs">\u2190 Todos os jobs</a></p>'
        f"</div>"
    )
    return _page(f"Status \u2014 {job.filename}", body, head_extra=head_extra)


# -- Search ----------------------------------------------------------------


def _render_search_page() -> str:
    form = (
        '<form id="search-form">'
        '<label for="query">Consulta</label>'
        '<input type="text" id="query" name="query" '
        'placeholder="Digite sua busca\u2026" required>'
        '<label for="n_results">M\u00e1x. resultados</label>'
        '<input type="text" id="n_results" name="n_results" value="10">'
        '<button type="submit">Buscar</button>'
        "</form>"
        '<div id="results"></div>'
        "<script>"
        "document.getElementById('search-form').addEventListener('submit',async e=>{"
        "e.preventDefault();"
        "const q=document.getElementById('query').value;"
        "const n=document.getElementById('n_results').value||10;"
        "const r=document.getElementById('results');"
        "r.innerHTML='<p>Buscando\u2026</p>';"
        "try{"
        "const resp=await fetch('/api/search',{method:'POST',"
        "headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({query:q,n_results:parseInt(n)})});"
        "const data=await resp.json();"
        "if(!resp.ok){r.innerHTML='<div class=\"error\">'"
        "+data.detail+'</div>';return;}"
        "if(!data.results.length){"
        "r.innerHTML='<p class=\"empty\">Nenhum resultado.</p>';return;}"
        "let h='<table><tr><th>Score</th><th>Texto</th>"
        "<th>Doc</th><th>P\u00e1g.</th></tr>';"
        "data.results.forEach(s=>{"
        "h+='<tr><td>'+s.similarity.toFixed(3)+'</td><td>'+s.text.substring(0,200)+'</td>'"
        "+'<td><a href=\"/documents/'+s.document_id+'\">'+s.source_file+'</a></td>'"
        "+'<td>'+s.page_number+'</td></tr>';});"
        "h+='</table><p class=\"detail\">'+data.total_matches+' resultados em '"
        "+data.processing_time_ms+'ms</p>';"
        "r.innerHTML=h;"
        "}catch(err){r.innerHTML='<div class=\"error\">'+err.message+'</div>';}"
        "});"
        "</script>"
    )
    return _page(
        "Busca Sem\u00e2ntica",
        f'<div class="card"><h1>Busca Sem\u00e2ntica</h1>{form}</div>',
    )


# -- Agent -----------------------------------------------------------------


def _render_agent_page() -> str:
    form = (
        '<form id="agent-form">'
        '<label for="query">Consulta</label>'
        '<input type="text" id="query" name="query" '
        'placeholder="Pergunte ao agente\u2026" required>'
        '<button type="submit">Perguntar</button>'
        "</form>"
        '<div id="results"></div>'
        '<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js">'
        "</script>"
        "<script>"
        "document.getElementById('agent-form')"
        ".addEventListener('submit',async e=>{"
        "e.preventDefault();"
        "const q=document.getElementById('query').value;"
        "const r=document.getElementById('results');"
        "r.innerHTML='<p>Processando\u2026 isso pode levar"
        " alguns segundos.</p>';"
        "try{"
        "const resp=await fetch('/api/agent/search',"
        "{method:'POST',"
        "headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({query:q})});"
        "const data=await resp.json();"
        "if(!resp.ok){"
        'r.innerHTML=\'<div class="error">'
        "'+data.detail+'</div>';return;}"
        "const md=marked.parse(data.answer||'');"
        'let h=\'<div class="card">'
        "<h2>Resposta</h2>"
        "<div class=\"md-body\">'+md+'</div>';"
        "if(data.sources.length){"
        'h+=\'<h2 style="margin-top:1rem">'
        "Fontes</h2><table>';"
        "h+='<tr><th>Documento</th>"
        "<th>P\u00e1g.</th><th>Trecho</th></tr>';"
        "data.sources.forEach(s=>{"
        "h+='<tr><td>'+s.filename+'</td>"
        "<td>'+s.page+'</td>'"
        "+'<td>'+s.text.substring(0,150)"
        "+'</td></tr>';});"
        "h+='</table>';}"
        'h+=\'<p class="detail">'
        "'+data.iterations+' itera\u00e7\u00f5es, '"
        "+data.processing_time_ms+'ms</p></div>';"
        "r.innerHTML=h;"
        "}catch(err){"
        'r.innerHTML=\'<div class="error">'
        "'+err.message+'</div>';}"
        "});"
        "</script>"
    )
    return _page(
        "Agente de Busca",
        f'<div class="card"><h1>Agente de Busca</h1>{form}</div>',
    )


# -- Jobs list -------------------------------------------------------------


def _render_jobs_list_page(jobs: list) -> str:
    if not jobs:
        body = (
            '<div class="card"><h1>Jobs</h1>'
            '<p class="empty">Nenhum job encontrado.</p></div>'
        )
        return _page("Jobs", body)

    rows = ""
    for job in jobs:
        label, badge_cls = _STATUS_LABELS.get(job.status, (job.status, "info"))
        rows += (
            f"<tr>"
            f'<td><a href="/jobs/{job.job_id}">{job.filename}</a></td>'
            f'<td><span class="badge {badge_cls}">{label}</span></td>'
            f"<td>{job.created_at:%Y-%m-%d %H:%M}</td>"
            f"</tr>"
        )
    table = (
        "<table>"
        "<tr><th>Arquivo</th><th>Status</th><th>Criado em</th></tr>"
        f"{rows}</table>"
    )
    return _page("Jobs", f'<div class="card"><h1>Jobs</h1>{table}</div>')


# -- Documents list --------------------------------------------------------


def _render_documents_list_page(docs: list[dict]) -> str:
    if not docs:
        body = (
            '<div class="card"><h1>Documentos</h1>'
            '<p class="empty">Nenhum documento encontrado.</p></div>'
        )
        return _page("Documentos", body)

    rows = ""
    for doc in docs:
        doc_id = doc.get("_id", "")
        label, badge_cls = _DOC_STATUS_LABELS.get(
            doc.get("status", ""), (doc.get("status", ""), "info")
        )
        pages = doc.get("total_pages") or "\u2014"
        rows += (
            f"<tr>"
            f'<td><a href="/documents/{doc_id}">{doc.get("filename", "")}</a></td>'
            f'<td><span class="badge {badge_cls}">{label}</span></td>'
            f"<td>{pages}</td>"
            f"</tr>"
        )
    table = (
        "<table>"
        "<tr><th>Arquivo</th><th>Status</th><th>P\u00e1ginas</th></tr>"
        f"{rows}</table>"
    )
    return _page("Documentos", f'<div class="card"><h1>Documentos</h1>{table}</div>')


# -- Document detail -------------------------------------------------------


def _render_document_detail_page(doc: dict) -> str:
    import json

    status = doc.get("status", "")
    label, badge_cls = _DOC_STATUS_LABELS.get(status, (status, "info"))
    pages = doc.get("total_pages") or "\u2014"
    size_kb = (doc.get("file_size") or 0) / 1024

    meta = (
        f'<div class="meta">'
        f'<span>Status: <span class="badge {badge_cls}">{label}</span></span>'
        f"<span>P\u00e1ginas: {pages}</span>"
        f"<span>Tamanho: {size_kb:.1f} KB</span>"
        f"</div>"
    )

    parsed = doc.get("parsed_document")
    parsed_html = ""
    if parsed:
        parsed_json = json.dumps(parsed, indent=2, ensure_ascii=False, default=str)
        parsed_html = (
            f'<h2>Documento parseado</h2><pre class="json">{parsed_json}</pre>'
        )

    body = (
        f'<div class="card">'
        f"<h1>{doc.get('filename', 'Documento')}</h1>"
        f"{meta}"
        f'<p class="detail">ID: <code>{doc.get("_id", "")}</code></p>'
        f"{parsed_html}"
        f'<p class="detail" style="margin-top:1.5rem">'
        f'<a href="/documents">\u2190 Todos os documentos</a></p>'
        f"</div>"
    )
    return _page(f"Documento \u2014 {doc.get('filename', '')}", body)


# -- Utility pages ---------------------------------------------------------


def _render_error_page(message: str) -> str:
    body = (
        f'<div class="card">'
        f"<h1>Erro</h1><p>{message}</p>"
        f'<p class="detail" style="margin-top:1rem">'
        f'<a href="/">\u2190 Voltar</a></p></div>'
    )
    return _page("Erro", body)


def _render_info_page(title: str, message: str) -> str:
    body = (
        f'<div class="card">'
        f"<h1>{title}</h1>"
        f'<div class="info-box">{message}</div>'
        f'<p class="detail" style="margin-top:1rem">'
        f'<a href="/">\u2190 Voltar</a></p></div>'
    )
    return _page(title, body)
