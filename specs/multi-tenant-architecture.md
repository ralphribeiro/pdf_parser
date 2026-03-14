# Arquitetura Multi-Tenant — Integração com Legal Stack

## Contexto

O doc parser foi construído como sistema single-tenant. Para integrá-lo à
[legal-stack](/lab/caseiro/legal-stack/), é necessário adicionar isolamento de
dados multi-usuário, autenticação via Authentik (IdP já existente) e automação
de ingestão via Nextcloud + n8n.

O doc parser substituirá o **Paperless-ngx** na stack, assumindo o papel de
processamento e busca semântica de documentos jurídicos.

---

## Stack existente (legal-stack)

| Serviço | Função | Auth |
|---|---|---|
| **Authentik** | SSO / IdP (OIDC + LDAP) | — |
| **Nextcloud** | Colaboração / arquivos (Group Folders) | OIDC via Authentik |
| **OnlyOffice** | Edição de documentos (integrado ao Nextcloud) | JWT com Nextcloud |
| **Planka** | Kanban / gestão de tarefas | OIDC via Authentik |
| **Dolibarr** | ERP/CRM | LDAP via Authentik |
| **n8n** | Automação de workflows | Forward Auth via Authentik |
| **Paperless-ngx** | Gestão documental (EDMS) — **será substituído** | OIDC via Authentik |

Infraestrutura compartilhada: PostgreSQL, MariaDB, Redis, Nginx (proxy reverso).

---

## Modelo de isolamento: workspace por grupo/pasta

### Por que não separar por usuário?

No caso de uso real, documentos jurídicos são compartilhados entre advogados
que trabalham no mesmo processo. Uma pasta no Nextcloud (Group Folder) é
compartilhada por N usuários, e todos precisam acessar os mesmos chunks no RAG.

Separar por `user_id` quebraria esse modelo: João faz upload, mas Maria não
encontraria o documento na busca semântica.

### Entidade de isolamento: `workspace_id`

O `workspace_id` corresponde a um **grupo do Authentik**, que por sua vez é
vinculado a um **Group Folder no Nextcloud**.

```
Authentik (source of truth)
  └── Grupo "processo-123"
       ├── João
       ├── Maria
       └── Pedro

Nextcloud
  └── Group Folder "Processo 123" → vinculado ao grupo "processo-123"

Doc Parser
  └── workspace_id = "processo-123"
  └── Permissão: token OIDC contém groups → verificar se workspace_id ∈ groups
```

O doc parser **nunca gerencia permissões** — apenas verifica se o
`workspace_id` do recurso requisitado está na lista de `groups` do token JWT.

---

## Isolamento por camada de dados

### MongoDB (metadados + documento parseado)

Adicionar campo `workspace_id` em cada documento:

```json
{
  "_id": "69b48d0c...",
  "workspace_id": "processo-123",
  "filename": "contrato.pdf",
  "status": "processed",
  "file_hash": "sha256...",
  "parsed_document": { ... }
}
```

Todas as queries incluem filtro por `workspace_id`:

```python
# Listar documentos do usuário
groups = token.claims["groups"]  # ["processo-123", "societario"]
collection.find({"workspace_id": {"$in": groups}})
```

Índice composto: `{ workspace_id: 1, created_at: -1 }`.

### ChromaDB (chunks vetoriais)

Collection única com filtro por metadata `workspace_id`:

```python
# Indexação
metadata = {
    "document_id": "69b48d0c...",
    "workspace_id": "processo-123",
    "source_file": "contrato.pdf",
    "page_number": 1,
    ...
}

# Busca (cross-workspace: busca em todos os workspaces do usuário)
results = collection.query(
    query_embeddings=[embedding],
    n_results=10,
    where={"workspace_id": {"$in": ["processo-123", "societario"]}},
)
```

**Por que collection única em vez de uma por workspace?**

- Permite busca cross-workspace (ex: "cláusulas de multa em todos os meus
  processos") com um único query
- Volume típico de escritório não justifica a complexidade de múltiplas
  collections
- O filtro por metadata `workspace_id` já garante isolamento adequado

### Redis (fila de jobs)

Prefixo de workspace no key e no Job schema:

```
job:processo-123:159bdb0f-...
```

---

## Autenticação: Authentik OIDC

Mesmo padrão já usado na legal-stack por Paperless, Planka e Nextcloud.

### Configuração no Authentik

1. Criar Application "doc-parser"
2. Criar Provider OIDC com:
   - Redirect URI: `https://{HOST_IP}:{DOC_PARSER_PORT}/auth/callback`
   - Scopes: `openid profile email groups`
3. Obter `CLIENT_ID` e `CLIENT_SECRET`

### Middleware FastAPI

O token JWT do Authentik contém:

```json
{
  "sub": "user-uuid",
  "preferred_username": "joao",
  "email": "joao@escritorio.com",
  "groups": ["processo-123", "societario", "admin"]
}
```

O middleware:

1. Valida o token JWT via OIDC discovery do Authentik
2. Extrai `sub` (user_id) e `groups` (workspaces)
3. Injeta no request state para uso nos endpoints

Endpoints filtram dados por `workspace_id ∈ user.groups`.

---

## Fluxo de ingestão: Nextcloud → n8n → Doc Parser

### Pré-requisitos

- App **Group Folders** instalado no Nextcloud
- Service account `svc-n8n` já configurado na legal-stack
- Grupo Authentik vinculado ao Group Folder

### Fluxo no n8n

```
┌─────────────────────────────────────────────────────────────┐
│ n8n Workflow: "Processar Documento Jurídico"                │
│                                                             │
│ 1. Trigger: polling da pasta /Processo-123/ no Nextcloud    │
│    (ou webhook do Nextcloud Activity API)                   │
│                                                             │
│ 2. Detecta novo PDF                                         │
│                                                             │
│ 3. Nextcloud: adiciona tag "processando" ao arquivo         │
│    (via WebDAV PROPPATCH)                                   │
│                                                             │
│ 4. Nextcloud: download do PDF                               │
│                                                             │
│ 5. Resolve workspace_id:                                    │
│    - Group Folder name → "processo-123"                     │
│    - (via Nextcloud API: /ocs/v1.php/cloud/groups)          │
│                                                             │
│ 6. HTTP Request: POST /api/jobs                             │
│    - Body: multipart (PDF)                                  │
│    - Header: X-Workspace-Id: processo-123                   │
│    - Auth: Bearer token (service account ou API key)        │
│                                                             │
│ 7. Polling: GET /api/jobs/{job_id}                          │
│    - Loop até status = "uploaded"                           │
│                                                             │
│ 8. Nextcloud: muda tag para "processado"                    │
│    (via WebDAV PROPPATCH)                                   │
└─────────────────────────────────────────────────────────────┘
```

### Tags no Nextcloud

| Tag | Significado |
|---|---|
| `processando` | PDF enviado ao doc parser, aguardando OCR/indexação |
| `processado` | Chunks indexados no ChromaDB, documento parseado no MongoDB |
| `erro` | Falha no processamento (verificar logs) |

---

## Interface de chat (RAG)

O doc parser não precisa implementar uma interface de chat. Um serviço externo
consome os dados via MCP:

### MCP Servers necessários

| MCP Server | Função |
|---|---|
| **[chroma-mcp](https://github.com/chroma-core/chroma-mcp)** | Busca semântica nos chunks (com filtro por `workspace_id`) |
| **[mongodb-mcp-server](https://github.com/mongodb-js/mongodb-mcp-server)** | Acesso a metadados, status, documento parseado |

### Opções de interface de chat

| Serviço | OIDC nativo | MCP | ChromaDB externo | Notas |
|---|---|---|---|---|
| **Open WebUI** | Sim | Sim (v0.6.31+) | Via MCP | Mais popular, 90k+ stars |
| **LibreChat** | Sim (OIDC/LDAP/SAML) | Sim | Via MCP | Melhor para enterprise multi-user |
| **AnythingLLM** | Não (proxy auth) | Sim | Nativo (connector) | Mais simples, workspace-based |

Para a legal-stack com Authentik, **Open WebUI** ou **LibreChat** são as
escolhas naturais pela compatibilidade OIDC direta.

### Isolamento no chat

O serviço de chat autentica via Authentik (mesmo IdP). Os MCP servers recebem
o contexto do usuário (via headers ou config) e filtram por `workspace_id`.

---

## Arquitetura completa

```
                         Authentik
                      (grupos = workspaces)
                    ┌────────┴────────┐
                    ▼                 ▼
               Nextcloud          Chat UI
             Group Folders    (Open WebUI / LibreChat)
                    │                 │
                    │ token OIDC      │ token OIDC → groups
                    │                 │
                n8n │                 │ MCP tools
         (detecta upload)            │
                    │                 │
                    ▼                 ▼
            ┌────────────────────────────────┐
            │        Doc Parser API          │
            │                                │
            │  middleware OIDC:              │
            │  extrai groups do token        │
            │  → workspace_id               │
            └──────────┬─────────────────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         MongoDB    ChromaDB    Redis
         filtro:    metadata:   prefix:
       workspace_id workspace_id workspace_id
```

---

## O que muda no código atual

### Novos conceitos

- `workspace_id`: campo em Job, Document (MongoDB), chunk metadata (ChromaDB)
- Middleware OIDC: validação de token, extração de groups
- Filtro por workspace em todos os endpoints de leitura

### Endpoints afetados

| Endpoint | Mudança |
|---|---|
| `POST /api/jobs` | Recebe `workspace_id` (header ou body) |
| `GET /api/jobs` | Filtra por workspaces do token |
| `GET /api/documents` | Filtra por workspaces do token |
| `GET /api/documents/{id}` | Verifica acesso ao workspace do documento |
| `POST /api/search` | Adiciona `where: {workspace_id: {$in: groups}}` |
| `POST /api/agent/search` | Idem ao search |

### Schemas afetados

- `Job`: adicionar `workspace_id: str`
- `DocumentStore.create_document()`: aceitar `workspace_id`
- `ChromaVectorStore.upsert_chunks()`: incluir `workspace_id` no metadata
- `ChromaVectorStore.query()`: filtrar por `workspace_id`

### Novos componentes

- `services/auth/`: middleware OIDC (validação JWT, extração de claims)
- Configuração: `OIDC_ISSUER_URL`, `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`

---

## Ordem de implementação sugerida

1. **Auth middleware** — OIDC com Authentik, extração de groups
2. **workspace_id no MongoDB** — campo + índice + filtro nos endpoints
3. **workspace_id no ChromaDB** — metadata nos chunks + filtro nas queries
4. **workspace_id nos Jobs** — campo no schema + filtro na listagem
5. **Endpoint de ingestão** — aceitar workspace_id no upload
6. **Integração n8n** — workflow Nextcloud → Doc Parser
7. **Chat UI** — deploy de Open WebUI/LibreChat com MCP servers
8. **Substituir Paperless** — remover do compose, adicionar doc parser

---

## Referências

- [Authentik OIDC Provider](https://docs.goauthentik.io/docs/providers/oauth2/)
- [Nextcloud Group Folders](https://apps.nextcloud.com/apps/groupfolders)
- [chroma-mcp](https://github.com/chroma-core/chroma-mcp)
- [mongodb-mcp-server](https://github.com/mongodb-js/mongodb-mcp-server)
- [Open WebUI MCP](https://docs.openwebui.com/features/extensibility/mcp/)
- [LibreChat](https://www.librechat.ai/)
- [n8n Nextcloud integration](https://n8n.io/integrations/nextcloud/)
