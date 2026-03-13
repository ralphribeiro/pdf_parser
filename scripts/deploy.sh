#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Script de deploy do Doc Parser
#
# Automatiza o ciclo completo: validação do host, build, deploy e healthcheck.
#
# Uso:
#   ./scripts/deploy.sh              # Deploy completo (build + up)
#   ./scripts/deploy.sh --rebuild    # Force rebuild sem cache
#   ./scripts/deploy.sh --down       # Parar todos os serviços
#   ./scripts/deploy.sh --status     # Verificar status dos serviços
#   ./scripts/deploy.sh --logs       # Mostrar logs (follow)
#   ./scripts/deploy.sh --reset      # Parar, limpar volumes e redeployar
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

COMPOSE_FILE="docker-compose.services.yml"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
ENV_FILE="${PROJECT_DIR}/.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
header(){ echo -e "\n${BOLD}═══ $* ═══${NC}"; }

dc() {
    docker compose -f "${PROJECT_DIR}/${COMPOSE_FILE}" "$@"
}

# ---------------------------------------------------------------------------
# Pré-requisitos do host
# ---------------------------------------------------------------------------

check_prerequisites() {
    header "1. Verificando pré-requisitos"

    # Docker
    if command -v docker &>/dev/null; then
        ok "Docker instalado: $(docker --version | head -1)"
    else
        fail "Docker não encontrado. Instale: https://docs.docker.com/engine/install/"
        exit 1
    fi

    # Docker Compose (v2)
    if docker compose version &>/dev/null; then
        ok "Docker Compose v2: $(docker compose version --short)"
    else
        fail "Docker Compose v2 não encontrado."
        exit 1
    fi

    # GPU AMD / ROCm (worker precisa para OCR com docTR)
    if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
        ok "Dispositivos GPU AMD encontrados (/dev/kfd, /dev/dri)"
    else
        warn "Dispositivos GPU AMD não encontrados."
        warn "O worker OCR requer GPU AMD com ROCm para docTR (PyTorch)."
        warn "Sem GPU, o worker não vai iniciar. Considere usar Tesseract (CPU) como fallback."
    fi

    # Grupos video/render
    if id -nG | grep -qw video && id -nG | grep -qw render; then
        ok "Usuário nos grupos video e render"
    else
        warn "Usuário não está nos grupos 'video' e/ou 'render'."
        warn "Execute: sudo usermod -aG video,render \$USER && newgrp video"
    fi

    # Arquivo .env
    if [ -f "${ENV_FILE}" ]; then
        ok "Arquivo .env encontrado"
    else
        warn "Arquivo .env não encontrado em ${ENV_FILE}"
        warn "Crie com pelo menos: EMBEDDING_API_URL=<url-do-servico-de-embeddings>"
    fi
}

# ---------------------------------------------------------------------------
# Validar variáveis de ambiente obrigatórias
# ---------------------------------------------------------------------------

check_env_vars() {
    header "2. Verificando variáveis de ambiente"

    local missing=0

    # EMBEDDING_API_URL é necessário para busca semântica
    if [ -f "${ENV_FILE}" ]; then
        # shellcheck disable=SC1090
        source "${ENV_FILE}" 2>/dev/null || true
    fi

    if [ -z "${EMBEDDING_API_URL:-}" ]; then
        warn "EMBEDDING_API_URL não definida."
        warn "A busca semântica não funcionará sem um serviço de embeddings."
        warn "Exemplo: EMBEDDING_API_URL=http://192.168.0.25:8080"
        missing=1
    else
        ok "EMBEDDING_API_URL=${EMBEDDING_API_URL}"
    fi

    EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding}"
    ok "EMBEDDING_MODEL=${EMBEDDING_MODEL}"

    # LLM (agente AI)
    if [ -z "${LLM_API_URL:-}" ]; then
        warn "LLM_API_URL não definida."
        warn "O agente de busca enriquecida (/api/agent/search) ficará desabilitado."
    else
        ok "LLM_API_URL=${LLM_API_URL}"
        ok "LLM_MODEL=${LLM_MODEL:-Qwen3.5-9B-Q4_K_M}"
    fi

    if [ $missing -eq 1 ]; then
        warn "Variáveis opcionais ausentes. O deploy continuará, mas algumas features podem não funcionar."
    fi
}

# ---------------------------------------------------------------------------
# Criar diretórios necessários
# ---------------------------------------------------------------------------

prepare_directories() {
    header "3. Preparando diretórios"

    mkdir -p "${DATA_DIR}"
    ok "Diretório de dados: ${DATA_DIR}"

    mkdir -p "${PROJECT_DIR}/output"
    ok "Diretório de output: ${PROJECT_DIR}/output"
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

build_images() {
    local no_cache="${1:-}"
    header "4. Build das imagens Docker"

    info "Base image: rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.7.1"
    info "Isso pode levar alguns minutos na primeira vez..."

    if [ "${no_cache}" = "--no-cache" ]; then
        info "Build sem cache (--no-cache)"
        dc build --no-cache
    else
        dc build
    fi

    ok "Imagens construídas com sucesso"
}

# ---------------------------------------------------------------------------
# Deploy (up)
# ---------------------------------------------------------------------------

start_services() {
    header "5. Iniciando serviços"

    info "Serviços que serão iniciados:"
    echo "  - redis       : Cache e fila de jobs (porta 6379)"
    echo "  - mongodb     : Document store persistente (porta 27017)"
    echo "  - chromadb    : Banco vetorial para busca semântica (porta 8000)"
    echo "  - api         : FastAPI — API REST + UI web (porta 8090)"
    echo "  - worker      : OCR worker com GPU AMD (sem porta exposta)"
    echo ""

    dc up -d

    ok "Todos os serviços iniciados em background"
}

# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------

wait_for_health() {
    header "6. Aguardando serviços ficarem saudáveis"

    # Redis
    info "Verificando Redis..."
    for i in $(seq 1 10); do
        if docker exec doc-parser-redis redis-cli ping 2>/dev/null | grep -q PONG; then
            ok "Redis: PONG"
            break
        fi
        [ "$i" -eq 10 ] && fail "Redis não respondeu após 10 tentativas"
        sleep 2
    done

    # MongoDB
    info "Verificando MongoDB..."
    for i in $(seq 1 10); do
        if docker exec doc-parser-mongodb mongosh --eval "db.runCommand({ping:1})" --quiet 2>/dev/null | grep -q "ok"; then
            ok "MongoDB: ok"
            break
        fi
        [ "$i" -eq 10 ] && fail "MongoDB não respondeu após 10 tentativas"
        sleep 2
    done

    # ChromaDB
    info "Verificando ChromaDB..."
    for i in $(seq 1 10); do
        if curl -sf http://localhost:8000/api/v1/heartbeat &>/dev/null; then
            ok "ChromaDB: heartbeat ok"
            break
        fi
        [ "$i" -eq 10 ] && warn "ChromaDB não respondeu (busca semântica pode não funcionar)"
        sleep 2
    done

    # API
    info "Verificando API (pode levar até 60s no primeiro start)..."
    for i in $(seq 1 30); do
        if curl -sf http://localhost:8090/api/jobs/healthcheck 2>/dev/null | grep -q "ok"; then
            ok "API: healthcheck ok"
            break
        fi
        [ "$i" -eq 30 ] && fail "API não respondeu após 60s"
        sleep 2
    done

    # Worker (sem endpoint, verifica se o container está rodando)
    info "Verificando Worker..."
    if docker ps --filter "name=doc-parser-worker" --filter "status=running" -q | grep -q .; then
        ok "Worker: container rodando"
    else
        warn "Worker não está rodando. Verifique: docker logs doc-parser-worker"
    fi
}

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------

print_summary() {
    header "Deploy concluído"

    echo ""
    echo -e "${BOLD}Endpoints disponíveis:${NC}"
    echo "  UI (upload):       http://localhost:8090/"
    echo "  API (jobs):        http://localhost:8090/api/jobs"
    echo "  API (search):      http://localhost:8090/api/search"
    echo "  API (agent):       http://localhost:8090/api/agent/search"
    echo "  API (documents):   http://localhost:8090/api/documents/{id}"
    echo "  API (health):      http://localhost:8090/api/jobs/healthcheck"
    echo ""
    echo -e "${BOLD}Fluxo de uso:${NC}"
    echo "  1. Upload:    curl -X POST http://localhost:8090/api/jobs -F 'file=@documento.pdf'"
    echo "  2. Status:    curl http://localhost:8090/api/jobs/{job_id}"
    echo "  3. Documento: curl http://localhost:8090/api/documents/{document_id}"
    echo "  4. Busca:     curl -X POST http://localhost:8090/api/search \\"
    echo "                  -H 'Content-Type: application/json' \\"
    echo "                  -d '{\"query\": \"texto de busca\"}'"
    echo "  5. Agente:    curl -X POST http://localhost:8090/api/agent/search \\"
    echo "                  -H 'Content-Type: application/json' \\"
    echo "                  -d '{\"query\": \"buscar info sobre X\"}'"
    echo ""
    echo -e "${BOLD}Comandos úteis:${NC}"
    echo "  Logs (todos):    docker compose -f ${COMPOSE_FILE} logs -f"
    echo "  Logs (worker):   docker logs -f doc-parser-worker"
    echo "  Logs (api):      docker logs -f doc-parser-api"
    echo "  Status:          docker compose -f ${COMPOSE_FILE} ps"
    echo "  Parar:           docker compose -f ${COMPOSE_FILE} down"
    echo "  Parar + limpar:  docker compose -f ${COMPOSE_FILE} down -v"
    echo "  MongoDB shell:   docker exec -it doc-parser-mongodb mongosh doc_parser"
    echo ""
}

# ---------------------------------------------------------------------------
# Comandos auxiliares
# ---------------------------------------------------------------------------

cmd_down() {
    header "Parando serviços"
    dc down
    ok "Todos os serviços parados"
}

cmd_status() {
    header "Status dos serviços"
    dc ps
}

cmd_logs() {
    dc logs -f
}

cmd_reset() {
    header "Reset completo (remove volumes!)"
    warn "Isso vai apagar TODOS os dados: Redis, MongoDB, ChromaDB, cache de modelos."
    echo ""
    read -rp "Tem certeza? (y/N) " confirm
    if [[ "${confirm}" =~ ^[yY]$ ]]; then
        dc down -v
        ok "Volumes removidos"
        main_deploy "--rebuild"
    else
        info "Cancelado."
    fi
}

# ---------------------------------------------------------------------------
# Deploy principal
# ---------------------------------------------------------------------------

main_deploy() {
    local rebuild_flag="${1:-}"

    check_prerequisites
    check_env_vars
    prepare_directories

    if [ "${rebuild_flag}" = "--rebuild" ]; then
        build_images "--no-cache"
    else
        build_images
    fi

    start_services
    wait_for_health
    print_summary
}

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

cd "${PROJECT_DIR}"

case "${1:-}" in
    --down)
        cmd_down
        ;;
    --status)
        cmd_status
        ;;
    --logs)
        cmd_logs
        ;;
    --reset)
        cmd_reset
        ;;
    --rebuild)
        main_deploy "--rebuild"
        ;;
    --help|-h)
        echo "Uso: $0 [opção]"
        echo ""
        echo "Opções:"
        echo "  (sem opção)   Deploy completo (build + up + healthcheck)"
        echo "  --rebuild     Force rebuild sem cache Docker"
        echo "  --down        Parar todos os serviços"
        echo "  --status      Mostrar status dos containers"
        echo "  --logs        Seguir logs de todos os serviços"
        echo "  --reset       Parar, limpar volumes e redeployar"
        echo "  --help        Mostrar esta ajuda"
        ;;
    *)
        main_deploy
        ;;
esac
