#!/usr/bin/env bash
# Start the FinSage-7B stack locally via Docker Compose.
#
# Modes:
#   --demo   CPU-only demo (frontend + API, no GPU/model). Recruiter-friendly.
#   --full   Full stack (vllm + api + frontend). Requires the merged model + GPU.
#   --gpu    Full stack with explicit NVIDIA GPU device reservations.
#
# Usage:
#   scripts/deploy_local.sh --demo
#   scripts/deploy_local.sh --full
#   scripts/deploy_local.sh --gpu
set -euo pipefail

COMPOSE_DIR="docker"
BASE="${COMPOSE_DIR}/docker-compose.yml"
DEMO="${COMPOSE_DIR}/docker-compose.demo.yml"
GPU="${COMPOSE_DIR}/docker-compose.gpu.yml"
MODEL_PATH="${MODEL_PATH:-checkpoints/finsage-7b-merged}"

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "Docker is not installed. See https://docs.docker.com/get-docker/"
  docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required ('docker compose')."
}

check_env_file() {
  if [[ ! -f .env ]]; then
    echo "WARNING: no .env found. Copy the template first:" >&2
    echo "    cp .env.example .env" >&2
    echo "Continuing with compose defaults (API_SECRET_KEY=change-me)." >&2
  fi
}

# Warn (do not hard-fail) if the placeholder secret is used in production.
check_secret() {
  local key="${API_SECRET_KEY:-change-me}"
  if [[ "${ENVIRONMENT:-production}" == "production" && "$key" == "change-me" ]]; then
    echo "WARNING: API_SECRET_KEY is the default 'change-me'. Set a strong secret" >&2
    echo "         before any public deployment (export API_SECRET_KEY=...)." >&2
  fi
}

check_model() {
  if [[ ! -e "$MODEL_PATH" ]]; then
    die "Merged model not found at '$MODEL_PATH'. Build it with training/merge_adapter.py or set MODEL_PATH."
  fi
}

MODE="${1:-}"
case "$MODE" in
  --demo)
    require_docker
    check_env_file
    echo "Starting FinSage-7B in DEMO mode (no GPU, no model server)…"
    echo "Frontend: http://localhost:3000   API: http://localhost:8080/v1"
    exec docker compose -f "$DEMO" up --build
    ;;
  --full)
    require_docker
    check_env_file
    check_secret
    check_model
    echo "Starting FinSage-7B FULL stack (vllm + api + frontend)…"
    echo "Frontend: http://localhost:3000   API: http://localhost:8080/v1   vLLM: http://localhost:8000/v1"
    exec docker compose -f "$BASE" up --build
    ;;
  --gpu)
    require_docker
    check_env_file
    check_secret
    check_model
    echo "Starting FinSage-7B FULL stack with GPU reservations…"
    echo "Requires the NVIDIA Container Toolkit. See docker/docker-compose.gpu.yml."
    exec docker compose -f "$BASE" -f "$GPU" up --build
    ;;
  -h | --help | "")
    usage
    [[ -z "$MODE" ]] && exit 1 || exit 0
    ;;
  *)
    echo "Unknown option: $MODE" >&2
    usage
    exit 1
    ;;
esac
