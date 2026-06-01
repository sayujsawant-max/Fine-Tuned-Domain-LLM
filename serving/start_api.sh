#!/usr/bin/env bash
# Start the FinSage-7B FastAPI service (Phase 8).
#
# Reads configuration from .env if present. Set API_RELOAD=true for autoreload
# during development.
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8080}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"

echo "Starting FinSage-7B API on ${HOST}:${PORT}"
echo "Proxying to vLLM at ${VLLM_BASE_URL}"
echo "Docs:    http://${HOST}:${PORT}/docs"
echo "Health:  http://${HOST}:${PORT}/v1/health"

RELOAD_FLAG=()
if [[ "${API_RELOAD:-false}" == "true" ]]; then
  echo "Autoreload enabled (development)."
  RELOAD_FLAG=(--reload)
fi

exec python -m uvicorn finsage.serving.app:app \
  --host "${HOST}" \
  --port "${PORT}" \
  "${RELOAD_FLAG[@]}"
