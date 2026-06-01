#!/usr/bin/env bash
# Start the FinSage-7B FastAPI service (Phase 9).
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HOST="${API_HOST:-localhost}"
PORT="${API_PORT:-8080}"

echo "Starting FinSage-7B API on ${HOST}:${PORT}"

exec python -m uvicorn finsage.serving.app:app \
  --host "${HOST}" \
  --port "${PORT}"
