#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server for FinSage-7B (Phase 8).
# Requires the 'serving' optional dependency group and a GPU.
set -euo pipefail

# Load .env if present (without overriding already-set vars).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL_PATH="${MERGED_MODEL_PATH:-./checkpoints/finsage-7b-merged}"
HOST="${VLLM_HOST:-localhost}"
PORT="${VLLM_PORT:-8000}"

echo "Starting vLLM server for ${MODEL_PATH} on ${HOST}:${PORT}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name finsage-7b \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 4096
