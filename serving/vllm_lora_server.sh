#!/usr/bin/env bash
# OPTIONAL: serve the base model + LoRA adapter directly, without merging first.
#
# NOTE: For Phase 7 we RECOMMEND serving the MERGED model (serving/vllm_server.sh).
# Direct LoRA serving in vLLM requires `--enable-lora` and a `--lora-modules`
# entry, and support/flags vary across vLLM versions. Treat this script as a
# convenience for experimentation, and verify the flags against your installed
# vLLM (`vllm serve --help`). It is not exercised by tests.
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

BASE_MODEL=${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-checkpoints/finsage-7b}
LORA_ADAPTER_NAME=${LORA_ADAPTER_NAME:-finsage-7b}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

if [[ ! -e "$LORA_ADAPTER_PATH" ]]; then
  echo "ERROR: LoRA adapter not found at '$LORA_ADAPTER_PATH'." >&2
  echo "Train it first (training/train.py) or set LORA_ADAPTER_PATH." >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: 'vllm' not found. Install serving deps: pip install -e '.[serving]'" >&2
  exit 1
fi

echo "Serving base '$BASE_MODEL' with LoRA '$LORA_ADAPTER_NAME' from '$LORA_ADAPTER_PATH'"
echo "If these flags are rejected by your vLLM version, prefer the merged-model path."

exec vllm serve "$BASE_MODEL" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-lora \
  --lora-modules "$LORA_ADAPTER_NAME=$LORA_ADAPTER_PATH"
