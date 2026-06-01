#!/usr/bin/env bash
# Start the vLLM OpenAI-compatible server for the merged FinSage-7B model (Phase 7).
# Requires the 'serving' optional dependency group (vLLM) and an NVIDIA GPU.
set -euo pipefail

# Load .env if present (without overriding already-set vars).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL_PATH=${MODEL_PATH:-checkpoints/finsage-7b-merged}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-finsage-7b}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8000}
DTYPE=${DTYPE:-auto}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}
API_KEY=${API_KEY:-}

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "ERROR: Merged model not found at '$MODEL_PATH'." >&2
  echo "Run training/merge_adapter.py first or set MODEL_PATH." >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "ERROR: 'vllm' not found. Install serving deps: pip install -e '.[serving]'" >&2
  exit 1
fi

# Build the argument list.
args=(
  serve "$MODEL_PATH"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --max-num-seqs "$MAX_NUM_SEQS"
)
if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  args+=(--trust-remote-code)
fi

# Log the command without exposing the API key.
echo "Starting: vllm ${args[*]}$([[ -n "$API_KEY" ]] && echo ' --api-key ***')"

if [[ -n "$API_KEY" ]]; then
  args+=(--api-key "$API_KEY")
fi

exec vllm "${args[@]}"
