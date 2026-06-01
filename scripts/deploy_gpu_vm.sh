#!/usr/bin/env bash
# Bootstrap a generic Ubuntu GPU VM (RunPod / Lambda / any cloud) to run the
# FinSage-7B full stack. Conservative by design — it checks prerequisites and
# prints guidance rather than forcing destructive changes.
#
# Usage:
#   scripts/deploy_gpu_vm.sh            # run the bootstrap + start the stack
#   scripts/deploy_gpu_vm.sh --help
#
# Manual steps you may still need (left as comments below):
#   - clone this repo and `cd` into it
#   - place the merged model at checkpoints/finsage-7b-merged
#   - set a strong API_SECRET_KEY in .env
set -euo pipefail

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

log() { echo "[deploy_gpu_vm] $*"; }

# 1. System packages -------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  log "Updating apt package lists (sudo may prompt)…"
  sudo apt-get update -y || log "apt-get update failed; continuing."
else
  log "apt-get not found — this helper targets Ubuntu/Debian. Adapt manually."
fi

# 2. Docker ----------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "Docker not found. Installing via the official convenience script…"
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh
  sudo usermod -aG docker "$USER" || true
  log "Docker installed. You may need to log out/in for group changes."
else
  log "Docker present: $(docker --version)"
fi

# 3. NVIDIA Container Toolkit ----------------------------------------------
# Required so containers can see the GPU. Install steps vary by distro; see
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
if command -v nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA driver present:"
  nvidia-smi -L || true
else
  log "WARNING: nvidia-smi not found. Install the NVIDIA driver before GPU serving."
fi

if docker info 2>/dev/null | grep -qi nvidia; then
  log "NVIDIA Docker runtime detected."
else
  log "WARNING: NVIDIA Container Toolkit not detected. Install it, then run:"
  log "    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  log "Verify with: docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi"
fi

# 4. Environment -----------------------------------------------------------
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    log "Created .env from .env.example. EDIT IT — set a strong API_SECRET_KEY."
  else
    log "WARNING: no .env or .env.example found. Run from the repo root."
  fi
fi

# 5. Start the GPU stack ---------------------------------------------------
if [[ ! -e "${MODEL_PATH:-checkpoints/finsage-7b-merged}" ]]; then
  log "WARNING: merged model not found at ${MODEL_PATH:-checkpoints/finsage-7b-merged}."
  log "Place it there (training/merge_adapter.py) before starting the full stack."
fi

log "Starting the GPU stack (Ctrl-C to stop)…"
exec docker compose -f docker/docker-compose.yml -f docker/docker-compose.gpu.yml up --build
