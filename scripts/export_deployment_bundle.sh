#!/usr/bin/env bash
# Package the files needed to deploy FinSage-7B into a self-contained bundle.
#
# Output: dist/finsage-deployment-bundle/
#
# Deliberately EXCLUDES model weights, checkpoints, raw/processed data, secrets,
# and experiment logs — only deployment code/config/docs are bundled.
#
# Usage:
#   scripts/export_deployment_bundle.sh
#   scripts/export_deployment_bundle.sh --help
set -euo pipefail

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

OUT="dist/finsage-deployment-bundle"

echo "Building deployment bundle at ${OUT}…"
rm -rf "$OUT"
mkdir -p "$OUT"

# Helper: copy a path into the bundle preserving its relative layout.
copy_into() {
  local src="$1"
  if [[ ! -e "$src" ]]; then
    echo "  skip (missing): $src"
    return
  fi
  local dest="${OUT}/$(dirname "$src")"
  mkdir -p "$dest"
  # Exclude caches and local env files defensively.
  cp -r "$src" "$dest/" 2>/dev/null
  echo "  added: $src"
}

copy_into docker
copy_into serving
copy_into frontend
copy_into src/finsage/serving
copy_into scripts/check_full_stack.py
copy_into scripts/deploy_local.sh
copy_into scripts/check_api_server.py
copy_into configs/deployment_config.yaml
copy_into configs/serving_config.yaml
copy_into .env.example
copy_into README.md
copy_into docs/deployment_guide.md
copy_into reports/deployment_checklist.md

# Strip artifacts that must never ship.
find "$OUT" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -type d -name "node_modules" -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -type d -name ".next" -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -type f \( -name ".env" -o -name ".env.local" -o -name "*.log" \) -delete 2>/dev/null || true
find "$OUT" -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" -o -name "*.pth" \) -delete 2>/dev/null || true

echo "Bundle ready: ${OUT}"
echo "Contents:"
find "$OUT" -maxdepth 2 -mindepth 1 | sort | sed 's/^/  /'
