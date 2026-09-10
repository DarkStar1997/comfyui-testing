#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="comfyui:2.14.0-cuda132-v0.35.0"

if [ -z "${HF_TOKEN:-}" ] && [ -f "${REPO_DIR}/.env" ]; then
  HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' "${REPO_DIR}/.env" | head -1 | tr -d "\"'")"
  export HF_TOKEN
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "error: HF_TOKEN not set (export it, or put HF_TOKEN=... in .env)" >&2
  exit 1
fi

mkdir -p "${REPO_DIR}/models"

exec docker run --rm -e HF_TOKEN \
  -v "${REPO_DIR}/models:/ComfyUI/models" \
  -v "${REPO_DIR}/download_models.py:/download_models.py:ro" \
  --entrypoint python "${IMAGE}" /download_models.py
