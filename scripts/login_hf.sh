#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./setup_env.sh
source "$ROOT_DIR/scripts/setup_env.sh"

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "Error: 'huggingface-cli' not found. Install 'huggingface_hub' first."
    exit 1
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "HF_TOKEN is not set, launching interactive Hugging Face login..."
    huggingface-cli login
fi

echo "Hugging Face login completed."
