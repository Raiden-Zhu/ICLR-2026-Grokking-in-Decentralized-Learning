#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./setup_env.sh
source "$ROOT_DIR/scripts/setup_env.sh"

if ! command -v wandb >/dev/null 2>&1; then
    echo "Error: 'wandb' command not found. Install requirements first."
    exit 1
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY is not set, launching interactive wandb login..."
    wandb login
fi

echo "wandb login completed."
