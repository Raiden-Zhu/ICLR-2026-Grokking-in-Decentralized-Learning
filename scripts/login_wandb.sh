#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./setup_env.sh
source "$ROOT_DIR/scripts/setup_env.sh"

PYTHON_CMD="$(repo_python)"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    "$PYTHON_CMD" - <<'PY'
import os
import sys

try:
    import wandb
except Exception as exc:
    raise SystemExit(f"Error: failed to import wandb: {exc}") from exc

key = os.environ.get("WANDB_API_KEY", "")
if not key:
    raise SystemExit("WANDB_API_KEY is empty")

if not wandb.login(key=key, relogin=True):
    raise SystemExit("wandb login failed")

print("wandb login completed.")
PY
else
    if ! command -v wandb >/dev/null 2>&1; then
        echo "Error: 'wandb' command not found. Install requirements first." >&2
        exit 1
    fi

    echo "WANDB_API_KEY is not set, launching interactive wandb login..."
    wandb login
    echo "wandb login completed."
fi
