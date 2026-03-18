#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./setup_env.sh
source "$ROOT_DIR/scripts/setup_env.sh"

PYTHON_CMD="$(repo_python)"

if [[ -n "${HF_TOKEN:-}" ]]; then
    "$PYTHON_CMD" - <<'PY'
import os

try:
    from huggingface_hub import login
except Exception as exc:
    raise SystemExit(f"Error: failed to import huggingface_hub.login: {exc}") from exc

token = os.environ.get("HF_TOKEN", "")
if not token:
    raise SystemExit("HF_TOKEN is empty")

login(token=token, add_to_git_credential=False)
print("Hugging Face login completed.")
PY
else
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "Error: 'huggingface-cli' not found. Install 'huggingface_hub' first." >&2
        exit 1
    fi

    echo "HF_TOKEN is not set, launching interactive Hugging Face login..."
    huggingface-cli login
    echo "Hugging Face login completed."
fi
