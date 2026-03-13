#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./setup_env.sh
source "$ROOT_DIR/scripts/setup_env.sh"

echo "[Preflight] Checking required commands..."
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 not found"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "Error: nvidia-smi not found"; exit 1; }

PYTHON_CMD="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_CMD="$ROOT_DIR/.venv/bin/python"
fi

echo "[Preflight] Python interpreter: $PYTHON_CMD"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "[Preflight] GPU count detected: $GPU_COUNT"

echo "[Preflight] Checking Python imports..."
"$PYTHON_CMD" - <<'PY'
import importlib

modules = ["torch", "wandb", "open_clip", "yaml", "tqdm"]
missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)

if missing:
    raise SystemExit(f"Missing Python packages: {', '.join(missing)}")

print("Python package check passed")
PY

echo "[Preflight] Done."
