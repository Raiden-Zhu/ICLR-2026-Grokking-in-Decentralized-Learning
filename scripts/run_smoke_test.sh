#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./setup_env.sh
source "$ROOT_DIR/scripts/setup_env.sh"

PYTHON_CMD="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_CMD="$ROOT_DIR/.venv/bin/python"
fi

"$PYTHON_CMD" "$ROOT_DIR/scripts/run_with_config.py" \
    --config "$ROOT_DIR/configs/examples/smoke_test_1gpu_1000steps.yaml" \
    --set "run.python=$PYTHON_CMD" \
    "$@"
