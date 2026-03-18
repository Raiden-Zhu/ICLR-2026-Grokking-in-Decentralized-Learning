#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_INSTALL=0
SKIP_CHECK=0
PIP_INDEX_URL_VALUE="${PIP_INDEX_URL:-}"
PIP_EXTRA_INDEX_URL_VALUE="${PIP_EXTRA_INDEX_URL:-}"

usage() {
    cat <<'EOF'
Usage: bash scripts/bootstrap_env.sh [options]

Options:
  --skip-install                 Create venv but skip requirements install
  --skip-check                   Skip post-install import checks
  --pip-index-url <url>          Use a custom pip index URL
  --pip-extra-index-url <url>    Use an extra pip index URL
  --use-tsinghua-mirror          Shortcut for Tsinghua PyPI mirror
  --help                         Show this help

Notes:
  - Mirror usage is optional and off by default.
  - You can also set PIP_INDEX_URL/PIP_EXTRA_INDEX_URL in your shell.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        --skip-check)
            SKIP_CHECK=1
            shift
            ;;
        --pip-index-url)
            if [[ $# -lt 2 ]]; then
                echo "Error: --pip-index-url requires a value"
                exit 1
            fi
            PIP_INDEX_URL_VALUE="$2"
            shift 2
            ;;
        --pip-extra-index-url)
            if [[ $# -lt 2 ]]; then
                echo "Error: --pip-extra-index-url requires a value"
                exit 1
            fi
            PIP_EXTRA_INDEX_URL_VALUE="$2"
            shift 2
            ;;
        --use-tsinghua-mirror)
            PIP_INDEX_URL_VALUE="https://pypi.tuna.tsinghua.edu.cn/simple"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN not found"
    exit 1
fi

assert_safe_venv_dir() {
    local resolved_venv_dir
    resolved_venv_dir="$($PYTHON_BIN -c 'import os; print(os.path.realpath(os.path.expanduser(os.environ["BOOTSTRAP_TARGET_DIR"])))')"

    local resolved_root_dir
    resolved_root_dir="$($PYTHON_BIN -c 'import os; print(os.path.realpath(os.environ["BOOTSTRAP_ROOT_DIR"]))')"

    local resolved_home_dir
    resolved_home_dir="$($PYTHON_BIN -c 'import os; print(os.path.realpath(os.path.expanduser("~")))')"

    if [[ -z "$resolved_venv_dir" ]]; then
        echo "Error: VENV_DIR resolved to an empty path"
        exit 1
    fi

    case "$resolved_venv_dir" in
        /|"$resolved_root_dir"|"$resolved_home_dir")
            echo "Error: Refusing to remove unsafe VENV_DIR path: $resolved_venv_dir"
            exit 1
            ;;
    esac
}

export BOOTSTRAP_TARGET_DIR="$VENV_DIR"
export BOOTSTRAP_ROOT_DIR="$ROOT_DIR"
assert_safe_venv_dir
unset BOOTSTRAP_TARGET_DIR
unset BOOTSTRAP_ROOT_DIR

echo "[Bootstrap] Using Python: $(command -v "$PYTHON_BIN")"
echo "[Bootstrap] Creating virtual environment at: $VENV_DIR"
if [[ -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
fi

if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo "[Bootstrap] python -m venv failed, trying virtualenv fallback"
    "$PYTHON_BIN" -m pip install --user virtualenv
    "$PYTHON_BIN" -m virtualenv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

PIP_INSTALL_ARGS=()
if [[ -n "$PIP_INDEX_URL_VALUE" ]]; then
    PIP_INSTALL_ARGS+=("--index-url" "$PIP_INDEX_URL_VALUE")
fi
if [[ -n "$PIP_EXTRA_INDEX_URL_VALUE" ]]; then
    PIP_INSTALL_ARGS+=("--extra-index-url" "$PIP_EXTRA_INDEX_URL_VALUE")
fi

if [[ ${#PIP_INSTALL_ARGS[@]} -gt 0 ]]; then
    echo "[Bootstrap] Using custom pip index options: ${PIP_INSTALL_ARGS[*]}"
fi

echo "[Bootstrap] Upgrading pip/setuptools/wheel"
python -m pip install "${PIP_INSTALL_ARGS[@]}" --upgrade pip setuptools wheel

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    echo "[Bootstrap] Installing project dependencies"
    python -m pip install "${PIP_INSTALL_ARGS[@]}" -r "$ROOT_DIR/requirements.txt"
fi

if [[ "$SKIP_CHECK" -eq 0 ]]; then
    echo "[Bootstrap] Running import checks"
    python - <<'PY'
import importlib

modules = ["torch", "wandb", "open_clip", "yaml", "tqdm"]
missing = []
for mod in modules:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

if missing:
    raise SystemExit(f"Missing imports after install: {', '.join(missing)}")

print("Import checks passed")
PY
fi

echo "[Bootstrap] Done"
echo "[Bootstrap] Activate later with: source $VENV_DIR/bin/activate"
