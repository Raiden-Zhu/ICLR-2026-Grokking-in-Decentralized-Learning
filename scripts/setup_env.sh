#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

load_env_file() {
    local env_file="$1"
    if [[ -f "$env_file" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$env_file"
        set +a
        echo "Loaded environment from $env_file"
    fi
}

# Load user-provided env files if they exist.
load_env_file "$ROOT_DIR/.env"
load_env_file "$ROOT_DIR/.env.local"

# Set safe defaults used by run scripts.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/cache/hf_home}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-$HF_HOME}"

if [[ -n "${WANDB_PROJECT:-}" ]]; then
    export WANDB_PROJECT
fi
if [[ -n "${WANDB_MODE:-}" ]]; then
    export WANDB_MODE
fi

echo "Environment is ready."
