#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

load_env_file() {
    local env_file="$1"
    local line=""
    local line_number=0
    local key=""
    local value=""

    [[ -f "$env_file" ]] || return 0

    while IFS= read -r line || [[ -n "$line" ]]; do
        line_number=$((line_number + 1))
        line="${line%$'\r'}"

        if [[ "$line" =~ ^[[:space:]]*$ ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi

        line="$(trim_whitespace "$line")"
        if [[ "$line" == export[[:space:]]* ]]; then
            line="$(trim_whitespace "${line#export}")"
        fi

        if [[ ! "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
            echo "Error: invalid env entry in $env_file:$line_number" >&2
            echo "Expected KEY=VALUE with optional surrounding quotes." >&2
            return 1
        fi

        key="${BASH_REMATCH[1]}"
        value="$(trim_whitespace "${BASH_REMATCH[2]}")"

        if [[ ${#value} -ge 2 ]]; then
            if [[ "$value" == \"*\" && "$value" == *\" ]]; then
                value="${value:1:${#value}-2}"
            elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
                value="${value:1:${#value}-2}"
            fi
        fi

        export "$key=$value"
    done < "$env_file"

    echo "Loaded environment from $env_file"
}

repo_python() {
    if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
        printf '%s\n' "$ROOT_DIR/.venv/bin/python"
    else
        printf '%s\n' "python3"
    fi
}

load_env_file "$ROOT_DIR/.env"
load_env_file "$ROOT_DIR/.env.local"

export HF_HOME="${HF_HOME:-$ROOT_DIR/cache/hf_home}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-$HF_HOME}"

if [[ -n "${HF_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT
fi

if [[ -n "${WANDB_PROJECT:-}" ]]; then
    export WANDB_PROJECT
fi
if [[ -n "${WANDB_MODE:-}" ]]; then
    export WANDB_MODE
fi

echo "Environment is ready."
