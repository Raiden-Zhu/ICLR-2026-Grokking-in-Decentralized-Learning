#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=./setup_env.sh
source "$ROOT_DIR/scripts/setup_env.sh"

PYTHON_CMD="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_CMD="$ROOT_DIR/.venv/bin/python"
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
    MODEL_NAME="$1"
    shift
else
    MODEL_NAME="${MODEL_NAME:-}"
fi

NUM_GPUS="${NUM_GPUS:-$($PYTHON_CMD -c 'import torch; print(torch.cuda.device_count())')}"
if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "Error: No GPUs available."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo "Detected GPUs: $NUM_GPUS"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

OVERRIDES=(--set "num_GPU=${NUM_GPUS}")

if [[ -n "$MODEL_NAME" ]]; then
    OVERRIDES+=(--set "model_name=${MODEL_NAME}")
fi

add_override_from_env() {
    local env_key="$1"
    local arg_key="$2"
    if [[ -n "${!env_key:-}" ]]; then
        OVERRIDES+=(--set "${arg_key}=${!env_key}")
    fi
}

add_override_from_env DATASET_NAME dataset_name
add_override_from_env IMAGE_SIZE image_size
add_override_from_env PRETRAINED pretrained
add_override_from_env OPTIMIZER optimizer_name
add_override_from_env LR lr
add_override_from_env AMP_ENABLED amp_enabled
add_override_from_env AMP_DTYPE amp_dtype
add_override_from_env ALPHA alpha
add_override_from_env NODE_DATASIZE node_datasize
add_override_from_env GOSSIP_TOPOLOGY gossip_topology
add_override_from_env R_SCHEDULE r_schedule
add_override_from_env R_START r_start
add_override_from_env R_END r_end
add_override_from_env NUM_NODES num_nodes
add_override_from_env K_STEPS k_steps
add_override_from_env MAX_STEPS max_steps
add_override_from_env BATCH_SIZE batch_size
add_override_from_env WEIGHT_DECAY weight_decay
add_override_from_env NON_IID nonIID
add_override_from_env WINDOW_SIZE window_size
add_override_from_env POINT1 point1
add_override_from_env LR_SCHEDULER lr_scheduler
add_override_from_env POST_MERGE_ROUNDS post_merge_rounds
add_override_from_env END_TOPOLOGY end_topology
add_override_from_env SEED seed

if [[ -n "${R_STARTS:-}" ]]; then
    # Example: R_STARTS="1.2,2.0"
    OVERRIDES+=(--set "r_starts=[${R_STARTS}]")
fi

"$PYTHON_CMD" "$ROOT_DIR/scripts/run_with_config.py" \
    --config "$ROOT_DIR/configs/paper/run_main_experiment.yaml" \
    --set "run.python=${PYTHON_CMD}" \
    "${OVERRIDES[@]}" \
    "$@"
