"""Worker-side training and synchronization helpers for the multi-GPU runtime."""

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from core import (
    compute_r,
    compute_consensus_error_from_buffer,
    copy_mean_state_to_model,
    copy_model_to_shared_buffer,
    copy_shared_buffer_to_model,
    copy_source_state_to_target,
    dense_to_sparse_gossip_matrix,
    get_sparse_gossip_matrix,
    gossip_update_flat_buffer,
)
from core.evaluation_runtime import (
    evaluate_average_model_from_shared_state,
    evaluate_local_models,
)
from core.optim import init_optimizer, init_scheduler
from models.get_model import get_model


def get_local_node_indices(rank, world_size, num_nodes, networks_per_gpu):
    """Return the inclusive node range owned by one worker process."""
    start_idx = rank * networks_per_gpu
    if rank == world_size - 1:
        end_idx = num_nodes
    else:
        end_idx = start_idx + networks_per_gpu
    return start_idx, end_idx, list(range(start_idx, end_idx))


def create_model_from_config(config, pretrained=None, schema_only=False):
    """Instantiate a model using the shared config and optional preload override."""
    model_pretrained = config.pretrained if pretrained is None else pretrained
    return get_model(
        model_name=config.model_name,
        pretrained=model_pretrained,
        schema_only=schema_only,
        **config.model_kwargs,
    )


def initialize_local_training_state(config, local_node_indices, device):
    """Build local models, optimizers, and schedulers for one worker."""
    local_networks = []
    base_seed = config.seed
    for node_index in local_node_indices:
        if config.diff_init and not config.pretrained:
            torch.manual_seed(base_seed + node_index)
        else:
            torch.manual_seed(base_seed)
        local_networks.append(create_model_from_config(config).to(device))

    local_optimizers = [
        init_optimizer(
            network=network,
            optimizer_name=config.optimizer_name,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        for network in local_networks
    ]
    local_schedulers = [
        init_scheduler(
            optimizer=optimizer,
            lr_scheduler_type=config.lr_scheduler,
            max_steps=config.max_steps,
            lr=config.lr,
        )
        for optimizer in local_optimizers
    ]
    return local_networks, local_optimizers, local_schedulers


def create_amp_runtime(config):
    """Create autocast and gradient-scaling settings for the worker."""
    amp_dtype_name = str(config.amp_dtype).lower()
    if amp_dtype_name not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported amp_dtype: {amp_dtype_name}. Use bf16, fp16, or fp32.")

    autocast_enabled = bool(config.amp_enabled) and amp_dtype_name in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=2**15,
        growth_factor=2.5,
        backoff_factor=0.5,
        growth_interval=500,
        enabled=bool(config.amp_enabled) and amp_dtype_name == "fp16",
    )
    return autocast_enabled, autocast_dtype, scaler


def clip_gradients(network, max_grad_abs_value=1.0):
    """Apply the existing per-parameter clipping policy."""
    for param in network.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-max_grad_abs_value, max_grad_abs_value)


def train_local_models_for_round(
    config,
    device,
    local_node_indices,
    local_networks,
    local_optimizers,
    local_schedulers,
    train_dataloaders_list,
    local_steps_completed,
    log_queue,
    autocast_enabled,
    autocast_dtype,
    scaler,
):
    """Run one local optimization round before the centralized gossip step."""
    communication_round_steps = int(config.k_steps)

    for local_idx, global_idx in enumerate(local_node_indices):
        if local_steps_completed[local_idx] >= config.max_steps:
            continue

        network = local_networks[local_idx]
        optimizer = local_optimizers[local_idx]
        scheduler = local_schedulers[local_idx]
        train_dataloader = train_dataloaders_list[local_idx]
        data_iter = iter(train_dataloader)

        network.train()
        mean_train_loss = 0.0
        correct_train = 0
        total_train = 0
        executed_steps = 0

        for _ in range(communication_round_steps):
            if local_steps_completed[local_idx] >= config.max_steps:
                break

            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                inputs, targets = next(data_iter)

            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                outputs = network(inputs)
                loss = nn.functional.cross_entropy(outputs, targets, label_smoothing=0.1)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            mean_train_loss += loss.item()
            clip_gradients(network)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            executed_steps += 1

            optimizer.zero_grad()

            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == targets).sum().item()
            total_train += targets.size(0)
            local_steps_completed[local_idx] += 1

            if local_steps_completed[local_idx] >= config.max_steps:
                break

        if executed_steps == 0:
            continue

        mean_train_loss /= executed_steps
        train_accuracy = 100 * correct_train / total_train if total_train else 0.0

        log_queue.put(
            {
                "type": "train",
                "network_idx": global_idx,
                "loss": mean_train_loss,
                "accuracy": train_accuracy,
                "step": local_steps_completed[local_idx],
                "k_steps": executed_steps,
            }
        )

        network.zero_grad()


def publish_local_state(local_node_indices, local_networks, shared_state_pool):
    """Publish local node states into the shared-memory source buffer."""
    for local_idx, global_idx in enumerate(local_node_indices):
        copy_model_to_shared_buffer(
            local_networks[local_idx],
            shared_state_pool,
            global_idx,
            buffer_name="source",
        )


def publish_local_step_progress(local_node_indices, local_steps_completed, shared_node_steps):
    """Publish per-node local progress for round scheduling and evaluation decisions."""
    for local_idx, global_idx in enumerate(local_node_indices):
        shared_node_steps[global_idx] = int(local_steps_completed[local_idx])


def resolve_reference_step(shared_node_steps):
    """Use the slowest logical node as the global reference step for shared schedules."""
    if len(shared_node_steps) == 0:
        return 0
    return min(int(step) for step in shared_node_steps)


def initialize_next_eval_step(eval_steps, max_steps):
    """Return the first scheduled evaluation step."""
    if eval_steps is None or eval_steps <= 0:
        return max_steps + 1
    return max(1, int(eval_steps))


def should_run_evaluation(current_step, next_eval_step, max_steps):
    """Decide whether the current communication round should trigger evaluation."""
    return current_step >= max_steps or current_step >= next_eval_step


def advance_next_eval_step(current_step, next_eval_step, eval_steps, max_steps):
    """Advance the next evaluation threshold after one evaluation round."""
    if eval_steps is None or eval_steps <= 0:
        return max_steps + 1

    step_stride = max(1, int(eval_steps))
    updated = next_eval_step
    while updated <= current_step and updated < max_steps:
        updated += step_stride
    return updated


def build_gossip_matrix_for_round(config, current_step, log_queue, is_disabled_topology):
    """Construct the communication operator used by the centralized aggregation phase."""
    if isinstance(config.gossip_topology, list):
        return dense_to_sparse_gossip_matrix(torch.tensor(config.gossip_topology, dtype=torch.float32))
    if is_disabled_topology(config.gossip_topology):
        return None

    r = compute_r(
        current_iter=current_step,
        start_iter=0,
        end_iter=config.max_steps,
        r_start=config.r_start,
        r_end=config.r_end,
        point1=config.point1,
        window_size=config.window_size,
        schedule=config.r_schedule,
    )
    log_queue.put({"type": "gossip_params", "r": r, "step": current_step})

    gossip_matrix, _ = get_sparse_gossip_matrix(
        config.num_nodes,
        topology=config.gossip_topology,
        r=r,
        wandb=None,
        current_iter=current_step,
        end_iter=config.max_steps,
    )
    return gossip_matrix


def run_rank_zero_gossip_round(
    config,
    shared_state_pool,
    shared_node_steps,
    log_queue,
    aggregation_device,
    is_disabled_topology,
):
    """Apply one centralized gossip round directly on the shared flat state buffer."""
    current_step = resolve_reference_step(shared_node_steps)
    gossip_matrix = build_gossip_matrix_for_round(
        config,
        current_step,
        log_queue,
        is_disabled_topology,
    )

    if gossip_matrix is not None:
        gossip_update_flat_buffer(
            shared_state_pool.flat_source,
            shared_state_pool.flat_target,
            gossip_matrix,
            compute_device=aggregation_device,
        )
        consensus_error = compute_consensus_error_from_buffer(shared_state_pool, buffer_name="target")
        log_queue.put({"type": "consensus_error", "error": consensus_error, "step": current_step})
    else:
        copy_source_state_to_target(shared_state_pool)

    return current_step


def load_broadcast_parameters(local_node_indices, local_networks, shared_state_pool, device):
    """Load centrally updated parameters back onto each worker's local models."""
    for local_idx, global_idx in enumerate(local_node_indices):
        copy_shared_buffer_to_model(
            local_networks[local_idx],
            shared_state_pool,
            global_idx,
            device=device,
            buffer_name="target",
        )


def run_rank_zero_control_phase(
    *,
    rank,
    config,
    shared_state_pool,
    log_queue,
    aggregation_device,
    shared_reference_step,
    shared_node_steps,
    shared_next_eval_step,
    shared_should_eval,
    is_disabled_topology,
):
    """Run the rank-0-only aggregation/control phase between publish and reload."""
    if rank != 0:
        return

    reference_step = run_rank_zero_gossip_round(
        config,
        shared_state_pool,
        shared_node_steps,
        log_queue,
        aggregation_device,
        is_disabled_topology,
    )
    shared_reference_step.value = int(reference_step)
    run_evaluation = should_run_evaluation(
        reference_step,
        shared_next_eval_step.value,
        config.max_steps,
    )
    shared_should_eval.value = int(run_evaluation)
    if run_evaluation:
        shared_next_eval_step.value = advance_next_eval_step(
            reference_step,
            shared_next_eval_step.value,
            config.eval_steps,
            config.max_steps,
        )


def run_optional_evaluation_phase(
    *,
    rank,
    config,
    local_node_indices,
    local_networks,
    valid_dataloaders_list,
    test_dataloader,
    calibration_loader,
    local_steps_completed,
    log_queue,
    shared_state_pool,
    shared_reference_step,
    shared_should_eval,
    aggregation_device,
    reestimate_batch_norm_stats,
):
    """Run the evaluation phase after parameter reload when the shared flag is enabled."""
    if not bool(shared_should_eval.value):
        return

    evaluate_local_models(
        local_node_indices,
        local_networks,
        valid_dataloaders_list,
        test_dataloader,
        local_steps_completed,
        log_queue,
    )
    if rank == 0:
        evaluate_average_model_from_shared_state(
            config,
            shared_state_pool,
            test_dataloader,
            calibration_loader,
            shared_reference_step.value,
            log_queue,
            aggregation_device,
            create_model_from_config=create_model_from_config,
            copy_mean_state_to_model=copy_mean_state_to_model,
            reestimate_batch_norm_stats=reestimate_batch_norm_stats,
        )


def initialize_worker_seed(config, rank):
    """Set the worker-local random seed policy used by the simulator."""
    worker_seed = config.seed + rank * 1000
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
