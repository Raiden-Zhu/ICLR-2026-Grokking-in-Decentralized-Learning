import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader
from models.get_model import get_model
from datasets import load_dataset
from datasets.dirichlet_sampling import (
    create_simple_preference,
    create_train_valid_dataloaders_multi,
    create_IID_preference,
    dirichlet_split,
)
from core import (
    compute_r,
    compute_consensus_error_from_buffer,
    copy_mean_state_to_model,
    copy_model_to_shared_buffer,
    copy_shared_buffer_to_model,
    copy_source_state_to_target,
    create_shared_state_pool,
    dense_to_sparse_gossip_matrix,
    gossip_update,
    get_gossip_matrix,
    get_sparse_gossip_matrix,
    gossip_update_flat_buffer,
)
from core.config_validation import (
    DEFAULT_TRAINING_ARGS,
    collect_alias_deprecation_warnings,
    validate_training_kwargs,
)
from core.logging import logging_process
from core.optim import init_optimizer, init_scheduler
from core.runtime_setup import (
    create_runtime_synchronization,
    finalize_worker_orchestration,
    initialize_shared_training_state,
    initialize_wandb_run,
    prepare_dataset_bundle,
    print_runtime_layout,
    reconstruct_final_networks_from_shared_state,
    resolve_runtime_devices,
    resolve_runtime_layout,
    save_convergence_model,
    set_multiprocessing_spawn,
    spawn_worker_processes,
    start_logging_thread,
    wait_for_worker_processes,
)
import torch.nn.functional as F
import wandb
import random

# Multi-processing imports
import torch.multiprocessing as mp
from torch.multiprocessing import Barrier

from copy import deepcopy
import os

# Note: wandb.login() will be called in main() only on rank 0


def is_disabled_topology(topology):
    """Return True when gossip communication is explicitly disabled."""
    if topology is None:
        return True
    if isinstance(topology, str):
        return topology.lower() == "none"
    return False


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable training configuration shared with worker processes."""

    num_nodes: int
    max_steps: int
    k_steps: int
    eval_steps: int
    model_name: str
    pretrained: bool
    optimizer_name: str
    lr: float
    momentum: float
    weight_decay: float
    lr_scheduler: str
    amp_enabled: bool
    amp_dtype: str
    gossip_topology: Any
    r_start: float
    r_end: float
    r_schedule: str
    point1: float
    window_size: float
    seed: int
    diff_init: bool
    end_topology: Optional[str]
    post_merge_rounds: int
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics produced for a single node and split."""

    split_name: str
    node_index: int
    step: int
    accuracy: float
    loss: float


BN_REESTIMATION_MAX_BATCHES = 32


def create_bn_reestimation_loader(train_dataset, batch_size):
    """Create a lightweight calibration loader for merged-model BN re-estimation."""
    if train_dataset is None:
        return None
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


def reestimate_batch_norm_stats(model, calibration_loader, device, max_batches=BN_REESTIMATION_MAX_BATCHES):
    """Refresh BatchNorm running statistics on a merged model without enabling dropout."""
    if calibration_loader is None:
        return

    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats
    ]
    if not batch_norm_layers:
        return

    was_training = model.training
    original_momenta = {module: module.momentum for module in batch_norm_layers}

    model.eval()
    for module in batch_norm_layers:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    with torch.no_grad():
        for batch_index, batch in enumerate(calibration_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(inputs.to(device, non_blocking=True))

    for module in batch_norm_layers:
        module.momentum = original_momenta[module]
        module.eval()

    if was_training:
        model.train()
    else:
        model.eval()

def distribute_networks(networks, num_gpus):
    """
    Distributes neural network models across multiple GPUs in a round-robin fashion.

    Parameters:
    networks: list of torch.nn.Module
        A list of neural network models to be distributed.
    num_gpus: int
        The number of available GPU devices.

    Returns:
    distributed_networks: list of torch.nn.Module
        The list of neural networks moved to the appropriate GPU devices.
    """
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    distributed_networks = []
    for i, network in enumerate(networks):
        device = devices[i % num_gpus]
        distributed_networks.append(network.to(device))
    return distributed_networks


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

    # Keep GradScaler only for fp16; bf16 is numerically stable enough without it.
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

            # AMP is kept here because it changes the numerical path of forward/backward,
            # while the rest of the communication logic assumes ordinary gradients.
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
    """Return the first scheduled evaluation step.

    eval_steps uses the same unit as max_steps and k_steps. Values <= 0 disable
    periodic evaluation and keep only the final evaluation.
    """
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


def build_gossip_matrix_for_round(config, current_step, log_queue):
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


def compute_consensus_error(networks):
    """Measure mean squared deviation from the average model."""
    mean_params = {}
    for key in networks[0].state_dict().keys():
        mean_params[key] = sum(net.state_dict()[key] for net in networks) / len(networks)

    consensus_error = 0.0
    for net in networks:
        param_diff = 0.0
        for key in net.state_dict().keys():
            param_diff += torch.sum((net.state_dict()[key] - mean_params[key]) ** 2).item()
        consensus_error += param_diff
    return consensus_error / len(networks)


def run_rank_zero_gossip_round(
    config,
    shared_state_pool,
    shared_node_steps,
    log_queue,
    aggregation_device,
):
    """Apply one centralized gossip round directly on the shared flat state buffer."""
    current_step = resolve_reference_step(shared_node_steps)
    gossip_matrix = build_gossip_matrix_for_round(config, current_step, log_queue)

    if gossip_matrix is not None:
        gossip_update_flat_buffer(
            shared_state_pool.flat_source,
            shared_state_pool.flat_target,
            gossip_matrix,
            compute_device=aggregation_device,
        )
        consensus_error = compute_consensus_error_from_buffer(shared_state_pool, buffer_name="target")
        log_queue.put(
            {"type": "consensus_error", "error": consensus_error, "step": current_step}
        )
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


def evaluate_model_metrics(network, dataloader, device):
    """Evaluate one model on one dataloader and return accuracy/loss."""
    if dataloader is None:
        return None

    network.eval()
    correct = 0
    total = 0
    cumulative_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            loss = F.cross_entropy(outputs, labels)
            cumulative_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0, 0.0
    return 100 * correct / total, cumulative_loss / len(dataloader)


def evaluate_network(network, dataloader, step, node_index, split_name):
    """Evaluate one local node model and package its metrics for logging."""
    assert split_name in {"valid", "test"}, "split_name must be either 'valid' or 'test'"
    device = next(network.parameters()).device
    metrics = evaluate_model_metrics(network, dataloader, device)
    if metrics is None:
        return None
    accuracy, loss = metrics
    return EvaluationResult(
        split_name=split_name,
        node_index=node_index,
        step=step,
        accuracy=accuracy,
        loss=loss,
    )


def log_evaluation_result(log_queue, result):
    """Forward evaluation metrics to the central logging thread."""
    if result is None:
        return
    log_queue.put(
        {
            "type": result.split_name,
            "network_idx": result.node_index,
            "accuracy": result.accuracy,
            "loss": result.loss,
            "step": result.step,
        }
    )


def evaluate_local_models(
    local_node_indices,
    local_networks,
    valid_dataloaders_list,
    test_dataloader,
    local_steps_completed,
    log_queue,
):
    """Evaluate local models after one centralized gossip/broadcast round."""
    for local_idx, global_idx in enumerate(local_node_indices):
        valid_result = evaluate_network(
            local_networks[local_idx],
            valid_dataloaders_list[local_idx],
            local_steps_completed[local_idx],
            global_idx,
            "valid",
        )
        log_evaluation_result(log_queue, valid_result)

        test_result = evaluate_network(
            local_networks[local_idx],
            test_dataloader,
            local_steps_completed[local_idx],
            global_idx,
            "test",
        )
        log_evaluation_result(log_queue, test_result)


def get_avg_model(networks, calibration_loader=None):
    """Average mergeable parameters across nodes into one consensus model."""
    convergence_model = deepcopy(networks[0])
    mergeable_parameter_keys = [
        key for key, value in networks[0].named_parameters() if value.requires_grad and value.is_floating_point()
    ]

    with torch.no_grad():
        for key in mergeable_parameter_keys:
            mean_value = sum(network.state_dict()[key] for network in networks) / len(networks)
            convergence_model.state_dict()[key].copy_(mean_value.to(dtype=convergence_model.state_dict()[key].dtype))

    reestimate_batch_norm_stats(
        convergence_model,
        calibration_loader,
        next(convergence_model.parameters()).device,
    )
    return convergence_model


def evaluate_average_model_from_shared_state(
    config,
    shared_state_pool,
    test_dataloader,
    calibration_loader,
    step,
    log_queue,
    aggregation_device,
):
    """Evaluate the average model reconstructed directly from the shared target buffer."""
    if test_dataloader is None:
        return

    avg_model = create_model_from_config(config).to(aggregation_device)
    copy_mean_state_to_model(avg_model, shared_state_pool, device=aggregation_device, buffer_name="target")
    reestimate_batch_norm_stats(avg_model, calibration_loader, aggregation_device)
    metrics = evaluate_model_metrics(avg_model, test_dataloader, aggregation_device)
    if metrics is not None:
        accuracy, loss = metrics
        log_queue.put(
            {
                "type": "avg_model",
                "test_accuracy": accuracy,
                "test_loss": loss,
                "step": step,
            }
        )
    del avg_model
    torch.cuda.empty_cache()


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
        )


def worker_process(
    rank,
    world_size,
    networks_per_gpu,
    config,
    train_dataloaders_list,
    valid_dataloaders_list,
    test_dataloader_data,
    calibration_loader_data,
    aggregation_device,
    barrier,
    shared_state_pool,
    shared_reference_step,
    shared_node_steps,
    shared_next_eval_step,
    shared_should_eval,
    log_queue,
):
    """Worker process for the simulator-style multi-GPU training loop."""
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    start_idx, end_idx, local_node_indices = get_local_node_indices(
        rank,
        world_size,
        config.num_nodes,
        networks_per_gpu,
    )
    if rank == 0:
        print(f"[Rank {rank}] Managing networks {start_idx} to {end_idx-1} on {device}")

    worker_seed = config.seed + rank * 1000
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)

    local_networks, local_optimizers, local_schedulers = initialize_local_training_state(
        config,
        local_node_indices,
        device,
    )
    autocast_enabled, autocast_dtype, scaler = create_amp_runtime(config)

    local_steps_completed = [0] * len(local_node_indices)
    test_dataloader = test_dataloader_data
    calibration_loader = calibration_loader_data

    while max(local_steps_completed) < config.max_steps:
        train_local_models_for_round(
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
        )

        publish_local_step_progress(
            local_node_indices,
            local_steps_completed,
            shared_node_steps,
        )

        # Phase 1 barrier: every worker finished local optimization for this round.
        barrier.wait()

        publish_local_state(
            local_node_indices,
            local_networks,
            shared_state_pool,
        )

        # Phase 2 barrier: shared source buffers now contain all local states.
        barrier.wait()

        run_rank_zero_control_phase(
            rank=rank,
            config=config,
            shared_state_pool=shared_state_pool,
            log_queue=log_queue,
            aggregation_device=aggregation_device,
            shared_reference_step=shared_reference_step,
            shared_node_steps=shared_node_steps,
            shared_next_eval_step=shared_next_eval_step,
            shared_should_eval=shared_should_eval,
        )

        # Phase 3 barrier: rank 0 has finished writing the updated shared target buffer.
        barrier.wait()

        load_broadcast_parameters(local_node_indices, local_networks, shared_state_pool, device)

        # Phase 4 barrier: each worker has reloaded the centrally aggregated parameters.
        barrier.wait()

        run_optional_evaluation_phase(
            rank=rank,
            config=config,
            local_node_indices=local_node_indices,
            local_networks=local_networks,
            valid_dataloaders_list=valid_dataloaders_list,
            test_dataloader=test_dataloader,
            calibration_loader=calibration_loader,
            local_steps_completed=local_steps_completed,
            log_queue=log_queue,
            shared_state_pool=shared_state_pool,
            shared_reference_step=shared_reference_step,
            shared_should_eval=shared_should_eval,
            aggregation_device=aggregation_device,
        )

        # Phase 5 barrier: finish logging/evaluation before starting the next round.
        barrier.wait()

    if rank == 0:
        print(f"[Rank {rank}] Training completed!")


def split_dataloaders_by_rank(train_dataloaders, valid_dataloaders, world_size, networks_per_gpu, num_nodes):
    """Partition logical-node dataloaders according to worker ownership."""
    train_dataloaders_split = []
    valid_dataloaders_split = []
    for rank in range(world_size):
        start_idx, end_idx, _ = get_local_node_indices(rank, world_size, num_nodes, networks_per_gpu)
        train_dataloaders_split.append(train_dataloaders[start_idx:end_idx])
        valid_dataloaders_split.append(valid_dataloaders[start_idx:end_idx])
    return train_dataloaders_split, valid_dataloaders_split


def run_post_merge_rounds(
    config,
    final_networks,
    test_dataloader,
    calibration_loader,
    log_queue,
    aggregation_device,
):
    """Run optional pure-gossip rounds after decentralized training finishes."""
    if config.post_merge_rounds <= 0:
        return

    print(f"\n{'='*60}")
    print(f"Training completed at step {config.max_steps}")
    print(f"Starting {config.post_merge_rounds} post-merge rounds...")

    post_topology = config.end_topology if config.end_topology else config.gossip_topology
    print(f"Using topology: {post_topology}")
    print(f"{'='*60}\n")

    for merge_round in range(config.post_merge_rounds):
        if not is_disabled_topology(post_topology):
            gossip_matrix, _ = get_gossip_matrix(
                config.num_nodes,
                topology=post_topology,
                r=config.r_end,
                wandb=None,
                current_iter=config.max_steps,
                end_iter=config.max_steps,
            )
            gossip_update(final_networks, gossip_matrix)

        avg_test_acc = 0.0
        avg_test_loss = 0.0
        num_evaluated_networks = 0
        for network in final_networks:
            metrics = evaluate_model_metrics(network, test_dataloader, aggregation_device)
            if metrics is None:
                continue
            accuracy, loss = metrics
            avg_test_acc += accuracy
            avg_test_loss += loss
            num_evaluated_networks += 1

        if num_evaluated_networks > 0:
            avg_test_acc /= num_evaluated_networks
            avg_test_loss /= num_evaluated_networks

        avg_model_metrics = None
        avg_model = None
        if test_dataloader is not None:
            avg_model = get_avg_model(final_networks, calibration_loader=calibration_loader).to(aggregation_device)
            avg_model.eval()
            avg_model_metrics = evaluate_model_metrics(avg_model, test_dataloader, aggregation_device)

        avg_model_test_acc = avg_model_metrics[0] if avg_model_metrics is not None else 0.0
        avg_model_test_loss = avg_model_metrics[1] if avg_model_metrics is not None else 0.0
        consensus_error = compute_consensus_error(final_networks)

        log_queue.put(
            {
                "type": "post_merge",
                "post_merge_round": merge_round + 1,
                "post_merge_avg_test_accuracy": avg_test_acc,
                "post_merge_avg_test_loss": avg_test_loss,
                "post_merge_avg_model_test_accuracy": avg_model_test_acc,
                "post_merge_avg_model_test_loss": avg_model_test_loss,
                "post_merge_consensus_error": consensus_error,
                "step": config.max_steps,
            }
        )

        if merge_round == config.post_merge_rounds - 1:
            log_queue.put(
                {
                    "type": "post_merge_final",
                    "avg_test_accuracy": avg_test_acc,
                    "avg_test_loss": avg_test_loss,
                    "avg_model_test_accuracy": avg_model_test_acc,
                    "avg_model_test_loss": avg_model_test_loss,
                    "avg_model_test_accuracy - avg_test_accuracy": avg_model_test_acc - avg_test_acc,
                    "consensus_error": consensus_error,
                    "step": config.max_steps,
                }
            )
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS (after {config.post_merge_rounds} post-merge rounds):")
            print(f"  Avg Test Accuracy: {avg_test_acc:.2f}%")
            print(f"  Avg Model Test Accuracy: {avg_model_test_acc:.2f}%")
            print(f"  Consensus Error: {consensus_error:.6f}")
            print(f"{'='*60}\n")
        else:
            print(
                f"Post-merge round {merge_round + 1}/{config.post_merge_rounds}: "
                f"Avg Test Acc={avg_test_acc:.2f}%, "
                f"Avg Model Test Acc={avg_model_test_acc:.2f}%, "
                f"Consensus Error={consensus_error:.6f}"
            )

        if avg_model is not None:
            del avg_model
            torch.cuda.empty_cache()

    print(f"Post-merge completed: {config.post_merge_rounds} rounds\n")


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_training_config(
    *,
    num_nodes,
    max_steps,
    k_steps,
    eval_steps,
    model_name,
    pretrained,
    optimizer_name,
    lr,
    momentum,
    weight_decay,
    lr_scheduler,
    amp_enabled,
    amp_dtype,
    gossip_topology,
    r_start,
    r_end,
    r_schedule,
    point1,
    window_size,
    seed,
    diff_init,
    end_topology,
    post_merge_rounds,
    model_kwargs,
):
    return TrainingConfig(
        num_nodes=num_nodes,
        max_steps=max_steps,
        k_steps=k_steps,
        eval_steps=eval_steps,
        model_name=model_name,
        pretrained=pretrained,
        optimizer_name=optimizer_name,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        gossip_topology=gossip_topology,
        r_start=r_start,
        r_end=r_end,
        r_schedule=r_schedule,
        point1=point1,
        window_size=window_size,
        seed=seed,
        diff_init=diff_init,
        end_topology=end_topology,
        post_merge_rounds=post_merge_rounds,
        model_kwargs=model_kwargs,
    )


def main(
    dataset_path=DEFAULT_TRAINING_ARGS["dataset_path"],
    dataset_name=DEFAULT_TRAINING_ARGS["dataset_name"],
    num_nodes=DEFAULT_TRAINING_ARGS["num_nodes"],
    num_GPU=DEFAULT_TRAINING_ARGS["num_GPU"],
    k_steps=DEFAULT_TRAINING_ARGS["k_steps"],
    eval_steps=DEFAULT_TRAINING_ARGS["eval_steps"],
    gossip_topology=DEFAULT_TRAINING_ARGS["gossip_topology"],
    max_steps=DEFAULT_TRAINING_ARGS["max_steps"],
    pretrained=DEFAULT_TRAINING_ARGS["pretrained"],
    model_name=DEFAULT_TRAINING_ARGS["model_name"],
    optimizer_name=DEFAULT_TRAINING_ARGS["optimizer_name"],
    lr=DEFAULT_TRAINING_ARGS["lr"],
    batch_size=DEFAULT_TRAINING_ARGS["batch_size"],
    momentum=DEFAULT_TRAINING_ARGS["momentum"],
    weight_decay=DEFAULT_TRAINING_ARGS["weight_decay"],
    lr_scheduler=DEFAULT_TRAINING_ARGS["lr_scheduler"],
    amp_enabled=DEFAULT_TRAINING_ARGS["amp_enabled"],
    amp_dtype=DEFAULT_TRAINING_ARGS["amp_dtype"],
    node_datasize=DEFAULT_TRAINING_ARGS["node_datasize"],
    nonIID=DEFAULT_TRAINING_ARGS["nonIID"],
    alpha=DEFAULT_TRAINING_ARGS["alpha"],
    image_size=DEFAULT_TRAINING_ARGS["image_size"],
    data_loading_workers=DEFAULT_TRAINING_ARGS["data_loading_workers"],
    train_data_ratio=DEFAULT_TRAINING_ARGS["train_data_ratio"],
    project_name=DEFAULT_TRAINING_ARGS["project_name"],
    r_start=DEFAULT_TRAINING_ARGS["r_start"],
    r_end=DEFAULT_TRAINING_ARGS["r_end"],
    r_schedule=DEFAULT_TRAINING_ARGS["r_schedule"],
    point1=DEFAULT_TRAINING_ARGS["point1"],
    window_size=DEFAULT_TRAINING_ARGS["window_size"],
    load_pickle=DEFAULT_TRAINING_ARGS["load_pickle"],
    seed=DEFAULT_TRAINING_ARGS["seed"],
    diff_init=DEFAULT_TRAINING_ARGS["diff_init"],
    end_topology=DEFAULT_TRAINING_ARGS["end_topology"],
    post_merge_rounds=DEFAULT_TRAINING_ARGS["post_merge_rounds"],
    data_sampling_mode=DEFAULT_TRAINING_ARGS["data_sampling_mode"],
    strict_loading=DEFAULT_TRAINING_ARGS["strict_loading"],
    max_failure_ratio=DEFAULT_TRAINING_ARGS["max_failure_ratio"],
    num_gpus=DEFAULT_TRAINING_ARGS["num_gpus"],
    non_iid=DEFAULT_TRAINING_ARGS["non_iid"],
    model_kwargs=DEFAULT_TRAINING_ARGS["model_kwargs"],
):
    """
    Multi-GPU distributed training using torch.multiprocessing.

    Key hyperparameters:
    - k_steps: local optimization steps inside each communication round.
    - eval_steps: evaluation interval measured in the same local-step unit as k_steps/max_steps.
    - max_steps: total local training steps per node (also called max_iters in some notes).
    - gossip_topology: communication structure (e.g., exponential, random, exponential+random).
    - post_merge_rounds: extra gossip-only rounds after decentralized training to approximate global merge.
    - diff_init: whether local models start from different random initializations.

    Compatibility note:
    - load_pickle is accepted only for backward compatibility and does not affect training behavior.
    """
    raw_args = {
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "num_nodes": num_nodes,
            "num_GPU": num_GPU,
            "k_steps": k_steps,
            "eval_steps": eval_steps,
            "gossip_topology": gossip_topology,
            "max_steps": max_steps,
            "pretrained": pretrained,
            "model_name": model_name,
            "optimizer_name": optimizer_name,
            "lr": lr,
            "batch_size": batch_size,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "lr_scheduler": lr_scheduler,
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype,
            "node_datasize": node_datasize,
            "nonIID": nonIID,
            "alpha": alpha,
            "image_size": image_size,
            "data_loading_workers": data_loading_workers,
            "train_data_ratio": train_data_ratio,
            "project_name": project_name,
            "r_start": r_start,
            "r_end": r_end,
            "r_schedule": r_schedule,
            "point1": point1,
            "window_size": window_size,
            # Legacy compatibility only; kept so older CLI/YAML inputs still parse.
            "load_pickle": load_pickle,
            "seed": seed,
            "diff_init": diff_init,
            "end_topology": end_topology,
            "post_merge_rounds": post_merge_rounds,
            "data_sampling_mode": data_sampling_mode,
            "strict_loading": strict_loading,
            "max_failure_ratio": max_failure_ratio,
            "num_gpus": num_gpus,
            "non_iid": non_iid,
            "model_kwargs": model_kwargs,
        }
    for message in collect_alias_deprecation_warnings(raw_args):
        print(f"[Deprecated] {message}")

    normalized_args = validate_training_kwargs(raw_args)

    dataset_path = normalized_args["dataset_path"]
    dataset_name = normalized_args["dataset_name"]
    num_nodes = normalized_args["num_nodes"]
    num_gpus = normalized_args["num_GPU"]
    k_steps = normalized_args["k_steps"]
    eval_steps = normalized_args["eval_steps"]
    gossip_topology = normalized_args["gossip_topology"]
    max_steps = normalized_args["max_steps"]
    pretrained = normalized_args["pretrained"]
    model_name = normalized_args["model_name"]
    optimizer_name = normalized_args["optimizer_name"]
    lr = normalized_args["lr"]
    batch_size = normalized_args["batch_size"]
    momentum = normalized_args["momentum"]
    weight_decay = normalized_args["weight_decay"]
    lr_scheduler = normalized_args["lr_scheduler"]
    amp_enabled = normalized_args["amp_enabled"]
    amp_dtype = normalized_args["amp_dtype"]
    node_datasize = normalized_args["node_datasize"]
    non_iid = normalized_args["nonIID"]
    alpha = normalized_args["alpha"]
    image_size = normalized_args["image_size"]
    data_loading_workers = normalized_args["data_loading_workers"]
    train_data_ratio = normalized_args["train_data_ratio"]
    project_name = normalized_args["project_name"]
    r_start = normalized_args["r_start"]
    r_end = normalized_args["r_end"]
    r_schedule = normalized_args["r_schedule"]
    point1 = normalized_args["point1"]
    window_size = normalized_args["window_size"]
    seed = normalized_args["seed"]
    diff_init = normalized_args["diff_init"]
    end_topology = normalized_args["end_topology"]
    post_merge_rounds = normalized_args["post_merge_rounds"]
    data_sampling_mode = normalized_args["data_sampling_mode"]
    strict_loading = normalized_args["strict_loading"]
    max_failure_ratio = normalized_args["max_failure_ratio"]
    model_kwargs = {} if normalized_args["model_kwargs"] is None else dict(normalized_args["model_kwargs"])

    set_multiprocessing_spawn()
    set_seed(seed)

    layout = resolve_runtime_layout(num_nodes, num_gpus)
    device_policy = resolve_runtime_devices()
    world_size = layout.world_size
    networks_per_gpu = layout.networks_per_gpu
    print_runtime_layout(num_nodes, layout)

    if r_start is None:
        r_start = max(num_nodes - 1, 0)
    
    run = initialize_wandb_run(
        project_name=project_name,
        dataset_name=dataset_name,
        model_name=model_name,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
        k_steps=k_steps,
        eval_steps=eval_steps,
        max_steps=max_steps,
        optimizer_name=optimizer_name,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        batch_size=batch_size,
        gossip_topology=gossip_topology,
        node_datasize=node_datasize,
        non_iid=non_iid,
        alpha=alpha,
        data_sampling_mode=data_sampling_mode,
        image_size=image_size,
        r_schedule=r_schedule,
        pretrained=pretrained,
        point1=point1,
        window_size=window_size,
        seed=seed,
        diff_init=diff_init,
        end_topology=end_topology,
        post_merge_rounds=post_merge_rounds,
    )
    
    dataset_bundle = prepare_dataset_bundle(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        strict_loading=strict_loading,
        max_failure_ratio=max_failure_ratio,
        non_iid=non_iid,
        alpha=alpha,
        num_nodes=num_nodes,
        node_datasize=node_datasize,
        train_data_ratio=train_data_ratio,
        data_loading_workers=data_loading_workers,
        data_sampling_mode=data_sampling_mode,
        world_size=world_size,
    )
    train_subset = dataset_bundle.train_subset
    valid_subset = dataset_bundle.valid_subset
    test_loader = dataset_bundle.test_loader
    calibration_dataset = dataset_bundle.calibration_dataset
    nb_class = dataset_bundle.nb_class
    train_dataloaders = dataset_bundle.train_dataloaders
    valid_dataloaders = dataset_bundle.valid_dataloaders
    
    config = build_training_config(
        num_nodes=num_nodes,
        max_steps=max_steps,
        k_steps=k_steps,
        eval_steps=eval_steps,
        model_name=model_name,
        pretrained=pretrained,
        optimizer_name=optimizer_name,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        gossip_topology=gossip_topology,
        r_start=r_start,
        r_end=r_end,
        r_schedule=r_schedule,
        point1=point1,
        window_size=window_size,
        seed=seed,
        diff_init=diff_init,
        end_topology=end_topology,
        post_merge_rounds=post_merge_rounds,
        model_kwargs=model_kwargs,
    )
    shared_state_pool, calibration_loader = initialize_shared_training_state(
        config=config,
        num_nodes=num_nodes,
        batch_size=batch_size,
        train_subset=train_subset,
        create_model_from_config=create_model_from_config,
        create_shared_state_pool=create_shared_state_pool,
        create_bn_reestimation_loader=create_bn_reestimation_loader,
    )

    (
        log_queue,
        shared_reference_step,
        shared_node_steps,
        shared_next_eval_step,
        shared_should_eval,
        barrier,
    ) = create_runtime_synchronization(
        world_size=world_size,
        num_nodes=num_nodes,
        eval_steps=eval_steps,
        max_steps=max_steps,
        initialize_next_eval_step=initialize_next_eval_step,
    )
    log_thread = start_logging_thread(
        log_queue,
        max_steps * num_nodes,
        num_nodes,
        logging_process,
    )
    
    # Split dataloaders for each GPU
    train_dataloaders_split, valid_dataloaders_split = split_dataloaders_by_rank(
        train_dataloaders,
        valid_dataloaders,
        world_size,
        networks_per_gpu,
        num_nodes,
    )
    
    processes = spawn_worker_processes(
        worker_target=worker_process,
        world_size=world_size,
        networks_per_gpu=networks_per_gpu,
        config=config,
        train_dataloaders_split=train_dataloaders_split,
        valid_dataloaders_split=valid_dataloaders_split,
        test_loader=test_loader,
        calibration_loader=calibration_loader,
        aggregation_device=device_policy.aggregation_device,
        barrier=barrier,
        shared_state_pool=shared_state_pool,
        shared_reference_step=shared_reference_step,
        shared_node_steps=shared_node_steps,
        shared_next_eval_step=shared_next_eval_step,
        shared_should_eval=shared_should_eval,
        log_queue=log_queue,
    )
    failed_workers = wait_for_worker_processes(processes)
    finalize_worker_orchestration(
        failed_workers=failed_workers,
        log_queue=log_queue,
        log_thread=log_thread,
        wandb_finish=wandb.finish,
    )
    print(f"\n{'='*60}")
    print(f"Multi-GPU training completed!")
    print(f"{'='*60}\n")

    final_networks = reconstruct_final_networks_from_shared_state(
        config=config,
        num_nodes=num_nodes,
        shared_state_pool=shared_state_pool,
        create_model_from_config=create_model_from_config,
        copy_shared_buffer_to_model=copy_shared_buffer_to_model,
        device=device_policy.aggregation_device,
    )

    run_post_merge_rounds(
        config,
        final_networks,
        test_loader,
        calibration_loader,
        log_queue,
        device_policy.aggregation_device,
    )
    convergence_model = save_convergence_model(
        final_networks=final_networks,
        calibration_loader=calibration_loader,
        get_avg_model=get_avg_model,
        save_path="convergence_model_multi_gpu.pth",
    )

    log_queue.put(None)
    log_thread.join()

    wandb.finish()
    
    return convergence_model


if __name__ == "__main__":
    import argparse
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    
    parser = argparse.ArgumentParser(description="Multi-GPU Distributed Training Script")
    
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_TRAINING_ARGS["dataset_path"], help="Path to the dataset directory")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_TRAINING_ARGS["dataset_name"], help="Name of the dataset")
    parser.add_argument("--num_nodes", type=int, default=DEFAULT_TRAINING_ARGS["num_nodes"], help="Total decentralized nodes/models in training")
    parser.add_argument("--num_GPU", "--num_gpus", dest="num_GPU", type=int, default=DEFAULT_TRAINING_ARGS["num_GPU"], help="Number of GPU worker processes; nodes are split across these GPUs")
    parser.add_argument("--k_steps", type=int, default=DEFAULT_TRAINING_ARGS["k_steps"], help="Local optimization steps between adjacent gossip communication rounds")
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_TRAINING_ARGS["eval_steps"], help="Run evaluation only when the local-step reference crosses this interval; final step is always evaluated")
    parser.add_argument("--gossip_topology", type=str, default=DEFAULT_TRAINING_ARGS["gossip_topology"], help="Gossip topology: exponential (fixed), random (sampled on complete graph), or exponential+random (random on exponential backbone)")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_TRAINING_ARGS["max_steps"], help="Maximum local training steps per node (aka max_iters in some experiment notes)")
    parser.add_argument("--pretrained", type=str2bool, nargs="?", const=True, default=DEFAULT_TRAINING_ARGS["pretrained"], help="Use pretrained model (True or False)")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_TRAINING_ARGS["model_name"],
        help="Model name (default: resnet18_cifar_stem; use resnet18_imagenet_stem for torchvision-style ImageNet stem)",
    )
    parser.add_argument("--optimizer_name", type=str, default=DEFAULT_TRAINING_ARGS["optimizer_name"], help="Optimizer name")
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAINING_ARGS["lr"], help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_TRAINING_ARGS["batch_size"], help="Batch size")
    parser.add_argument("--momentum", type=float, default=DEFAULT_TRAINING_ARGS["momentum"], help="Momentum (for SGD)")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_TRAINING_ARGS["weight_decay"], help="Weight decay")
    parser.add_argument("--lr_scheduler", type=str, default=DEFAULT_TRAINING_ARGS["lr_scheduler"], help="Learning rate scheduler")
    parser.add_argument("--amp_enabled", type=str2bool, nargs="?", const=True, default=DEFAULT_TRAINING_ARGS["amp_enabled"], help="Enable automatic mixed precision")
    parser.add_argument("--amp_dtype", type=str, default=DEFAULT_TRAINING_ARGS["amp_dtype"], choices=["bf16", "fp16", "fp32"], help="AMP compute dtype; fp16 uses GradScaler, bf16 does not")
    parser.add_argument("--node_datasize", type=int, default=DEFAULT_TRAINING_ARGS["node_datasize"], help="Number of training samples assigned to each node")
    parser.add_argument("--nonIID", "--non_iid", dest="nonIID", type=str2bool, nargs="?", const=True, default=DEFAULT_TRAINING_ARGS["nonIID"], help="Enable non-IID partitioning across nodes")
    parser.add_argument("--alpha", type=float, default=DEFAULT_TRAINING_ARGS["alpha"], help="Dirichlet concentration for non-IID split; smaller means more heterogeneous")
    parser.add_argument("--image_size", type=int, default=DEFAULT_TRAINING_ARGS["image_size"], help="Image size")
    parser.add_argument("--strict_loading", type=str2bool, nargs="?", const=True, default=DEFAULT_TRAINING_ARGS["strict_loading"], help="Fail fast on TinyImageNet corrupted samples")
    parser.add_argument("--max_failure_ratio", type=float, default=DEFAULT_TRAINING_ARGS["max_failure_ratio"], help="Max tolerated TinyImageNet corrupted-sample ratio before raising an error")
    parser.add_argument("--data_loading_workers", type=int, default=DEFAULT_TRAINING_ARGS["data_loading_workers"], help="Number of workers for data loading")
    parser.add_argument("--train_data_ratio", type=float, default=DEFAULT_TRAINING_ARGS["train_data_ratio"], help="Train data ratio")
    parser.add_argument("--data_sampling_mode", type=str, default=DEFAULT_TRAINING_ARGS["data_sampling_mode"], choices=["fixed", "resample"], help="How each node turns its class distribution into concrete samples: fixed subset or resampled stream")
    parser.add_argument("--project_name", type=str, default=DEFAULT_TRAINING_ARGS["project_name"], help="Project name for logging")
    parser.add_argument("--r_start", type=float, default=DEFAULT_TRAINING_ARGS["r_start"], help="Starting extra-neighbor count r (excluding self)")
    parser.add_argument("--r_end", type=float, default=DEFAULT_TRAINING_ARGS["r_end"], help="Ending extra-neighbor count r (excluding self)")
    parser.add_argument("--r_schedule", type=str, default=DEFAULT_TRAINING_ARGS["r_schedule"], help="Schedule for r transition from r_start to r_end")
    parser.add_argument("--point1", type=float, default=DEFAULT_TRAINING_ARGS["point1"], help="Control point for truncate schedules (switch/start position)")
    parser.add_argument("--window_size", type=float, default=DEFAULT_TRAINING_ARGS["window_size"], help="Window size used by truncate_v2 schedule")
    parser.add_argument("--end_topology", type=str, default=DEFAULT_TRAINING_ARGS["end_topology"], help="Optional topology used during post-merge rounds; defaults to gossip_topology")
    parser.add_argument("--post_merge_rounds", type=int, default=DEFAULT_TRAINING_ARGS["post_merge_rounds"], help="Extra gossip merge rounds after decentralized training to approximate global merge")
    parser.add_argument("--load_pickle", dest="load_pickle", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=DEFAULT_TRAINING_ARGS['seed'], help='Random seed for reproducibility')
    parser.add_argument("--diff_init", type=str2bool, default=DEFAULT_TRAINING_ARGS["diff_init"], help="Use different random initialization across local models (for initialization sensitivity studies)")
    
    args = parser.parse_args()

    normalized_args = validate_training_kwargs(vars(args))
    main(**normalized_args)
