import random

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core import (
    copy_shared_buffer_to_model,
    create_shared_state_pool,
    gossip_update,
    get_gossip_matrix,
)
from core.config_validation import DEFAULT_TRAINING_ARGS
from core.entrypoint import build_main_argument_parser, normalize_main_kwargs
from core.evaluation_runtime import get_avg_model, run_post_merge_rounds
from core.logging import logging_process
from core.model_runtime import (
    compute_consensus_error,
    create_bn_reestimation_loader,
    is_disabled_topology,
    reestimate_batch_norm_stats,
)
from core.worker_runtime import (
    create_amp_runtime,
    create_model_from_config,
    get_local_node_indices,
    initialize_local_training_state,
    initialize_next_eval_step,
    initialize_worker_seed,
    load_broadcast_parameters,
    publish_local_state,
    publish_local_step_progress,
    run_optional_evaluation_phase,
    run_rank_zero_control_phase,
    train_local_models_for_round,
)
from core.runtime_setup import (
    finalize_worker_orchestration,
    initialize_wandb_run,
    prepare_training_runtime,
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
import wandb

# Note: wandb.login() will be called in main() only on rank 0


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

    initialize_worker_seed(config, rank)

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
            is_disabled_topology=is_disabled_topology,
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
            reestimate_batch_norm_stats=reestimate_batch_norm_stats,
        )

        # Phase 5 barrier: finish logging/evaluation before starting the next round.
        barrier.wait()

    if rank == 0:
        print(f"[Rank {rank}] Training completed!")


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
    normalized_args = normalize_main_kwargs(raw_args)

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
    
    runtime = prepare_training_runtime(
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
        networks_per_gpu=networks_per_gpu,
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
        diff_init=diff_init,
        end_topology=end_topology,
        post_merge_rounds=post_merge_rounds,
        model_kwargs=model_kwargs,
        config_cls=TrainingConfig,
        get_local_node_indices=get_local_node_indices,
        create_model_from_config=create_model_from_config,
        create_shared_state_pool=create_shared_state_pool,
        create_bn_reestimation_loader=create_bn_reestimation_loader,
        initialize_next_eval_step=initialize_next_eval_step,
    )
    config = runtime.config
    test_loader = runtime.dataset_bundle.test_loader
    calibration_loader = runtime.calibration_loader
    shared_state_pool = runtime.shared_state_pool
    synchronization = runtime.synchronization
    log_thread = start_logging_thread(
        synchronization.log_queue,
        max_steps * num_nodes,
        num_nodes,
        logging_process,
    )

    processes = spawn_worker_processes(
        worker_target=worker_process,
        world_size=world_size,
        networks_per_gpu=networks_per_gpu,
        config=config,
        train_dataloaders_split=runtime.train_dataloaders_split,
        valid_dataloaders_split=runtime.valid_dataloaders_split,
        test_loader=test_loader,
        calibration_loader=calibration_loader,
        aggregation_device=device_policy.aggregation_device,
        barrier=synchronization.barrier,
        shared_state_pool=shared_state_pool,
        shared_reference_step=synchronization.shared_reference_step,
        shared_node_steps=synchronization.shared_node_steps,
        shared_next_eval_step=synchronization.shared_next_eval_step,
        shared_should_eval=synchronization.shared_should_eval,
        log_queue=synchronization.log_queue,
    )
    failed_workers = wait_for_worker_processes(processes)
    finalize_worker_orchestration(
        failed_workers=failed_workers,
        log_queue=synchronization.log_queue,
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
        synchronization.log_queue,
        device_policy.aggregation_device,
        compute_consensus_error=compute_consensus_error,
        get_gossip_matrix=get_gossip_matrix,
        gossip_update=gossip_update,
        is_disabled_topology=is_disabled_topology,
        reestimate_batch_norm_stats=reestimate_batch_norm_stats,
    )
    convergence_model = save_convergence_model(
        final_networks=final_networks,
        calibration_loader=calibration_loader,
        get_avg_model=get_avg_model,
        save_path="convergence_model_multi_gpu.pth",
    )
    synchronization.log_queue.put(None)
    log_thread.join()
    wandb.finish()
    
    return convergence_model


if __name__ == "__main__":
    parser = build_main_argument_parser()
    args = parser.parse_args()
    normalized_args = normalize_main_kwargs(vars(args))
    main(**normalized_args)
