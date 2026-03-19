"""Low-risk runtime setup helpers extracted from the main entrypoint."""

import os
import time
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.multiprocessing as mp
import wandb
from torch.multiprocessing import Barrier

from datasets import load_dataset
from datasets.dirichlet_sampling import (
    create_IID_preference,
    create_simple_preference,
    create_train_valid_dataloaders_multi,
    dirichlet_split,
)


TRAINING_CONFIG_KEYS = (
    "num_nodes",
    "max_steps",
    "k_steps",
    "eval_steps",
    "model_name",
    "pretrained",
    "optimizer_name",
    "lr",
    "momentum",
    "weight_decay",
    "lr_scheduler",
    "amp_enabled",
    "amp_dtype",
    "gossip_topology",
    "r_start",
    "r_end",
    "r_schedule",
    "point1",
    "window_size",
    "seed",
    "diff_init",
    "end_topology",
    "post_merge_rounds",
    "model_kwargs",
)


def _pick_kwargs(source: Dict[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    """Select a stable subset of keyword arguments from a source mapping."""
    return {key: source[key] for key in keys}


@dataclass(frozen=True)
class RuntimeLayout:
    world_size: int
    networks_per_gpu: int


@dataclass(frozen=True)
class RuntimeDevicePolicy:
    aggregation_device: str


@dataclass(frozen=True)
class DatasetBundle:
    train_subset: Any
    valid_subset: Any
    test_loader: Any
    calibration_dataset: Any
    nb_class: int
    train_dataloaders: Sequence[Any]
    valid_dataloaders: Sequence[Any]


@dataclass(frozen=True)
class RuntimeSynchronization:
    log_queue: Any
    shared_reference_step: Any
    shared_node_steps: Any
    shared_next_eval_step: Any
    shared_should_eval: Any
    barrier: Barrier


@dataclass(frozen=True)
class PreparedTrainingRuntime:
    dataset_bundle: DatasetBundle
    config: Any
    shared_state_pool: Any
    calibration_loader: Any
    synchronization: RuntimeSynchronization
    train_dataloaders_split: Sequence[Any]
    valid_dataloaders_split: Sequence[Any]


def set_multiprocessing_spawn() -> None:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def resolve_runtime_layout(num_nodes: int, requested_gpus: int) -> RuntimeLayout:
    available_gpus = torch.cuda.device_count()
    num_gpus = requested_gpus
    if num_gpus > available_gpus:
        print(
            f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available. "
            f"Using {available_gpus}."
        )
        num_gpus = available_gpus
    if num_gpus > num_nodes:
        print(
            f"Warning: Requested {num_gpus} GPU workers for {num_nodes} nodes. "
            f"Using {num_nodes} workers instead."
        )
        num_gpus = num_nodes
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")

    world_size = num_gpus
    networks_per_gpu = num_nodes // world_size
    return RuntimeLayout(world_size=world_size, networks_per_gpu=networks_per_gpu)


def resolve_runtime_devices() -> RuntimeDevicePolicy:
    # Keep the current behavior: central aggregation and orchestration stay on rank 0's GPU.
    return RuntimeDevicePolicy(aggregation_device="cuda:0")


def print_runtime_layout(num_nodes: int, layout: RuntimeLayout) -> None:
    print(f"{'='*60}")
    print("Multi-GPU Training Configuration:")
    print(f"  Total nodes: {num_nodes}")
    print(f"  GPUs: {layout.world_size}")
    print(f"  Networks per GPU: {layout.networks_per_gpu}")
    print(
        f"  Remainder networks on last GPU: "
        f"{num_nodes - layout.networks_per_gpu * layout.world_size}"
    )
    print(f"{'='*60}\n")


def _configure_wandb_step_metric(run):
    """Bind all logged metrics to the explicit simulator step axis."""
    run.define_metric("step")
    run.define_metric("*", step_metric="step")


def initialize_wandb_run(
    *,
    project_name: str,
    dataset_name: str,
    model_name: str,
    num_nodes: int,
    num_gpus: int,
    k_steps: int,
    eval_steps: int,
    max_steps: int,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    lr_scheduler: str,
    amp_enabled: bool,
    amp_dtype: str,
    batch_size: int,
    gossip_topology: Any,
    node_datasize: int,
    non_iid: bool,
    alpha: float,
    data_sampling_mode: str,
    image_size: int,
    r_schedule: str,
    pretrained: bool,
    point1: float,
    window_size: float,
    seed: int,
    diff_init: bool,
    end_topology: Any,
    post_merge_rounds: int,
):
    wandb_api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        try:
            wandb.login()
        except Exception as exc:
            print(f"Warning: wandb login skipped ({exc}). Falling back to offline mode.")
            os.environ.setdefault("WANDB_MODE", "offline")

    os.environ["WANDB_SILENT"] = "true"

    run = wandb.init(
        project=project_name,
        config={
            "dataset_name": dataset_name,
            "model_name": model_name,
            "num_nodes": num_nodes,
            "num_GPU": num_gpus,
            "k_steps": k_steps,
            "eval_steps": eval_steps,
            "max_steps": max_steps,
            "optimizer": optimizer_name,
            "lr": lr,
            "momentum": momentum if optimizer_name.lower() == "sgd" else None,
            "weight_decay": weight_decay,
            "lr_scheduler": lr_scheduler,
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype,
            "batch_size": batch_size,
            "gossip_topology": gossip_topology,
            "node_datasize": node_datasize,
            "nonIID": non_iid,
            "alpha": alpha,
            "data_sampling_mode": data_sampling_mode,
            "image_size": image_size,
            "r_schedule": r_schedule,
            "pretrain": pretrained,
            "point1": point1,
            "window_size": window_size,
            "seed": seed,
            "diff_init": diff_init,
            "end_topology": end_topology,
            "post_merge_rounds": post_merge_rounds,
        },
    )
    _configure_wandb_step_metric(run)
    run.name = "_".join(
        [f"{key}={val}" for key, val in run.config.items() if key != "dataset_name"]
    )
    return run


def prepare_dataset_bundle(
    *,
    dataset_path: str,
    dataset_name: str,
    image_size: int,
    batch_size: int,
    seed: int,
    strict_loading: bool,
    max_failure_ratio: float,
    non_iid: bool,
    alpha: float,
    num_nodes: int,
    node_datasize: int,
    train_data_ratio: float,
    data_loading_workers: int,
    data_sampling_mode: str,
    world_size: int,
) -> DatasetBundle:
    train_subset, valid_subset, test_loader, _, nb_class, calibration_dataset = load_dataset(
        root=dataset_path,
        name=dataset_name,
        image_size=image_size,
        train_batch_size=batch_size,
        valid_batch_size=batch_size,
        return_dataloader=False,
        seed=seed,
        strict_loading=strict_loading,
        max_failure_ratio=max_failure_ratio,
    )

    if non_iid:
        if alpha > 0:
            all_class_weights = dirichlet_split(num_nodes, nb_class, alpha, seed=seed)
        else:
            all_class_weights = create_simple_preference(
                num_nodes, nb_class, important_prob=0.8
            )
    else:
        all_class_weights = create_IID_preference(num_nodes, nb_class)

    train_dataloaders, valid_dataloaders = create_train_valid_dataloaders_multi(
        train_dataset=train_subset,
        valid_dataset=valid_subset,
        nb_dataloader=num_nodes,
        samples_per_loader=node_datasize,
        batch_size=batch_size,
        all_class_weights=all_class_weights,
        nb_class=nb_class,
        train_ratio=train_data_ratio,
        num_workers=data_loading_workers // world_size,
        seed=seed,
        sampling_mode=data_sampling_mode,
        valid_sampling_mode="fixed",
    )

    return DatasetBundle(
        train_subset=train_subset,
        valid_subset=valid_subset,
        test_loader=test_loader,
        calibration_dataset=calibration_dataset,
        nb_class=nb_class,
        train_dataloaders=train_dataloaders,
        valid_dataloaders=valid_dataloaders,
    )


def create_runtime_synchronization(
    *,
    world_size: int,
    num_nodes: int,
    eval_steps: int,
    max_steps: int,
    initialize_next_eval_step: Callable[[int, int], int],
):
    log_queue = mp.Queue()
    shared_reference_step = mp.Value("i", 0)
    shared_node_steps = mp.Array("i", [0] * num_nodes)
    shared_next_eval_step = mp.Value(
        "i", initialize_next_eval_step(eval_steps, max_steps)
    )
    shared_should_eval = mp.Value("i", 0)
    barrier = Barrier(world_size)
    return RuntimeSynchronization(
        log_queue=log_queue,
        shared_reference_step=shared_reference_step,
        shared_node_steps=shared_node_steps,
        shared_next_eval_step=shared_next_eval_step,
        shared_should_eval=shared_should_eval,
        barrier=barrier,
    )


def split_dataloaders_by_rank(
    train_dataloaders,
    valid_dataloaders,
    world_size: int,
    networks_per_gpu: int,
    num_nodes: int,
    get_local_node_indices: Callable[[int, int, int, int], Tuple[int, int, Iterable[int]]],
):
    """Partition logical-node dataloaders according to worker ownership."""
    train_dataloaders_split = []
    valid_dataloaders_split = []
    for rank in range(world_size):
        start_idx, end_idx, _ = get_local_node_indices(
            rank, world_size, num_nodes, networks_per_gpu
        )
        train_dataloaders_split.append(train_dataloaders[start_idx:end_idx])
        valid_dataloaders_split.append(valid_dataloaders[start_idx:end_idx])
    return train_dataloaders_split, valid_dataloaders_split


def build_training_config(*, config_cls: Callable[..., Any], **config_kwargs):
    return config_cls(**config_kwargs)


def _build_training_config_kwargs(runtime_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the normalized runtime fields that belong to TrainingConfig."""
    return _pick_kwargs(runtime_kwargs, TRAINING_CONFIG_KEYS)


def start_logging_thread(log_queue, total_steps: int, num_nodes: int, logging_process):
    log_thread = threading.Thread(
        target=logging_process,
        args=(log_queue, total_steps, num_nodes),
    )
    log_thread.start()
    return log_thread


def spawn_worker_processes(
    *,
    worker_target,
    world_size: int,
    networks_per_gpu: int,
    config,
    train_dataloaders_split,
    valid_dataloaders_split,
    test_loader,
    calibration_loader,
    aggregation_device: str,
    barrier,
    shared_state_pool,
    shared_reference_step,
    shared_node_steps,
    shared_next_eval_step,
    shared_should_eval,
    log_queue,
):
    print(f"Spawning {world_size} worker processes...")
    processes = []
    for rank in range(world_size):
        process = mp.Process(
            target=worker_target,
            args=(
                rank,
                world_size,
                networks_per_gpu,
                config,
                train_dataloaders_split[rank],
                valid_dataloaders_split[rank],
                test_loader,
                calibration_loader,
                aggregation_device,
                barrier,
                shared_state_pool,
                shared_reference_step,
                shared_node_steps,
                shared_next_eval_step,
                shared_should_eval,
                log_queue,
            ),
        )
        process.start()
        processes.append(process)
    return processes


def _terminate_process_group(processes, *, join_timeout_s: float = 5.0) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()

    deadline = time.monotonic() + max(0.0, float(join_timeout_s))
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)

    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()

    for process in processes:
        if process.is_alive():
            process.join(timeout=0.1)


def wait_for_worker_processes(processes, *, poll_interval_s: float = 0.2) -> List[Tuple[int, int]]:
    failed_workers = []
    remaining = list(processes)

    while remaining:
        alive_processes = []
        failure_detected = False

        for process in remaining:
            process.join(timeout=0)
            exitcode = process.exitcode
            if exitcode is None:
                alive_processes.append(process)
                continue
            if exitcode != 0:
                failed_workers.append((process.pid, exitcode))
                failure_detected = True

        if failure_detected:
            _terminate_process_group(alive_processes)
            for process in alive_processes:
                if process.exitcode not in (None, 0):
                    failed_workers.append((process.pid, process.exitcode))
            break

        if not alive_processes:
            break

        remaining = alive_processes
        time.sleep(max(0.0, float(poll_interval_s)))

    return failed_workers


def initialize_shared_training_state(
    *,
    config,
    num_nodes: int,
    batch_size: int,
    train_subset,
    create_model_from_config: Callable[..., Any],
    create_shared_state_pool: Callable[[Any, int], Any],
    create_bn_reestimation_loader: Callable[[Any, int], Any],
):
    reference_model = create_model_from_config(config, schema_only=True).cpu()
    shared_state_pool = create_shared_state_pool(reference_model, num_nodes)
    del reference_model

    calibration_loader = create_bn_reestimation_loader(train_subset, batch_size)
    return shared_state_pool, calibration_loader


def prepare_training_runtime(
    *,
    dataset_path: str,
    dataset_name: str,
    image_size: int,
    batch_size: int,
    seed: int,
    strict_loading: bool,
    max_failure_ratio: float,
    non_iid: bool,
    alpha: float,
    num_nodes: int,
    node_datasize: int,
    train_data_ratio: float,
    data_loading_workers: int,
    data_sampling_mode: str,
    world_size: int,
    networks_per_gpu: int,
    max_steps: int,
    k_steps: int,
    eval_steps: int,
    model_name: str,
    pretrained: bool,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    lr_scheduler: str,
    amp_enabled: bool,
    amp_dtype: str,
    gossip_topology: Any,
    r_start: float,
    r_end: float,
    r_schedule: str,
    point1: float,
    window_size: float,
    diff_init: bool,
    end_topology: Optional[str],
    post_merge_rounds: int,
    model_kwargs: Dict[str, Any],
    config_cls: Callable[..., Any],
    get_local_node_indices: Callable[[int, int, int, int], Tuple[int, int, Iterable[int]]],
    create_model_from_config: Callable[..., Any],
    create_shared_state_pool: Callable[[Any, int], Any],
    create_bn_reestimation_loader: Callable[[Any, int], Any],
    initialize_next_eval_step: Callable[[int, int], int],
) -> PreparedTrainingRuntime:
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
    config = build_training_config(
        config_cls=config_cls,
        **_build_training_config_kwargs(locals()),
    )
    shared_state_pool, calibration_loader = initialize_shared_training_state(
        config=config,
        num_nodes=num_nodes,
        batch_size=batch_size,
        train_subset=dataset_bundle.train_subset,
        create_model_from_config=create_model_from_config,
        create_shared_state_pool=create_shared_state_pool,
        create_bn_reestimation_loader=create_bn_reestimation_loader,
    )
    synchronization = create_runtime_synchronization(
        world_size=world_size,
        num_nodes=num_nodes,
        eval_steps=eval_steps,
        max_steps=max_steps,
        initialize_next_eval_step=initialize_next_eval_step,
    )
    train_dataloaders_split, valid_dataloaders_split = split_dataloaders_by_rank(
        dataset_bundle.train_dataloaders,
        dataset_bundle.valid_dataloaders,
        world_size,
        networks_per_gpu,
        num_nodes,
        get_local_node_indices,
    )
    return PreparedTrainingRuntime(
        dataset_bundle=dataset_bundle,
        config=config,
        shared_state_pool=shared_state_pool,
        calibration_loader=calibration_loader,
        synchronization=synchronization,
        train_dataloaders_split=train_dataloaders_split,
        valid_dataloaders_split=valid_dataloaders_split,
    )


def finalize_worker_orchestration(
    *,
    failed_workers: Sequence[Tuple[int, int]],
    log_queue,
    log_thread,
    wandb_finish: Callable[[], None],
) -> None:
    if not failed_workers:
        return

    worker_msg = ", ".join([f"pid={pid}, exitcode={code}" for pid, code in failed_workers])
    log_queue.put(None)
    log_thread.join()
    wandb_finish()
    raise RuntimeError(f"Worker process failure detected: {worker_msg}")


def reconstruct_final_networks_from_shared_state(
    *,
    config,
    num_nodes: int,
    shared_state_pool,
    create_model_from_config: Callable[..., Any],
    copy_shared_buffer_to_model: Callable[..., None],
    device: str = "cuda:0",
) -> List[Any]:
    final_networks = []
    for node_index in range(num_nodes):
        network = create_model_from_config(config).to(device)
        copy_shared_buffer_to_model(
            network,
            shared_state_pool,
            node_index,
            device=device,
            buffer_name="target",
        )
        network.eval()
        final_networks.append(network)
    return final_networks


def save_convergence_model(
    *,
    final_networks: Sequence[Any],
    calibration_loader,
    get_avg_model: Callable[..., Any],
    save_path: str,
):
    convergence_model = get_avg_model(final_networks, calibration_loader=calibration_loader)
    torch.save(convergence_model.state_dict(), save_path)
    return convergence_model
