"""CLI helpers for the active multi-GPU training entrypoint."""

import argparse
from typing import Any, Dict

from core.config_validation import (
    DEFAULT_TRAINING_ARGS,
    collect_alias_deprecation_warnings,
    validate_training_kwargs,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"yes", "true", "t", "y", "1"}:
        return True
    if lowered in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def normalize_main_kwargs(raw_args: Dict[str, Any]) -> Dict[str, Any]:
    for message in collect_alias_deprecation_warnings(raw_args):
        print(f"[Deprecated] {message}")
    return validate_training_kwargs(raw_args)


def build_main_argument_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_ARGS["seed"], help="Random seed for reproducibility")
    parser.add_argument("--diff_init", type=str2bool, default=DEFAULT_TRAINING_ARGS["diff_init"], help="Use different random initialization across local models (for initialization sensitivity studies)")
    return parser
