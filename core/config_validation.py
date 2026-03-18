"""Shared defaults and validation for training entrypoints."""

from typing import Any, Dict, List

DEFAULT_TRAINING_ARGS: Dict[str, Any] = {
    "dataset_path": "datasets/downloads",
    "dataset_name": "cifar100",
    "num_nodes": 16,
    "num_GPU": 1,
    "k_steps": 100,
    "eval_steps": 100,
    "gossip_topology": "random",
    "max_steps": 6000,
    "pretrained": True,
    "model_name": "resnet18_cifar_stem",
    "optimizer_name": "adamw",
    "lr": 1e-3,
    "batch_size": 64,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "lr_scheduler": "none",
    "amp_enabled": True,
    "amp_dtype": "bf16",
    "node_datasize": 4096,
    "nonIID": True,
    "alpha": 0.1,
    "image_size": 32,
    "data_loading_workers": 8,
    "train_data_ratio": 0.8,
    "project_name": "SingleMerge_MultiGPU",
    "r_start": 1.0,
    "r_end": 3.0,
    "r_schedule": "fixed",
    "point1": 0.0,
    "window_size": 0.1,
    # Legacy no-op kept only so older CLI/YAML configs continue to parse.
    "load_pickle": 1,
    "seed": 42,
    "diff_init": False,
    "end_topology": None,
    "post_merge_rounds": 0,
    "data_sampling_mode": "fixed",
    "strict_loading": False,
    "max_failure_ratio": 0.005,
    "num_gpus": None,
    "non_iid": None,
    "model_kwargs": None,
}

SUPPORTED_OPTIMIZERS = {"sgd", "adam", "adamw"}
SUPPORTED_LR_SCHEDULERS = {
    "cosine",
    "step",
    "warmup_cosine",
    "constant_then_zero",
    "none",
}
SUPPORTED_DATA_SAMPLING_MODES = {"fixed", "resample"}
SUPPORTED_AMP_DTYPES = {"bf16", "fp16", "fp32"}
SUPPORTED_R_SCHEDULES = {
    "fixed",
    "linear",
    "cosine",
    "slow_decrease",
    "slow_grow",
    "truncate",
    "truncate_v2",
}
SUPPORTED_BASE_TOPOLOGIES = {"ring", "left", "complete", "exponential"}
SUPPORTED_GOSSIP_TOPOLOGIES = {
    "none",
    "localtraining",
    "ring",
    "left",
    "complete",
    "exponential",
    "random",
    "ringtocomplete",
    "completetoring",
    "lefttocomplete",
    "completetorandom",
}
ALIAS_KEYS = {
    "num_gpus": "num_GPU",
    "non_iid": "nonIID",
}
KNOWN_TRAINING_KEYS = set(DEFAULT_TRAINING_ARGS)


def collect_alias_deprecation_warnings(values: Dict[str, Any]) -> List[str]:
    warnings = []
    for alias, canonical in ALIAS_KEYS.items():
        if alias not in values or values[alias] is None:
            continue
        warnings.append(
            f"Config key '{alias}' is deprecated and will be removed in a future cleanup; use '{canonical}' instead."
        )
    return warnings


def _normalize_aliases(values: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(values)
    for alias, canonical in ALIAS_KEYS.items():
        if alias in normalized and normalized[alias] is not None:
            if canonical in normalized and normalized[canonical] not in (None, normalized[alias]):
                raise ValueError(
                    f"Conflicting values for {canonical} and {alias}: "
                    f"{normalized[canonical]!r} vs {normalized[alias]!r}"
                )
            normalized[canonical] = normalized[alias]
        normalized.pop(alias, None)
    return normalized


def _normalize_optional_string(value: Any) -> Any:
    if value is None:
        return None
    return str(value).strip().lower()


def _validate_topology_value(value: Any, *, field_name: str) -> Any:
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in SUPPORTED_GOSSIP_TOPOLOGIES:
        return normalized

    if "+" in normalized:
        parts = normalized.split("+")
        if len(parts) == 2 and parts[1] == "random" and parts[0] in SUPPORTED_BASE_TOPOLOGIES:
            return normalized

    supported = ", ".join(sorted(SUPPORTED_GOSSIP_TOPOLOGIES))
    raise ValueError(
        f"Unsupported {field_name}={value!r}. Supported: {supported}, or 'base+random' "
        f"where base is one of {', '.join(sorted(SUPPORTED_BASE_TOPOLOGIES))}"
    )


def normalize_training_kwargs(values: Dict[str, Any], apply_defaults: bool = True) -> Dict[str, Any]:
    normalized = dict(DEFAULT_TRAINING_ARGS) if apply_defaults else {}
    normalized.update(_normalize_aliases(values))

    if "optimizer_name" in normalized and normalized["optimizer_name"] is not None:
        normalized["optimizer_name"] = str(normalized["optimizer_name"]).lower()
    if "lr_scheduler" in normalized and normalized["lr_scheduler"] is not None:
        normalized["lr_scheduler"] = str(normalized["lr_scheduler"]).lower()
    if "data_sampling_mode" in normalized and normalized["data_sampling_mode"] is not None:
        normalized["data_sampling_mode"] = str(normalized["data_sampling_mode"]).lower()
    if "gossip_topology" in normalized:
        normalized["gossip_topology"] = _normalize_optional_string(normalized["gossip_topology"])
    if "end_topology" in normalized:
        normalized["end_topology"] = _normalize_optional_string(normalized["end_topology"])
    if "r_schedule" in normalized and normalized["r_schedule"] is not None:
        normalized["r_schedule"] = str(normalized["r_schedule"]).lower()

    return normalized


def validate_training_kwargs(values: Dict[str, Any], require_all: bool = True) -> Dict[str, Any]:
    normalized = normalize_training_kwargs(values, apply_defaults=require_all)

    unknown_keys = sorted(set(normalized) - KNOWN_TRAINING_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown training config keys: {', '.join(unknown_keys)}")

    required_positive_ints = [
        "num_nodes",
        "num_GPU",
        "k_steps",
        "max_steps",
        "batch_size",
        "node_datasize",
        "image_size",
    ]
    if require_all:
        required_positive_ints.append("seed")
    for key in required_positive_ints:
        if key in normalized and normalized[key] is not None and int(normalized[key]) <= 0:
            raise ValueError(f"{key} must be > 0, got {normalized[key]!r}")

    if "data_loading_workers" in normalized and normalized["data_loading_workers"] is not None:
        if int(normalized["data_loading_workers"]) < 0:
            raise ValueError(
                f"data_loading_workers must be >= 0, got {normalized['data_loading_workers']!r}"
            )

    if "eval_steps" in normalized and normalized["eval_steps"] is not None:
        int(normalized["eval_steps"])

    if "train_data_ratio" in normalized and normalized["train_data_ratio"] is not None:
        train_data_ratio = float(normalized["train_data_ratio"])
        if not 0 < train_data_ratio <= 1:
            raise ValueError(
                f"train_data_ratio must be in (0, 1], got {normalized['train_data_ratio']!r}"
            )

    if "max_failure_ratio" in normalized and normalized["max_failure_ratio"] is not None:
        max_failure_ratio = float(normalized["max_failure_ratio"])
        if not 0 <= max_failure_ratio <= 1:
            raise ValueError(
                f"max_failure_ratio must be in [0, 1], got {normalized['max_failure_ratio']!r}"
            )

    if "post_merge_rounds" in normalized and normalized["post_merge_rounds"] is not None:
        if int(normalized["post_merge_rounds"]) < 0:
            raise ValueError(
                f"post_merge_rounds must be >= 0, got {normalized['post_merge_rounds']!r}"
            )

    if "optimizer_name" in normalized and normalized["optimizer_name"] is not None:
        if normalized["optimizer_name"] not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer_name={normalized['optimizer_name']!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_OPTIMIZERS))}"
            )

    if "lr_scheduler" in normalized and normalized["lr_scheduler"] is not None:
        if normalized["lr_scheduler"] not in SUPPORTED_LR_SCHEDULERS:
            raise ValueError(
                f"Unsupported lr_scheduler={normalized['lr_scheduler']!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_LR_SCHEDULERS))}"
            )

    if "data_sampling_mode" in normalized and normalized["data_sampling_mode"] is not None:
        if normalized["data_sampling_mode"] not in SUPPORTED_DATA_SAMPLING_MODES:
            raise ValueError(
                f"Unsupported data_sampling_mode={normalized['data_sampling_mode']!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_DATA_SAMPLING_MODES))}"
            )

    if "amp_dtype" in normalized and normalized["amp_dtype"] is not None:
        amp_dtype = str(normalized["amp_dtype"]).lower()
        if amp_dtype not in SUPPORTED_AMP_DTYPES:
            raise ValueError(
                f"Unsupported amp_dtype={normalized['amp_dtype']!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_AMP_DTYPES))}"
            )
        normalized["amp_dtype"] = amp_dtype

    if "r_schedule" in normalized and normalized["r_schedule"] is not None:
        if normalized["r_schedule"] not in SUPPORTED_R_SCHEDULES:
            raise ValueError(
                f"Unsupported r_schedule={normalized['r_schedule']!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_R_SCHEDULES))}"
            )

    if "gossip_topology" in normalized:
        normalized["gossip_topology"] = _validate_topology_value(
            normalized["gossip_topology"],
            field_name="gossip_topology",
        )

    if "end_topology" in normalized:
        normalized["end_topology"] = _validate_topology_value(
            normalized["end_topology"],
            field_name="end_topology",
        )

    return normalized
