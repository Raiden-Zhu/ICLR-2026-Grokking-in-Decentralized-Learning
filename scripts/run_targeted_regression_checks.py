#!/usr/bin/env python3
"""Lightweight regression checks for safe refactors."""

import inspect
import queue
import yaml
from pathlib import Path

from torch.utils.data import Dataset

import core.logging as logging_module
from core.entrypoint import build_main_argument_parser, normalize_main_kwargs
from datasets.common import DatasetView, build_split_dataset_views, finalize_classification_dataset
from datasets.cifar10 import load_cifar10
from datasets.cifar100 import load_cifar100
from datasets.tinyimagenet import load_tinyimagenet
from main_multi_GPU import main, _normalize_main_runtime_args

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets" / "downloads"


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


class DummyDataset(Dataset):
    def __init__(self, size=20, num_classes=4):
        self.samples = list(range(size))
        self.targets = [i % num_classes for i in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


def check_cli_and_normalization():
    parser = build_main_argument_parser()
    args = parser.parse_args([])
    main_params = set(inspect.signature(main).parameters)
    arg_keys = set(vars(args))
    assert_true(arg_keys <= main_params, "Parser args must remain accepted by main()")

    normalized = normalize_main_kwargs(vars(args))
    normalized_runtime = _normalize_main_runtime_args(vars(args))
    for key, value in normalized.items():
        if key == "model_kwargs":
            expected = {} if value is None else value
            assert_true(
                normalized_runtime[key] == expected,
                f"Normalization mismatch on {key}: {normalized_runtime[key]!r} vs {expected!r}",
            )
            continue
        assert_true(
            normalized_runtime[key] == value,
            f"Normalization mismatch on {key}: {normalized_runtime[key]!r} vs {value!r}",
        )

    print("[OK] CLI/main compatibility and normalization")


def check_shared_dataset_helpers():
    base = DummyDataset()
    train_subset, valid_subset, calibration_view = build_split_dataset_views(
        base,
        train_transform=lambda x: ("train", x),
        eval_transform=lambda x: ("eval", x),
        split=0.8,
        seed=42,
    )
    assert_true(len(train_subset) + len(valid_subset) == len(base), "Split helper changed sample count")
    assert_true(isinstance(calibration_view, DatasetView), "Calibration view must remain a DatasetView")
    assert_true(calibration_view[0][0][0] == "eval", "Calibration view must preserve eval transform")

    dataset_pack = finalize_classification_dataset(
        train_subset=train_subset,
        valid_subset=valid_subset,
        test_set=DatasetView(base, transform=lambda x: ("test", x)),
        calibration_view=calibration_view,
        image_shape=(3, 32, 32),
        num_classes=4,
        train_batch_size=2,
        valid_batch_size=2,
        return_dataloader=False,
    )
    dataloader_pack = finalize_classification_dataset(
        train_subset=train_subset,
        valid_subset=valid_subset,
        test_set=DatasetView(base, transform=lambda x: ("test", x)),
        calibration_view=calibration_view,
        image_shape=(3, 32, 32),
        num_classes=4,
        train_batch_size=2,
        valid_batch_size=2,
        return_dataloader=True,
    )
    assert_true(len(dataset_pack) == 6, "Dataset helper must preserve 6-tuple return shape")
    assert_true(len(dataloader_pack) == 6, "Dataloader helper must preserve 6-tuple return shape")

    print("[OK] Shared dataset helper semantics")


def _check_dataset_tuple(name, loader_fn, expected_shape, expected_classes):
    data = loader_fn(str(DATASETS_DIR), return_dataloader=False)
    assert_true(len(data) == 6, f"{name} loader must preserve 6-tuple structure")
    train_subset, valid_subset, _test_loader, image_shape, num_classes, calibration_view = data
    assert_true(train_subset is not None, f"{name} train subset missing")
    assert_true(valid_subset is not None, f"{name} valid subset missing for default split<1")
    assert_true(image_shape == expected_shape, f"{name} image shape changed: {image_shape!r}")
    assert_true(num_classes == expected_classes, f"{name} class count changed: {num_classes!r}")
    assert_true(isinstance(calibration_view, DatasetView), f"{name} calibration view type changed")
    print(f"[OK] {name} loader structure")


def maybe_check_real_datasets():
    _check_dataset_tuple("CIFAR-10", load_cifar10, (3, 32, 32), 10)

    cifar100_dir = DATASETS_DIR / "cifar-100-python"
    if cifar100_dir.exists():
        _check_dataset_tuple("CIFAR-100", load_cifar100, (3, 32, 32), 100)
    else:
        print("[SKIP] CIFAR-100 cache not found locally")

    tiny_train = DATASETS_DIR / "tiny-imagenet" / "tiny-imagenet_train.npz"
    tiny_val = DATASETS_DIR / "tiny-imagenet" / "tiny-imagenet_val.npz"
    if tiny_train.exists() and tiny_val.exists():
        _check_dataset_tuple("TinyImageNet", load_tinyimagenet, (3, 224, 224), 200)
    else:
        print("[SKIP] TinyImageNet cache not found locally")


def check_config_launcher_defaults():
    config_paths = sorted((ROOT_DIR / "configs").glob("**/*.yaml"))
    hardcoded_python3 = []
    for path in config_paths:
        data = yaml.safe_load(path.read_text()) or {}
        if data.get("run", {}).get("python") == "python3":
            hardcoded_python3.append(path.relative_to(ROOT_DIR).as_posix())

    assert_true(
        not hardcoded_python3,
        f"Configs must not hardcode run.python=python3: {hardcoded_python3!r}",
    )
    print("[OK] Config launcher interpreter defaults")


def check_logging_step_alignment():
    recorded_payloads = []

    class DummyTqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total")

        def update(self, _n):
            return None

        def close(self):
            return None

    original_log = logging_module.wandb.log
    original_tqdm = logging_module.tqdm
    try:
        logging_module.wandb.log = lambda payload: recorded_payloads.append(dict(payload))
        logging_module.tqdm = DummyTqdm

        log_queue = queue.Queue()
        items = [
            {"type": "train", "network_idx": 0, "loss": 1.0, "accuracy": 10.0, "step": 100, "k_steps": 50},
            {"type": "train", "network_idx": 1, "loss": 3.0, "accuracy": 30.0, "step": 100, "k_steps": 50},
            {"type": "valid", "network_idx": 0, "loss": 1.5, "accuracy": 15.0, "step": 100, "k_steps": 50},
            {"type": "valid", "network_idx": 1, "loss": 2.5, "accuracy": 25.0, "step": 100, "k_steps": 50},
            {"type": "test", "network_idx": 0, "loss": 4.0, "accuracy": 40.0, "step": 100, "k_steps": 50},
            {"type": "avg_model", "test_accuracy": 65.0, "test_loss": 1.2, "step": 100, "k_steps": 50},
            {"type": "test", "network_idx": 1, "loss": 2.0, "accuracy": 60.0, "step": 100, "k_steps": 50},
        ]
        for item in items:
            log_queue.put(item)
        log_queue.put(None)

        logging_module.logging_process(log_queue, total_steps=1000, num_nodes=2)
    finally:
        logging_module.wandb.log = original_log
        logging_module.tqdm = original_tqdm

    train_payload = next((p for p in recorded_payloads if "avg_train_accuracy" in p), None)
    valid_payload = next((p for p in recorded_payloads if "avg_valid_accuracy" in p), None)
    test_payload = next((p for p in recorded_payloads if "test_accuracy/network_0" in p), None)
    merged_payload = next(
        (
            p for p in recorded_payloads
            if "avg_test_accuracy" in p and "avg_model_test_accuracy" in p
        ),
        None,
    )
    gap_only_payloads = [
        p for p in recorded_payloads
        if set(p.keys()) == {"avg_model_test_accuracy - avg_test_accuracy", "step"}
    ]
    standalone_avg_model_payloads = [
        p for p in recorded_payloads
        if "avg_model_test_accuracy" in p and "avg_test_accuracy" not in p
    ]

    assert_true(train_payload is not None, "Train payload must be emitted")
    assert_true(valid_payload is not None, "Valid payload must be emitted")
    assert_true(test_payload is not None, "Per-node test payload must be emitted")
    assert_true(merged_payload is not None, "Merged avg-test/avg-model payload must be emitted")
    assert_true(train_payload["step"] == 100, f"Train step drifted: {train_payload!r}")
    assert_true(valid_payload["step"] == 100, f"Valid step drifted: {valid_payload!r}")
    assert_true(test_payload["step"] == 100, f"Test step drifted: {test_payload!r}")
    assert_true(merged_payload["step"] == 100, f"Merged payload step drifted: {merged_payload!r}")
    assert_true(train_payload["round"] == 2, f"Train round drifted: {train_payload!r}")
    assert_true(valid_payload["round"] == 2, f"Valid round drifted: {valid_payload!r}")
    assert_true(test_payload["round"] == 2, f"Test round drifted: {test_payload!r}")
    assert_true(merged_payload["round"] == 2, f"Merged payload round drifted: {merged_payload!r}")
    assert_true(
        "avg_test_accuracy" not in test_payload,
        f"Per-node test payload should not duplicate avg test metrics: {test_payload!r}",
    )
    assert_true(
        merged_payload["avg_model_test_accuracy - avg_test_accuracy"] == 15.0,
        f"Unexpected mergeability gap payload: {merged_payload!r}",
    )
    assert_true(
        not gap_only_payloads,
        f"Gap metric must not be emitted as a standalone payload: {gap_only_payloads!r}",
    )
    assert_true(
        not standalone_avg_model_payloads,
        f"Avg-model metrics must not be emitted without the matching avg-test payload: {standalone_avg_model_payloads!r}",
    )

    print("[OK] Logging step alignment and mergeability payloads")


def main_check():
    check_cli_and_normalization()
    check_shared_dataset_helpers()
    check_config_launcher_defaults()
    check_logging_step_alignment()
    maybe_check_real_datasets()
    print("TARGETED_REGRESSION_CHECKS_OK")


if __name__ == "__main__":
    main_check()
