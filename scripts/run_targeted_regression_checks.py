#!/usr/bin/env python3
"""Lightweight regression checks for recent safe refactors."""

import inspect
from pathlib import Path

from torch.utils.data import Dataset

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


def main_check():
    check_cli_and_normalization()
    check_shared_dataset_helpers()
    maybe_check_real_datasets()
    print("TARGETED_REGRESSION_CHECKS_OK")


if __name__ == "__main__":
    main_check()
