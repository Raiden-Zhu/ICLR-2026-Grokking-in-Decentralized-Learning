import glob
import os
import shutil
import urllib.request
import zipfile
from typing import Tuple

import numpy as np
import torchvision.transforms as tfs
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .common import DatasetView


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as progress:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=progress.update_to)


def safe_extract_zip(zip_ref: zipfile.ZipFile, destination: str) -> None:
    destination_real = os.path.realpath(destination)
    os.makedirs(destination_real, exist_ok=True)

    for member in zip_ref.infolist():
        member_path = os.path.realpath(os.path.join(destination_real, member.filename))
        if member_path != destination_real and not member_path.startswith(
            destination_real + os.sep
        ):
            raise RuntimeError(
                f"Unsafe path detected while extracting TinyImageNet archive: {member.filename}"
            )

    zip_ref.extractall(destination_real)


class TinyImageNet(Dataset):
    base_url = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
    train_cache_name = "tiny-imagenet_train.npz"
    val_cache_name = "tiny-imagenet_val.npz"
    legacy_train_cache_name = "tiny-imagenet_train.pkl"
    legacy_val_cache_name = "tiny-imagenet_val.pkl"

    def __init__(
        self,
        root="./data",
        train=True,
        transform=None,
        download=False,
        strict_loading=False,
        max_failure_ratio=0.005,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.tiny_imagenet_dir = os.path.join(self.root, "tiny-imagenet")
        self.strict_loading = bool(strict_loading)
        self.max_failure_ratio = float(max_failure_ratio)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it")

        data_path = self._processed_path(train)
        dat = self._load_processed_data(data_path)
        self.data = dat["data"]
        self.targets = dat["targets"]

    def __getitem__(self, item):
        data, targets = Image.fromarray(self.data[item]), self.targets[item]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        return len(self.data)

    def _processed_path(self, train: bool) -> str:
        filename = self.train_cache_name if train else self.val_cache_name
        return os.path.join(self.tiny_imagenet_dir, filename)

    def _legacy_processed_path(self, train: bool) -> str:
        filename = self.legacy_train_cache_name if train else self.legacy_val_cache_name
        return os.path.join(self.tiny_imagenet_dir, filename)

    def _check_exists(self) -> bool:
        return os.path.exists(self._processed_path(True)) and os.path.exists(
            self._processed_path(False)
        )

    def _load_processed_data(self, data_path: str):
        with np.load(data_path, allow_pickle=False) as data:
            return {
                "data": data["data"],
                "targets": data["targets"],
            }

    def _save_processed_data(self, output_path: str, data: np.ndarray, targets: np.ndarray) -> None:
        tmp_path = f"{output_path}.tmp.npz"
        np.savez_compressed(tmp_path, data=data, targets=targets)
        os.replace(tmp_path, output_path)

    def _report_loading_failures(
        self, split_name, failed_count, total_count, failed_examples
    ):
        if failed_count <= 0:
            return

        failure_ratio = failed_count / max(total_count, 1)
        print(
            f"[TinyImageNet] {split_name}: skipped {failed_count}/{total_count} corrupted images "
            f"({failure_ratio:.2%})."
        )
        for image_path, error_msg in failed_examples[:5]:
            print(f"  - {image_path}: {error_msg}")

        if failure_ratio > self.max_failure_ratio:
            raise RuntimeError(
                f"[TinyImageNet] {split_name} corruption ratio {failure_ratio:.2%} exceeds "
                f"max_failure_ratio={self.max_failure_ratio:.2%}."
            )

    def _process_train_data(self, train_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        print("Processing training data...")
        classes = sorted(os.listdir(train_dir))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        images = []
        labels = []
        total_images = 0
        failed_images = 0
        failed_examples = []

        for class_dir in tqdm(classes):
            class_path = os.path.join(train_dir, class_dir, "images")
            class_idx = class_to_idx[class_dir]

            for img_path in glob.glob(os.path.join(class_path, "*.JPEG")):
                total_images += 1
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(np.array(img))
                    labels.append(class_idx)
                except (OSError, ValueError, RuntimeError) as exc:
                    failed_images += 1
                    if self.strict_loading:
                        raise RuntimeError(
                            f"Failed to load training image: {img_path}"
                        ) from exc
                    if len(failed_examples) < 20:
                        failed_examples.append((img_path, f"{type(exc).__name__}: {exc}"))
                    continue

        self._report_loading_failures("train", failed_images, total_images, failed_examples)
        if not images:
            raise RuntimeError("No valid training images were loaded from TinyImageNet")

        return np.stack(images), np.array(labels, dtype=np.int64)

    def _process_val_data(self, val_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        print("Processing validation data...")
        val_anno_path = os.path.join(val_dir, "val_annotations.txt")
        with open(val_anno_path, "r", encoding="utf-8") as handle:
            val_anno = handle.readlines()

        img_to_class = {line.split("\t")[0]: line.split("\t")[1] for line in val_anno}
        classes = sorted(set(img_to_class.values()))
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        images = []
        labels = []
        total_images = 0
        failed_images = 0
        failed_examples = []

        val_images_dir = os.path.join(val_dir, "images")
        for img_name in tqdm(sorted(os.listdir(val_images_dir))):
            img_path = os.path.join(val_images_dir, img_name)
            total_images += 1
            try:
                img = Image.open(img_path).convert("RGB")
                class_name = img_to_class[img_name]
                class_idx = class_to_idx[class_name]
                images.append(np.array(img))
                labels.append(class_idx)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                failed_images += 1
                if self.strict_loading:
                    raise RuntimeError(
                        f"Failed to load validation image: {img_path}"
                    ) from exc
                if len(failed_examples) < 20:
                    failed_examples.append((img_path, f"{type(exc).__name__}: {exc}"))
                continue

        self._report_loading_failures("val", failed_images, total_images, failed_examples)
        if not images:
            raise RuntimeError("No valid validation images were loaded from TinyImageNet")

        return np.stack(images), np.array(labels, dtype=np.int64)

    def download(self):
        if self._check_exists():
            print("Files already exist")
            return

        os.makedirs(self.tiny_imagenet_dir, exist_ok=True)

        legacy_paths = [
            self._legacy_processed_path(True),
            self._legacy_processed_path(False),
        ]
        if any(os.path.exists(path) for path in legacy_paths):
            print(
                "[TinyImageNet] Found legacy pickle caches. They are ignored for safety; "
                "the dataset will be regenerated as .npz files."
            )

        zip_path = os.path.join(self.root, "tiny-imagenet-200.zip")
        if not os.path.exists(zip_path):
            print("Downloading TinyImageNet...")
            download_url(self.base_url, zip_path)

        extract_root = os.path.join(self.root, ".tiny-imagenet-extract")
        if os.path.exists(extract_root):
            shutil.rmtree(extract_root)

        try:
            print("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                safe_extract_zip(zip_ref, extract_root)

            extracted_dir = os.path.join(extract_root, "tiny-imagenet-200")
            if not os.path.isdir(extracted_dir):
                raise RuntimeError(
                    "TinyImageNet archive extracted successfully, but the expected "
                    "tiny-imagenet-200 directory was not found."
                )

            print("Processing dataset...")
            train_data, train_labels = self._process_train_data(
                os.path.join(extracted_dir, "train")
            )
            self._save_processed_data(
                self._processed_path(True), train_data, train_labels
            )

            val_data, val_labels = self._process_val_data(
                os.path.join(extracted_dir, "val")
            )
            self._save_processed_data(self._processed_path(False), val_data, val_labels)
        finally:
            if os.path.exists(extract_root):
                shutil.rmtree(extract_root)

        print("Done!")


def load_tinyimagenet(
    root,
    train_transforms=None,
    valid_transforms=None,
    image_size=224,
    train_batch_size=64,
    valid_batch_size=64,
    split=0.9,
    seed=42,
    strict_loading=False,
    max_failure_ratio=0.005,
    return_dataloader=False,
):
    if split is None:
        split = 0.9

    if train_transforms is None:
        train_transforms = tfs.Compose(
            [
                tfs.RandomHorizontalFlip(p=0.5),
                tfs.RandomApply([tfs.RandomCrop(image_size, padding=4)], p=0.5),
                tfs.RandomApply([tfs.RandomRotation(10)], p=0.3),
                tfs.RandomApply(
                    [
                        tfs.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2
                        )
                    ],
                    p=0.3,
                ),
                tfs.Resize((image_size, image_size)),
                tfs.ToTensor(),
                tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    if valid_transforms is None:
        valid_transforms = tfs.Compose(
            [
                tfs.Resize((image_size, image_size)),
                tfs.ToTensor(),
                tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    base_train_set = TinyImageNet(
        root,
        True,
        transform=None,
        download=True,
        strict_loading=strict_loading,
        max_failure_ratio=max_failure_ratio,
    )
    train_set = DatasetView(base_train_set, transform=train_transforms)
    test_set = DatasetView(
        TinyImageNet(
            root,
            False,
            transform=None,
            download=True,
            strict_loading=strict_loading,
            max_failure_ratio=max_failure_ratio,
        ),
        transform=valid_transforms,
    )

    valid_subset = None
    if split < 1.0:
        labels = base_train_set.targets
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=1 - split, random_state=seed
        )
        for train_idx, val_idx in splitter.split(range(len(labels)), labels):
            train_subset = DatasetView(base_train_set, train_idx, train_transforms)
            valid_subset = DatasetView(base_train_set, val_idx, valid_transforms)
    else:
        train_subset = train_set

    test_loader = DataLoader(test_set, batch_size=valid_batch_size, drop_last=False)
    calibration_view = DatasetView(base_train_set, transform=valid_transforms)
    if return_dataloader:
        train_loader = DataLoader(
            train_subset, batch_size=train_batch_size, shuffle=True, drop_last=True
        )
        valid_loader = (
            DataLoader(valid_subset, batch_size=valid_batch_size, drop_last=False)
            if valid_subset is not None
            else None
        )

        return (
            train_loader,
            valid_loader,
            test_loader,
            (3, image_size, image_size),
            200,
            calibration_view,
        )
    return train_subset, valid_subset, test_loader, (3, image_size, image_size), 200, calibration_view
