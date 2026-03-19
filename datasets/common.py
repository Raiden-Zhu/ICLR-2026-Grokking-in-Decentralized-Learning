from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset


class DatasetView(Dataset):
    """Apply a transform to an existing dataset view without copying its samples."""

    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.indices = None if indices is None else list(indices)
        self.transform = transform

    def __len__(self):
        if self.indices is None:
            return len(self.dataset)
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = idx if self.indices is None else self.indices[idx]
        image, target = self.dataset[source_idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    @property
    def targets(self):
        return get_dataset_targets(self)

    @property
    def labels(self):
        return self.targets


def get_dataset_targets(dataset):
    """Return label targets for datasets and lightweight dataset views."""
    if isinstance(dataset, Subset):
        parent_targets = get_dataset_targets(dataset.dataset)
        return [parent_targets[idx] for idx in dataset.indices]

    if isinstance(dataset, DatasetView):
        parent_targets = get_dataset_targets(dataset.dataset)
        if dataset.indices is None:
            return parent_targets
        return [parent_targets[idx] for idx in dataset.indices]

    if hasattr(dataset, "targets"):
        return dataset.targets
    if hasattr(dataset, "labels"):
        return dataset.labels

    raise ValueError("Dataset must expose targets or labels")


def build_split_dataset_views(base_train_set, train_transform, eval_transform, *, split, seed):
    """Build train/valid/calibration dataset views while preserving current split semantics."""
    train_set = DatasetView(base_train_set, transform=train_transform)
    calibration_view = DatasetView(base_train_set, transform=eval_transform)

    if split < 1.0:
        labels = get_dataset_targets(base_train_set)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - split,
            random_state=seed,
        )
        for train_idx, val_idx in splitter.split(range(len(labels)), labels):
            train_subset = DatasetView(base_train_set, train_idx, train_transform)
            valid_subset = DatasetView(base_train_set, val_idx, eval_transform)
            return train_subset, valid_subset, calibration_view

    return train_set, None, calibration_view


def finalize_classification_dataset(
    *,
    train_subset,
    valid_subset,
    test_set,
    calibration_view,
    image_shape,
    num_classes,
    train_batch_size,
    valid_batch_size,
    return_dataloader,
):
    """Finalize the common dataset return structure without changing output semantics."""
    test_loader = DataLoader(test_set, batch_size=valid_batch_size, drop_last=False)

    if return_dataloader:
        train_loader = DataLoader(
            train_subset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
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
            image_shape,
            num_classes,
            calibration_view,
        )

    return (
        train_subset,
        valid_subset,
        test_loader,
        image_shape,
        num_classes,
        calibration_view,
    )
