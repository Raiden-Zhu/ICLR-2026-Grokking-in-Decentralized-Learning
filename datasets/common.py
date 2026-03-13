from torch.utils.data import Dataset, Subset


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