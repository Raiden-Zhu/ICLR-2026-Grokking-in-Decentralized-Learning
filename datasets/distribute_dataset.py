"""Legacy dataset-partition compatibility helpers.

This module is no longer used by the active simulator data path, which now
relies on DatasetView-based subsets and dirichlet_sampling.py. It is kept only
as a small compatibility helper for older external scripts.
"""

import random
from typing import Any, Optional

from torch.utils.data import Dataset


class DistributedDataset(Dataset):
    def __init__(self, dataset: Dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]

    def __len__(self):
        return len(self.indices)


def distribute_dataset(
    dataset: Dataset,
    split: Any,
    rank: int,
    size: Optional[int] = None,
    seed: int = 777,
    dirichlet: bool = False,
):
    if dirichlet:
        return DistributedDataset(dataset, split[rank])

    dataset_size = len(dataset) if size is None else int(size)
    generator = random.Random(seed)
    indices = list(range(dataset_size))
    generator.shuffle(indices)

    index_splits = []
    for cutoff in split:
        boundary = int(cutoff * dataset_size)
        index_splits.append(indices[:boundary])
        indices = indices[boundary:]

    return DistributedDataset(dataset, index_splits[rank])
