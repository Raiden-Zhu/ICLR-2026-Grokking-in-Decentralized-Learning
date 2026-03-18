import numpy as np
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
import torch
from tqdm import tqdm
import random
import time
import os
import json
import hashlib
from .common import get_dataset_targets
# from sklearn.model_selection import train_test_split


SUPPORTED_SAMPLING_MODES = {"fixed", "resample"}



def dirichlet_split_noniid(train_labels, alpha, n_clients):
    """
    Split sample indices into n_clients subsets using a Dirichlet distribution with concentration alpha.
    """
    n_classes = train_labels.max() + 1
    # (K, N) class-label distribution matrix. Each row stores how one class is distributed across clients.
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, ...) Sample-index lists for each class.
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    # Store the sample-index list for each client.
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # k_idcs contains all indices for one class, and fracs gives that class's proportions across clients.
        # np.split partitions the indices of class k into N subsets according to fracs.
        # i is the client index, and idcs is the subset of sample indices assigned to that client.
        # (np.cumsum(fracs)[:-1] * len(k_idcs)) computes the split points for this class across clients.

        for i, idcs in enumerate(
            np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))
        ):
            client_idcs[i] += [idcs]
    # Concatenate per-class index lists into one index array per client.
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def dirichlet_split(n, num_classes, dir_alpha, seed=42):

    # Generate a NumPy matrix of Dirichlet-distributed class weights.
    rng = np.random.default_rng(seed)
    return rng.dirichlet([dir_alpha] * num_classes, n)

def _build_dataloader_kwargs(num_workers):
    effective_num_workers = max(0, int(num_workers))
    dataloader_kwargs = {
        "num_workers": effective_num_workers,
        "pin_memory": True,
    }
    if effective_num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
        dataloader_kwargs["persistent_workers"] = True
    return dataloader_kwargs


def _normalize_sampling_mode(mode, *, field_name):
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_SAMPLING_MODES:
        supported = ", ".join(sorted(SUPPORTED_SAMPLING_MODES))
        raise ValueError(f"{field_name} must be one of: {supported}")
    return normalized


def _get_class_weights_for_index(i, all_class_weights, nb_class):
    if all_class_weights is not None:
        class_weights = np.asarray(all_class_weights[i], dtype=np.float64)
    else:
        class_weights = np.ones(nb_class, dtype=np.float64) / nb_class

    if class_weights.ndim != 1 or class_weights.shape[0] != nb_class:
        raise ValueError("class_weights must be a 1D vector with length nb_class")

    total_weight = class_weights.sum()
    if total_weight <= 0:
        raise ValueError("class_weights must sum to a positive value")
    return class_weights / total_weight


def _create_subset_dataloader(dataset, subset_indices, batch_size, num_workers):
    subset = Subset(dataset, subset_indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        **_build_dataloader_kwargs(num_workers),
    )


def sample_fixed_subset_indices(dataset, subset_size, class_weights, seed):
    """Sample one fixed subset once for the whole training run.

    Sampling is without replacement inside one node dataset, but different nodes may
    still overlap because each node keeps an independently sampled fixed local subset.
    """
    subset_size = int(subset_size)
    if subset_size < 0:
        raise ValueError("subset_size must be non-negative")
    if subset_size == 0:
        return np.empty((0,), dtype=np.int64)

    dataset_size = len(dataset)
    if subset_size > dataset_size:
        raise ValueError(
            f"Requested fixed subset of size {subset_size}, but dataset only has {dataset_size} samples"
        )

    targets = get_dataset_targets(dataset)
    weights = np.asarray(class_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("class_weights must be a 1D vector")

    sample_probabilities = weights[targets] + 1e-12
    sample_probabilities /= sample_probabilities.sum()

    rng = np.random.default_rng(seed)
    return rng.choice(
        dataset_size,
        size=subset_size,
        replace=False,
        p=sample_probabilities,
    ).astype(np.int64)


def create_fixed_dataloader_for_index(
    i,
    train_dataset,
    valid_dataset,
    samples_per_loader,
    batch_size,
    all_class_weights,
    nb_class,
    train_ratio,
    num_workers,
    seed,
):
    """Create fixed per-node train/valid dataloaders sampled once at startup."""
    class_weights = _get_class_weights_for_index(i, all_class_weights, nb_class)

    train_samples = int(samples_per_loader)
    valid_samples = int(samples_per_loader * (1 - train_ratio) / train_ratio)

    train_indices = sample_fixed_subset_indices(
        train_dataset,
        train_samples,
        class_weights,
        seed=seed + i,
    )
    train_dataloader = _create_subset_dataloader(
        train_dataset,
        train_indices,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    valid_dataloader = None
    if valid_dataset is not None and valid_samples > 0:
        valid_indices = sample_fixed_subset_indices(
            valid_dataset,
            valid_samples,
            class_weights,
            seed=seed + 10000 + i,
        )
        valid_dataloader = _create_subset_dataloader(
            valid_dataset,
            valid_indices,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    return train_dataloader, valid_dataloader


# num_samples refers to the number of batches in the original dataset (e.g. 97 with batch size 512)
# further divided by the node count, e.g. 97 // 16 = 6.
class nonIIDSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        num_samples,
        class_weights,
        nb_classes,
        limit_samples=1000,
        cache_dir="cache/class_indices",
        enable_cache=True,
        seed=42,
        node_index=0,
        stream_offset=0,
    ):
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.class_weights = np.asarray(class_weights, dtype=np.float64)
        self.nb_classes = nb_classes
        self.enable_cache = enable_cache
        self.seed = int(seed)
        self.node_index = int(node_index)
        self.stream_offset = int(stream_offset)
        self.iteration_counter = 0

        if self.num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        
        # Try to load class_indices from cache.
        if self.enable_cache:
            cache_loaded = self._load_class_indices_from_cache(cache_dir)
        else:
            cache_loaded = False
        
        # Rebuild the cache if loading fails.
        if not cache_loaded:
            self.class_indices = self._build_class_indices(limit_samples)
            
            # Save the rebuilt indices to cache.
            if self.enable_cache:
                self._save_class_indices_to_cache(cache_dir)

        self.class_indices = [np.asarray(indices, dtype=np.int64) for indices in self.class_indices]
        self.class_probabilities = self._build_class_probabilities()

    def _build_class_indices(self, limit_samples):
        targets = None
        try:
            targets = get_dataset_targets(self.dataset)
        except ValueError:
            targets = None

        class_indices = [[] for _ in range(self.nb_classes)]
        if targets is not None:
            for idx, label in enumerate(np.asarray(targets, dtype=np.int64)):
                class_indices[int(label)].append(idx)
            return class_indices

        with tqdm(total=len(self.dataset), desc="Initializing Sampler") as pbar:
            for idx, (_, label) in enumerate(self.dataset):
                class_indices[int(label)].append(idx)
                pbar.update(1)
        return class_indices

    def _build_class_probabilities(self):
        if self.class_weights.ndim != 1 or self.class_weights.shape[0] != self.nb_classes:
            raise ValueError("class_weights must be a 1D vector with length nb_classes")

        class_probabilities = np.clip(self.class_weights.copy(), a_min=0.0, a_max=None)
        available_mask = np.asarray([len(indices) > 0 for indices in self.class_indices], dtype=bool)
        class_probabilities[~available_mask] = 0.0

        total_probability = class_probabilities.sum()
        if total_probability <= 0:
            if not available_mask.any():
                raise ValueError("Sampler cannot draw from an empty dataset")
            class_probabilities = available_mask.astype(np.float64)
            total_probability = class_probabilities.sum()

        return class_probabilities / total_probability
    
    def _get_cache_key(self):
        """Build a cache key from dataset characteristics."""
        # Use dataset type, length, and class count as the base identifier.
        dataset_name = type(self.dataset).__name__
        dataset_len = len(self.dataset)

        # Try to retrieve the dataset root path when available.
        dataset_root = ""
        if hasattr(self.dataset, 'root'):
            dataset_root = str(self.dataset.root)
        elif hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'root'):
            dataset_root = str(self.dataset.dataset.root)

        dataset_view = ""
        if hasattr(self.dataset, "indices") and self.dataset.indices is not None:
            subset_indices = np.asarray(self.dataset.indices, dtype=np.int64)
            dataset_view = hashlib.md5(subset_indices.tobytes()).hexdigest()[:12]

        # Build a stable hash for this dataset view.
        key_string = f"{dataset_name}_{dataset_len}_{self.nb_classes}_{dataset_root}_{dataset_view}"
        hash_key = hashlib.md5(key_string.encode()).hexdigest()[:12]

        return f"{dataset_name}_{dataset_len}_{self.nb_classes}_{hash_key}.npz"

    def _serialize_class_indices(self):
        counts = np.asarray([len(indices) for indices in self.class_indices], dtype=np.int64)
        arrays = [np.asarray(indices, dtype=np.int64) for indices in self.class_indices]
        flat_indices = (
            np.concatenate(arrays).astype(np.int64, copy=False)
            if arrays and any(len(indices) > 0 for indices in arrays)
            else np.empty((0,), dtype=np.int64)
        )
        return counts, flat_indices

    def _deserialize_class_indices(self, counts, flat_indices):
        counts = np.asarray(counts, dtype=np.int64)
        flat_indices = np.asarray(flat_indices, dtype=np.int64)

        if counts.ndim != 1 or len(counts) != self.nb_classes:
            raise ValueError("Cached class_indices has invalid class-count metadata")
        if np.any(counts < 0):
            raise ValueError("Cached class_indices contains negative class counts")

        total_indices = int(counts.sum())
        if total_indices != len(flat_indices):
            raise ValueError(
                "Cached class_indices is inconsistent: flat index length does not match counts"
            )

        class_indices = []
        start_idx = 0
        for count in counts.tolist():
            end_idx = start_idx + int(count)
            class_indices.append(flat_indices[start_idx:end_idx].copy())
            start_idx = end_idx
        return class_indices

    def _load_class_indices_from_cache(self, cache_dir):
        """Load class_indices from cache."""
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, self._get_cache_key())

            if not os.path.exists(cache_file):
                return False

            with np.load(cache_file, allow_pickle=False) as cache_data:
                self.class_indices = self._deserialize_class_indices(
                    cache_data["counts"],
                    cache_data["flat_indices"],
                )
            print(f"✓ Loaded class_indices from cache: {cache_file}")
            return True
        except Exception as e:
            print(f"⚠ Failed to load cache: {e}")
            return False

    def _save_class_indices_to_cache(self, cache_dir):
        """Save class_indices to cache."""
        try:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, self._get_cache_key())
            counts, flat_indices = self._serialize_class_indices()
            temp_file = f"{cache_file}.tmp"

            with open(temp_file, 'wb') as handle:
                np.savez_compressed(handle, counts=counts, flat_indices=flat_indices)
            os.replace(temp_file, cache_file)
            print(f"✓ Saved class_indices to cache: {cache_file}")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")

    def __iter__(self):
        iteration_seed = (
            self.seed
            + self.node_index * 1000003
            + self.stream_offset * 10007
            + self.iteration_counter
        )
        self.iteration_counter += 1

        rng = np.random.default_rng(iteration_seed)
        class_choices = rng.choice(
            self.nb_classes,
            size=self.num_samples,
            replace=True,
            p=self.class_probabilities,
        )

        samples = np.empty(self.num_samples, dtype=np.int64)
        for class_idx in np.unique(class_choices):
            class_positions = np.where(class_choices == class_idx)[0]
            class_members = self.class_indices[int(class_idx)]
            samples[class_positions] = rng.choice(
                class_members,
                size=len(class_positions),
                replace=True,
            )

        return iter(samples.tolist())

    def __len__(self):
        return self.num_samples


def record_datasequence(sampler):
    # Collect the sampled data-order indices generated by the sampler.
    sampled_indices = list(sampler)
    sampled_indices = [int(idx) for idx in sampled_indices]
    # Get the current timestamp.
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create the output directory.
    save_dir = "/mnt/csp/mmvision/home/lwh/DLS/datasequence/"
    os.makedirs(save_dir, exist_ok=True)

    # Save sampled indices to a JSON file.
    file_path = os.path.join(save_dir, f"sampled_indices_{timestamp}.json")
    with open(file_path, "w") as f:
        json.dump(sampled_indices, f, indent=4)

    print(f"Sampled indices saved to {file_path}")


# def create_train_test_dataloaders(
#     train_dataset,
#     valid_dataset,
#     nb_dataloader,
#     samples_per_loader,
#     batch_size=32,
#     all_class_weights=None,
#     nb_class=10,
#     train_ratio=0.9,
# ):
#     train_dataloaders = []
#     test_dataloaders = []
#     for i in range(nb_dataloader):
#         # Create a unique class distribution for each dataloader
#         if all_class_weights is not None:
#             class_weights = all_class_weights[i]
#         else:
#             class_weights = np.random.dirichlet(np.ones(nb_class))
#         # Calculate split sizes
#         train_samples = int(samples_per_loader * train_ratio)
#         test_samples = samples_per_loader - train_samples

#         # Create new samplers with same class weights but different sample sizes
#         train_sampler = nonIIDSampler(
#             dataset=train_dataset,
#             num_samples=train_samples,
#             class_weights=class_weights,
#             nb_classes=nb_class,
#         )  #  dataset, num_samples, class_weights, nb_classes, limit_samples=1000

#         test_sampler = nonIIDSampler(
#             dataset=valid_dataset,
#             num_samples=test_samples,
#             class_weights=class_weights,
#             nb_classes=nb_class,
#         )

#         # Create new dataloaders
#         train_dataloader = DataLoader(
#             train_dataset, batch_size=batch_size, sampler=train_sampler
#         )

#         test_dataloader = DataLoader(
#             valid_dataset, batch_size=batch_size, sampler=test_sampler
#         )

#         train_dataloaders.append(train_dataloader)
#         test_dataloaders.append(test_dataloader)
#     return train_dataloaders, test_dataloaders


def create_dataloaders(
    dataset,
    nb_dataloader,
    samples_per_loader,
    batch_size=32,
    all_class_weights=None,
    nb_class=10,
):
    class_weights_all = []
    dataloaders = []
    for i in range(nb_dataloader):
        # Create a unique class distribution for each dataloader
        if all_class_weights is not None:
            class_weights = all_class_weights[i]
        else:
            class_weights = np.random.dirichlet(np.ones(nb_class))
        class_weights_all.append(class_weights)
        sampler = nonIIDSampler(dataset, samples_per_loader, class_weights, nb_class)
        # record_datasequence(sampler)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
        )
        dataloaders.append(dataloader)

    return dataloaders, class_weights_all


def create_dataloader_for_index(
    i,
    train_dataset,
    valid_dataset,
    samples_per_loader,
    batch_size,
    all_class_weights,
    nb_class,
    train_ratio,
    num_workers,
    seed,
    valid_sampling_mode="resample",
):
    class_weights = _get_class_weights_for_index(i, all_class_weights, nb_class)

    # Calculate split sizes
    train_samples = int(samples_per_loader)
    valid_samples = int(samples_per_loader * (1 - train_ratio) / train_ratio)

    train_sampler = nonIIDSampler(
        dataset=train_dataset,
        num_samples=train_samples,
        class_weights=class_weights,
        nb_classes=nb_class,
        seed=seed,
        node_index=i,
        stream_offset=0,
    )

    dataloader_kwargs = _build_dataloader_kwargs(num_workers)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        **dataloader_kwargs,
    )

    valid_dataloader = None
    if valid_dataset is not None and valid_samples > 0:
        valid_sampling_mode = _normalize_sampling_mode(
            valid_sampling_mode,
            field_name="valid_sampling_mode",
        )
        if valid_sampling_mode == "resample":
            valid_sampler = nonIIDSampler(
                dataset=valid_dataset,
                num_samples=valid_samples,
                class_weights=class_weights,
                nb_classes=nb_class,
                seed=seed + 20000,
                node_index=i,
                stream_offset=1,
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                sampler=valid_sampler,
                **dataloader_kwargs,
            )
        else:
            valid_indices = sample_fixed_subset_indices(
                valid_dataset,
                valid_samples,
                class_weights,
                seed=seed + 10000 + i,
            )
            valid_dataloader = _create_subset_dataloader(
                valid_dataset,
                valid_indices,
                batch_size=batch_size,
                num_workers=num_workers,
            )

    return train_dataloader, valid_dataloader


def create_train_valid_dataloaders_multi(
    train_dataset,
    valid_dataset,
    nb_dataloader,
    samples_per_loader,
    batch_size=32,
    all_class_weights=None,
    nb_class=10,
    train_ratio=0.8,
    num_workers=8,
    seed=42,
    sampling_mode="fixed",
    valid_sampling_mode=None,
):
    """Create one train/valid dataloader pair per node.

    sampling_mode controls how training samples are produced for each node while
    keeping the same class weights. In resample mode, training data is redrawn
    every time the dataloader iterator is recreated. Validation defaults to a
    fixed subset so evaluation remains comparable across rounds.
    """
    sampling_mode = _normalize_sampling_mode(sampling_mode, field_name="sampling_mode")
    if valid_sampling_mode is None:
        valid_sampling_mode = "fixed"
    valid_sampling_mode = _normalize_sampling_mode(
        valid_sampling_mode,
        field_name="valid_sampling_mode",
    )

    train_dataloaders = []
    valid_dataloaders = []

    for i in range(nb_dataloader):
        if sampling_mode == "fixed":
            train_dl, valid_dl = create_fixed_dataloader_for_index(
                i=i,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                samples_per_loader=samples_per_loader,
                batch_size=batch_size,
                all_class_weights=all_class_weights,
                nb_class=nb_class,
                train_ratio=train_ratio,
                num_workers=num_workers,
                seed=seed,
            )
        else:
            train_dl, valid_dl = create_dataloader_for_index(
                i=i,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                samples_per_loader=samples_per_loader,
                batch_size=batch_size,
                all_class_weights=all_class_weights,
                nb_class=nb_class,
                train_ratio=train_ratio,
                num_workers=num_workers,
                seed=seed,
                valid_sampling_mode=valid_sampling_mode,
            )
        train_dataloaders.append(train_dl)
        valid_dataloaders.append(valid_dl)

    return train_dataloaders, valid_dataloaders


def create_train_test_dataloaders_multi(
    train_dataset,
    valid_dataset,
    nb_dataloader,
    samples_per_loader,
    batch_size=32,
    all_class_weights=None,
    nb_class=10,
    train_ratio=0.8,
    num_workers=8,
    seed=42,
):
    return create_train_valid_dataloaders_multi(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        nb_dataloader=nb_dataloader,
        samples_per_loader=samples_per_loader,
        batch_size=batch_size,
        all_class_weights=all_class_weights,
        nb_class=nb_class,
        train_ratio=train_ratio,
        num_workers=num_workers,
        seed=seed,
        sampling_mode="resample",
        valid_sampling_mode="resample",
    )


def create_fixed_train_valid_dataloaders_multi(
    train_dataset,
    valid_dataset,
    nb_dataloader,
    samples_per_loader,
    batch_size=32,
    all_class_weights=None,
    nb_class=10,
    train_ratio=0.8,
    num_workers=8,
    seed=42,
):
    """Backward-compatible wrapper for the fixed subset mode."""
    return create_train_valid_dataloaders_multi(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        nb_dataloader=nb_dataloader,
        samples_per_loader=samples_per_loader,
        batch_size=batch_size,
        all_class_weights=all_class_weights,
        nb_class=nb_class,
        train_ratio=train_ratio,
        num_workers=num_workers,
        seed=seed,
        sampling_mode="fixed",
        valid_sampling_mode="fixed",
    )


def create_simple_preference(n, nb_class, important_prob=0.5):
    all_class_weights = np.zeros((n, nb_class))
    if nb_class > n:
        nb_important = nb_class // n
    else:
        nb_important = 1
    for i in range(n):
        # generate nb_important int between 0 and nb_class-1 (inclusive)
        important_classes = np.random.randint(0, nb_class, nb_important)
        all_class_weights[i, important_classes] = important_prob / nb_important
        # the rest index which is not in the important_class should be (1-important_prob) / (nb_class - nb_important)
        all_class_weights[i, np.setdiff1d(np.arange(nb_class), important_classes)] = (
            1 - important_prob
        ) / (nb_class - nb_important)
    return all_class_weights


def create_IID_preference(n, nb_class):
    all_class_weights = np.zeros((n, nb_class))
    for i in range(n):
        all_class_weights[i] = np.ones(nb_class) / nb_class
    return all_class_weights


if __name__ == "__main__":
    # from torchvision.datasets import CIFAR10
    # from torchvision import transforms

    # # Create 5 dataloaders with 10000 samples each
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    # # Load CIFAR-10 dataset
    # cifar10 = CIFAR10(root="./data", train=True, download=True, transform=transform)
    # n_loaders = 5
    # samples_per_loader = 10000
    # dataloaders = create_nonIID_dataloaders(cifar10, n_loaders, samples_per_loader)

    # # Verify the class distribution in each dataloader
    # for i, dataloader in enumerate(dataloaders):
    #     class_counts = [0] * 10
    #     for _, labels in dataloader:
    #         for label in labels:
    #             class_counts[label] += 1
    #     print(f"Dataloader {i} class distribution:")
    #     print(class_counts)
    #     print()
    print(create_simple_preference(16, 10))
