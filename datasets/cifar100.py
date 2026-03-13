import torchvision.transforms as tfs
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CIFAR100
from .distribute_dataset import distribute_dataset
from .common import DatasetView
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset


class SubsetDataset(Dataset):
    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num_samples = min(num_samples, len(dataset))
        self.indices = list(range(self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {self.num_samples}"
            )
        return self.dataset[self.indices[idx]]


def load_cifar100(
    root,
    transforms=None,
    valid_transforms=None,
    image_size=32,
    train_batch_size=64,
    valid_batch_size=64,
    split=0.95,
    seed=42,
    return_dataloader=False,
):

    train_subset = None
    valid_subset = None

    if transforms is None:
        transforms_train = tfs.Compose(
            [
                tfs.Resize((image_size, image_size)),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )
    else:
        transforms_train = transforms

    transforms_test = transforms_train if valid_transforms is None else valid_transforms

    if train_batch_size is None:
        train_batch_size = 1
    if split is None:
        split = 0.8
    base_train_set = CIFAR100(root, True, transform=None, download=True)
    train_set = DatasetView(base_train_set, transform=transforms_train)
    test_set = DatasetView(CIFAR100(root, False, transform=None, download=True), transform=transforms_test)

    # split train_set into train_set and test_set make sure the class distribution is the same
    if split < 1.0:
        labels = base_train_set.targets
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=1 - split, random_state=seed
        )
        for train_idx, val_idx in splitter.split(range(len(labels)), labels):
            train_subset = DatasetView(base_train_set, train_idx, transforms_train)
            valid_subset = DatasetView(base_train_set, val_idx, transforms_test)
    else:
        train_subset = train_set

    test_loader = DataLoader(test_set, batch_size=valid_batch_size, drop_last=False)
    if return_dataloader:
        train_loader = DataLoader(
            train_subset, batch_size=train_batch_size, shuffle=True, drop_last=True
        )
        valid_loader = (
            DataLoader(valid_subset, batch_size=valid_batch_size, drop_last=False)
            if valid_subset is not None
            else None
        )

        return train_loader, valid_loader, test_loader, (3, image_size, image_size), 100, DatasetView(base_train_set, transform=transforms_test)
    return train_subset, valid_subset, test_loader, (3, image_size, image_size), 100, DatasetView(base_train_set, transform=transforms_test)
