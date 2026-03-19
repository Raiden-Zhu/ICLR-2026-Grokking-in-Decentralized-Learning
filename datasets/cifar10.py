import torchvision.transforms as tfs
from torchvision.transforms import RandAugment
from torchvision.datasets import CIFAR10

from .common import DatasetView, build_split_dataset_views, finalize_classification_dataset


def load_cifar10(
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
    if transforms is None:
        transforms_train = tfs.Compose(
            [
                tfs.RandomCrop(32, padding=4),
                tfs.RandomHorizontalFlip(),
                RandAugment(num_ops=2, magnitude=9),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )
        transforms_test = tfs.Compose(
            [
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )
    else:
        transforms_train = transforms
        transforms_test = transforms if valid_transforms is None else valid_transforms

    if valid_transforms is not None:
        transforms_test = valid_transforms

    if train_batch_size is None:
        train_batch_size = 1
    if split is None:
        split = 0.8

    base_train_set = CIFAR10(root, True, transform=None, download=True)
    test_set = DatasetView(
        CIFAR10(root, False, transform=None, download=True),
        transform=transforms_test,
    )
    train_subset, valid_subset, calibration_view = build_split_dataset_views(
        base_train_set,
        transforms_train,
        transforms_test,
        split=split,
        seed=seed,
    )

    return finalize_classification_dataset(
        train_subset=train_subset,
        valid_subset=valid_subset,
        test_set=test_set,
        calibration_view=calibration_view,
        image_shape=(3, image_size, image_size),
        num_classes=10,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
        return_dataloader=return_dataloader,
    )
