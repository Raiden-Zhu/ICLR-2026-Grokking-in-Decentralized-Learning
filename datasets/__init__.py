from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .tinyimagenet import load_tinyimagenet

def load_dataset(
    root,
    name,
    image_size,
    return_dataloader,
    train_batch_size=64,
    valid_batch_size=64,
    distribute=False,
    split=None,
    rank=0,
    seed=666,
    strict_loading=False,
    max_failure_ratio=0.005,
    debug=False,
):
    if name.lower() == "cifar100":
        return load_cifar100(
            root=root,
            image_size=image_size,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            split=split,
            seed=seed,
            return_dataloader=return_dataloader,
        )
    if name.lower() == "cifar10":
        return load_cifar10(
            root=root,
            image_size=image_size,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            split=split,
            seed=seed,
            return_dataloader=return_dataloader,
        )
    if name.lower() == "tinyimagenet":
        return load_tinyimagenet(
            root=root,
            image_size=image_size,
            train_batch_size=train_batch_size,
            valid_batch_size=valid_batch_size,
            split=split,
            seed=seed,
            strict_loading=strict_loading,
            max_failure_ratio=max_failure_ratio,
            return_dataloader=return_dataloader,
        )
    raise ValueError(f"Unsupported dataset: {name}")
