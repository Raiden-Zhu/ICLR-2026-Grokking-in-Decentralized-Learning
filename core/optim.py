"""Optimizer and scheduler factories for training."""

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


def init_optimizer(network, optimizer_name, lr, momentum, weight_decay):
    """Create a torch optimizer from config values."""
    name = optimizer_name.lower()
    if name == "sgd":
        return optim.SGD(
            network.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}. Supported: sgd, adam, adamw"
    )


def init_scheduler(optimizer, lr_scheduler_type, max_steps, lr):
    """Create a learning-rate scheduler from config values."""
    scheduler_name = lr_scheduler_type.lower()
    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, max_steps // 2, gamma=0.1)
    if scheduler_name == "warmup_cosine":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=max_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    if scheduler_name == "constant_then_zero":
        def lr_lambda(step):
            threshold = int(max_steps * 0.5)
            return 1.0 if step < threshold else 0.0

        return LambdaLR(optimizer, lr_lambda)
    if scheduler_name == "none":
        return None
    raise ValueError(
        "Unsupported lr_scheduler: "
        f"{lr_scheduler_type}. Supported: cosine, step, warmup_cosine, constant_then_zero, none"
    )
