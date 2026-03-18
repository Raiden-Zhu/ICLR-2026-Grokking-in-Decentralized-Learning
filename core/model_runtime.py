"""Model-side runtime helpers shared by training, evaluation, and post-merge flows."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BN_REESTIMATION_MAX_BATCHES = 32


def is_disabled_topology(topology):
    """Return True when gossip communication is explicitly disabled."""
    if topology is None:
        return True
    if isinstance(topology, str):
        return topology.lower() == "none"
    return False


def create_bn_reestimation_loader(train_dataset, batch_size):
    """Create a lightweight calibration loader for merged-model BN re-estimation."""
    if train_dataset is None:
        return None
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )


def reestimate_batch_norm_stats(model, calibration_loader, device, max_batches=BN_REESTIMATION_MAX_BATCHES):
    """Refresh BatchNorm running statistics on a merged model without enabling dropout."""
    if calibration_loader is None:
        return

    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats
    ]
    if not batch_norm_layers:
        return

    was_training = model.training
    original_momenta = {module: module.momentum for module in batch_norm_layers}

    model.eval()
    for module in batch_norm_layers:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    with torch.no_grad():
        for batch_index, batch in enumerate(calibration_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(inputs.to(device, non_blocking=True))

    for module in batch_norm_layers:
        module.momentum = original_momenta[module]
        module.eval()

    if was_training:
        model.train()
    else:
        model.eval()


def compute_consensus_error(networks):
    """Measure mean squared deviation from the average model."""
    mean_params = {}
    for key in networks[0].state_dict().keys():
        mean_params[key] = sum(net.state_dict()[key] for net in networks) / len(networks)

    consensus_error = 0.0
    for net in networks:
        param_diff = 0.0
        for key in net.state_dict().keys():
            param_diff += torch.sum((net.state_dict()[key] - mean_params[key]) ** 2).item()
        consensus_error += param_diff
    return consensus_error / len(networks)
