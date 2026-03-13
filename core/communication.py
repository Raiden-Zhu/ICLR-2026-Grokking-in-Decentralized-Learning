"""Communication helpers for gossip-based distributed training.

These utilities are intentionally small: they define the communication schedule
and the parameter-averaging primitives used by the higher-level training loop.
In the current multi-GPU setup, workers publish local states into shared CPU
buffers, rank 0 applies gossip centrally, and the updated states are broadcast
back to worker processes.
"""

import numpy as np
import torch

from core.gossip_matrix import (
    SparseGossipMatrix,
    get_gossip_matrix,
    get_sparse_gossip_matrix,
)


def compute_r(
    current_iter,
    start_iter,
    end_iter,
    r_start,
    r_end,
    point1,
    window_size,
    schedule="linear",
):
    """Compute dynamic r according to the configured schedule."""
    if schedule == "fixed":
        # Keep r constant at r_start for all iterations.
        return max(r_start, 0)
    if current_iter < start_iter:
        # Before schedule starts, use the initial r.
        return max(r_start, 0)
    if current_iter >= end_iter:
        # After schedule ends, clamp to the final r.
        return max(r_end, 0)

    progress = (current_iter - start_iter) / (end_iter - start_iter)
    if schedule == "linear":
        # Linearly interpolate from r_start to r_end.
        r = r_start - (r_start - r_end) * progress
    elif schedule == "cosine":
        # Smooth cosine decay from r_start to r_end.
        r = r_end + 0.5 * (r_start - r_end) * (1 + np.cos(np.pi * progress))
    elif schedule == "slow_decrease":
        # Exponential decay that drops quickly early and flattens later.
        decay_rate = 10
        r = r_end + (r_start - r_end) * np.exp(-decay_rate * progress)
    elif schedule == "slow_grow":
        # Very late-stage growth toward a larger target value.
        adjusted = progress**1
        r = r_start + (16 - r_start) * np.log10(1 + (adjusted**80) * 9)
    elif schedule == "truncate":
        # v1: single switch from r_start to r_end at point1.
        r = r_start if progress < point1 else r_end
    elif schedule == "truncate_v2":
        # v2: temporary r_end window [point1, point1 + window_size), then return to r_start.
        point2 = point1 + window_size
        if progress < point1:
            r = r_start
        elif progress < point2:
            r = r_end
        else:
            r = r_start
    else:
        raise ValueError(f"Unknown schedule {schedule}")

    return max(r, 0)


def gossip_update(networks, gossip_matrix):
    """Apply one gossip averaging step to model parameters.

    Args:
        networks: Ordered list of node models. Their order must match the rows of
            the gossip matrix.
        gossip_matrix: Row-stochastic communication matrix where row i specifies
            how node i mixes parameters from all nodes.
    """
    device = next(networks[0].parameters()).device
    gossip_matrix = gossip_matrix.to(device, dtype=torch.float32)
    mergeable_parameter_keys = [
        key for key, value in networks[0].named_parameters() if value.requires_grad and value.is_floating_point()
    ]

    with torch.no_grad():
        for key in mergeable_parameter_keys:
            params = torch.stack([net.state_dict()[key] for net in networks])
            params = params.to(dtype=torch.float32)
            params_reshaped = params.reshape(len(networks), -1)
            updated_params = torch.matmul(gossip_matrix, params_reshaped)
            updated_params = updated_params.reshape(params.shape)

            for net, new_param in zip(networks, updated_params):
                orig_param = net.state_dict()[key]
                net.state_dict()[key].copy_(new_param.to(dtype=orig_param.dtype))


def gossip_update_flat_buffer(
    source_buffer,
    target_buffer,
    gossip_matrix,
    compute_device="cuda:0",
    chunk_size=2_000_000,
):
    """Apply one gossip averaging step directly to a flat node-parameter matrix.

    Args:
        source_buffer: Shared CPU tensor of shape [num_nodes, total_float_state_numel].
        target_buffer: Shared CPU tensor with the same shape used for the result.
        gossip_matrix: Either a dense row-stochastic matrix or a sparse row-wise
            operator where row i specifies how node i mixes states from all nodes.
        compute_device: Device used for the weighted accumulation.
        chunk_size: Number of flattened state elements processed per chunk.
    """
    if source_buffer.shape != target_buffer.shape:
        raise ValueError("source_buffer and target_buffer must have the same shape")
    if source_buffer.ndim != 2:
        raise ValueError("Flat gossip buffers must be 2D tensors")

    device = torch.device(compute_device)

    with torch.no_grad():
        total_numel = source_buffer.shape[1]
        if isinstance(gossip_matrix, SparseGossipMatrix):
            if gossip_matrix.num_nodes != source_buffer.shape[0]:
                raise ValueError("Sparse gossip operator size does not match the flat buffer")

            device_rows = [
                (
                    neighbors.to(device=device, dtype=torch.long),
                    weights.to(device=device, dtype=torch.float32).unsqueeze(1),
                )
                for neighbors, weights in zip(gossip_matrix.row_indices, gossip_matrix.row_weights)
            ]

            for start in range(0, total_numel, chunk_size):
                end = min(start + chunk_size, total_numel)
                chunk = source_buffer[:, start:end].to(device=device, dtype=torch.float32)
                updated_chunk = torch.empty_like(chunk)
                for row_idx, (neighbors, weights) in enumerate(device_rows):
                    updated_chunk[row_idx].copy_(
                        (chunk.index_select(0, neighbors) * weights).sum(dim=0)
                    )
                target_buffer[:, start:end].copy_(updated_chunk.cpu())
        else:
            gossip_matrix = gossip_matrix.to(device=device, dtype=torch.float32)
            for start in range(0, total_numel, chunk_size):
                end = min(start + chunk_size, total_numel)
                chunk = source_buffer[:, start:end].to(device=device, dtype=torch.float32)
                updated_chunk = torch.matmul(gossip_matrix, chunk)
                target_buffer[:, start:end].copy_(updated_chunk.cpu())


__all__ = [
    "compute_r",
    "gossip_update",
    "gossip_update_flat_buffer",
    "get_gossip_matrix",
    "get_sparse_gossip_matrix",
]
