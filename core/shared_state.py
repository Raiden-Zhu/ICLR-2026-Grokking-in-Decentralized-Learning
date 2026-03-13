"""Shared-memory model state helpers for multi-process gossip training."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class FlatStateEntry:
    """Metadata for one floating-point state entry packed into the flat buffer."""

    key: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    start: int
    end: int
    average_in_merged_model: bool


@dataclass(frozen=True)
class BufferStateEntry:
    """Metadata for one model buffer copied through between communication rounds."""

    key: str
    shape: Tuple[int, ...]
    dtype: torch.dtype


@dataclass
class SharedStatePool:
    """Double-buffered shared storage for floating state and non-floating buffers."""

    flat_source: torch.Tensor
    flat_target: torch.Tensor
    buffer_source: Dict[str, torch.Tensor]
    buffer_target: Dict[str, torch.Tensor]
    float_entries: List[FlatStateEntry]
    buffer_entries: List[BufferStateEntry]


def _shared_empty(shape, dtype):
    tensor = torch.empty(shape, dtype=dtype)
    tensor.share_memory_()
    return tensor


def create_shared_state_pool(reference_model, num_nodes):
    """Create double-buffered shared CPU storage for floating state and buffers."""
    float_entries = []
    buffer_entries = []
    buffer_source = {}
    buffer_target = {}

    parameter_keys = {key for key, _ in reference_model.named_parameters()}
    merged_model_parameter_keys = {
        key
        for key, value in reference_model.named_parameters()
        if value.requires_grad and value.is_floating_point()
    }

    offset = 0
    for key, value in reference_model.state_dict().items():
        shape = tuple(value.shape)
        if value.is_floating_point():
            next_offset = offset + value.numel()
            float_entries.append(
                FlatStateEntry(
                    key=key,
                    shape=shape,
                    dtype=value.dtype,
                    start=offset,
                    end=next_offset,
                    average_in_merged_model=key in merged_model_parameter_keys,
                )
            )
            offset = next_offset
        elif key not in parameter_keys:
            buffer_entries.append(BufferStateEntry(key=key, shape=shape, dtype=value.dtype))
            storage_shape = (num_nodes, *shape)
            buffer_source[key] = _shared_empty(storage_shape, value.dtype)
            buffer_target[key] = _shared_empty(storage_shape, value.dtype)

    flat_source = _shared_empty((num_nodes, offset), torch.float32)
    flat_target = _shared_empty((num_nodes, offset), torch.float32)

    return SharedStatePool(
        flat_source=flat_source,
        flat_target=flat_target,
        buffer_source=buffer_source,
        buffer_target=buffer_target,
        float_entries=float_entries,
        buffer_entries=buffer_entries,
    )


def _select_flat_buffer(shared_state_pool, buffer_name):
    if buffer_name == "source":
        return shared_state_pool.flat_source
    if buffer_name == "target":
        return shared_state_pool.flat_target
    raise ValueError(f"Unknown buffer_name: {buffer_name}")


def _select_buffer_storage(shared_state_pool, buffer_name):
    if buffer_name == "source":
        return shared_state_pool.buffer_source
    if buffer_name == "target":
        return shared_state_pool.buffer_target
    raise ValueError(f"Unknown buffer_name: {buffer_name}")


def copy_model_to_shared_buffer(model, shared_state_pool, node_index, buffer_name="source"):
    """Copy one model's current state into the selected shared CPU buffer."""
    flat_buffer = _select_flat_buffer(shared_state_pool, buffer_name)
    buffer_storage = _select_buffer_storage(shared_state_pool, buffer_name)
    state_dict = model.state_dict()

    with torch.no_grad():
        if shared_state_pool.float_entries:
            float_row = torch.cat(
                [
                    state_dict[entry.key].detach().reshape(-1).to(dtype=torch.float32)
                    for entry in shared_state_pool.float_entries
                ]
            )
            flat_buffer[node_index].copy_(float_row.to(device="cpu"))

        for entry in shared_state_pool.buffer_entries:
            buffer_storage[entry.key][node_index].copy_(
                state_dict[entry.key].detach().to(device="cpu")
            )


def copy_shared_buffer_to_model(model, shared_state_pool, node_index, device, buffer_name="target"):
    """Load one node state from the selected shared buffer into a model instance."""
    flat_buffer = _select_flat_buffer(shared_state_pool, buffer_name)
    buffer_storage = _select_buffer_storage(shared_state_pool, buffer_name)
    state_dict = model.state_dict()

    with torch.no_grad():
        flat_row = flat_buffer[node_index].to(device=device, dtype=torch.float32)
        for entry in shared_state_pool.float_entries:
            target = state_dict[entry.key]
            source = flat_row[entry.start:entry.end].view(entry.shape)
            target.copy_(source.to(dtype=target.dtype))

        for entry in shared_state_pool.buffer_entries:
            target = state_dict[entry.key]
            source = buffer_storage[entry.key][node_index]
            target.copy_(source.to(device=device, dtype=target.dtype))


def copy_source_state_to_target(shared_state_pool):
    """Copy the published mergeable params and buffers into the target buffer."""
    with torch.no_grad():
        shared_state_pool.flat_target.copy_(shared_state_pool.flat_source)
        for key, value in shared_state_pool.buffer_source.items():
            shared_state_pool.buffer_target[key].copy_(value)


def copy_mean_state_to_model(
    model,
    shared_state_pool,
    device,
    buffer_name="target",
    reference_node_index=0,
):
    """Load mean mergeable params and reference floating buffers into one model instance."""
    flat_buffer = _select_flat_buffer(shared_state_pool, buffer_name)
    buffer_storage = _select_buffer_storage(shared_state_pool, buffer_name)
    state_dict = model.state_dict()

    with torch.no_grad():
        if shared_state_pool.float_entries:
            mean_row = flat_buffer.mean(dim=0).to(device=device, dtype=torch.float32)
            reference_row = flat_buffer[reference_node_index].to(device=device, dtype=torch.float32)
            for entry in shared_state_pool.float_entries:
                target = state_dict[entry.key]
                if entry.average_in_merged_model:
                    source = mean_row[entry.start:entry.end].view(entry.shape)
                else:
                    source = reference_row[entry.start:entry.end].view(entry.shape)
                target.copy_(source.to(dtype=target.dtype))

        for entry in shared_state_pool.buffer_entries:
            target = state_dict[entry.key]
            source = buffer_storage[entry.key][reference_node_index]
            target.copy_(source.to(device=device, dtype=target.dtype))


def compute_consensus_error_from_buffer(shared_state_pool, buffer_name="target"):
    """Compute consensus error directly from mergeable parameters in the flat buffer."""
    flat_buffer = _select_flat_buffer(shared_state_pool, buffer_name)

    with torch.no_grad():
        if flat_buffer.numel() == 0:
            return 0.0

        consensus_slices = [
            flat_buffer[:, entry.start:entry.end]
            for entry in shared_state_pool.float_entries
            if entry.average_in_merged_model
        ]
        if not consensus_slices:
            return 0.0

        consensus_buffer = torch.cat(consensus_slices, dim=1)
        mean_flat = consensus_buffer.mean(dim=0, keepdim=True)
        return ((consensus_buffer - mean_flat) ** 2).sum(dim=1).mean().item()