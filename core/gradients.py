"""Gradient helpers used by the current multi-GPU training path."""

import torch


def get_gradient_vector(network):
    """Flatten the current in-memory gradients of a model into one vector."""
    gradient_chunks = []
    for param in network.parameters():
        if param.grad is not None:
            gradient_chunks.append(param.grad.view(-1))

    if not gradient_chunks:
        return torch.zeros(1, dtype=torch.float16)
    return torch.cat(gradient_chunks).to(torch.float16)


__all__ = ["get_gradient_vector"]