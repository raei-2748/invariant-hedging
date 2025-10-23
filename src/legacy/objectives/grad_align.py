"""Gradient alignment utilities for HIRM."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def _flatten_gradients(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    pieces = [g.reshape(-1) for g in grads if g is not None]
    if not pieces:
        raise ValueError("At least one gradient tensor is required per environment.")
    return torch.cat(pieces)


def normalized_head_grads(
    head_grads: Sequence[Sequence[torch.Tensor]],
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return L2-normalised flattened gradients for each environment."""

    if not head_grads:
        raise ValueError("No gradients provided for normalisation.")

    flattened: list[torch.Tensor] = []
    expected_dim: int | None = None
    for env_grads in head_grads:
        vector = _flatten_gradients(env_grads)
        if expected_dim is None:
            expected_dim = vector.numel()
        elif vector.numel() != expected_dim:
            raise ValueError("All gradient vectors must share the same dimensionality.")
        norm = torch.linalg.norm(vector)
        if norm <= eps:
            vector = torch.zeros_like(vector)
        else:
            vector = vector / norm
        flattened.append(vector)
    return torch.stack(flattened, dim=0)


def pairwise_cosine(vectors: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Compute pairwise cosine similarities for the given row vectors."""

    if vectors.ndim != 2:
        raise ValueError("Expected a 2D tensor of shape (n_envs, dim).")
    if vectors.shape[0] < 2:
        return vectors.new_zeros(0)
    normalised = F.normalize(vectors, p=2, dim=1, eps=eps)
    sim_matrix = normalised @ normalised.T
    idx = torch.triu_indices(sim_matrix.shape[0], sim_matrix.shape[1], offset=1)
    return sim_matrix[idx[0], idx[1]]


def env_variance(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Variance of per-environment scalar quantities."""

    if not values:
        raise ValueError("No environment values provided for variance computation.")
    scalars = []
    for value in values:
        if value.ndim == 0:
            scalars.append(value)
        else:
            scalars.append(value.reshape(-1).mean())
    stacked = torch.stack(scalars)
    if stacked.numel() <= 1:
        return torch.zeros((), device=stacked.device, dtype=stacked.dtype)
    return stacked.var(unbiased=False)


__all__ = ["normalized_head_grads", "pairwise_cosine", "env_variance"]
