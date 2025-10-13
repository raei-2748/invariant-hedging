"""Penalty functions operating on flattened head gradients."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


def _ensure_grad_list(grad_list: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    grads: List[torch.Tensor] = []
    for grad in grad_list:
        if not isinstance(grad, torch.Tensor):
            raise TypeError("Gradient entries must be `torch.Tensor` instances.")
        grads.append(grad.reshape(-1))
    if not grads:
        raise ValueError("At least one gradient vector is required to compute a penalty.")
    expected_dim = grads[0].numel()
    for grad in grads[1:]:
        if grad.numel() != expected_dim:
            raise ValueError("All gradient vectors must share the same dimensionality.")
    return grads


def _normalise_rows(stacked: torch.Tensor, eps: float) -> torch.Tensor:
    if stacked.ndim != 2:
        raise ValueError("Expected a 2D tensor with gradients stacked along dim=0.")
    if stacked.shape[0] == 0:
        return stacked
    norms = torch.linalg.norm(stacked, ord=2, dim=1, keepdim=True)
    safe_norms = torch.clamp(norms, min=eps)
    scaled = stacked / safe_norms
    scaled = torch.where(norms <= eps, torch.zeros_like(scaled), scaled)
    return scaled


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    reduction = reduction.lower()
    if reduction == "mean":
        return values.mean() if values.numel() > 0 else values.new_zeros(())
    if reduction == "sum":
        return values.sum() if values.numel() > 0 else values.new_zeros(())
    if reduction == "none":
        return values
    raise ValueError(f"Unsupported reduction '{reduction}'.")


def cosine_alignment_penalty(
    grad_list: Sequence[torch.Tensor],
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    """Average pairwise cosine misalignment between environment gradients."""

    grads = _ensure_grad_list(grad_list)
    device = grads[0].device
    dtype = grads[0].dtype
    if len(grads) < 2:
        return torch.zeros((), device=device, dtype=dtype)

    stacked = torch.stack([g.to(device=device, dtype=dtype) for g in grads], dim=0)
    normalised = _normalise_rows(stacked, eps=eps)
    cos_matrix = normalised @ normalised.T
    indices = torch.triu_indices(cos_matrix.shape[0], cos_matrix.shape[1], offset=1)
    pairwise = cos_matrix[indices[0], indices[1]]
    penalty = 1.0 - pairwise
    return _reduce(penalty, reduction)


def varnorm_penalty(
    grad_list: Sequence[torch.Tensor],
    eps: float = 1e-12,
    reduction: str = "mean",
) -> torch.Tensor:
    """Variance of L2-normalised gradients across environments."""

    grads = _ensure_grad_list(grad_list)
    device = grads[0].device
    dtype = grads[0].dtype
    if len(grads) < 2:
        return torch.zeros((), device=device, dtype=dtype)

    stacked = torch.stack([g.to(device=device, dtype=dtype) for g in grads], dim=0)
    normalised = _normalise_rows(stacked, eps=eps)
    variances = normalised.var(dim=0, unbiased=False)
    if reduction.lower() == "none":
        return variances
    return _reduce(variances, reduction)


__all__ = ["cosine_alignment_penalty", "varnorm_penalty"]

