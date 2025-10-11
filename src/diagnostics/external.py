"""Aggregate diagnostics for per-environment risk metrics."""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ClampConfig = Optional[Tuple[Optional[float], Optional[float]]]


def _vector(values: Iterable[float]) -> torch.Tensor:
    data = list(values)
    if not data:
        return torch.empty(0, dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32)


def _matrix(values: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2:
        raise ValueError("Expected 2D data for diagnostic aggregation")
    return tensor.clone().detach()


def _apply_clamp(tensor: torch.Tensor, clamp: ClampConfig) -> torch.Tensor:
    if clamp is None:
        return tensor
    min_val, max_val = clamp
    if min_val is None and max_val is None:
        return tensor
    return tensor.clamp(min=min_val if min_val is not None else -math.inf, max=max_val if max_val is not None else math.inf)


def _trim_vector(tensor: torch.Tensor, trim_fraction: float) -> torch.Tensor:
    if trim_fraction <= 0.0 or tensor.numel() == 0:
        return tensor
    if trim_fraction >= 0.5:
        raise ValueError("trim_fraction must be < 0.5")
    trim_count = int(math.floor(tensor.numel() * trim_fraction))
    if trim_count == 0:
        return tensor
    if trim_count * 2 >= tensor.numel():
        return tensor.narrow(0, trim_count, tensor.numel() - trim_count)
    sorted_vals, _ = torch.sort(tensor)
    return sorted_vals[trim_count : tensor.numel() - trim_count]


def _trim_matrix(tensor: torch.Tensor, trim_fraction: float) -> torch.Tensor:
    if trim_fraction <= 0.0 or tensor.shape[0] == 0:
        return tensor
    if trim_fraction >= 0.5:
        raise ValueError("trim_fraction must be < 0.5")
    rows = tensor.shape[0]
    trim_count = int(math.floor(rows * trim_fraction))
    if trim_count == 0:
        return tensor
    if trim_count * 2 >= rows:
        norms = torch.linalg.vector_norm(tensor, dim=1)
        keep = norms.argmin().unsqueeze(0)
        return tensor.index_select(0, keep)
    norms = torch.linalg.vector_norm(tensor, dim=1)
    order = torch.argsort(norms)
    keep = order[trim_count : rows - trim_count]
    return tensor.index_select(0, keep)


def _sanitize_vector(values: Iterable[float], trim_fraction: float, clamp: ClampConfig) -> torch.Tensor:
    tensor = _vector(values)
    tensor = _apply_clamp(tensor, clamp)
    tensor = _trim_vector(tensor, trim_fraction)
    return tensor


def _sanitize_matrix(values: Sequence[Sequence[float]] | torch.Tensor, trim_fraction: float, clamp: ClampConfig) -> torch.Tensor:
    tensor = _matrix(values)
    tensor = _apply_clamp(tensor, clamp)
    tensor = _trim_matrix(tensor, trim_fraction)
    return tensor


def compute_ig(values: Iterable[float], trim_fraction: float = 0.0, clamp: ClampConfig = None) -> Optional[float]:
    tensor = _sanitize_vector(values, trim_fraction, clamp)
    if tensor.numel() <= 1:
        return None
    return float((tensor.max() - tensor.min()).item())


def compute_wg(
    train_values: Iterable[float],
    test_values: Iterable[float],
    trim_fraction: float = 0.0,
    clamp: ClampConfig = None,
) -> Optional[float]:
    train_tensor = _sanitize_vector(train_values, trim_fraction, clamp)
    test_tensor = _sanitize_vector(test_values, trim_fraction, clamp)
    if train_tensor.numel() == 0 or test_tensor.numel() == 0:
        return None
    return float(test_tensor.max().item() - train_tensor.max().item())


def _covariance_dispersion(tensor: torch.Tensor, mode: str, eps: float) -> float:
    if tensor.shape[0] <= 1:
        return 0.0
    centered = tensor - tensor.mean(dim=0, keepdim=True)
    cov = centered.T.matmul(centered) / max(tensor.shape[0] - 1, 1)
    if mode == "trace":
        return float(torch.trace(cov).item())
    if mode == "frobenius":
        return float(torch.linalg.matrix_norm(cov, ord="fro").item())
    raise ValueError(f"Unsupported covariance dispersion '{mode}'")


def _alignment_score(tensor: torch.Tensor, alignment: str, eps: float) -> float:
    if tensor.shape[0] == 0:
        return 0.0
    mean_vec = tensor.mean(dim=0, keepdim=True)
    if alignment == "cosine":
        scores = F.cosine_similarity(tensor, mean_vec.expand_as(tensor), dim=1, eps=eps)
    elif alignment == "dot":
        denom = torch.linalg.vector_norm(mean_vec) * torch.linalg.vector_norm(tensor, dim=1) + eps
        scores = tensor.matmul(mean_vec.squeeze(0)) / denom
    else:
        raise ValueError(f"Unsupported alignment metric '{alignment}'")
    return float(scores.mean().item())


def compute_variation_ratio(
    matrix: Sequence[Sequence[float]] | torch.Tensor,
    trim_fraction: float = 0.0,
    clamp: ClampConfig = None,
    alignment: str = "cosine",
    covariance_dispersion: str = "trace",
    eps: float = 1e-8,
) -> Optional[float]:
    tensor = _sanitize_matrix(matrix, trim_fraction, clamp)
    if tensor.shape[0] <= 1:
        return None
    align = _alignment_score(tensor, alignment, eps)
    dispersion = _covariance_dispersion(tensor, covariance_dispersion, eps)
    variation = max(1.0 - align, 0.0)
    scaled = variation * (dispersion / max(tensor.shape[1], 1))
    return float(scaled)


def compute_expected_risk(
    values: Iterable[float],
    trim_fraction: float = 0.0,
    clamp: ClampConfig = None,
) -> Optional[float]:
    tensor = _sanitize_vector(values, trim_fraction, clamp)
    if tensor.numel() == 0:
        return None
    return float(tensor.mean().item())


def compute_tail_risk(
    values: Iterable[float],
    trim_fraction: float = 0.0,
    clamp: ClampConfig = None,
    covariance_dispersion: str = "trace",
    quantile: float = 0.95,
    eps: float = 1e-8,
) -> Optional[float]:
    tensor = _sanitize_vector(values, trim_fraction, clamp)
    if tensor.numel() == 0:
        return None
    if tensor.numel() == 1:
        tail = tensor[0]
    else:
        q = torch.tensor(float(quantile), dtype=tensor.dtype)
        tail = torch.quantile(tensor, q.clamp(eps, 1 - eps))
    dispersion = _covariance_dispersion(tensor.unsqueeze(1), covariance_dispersion, eps)
    return float(tail.item() + dispersion)


__all__ = [
    "compute_ig",
    "compute_wg",
    "compute_variation_ratio",
    "compute_expected_risk",
    "compute_tail_risk",
]
