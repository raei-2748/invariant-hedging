"""Invariant subspace diagnostics for evaluation outputs."""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ClampConfig = Optional[Tuple[Optional[float], Optional[float]]]


def _as_tensor(data: Sequence[Sequence[float]] | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2:
        raise ValueError("Invariant diagnostics expect a 2D tensor")
    return tensor.clone().detach()


def _apply_clamp(matrix: torch.Tensor, clamp: ClampConfig) -> torch.Tensor:
    if clamp is None:
        return matrix
    min_val, max_val = clamp
    if min_val is None and max_val is None:
        return matrix
    return matrix.clamp(min=min_val if min_val is not None else -math.inf, max=max_val if max_val is not None else math.inf)


def _trim_rows(matrix: torch.Tensor, trim_fraction: float) -> torch.Tensor:
    if trim_fraction <= 0.0:
        return matrix
    if trim_fraction >= 0.5:
        raise ValueError("trim_fraction must be < 0.5")
    rows = matrix.shape[0]
    trim_count = int(math.floor(rows * trim_fraction))
    if trim_count == 0:
        return matrix
    if trim_count * 2 >= rows:
        # keep the central row with smallest norm to avoid empty tensor
        norms = torch.linalg.vector_norm(matrix, dim=1)
        keep = norms.argmin().unsqueeze(0)
        return matrix.index_select(0, keep)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    sorted_indices = torch.argsort(norms)
    keep = sorted_indices[trim_count : rows - trim_count]
    return matrix.index_select(0, keep)


def _prepare_matrix(
    data: Sequence[Sequence[float]] | torch.Tensor,
    trim_fraction: float = 0.0,
    clamp: ClampConfig = None,
) -> torch.Tensor:
    matrix = _as_tensor(data)
    matrix = _apply_clamp(matrix, clamp)
    matrix = _trim_rows(matrix, trim_fraction)
    return matrix


def compute_C1(matrix: Sequence[Sequence[float]] | torch.Tensor, alignment: str = "cosine", eps: float = 1e-8) -> Optional[float]:
    prepared = _as_tensor(matrix)
    if prepared.shape[0] == 0:
        return None
    mean_vec = prepared.mean(dim=0, keepdim=True)
    if torch.linalg.vector_norm(mean_vec) < eps:
        return 0.0
    if alignment == "cosine":
        scores = F.cosine_similarity(prepared, mean_vec.expand_as(prepared), dim=1, eps=eps)
    elif alignment == "dot":
        mean = mean_vec.squeeze(0)
        denom = torch.linalg.vector_norm(mean) * torch.linalg.vector_norm(prepared, dim=1) + eps
        scores = prepared.matmul(mean) / denom
    else:
        raise ValueError(f"Unsupported alignment metric '{alignment}'")
    return float(scores.mean().item())


def compute_C2(matrix: Sequence[Sequence[float]] | torch.Tensor) -> Optional[float]:
    prepared = _as_tensor(matrix)
    if prepared.shape[0] == 0:
        return None
    centered = prepared - prepared.mean(dim=0, keepdim=True)
    dispersion = torch.linalg.vector_norm(centered, dim=1)
    return float(dispersion.mean().item())


def _covariance_dispersion(matrix: torch.Tensor, mode: str, eps: float) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    cov = centered.T.matmul(centered) / max(matrix.shape[0] - 1, 1)
    if mode == "trace":
        return float(torch.trace(cov).item())
    if mode == "frobenius":
        return float(torch.linalg.matrix_norm(cov, ord="fro").item())
    raise ValueError(f"Unsupported covariance dispersion '{mode}'")


def compute_C3(
    matrix: Sequence[Sequence[float]] | torch.Tensor,
    covariance_dispersion: str = "trace",
    eps: float = 1e-8,
) -> Optional[float]:
    prepared = _as_tensor(matrix)
    if prepared.shape[0] == 0:
        return None
    return _covariance_dispersion(prepared, covariance_dispersion, eps)


def compute_ISI(
    matrix: Sequence[Sequence[float]] | torch.Tensor,
    trim_fraction: float = 0.0,
    clamp: ClampConfig = None,
    alignment: str = "cosine",
    covariance_dispersion: str = "trace",
    eps: float = 1e-8,
) -> Mapping[str, Optional[float]]:
    prepared = _prepare_matrix(matrix, trim_fraction=trim_fraction, clamp=clamp)
    if prepared.shape[0] == 0:
        return {"C1": None, "C2": None, "C3": None, "ISI": None}
    c1 = compute_C1(prepared, alignment=alignment, eps=eps)
    c2 = compute_C2(prepared)
    c3 = compute_C3(prepared, covariance_dispersion=covariance_dispersion, eps=eps)
    if c1 is None or c2 is None or c3 is None:
        isi_value = None
    else:
        isi_value = float(c1 / (1.0 + c2 + c3))
    return {"C1": c1, "C2": c2, "C3": c3, "ISI": isi_value}


__all__ = ["compute_C1", "compute_C2", "compute_C3", "compute_ISI"]
