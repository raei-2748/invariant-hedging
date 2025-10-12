"""Invariance diagnostics utilities.

This module implements the core C1/C2/C3 components of the invariance
diagnostics described in PR-04 as well as the aggregated Invariance Spectrum
Index (ISI).  The functions operate on simple ``torch.Tensor`` inputs so they
are easy to reuse from both the training code and the accompanying unit tests.

The computations intentionally avoid any side-effects (e.g. no gradient
updates) so they can be used during evaluation without impacting the trained
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ISINormalizationConfig:
    """Normalization parameters for the C1 and C3 stability components."""

    c1_max_dispersion: float = 1.0
    c3_max_distance: float = 1.0


def _as_tensor(values: Iterable[torch.Tensor | float]) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            tensors.append(value.reshape(-1).to(dtype=torch.float32))
        else:
            tensors.append(torch.tensor([float(value)], dtype=torch.float32))
    if not tensors:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(tensors)


def _safe_divide(num: torch.Tensor, denom: float) -> torch.Tensor:
    if denom <= 0:
        return torch.zeros_like(num)
    return num / float(denom)


def compute_C1_global_stability(
    env2risk: Mapping[str, torch.Tensor | float],
    norm_cfg: ISINormalizationConfig | Mapping[str, float] | None,
) -> float:
    """Compute the C1 (global stability) component.

    Parameters
    ----------
    env2risk:
        Mapping from environment identifier to a tensor (or scalar) with risk
        values.  Each tensor is flattened prior to aggregation which allows
        callers to pass either batched values or already aggregated scalars.
    norm_cfg:
        Configuration specifying the maximum dispersion used to map the
        variance/range to the stability score.  ``None`` defaults to a maximum
        dispersion of ``1.0``.
    """

    if not env2risk:
        return 1.0

    if isinstance(norm_cfg, Mapping):
        max_disp = float(norm_cfg.get("c1_max_dispersion", 1.0))
    elif isinstance(norm_cfg, ISINormalizationConfig):
        max_disp = float(norm_cfg.c1_max_dispersion)
    else:
        max_disp = 1.0

    per_env_means: List[torch.Tensor] = []
    for tensor in env2risk.values():
        flattened = _as_tensor([tensor])
        if flattened.numel() == 0:
            continue
        per_env_means.append(flattened.float().mean())

    if len(per_env_means) <= 1:
        return 1.0

    stacked = torch.stack(per_env_means)
    dispersion = stacked.var(unbiased=False)
    normalized = torch.clamp(_safe_divide(dispersion, max_disp), min=0.0, max=1.0)
    stability = 1.0 - normalized.item()
    return float(max(0.0, min(1.0, stability)))


def compute_C2_mechanistic_stability(
    head_grads: Sequence[torch.Tensor | Sequence[float]],
) -> float:
    """Compute the C2 (mechanistic stability) component."""

    grads: List[torch.Tensor] = []
    for grad in head_grads:
        if isinstance(grad, torch.Tensor):
            tensor = grad.reshape(-1).float()
        else:
            tensor = torch.tensor(list(grad), dtype=torch.float32)
        if tensor.numel() == 0:
            continue
        grads.append(tensor)

    if len(grads) <= 1:
        return 1.0

    cos_values: List[torch.Tensor] = []
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            a = grads[i]
            b = grads[j]
            cos = F.cosine_similarity(a, b, dim=0, eps=1e-12)
            cos_values.append((cos + 1.0) * 0.5)

    if not cos_values:
        return 1.0

    avg = torch.stack(cos_values).mean().clamp(0.0, 1.0)
    return float(avg.item())


def compute_C3_structural_stability(
    env2repr: Mapping[str, torch.Tensor | Sequence[float]],
    norm_cfg: ISINormalizationConfig | Mapping[str, float] | None,
) -> float:
    """Compute the C3 (structural stability) component."""

    if not env2repr:
        return 1.0

    if isinstance(norm_cfg, Mapping):
        max_distance = float(norm_cfg.get("c3_max_distance", 1.0))
    elif isinstance(norm_cfg, ISINormalizationConfig):
        max_distance = float(norm_cfg.c3_max_distance)
    else:
        max_distance = 1.0

    repr_means: List[torch.Tensor] = []
    for tensor in env2repr.values():
        if isinstance(tensor, torch.Tensor):
            mean_vec = tensor.reshape(tensor.shape[0], -1).mean(dim=0)
        else:
            tensor_t = torch.tensor(list(tensor), dtype=torch.float32)
            mean_vec = tensor_t.reshape(1, -1).mean(dim=0)
        repr_means.append(mean_vec.float())

    if len(repr_means) <= 1:
        return 1.0

    pairwise_distances: List[torch.Tensor] = []
    for i in range(len(repr_means)):
        for j in range(i + 1, len(repr_means)):
            dist = F.pairwise_distance(
                repr_means[i].unsqueeze(0), repr_means[j].unsqueeze(0), p=2
            )
            pairwise_distances.append(dist.squeeze(0))

    if not pairwise_distances:
        return 1.0

    avg_distance = torch.stack(pairwise_distances).mean()
    normalized = torch.clamp(_safe_divide(avg_distance, max_distance), min=0.0, max=1.0)
    stability = 1.0 - normalized.item()
    return float(max(0.0, min(1.0, stability)))


def compute_ISI(
    C1: float,
    C2: float,
    C3: float,
    weights: Sequence[float] | None = None,
) -> float:
    """Aggregate the C-components into the final ISI score."""

    if weights is None:
        weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    if len(tuple(weights)) != 3:
        raise ValueError("ISI weights must have exactly three entries")

    w1, w2, w3 = (float(w) for w in weights)
    total = w1 + w2 + w3
    if total <= 0:
        raise ValueError("ISI weights must sum to a positive value")

    normalized_weights = (w1 / total, w2 / total, w3 / total)
    isi = (
        normalized_weights[0] * float(C1)
        + normalized_weights[1] * float(C2)
        + normalized_weights[2] * float(C3)
    )
    return float(max(0.0, min(1.0, isi)))


__all__ = [
    "ISINormalizationConfig",
    "compute_C1_global_stability",
    "compute_C2_mechanistic_stability",
    "compute_C3_structural_stability",
    "compute_ISI",
]

