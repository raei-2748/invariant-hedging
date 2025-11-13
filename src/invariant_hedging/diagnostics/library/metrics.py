"""Diagnostic metrics for invariant hedging runs."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch


def invariant_gap(per_env_values: Sequence[float]) -> float:
    tensor = torch.tensor(per_env_values, dtype=torch.float32)
    if tensor.numel() <= 1:
        return 0.0
    return float(tensor.std(unbiased=False).item())


def worst_group(per_env_values: Sequence[float]) -> float:
    if not per_env_values:
        return 0.0
    return float(max(per_env_values))


def mechanistic_sensitivity(sensitivities: Iterable[float]) -> float:
    values = list(sensitivities)
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item())


__all__ = ["invariant_gap", "worst_group", "mechanistic_sensitivity"]
