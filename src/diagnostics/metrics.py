"""Diagnostic metrics for invariant hedging runs."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch


def invariant_gap(per_env_values: Sequence[float]) -> float:
    tensor = torch.tensor(per_env_values, dtype=torch.float32)
    if tensor.numel() <= 1:
        return 0.0
    return float(tensor.std(unbiased=False).item())


def worst_group(
    per_env_values: Sequence[float | torch.Tensor], *, mode: str = "reward"
) -> float:
    """Return the worst performance across environments for the given ``mode``.

    ``mode`` determines whether lower values correspond to better outcomes.
    For reward/utility metrics the worst group is the minimum reward, whereas
    for loss-like metrics (where higher numbers are worse) the worst group is
    the maximum value.
    """

    if not per_env_values:
        return 0.0

    values: list[float] = []
    for value in per_env_values:
        if isinstance(value, torch.Tensor):
            tensor = value.reshape(-1)
            if tensor.numel() != 1:
                raise ValueError("worst_group expects scalar per-environment values")
            values.append(float(tensor.item()))
        else:
            values.append(float(value))

    if not values:
        return 0.0

    if mode == "reward":
        reducer = min
    elif mode == "loss":
        reducer = max
    else:
        raise ValueError(f"Unsupported worst_group mode: {mode!r}")

    return float(reducer(values))


def mechanistic_sensitivity(sensitivities: Iterable[float]) -> float:
    values = list(sensitivities)
    if not values:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item())


__all__ = ["invariant_gap", "worst_group", "mechanistic_sensitivity"]
