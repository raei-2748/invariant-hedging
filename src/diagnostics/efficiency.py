"""Efficiency metrics for diagnostics exports."""

from __future__ import annotations

from typing import Iterable, Mapping

import torch


def compute_ER(outcomes: torch.Tensor | Iterable[float]) -> float:
    """Average economic return for the diagnostic probe."""

    if isinstance(outcomes, torch.Tensor):
        tensor = outcomes.reshape(-1).float()
    else:
        tensor = torch.tensor(list(outcomes), dtype=torch.float32)
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.mean().item())


def compute_TR(positions: torch.Tensor) -> float:
    """Mean absolute turnover computed from position deltas."""

    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim < 2:
        # Interpret as (batch, time) even if squeezed
        positions = positions.reshape(positions.shape[0], -1)
    deltas = positions.diff(dim=-1)
    turnover = deltas.abs().mean()
    return float(turnover.item()) if turnover.numel() else 0.0


__all__ = ["compute_ER", "compute_TR"]

