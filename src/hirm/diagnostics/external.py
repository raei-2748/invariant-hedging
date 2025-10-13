"""External stability and robustness diagnostics."""

from __future__ import annotations

from typing import Mapping

import torch


def compute_IG(env2outcome: Mapping[str, torch.Tensor | float]) -> float:
    """Compute the invariance gap as the dispersion of outcomes across envs."""

    if not env2outcome:
        return 0.0
    values = []
    for outcome in env2outcome.values():
        if isinstance(outcome, torch.Tensor):
            tensor = outcome.reshape(-1).float()
            if tensor.numel() == 0:
                continue
            values.append(tensor.mean())
        else:
            values.append(torch.tensor(float(outcome), dtype=torch.float32))
    if len(values) <= 1:
        return 0.0
    stacked = torch.stack(values)
    return float(stacked.std(unbiased=False).item())


def compute_WG(env2risk: Mapping[str, torch.Tensor | float]) -> float:
    """Return the worst (largest) risk across environments."""

    if not env2risk:
        return 0.0
    max_value = None
    for risk in env2risk.values():
        if isinstance(risk, torch.Tensor):
            tensor = risk.reshape(-1).float()
            if tensor.numel() == 0:
                continue
            candidate = tensor.mean().item()
        else:
            candidate = float(risk)
        if max_value is None or candidate > max_value:
            max_value = candidate
    return float(max_value or 0.0)


def compute_VR(env2risk: Mapping[str, torch.Tensor | float]) -> float:
    """Variance of per-environment risk."""

    if not env2risk:
        return 0.0
    values = []
    for risk in env2risk.values():
        if isinstance(risk, torch.Tensor):
            tensor = risk.reshape(-1).float()
            if tensor.numel() == 0:
                continue
            values.append(tensor.mean())
        else:
            values.append(torch.tensor(float(risk), dtype=torch.float32))
    if len(values) <= 1:
        return 0.0
    stacked = torch.stack(values)
    return float(stacked.var(unbiased=False).item())


__all__ = ["compute_IG", "compute_WG", "compute_VR"]

