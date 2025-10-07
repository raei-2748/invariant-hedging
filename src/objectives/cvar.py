"""Conditional Value-at-Risk utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CVaRResult:
    value: float
    lower: float
    upper: float


def cvar_loss(losses: torch.Tensor, alpha: float) -> torch.Tensor:
    if losses.ndim != 1:
        losses = losses.reshape(-1)
    var = torch.quantile(losses, alpha)
    tail = losses[losses >= var]
    if tail.numel() == 0:
        return losses.mean()
    return tail.mean()


def cvar_from_pnl(pnl: torch.Tensor, alpha: float) -> torch.Tensor:
    losses = -pnl
    return cvar_loss(losses, alpha)


def bootstrap_cvar_ci(
    pnl: torch.Tensor,
    alpha: float,
    num_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CVaRResult:
    rng = np.random.default_rng(seed)
    losses = (-pnl).detach().cpu().numpy().reshape(-1)
    estimates = []
    n = len(losses)
    for _ in range(num_samples):
        indices = rng.integers(0, n, size=n)
        sample = torch.as_tensor(losses[indices])
        estimates.append(float(cvar_loss(sample, alpha)))
    mean_estimate = float(np.mean(estimates))
    lower = float(np.quantile(estimates, (1 - confidence) / 2))
    upper = float(np.quantile(estimates, 1 - (1 - confidence) / 2))
    return CVaRResult(value=mean_estimate, lower=lower, upper=upper)


def differentiable_cvar(losses: torch.Tensor, alpha: float, eps: float = 1e-6) -> torch.Tensor:
    """Smooth approximation to CVaR using the Rockafellar-Uryasev formulation."""
    tau = torch.quantile(losses, alpha)
    hinge = torch.clamp(losses - tau, min=0.0)
    tail_fraction = max(eps, 1.0 - alpha)
    return tau + hinge.mean() / tail_fraction
