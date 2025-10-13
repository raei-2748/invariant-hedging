"""Entropic risk measures for comparison baselines."""
from __future__ import annotations

import torch


def entropic_risk(losses: torch.Tensor, eta: float) -> torch.Tensor:
    losses = losses.reshape(-1)
    return (1.0 / eta) * torch.log(torch.mean(torch.exp(eta * losses)))


def entropic_risk_from_pnl(pnl: torch.Tensor, eta: float) -> torch.Tensor:
    return entropic_risk(-pnl, eta)
