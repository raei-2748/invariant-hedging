"""Statistical utilities for evaluation and reporting."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch


def cumulative_paths(step_pnl: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(step_pnl, dim=1)


def max_drawdown(step_pnl: torch.Tensor) -> torch.Tensor:
    equity = cumulative_paths(step_pnl)
    running_max = torch.cummax(equity, dim=1)[0]
    drawdowns = equity - running_max
    return -drawdowns.min(dim=1).values


def sharpe_ratio(step_pnl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = step_pnl.mean(dim=1)
    std = step_pnl.std(dim=1, unbiased=False)
    daily_sharpe = mean / (std + eps)
    return daily_sharpe * math.sqrt(252)


def sortino_ratio(step_pnl: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = step_pnl.mean(dim=1)
    downside = torch.clamp(-step_pnl, min=0.0)
    downside_var = (downside**2).mean(dim=1)
    downside_std = torch.sqrt(downside_var + eps)
    daily_sortino = mean / (downside_std + eps)
    return daily_sortino * math.sqrt(252)


def turnover_ratio(turnover: torch.Tensor, notional: float) -> torch.Tensor:
    return turnover / (notional + 1e-8)


def qq_plot_data(pnl: torch.Tensor, reference: torch.Tensor) -> Dict[str, np.ndarray]:
    pnl_sorted, _ = torch.sort(pnl)
    ref_sorted, _ = torch.sort(reference)
    n = min(pnl_sorted.numel(), ref_sorted.numel())
    q = torch.linspace(0, 1, n)
    pnl_quant = torch.quantile(pnl_sorted, q)
    ref_quant = torch.quantile(ref_sorted, q)
    return {"quantiles": q.numpy(), "model": pnl_quant.numpy(), "reference": ref_quant.numpy()}


def bootstrap_mean_ci(values: torch.Tensor, samples: int = 1000, confidence: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = values.detach().cpu().numpy().reshape(-1)
    est = []
    n = len(arr)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        est.append(arr[idx].mean())
    lower = float(np.quantile(est, (1 - confidence) / 2))
    upper = float(np.quantile(est, 1 - (1 - confidence) / 2))
    return float(np.mean(est)), lower, upper
