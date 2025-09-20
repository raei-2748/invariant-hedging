from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch

from hirm_experiment.training.pnl import HedgingOutputs


def cvar(losses: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    sorted_losses, _ = torch.sort(losses, descending=True)
    cutoff = max(int((1.0 - alpha) * losses.shape[0]), 1)
    tail = sorted_losses[:cutoff]
    return tail.mean()


def max_drawdown(step_pnl: torch.Tensor) -> torch.Tensor:
    equity = step_pnl.cumsum(dim=1)
    peak = torch.cummax(equity, dim=1).values
    drawdown = peak - equity
    return drawdown.max(dim=1).values.mean()


def sharpe_ratio(pnl: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = pnl.mean()
    std = pnl.std(unbiased=False)
    return mean / (std + eps)


def confidence_interval(samples: torch.Tensor, level: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = samples.mean()
    std = samples.std(unbiased=False)
    z = 1.96 if level == 0.95 else torch.tensor(1.96)
    half_width = z * std / torch.sqrt(torch.tensor(float(samples.shape[0])))
    return mean - half_width, mean + half_width


def compute_metrics(outputs: HedgingOutputs, alpha: float = 0.95) -> Dict[str, float]:
    pnl = outputs.episode_pnl.detach()
    losses = -pnl
    metrics = {
        "mean_pnl": float(pnl.mean().cpu()),
        "std_pnl": float(pnl.std(unbiased=False).cpu()),
        "cvar_95": float(cvar(losses, alpha).cpu()),
        "turnover": float(outputs.turnover.mean().cpu()),
        "max_drawdown": float(max_drawdown(outputs.step_pnl.detach()).cpu()),
        "sharpe": float(sharpe_ratio(pnl).cpu()),
    }
    ci_low, ci_high = confidence_interval(pnl, alpha)
    metrics["pnl_ci_low"] = float(ci_low.cpu())
    metrics["pnl_ci_high"] = float(ci_high.cpu())
    return metrics


def aggregate_seed_metrics(seed_metrics: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    collected: Dict[str, list] = {}
    for metrics in seed_metrics:
        for key, value in metrics.items():
            collected.setdefault(key, []).append(value)
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in collected.items():
        tensor = torch.tensor(values, dtype=torch.float32)
        mean = tensor.mean()
        std = tensor.std(unbiased=False)
        ci_low, ci_high = confidence_interval(tensor, 0.95)
        summary[key] = {
            "mean": float(mean),
            "std": float(std),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }
    return summary
