"""Optimiser helpers and schedules used by the training engine."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from omegaconf import DictConfig


def lambda_at_step(step: int, *, target: float, schedule: str, warmup_steps: int) -> float:
    schedule = schedule.lower()
    if schedule not in {"none", "linear", "cosine"}:
        raise ValueError(f"Unsupported IRM schedule: {schedule}")
    if schedule == "none" or warmup_steps <= 0:
        return target
    progress = min(1.0, float(step) / float(max(1, warmup_steps)))
    if schedule == "linear":
        return target * progress
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return target * cosine


def label_smoothing(pnl: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return pnl
    mean = pnl.mean()
    return (1 - smoothing) * pnl + smoothing * mean


def setup_optimizer(
    policy: torch.nn.Module,
    cfg: DictConfig,
    extra_params: Iterable[torch.nn.Parameter] | None = None,
) -> torch.optim.Optimizer:
    name = cfg.optimizer.name.lower()
    params = [p for p in policy.parameters() if p.requires_grad]
    if extra_params is not None:
        params.extend(param for param in extra_params if param.requires_grad)
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay
    if cfg.model.name == "erm_reg":
        weight_decay = cfg.model.regularization.get("weight_decay", weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")


def setup_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    name = cfg.scheduler.name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.steps, eta_min=cfg.optimizer.lr * 0.1
        )
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


__all__ = [
    "lambda_at_step",
    "label_smoothing",
    "setup_optimizer",
    "setup_scheduler",
]
