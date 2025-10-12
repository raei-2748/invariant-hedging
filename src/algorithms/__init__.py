"""Factory helpers for training algorithms."""
from __future__ import annotations

from typing import Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler

from ..irm.configs import IRMConfig
from .common import Algorithm
from .erm import ERMAlgorithm
from .irm import IRMAlgorithm, IRMSchedule
from .hirm import HIRMAlgorithm
from .vrex import VRExAlgorithm


def build_algorithm(
    name: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: Optional[float],
    config,
) -> Algorithm:
    key = name.lower()
    if key == "erm":
        return ERMAlgorithm(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
        )
    if key == "vrex":
        penalty_weight = float(getattr(config, "penalty_weight", 1.0))
        return VRExAlgorithm(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
            penalty_weight=penalty_weight,
        )
    if key == "irm":
        schedule = IRMSchedule(
            target=float(getattr(config, "lambda_target", getattr(config, "penalty_weight", 1.0))),
            warmup_steps=int(getattr(config, "warmup_steps", getattr(config, "ramp_steps", 0))),
            schedule=str(getattr(config, "schedule", "linear")),
            init=float(getattr(config, "lambda_init", 0.0)),
            delay_steps=int(getattr(config, "delay_steps", getattr(config, "pretrain_steps", 0))),
        )
        return IRMAlgorithm(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
            penalty_schedule=schedule,
        )
    if key == "hirm":
        irm_cfg = IRMConfig.from_config(getattr(config, "irm", None))
        detach = bool(getattr(config, "detach_features", False))
        return HIRMAlgorithm(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
            irm_config=irm_cfg,
            detach_features=detach,
        )
    raise ValueError(f"Unsupported algorithm '{name}'.")


__all__ = [
    "build_algorithm",
    "ERMAlgorithm",
    "IRMAlgorithm",
    "IRMSchedule",
    "HIRMAlgorithm",
    "VRExAlgorithm",
]
