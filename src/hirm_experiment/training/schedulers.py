from __future__ import annotations

import math
from typing import Optional

from torch.optim import Optimizer


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_num = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_num += 1
        for idx, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self._lr_for_step(self.base_lrs[idx])

    def _lr_for_step(self, base_lr: float) -> float:
        if self.step_num <= self.warmup_steps and self.warmup_steps > 0:
            return base_lr * self.step_num / self.warmup_steps
        progress = min(max(self.step_num - self.warmup_steps, 0), self.total_steps - self.warmup_steps)
        if self.total_steps == self.warmup_steps:
            cosine = 0.0
        else:
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress / (self.total_steps - self.warmup_steps)))
        return self.min_lr + (base_lr - self.min_lr) * cosine


def build_scheduler(optimizer: Optimizer, config: dict, total_steps: int) -> Optional[WarmupCosineScheduler]:
    name = config.get("name")
    if name is None:
        return None
    if name == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=int(config.get("warmup_steps", 0)),
            min_lr=float(config.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")
