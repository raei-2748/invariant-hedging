"""Invariant risk minimisation objective."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

from ..objectives import penalties
from ._base import OptimizerStepMixin, stack_losses
from .common import TrainBatch


@dataclass
class IRMSchedule:
    target: float
    warmup_steps: int
    schedule: str
    init: float = 0.0
    delay_steps: int = 0

    def weight(self, step: int) -> float:
        if step <= self.delay_steps:
            return float(self.init)
        if self.warmup_steps <= 0:
            return float(self.target)
        effective = max(0, step - self.delay_steps)
        progress = min(1.0, effective / float(self.warmup_steps))
        mode = self.schedule.lower()
        if mode == "linear":
            return float(self.init + (self.target - self.init) * progress)
        if mode == "cosine":
            cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
            value = self.init + (self.target - self.init) * cosine
            return float(value)
        if mode == "none":
            return float(self.target)
        raise ValueError(f"Unsupported IRM schedule '{self.schedule}'.")


class IRMAlgorithm(OptimizerStepMixin):
    """ERM augmented with an invariance penalty."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        grad_clip: Optional[float],
        penalty_schedule: IRMSchedule,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
        )
        device = next(model.parameters()).device
        self._dummy = torch.tensor(1.0, device=device, requires_grad=True)
        self._schedule = penalty_schedule

    @property
    def representation_scale(self) -> torch.Tensor | None:
        return self._dummy

    def step(self, batch: TrainBatch) -> dict[str, float]:
        self._optimizer.zero_grad(set_to_none=True)
        losses = stack_losses(batch.env_losses)
        risk = losses.mean()
        weight = self._schedule.weight(batch.step)
        penalty = penalties.irm_penalty((env.loss for env in batch.env_losses), self._dummy)
        total = risk + weight * penalty
        self._backward_and_step(total)
        return {
            "train/loss": float(risk.detach().item()),
            "train/penalty": float((weight * penalty).detach().item()),
            "train/irm_penalty": float(penalty.detach().item()),
            "train/lambda": float(weight),
        }


__all__ = ["IRMAlgorithm", "IRMSchedule"]
