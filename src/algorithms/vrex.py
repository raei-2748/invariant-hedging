"""Variance risk extrapolation (V-REx) objective."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler

from ..objectives import penalties
from ._base import OptimizerStepMixin, stack_losses
from .common import TrainBatch


class VRExAlgorithm(OptimizerStepMixin):
    """V-REx penalty encouraging equal risk across environments."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        grad_clip: Optional[float],
        penalty_weight: float,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
        )
        self._penalty_weight = float(penalty_weight)

    @property
    def representation_scale(self) -> torch.Tensor | None:
        return None

    def step(self, batch: TrainBatch) -> Dict[str, float]:
        self._optimizer.zero_grad(set_to_none=True)
        losses = stack_losses(batch.env_losses)
        risk = losses.mean()
        penalty = penalties.vrex_penalty(env.loss for env in batch.env_losses)
        total = risk + self._penalty_weight * penalty
        self._backward_and_step(total)
        return {
            "train/loss": float(risk.detach().item()),
            "train/penalty": float((self._penalty_weight * penalty).detach().item()),
        }


__all__ = ["VRExAlgorithm"]
