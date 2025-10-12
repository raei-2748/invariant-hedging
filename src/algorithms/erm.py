"""Empirical risk minimisation training loop."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler

from ._base import OptimizerStepMixin, stack_losses
from .common import TrainBatch


class ERMAlgorithm(OptimizerStepMixin):
    """Vanilla ERM objective averaging environment losses."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[_LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        grad_clip: Optional[float],
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            grad_clip=grad_clip,
        )

    @property
    def representation_scale(self) -> torch.Tensor | None:
        return None

    def step(self, batch: TrainBatch) -> Dict[str, float]:
        self._optimizer.zero_grad(set_to_none=True)
        loss_tensor = stack_losses(batch.env_losses)
        total_loss = loss_tensor.mean()
        self._backward_and_step(total_loss)
        return {
            "train/loss": float(total_loss.detach().item()),
            "train/penalty": 0.0,
        }


__all__ = ["ERMAlgorithm"]
