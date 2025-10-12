"""Base helpers shared across algorithm implementations."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler

from .common import Algorithm, TrainBatch, stack_losses


class OptimizerStepMixin:
    """Utility mixin providing gradient scaling and optimiser stepping."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        scheduler: Optional[_LRScheduler],
        grad_clip: Optional[float],
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scaler = scaler
        self._grad_clip = grad_clip

    def _backward_and_step(self, loss: torch.Tensor) -> None:
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            if self._grad_clip is not None and self._grad_clip > 0:
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss.backward()
            if self._grad_clip is not None and self._grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
            self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()

    def state_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "optimizer": self._optimizer.state_dict(),
        }
        if self._scheduler is not None:
            payload["scheduler"] = self._scheduler.state_dict()
        if self._scaler is not None:
            payload["scaler"] = self._scaler.state_dict()
        return payload


__all__ = ["OptimizerStepMixin", "Algorithm", "TrainBatch", "stack_losses"]
