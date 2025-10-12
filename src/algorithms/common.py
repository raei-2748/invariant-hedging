"""Common types and base helpers for training algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Protocol

import torch


@dataclass
class EnvLoss:
    """Per-environment loss payload passed to algorithms."""

    name: str
    loss: torch.Tensor
    payload: Dict[str, Any]


@dataclass
class TrainBatch:
    """Container aggregating environment losses for a single optimisation step."""

    step: int
    env_losses: List[EnvLoss]


class Algorithm(Protocol):
    """Minimal protocol shared by training algorithms."""

    def step(self, batch: TrainBatch) -> Dict[str, float]:
        """Perform one optimisation step and return logging metrics."""

    def state_dict(self) -> Dict[str, Any]:
        """Return serialisable algorithm state (optimiser, schedulers, etc.)."""

    @property
    def representation_scale(self) -> torch.Tensor | None:
        """Optional scaling tensor injected into the policy representation."""


def stack_losses(env_losses: Iterable[EnvLoss]) -> torch.Tensor:
    losses = [env.loss.reshape(()) for env in env_losses]
    if not losses:
        raise ValueError("At least one environment loss is required per optimisation step.")
    return torch.stack(losses)


__all__ = ["Algorithm", "EnvLoss", "TrainBatch", "stack_losses"]
