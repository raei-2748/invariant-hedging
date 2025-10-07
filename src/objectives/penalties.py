"""Regularization penalties used in distributionally robust training."""

from __future__ import annotations

from typing import Iterable

import torch


def irm_penalty(env_losses: Iterable[torch.Tensor], dummy: torch.Tensor) -> torch.Tensor:
    penalties: list[torch.Tensor] = []
    for loss in env_losses:
        grad = torch.autograd.grad(loss, dummy, create_graph=True)[0]
        penalties.append(grad.pow(2))
    return torch.mean(torch.stack(penalties))


def vrex_penalty(env_losses: Iterable[torch.Tensor]) -> torch.Tensor:
    losses = torch.stack([loss for loss in env_losses])
    return losses.var(unbiased=False)


def groupdro_objective(weights: torch.Tensor, env_losses: Iterable[torch.Tensor]) -> torch.Tensor:
    losses = torch.stack([loss for loss in env_losses])
    return torch.dot(weights, losses)


def update_groupdro_weights(
    weights: torch.Tensor,
    env_losses: Iterable[torch.Tensor],
    step_size: float,
) -> torch.Tensor:
    losses = torch.stack([loss.detach() for loss in env_losses])
    new_weights = weights * torch.exp(step_size * losses)
    new_weights = torch.clamp(new_weights, min=1e-6)
    return new_weights / new_weights.sum()
