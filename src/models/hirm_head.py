"""Risk-only IRM head used for HIRM-Head experiments."""
from __future__ import annotations

import torch
from torch import nn


class HIRMHead(nn.Module):
    """Risk estimator head equipped with an IRMv1 scaling parameter."""

    def __init__(self, base_repr: nn.Module, risk_head: nn.Module) -> None:
        super().__init__()
        self.repr = base_repr
        self.risk_head = risk_head
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return risk predictions for the provided features."""

        phi = self.repr(features)
        return self.risk_head(phi)


def irm_penalty(per_env_loss: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """IRMv1 penalty on the provided scalar per-environment loss."""

    if per_env_loss.ndim != 0:
        per_env_loss = per_env_loss.mean()
    grad = torch.autograd.grad(per_env_loss * (w ** 2), [w], create_graph=True)[0]
    return torch.sum(grad ** 2)


__all__ = ["HIRMHead", "irm_penalty"]

