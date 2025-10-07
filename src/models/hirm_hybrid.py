"""Hybrid risk head combining invariant and adaptive estimates."""

from __future__ import annotations

import torch
from torch import nn


class HIRMHybrid(nn.Module):
    """Two-head risk estimator with a learnable gating coefficient."""

    def __init__(
        self,
        base_repr: nn.Module,
        risk_head_inv: nn.Module,
        risk_head_adapt: nn.Module,
        *,
        alpha_init: float = 0.0,
        freeze_alpha: bool = False,
    ) -> None:
        super().__init__()
        self.repr = base_repr
        self.h_inv = risk_head_inv
        self.h_adapt = risk_head_adapt
        self.alpha = nn.Parameter(
            torch.tensor(alpha_init, dtype=torch.float32), requires_grad=not freeze_alpha
        )
        self.w_inv = nn.Parameter(torch.ones(1))

    def forward(self, features: torch.Tensor):
        phi = self.repr(features)
        r_inv = self.h_inv(phi)
        r_adapt = self.h_adapt(phi)
        gate = torch.sigmoid(self.alpha)
        r_hat = gate * r_inv + (1 - gate) * r_adapt
        return r_hat, r_inv, r_adapt, gate

    def gate_value(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha.detach())


__all__ = ["HIRMHybrid"]
