"""Hybrid policy with invariant and adaptive risk heads."""

from __future__ import annotations

import torch
from torch import nn

from .heads import RepresentationHead, RiskHead


class TwoHeadPolicy(nn.Module):
    """Policy with shared encoder and two risk heads for HIRM-Hybrid."""

    def __init__(
        self,
        feature_dim: int,
        num_envs: int,
        hidden_width: int,
        hidden_depth: int,
        dropout: float,
        layer_norm: bool,
        representation_dim: int,
        adapter_hidden: int,
        max_position: float,
        risk_hidden: int,
        alpha_init: float = 0.0,
        freeze_alpha: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        input_dim = feature_dim
        for _ in range(hidden_depth):
            layers.append(nn.Linear(input_dim, hidden_width))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_width))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_width
        self.encoder = nn.Sequential(*layers)
        self.representation = RepresentationHead(hidden_width, hidden_width, representation_dim)
        self.env_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(representation_dim, adapter_hidden),
                    nn.ReLU(),
                    nn.Linear(adapter_hidden, 1),
                )
                for _ in range(num_envs)
            ]
        )
        self.max_position = max_position
        self.inv_head = RiskHead(representation_dim, risk_hidden)
        self.adapt_head = RiskHead(representation_dim, risk_hidden)
        alpha = torch.tensor(alpha_init, dtype=torch.float32)
        self.alpha = nn.Parameter(alpha, requires_grad=not freeze_alpha)

    def forward(
        self,
        features: torch.Tensor,
        env_index: int,
        representation_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.encoder(features)
        rep = self.representation(h)
        scale = representation_scale if representation_scale is not None else 1.0
        scaled_rep = rep * scale
        adapter = self.env_adapters[env_index]
        raw_action = adapter(scaled_rep)
        action = torch.tanh(raw_action) * self.max_position
        return {
            "action": action,
            "raw_action": raw_action,
            "representation": rep,
        }

    def invariant_risk(self, representation: torch.Tensor) -> torch.Tensor:
        return self.inv_head(representation)

    def adaptive_risk(self, representation: torch.Tensor) -> torch.Tensor:
        return self.adapt_head(representation)

    def mixed_risk(self, representation: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha)
        inv = self.invariant_risk(representation)
        adapt = self.adaptive_risk(representation)
        return gate * inv + (1 - gate) * adapt

    def gate_value(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha.detach())


__all__ = ["TwoHeadPolicy"]
