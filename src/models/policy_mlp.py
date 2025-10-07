"""MLP policy network with environment adapters and representation head."""

from __future__ import annotations

import torch
from torch import nn

from .heads import RepresentationHead


class PolicyMLP(nn.Module):
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
        self.backbone = nn.Sequential(*layers)
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

    def forward(
        self,
        features: torch.Tensor,
        env_index: int,
        representation_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h = self.backbone(features)
        rep = self.representation(h)
        scale = representation_scale if representation_scale is not None else 1.0
        scaled_rep = rep * scale
        adapter = self.env_adapters[env_index]
        raw_action = adapter(scaled_rep)
        action = torch.tanh(raw_action) * self.max_position
        return {"action": action, "raw_action": raw_action, "representation": rep}
