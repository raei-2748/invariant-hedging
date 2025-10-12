"""MLP policy network with environment adapters and representation head."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .feature_extractor import FeatureExtractor
from .policy import Policy


class PolicyMLP(Policy):
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
        *,
        head_name: str | None = None,
    ) -> None:
        super().__init__(head_name=head_name or "decision_head")
        self.feature_extractor = FeatureExtractor(
            feature_dim=feature_dim,
            hidden_width=hidden_width,
            hidden_depth=hidden_depth,
            dropout=dropout,
            layer_norm=layer_norm,
            representation_dim=representation_dim,
        )
        # Preserve legacy attribute names for downstream utilities.
        self.backbone = self.feature_extractor.backbone
        self.representation = self.feature_extractor.representation
        adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(representation_dim, adapter_hidden),
                    nn.ReLU(),
                    nn.Linear(adapter_hidden, 1),
                )
                for _ in range(num_envs)
            ]
        )
        self.env_adapters = adapters
        # Provide a consistent head alias for psi-parameter selection.
        self.decision_head = adapters
        self.max_position = max_position

    def forward(
        self,
        features: torch.Tensor,
        env_index: int,
        representation_scale: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        rep = self.feature_extractor(features, detach_for_penalty=self.should_detach_features())
        scale = representation_scale if representation_scale is not None else 1.0
        scaled_rep = rep * scale
        adapter = self.env_adapters[env_index]
        raw_action = adapter(scaled_rep)
        action = torch.tanh(raw_action) * self.max_position
        return {"action": action, "raw_action": raw_action, "representation": rep}
