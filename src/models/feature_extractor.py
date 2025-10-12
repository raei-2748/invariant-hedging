"""Backbone network used by hedging policies."""
from __future__ import annotations

from torch import nn

from .heads import RepresentationHead


class FeatureExtractor(nn.Module):
    """Shared backbone feeding environment-specific heads."""

    def __init__(
        self,
        feature_dim: int,
        hidden_width: int,
        hidden_depth: int,
        dropout: float,
        layer_norm: bool,
        representation_dim: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
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
        representation_input = hidden_width if hidden_depth > 0 else feature_dim
        self.representation = RepresentationHead(
            representation_input, hidden_width, representation_dim
        )

    def forward(self, features, *, detach_for_penalty: bool = False):
        hidden = self.backbone(features)
        representation = self.representation(hidden)
        if detach_for_penalty:
            representation = representation.detach()
        return representation


__all__ = ["FeatureExtractor"]
