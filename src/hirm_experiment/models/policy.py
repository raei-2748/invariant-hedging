from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


@dataclass
class HedgingPolicyConfig:
    input_dim: int
    hidden_dims: List[int]
    dropout: float = 0.0
    activation: str = "relu"
    output_dim: int = 1
    bounded_output: bool = True
    output_scale: float = 1.0


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class HedgingPolicy(nn.Module):
    def __init__(self, config: HedgingPolicyConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = config.input_dim
        act = _activation(config.activation)
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act.__class__())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.output_dim))
        self.network = nn.Sequential(*layers)
        self.bounded_output = config.bounded_output
        self.output_scale = config.output_scale

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.network(features)
        if self.bounded_output:
            out = torch.tanh(out) * self.output_scale
        return out
