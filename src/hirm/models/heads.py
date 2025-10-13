"""Representation heads used by IRM penalties."""
from __future__ import annotations

import torch
from torch import nn


class RepresentationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RiskHead(nn.Module):
    """Simple risk estimator head used by HIRM variants."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        layers = []
        if hidden_dim > 0:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)])
        else:
            layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
