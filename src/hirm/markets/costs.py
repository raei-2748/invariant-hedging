"""Execution cost models used across synthetic and real datasets."""
from __future__ import annotations

from typing import Dict

import torch


def execution_cost(trade: torch.Tensor, spot: torch.Tensor, config: Dict[str, float]) -> torch.Tensor:
    """Linear + quadratic execution cost expressed in cash units."""
    linear_bps = float(config.get("linear_bps", 0.0)) * 1e-4
    quadratic = float(config.get("quadratic", 0.0))
    slippage_multiplier = float(config.get("slippage_multiplier", 1.0))
    notional_trade = torch.abs(trade) * spot
    linear_cost = notional_trade * linear_bps
    quadratic_cost = (trade ** 2) * quadratic
    return slippage_multiplier * (linear_cost + quadratic_cost)


def apply_transaction_costs(positions: torch.Tensor, spot: torch.Tensor, config: Dict[str, float]) -> torch.Tensor:
    trades = torch.diff(torch.cat([torch.zeros_like(positions[..., :1]), positions], dim=-1), dim=-1)
    costs = execution_cost(trades, spot[..., 1:], config)
    return costs
