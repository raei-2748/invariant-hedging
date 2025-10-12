"""Liquidity stress cost model utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class LiquidityStressConfig:
    base_spread_bps: float
    vol_slope_bps: float
    size_slope_bps: float
    slippage_coeff: float
    eps: float = 1e-12

    def to_dict(self) -> Dict[str, float]:
        return {
            "base_spread_bps": self.base_spread_bps,
            "vol_slope_bps": self.vol_slope_bps,
            "size_slope_bps": self.size_slope_bps,
            "slippage_coeff": self.slippage_coeff,
        }


def _ensure_array(values) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64)


def spread_bps(variance, trade_size, config: LiquidityStressConfig) -> np.ndarray:
    variance = np.maximum(_ensure_array(variance), 0.0)
    trade_size = _ensure_array(trade_size)
    vol_term = config.vol_slope_bps * np.sqrt(variance)
    size_term = config.size_slope_bps * np.abs(trade_size) / (1.0 + config.eps)
    raw_bps = config.base_spread_bps + vol_term + size_term
    return np.maximum(raw_bps, 0.0)


def liquidity_costs(
    variance,
    trade_size,
    price_change,
    notional,
    config: LiquidityStressConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    variance = _ensure_array(variance)
    trade_size = _ensure_array(trade_size)
    price_change = _ensure_array(price_change)
    bps = spread_bps(variance, trade_size, config)
    notional_array = np.abs(notional * trade_size)
    spread_cost = bps * notional_array / 10_000.0
    slippage = config.slippage_coeff * np.square(price_change * trade_size)
    total_cost = spread_cost + slippage
    total_cost = np.maximum(total_cost, 0.0)
    summary = {
        "mean_spread_bps": float(np.mean(bps) if bps.size else 0.0),
        "mean_slippage": float(np.mean(slippage) if slippage.size else 0.0),
        "turnover": float(np.sum(np.abs(trade_size))),
    }
    return total_cost, summary
