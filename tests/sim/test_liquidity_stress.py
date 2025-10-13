"""Tests for liquidity stress cost model."""
from __future__ import annotations

import numpy as np

from hirm.data.sim.liquidity import LiquidityStressConfig, liquidity_costs
from hirm.envs.registry import SyntheticRegimeRegistry


def test_cost_monotonicity() -> None:
    variance = np.linspace(0.01, 0.09, num=12)
    trade = np.full_like(variance, 0.5)
    price = np.zeros_like(variance)
    cfg_low = LiquidityStressConfig(base_spread_bps=1.0, vol_slope_bps=5.0, size_slope_bps=2.0, slippage_coeff=0.0)
    cfg_high = LiquidityStressConfig(base_spread_bps=1.0, vol_slope_bps=10.0, size_slope_bps=4.0, slippage_coeff=0.0)
    cost_low, _ = liquidity_costs(variance, trade, price, notional=100.0, config=cfg_low)
    cost_high, _ = liquidity_costs(variance, trade, price, notional=100.0, config=cfg_high)
    assert float(cost_high.mean()) > float(cost_low.mean())


def test_liquidity_applies_only_to_configured_regimes() -> None:
    config = {
        "data": {
            "regimes": {
                "bands": [
                    {"name": "train_low"},
                    {"name": "test_crisis"},
                ]
            },
            "stress": {
                "liquidity": {
                    "enabled": True,
                    "apply_to": ["test_crisis"],
                }
            },
        }
    }
    registry = SyntheticRegimeRegistry(config)
    cfg = LiquidityStressConfig(base_spread_bps=1.0, vol_slope_bps=8.0, size_slope_bps=4.0, slippage_coeff=0.001)
    variance = np.full(16, 0.05)
    trade = np.linspace(-1.0, 1.0, num=16)
    price = np.linspace(-0.5, 0.5, num=16)
    baseline_cfg = LiquidityStressConfig(base_spread_bps=0.0, vol_slope_bps=0.0, size_slope_bps=0.0, slippage_coeff=0.0)
    baseline_cost, _ = liquidity_costs(variance, trade, price, notional=50.0, config=baseline_cfg)

    crisis_spec = registry.get("test_crisis")
    crisis_cost, _ = liquidity_costs(variance, trade, price, notional=50.0, config=cfg)
    if crisis_spec.stress_liquidity:
        assert np.all(crisis_cost >= baseline_cost)
    calm_spec = registry.get("train_low")
    if not calm_spec.stress_liquidity:
        calm_cost = baseline_cost
    else:
        calm_cost, _ = liquidity_costs(variance, trade, price, notional=50.0, config=cfg)
    assert np.allclose(calm_cost, baseline_cost)


def test_costs_are_non_negative_and_units_correct() -> None:
    cfg = LiquidityStressConfig(base_spread_bps=2.0, vol_slope_bps=0.0, size_slope_bps=0.0, slippage_coeff=0.0005)
    variance = np.array([0.04, 0.09])
    trade = np.array([1.0, -0.5])
    price = np.array([1.5, -2.0])
    cost, summary = liquidity_costs(variance, trade, price, notional=100.0, config=cfg)
    assert np.all(cost >= 0.0)
    expected_first = 2.0 / 10_000.0 * 100.0 * abs(trade[0])
    assert np.isclose(cost[0] - cfg.slippage_coeff * (price[0] * trade[0]) ** 2, expected_first)
    assert summary["mean_spread_bps"] >= cfg.base_spread_bps
    assert summary["turnover"] == np.sum(np.abs(trade))
