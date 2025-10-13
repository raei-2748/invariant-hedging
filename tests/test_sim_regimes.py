"""Regression tests for the deterministic Heston synthetic pipeline."""
from __future__ import annotations

import torch

from hirm.data.sim.heston import (
    REGIME_ORDER,
    SPLIT_OFFSETS,
    HestonRegimeSimulator,
)


def _expected_schedule(seed: int, split: str, total: int) -> list[str]:
    import numpy as np

    rng = np.random.default_rng((seed + SPLIT_OFFSETS.get(split, 0)) & 0xFFFFFFFF)
    order = list(REGIME_ORDER)
    schedule: list[str] = []
    while len(schedule) < total:
        rng.shuffle(order)
        schedule.extend(order)
    return schedule[:total]


def test_regime_schedule_is_deterministic() -> None:
    simulator = HestonRegimeSimulator(base_seed=123, spot0=100.0, rate=0.01)
    schedule = simulator.regime_schedule("train", 8)
    assert schedule == _expected_schedule(123, "train", 8)
    # Repeat to ensure determinism on subsequent calls.
    assert simulator.regime_schedule("train", 8) == schedule


def test_feature_toggles_adjust_outputs() -> None:
    cost_meta = {"linear_bps": 5.0, "quadratic": 0.0, "slippage_multiplier": 1.0, "notional": 1.0}
    base = HestonRegimeSimulator(base_seed=7, spot0=100.0, rate=0.01)
    jumps = HestonRegimeSimulator(base_seed=7, spot0=100.0, rate=0.01, use_jumps=True)
    liquid = HestonRegimeSimulator(base_seed=7, spot0=100.0, rate=0.01, use_liquidity_spread=True)

    base_batch = base.generate_batch(
        split="train", regime="high", num_episodes=4, steps=16, maturity_days=16, cost_meta=cost_meta
    )
    jump_batch = jumps.generate_batch(
        split="train", regime="high", num_episodes=4, steps=16, maturity_days=16, cost_meta=cost_meta
    )
    liquidity_batch = liquid.generate_batch(
        split="train", regime="high", num_episodes=4, steps=16, maturity_days=16, cost_meta=cost_meta
    )

    assert not torch.allclose(base_batch.spot, jump_batch.spot)
    assert base_batch.meta["liquidity_spread_bps"] == 0.0
    assert liquidity_batch.meta["liquidity_spread_bps"] > 0.0
