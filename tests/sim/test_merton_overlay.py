"""Tests for the Merton jump overlay utilities."""
from __future__ import annotations

import pandas as pd
import torch

from hirm.data.sim import HestonParams, simulate_heston
from hirm.data.sim.merton import overlay_merton_jumps
from hirm.objectives.cvar import cvar_from_pnl


def _pnl_from_path(df: pd.DataFrame) -> float:
    return float(df["spot"].iloc[-1] - df["spot"].iloc[0])


def test_overlay_is_reproducible() -> None:
    params = HestonParams(s0=100.0, v0=0.04, mu=0.0, kappa=1.4, theta=0.04, sigma=0.3, rho=-0.5, dt_days=1, days=60)
    base = simulate_heston(params, seed=7)
    overlay_a, summary_a = overlay_merton_jumps(base, lam=0.2, mu_j=-0.05, sigma_j=0.1, seed_offset=101)
    overlay_b, summary_b = overlay_merton_jumps(base, lam=0.2, mu_j=-0.05, sigma_j=0.1, seed_offset=101)
    pd.testing.assert_frame_equal(overlay_a, overlay_b)
    assert summary_a == summary_b


def test_left_tail_thickens_with_jumps() -> None:
    params = HestonParams(s0=100.0, v0=0.05, mu=0.0, kappa=1.8, theta=0.04, sigma=0.4, rho=-0.6, dt_days=1, days=90)
    pnl_base = []
    pnl_jump = []
    for seed in range(16):
        path = simulate_heston(params, seed=seed)
        pnl_base.append(_pnl_from_path(path))
        stressed, _ = overlay_merton_jumps(path, lam=0.5, mu_j=-0.08, sigma_j=0.12, seed_offset=seed + 99)
        pnl_jump.append(_pnl_from_path(stressed))
    cvar_base = cvar_from_pnl(torch.tensor(pnl_base, dtype=torch.float32), alpha=0.95)
    cvar_jump = cvar_from_pnl(torch.tensor(pnl_jump, dtype=torch.float32), alpha=0.95)
    assert float(cvar_jump - cvar_base) > 0.0


def test_disabled_overlay_returns_identical_path() -> None:
    params = HestonParams(s0=100.0, v0=0.04, mu=0.0, kappa=1.1, theta=0.04, sigma=0.25, rho=-0.4, dt_days=1, days=45)
    path = simulate_heston(params, seed=11)
    overlay, summary = overlay_merton_jumps(path, lam=0.0, mu_j=-0.08, sigma_j=0.12, seed_offset=123)
    pd.testing.assert_frame_equal(path.assign(jump_size=0.0), overlay)
    assert summary.count == 0
    assert summary.mean == 0.0
    assert summary.std == 0.0
