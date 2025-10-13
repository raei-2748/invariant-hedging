"""Regression tests for the SPY/E-mini real data loader."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hirm.data.real.spy_emini import SpyEminiDataModule, WINDOW_DEFINITIONS


def test_each_window_produces_expected_range(tmp_path: Path) -> None:
    dates = pd.date_range("2008-01-01", "2022-12-31", freq="7D")
    df = pd.DataFrame(
        {
            "date": dates,
            "spot": 100.0 + 0.1 * pd.Series(range(len(dates))),
            "mid_price": 5.0 + 0.05 * pd.Series(range(len(dates))),
            "implied_vol": 0.2 + 0.001 * pd.Series(range(len(dates))),
        }
    )
    data_path = tmp_path / "spy_full.csv"
    df.to_csv(data_path, index=False)

    module = SpyEminiDataModule(
        {
            "spy_path": str(data_path),
            "mode": "full",
            "rate": 0.01,
            "include_gfc": True,
            "base_linear_bps": 5.0,
            "base_quadratic": 0.0,
            "base_slippage_multiplier": 1.0,
        }
    )

    env_order = module.env_order
    batches = module.prepare("train", env_order)

    for name in env_order:
        expected = WINDOW_DEFINITIONS[name]
        batch = batches[name]
        assert batch.meta["start_date"] == expected["start"]
        assert batch.meta["end_date"] == expected["end"]
