from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.report.ire3d import build_ire_coordinates, write_ire_assets


def _make_raw() -> pd.DataFrame:
    records = []
    regimes = ["train_main", "crisis_2020"]
    for seed in range(6):
        for regime in regimes:
            base = seed if regime == "train_main" else seed + 0.5
            records.extend(
                [
                    {"run_path": "run", "seed": seed, "regime": regime, "metric": "IG", "value": 0.3 + 0.02 * base},
                    {"run_path": "run", "seed": seed, "regime": regime, "metric": "CVaR_95", "value": -0.1 - 0.03 * base},
                    {"run_path": "run", "seed": seed, "regime": regime, "metric": "ER_mean_pnl", "value": 0.2 - 0.01 * base},
                    {"run_path": "run", "seed": seed, "regime": regime, "metric": "TR_turnover", "value": 1.0 + 0.02 * base},
                ]
            )
    return pd.DataFrame(records)


def _config() -> dict:
    return {
        "report": {
            "regimes_order": ["train_main", "crisis_2020"],
            "metrics": {"efficiency": ["ER_mean_pnl", "TR_turnover"]},
            "ire3d": {
                "winsor_pct": [5, 95],
                "axis_I": "IG",
                "axis_R_source": "CVaR_95",
                "E_alpha_mode": "sd_equalize",
                "projections": ["top", "front", "side"],
            },
        }
    }


def test_ire3d_coordinates_and_assets(tmp_path: Path) -> None:
    raw = _make_raw()
    config = _config()
    result = build_ire_coordinates(raw, config)
    assert ((result.points[["I_star", "R_star", "E_star"]] >= 0) & (result.points[["I_star", "R_star", "E_star"]] <= 1)).all().all()

    pytest.importorskip("plotly")
    write_ire_assets(result.points, config, tmp_path)
    assert (tmp_path / "interactive" / "ire_3d.html").exists()
    for projection in ["top", "front", "side"]:
        assert (tmp_path / "figures" / f"ire_3d_{projection}.pdf").exists()
