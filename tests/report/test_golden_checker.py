from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.check_golden import compare_directories, compare_tables


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "ERM",
                "split": "mini_crisis",
                "n_seeds": 2,
                "es95_mean": -2.45,
                "es95_ci_low": -2.65,
                "es95_ci_high": -2.25,
                "meanpnl_mean": 0.78,
                "meanpnl_ci_low": 0.60,
                "meanpnl_ci_high": 0.96,
                "turnover_mean": 1.32,
                "turnover_ci_low": 1.20,
                "turnover_ci_high": 1.44,
                "d_es95_vs_ERM_pct": 0.0,
                "d_meanpnl_vs_ERM_pct": 0.0,
                "d_turnover_vs_ERM_pct": 0.0,
                "commit": "abc1234",
                "phase": "mini",
                "config_tag": "paper-lite",
                "timestamp": "2024-01-15T12:00:00+00:00",
            },
            {
                "method": "IRM",
                "split": "mini_crisis",
                "n_seeds": 2,
                "es95_mean": -2.28,
                "es95_ci_low": -2.52,
                "es95_ci_high": -2.04,
                "meanpnl_mean": 0.81,
                "meanpnl_ci_low": 0.66,
                "meanpnl_ci_high": 0.96,
                "turnover_mean": 1.18,
                "turnover_ci_low": 1.05,
                "turnover_ci_high": 1.31,
                "d_es95_vs_ERM_pct": 6.94,
                "d_meanpnl_vs_ERM_pct": 3.85,
                "d_turnover_vs_ERM_pct": -10.61,
                "commit": "abc1234",
                "phase": "mini",
                "config_tag": "paper-lite",
                "timestamp": "2024-01-15T12:00:00+00:00",
            },
        ]
    )


def test_compare_directories_pass(tmp_path: Path) -> None:
    golden_dir = tmp_path / "goldens"
    tables_dir = tmp_path / "tables"
    golden_dir.mkdir()
    tables_dir.mkdir()

    frame = _sample_frame()
    frame.to_csv(golden_dir / "paper-lite.csv", index=False)

    current = frame.copy()
    current.loc[current["method"] == "IRM", "es95_mean"] *= 1.01
    current.loc[current["method"] == "ERM", "timestamp"] = "2024-02-01T00:00:00+00:00"
    current.to_csv(tables_dir / "paper-lite.csv", index=False)

    issues, drifts = compare_directories(golden_dir, tables_dir, ignore_columns={"timestamp", "commit"})
    assert not issues
    assert not drifts


def test_compare_directories_detects_drift(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.csv"
    current_path = tmp_path / "current.csv"

    frame = _sample_frame()
    frame.to_csv(golden_path, index=False)

    current = frame.copy()
    current.loc[current["method"] == "IRM", "meanpnl_mean"] = 1.2
    current.to_csv(current_path, index=False)

    issues, drifts = compare_tables(
        golden_path,
        current_path,
        rel_tol=0.025,
        abs_tol=1e-8,
        ignore_columns={"timestamp", "commit"},
    )
    assert not issues
    assert drifts
    columns = {drift.column for drift in drifts}
    assert "meanpnl_mean" in columns
