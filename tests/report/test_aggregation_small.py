from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from src.report.aggregate import AggregateResult, aggregate_runs


def _write_diagnostics_table(path: Path, seeds: list[int], regimes: list[str], metrics: dict[str, dict[str, float]]) -> None:
    records: list[dict[str, object]] = []
    for seed in seeds:
        for regime in regimes:
            for metric, values in metrics.items():
                records.append(
                    {
                        "env": regime,
                        "split": "test",
                        "seed": seed,
                        "algo": "erm",
                        "metric": metric,
                        "value": values[regime] + seed * 0.01,
                    }
                )
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)


def test_cross_seed_aggregation(tmp_path: Path) -> None:
    regimes = ["train_main", "crisis_2020"]
    metrics = {
        "ISI": {"train_main": 0.8, "crisis_2020": 0.7},
        "CVaR_95": {"train_main": -0.2, "crisis_2020": -0.3},
        "ER_mean_pnl": {"train_main": 0.1, "crisis_2020": 0.05},
        "TR_turnover": {"train_main": 1.2, "crisis_2020": 1.4},
    }
    table_path = tmp_path / "diagnostics.parquet"
    _write_diagnostics_table(table_path, seeds=[0, 1, 2], regimes=regimes, metrics=metrics)

    config = {
        "report": {
            "diagnostics_table": str(table_path),
            "split": "test",
            "algo": "erm",
            "seeds": 3,
            "regimes_order": regimes,
            "confidence_level": 0.95,
            "metrics": {
                "invariance": ["ISI"],
                "robustness": ["CVaR_95"],
                "efficiency": ["ER_mean_pnl", "TR_turnover"],
            },
            "figures": {"dpi_preview": 120},
            "latex": {"column_format": "lrr", "table_float": "t", "booktabs": True},
            "generate_3d": False,
        }
    }

    result = aggregate_runs(config)
    assert isinstance(result, AggregateResult)
    assert set(result.regimes) == set(regimes)
    assert result.seeds == [0, 1, 2]
    assert result.diagnostics_path == table_path
    assert all(result.summary["n"] == 3)
    mean_isi = result.summary[
        (result.summary["metric"] == "ISI") & (result.summary["regime"] == "train_main")
    ]["mean"].iloc[0]
    assert mean_isi == pytest.approx(0.81, rel=1e-3)
