from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from hirm.report.aggregate import AggregateResult, aggregate_runs

def _make_seed_run(base: Path, stamp: str, seed: int, regimes: list[str], metrics: dict[str, dict[str, float]]) -> None:
    run_dir = base / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for regime in regimes:
        for metric, values in metrics.items():
            records.append({"regime": regime, "metric": metric, "value": values[regime]})
    df = pd.DataFrame(records)
    df.to_csv(run_dir / f"diagnostics_seed_{seed}.csv", index=False)
    with open(run_dir / "final_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({metric: {regime: values[regime] for regime in regimes} for metric, values in metrics.items()}, handle)
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump({"timestamp": seed, "model_family": "ERM"}, handle)
    with open(run_dir / "diagnostics_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"seed": seed, "regimes": regimes}, handle)


def test_cross_seed_aggregation(tmp_path: Path) -> None:
    regimes = ["train_main", "crisis_2020"]
    metrics = {
        "ISI": {"train_main": 0.8, "crisis_2020": 0.7},
        "CVaR_95": {"train_main": -0.2, "crisis_2020": -0.3},
        "ER_mean_pnl": {"train_main": 0.1, "crisis_2020": 0.05},
        "TR_turnover": {"train_main": 1.2, "crisis_2020": 1.4},
    }
    for seed in range(3):
        perturbed = {metric: {reg: value + seed * 0.01 for reg, value in values.items()} for metric, values in metrics.items()}
        _make_seed_run(tmp_path, f"run_{seed}", seed, regimes, perturbed)

    config = {
        "report": {
            "seeds": 3,
            "seed_dirs": [str(tmp_path / "*")],
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
    assert all(result.summary["n"] == 3)
    mean_isi = result.summary[(result.summary["metric"] == "ISI") & (result.summary["regime"] == "train_main")]["mean"].iloc[0]
    assert mean_isi == pytest.approx(0.81, rel=1e-3)