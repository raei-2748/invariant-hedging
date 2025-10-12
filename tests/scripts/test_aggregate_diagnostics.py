import json

import pandas as pd
import pytest

from scripts.aggregate_diagnostics import aggregate_run


def test_aggregate_diagnostics_creates_tables(tmp_path):
    run_dir = tmp_path / "demo_run"
    data_dir = run_dir / "seeds" / "seed42" / "regimeA" / "test"
    data_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"probe_id": "p0", "env": "env1", "risk": 0.1},
            {"probe_id": "p0", "env": "env2", "risk": 0.2},
            {"probe_id": "p1", "env": "env1", "risk": 0.2},
            {"probe_id": "p1", "env": "env2", "risk": 0.4},
        ]
    ).to_csv(data_dir / "risk.csv", index=False)

    pd.DataFrame(
        [
            {"probe_id": "p0", "env_i": "env1", "env_j": "env2", "cosine": 0.0},
            {"probe_id": "p1", "env_i": "env1", "env_j": "env2", "cosine": 0.5},
        ]
    ).to_csv(data_dir / "alignment_head.csv", index=False)

    pd.DataFrame(
        [
            {"probe_id": "p0", "dispersion": 0.2},
            {"probe_id": "p1", "dispersion": 0.4},
        ]
    ).to_csv(data_dir / "feature_dispersion.csv", index=False)

    pd.DataFrame(
        [
            {"step": 0, "risk": 0.15},
            {"step": 1, "risk": 0.20},
            {"step": 2, "risk": 0.25},
            {"step": 3, "risk": 0.30},
        ]
    ).to_csv(data_dir / "risk_series.csv", index=False)

    pd.DataFrame(
        [
            {"step": 0, "pnl": 0.1, "a0": 0.0, "a1": 0.0},
            {"step": 1, "pnl": 0.2, "a0": 0.5, "a1": 0.0},
            {"step": 2, "pnl": -0.1, "a0": 0.0, "a1": 0.5},
            {"step": 3, "pnl": 0.0, "a0": 0.0, "a1": 0.2},
        ]
    ).to_csv(data_dir / "pnl.csv", index=False)

    with (data_dir / "cvar95.json").open("w", encoding="utf-8") as handle:
        json.dump({"cvar95": 0.12}, handle)

    tables = aggregate_run(run_dir)

    for name in [
        "diagnostics_summary",
        "invariance_diagnostics",
        "robustness_diagnostics",
        "efficiency_diagnostics",
        "capital_efficiency_frontier",
    ]:
        output_path = run_dir / "tables" / f"{name}.csv"
        assert output_path.exists()
        assert not tables[name].empty

    summary = tables["diagnostics_summary"].iloc[0]
    assert summary["seed"] == 42
    assert summary["regime_name"] == "regimeA"
    assert summary["split"] == "test"
    assert summary["cvar95"] == 0.12
    assert summary["turnover"] == pytest.approx(0.5023689270621825)
    assert summary["mean_pnl"] == pytest.approx(0.05)
    assert summary["ER"] == pytest.approx(0.5)
    assert summary["TR"] == pytest.approx(1.674563090207275)
    assert summary["ISI"] == pytest.approx(0.7729166666666666)
    assert summary["IG"] == pytest.approx(0.15)
    assert summary["WG"] == pytest.approx(0.3)
    assert summary["VR"] == pytest.approx(0.24845199749997662)

    invariance = tables["invariance_diagnostics"]
    aggregate_rows = invariance[invariance["type"] == "aggregate"]
    assert aggregate_rows.iloc[0]["C1"] == pytest.approx(0.99375)
    assert aggregate_rows.iloc[0]["C2"] == pytest.approx(0.625)
    assert aggregate_rows.iloc[0]["C3"] == pytest.approx(0.7)

    frontier = tables["capital_efficiency_frontier"]
    assert frontier.iloc[0]["mean_pnl"] == pytest.approx(0.05)
    assert frontier.iloc[0]["cvar95"] == 0.12
