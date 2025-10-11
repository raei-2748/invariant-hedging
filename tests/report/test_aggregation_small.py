from __future__ import annotations

import json
import pandas as pd

from src.report.aggregate import run_aggregation


def _get_stat(stats_frame: pd.DataFrame, metric: str, regime: str) -> pd.Series:
    subset = stats_frame[(stats_frame["metric"] == metric) & (stats_frame["regime"] == regime)]
    assert not subset.empty, f"Missing statistics for {metric} / {regime}"
    return subset.iloc[0]


def test_aggregate_pipeline_smoke(sample_report_config):
    result = run_aggregation(sample_report_config, lite=True, skip_3d=True)
    stats = result.stats_frame

    isi_train = _get_stat(stats, "ISI", "train_main")
    assert round(float(isi_train["mean"]), 3) == 1.1
    assert int(isi_train["n"]) == 3

    cvar_crisis = _get_stat(stats, "CVaR_95", "crisis_2020")
    assert cvar_crisis["mean"] > 0
    assert cvar_crisis["ci_high"] >= cvar_crisis["ci_low"]

    tables_dir = result.outputs_dir / "tables"
    figures_dir = result.outputs_dir / "figures"
    manifests_dir = result.outputs_dir / "manifests"

    assert (tables_dir / "main_scorecard.tex").exists()
    assert (tables_dir / "robustness_table.csv").exists()
    assert (figures_dir / "heatmap_invariance.pdf").exists()
    assert (figures_dir / "efficiency_frontier.pdf").exists()

    manifest_path = manifests_dir / "aggregate_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "git_hash" in manifest
    assert manifest["regimes_order"] == sample_report_config["report"]["regimes_order"]
