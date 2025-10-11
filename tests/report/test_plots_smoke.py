from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.report.plots import (
    plot_efficiency_frontier,
    plot_heatmaps,
    plot_qq,
    plot_scorecard,
    plot_seed_distributions,
)


def _make_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["ISI", "ISI", "CVaR_95", "CVaR_95", "ER_mean_pnl", "ER_mean_pnl", "TR_turnover", "TR_turnover"],
            "regime": [
                "train_main",
                "crisis_2020",
                "train_main",
                "crisis_2020",
                "train_main",
                "crisis_2020",
                "train_main",
                "crisis_2020",
            ],
            "mean": [0.8, 0.7, -0.2, -0.3, 0.1, 0.05, 1.2, 1.4],
            "ci_half_width": [0.05, 0.04, 0.02, 0.03, 0.01, 0.02, 0.05, 0.06],
        }
    )


def _make_raw() -> pd.DataFrame:
    records = []
    for seed in range(6):
        for regime in ["train_main", "crisis_2020"]:
            records.extend(
                [
                    {"run_path": "run", "seed": seed, "regime": regime, "metric": "ISI", "value": 0.7 + 0.01 * seed},
                    {"run_path": "run", "seed": seed, "regime": regime, "metric": "CVaR_95", "value": -0.2 - 0.01 * seed},
                    {"run_path": "run", "seed": seed, "regime": regime, "metric": "ER_mean_pnl", "value": 0.1 - 0.005 * seed},
                    {"run_path": "run", "seed": seed, "regime": regime, "metric": "TR_turnover", "value": 1.0 + 0.02 * seed},
                ]
            )
    return pd.DataFrame(records)


def test_plotting_smoke(tmp_path: Path) -> None:
    summary = _make_summary()
    raw = _make_raw()
    config = {
        "regimes_order": ["train_main", "crisis_2020"],
        "metrics": {
            "invariance": ["ISI"],
            "robustness": ["CVaR_95"],
            "efficiency": ["ER_mean_pnl", "TR_turnover"],
        },
        "figures": {"dpi_preview": 100},
        "qq": {"reference": "gaussian", "bins": 20},
    }

    plot_scorecard(summary, config, tmp_path)
    plot_heatmaps(summary, config, tmp_path)
    plot_seed_distributions(raw, ["CVaR_95"], config, tmp_path)
    plot_efficiency_frontier(raw, "ER_mean_pnl", "TR_turnover", config, tmp_path)
    plot_qq(raw, config, tmp_path, "ER_mean_pnl")

    expected = [
        tmp_path / "scorecard.pdf",
        tmp_path / "heatmap_invariance.pdf",
        tmp_path / "distribution_CVaR_95.pdf",
        tmp_path / "efficiency_frontier.pdf",
        tmp_path / "qq_ER_mean_pnl.pdf",
    ]
    for path in expected:
        assert path.exists()
