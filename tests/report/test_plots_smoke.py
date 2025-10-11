from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.report import plots


def _make_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = pd.DataFrame(
        [
            {"metric": "ISI", "regime": "train_main", "mean": 1.0, "ci_half_width": 0.1},
            {"metric": "ISI", "regime": "crisis_2020", "mean": 1.2, "ci_half_width": 0.1},
            {"metric": "CVaR_95", "regime": "train_main", "mean": 0.3, "ci_half_width": 0.05},
            {"metric": "CVaR_95", "regime": "crisis_2020", "mean": 0.5, "ci_half_width": 0.07},
            {"metric": "ER_mean_pnl", "regime": "train_main", "mean": 0.8, "ci_half_width": 0.04},
            {"metric": "ER_mean_pnl", "regime": "crisis_2020", "mean": 0.6, "ci_half_width": 0.05},
            {"metric": "TR_turnover", "regime": "train_main", "mean": 0.2, "ci_half_width": 0.03},
            {"metric": "TR_turnover", "regime": "crisis_2020", "mean": 0.3, "ci_half_width": 0.02},
        ]
    )
    rows = []
    for seed in range(3):
        rows.extend(
            [
                {"seed": seed, "regime": "train_main", "metric": "ISI", "value": 1.0 + 0.1 * seed},
                {"seed": seed, "regime": "crisis_2020", "metric": "ISI", "value": 1.1 + 0.05 * seed},
                {"seed": seed, "regime": "train_main", "metric": "CVaR_95", "value": 0.3 + 0.1 * seed},
                {"seed": seed, "regime": "crisis_2020", "metric": "CVaR_95", "value": 0.5 + 0.1 * seed},
                {"seed": seed, "regime": "train_main", "metric": "ER_mean_pnl", "value": 0.7 + 0.05 * seed},
                {"seed": seed, "regime": "crisis_2020", "metric": "ER_mean_pnl", "value": 0.6 + 0.02 * seed},
                {"seed": seed, "regime": "train_main", "metric": "TR_turnover", "value": 0.2 + 0.03 * seed},
                {"seed": seed, "regime": "crisis_2020", "metric": "TR_turnover", "value": 0.25 + 0.04 * seed},
            ]
        )
    seed_frame = pd.DataFrame(rows)
    return seed_frame, stats


def test_plot_helpers(tmp_path: Path) -> None:
    seed_frame, stats = _make_frames()
    regimes = ["train_main", "crisis_2020"]

    plots.plot_heatmap(stats, ["ISI"], regimes, title="Test", output_path=tmp_path / "heatmap")
    assert (tmp_path / "heatmap.pdf").exists()
    assert (tmp_path / "heatmap.png").exists()

    plots.plot_seed_distribution(seed_frame, "ISI", regimes, output_path=tmp_path / "seed")
    assert (tmp_path / "seed.pdf").exists()

    plots.plot_efficiency_frontier(seed_frame, regimes, output_path=tmp_path / "frontier")
    assert (tmp_path / "frontier.pdf").exists()

    plots.plot_qq(seed_frame, "train_main", output_path=tmp_path / "qq")
    assert (tmp_path / "qq.pdf").exists()

    plots.generate_all_plots(
        seed_frame,
        stats,
        metrics_config={"invariance": ["ISI"], "robustness": ["CVaR_95"], "efficiency": ["ER_mean_pnl", "TR_turnover"]},
        regimes_order=regimes,
        figure_config={"dpi_preview": 80, "qq_reference": "gaussian"},
        output_dir=tmp_path / "bundle",
        lite=True,
    )
    assert (tmp_path / "bundle" / "heatmap_invariance.pdf").exists()
