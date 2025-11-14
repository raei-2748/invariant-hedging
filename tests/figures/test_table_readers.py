from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from invariant_hedging.reporting.tables import (
    maybe_filter_regimes,
    maybe_filter_seeds,
    read_alignment_head,
    read_capital_efficiency_frontier,
    read_diagnostics_summary,
    read_invariance_diagnostics,
)


def _write_csv(path: Path, data: dict[str, list]) -> None:
    frame = pd.DataFrame(data)
    frame.to_csv(path, index=False)


def test_read_invariance_diagnostics_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "invariance_diagnostics.csv"
    _write_csv(
        csv_path,
        {
            "seed": [1],
            "regime_name": ["train"],
            "split": ["test"],
            "ISI": [0.5],
            "IG": [0.2],
            "IG_norm": [1.0],
            "C1": [0.1],
            "C2": [0.1],
            "C3": [0.1],
        },
    )

    frame = read_invariance_diagnostics(csv_path)
    assert not frame.empty
    assert list(frame.columns)[:3] == ["seed", "regime_name", "split"]


def test_read_invariance_missing_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "invariance_diagnostics.csv"
    _write_csv(
        csv_path,
        {
            "seed": [1],
            "regime_name": ["train"],
            "split": ["test"],
            "ISI": [0.5],
            "IG": [0.2],
            "IG_norm": [1.0],
            "C1": [0.1],
            "C2": [0.1],
            # Missing C3
        },
    )

    with pytest.raises(ValueError):
        read_invariance_diagnostics(csv_path)


def test_read_capital_efficiency_frontier(tmp_path: Path) -> None:
    csv_path = tmp_path / "capital_efficiency_frontier.csv"
    _write_csv(
        csv_path,
        {
            "model": ["ERM"],
            "seed": [1],
            "regime_name": ["train"],
            "mean_pnl": [0.3],
            "cvar95": [0.2],
            "ER": [0.25],
            "TR": [0.5],
        },
    )

    frame = read_capital_efficiency_frontier(csv_path)
    assert pytest.approx(frame.loc[0, "ER"], rel=1e-6) == 0.25


def test_capital_efficiency_alias_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "capital_efficiency_frontier.csv"
    _write_csv(
        csv_path,
        {
            "method": ["GroupDRO"],
            "seed": [3],
            "regime": ["stress"],
            "meanpnl_crisis": [0.12],
            "es95_crisis": [0.45],
            "expected_return": [0.18],
            "turnover_crisis": [0.32],
        },
    )

    frame = read_capital_efficiency_frontier(csv_path)
    assert list(frame.columns)[:4] == ["model", "seed", "regime_name", "mean_pnl"]


def test_read_diagnostics_summary_filters(tmp_path: Path) -> None:
    csv_path = tmp_path / "diagnostics_summary.csv"
    _write_csv(
        csv_path,
        {
            "model": ["ERM", "HIRM"],
            "seed": [1, 2],
            "regime_name": ["crisis_a", "crisis_b"],
            "split": ["test", "test"],
            "ISI": [0.3, 0.4],
            "IG": [0.2, 0.1],
            "IG_norm": [1.0, 1.2],
            "CVaR95": [0.5, 0.6],
            "mean_pnl": [0.7, 0.8],
            "TR": [0.9, 1.0],
            "ER": [0.4, 0.6],
        },
    )

    frame = read_diagnostics_summary(csv_path)
    filtered = maybe_filter_seeds(frame, [2])
    assert filtered["seed"].unique().tolist() == [2]

    filtered_regime = maybe_filter_regimes(frame, ["crisis_b"])
    assert filtered_regime["regime_name"].unique().tolist() == ["crisis_b"]


def test_diagnostics_summary_alias_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "diagnostics_summary.csv"
    _write_csv(
        csv_path,
        {
            "method": ["ERM"],
            "seed": [5],
            "regime": ["crisis"],
            "split": ["test"],
            "isi": [0.4],
            "ig": [0.12],
            "ig_norm": [1.5],
            "es95_crisis": [0.8],
            "meanpnl_crisis": [0.05],
            "turnover_crisis": [0.2],
            "expected_return": [0.11],
        },
    )

    frame = read_diagnostics_summary(csv_path)
    assert set(["model", "CVaR95", "mean_pnl", "TR", "ER"]).issubset(frame.columns)


def test_read_alignment_head(tmp_path: Path) -> None:
    csv_path = tmp_path / "alignment_head.csv"
    _write_csv(
        csv_path,
        {
            "epoch": [0, 0],
            "step": [1, 2],
            "pair": [0, 1],
            "penalty_value": [0.1, 0.1],
            "avg_risk": [0.2, 0.2],
            "cosine_alignment": [0.9, 0.85],
        },
    )

    frame = read_alignment_head(csv_path)
    assert list(frame.columns)[-2:] == ["avg_risk", "cosine_alignment"]


def test_alignment_head_alias_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "alignment_head.csv"
    _write_csv(
        csv_path,
        {
            "step": [1, 2],
            "pair_index": [0, 0],
            "penalty": [0.1, 0.1],
            "alignment": [0.7, 0.65],
        },
    )

    frame = read_alignment_head(csv_path)
    assert list(frame.columns) == ["step", "pair", "penalty_value", "cosine_alignment"]

