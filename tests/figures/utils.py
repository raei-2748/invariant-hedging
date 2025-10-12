from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_run_directory(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "20240101_test"
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "seed": [1, 1, 2],
            "regime_name": ["crisis_a", "crisis_b", "crisis_a"],
            "split": ["train", "val", "test"],
            "ISI": [0.5, 0.4, 0.3],
            "IG": [0.3, 0.25, 0.2],
            "IG_norm": [1.2, 1.1, 0.9],
            "C1": [0.1, 0.1, 0.1],
            "C2": [0.1, 0.1, 0.1],
            "C3": [0.1, 0.1, 0.1],
        }
    ).to_csv(tables_dir / "invariance_diagnostics.csv", index=False)

    pd.DataFrame(
        {
            "model": ["ERM", "HIRM"],
            "seed": [1, 2],
            "regime_name": ["crisis_a", "crisis_b"],
            "mean_pnl": [0.6, 0.7],
            "cvar95": [0.2, 0.3],
            "ER": [0.5, 0.55],
            "TR": [0.4, 0.5],
        }
    ).to_csv(tables_dir / "capital_efficiency_frontier.csv", index=False)

    pd.DataFrame(
        {
            "model": ["ERM", "HIRM", "GroupDRO"],
            "seed": [1, 2, 1],
            "regime_name": ["crisis_a", "crisis_b", "crisis_a"],
            "split": ["test", "test", "test"],
            "ISI": [0.3, 0.4, 0.35],
            "IG": [0.2, 0.1, 0.15],
            "IG_norm": [1.0, 1.1, 0.9],
            "CVaR95": [0.45, 0.5, 0.55],
            "mean_pnl": [0.65, 0.7, 0.66],
            "TR": [0.6, 0.7, 0.65],
            "ER": [0.5, 0.55, 0.52],
        }
    ).to_csv(tables_dir / "diagnostics_summary.csv", index=False)

    pd.DataFrame(
        {
            "epoch": [0, 0, 1],
            "step": [1, 2, 3],
            "pair": [0, 1, 0],
            "penalty_value": [0.1, 0.1, 0.15],
            "avg_risk": [0.2, 0.2, 0.25],
            "cosine_alignment": [0.9, 0.85, 0.88],
        }
    ).to_csv(tables_dir / "alignment_head.csv", index=False)

    return run_dir

