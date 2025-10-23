from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.reporting.latex import build_table, save_latex_table, write_table_csv


def test_build_table_with_booktabs(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "metric": ["ISI", "ISI"],
            "regime": ["train_main", "crisis_2020"],
            "mean": [0.8, 0.7],
            "ci_half_width": [0.05, 0.04],
        }
    )
    config = {"latex": {"column_format": "lrr", "table_float": "t", "booktabs": True}}
    text = build_table(summary, ["ISI"], ["train_main", "crisis_2020"], config, "Invariance", "tab:invariance")
    assert "\\toprule" in text
    assert "ISI" in text
    tex_path = tmp_path / "invariance.tex"
    save_latex_table(text, tex_path)
    assert tex_path.exists()

    pivot = pd.DataFrame(
        {
            "metric": ["ISI"],
            "mean_train_main": [0.8],
            "ci_half_width_train_main": [0.05],
            "std_train_main": [0.01],
            "n_train_main": [3],
        }
    )
    csv_path = tmp_path / "invariance.csv"
    write_table_csv(pivot, csv_path)
    assert csv_path.exists()
