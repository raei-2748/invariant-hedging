"""LaTeX table helpers for the reporting pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Sequence

import pandas as pd

BACKSLASH = chr(92)
ROW_BREAK = BACKSLASH * 2

LATEX_SPECIALS = {
    "_": BACKSLASH + "_",
    "%": BACKSLASH + "%",
}

def escape_latex(text: str) -> str:
    result = str(text)
    for src, dst in LATEX_SPECIALS.items():
        result = result.replace(src, dst)
    return result

def format_mean_ci(mean: float, ci: float) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.3f} {BACKSLASH}pm {ci:.3f}"

def build_table(
    summary: pd.DataFrame,
    metrics: Sequence[str],
    regimes: Sequence[str],
    config: Mapping[str, Mapping[str, str]],
    caption: str,
    label: str,
) -> str:
    latex_cfg = config.get("latex", {})
    column_format = latex_cfg.get("column_format", "lrrrrr")
    table_float = latex_cfg.get("table_float", "t")
    booktabs = latex_cfg.get("booktabs", True)

    header = ["Metric"] + [escape_latex(regime) for regime in regimes]
    lines: List[str] = []
    lines.append(BACKSLASH + "begin{table}[" + table_float + "]")
    lines.append(BACKSLASH + "centering")
    lines.append(BACKSLASH + "begin{tabular}{" + column_format + "}")
    if booktabs:
        lines.append(BACKSLASH + "toprule")
    lines.append(" " + " & ".join(header) + " " + ROW_BREAK)
    if booktabs:
        lines.append(BACKSLASH + "midrule")

    for metric in metrics:
        metric_rows = summary[(summary["metric"] == metric) & (summary["regime"].isin(regimes))]
        if metric_rows.empty:
            continue
        values = []
        for regime in regimes:
            row = metric_rows[metric_rows["regime"] == regime]
            if row.empty:
                values.append("--")
            else:
                mean = float(row["mean"].iloc[0])
                ci = float(row["ci_half_width"].iloc[0])
                values.append(format_mean_ci(mean, ci))
        line = escape_latex(metric) + " & " + " & ".join(values) + " " + ROW_BREAK
        lines.append(line)
    if booktabs:
        lines.append(BACKSLASH + "bottomrule")
    lines.append(BACKSLASH + "end{tabular}")
    lines.append(BACKSLASH + "caption{" + escape_latex(caption) + "}")
    lines.append(BACKSLASH + "label{" + escape_latex(label) + "}")
    lines.append(BACKSLASH + "end{table}")
    return "\n".join(lines)

def save_latex_table(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)

def write_table_csv(pivot: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(path, index=False)

__all__ = [
    "escape_latex",
    "format_mean_ci",
    "build_table",
    "save_latex_table",
    "write_table_csv",
]
