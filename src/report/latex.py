"""LaTeX table helpers for report aggregation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import pandas as pd


def escape_latex(text: str) -> str:
    return text.replace("_", "\\_")


def _format_value(mean: float, ci_half_width: float) -> str:
    if math.isnan(mean):
        return "--"
    if math.isnan(ci_half_width) or ci_half_width == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} \\pm {ci_half_width:.3f}"


def _build_table_frame(
    stats_frame: pd.DataFrame,
    metrics: Sequence[str],
    regimes_order: Sequence[str],
) -> pd.DataFrame:
    subset = stats_frame[stats_frame["metric"].isin(metrics)].copy()
    subset = subset.set_index(["metric", "regime"])
    data = {}
    for metric in metrics:
        row = {}
        for regime in regimes_order:
            try:
                record = subset.loc[(metric, regime)]
            except KeyError:
                row[regime] = "--"
                continue
            row[regime] = _format_value(float(record["mean"]), float(record["ci_half_width"]))
        data[metric] = row
    table = pd.DataFrame.from_dict(data, orient="index", columns=list(regimes_order))
    table.index.name = "Metric"
    return table


def render_booktabs_table(
    table: pd.DataFrame,
    group_labels: Mapping[str, Sequence[str]],
    *,
    table_float: str = "t",
    column_format: str = "lrrrr",
    booktabs: bool = True,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    columns = ["Metric"] + list(table.columns)
    lines = [f"\\begin{{table}}[{table_float}]", "\\centering"]
    lines.append(f"\\begin{{tabular}}{{{column_format}}}")
    if booktabs:
        lines.append("\\toprule")
    header = " & ".join(["Metric"] + [escape_latex(str(c)) for c in table.columns]) + " \\\\"  # noqa: E501
    lines.append(header)
    if booktabs:
        lines.append("\\midrule")

    for group_name, metrics in group_labels.items():
        span = len(columns)
        lines.append(f"\\multicolumn{{{span}}}{{l}}{{\\textbf{{{escape_latex(group_name)}}}}}\\\\")
        for metric in metrics:
            if metric not in table.index:
                continue
            row = table.loc[metric]
            values = " & ".join([escape_latex(metric)] + [str(row[col]) for col in table.columns])
            lines.append(f"{values}\\\\")
        if booktabs:
            lines.append("\\midrule")

    if booktabs:
        lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    if caption:
        lines.append(f"\\caption{{{escape_latex(caption)}}}")
    if label:
        lines.append(f"\\label{{{escape_latex(label)}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def write_table(
    stats_frame: pd.DataFrame,
    *,
    metrics: Sequence[str],
    regimes_order: Sequence[str],
    group_name: str,
    output_dir: Path,
    latex_config: Mapping[str, object],
    table_name: str,
    caption: str | None = None,
    label: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_frame = _build_table_frame(stats_frame, metrics, regimes_order)
    table_frame.to_csv(output_dir / f"{table_name}.csv")

    table_float = str(latex_config.get("table_float", "t"))
    column_format = str(latex_config.get("column_format", "lrrrrr"))
    booktabs = bool(latex_config.get("booktabs", True))

    latex_text = render_booktabs_table(
        table_frame,
        group_labels={group_name: metrics},
        table_float=table_float,
        column_format=column_format,
        booktabs=booktabs,
        caption=caption,
        label=label,
    )
    (output_dir / f"{table_name}.tex").write_text(latex_text, encoding="utf-8")


def write_all_tables(
    stats_frame: pd.DataFrame,
    *,
    metrics_config: Mapping[str, Sequence[str]],
    regimes_order: Sequence[str],
    latex_config: Mapping[str, object],
    output_dir: Path,
) -> None:
    """Write the main set of tables as defined in the configuration."""

    output_dir.mkdir(parents=True, exist_ok=True)
    block_titles = {
        "invariance": "Invariance",
        "robustness": "Robustness",
        "efficiency": "Efficiency",
    }

    all_metrics: list[str] = []
    for block in ("invariance", "robustness", "efficiency"):
        all_metrics.extend(metrics_config.get(block, ()))

    if all_metrics:
        table_frame = _build_table_frame(stats_frame, all_metrics, regimes_order)
        latex_text = render_booktabs_table(
            table_frame,
            group_labels={block_titles.get(block, block.title()): tuple(metrics_config.get(block, ())) for block in metrics_config},
            table_float=str(latex_config.get("table_float", "t")),
            column_format=str(latex_config.get("column_format", "lrrrrr")),
            booktabs=bool(latex_config.get("booktabs", True)),
            caption="Main scorecard",
            label="tab:scorecard",
        )
        (output_dir / "main_scorecard.tex").write_text(latex_text, encoding="utf-8")
        table_frame.to_csv(output_dir / "main_scorecard.csv")

    for block, metrics in metrics_config.items():
        if not metrics:
            continue
        write_table(
            stats_frame,
            metrics=metrics,
            regimes_order=regimes_order,
            group_name=block_titles.get(block, block.title()),
            output_dir=output_dir,
            latex_config=latex_config,
            table_name=f"{block}_table",
            caption=f"{block_titles.get(block, block.title())} metrics",
            label=f"tab:{block}",
        )
