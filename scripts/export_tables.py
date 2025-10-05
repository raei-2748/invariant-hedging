#!/usr/bin/env python3
"""Export scorecard summaries to Markdown and LaTeX tables."""
from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

LOGGER = logging.getLogger("export_tables")

REQUIRED_COLUMNS = [
    "method",
    "n_seeds",
    "es95_mean",
    "es95_ci_low",
    "es95_ci_high",
    "meanpnl_mean",
    "meanpnl_ci_low",
    "meanpnl_ci_high",
    "turnover_mean",
    "turnover_ci_low",
    "turnover_ci_high",
    "d_es95_vs_ERM_pct",
    "d_meanpnl_vs_ERM_pct",
    "d_turnover_vs_ERM_pct",
]

MARKDOWN_HEADERS = [
    "Method",
    "Seeds",
    "ES95 (95% CI)",
    "Mean PnL (95% CI)",
    "Turnover (95% CI)",
    "ΔES95 vs ERM (%)",
    "ΔMeanPnL vs ERM (%)",
    "ΔTurnover vs ERM (%)",
]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    return rows


def _validate_columns(rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Scorecard is empty; cannot export tables")
    missing = [col for col in REQUIRED_COLUMNS if col not in rows[0]]
    if missing:
        raise KeyError(f"Scorecard missing required columns: {missing}")


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _format_ci(mean: object, low: object, high: object) -> str:
    mean_val = _to_float(mean)
    low_val = _to_float(low)
    high_val = _to_float(high)
    if any(math.isnan(v) for v in (mean_val, low_val, high_val)):
        return "NA"
    return f"{mean_val:.4f} [{low_val:.4f}, {high_val:.4f}]"


def _format_pct(value: object) -> str:
    val = _to_float(value)
    if math.isnan(val):
        return "NA"
    return f"{val:.2f}%"


def _markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    table_rows: List[List[str]] = [MARKDOWN_HEADERS, ["---"] * len(MARKDOWN_HEADERS)]
    for row in rows:
        table_rows.append(
            [
                str(row.get("method", "")),
                str(row.get("n_seeds", "")),
                _format_ci(row.get("es95_mean"), row.get("es95_ci_low"), row.get("es95_ci_high")),
                _format_ci(row.get("meanpnl_mean"), row.get("meanpnl_ci_low"), row.get("meanpnl_ci_high")),
                _format_ci(row.get("turnover_mean"), row.get("turnover_ci_low"), row.get("turnover_ci_high")),
                _format_pct(row.get("d_es95_vs_ERM_pct")),
                _format_pct(row.get("d_meanpnl_vs_ERM_pct")),
                _format_pct(row.get("d_turnover_vs_ERM_pct")),
            ]
        )
    return "\n".join("| " + " | ".join(row) + " |" for row in table_rows)


def _latex_table(rows: Sequence[Mapping[str, object]]) -> str:
    header = r"\begin{tabular}{lrrrrrrr}\toprule"
    lines = [header]
    lines.append(
        r"Method & Seeds & ES95 (95\% CI) & Mean PnL (95\% CI) & Turnover (95\% CI) & "
        r"\(\Delta\)ES95 vs ERM (\%) & \(\Delta\)MeanPnL vs ERM (\%) & \(\Delta\)Turnover vs ERM (\%) \\"
    )
    lines.append(r"\midrule")
    for row in rows:
        fields = [
            str(row.get("method", "")),
            str(row.get("n_seeds", "")),
            _format_ci(row.get("es95_mean"), row.get("es95_ci_low"), row.get("es95_ci_high")),
            _format_ci(row.get("meanpnl_mean"), row.get("meanpnl_ci_low"), row.get("meanpnl_ci_high")),
            _format_ci(row.get("turnover_mean"), row.get("turnover_ci_low"), row.get("turnover_ci_high")),
            _format_pct(row.get("d_es95_vs_ERM_pct")),
            _format_pct(row.get("d_meanpnl_vs_ERM_pct")),
            _format_pct(row.get("d_turnover_vs_ERM_pct")),
        ]
        lines.append(" & ".join(fields) + r" \\")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", required=True, help="Input scorecard CSV")
    parser.add_argument("--out_md", required=True, help="Markdown output path")
    parser.add_argument("--out_tex", required=True, help="LaTeX output path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    rows = _load_rows(Path(args.scorecard))
    _validate_columns(rows)
    rows.sort(key=lambda item: item.get("method", ""))

    md_path = Path(args.out_md)
    tex_path = Path(args.out_tex)
    _ensure_parent(md_path)
    _ensure_parent(tex_path)

    markdown = _markdown_table(rows)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(markdown + "\n")
    LOGGER.info("Wrote Markdown table to %s", md_path)

    latex = _latex_table(rows)
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write(latex + "\n")
    LOGGER.info("Wrote LaTeX table to %s", tex_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
