from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import (
    build_scorecard,
    ensure_assets_dir,
    load_report_inputs,
    resolve_diagnostic_columns,
    summarise_by_method,
)

LOGGER = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _format_ci(mean: float, low: float, high: float, precision: int = 4) -> str:
    if any(np.isnan([mean, low, high])):
        return "NA"
    return f"{mean:.{precision}f} [{low:.{precision}f}, {high:.{precision}f}]"


def _format_pct(value: float) -> str:
    if np.isnan(value):
        return "NA"
    return f"{value:.2f}%"


def _write_markdown_table(scorecard: pd.DataFrame, out_path: Path) -> None:
    headers = [
        "Method",
        "Seeds",
        "ES90 (95% CI)",
        "ES95 (95% CI)",
        "ES99 (95% CI)",
        "Mean PnL (95% CI)",
        "Turnover (95% CI)",
        "ΔES95 vs ERM (%)",
        "ΔMeanPnL vs ERM (%)",
        "ΔTurnover vs ERM (%)",
    ]
    rows = [headers, ["---"] * len(headers)]
    for _, row in scorecard.iterrows():
        rows.append(
            [
                str(row.get("method", "")),
                str(row.get("n_seeds", "")),
                _format_ci(row.get("es90_mean", np.nan), row.get("es90_ci_low", np.nan), row.get("es90_ci_high", np.nan)),
                _format_ci(row.get("es95_mean", np.nan), row.get("es95_ci_low", np.nan), row.get("es95_ci_high", np.nan)),
                _format_ci(row.get("es99_mean", np.nan), row.get("es99_ci_low", np.nan), row.get("es99_ci_high", np.nan)),
                _format_ci(row.get("meanpnl_mean", np.nan), row.get("meanpnl_ci_low", np.nan), row.get("meanpnl_ci_high", np.nan)),
                _format_ci(row.get("turnover_mean", np.nan), row.get("turnover_ci_low", np.nan), row.get("turnover_ci_high", np.nan)),
                _format_pct(float(row.get("d_es95_vs_ERM_pct", np.nan))),
                _format_pct(float(row.get("d_meanpnl_vs_ERM_pct", np.nan))),
                _format_pct(float(row.get("d_turnover_vs_ERM_pct", np.nan))),
            ]
        )
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write("| " + " | ".join(row) + " |\n")
    LOGGER.info("Wrote Markdown table to %s", out_path)


def _write_latex_table(scorecard: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        (
            r"Method & Seeds & ES90 (95\% CI) & ES95 (95\% CI) & ES99 (95\% CI) & "
            r"Mean PnL (95\% CI) & Turnover (95\% CI) & "
            r"\(\Delta\)ES95 vs ERM (\%) & \(\Delta\)MeanPnL vs ERM (\%) & \(\Delta\)Turnover vs ERM (\%) "
            r"\\"
        ),
        r"\midrule",
    ]
    for _, row in scorecard.iterrows():
        cells = [
            str(row.get("method", "")),
            str(row.get("n_seeds", "")),
            _format_ci(row.get("es90_mean", np.nan), row.get("es90_ci_low", np.nan), row.get("es90_ci_high", np.nan)),
            _format_ci(row.get("es95_mean", np.nan), row.get("es95_ci_low", np.nan), row.get("es95_ci_high", np.nan)),
            _format_ci(row.get("es99_mean", np.nan), row.get("es99_ci_low", np.nan), row.get("es99_ci_high", np.nan)),
            _format_ci(row.get("meanpnl_mean", np.nan), row.get("meanpnl_ci_low", np.nan), row.get("meanpnl_ci_high", np.nan)),
            _format_ci(row.get("turnover_mean", np.nan), row.get("turnover_ci_low", np.nan), row.get("turnover_ci_high", np.nan)),
            _format_pct(float(row.get("d_es95_vs_ERM_pct", np.nan))),
            _format_pct(float(row.get("d_meanpnl_vs_ERM_pct", np.nan))),
            _format_pct(float(row.get("d_turnover_vs_ERM_pct", np.nan))),
        ]
        lines.append(" & ".join(cells) + ' \\')
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    LOGGER.info("Wrote LaTeX table to %s", out_path)


def _diagnostic_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics is None or diagnostics.empty:
        return pd.DataFrame()
    diag_cols = resolve_diagnostic_columns(diagnostics)
    if not diag_cols:
        return pd.DataFrame()
    mapping: Mapping[str, str] = {key: value for key, value in diag_cols.items() if value in diagnostics.columns}
    if not mapping:
        return pd.DataFrame()
    renamed = diagnostics.rename(columns={v: k for k, v in mapping.items()})
    columns = {key: key for key in mapping.keys()}
    summary = summarise_by_method(renamed, columns)
    return summary


def _write_diagnostic_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    components = [col for col in ("ig", "wg", "msi") if f"{col}_mean" in summary.columns]
    if not components:
        return
    headers = ["Method", "Seeds"] + [f"{col.upper()} (95% CI)" for col in components]
    rows = [headers, ["---"] * len(headers)]
    for _, row in summary.iterrows():
        values = [
            str(row.get("method", "")),
            str(row.get("n_seeds", "")),
        ]
        for col in components:
            values.append(
                _format_ci(
                    row.get(f"{col}_mean", np.nan),
                    row.get(f"{col}_ci_low", np.nan),
                    row.get(f"{col}_ci_high", np.nan),
                    precision=3,
                )
            )
        rows.append(values)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write("| " + " | ".join(row) + " |\n")
    LOGGER.info("Wrote diagnostic Markdown table to %s", out_path)


def _write_diagnostic_csv(summary: pd.DataFrame, out_path: Path) -> None:
    summary.to_csv(out_path, index=False)
    LOGGER.info("Wrote diagnostic summary to %s", out_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase-2 report tables")
    parser.add_argument("--per-seed", action="append", type=Path, help="Explicit per-seed CSV inputs")
    parser.add_argument("--scorecard", type=Path, default=None, help="Optional existing scorecard.csv")
    parser.add_argument("--assets-dir", type=Path, default=None, help="Directory for table outputs")
    parser.add_argument("--search-root", action="append", type=Path, help="Additional directories to scan for CSVs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    assets_dir = ensure_assets_dir(args.assets_dir)
    search_roots = tuple(args.search_root) if args.search_root else None
    inputs = load_report_inputs(
        per_seed_paths=args.per_seed or None,
        scorecard_path=args.scorecard,
        search_roots=search_roots,
    )

    scorecard = build_scorecard(inputs.per_seed)
    if scorecard.empty and inputs.scorecard is not None and not inputs.scorecard.empty:
        scorecard = inputs.scorecard
    if scorecard.empty:
        LOGGER.warning("No per-seed metrics available to build scorecard")
    else:
        scorecard_path = assets_dir / "scorecard.csv"
        scorecard.to_csv(scorecard_path, index=False)
        LOGGER.info("Wrote scorecard CSV to %s", scorecard_path)
        _write_markdown_table(scorecard, assets_dir / "table_crisis.md")
        _write_latex_table(scorecard, assets_dir / "table_crisis.tex")

    diagnostics_summary = _diagnostic_summary(inputs.diagnostics)
    if diagnostics_summary.empty:
        LOGGER.info("No diagnostics summary produced")
    else:
        _write_diagnostic_csv(diagnostics_summary, assets_dir / "diagnostics_summary.csv")
        _write_diagnostic_markdown(diagnostics_summary, assets_dir / "table_diagnostics.md")

    generated = (not scorecard.empty) or (not diagnostics_summary.empty)
    if not generated:
        LOGGER.warning("No tables generated; verify per-seed CSV inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
