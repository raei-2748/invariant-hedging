#!/usr/bin/env python3
"""Plot hyperparameter sweeps (λ for IRM-Head, β for V-REx) against CVaR-95."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plot_utils import compute_ci, ensure_cols, filter_frame, load_csv, save_png_with_meta

LOGGER = logging.getLogger("plot_sweeps")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", required=True, help="Path to scoreboard CSV")
    parser.add_argument("--out", required=True, help="Base output PNG path")
    parser.add_argument("--reg", default="crisis", help="Regime filter (default: crisis)")
    parser.add_argument("--models", nargs="*", help="Optional list of models to include")
    parser.add_argument("--title", default=None, help="Optional plot title prefix")
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _summaries(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
    records = []
    for value, group in df.groupby(param_col):
        try:
            stats = compute_ci(group["cvar95"])
        except ValueError:
            continue
        record: Dict[str, object] = {
            "param": value,
            "mean": stats.mean,
            "lower": stats.lower,
            "upper": stats.upper,
            "count": stats.count,
        }
        if "best_flag" in group.columns and group["best_flag"].any():
            record["best"] = True
        else:
            record["best"] = False
        records.append(record)
    return pd.DataFrame.from_records(records)


def _plot_curve(summary: pd.DataFrame, *, ax: plt.Axes, label: str, title: str | None) -> Dict[str, object]:
    summary = summary.sort_values("param")
    params = summary["param"].to_numpy()
    means = summary["mean"].to_numpy()
    lower_err = means - summary["lower"].to_numpy()
    upper_err = summary["upper"].to_numpy() - means
    ax.errorbar(params, means, yerr=[lower_err, upper_err], fmt="-o", capsize=4)
    if summary["best"].any():
        best_rows = summary[summary["best"]]
        ax.scatter(best_rows["param"], best_rows["mean"], color="#d62728", zorder=5, label="Selected")
        ax.legend(loc="best", frameon=False)
    ax.set_xlabel(label)
    ax.set_ylabel("CVaR-95 (per notional; lower is better)")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return {
        "points": summary.to_dict(orient="records"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    df = load_csv(args.input)
    ensure_cols(df, ["model", "reg", "cvar95"])
    filtered = filter_frame(df, reg=args.reg, models=args.models)
    if filtered.empty:
        LOGGER.error("No rows remain after applying filters (reg=%s, models=%s)", args.reg, args.models)
        return 1

    produced = 0
    base = Path(args.out)

    if "lambda" in filtered.columns:
        irm = filtered[filtered["lambda"].notna()]
        if not irm.empty:
            irm_summary = _summaries(irm, "lambda")
            if not irm_summary.empty:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                extra = _plot_curve(
                    irm_summary,
                    ax=ax,
                    label="λ (IRM-Head)",
                    title=(args.title + " — IRM-Head") if args.title else "IRM-Head λ Sweep",
                )
                fig.tight_layout()
                out_path = base.with_name(f"{base.stem}_irm_lambda{base.suffix}")
                meta = {
                    "input": str(Path(args.input).resolve()),
                    "filters": {"reg": args.reg, "models": args.models},
                    "sweep": "lambda",
                    "data": extra["points"],
                    "args": vars(args),
                }
                save_png_with_meta(fig, out_path, meta, dpi=args.dpi)
                LOGGER.info("Saved IRM λ sweep to %s", out_path)
                produced += 1

    if "beta" in filtered.columns:
        vrex = filtered[filtered["beta"].notna()]
        if not vrex.empty:
            vrex_summary = _summaries(vrex, "beta")
            if not vrex_summary.empty:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                extra = _plot_curve(
                    vrex_summary,
                    ax=ax,
                    label="β (V-REx)",
                    title=(args.title + " — V-REx") if args.title else "V-REx β Sweep",
                )
                fig.tight_layout()
                out_path = base.with_name(f"{base.stem}_vrex_beta{base.suffix}")
                meta = {
                    "input": str(Path(args.input).resolve()),
                    "filters": {"reg": args.reg, "models": args.models},
                    "sweep": "beta",
                    "data": extra["points"],
                    "args": vars(args),
                }
                save_png_with_meta(fig, out_path, meta, dpi=args.dpi)
                LOGGER.info("Saved V-REx β sweep to %s", out_path)
                produced += 1

    if produced == 0:
        LOGGER.error("No sweep plots were produced; ensure lambda/beta columns exist with data")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
