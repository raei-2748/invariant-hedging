#!/usr/bin/env python3
"""Plot Crisis CVaR-95 per model with 95% confidence intervals."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from ._plot_utils import compute_ci, ensure_cols, filter_frame, load_csv, save_png_with_meta

LOGGER = logging.getLogger("plot_cvar_by_method")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", required=True, help="Path to scoreboard CSV")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--reg", default="crisis", help="Regime filter (default: crisis)")
    parser.add_argument("--models", nargs="*", help="Optional list of models to include")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    df = load_csv(args.input)
    ensure_cols(df, ["model", "seed", "reg", "cvar95"])
    filtered = filter_frame(df, reg=args.reg, models=args.models)
    if filtered.empty:
        LOGGER.error("No rows remain after applying filters (reg=%s, models=%s)", args.reg, args.models)
        return 1

    groups = []
    for model, group in filtered.groupby("model"):
        try:
            stats = compute_ci(group["cvar95"])
        except ValueError:
            LOGGER.warning("Skipping model %s due to insufficient data", model)
            continue
        groups.append((model, stats))

    if not groups:
        LOGGER.error("No models with valid CVaR-95 data to plot")
        return 1

    groups.sort(key=lambda item: item[0])
    labels = [g[0] for g in groups]
    means = [g[1].mean for g in groups]
    lower_err = [g[1].mean - g[1].lower for g in groups]
    upper_err = [g[1].upper - g[1].mean for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = range(len(labels))
    ax.errorbar(
        x=list(x_positions),
        y=means,
        yerr=[lower_err, upper_err],
        fmt="o",
        capsize=6,
        markersize=6,
        color="#1f77b4",
        ecolor="#1f77b4",
    )

    for idx, mean in enumerate(means):
        ax.annotate(f"{mean:.3f}", xy=(idx, mean), xytext=(0, 8), textcoords="offset points", ha="center")

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("CVaR-95 (per notional; lower is better)")
    ax.set_xlabel("Model")
    ax.set_title(args.title or "Crisis CVaR-95 by Method")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.out)
    meta = {
        "input": str(Path(args.input).resolve()),
        "filters": {"reg": args.reg, "models": args.models},
        "records": [
            {
                "model": label,
                "mean_cvar95": stat.mean,
                "lower95": stat.lower,
                "upper95": stat.upper,
                "samples": stat.count,
            }
            for label, stat in groups
        ],
        "args": vars(args),
    }
    save_png_with_meta(fig, out_path, meta, dpi=args.dpi)
    LOGGER.info("Saved plot to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
