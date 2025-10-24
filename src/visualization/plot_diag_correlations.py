#!/usr/bin/env python3
"""Plot diagnostic correlations (IG, MSI) against CVaR-95."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from ._plot_utils import (
    ensure_cols,
    filter_frame,
    load_csv,
    pearson_corr,
    save_png_with_meta,
    spearman_corr,
)

LOGGER = logging.getLogger("plot_diag_correlations")


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


def _build_meta(args: argparse.Namespace, diag: str, stats: Dict[str, float]) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "input": str(Path(args.input).resolve()),
        "filters": {"reg": args.reg, "models": args.models},
        "diagnostic": diag,
        "stats": stats,
        "args": vars(args),
    }
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    df = load_csv(args.input)
    ensure_cols(df, ["model", "reg", "cvar95"])
    filtered = filter_frame(df, reg=args.reg, models=args.models)
    if filtered.empty:
        LOGGER.error("No rows remain after applying filters (reg=%s, models=%s)", args.reg, args.models)
        return 1

    diagnostics = ["IG", "MSI"]
    produced = 0
    for diag in diagnostics:
        if diag not in filtered.columns:
            LOGGER.warning("Column %s missing; skipping plot", diag)
            continue
        subset = filtered[[diag, "cvar95"]].dropna()
        if subset.empty:
            LOGGER.warning("Column %s present but no finite values; skipping", diag)
            continue
        x = subset[diag]
        y = subset["cvar95"]
        try:
            pearson = pearson_corr(x, y)
            spearman = spearman_corr(x, y)
        except ValueError:
            LOGGER.warning("Insufficient data to compute correlations for %s", diag)
            continue

        stats = {
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "n": int(subset.shape[0]),
        }
        print(f"{diag} vs CVaR-95 | Pearson r={pearson:.4f}, Spearman rho={spearman:.4f}, n={stats['n']}")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, color="#d62728", alpha=0.8)
        if subset.shape[0] >= 2:
            coeffs = np.polyfit(x, y, deg=1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, coeffs[0] * xs + coeffs[1], color="#1f77b4", linestyle="--", linewidth=1.5)
        ax.set_xlabel(f"{diag}")
        ax.set_ylabel("CVaR-95 (per notional; lower is better)")
        title_root = args.title or "Diagnostics vs CVaR-95"
        ax.set_title(f"{title_root} — {diag}")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        base = Path(args.out)
        out_path = base.with_name(f"{base.stem}_{diag.lower()}{base.suffix}")
        meta = _build_meta(args, diag, stats)
        save_png_with_meta(fig, out_path, meta, dpi=args.dpi)
        LOGGER.info("Saved %s correlation plot to %s", diag, out_path)
        produced += 1

    if produced == 0:
        LOGGER.error("No diagnostic plots were produced; ensure IG/MSI columns exist")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
