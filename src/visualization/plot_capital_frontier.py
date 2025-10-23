#!/usr/bin/env python3
"""Plot mean PnL against |CVaR-95| with turnover-encoded markers."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plot_utils import ensure_cols, filter_frame, load_csv, save_png_with_meta

LOGGER = logging.getLogger("plot_capital_frontier")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", required=True, help="Path to scoreboard or scorecard CSV")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--reg", default="crisis", help="Regime filter (scoreboard mode only; default: crisis)")
    parser.add_argument("--models", nargs="*", help="Optional list of models to include")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _compute_sizes(turnover: pd.Series) -> np.ndarray:
    values = turnover.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full_like(values, 120.0)
    vals = values[finite]
    min_val = float(vals.min())
    max_val = float(vals.max())
    if max_val - min_val < 1e-12:
        sizes = np.full_like(values, 180.0)
    else:
        scaled = (values - min_val) / (max_val - min_val)
        sizes = 80.0 + 260.0 * scaled
    sizes[~finite] = 80.0
    return sizes


def _prepare_points(
    frame: pd.DataFrame,
    *,
    reg: str | None,
    models: Sequence[str] | None,
) -> tuple[pd.DataFrame, str]:
    """Normalise heterogeneous score inputs to a common plotting schema."""

    scoreboard_cols = {"model", "seed", "reg", "cvar95", "mean_pnl"}
    scorecard_cols = {"method", "es95_mean", "meanpnl_mean"}

    if scoreboard_cols.issubset(frame.columns):
        ensure_cols(frame, scoreboard_cols)
        ensure_cols(frame, {"turnover"}, soft=True)
        filtered = filter_frame(frame, reg=reg, models=models)
        filtered = filtered.copy()
        if filtered.empty:
            return filtered, "scoreboard"
        if "turnover" not in filtered:
            filtered["turnover"] = np.nan
        filtered["abs_cvar95"] = filtered["cvar95"].abs()
        filtered["label"] = [f"{row['model']} (seed {row['seed']})" for _, row in filtered.iterrows()]
        return filtered[["model", "seed", "reg", "abs_cvar95", "mean_pnl", "turnover", "label"]], "scoreboard"

    if scorecard_cols.issubset(frame.columns):
        ensure_cols(frame, scorecard_cols)
        ensure_cols(frame, {"turnover_mean"}, soft=True)
        working = frame.copy()
        if models:
            working = working[working["method"].isin(models)]
        if reg:
            LOGGER.warning("Ignoring reg=%s filter for aggregated scorecard input", reg)
        if working.empty:
            return working, "scorecard"
        working["model"] = working["method"]
        working["seed"] = np.nan
        working["reg"] = None
        working["abs_cvar95"] = working["es95_mean"].abs()
        working["mean_pnl"] = working["meanpnl_mean"]
        working["turnover"] = working.get("turnover_mean", np.nan)
        working["label"] = working["method"]
        return working[["model", "seed", "reg", "abs_cvar95", "mean_pnl", "turnover", "label"]], "scorecard"

    raise KeyError(
        "Input CSV does not contain the expected columns for either scoreboard "
        "(`model`, `seed`, `reg`, `cvar95`, `mean_pnl`) or scorecard "
        "(`method`, `es95_mean`, `meanpnl_mean`) inputs."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    df = load_csv(args.input)
    prepared, schema = _prepare_points(df, reg=args.reg, models=args.models)
    if prepared.empty:
        LOGGER.error("No rows remain after applying filters (schema=%s, reg=%s, models=%s)", schema, args.reg, args.models)
        return 1

    prepared = prepared.dropna(subset=["abs_cvar95", "mean_pnl"])
    if prepared.empty:
        LOGGER.error("Filtered data has no finite CVaR-95 / mean_pnl values (schema=%s)", schema)
        return 1

    sizes = _compute_sizes(prepared["turnover"])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        prepared["abs_cvar95"],
        prepared["mean_pnl"],
        s=sizes,
        c="#1f77b4",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.6,
    )

    sorted_eff = prepared.sort_values(["abs_cvar95", "mean_pnl"], ascending=[True, False]).head(3)
    for _, row in sorted_eff.iterrows():
        label = row["label"]
        ax.annotate(
            label,
            xy=(row["abs_cvar95"], row["mean_pnl"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("|CVaR-95| (per notional)")
    ax.set_ylabel("Mean PnL")
    ax.set_title(args.title or "Capital Frontier: Mean PnL vs Risk")
    ax.grid(alpha=0.3)

    # Marker size legend proxy.
    finite_turn = prepared["turnover"].dropna()
    if not finite_turn.empty:
        avg_turn = float(finite_turn.mean())
        legend_txt = f"Marker size ∝ turnover\nAvg turnover: {avg_turn:.2f}"
        ax.text(0.98, 0.02, legend_txt, transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6))

    fig.tight_layout()

    out_path = Path(args.out)
    meta = {
        "input": str(Path(args.input).resolve()),
        "schema": schema,
        "filters": {"reg": args.reg, "models": args.models},
        "points": prepared[["model", "seed", "reg", "abs_cvar95", "mean_pnl", "turnover"]].to_dict(orient="records"),
        "args": vars(args),
    }
    save_png_with_meta(fig, out_path, meta, dpi=args.dpi)
    plt.close(fig)
    LOGGER.info("Saved capital frontier plot to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
