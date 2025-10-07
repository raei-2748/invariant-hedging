#!/usr/bin/env python3
"""Visualize the capital-efficiency frontier across methods."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("plot_capital_frontier")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _ensure_outdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", required=True, help="Path to scorecard.csv")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument(
        "--notional", type=float, default=1.0, help="Notional scaling (unused placeholder)"
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output figure DPI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _load_scorecard(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _pareto_front(points: list[tuple[str, float, float]]) -> list[str]:
    dominant: list[str] = []
    for name_i, risk_i, pnl_i in points:
        if math.isnan(risk_i) or math.isnan(pnl_i):
            continue
        dominated = False
        for name_j, risk_j, pnl_j in points:
            if name_i == name_j:
                continue
            if math.isnan(risk_j) or math.isnan(pnl_j):
                continue
            better_or_equal_risk = risk_j <= risk_i + 1e-9
            better_or_equal_pnl = pnl_j >= pnl_i - 1e-9
            strictly_better = (risk_j < risk_i - 1e-9) or (pnl_j > pnl_i + 1e-9)
            if better_or_equal_risk and better_or_equal_pnl and strictly_better:
                dominated = True
                break
        if not dominated:
            dominant.append(name_i)
    return dominant


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    df = _load_scorecard(Path(args.scorecard))
    required = {"method", "es95_mean", "meanpnl_mean", "turnover_mean", "n_seeds"}
    if not required.issubset(df.columns):
        raise KeyError("scorecard CSV missing required columns")
    df = df.copy()
    df["abs_es95"] = df["es95_mean"].abs()
    df = df[df["n_seeds"] > 0]
    if df.empty:
        LOGGER.error("Scorecard does not contain any methods with valid seeds")
        return 1

    methods = df["method"].tolist()
    risks = df["abs_es95"].to_numpy(dtype=float)
    pnls = df["meanpnl_mean"].to_numpy(dtype=float)
    turnovers = df["turnover_mean"].to_numpy(dtype=float)

    pareto = _pareto_front(list(zip(methods, risks, pnls)))

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    min_turn = np.nanmin(turnovers)
    max_turn = np.nanmax(turnovers)
    turn_range = max(max_turn - min_turn, 1e-6)

    for idx, method in enumerate(methods):
        color = cmap(idx % cmap.N)
        risk = risks[idx]
        pnl = pnls[idx]
        turn = turnovers[idx]
        if math.isnan(risk) or math.isnan(pnl):
            LOGGER.warning("Skipping method %s due to NaNs", method)
            continue
        size = 120 * (1 + (0 if math.isnan(turn) else (turn - min_turn) / turn_range))
        edgecolor = "black" if method in pareto else color
        facecolor = color if method in pareto else (*color[:3], 0.5)
        ax.scatter(
            risk,
            pnl,
            s=size,
            color=facecolor,
            edgecolor=edgecolor,
            linewidths=1.0,
            alpha=0.85,
            zorder=3,
        )
        ax.text(risk, pnl, f" {method}", ha="left", va="center", fontsize=9)

    ax.set_xlabel("|ES95 Mean|")
    ax.set_ylabel("Mean PnL")
    ax.set_title("Capital-Efficiency Frontier")
    ax.grid(alpha=0.3)
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            label="Pareto-dominant",
            markerfacecolor="none",
            markersize=8,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            label="Non-dominant",
            markerfacecolor="gray",
            alpha=0.5,
            markersize=8,
        ),
    ]
    ax.legend(handles=legend_elements, loc="best", frameon=False)
    fig.tight_layout()

    out_path = Path(args.out)
    _ensure_outdir(out_path)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    LOGGER.info("Saved capital frontier plot to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
