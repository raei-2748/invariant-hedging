#!/usr/bin/env python3
"""Scatter plot of IG vs. crisis ES95 with regression fit."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("plot_ig_vs_cvar")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _ensure_outdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, help="Path to diagnostics_all.csv")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--dpi", type=int, default=200, help="Output figure DPI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2:
        return math.nan, math.nan
    try:  # pragma: no cover - prefer SciPy when available
        from scipy import stats

        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        xm = x - x_mean
        ym = y - y_mean
        denom = math.sqrt(float(np.sum(xm**2)) * float(np.sum(ym**2)))
        if denom == 0:
            return math.nan, math.nan
        r = float(np.sum(xm * ym) / denom)
        df = x.size - 2
        if df <= 0:
            return r, math.nan
        # two-sided p-value using Student's t-distribution approximation
        t_stat = abs(r) * math.sqrt(df / max(1e-12, 1 - r * r))
        try:
            from statistics import NormalDist

            # Approximate using normal distribution when SciPy is unavailable
            p = 2 * (1 - NormalDist().cdf(t_stat))
        except Exception:
            p = math.nan
        return r, float(p)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    diagnostics_path = Path(args.diagnostics)
    if not diagnostics_path.exists():
        raise FileNotFoundError(diagnostics_path)
    df = pd.read_csv(diagnostics_path)
    if not {"ig", "es95_crisis", "method"}.issubset(df.columns):
        raise KeyError("diagnostics CSV must contain 'ig', 'es95_crisis', and 'method' columns")
    df = df.dropna(subset=["ig", "es95_crisis"])
    if df.empty:
        LOGGER.error("No diagnostic records with finite IG and ES95 values")
        return 1

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    method_colors: dict[str, tuple[float, float, float, float]] = {}

    for idx, method in enumerate(sorted(df["method"].unique())):
        method_colors[method] = cmap(idx % cmap.N)

    for method, group in df.groupby("method"):
        color = method_colors.get(method, (0.2, 0.2, 0.2, 1.0))
        ax.scatter(
            group["ig"].to_numpy(dtype=float),
            group["es95_crisis"].to_numpy(dtype=float),
            label=method,
            color=color,
            edgecolor="black",
            linewidths=0.4,
            alpha=0.8,
        )

    x = df["ig"].to_numpy(dtype=float)
    y = df["es95_crisis"].to_numpy(dtype=float)
    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), num=200)
        ax.plot(
            xs,
            slope * xs + intercept,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Least-squares fit",
        )
        r, p = _pearsonr(x, y)
        caption = f"r = {r:.3f}, p = {p:.3g}" if not math.isnan(r) else "r unavailable"
        ax.text(0.02, 0.95, caption, transform=ax.transAxes, ha="left", va="top", fontsize=11)
    else:
        LOGGER.warning("Not enough points to compute regression")

    ax.set_xlabel("IG (Train ES95 gap)")
    ax.set_ylabel("Crisis ES95")
    ax.set_title("IG vs. Crisis ES95")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    out_path = Path(args.out)
    _ensure_outdir(out_path)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    LOGGER.info("Saved IG vs ES95 scatter to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
