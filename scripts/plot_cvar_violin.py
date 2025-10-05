#!/usr/bin/env python3
"""Create a violin plot showing crisis ES95 dispersion by method."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("plot_cvar_violin")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _t_critical(df: int, confidence: float = 0.95) -> float:
    if df <= 0:
        return 0.0
    alpha = 1.0 - confidence
    try:  # pragma: no cover - prefer SciPy when available
        from scipy import stats

        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        from statistics import NormalDist

        return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _confidence_interval(values: Sequence[float]) -> tuple[float, float, float]:
    cleaned = np.asarray([v for v in values if not math.isnan(v)])
    if cleaned.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(cleaned.mean())
    if cleaned.size == 1:
        return mean, mean, mean
    std = float(cleaned.std(ddof=1))
    if math.isnan(std):
        return mean, math.nan, math.nan
    margin = _t_critical(cleaned.size - 1) * std / math.sqrt(cleaned.size)
    return mean, mean - margin, mean + margin


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def _ensure_outdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", required=True, help="Path to aggregated scorecard.csv")
    parser.add_argument("--diagnostics", required=True, help="Path to diagnostics_all.csv")
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--dpi", type=int, default=200, help="Output figure DPI")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _method_order(scorecard: pd.DataFrame, diagnostics: pd.DataFrame) -> List[str]:
    ordered = []
    if "method" in scorecard:
        ordered.extend(list(dict.fromkeys(scorecard["method"].tolist())))
    diag_methods = list(dict.fromkeys(diagnostics.get("method", pd.Series(dtype=str)).tolist()))
    for method in diag_methods:
        if method not in ordered:
            ordered.append(method)
    return ordered


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    scorecard_df = _load_csv(Path(args.scorecard))
    diagnostics_df = _load_csv(Path(args.diagnostics))
    if "es95_crisis" not in diagnostics_df.columns:
        raise KeyError("diagnostics CSV must contain 'es95_crisis' column")
    diagnostics_df = diagnostics_df.dropna(subset=["es95_crisis"])
    method_order = _method_order(scorecard_df, diagnostics_df)
    if not method_order:
        LOGGER.error("No methods available to plot")
        return 1

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    positions = np.arange(len(method_order))

    for idx, method in enumerate(method_order):
        subset = diagnostics_df[diagnostics_df["method"] == method]["es95_crisis"].dropna()
        values = subset.to_numpy(dtype=float)
        if values.size == 0:
            LOGGER.warning("Skipping method %s with no crisis ES95 data", method)
            continue
        color = cmap(idx % cmap.N)
        parts = ax.violinplot(
            values,
            positions=[positions[idx]],
            widths=0.8,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.35)
        jitter = (np.random.rand(values.size) - 0.5) * 0.12
        ax.scatter(
            np.full(values.size, positions[idx]) + jitter,
            values,
            color=color,
            edgecolor="black",
            linewidths=0.4,
            alpha=0.8,
            zorder=3,
        )
        mean, low, high = _confidence_interval(values)
        if not math.isnan(mean) and not math.isnan(low) and not math.isnan(high):
            ax.errorbar(
                positions[idx],
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=1.2,
                capsize=4,
                markersize=5,
                zorder=4,
            )
            ax.text(
                positions[idx],
                mean,
                f" {mean:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                color="black",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(method_order, rotation=20, ha="right")
    ax.set_ylabel("Crisis ES95")
    ax.set_xlabel("Method")
    ax.set_title("Crisis ES95 Dispersion by Method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.out)
    _ensure_outdir(out_path)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    LOGGER.info("Saved violin plot to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
