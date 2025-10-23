#!/usr/bin/env python3
"""Generate QQ plots for PnL tails from evaluation metrics."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import NormalDist
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from _plot_utils import save_png_with_meta

LOGGER = logging.getLogger("plot_qq_tails")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="One or more run directories containing metrics.jsonl")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=200, help="Figure resolution")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _find_metrics_file(run_dir: Path) -> Path | None:
    direct = run_dir / "metrics.jsonl"
    if direct.exists():
        return direct
    candidates = sorted(run_dir.rglob("metrics.jsonl"))
    return candidates[0] if candidates else None


def _extract_episode_pnls(metrics_path: Path) -> List[float]:
    pnls: List[float] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSON line in %s", metrics_path)
                continue
            for key, value in record.items():
                key_lower = key.lower()
                if "pnl" not in key_lower:
                    continue
                if "mean" in key_lower or "avg" in key_lower:
                    continue
                if isinstance(value, (list, tuple)):
                    pnls.extend(float(v) for v in value if _is_number(v))
                elif isinstance(value, (int, float)) and ("episode" in key_lower or "sample" in key_lower or "dist" in key_lower):
                    pnls.append(float(value))
    return pnls


def _is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _qq_data(values: np.ndarray) -> Dict[str, np.ndarray]:
    qs = np.linspace(1e-3, 0.5, min(200, max(50, values.size)))
    empirical = np.quantile(values, qs, method="linear")
    normal = np.array([NormalDist().inv_cdf(float(q)) for q in qs])
    return {"quantiles": qs, "empirical": empirical, "theoretical": normal}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    series_data = []
    meta_runs = []

    for run in args.runs:
        run_path = Path(run)
        if not run_path.exists():
            LOGGER.warning("Run directory %s not found; skipping", run_path)
            continue
        metrics_path = _find_metrics_file(run_path)
        if metrics_path is None:
            LOGGER.warning("metrics.jsonl not found under %s; skipping", run_path)
            continue
        pnls = _extract_episode_pnls(metrics_path)
        if not pnls:
            LOGGER.warning("No episode-level PnL values found in %s; skipping", run_path)
            continue
        arr = np.asarray(pnls, dtype=float)
        if arr.size < 5:
            LOGGER.warning("Only %d PnL samples for %s; skipping", arr.size, run_path)
            continue
        series_data.append((run_path.name, arr))
        meta_runs.append({
            "run": str(run_path.resolve()),
            "metrics_file": str(metrics_path.resolve()),
            "num_samples": int(arr.size),
        })

    if not series_data:
        LOGGER.error("No usable runs provided; ensure metrics.jsonl contain episode-level PnL data")
        return 1

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, arr in series_data:
        qq = _qq_data(np.sort(arr))
        ax.plot(qq["theoretical"], qq["empirical"], marker="o", markersize=3, linestyle="-", label=label)

    vline = NormalDist().inv_cdf(0.05)
    ax.axvline(vline, color="gray", linestyle=":", linewidth=1.2, label="5% tail")

    ax.set_xlabel("Standard Normal Quantile (lower tail)")
    ax.set_ylabel("Empirical PnL Quantile")
    ax.set_title(args.title or "QQ Plot of PnL Lower Tail")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    out_path = Path(args.out)
    meta = {
        "runs": meta_runs,
        "args": vars(args),
    }
    save_png_with_meta(fig, out_path, meta, dpi=args.dpi)
    LOGGER.info("Saved QQ tails plot to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
