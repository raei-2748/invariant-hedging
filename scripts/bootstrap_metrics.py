#!/usr/bin/env python3
"""Bootstrap confidence intervals for metrics stored in final_metrics.json files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable
import numpy as np
def _expand_paths(patterns: Iterable[str]) -> list[Path]:
    expanded: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            metrics_path = candidate / "final_metrics.json"
            if metrics_path.exists():
                expanded.append(metrics_path)
            continue
        if candidate.is_file():
            expanded.append(candidate)
            continue
        for match in Path().glob(pattern):
            if match.is_dir():
                metrics_file = match / "final_metrics.json"
                if metrics_file.exists():
                    expanded.append(metrics_file)
            elif match.is_file():
                expanded.append(match)
    return sorted({path.resolve() for path in expanded})
def _load_metric(path: Path, metric: str) -> float:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if metric not in payload:
        raise KeyError(f"Metric '{metric}' missing from {path}")
    return float(payload[metric])
def _bootstrap(values: np.ndarray, samples: int, confidence: float, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        raise ValueError("No values provided for bootstrap")
    estimates = np.empty(samples, dtype=float)
    for i in range(samples):
        indices = rng.integers(0, n, size=n)
        estimates[i] = values[indices].mean()
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    return float(values.mean()), float(np.quantile(estimates, lower_q)), float(np.quantile(estimates, upper_q))
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories, final_metrics.json files, or glob patterns (e.g. 'runs/2025*/')",
    )
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        help="Metric key to analyse (repeat for multiple metrics)",
    )
    parser.add_argument("--samples", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for the interval")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility")
    args = parser.parse_args()
    metric_files = _expand_paths(args.paths)
    if not metric_files:
        raise SystemExit("No final_metrics.json files found for the provided paths")
    header = ["metric", "mean", "ci_lower", "ci_upper", "n"]
    print(",".join(header))
    for metric in args.metric:
        values = np.array([_load_metric(path, metric) for path in metric_files], dtype=float)
        mean, lower, upper = _bootstrap(values, args.samples, args.confidence, args.seed)
        print(f"{metric},{mean:.6f},{lower:.6f},{upper:.6f},{len(values)}")
if __name__ == "__main__":
    main()
