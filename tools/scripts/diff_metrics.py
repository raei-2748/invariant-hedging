#!/usr/bin/env python3
"""Numerically compare two metric logs for reproducibility checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Tuple


def _load_metrics(path: Path) -> Dict[Tuple[int | None, str], float]:
    data: Dict[Tuple[int | None, str], float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            step = record.pop("step", None)
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    data[(step, key)] = float(value)
    return data


def _compare(first: Dict[Tuple[int | None, str], float], second: Dict[Tuple[int | None, str], float]) -> Tuple[int, float, float]:
    keys = sorted(set(first) & set(second))
    if not keys:
        raise ValueError("No overlapping metrics between the provided files.")
    deltas = [abs(first[key] - second[key]) for key in keys]
    return len(keys), max(deltas), mean(deltas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs=2, type=Path, help="Two metrics.jsonl files to compare")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Mean absolute difference tolerance")
    args = parser.parse_args()

    first = _load_metrics(args.paths[0])
    second = _load_metrics(args.paths[1])
    count, max_delta, mean_delta = _compare(first, second)
    print(f"Compared {count} overlapping metrics.")
    print(f"max|Δ|={max_delta:.3e}, mean|Δ|={mean_delta:.3e}")
    if mean_delta > args.tolerance:
        import sys

        print(
            f"Mean absolute difference {mean_delta:.3e} exceeds tolerance {args.tolerance:.3e}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":

    raise SystemExit(main())
