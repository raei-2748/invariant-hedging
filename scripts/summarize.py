#!/usr/bin/env python3
"""Scan run directories for final metrics and emit a CSV summary."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def _load_yaml(path: Path) -> dict:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def collect_rows(root: Path, pattern: str, steps_filter: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(root.glob(pattern)):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "final_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as fh:
                metrics = json.load(fh)
        except json.JSONDecodeError:
            continue

        config = _load_yaml(run_dir / "config.yaml")
        train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
        seed = train_cfg.get("seed")
        steps = train_cfg.get("steps")

        if steps_filter is not None and steps != steps_filter:
            continue

        row: dict[str, object] = {
            "run": run_dir.name,
            "seed": seed,
            "steps": steps,
        }
        for key, value in metrics.items():
            row[key.replace("/", "_")] = value
        rows.append(row)
    return rows


def fieldnames_from_rows(rows: Sequence[dict[str, object]]) -> list[str]:
    ordered: list[str] = ["run", "seed", "steps"]
    for row in rows:
        for key in row:
            if key in ordered:
                continue
            ordered.append(key)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="runs", help="Root directory containing Hydra run folders"
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern (relative to root) selecting runs, e.g. '20250927_*'",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Only include runs whose train.steps matches this value",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = collect_rows(root, args.pattern, args.steps)
    if not rows:
        return

    fieldnames = fieldnames_from_rows(rows)
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
