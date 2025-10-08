"""Aggregate diagnostic metrics from Phase-2 runs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any, Dict, List

EXPECTED_COLUMNS = [
    "method",
    "seed",
    "cvar95",
    "mean",
    "sortino",
    "turnover",
    "IG",
    "WG",
    "MSI",
    "commit",
]


def _load_metrics(path: pathlib.Path) -> Dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    return {key: data.get(key, "") for key in EXPECTED_COLUMNS}


def _discover_runs(root: pathlib.Path) -> List[pathlib.Path]:
    return sorted(root.glob("**/final_metrics.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Directory containing run outputs")
    parser.add_argument("--out", required=True, help="Path to the aggregated CSV")
    args = parser.parse_args()

    run_root = pathlib.Path(args.runs)
    rows = [_load_metrics(path) for path in _discover_runs(run_root)]

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=EXPECTED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("")


if __name__ == "__main__":
    main()
