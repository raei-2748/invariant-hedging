"""IO helpers for simulation provenance and diagnostics exports."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _ensure_path(path) -> Path:
    return Path(path)


def write_sim_params_json(path, params: Mapping) -> None:
    dest = _ensure_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(dict(params), handle, indent=2, sort_keys=True)


def write_stress_summary_json(path, summary: Mapping) -> None:
    dest = _ensure_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(dict(summary), handle, indent=2, sort_keys=True)


def write_alignment_csv(path: Path | str, rows: Sequence[Mapping[str, object]]) -> None:
    """Append alignment diagnostics to a CSV file."""

    if not rows:
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    first_row = rows[0]
    fieldnames: Iterable[str] = list(first_row.keys())
    needs_header = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
