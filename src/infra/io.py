"""JSON writers for simulation provenance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


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
