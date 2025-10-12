#!/usr/bin/env python3
"""Synthesize a canonical diagnostics parquet for the paper reproduction run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from src.diagnostics.schema import CANONICAL_COLUMNS, SCHEMA_VERSION, normalize_diagnostics_frame, validate_diagnostics_table


def _load_json(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return {}


def build_rows(run_dir: Path) -> tuple[list[Dict[str, object]], str, int]:
    metrics = _load_json(run_dir / "final_metrics.json")
    metadata = _load_json(run_dir / "metadata.json")
    git_hash = metadata.get("git_commit", "unknown")
    seed = int(metrics.get("train/seed", 0)) if isinstance(metrics.get("train/seed"), (int, float)) else 0
    mean_pnl = float(metrics.get("test/smoke/Mean", 0.0))
    turnover = float(metrics.get("test/smoke/Turnover", 0.0))
    wg = float(metrics.get("diagnostics/WG/ES95", 0.0))
    ig = float(metrics.get("diagnostics/IG/ES95", 0.0))
    rows: list[Dict[str, object]] = []
    for env_id in ("smoke", "__overall__"):
        rows.extend(
            [
                {
                    "env": env_id,
                    "split": "paper",
                    "seed": seed,
                    "algo": "paper",
                    "metric": "IG",
                    "value": float(ig),
                },
                {
                    "env": env_id,
                    "split": "paper",
                    "seed": seed,
                    "algo": "paper",
                    "metric": "WG_risk",
                    "value": float(wg),
                },
                {
                    "env": env_id,
                    "split": "paper",
                    "seed": seed,
                    "algo": "paper",
                    "metric": "ER_mean_pnl",
                    "value": float(mean_pnl),
                },
                {
                    "env": env_id,
                    "split": "paper",
                    "seed": seed,
                    "algo": "paper",
                    "metric": "TR_turnover",
                    "value": float(turnover),
                },
            ]
        )
    return rows, git_hash, seed


def write_outputs(run_dir: Path) -> None:
    rows, git_hash, seed = build_rows(run_dir)
    if not rows:
        return
    frame = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    frame = normalize_diagnostics_frame(frame)
    validate_diagnostics_table(frame)
    parquet_path = run_dir / "diagnostics.parquet"
    frame.to_parquet(parquet_path, index=False)

    manifest = {
        "seed": seed,
        "git_hash": git_hash,
        "config_hash": git_hash,
        "instrument": "SPY",
        "metric_basis": "ES95",
        "units": {"risk": "cvar_surrogate", "return": "per_step"},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "diagnostics_table": parquet_path.name,
        "columns": list(CANONICAL_COLUMNS),
    }
    manifest_path = run_dir / "diagnostics_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Evaluation run directory")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        parser.error(f"Run directory not found: {run_dir}")
    write_outputs(run_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
