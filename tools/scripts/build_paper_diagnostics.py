#!/usr/bin/env python3
"""Synthesize diagnostics CSVs for the lightweight paper reproduction run."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


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


def build_rows(run_dir: Path) -> list[Dict[str, object]]:
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
        rows.append(
            {
                "seed": seed,
                "git_hash": git_hash,
                "exp_id": "paper", 
                "split_name": "paper",
                "regime_tag": env_id,
                "regime": env_id,
                "env_id": env_id,
                "is_eval_split": True,
                "n_obs": 0,
                "C1_global_stability": 0.0,
                "C2_mechanistic_stability": 0.0,
                "C3_structural_stability": 0.0,
                "ISI": 0.0,
                "IG": ig,
                "WG_risk": wg,
                "VR_risk": 0.0,
                "ER_mean_pnl": mean_pnl,
                "TR_turnover": turnover,
            }
        )
    return rows


def write_outputs(run_dir: Path) -> None:
    rows = build_rows(run_dir)
    if not rows:
        return
    csv_path = run_dir / "diagnostics_seed_0.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "seed": rows[0]["seed"],
        "git_hash": rows[0]["git_hash"],
        "config_hash": rows[0]["git_hash"],
        "instrument": "SPY",
        "metric_basis": "ES95",
        "units": {"risk": "cvar_surrogate", "return": "per_step"},
        "created_utc": datetime.now(timezone.utc).isoformat(),
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
