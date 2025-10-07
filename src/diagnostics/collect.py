"""Collect diagnostics from run directories into a CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import yaml


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {}


def _read_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        try:
            return yaml.safe_load(handle) or {}
        except yaml.YAMLError:
            return {}


def _resolve(metrics: Mapping[str, object], candidates: Sequence[str]) -> object:
    for key in candidates:
        if key in metrics:
            return metrics[key]
    return ""


def _collect_row(run_dir: Path) -> Dict[str, object]:
    metrics = _read_json(run_dir / "final_metrics.json")
    metadata = _read_json(run_dir / "metadata.json")
    config = _read_yaml(run_dir / "config.yaml")

    method = config.get("model", {}).get("name") if isinstance(config, dict) else None
    if not method and isinstance(config, dict):
        method = config.get("method") or config.get("model_name")

    seed = None
    if isinstance(config, dict):
        train_cfg = config.get("train") or {}
        runtime_cfg = config.get("runtime") or {}
        seed = train_cfg.get("seed") or runtime_cfg.get("seed") or config.get("seed")

    cvar = _resolve(
        metrics,
        [
            "test/crisis/cvar",
            "test/crisis_cvar",
            "test/crisis/CVaR95",
            "test/crisis/CVaR",
            "test/crisis/ES95",
        ],
    )
    mean = _resolve(
        metrics,
        [
            "test/crisis/mean",
            "test/crisis_mean",
            "test/crisis_mean_pnl",
        ],
    )
    sortino = _resolve(
        metrics,
        [
            "test/crisis/sortino",
            "test/crisis_sortino",
        ],
    )
    turnover = _resolve(
        metrics,
        [
            "test/crisis/turnover",
            "test/crisis_turnover",
        ],
    )
    ig = _resolve(
        metrics,
        [
            "diagnostics/IG/cvar",
            "diagnostics/IG/ES95",
            "diagnostics/IG/value",
        ],
    )
    wg = _resolve(
        metrics,
        [
            "diagnostics/WG/cvar",
            "diagnostics/WG/ES95",
            "diagnostics/WG/value",
        ],
    )
    msi = _resolve(
        metrics,
        [
            "diagnostics/MSI/value",
            "diagnostics/MSI/mean",
        ],
    )
    commit = metadata.get("git_commit", "") if isinstance(metadata, dict) else ""

    return {
        "method": method or "",
        "seed": seed if seed is not None else "",
        "cvar95": cvar,
        "mean": mean,
        "sortino": sortino,
        "turnover": turnover,
        "IG": ig,
        "WG": wg,
        "MSI": msi,
        "commit": commit,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="Directory containing run outputs")
    parser.add_argument("--out", required=True, help="Path to the aggregated CSV")
    args = parser.parse_args(argv)

    run_root = Path(args.runs)
    rows: List[Dict[str, object]] = []
    for metrics_path in run_root.glob("**/final_metrics.json"):
        rows.append(_collect_row(metrics_path.parent))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("")


if __name__ == "__main__":  # pragma: no cover
    main()
