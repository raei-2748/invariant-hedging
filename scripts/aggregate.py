#!/usr/bin/env python
"""CLI entry-point for the reporting aggregation pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.report.aggregate import run_aggregation


def _load_config(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate diagnostics into publication-grade assets")
    parser.add_argument("--config", type=Path, required=True, help="Path to the aggregation YAML config")
    parser.add_argument("--lite", action="store_true", help="Enable lightweight mode for quick iterations")
    parser.add_argument("--skip-3d", action="store_true", help="Skip I–R–E 3D generation even if enabled in config")
    parser.add_argument("--out", type=Path, default=None, help="Override output directory defined in the config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    result = run_aggregation(config, lite=args.lite, skip_3d=args.skip_3d, output_dir=args.out)
    summary = {
        "output_dir": str(result.outputs_dir),
        "runs": [str(p) for p in result.selected_runs],
        "tables": sorted(str(p) for p in (result.outputs_dir / "tables").glob("*.tex")),
        "figures": sorted(str(p) for p in (result.outputs_dir / "figures").glob("*.pdf")),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
