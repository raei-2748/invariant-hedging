"""Lightweight reporting utilities for CI smoke tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class RunSummary:
    """Snapshot of a single training run."""

    run_path: Path
    final_metrics: dict
    latest_event: dict

    def to_dict(self) -> dict:
        return {
            "run_path": str(self.run_path),
            "final_metrics": self.final_metrics,
            "latest_event": self.latest_event,
        }


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _load_latest_event(metrics_path: Path) -> dict:
    latest: dict | None = None
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    latest = payload
    except OSError:
        return {}
    return latest or {}


def _candidate_run_dirs(root: Path) -> Iterable[Path]:
    if (root / "final_metrics.json").exists() or (root / "metrics.jsonl").exists():
        yield root
        return
    for child in sorted(root.glob("*/")):
        if not child.is_dir():
            continue
        if (child / "final_metrics.json").exists() or (child / "metrics.jsonl").exists():
            yield child


def discover_runs(run_specs: Sequence[str]) -> List[Path]:
    runs: List[Path] = []
    for spec in run_specs:
        base = Path(spec).expanduser()
        if not base.exists():
            continue
        for candidate in _candidate_run_dirs(base):
            runs.append(candidate)
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in runs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def build_run_summary(run_dir: Path) -> RunSummary:
    final_metrics = _load_json(run_dir / "final_metrics.json")
    latest_event = _load_latest_event(run_dir / "metrics.jsonl")
    return RunSummary(run_path=run_dir, final_metrics=final_metrics, latest_event=latest_event)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories to summarise")
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Ignored flag maintained for compatibility; prevents plotting in CI.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the generated JSON summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = discover_runs(args.runs)
    if not run_dirs:
        raise SystemExit("No training runs discovered for reporting")

    summaries = [build_run_summary(run_dir) for run_dir in run_dirs]
    payload = {
        "runs": [summary.to_dict() for summary in summaries],
        "count": len(summaries),
    }

    output_path = Path(args.output) if args.output else Path(args.runs[0]) / "ci_report_lite.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote CI summary for {len(summaries)} run(s) to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
