#!/usr/bin/env python3
"""Validate golden metric baselines stored in the repository."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping


def _load_manifest(manifest_path: Path) -> Mapping[str, object]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Golden manifest must be a JSON object")
    return payload


def _load_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and "metrics" in payload:
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise TypeError(f"Unexpected 'metrics' structure in {path}")
    elif isinstance(payload, Mapping):
        metrics = payload
    else:
        raise TypeError(f"Unsupported JSON payload in {path}")

    flattened: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, Mapping) and "value" in value:
            value = value["value"]
        if isinstance(value, (int, float)):
            flattened[key] = float(value)
    return flattened


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="tools/golden_metrics.json",
        help="Path to the golden metrics manifest (relative to the repository root).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not manifest_path.exists():
        print(f"Golden manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = _load_manifest(manifest_path)
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        print("Manifest 'entries' must be a list", file=sys.stderr)
        return 1

    total_checks = 0
    failures: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            failures.append("Manifest entry is not an object")
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            failures.append("Manifest entry missing 'path'")
            continue
        target = Path(raw_path)
        if not target.is_absolute():
            target = repo_root / target
        if not target.exists():
            failures.append(f"Golden file missing: {target}")
            continue
        try:
            actual_metrics = _load_metrics(target)
        except Exception as exc:  # pragma: no cover - defensive
            failures.append(f"Failed to load metrics from {target}: {exc}")
            continue

        expected = entry.get("metrics", {})
        if not isinstance(expected, Mapping):
            failures.append(f"Expected metrics must be an object for {target}")
            continue
        rtol = float(entry.get("rtol", 1e-6))
        atol = float(entry.get("atol", 1e-6))

        for metric, expected_value in expected.items():
            total_checks += 1
            if metric not in actual_metrics:
                failures.append(f"Metric '{metric}' missing in {target}")
                continue
            observed = actual_metrics[metric]
            if not isinstance(expected_value, (int, float)):
                failures.append(
                    f"Expected value for '{metric}' must be numeric in manifest"
                )
                continue
            if not math.isclose(
                observed, float(expected_value), rel_tol=rtol, abs_tol=atol
            ):
                failures.append(
                    f"Metric '{metric}' mismatch in {target}: expected {expected_value}, observed {observed}"
                )

    if failures:
        for message in failures:
            print(message, file=sys.stderr)
        return 1

    print(f"Validated {total_checks} golden metrics across {len(entries)} files.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
