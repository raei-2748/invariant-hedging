#!/usr/bin/env python3
"""Paper provenance manifest builder."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def _split_csv(raw: str) -> List[str]:
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


def _split_ints(raw: str) -> List[int]:
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return values


def _method_slug(method: str) -> str:
    return method.replace("/", "__")


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git(command: Sequence[str]) -> str | None:
    try:
        output = subprocess.check_output(command, cwd=REPO_ROOT)
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.decode().strip()


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = None
            continue
        version = getattr(module, "__version__", None)
        if version is None and hasattr(module, "version"):
            candidate = getattr(module, "version")
            if isinstance(candidate, str):
                version = candidate
        versions[name] = version if isinstance(version, str) else None
    return versions


def _load_json(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Metrics file not found: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Metrics file at {path} does not contain an object")
    return data


def collect_metrics(run_root: Path, methods: Sequence[str], seeds: Sequence[int], eval_windows: Sequence[str]) -> List[dict[str, object]]:
    entries: List[dict[str, object]] = []
    for method in methods:
        slug = _method_slug(method)
        for seed in seeds:
            run_dir = run_root / slug / f"seed_{seed}"
            train_metrics_path = run_dir / "final_metrics.json"
            metrics = _load_json(train_metrics_path)
            entries.append(
                {
                    "stage": "train",
                    "method": method,
                    "seed": seed,
                    "path": _relative_path(train_metrics_path),
                    "metrics": metrics,
                }
            )
            for window in eval_windows:
                eval_dir = run_dir / "eval" / window
                eval_metrics_path = eval_dir / "final_metrics.json"
                eval_metrics = _load_json(eval_metrics_path)
                entries.append(
                    {
                        "stage": "eval",
                        "method": method,
                        "seed": seed,
                        "window": window,
                        "path": _relative_path(eval_metrics_path),
                        "metrics": eval_metrics,
                    }
                )
    return entries


def build_grid(methods: Sequence[str], seeds: Sequence[int], eval_windows: Sequence[str]) -> List[dict[str, object]]:
    return [
        {"method": method, "seed": seed, "eval_window": window}
        for method in methods
        for seed in seeds
        for window in eval_windows
    ]


def _resolve_provenance_functions() -> Tuple[callable, callable]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.evaluation.reporting.paper_provenance import collect_provenance, write_provenance

    return collect_provenance, write_provenance


def _manifest_from_grid(args: argparse.Namespace) -> None:
    methods = _split_csv(args.methods)
    seeds = _split_ints(args.seeds)
    eval_windows = _split_csv(args.eval_windows)
    if not methods or not seeds or not eval_windows:
        raise SystemExit("At least one method, seed, and evaluation window is required")

    run_root = Path(args.run_root or (REPO_ROOT / "runs" / "paper")).expanduser().resolve()
    output_path = Path(args.output or (run_root / "paper_provenance.json")).expanduser().resolve()
    metrics_path = Path(args.metrics_output or (run_root / "final_metrics.json")).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_entries = collect_metrics(run_root, methods, seeds, eval_windows)

    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": _relative_path(run_root),
        "configurations": {
            "methods": methods,
            "seeds": seeds,
            "evaluation_windows": eval_windows,
            "grid": build_grid(methods, seeds, eval_windows),
        },
        "git": {
            "sha": _git(["git", "rev-parse", "HEAD"]),
            "status": _git(["git", "status", "--short"]),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": _package_versions(["torch", "numpy", "pandas", "hydra", "omegaconf"]),
        "environment": dict(os.environ),
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)

    metrics_doc = {"generated_at": provenance["generated_at"], "runs": metrics_entries}
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_doc, fh, indent=2)

    print(f"Provenance written to {output_path}")
    print(f"Final metrics summary written to {metrics_path}")


def _legacy_manifest(args: argparse.Namespace) -> None:
    collect_provenance, write_provenance = _resolve_provenance_functions()
    data = collect_provenance(args.run_dir)
    indent = 2 if args.pretty or args.output else None
    text = json.dumps(data, indent=indent, sort_keys=True)
    if args.output is not None:
        write_provenance(args.output, data)
    print(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", help="Comma-separated list of Hydra training configs")
    parser.add_argument("--seeds", help="Comma-separated list of seeds")
    parser.add_argument("--eval-windows", help="Comma-separated list of evaluation windows")
    parser.add_argument("--run-root", help="Root directory containing the paper runs")
    parser.add_argument("--output", help="Path to write the provenance JSON")
    parser.add_argument("--metrics-output", help="Path to write the consolidated metrics JSON")
    parser.add_argument("--run-dir", type=Path, default=None, help="Legacy: run directory to fingerprint")
    parser.add_argument("--pretty", action="store_true", help="Legacy: pretty-print provenance JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.methods or args.seeds or args.eval_windows:
        if not (args.methods and args.seeds and args.eval_windows):
            raise SystemExit("--methods, --seeds, and --eval-windows must be provided together")
        _manifest_from_grid(args)
    else:
        _legacy_manifest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
