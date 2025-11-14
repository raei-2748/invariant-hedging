#!/usr/bin/env python3
"""Locate the best checkpoint in a run directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _from_manifest(manifest_path: Path) -> Path | None:
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except Exception:
        return None
    if not manifest:
        return None
    entry = manifest[0]
    rel_path = entry.get("path")
    if not isinstance(rel_path, str):
        return None
    return manifest_path.parent / rel_path


def _from_glob(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory containing checkpoints")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
        return 1

    manifest_path = checkpoint_dir / "manifest.json"
    checkpoint = _from_manifest(manifest_path)
    if checkpoint is None:
        checkpoint = _from_glob(checkpoint_dir)
    if checkpoint is None or not checkpoint.exists():
        print("No checkpoints saved in run directory", file=sys.stderr)
        return 1

    print(str(checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
