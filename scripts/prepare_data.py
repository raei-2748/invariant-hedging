#!/usr/bin/env python3
"""Stage required datasets for the paper reproduction pipeline."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Iterable


def _compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    shutil.copy2(source, target)
    return target


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Stage the minimal dataset used by the CI smoke pipeline.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Override the staging directory (relative to the repository root).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "data" / "spy_sample.csv"
    if not source.exists():
        print(
            "Missing data/spy_sample.csv. "
            "Download the SPY OHLC sample described in README.md and place it under data/.",
            file=sys.stderr,
        )
        return 1

    if args.out is not None:
        stage_dir = args.out
        if not stage_dir.is_absolute():
            stage_dir = (repo_root / stage_dir).resolve()
    elif args.mini:
        stage_dir = repo_root / "artifacts" / "data-mini"
    else:
        stage_dir = repo_root / "outputs" / "paper_data"

    target = _stage(source, stage_dir)
    checksum = _compute_checksum(source)
    try:
        relative_target = target.relative_to(repo_root)
    except ValueError:
        relative_target = target
    print(
        "Staged dataset: {src} → {dst} (sha256={checksum})".format(
            src=source.relative_to(repo_root),
            dst=relative_target,
            checksum=checksum,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
