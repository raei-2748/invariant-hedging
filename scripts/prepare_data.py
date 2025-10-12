#!/usr/bin/env python3
"""Stage required datasets for the paper reproduction pipeline."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "data" / "spy_sample.csv"
    if not source.exists():
        print(
            "Missing data/spy_sample.csv. "
            "Download the SPY OHLC sample described in README.md and place it under data/.",
            file=sys.stderr,
        )
        return 1

    stage_dir = repo_root / "outputs" / "paper_data"
    stage_dir.mkdir(parents=True, exist_ok=True)
    target = stage_dir / source.name

    shutil.copy2(source, target)
    print(f"Staged dataset: {source} -> {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
