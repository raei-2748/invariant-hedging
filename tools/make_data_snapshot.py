#!/usr/bin/env python3
"""Build reproducible data snapshots for development and CI."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable

from invariant_hedging import get_repo_root

REPO_ROOT = get_repo_root()
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", REPO_ROOT / "data")


def as_path(path_like: os.PathLike[str] | str) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        return REPO_ROOT / path
    return path


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum_file(path: Path, digest: str) -> None:
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def build_full_snapshot(raw_source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_source, destination)
    digest = compute_sha256(destination)
    write_checksum_file(destination, digest)
    return digest


def iter_filtered_rows(raw_source: Path) -> Iterable[list[str]]:
    with raw_source.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"The file {raw_source} does not include a header row.")
        yield reader.fieldnames
        strikes_to_keep = {"310.0", "330.0"}
        option_types = {"call", "put"}
        for row in reader:
            if row.get("strike") not in strikes_to_keep:
                continue
            if row.get("option_type") not in option_types:
                continue
            yield [row[field] for field in reader.fieldnames]


def build_mini_snapshot(raw_source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for record in iter_filtered_rows(raw_source):
            writer.writerow(record)
    digest = compute_sha256(destination)
    write_checksum_file(destination, digest)
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full", "mini", "both"),
        default="both",
        help="Which snapshot(s) to materialise (default: both)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory that contains the raw/ tree (defaults to $DATA_DIR or repo/data).",
    )
    return parser.parse_args()


def ensure_raw_dataset(raw_dir: Path) -> Path:
    raw_file = raw_dir / "spy_options_synthetic.csv"
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Missing {raw_file}. Run tools/fetch_data.sh before making snapshots."
        )
    return raw_file


def main() -> int:
    args = parse_args()
    data_dir = as_path(args.data_dir)
    raw_dir = data_dir / "raw"
    raw_file = ensure_raw_dataset(raw_dir)

    snapshot_dir = raw_dir / "snapshots"
    full_target = snapshot_dir / "full" / "spy_options_full.csv"
    mini_target = snapshot_dir / "mini" / "spy_options_mini.csv"

    if args.mode in {"full", "both"}:
        digest = build_full_snapshot(raw_file, full_target)
        print(f"[make_data_snapshot] Wrote full snapshot: {full_target} (sha256={digest})")

    if args.mode in {"mini", "both"}:
        digest = build_mini_snapshot(raw_file, mini_target)
        print(f"[make_data_snapshot] Wrote mini snapshot: {mini_target} (sha256={digest})")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
