"""CLI utility for loading SPY slices defined via YAML split configs.

This module provides deterministic slicing of the historical SPY dataset based on
paper-aligned windows. The CLI entry point can be invoked with
``python -m src.data.spy_loader --split configs/splits/spy_train.yaml``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}


@dataclass(frozen=True)
class SplitConfig:
    """Typed representation of a split configuration."""

    path: Path
    name: str
    instrument: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    regime_tag: str
    notes: Optional[str] = None
    no_overlap_with_tests: bool = False
    include_gfc: bool = False
    raw: Dict[str, object] = None

    def to_metadata(self) -> Dict[str, object]:
        meta = {
            "split_name": self.name,
            "instrument": self.instrument,
            "start_date": self.start_date.date().isoformat(),
            "end_date": self.end_date.date().isoformat(),
            "regime_tag": self.regime_tag,
        }
        if self.notes is not None:
            meta["notes"] = self.notes
        if self.no_overlap_with_tests:
            meta["no_overlap_with_tests"] = True
        if self.include_gfc:
            meta["include_gfc"] = True
        return meta


def _parse_timestamp(value: object, field: str, source: Path) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas error formatting is descriptive
        raise ValueError(f"Failed to parse {field}='{value}' in {source}") from exc
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    ts = ts.normalize()
    return ts


def load_split_config(path: Path, *, validate_overlap: bool = True) -> SplitConfig:
    if not path.exists():
        raise FileNotFoundError(f"Split YAML not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    required_fields = ["name", "instrument", "start_date", "end_date", "regime_tag"]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise ValueError(f"Missing required fields {missing} in split config {path}")

    start_ts = _parse_timestamp(data["start_date"], "start_date", path)
    end_ts = _parse_timestamp(data["end_date"], "end_date", path)
    if end_ts < start_ts:
        raise ValueError(
            f"Split {data['name']} in {path} has end_date before start_date: "
            f"{end_ts.date().isoformat()} < {start_ts.date().isoformat()}"
        )

    config = SplitConfig(
        path=path,
        name=str(data["name"]),
        instrument=str(data["instrument"]),
        start_date=start_ts,
        end_date=end_ts,
        regime_tag=str(data["regime_tag"]),
        notes=data.get("notes"),
        no_overlap_with_tests=bool(data.get("no_overlap_with_tests", False)),
        include_gfc=bool(data.get("include_gfc", False)),
        raw=data,
    )

    if validate_overlap and config.no_overlap_with_tests:
        _ensure_no_overlap_with_tests(config)

    return config


def _ensure_no_overlap_with_tests(validation_config: SplitConfig) -> None:
    split_dir = validation_config.path.parent
    for test_yaml in sorted(split_dir.glob("spy_test_*.yaml")):
        test_config = load_split_config(test_yaml, validate_overlap=False)
        if _ranges_overlap(
            validation_config.start_date,
            validation_config.end_date,
            test_config.start_date,
            test_config.end_date,
        ):
            raise ValueError(
                "Validation split {val} ({val_start}–{val_end}) overlaps test split "
                "{test} ({test_start}–{test_end}). Adjust YAML boundaries to maintain "
                "disjoint regimes.".format(
                    val=validation_config.name,
                    val_start=validation_config.start_date.date().isoformat(),
                    val_end=validation_config.end_date.date().isoformat(),
                    test=test_config.name,
                    test_start=test_config.start_date.date().isoformat(),
                    test_end=test_config.end_date.date().isoformat(),
                )
            )


def _ranges_overlap(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    return latest_start <= earliest_end


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "CSV is missing required columns: {missing}".format(
                missing=", ".join(sorted(missing))
            )
        )
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Encountered unparsable dates in CSV.")
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    df["Date"] = df["Date"].dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def slice_for_split(df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    mask = (df["Date"] >= config.start_date) & (df["Date"] <= config.end_date)
    sliced = df.loc[mask].copy()
    if sliced.empty:
        raise ValueError(
            f"Slice for split {config.name} ({config.start_date.date().isoformat()}–"
            f"{config.end_date.date().isoformat()}) is empty. Check the CSV data and YAML."
        )
    sliced = sliced.sort_values("Date").reset_index(drop=True)
    if sliced["Date"].duplicated().any():
        duplicates = sliced.loc[sliced["Date"].duplicated(), "Date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(
            "Duplicate business days detected in slice for split {name}: {dates}".format(
                name=config.name, dates=", ".join(duplicates)
            )
        )
    return sliced


def compute_log_returns(slice_df: pd.DataFrame) -> pd.DataFrame:
    slice_df = slice_df.copy()
    slice_df["log_ret"] = np.log(slice_df["Adj Close"] / slice_df["Adj Close"].shift(1))
    slice_df = slice_df.dropna(subset=["log_ret"]).reset_index(drop=True)
    if slice_df.empty:
        raise ValueError("Slice has insufficient rows to compute log returns after dropping the first day.")
    return slice_df


def attach_metadata_columns(slice_df: pd.DataFrame, config: SplitConfig) -> pd.DataFrame:
    enriched = slice_df.copy()
    enriched["regime_tag"] = config.regime_tag
    enriched["split_name"] = config.name
    return enriched


def _collect_git_hash() -> Optional[str]:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_metadata(
    run_dir: Path,
    config: SplitConfig,
    row_count: int,
    csv_path: Path,
) -> None:
    metadata = config.to_metadata()
    timestamp = datetime.now(UTC).replace(microsecond=0)
    metadata.update(
        {
            "row_count": int(row_count),
            "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "csv_path": str(csv_path.resolve()),
            "split_yaml_path": str(config.path.resolve()),
            "git_hash": _collect_git_hash(),
            "python_version": sys.version,
            "platform": platform.platform(),
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    split_copy_path = run_dir / "split.yaml"
    shutil.copy2(config.path, split_copy_path)


def run_loader(
    split_path: Path,
    csv_path: Path,
    out_parquet: Optional[Path],
    runs_dir: Path,
) -> pd.DataFrame:
    config = load_split_config(split_path)
    df = load_raw_csv(csv_path)
    sliced = slice_for_split(df, config)
    sliced = compute_log_returns(sliced)
    enriched = attach_metadata_columns(sliced, config)

    if out_parquet is not None:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        try:
            enriched.to_parquet(out_parquet)
        except ImportError as exc:  # pragma: no cover - dependent on optional deps
            raise RuntimeError(
                "Writing parquet requires optional dependency pyarrow or fastparquet."
            ) from exc

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    unique_suffix = os.urandom(3).hex()
    run_dir = runs_dir / f"{run_stamp}_{unique_suffix}"
    write_metadata(run_dir, config, len(enriched), csv_path)

    first_date = enriched["Date"].iloc[0].date().isoformat()
    last_date = enriched["Date"].iloc[-1].date().isoformat()
    summary = (
        f"{config.name}: {first_date} -> {last_date} "
        f"({len(enriched)} rows) [regime: {config.regime_tag}]"
    )
    print(summary)

    return enriched


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slice SPY data according to a split YAML")
    parser.add_argument("--split", required=True, type=Path, help="Path to split YAML")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/spy.csv"),
        help="Path to raw SPY CSV (default: data/raw/spy.csv)",
    )
    parser.add_argument(
        "--out_parquet",
        type=Path,
        default=None,
        help="Optional path to write the sliced parquet dataset",
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("runs"),
        help="Directory where run metadata will be stored",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_loader(args.split, args.csv, args.out_parquet, args.runs_dir)


if __name__ == "__main__":
    main()
