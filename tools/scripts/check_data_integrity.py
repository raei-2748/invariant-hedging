#!/usr/bin/env python3
"""Validate sample and real-data staging for the HIRM pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_OPTIONMETRICS = {
    "trade_date",
    "expiration",
    "strike",
    "option_type",
    "bid",
    "ask",
    "mid",
    "implied_vol",
}


def _fail(message: str) -> None:
    print(f"[data-check] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def _validate_sample(sample_csv: Path) -> None:
    if not sample_csv.exists():
        _fail(f"Sample dataset missing: {sample_csv}")
    df = pd.read_csv(sample_csv)
    required = {"date", "spot", "vol"}
    missing = required - set(df.columns)
    if missing:
        _fail(f"Sample CSV {sample_csv} missing columns {sorted(missing)}")
    if df[required].isna().any().any():
        _fail(f"Sample CSV {sample_csv} contains NaNs in {required}")
    if (df["spot"] <= 0).any():
        _fail(f"Sample CSV {sample_csv} contains non-positive spot values")


def _validate_optionmetrics_file(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_OPTIONMETRICS - set(df.columns)
    if missing:
        _fail(f"{csv_path} missing OptionMetrics columns {sorted(missing)}")
    numeric_cols = ["bid", "ask", "mid", "implied_vol"]
    if df[numeric_cols].isna().any().any():
        _fail(f"{csv_path} has NaNs in {numeric_cols}")
    if (df[numeric_cols] < 0).any().any():
        _fail(f"{csv_path} has negative prices or vols")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    if df["trade_date"].isna().any():
        _fail(f"{csv_path} has unparsable trade_date entries")
    min_date = df["trade_date"].min()
    max_date = df["trade_date"].max()
    if min_date.year < 2000 or max_date.year > 2035:
        _fail(f"{csv_path} trade_date range ({min_date}–{max_date}) outside [2000, 2035]")


def _validate_optionmetrics(optionmetrics_dir: Path) -> None:
    csvs = sorted(optionmetrics_dir.glob("*.csv"))
    if not csvs:
        _fail(f"No OptionMetrics CSVs found under {optionmetrics_dir}")
    for csv_path in csvs:
        _validate_optionmetrics_file(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root data directory (default: %(default)s)")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    sample_csv = data_root / "raw" / "spy_options_synthetic.csv"
    optionmetrics_dir = data_root / "raw" / "optionmetrics"

    _validate_sample(sample_csv)
    if optionmetrics_dir.exists():
        _validate_optionmetrics(optionmetrics_dir)
    else:
        print(f"[data-check] Skipping OptionMetrics validation (directory missing: {optionmetrics_dir})")

    print("[data-check] All validations passed.")


if __name__ == "__main__":
    main()
