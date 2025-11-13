#!/usr/bin/env python3
"""Validate CSV schemas for real data anchors."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class FileReport:
    path: Path
    exists: bool
    missing_columns: List[str]
    row_count: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    notes: List[str] | None = None


DATE_COLUMNS = ("trade_date", "date")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="CSV files to validate",
    )
    parser.add_argument(
        "--required",
        type=str,
        required=True,
        help="Comma-separated list of required columns",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional markdown file to append summary rows",
    )
    return parser.parse_args()


def detect_date_column(columns: List[str]) -> str | None:
    for candidate in DATE_COLUMNS:
        if candidate in columns:
            return candidate
    return None


def summarize_file(path: Path, required_cols: List[str]) -> FileReport:
    if not path.exists():
        return FileReport(path=path, exists=False, missing_columns=required_cols, notes=["file missing"])

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - surfaces ingest failures
        return FileReport(
            path=path,
            exists=True,
            missing_columns=required_cols,
            notes=[f"failed to read CSV: {exc}"],
        )

    columns = list(df.columns)
    missing = [col for col in required_cols if col not in columns]
    date_col = detect_date_column(columns)
    notes: List[str] = []
    start_date = end_date = None

    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            valid = dates.dropna()
            if not valid.empty:
                start_date = valid.min().strftime("%Y-%m-%d")
                end_date = valid.max().strftime("%Y-%m-%d")
            else:
                notes.append(f"no valid timestamps in {date_col}")
        except Exception as exc:  # pragma: no cover - diagnostic only
            notes.append(f"failed to parse dates in {date_col}: {exc}")
    else:
        notes.append("no trade_date/date column found")

    return FileReport(
        path=path,
        exists=True,
        missing_columns=missing,
        row_count=int(len(df)),
        start_date=start_date,
        end_date=end_date,
        notes=notes or None,
    )


def emit_markdown(reports: List[FileReport]) -> str:
    output = ["| file | exists | rows | date_range | missing_columns | notes |", "| --- | --- | --- | --- | --- | --- |"]
    for report in reports:
        date_range = ""
        if report.start_date and report.end_date:
            date_range = f"{report.start_date} → {report.end_date}"
        missing = ", ".join(report.missing_columns) if report.missing_columns else ""
        notes = ", ".join(report.notes) if report.notes else ""
        output.append(
            "| {file} | {exists} | {rows} | {date_range} | {missing} | {notes} |".format(
                file=report.path,
                exists="yes" if report.exists else "no",
                rows=report.row_count if report.row_count is not None else "",
                date_range=date_range,
                missing=missing,
                notes=notes,
            )
        )
    return "\n".join(output)


def main() -> int:
    args = parse_args()
    required_cols = [col.strip() for col in args.required.split(",") if col.strip()]

    reports = [summarize_file(Path(file), required_cols) for file in args.files]

    markdown = emit_markdown(reports)
    print(markdown)

    if args.out:
        header_needed = not args.out.exists()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8", newline="") as fh:
            if header_needed:
                fh.write(markdown + "\n")
            else:
                # Append only the table rows (skip header) to avoid duplication.
                rows = markdown.splitlines()[2:]
                for row in rows:
                    fh.write(row + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
