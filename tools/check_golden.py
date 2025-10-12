"""Regression checker for paper-lite aggregate metrics."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_REL_TOL = 0.025
DEFAULT_ABS_TOL = 1e-8
DEFAULT_IGNORE = {"timestamp", "commit"}


@dataclass(frozen=True)
class Drift:
    """Represents a single cell where the current value drifts from the golden."""

    file: Path
    column: str
    key: tuple[tuple[str, object], ...]
    expected: float
    actual: float
    abs_diff: float
    rel_diff: float

    def format(self) -> str:
        key_repr = ", ".join(f"{name}={value}" for name, value in self.key)
        rel_pct = self.rel_diff * 100.0
        return (
            f"{self.file} [{key_repr}] column '{self.column}' drifted: "
            f"expected {self.expected:.6g}, got {self.actual:.6g} "
            f"(Δ={self.abs_diff:.6g}, rel={rel_pct:.2f}%)"
        )


@dataclass(frozen=True)
class StructureIssue:
    file: Path
    message: str

    def format(self) -> str:
        return f"{self.file}: {self.message}"


def _normalise_frame(df: pd.DataFrame, ignore: Iterable[str]) -> pd.DataFrame:
    drop = [col for col in ignore if col in df.columns]
    if drop:
        df = df.drop(columns=drop)
    return df


def _infer_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols: list[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(column)
    return numeric_cols


def _build_index(df: pd.DataFrame, numeric_cols: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    non_numeric = [col for col in df.columns if col not in numeric_cols]
    if not non_numeric:
        df = df.assign(__row_id__=np.arange(len(df)))
        non_numeric = ["__row_id__"]
    if df.duplicated(subset=non_numeric).any():
        raise ValueError(f"Non-unique keys detected using columns {non_numeric}")
    df = df.set_index(non_numeric).sort_index()
    return df, non_numeric


def compare_tables(
    golden_path: Path,
    current_path: Path,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
    ignore_columns: Iterable[str] = DEFAULT_IGNORE,
) -> tuple[list[StructureIssue], list[Drift]]:
    try:
        golden = pd.read_csv(golden_path)
    except FileNotFoundError:
        return [StructureIssue(golden_path, "missing golden file")], []
    try:
        current = pd.read_csv(current_path)
    except FileNotFoundError:
        return [StructureIssue(current_path, "missing current file")], []

    ignore = set(ignore_columns)
    golden = _normalise_frame(golden, ignore)
    current = _normalise_frame(current, ignore)

    if set(golden.columns) != set(current.columns):
        missing = set(golden.columns) - set(current.columns)
        extra = set(current.columns) - set(golden.columns)
        details = []
        if missing:
            details.append(f"missing columns {sorted(missing)}")
        if extra:
            details.append(f"unexpected columns {sorted(extra)}")
        issue = StructureIssue(current_path, "; ".join(details))
        return [issue], []

    golden = golden[sorted(golden.columns)]
    current = current[sorted(current.columns)]

    numeric_cols = _infer_numeric_columns(golden)
    current_numeric = _infer_numeric_columns(current)
    if set(numeric_cols) != set(current_numeric):
        issue = StructureIssue(
            current_path,
            f"numeric column mismatch (golden={sorted(numeric_cols)}, current={sorted(current_numeric)})",
        )
        return [issue], []

    try:
        golden_indexed, key_columns = _build_index(golden, numeric_cols)
        current_indexed, _ = _build_index(current, numeric_cols)
    except ValueError as exc:
        issue = StructureIssue(current_path, str(exc))
        return [issue], []

    missing_rows = golden_indexed.index.difference(current_indexed.index)
    extra_rows = current_indexed.index.difference(golden_indexed.index)
    issues: list[StructureIssue] = []
    if len(missing_rows) > 0:
        details = "; ".join(str(tuple(idx)) for idx in missing_rows[:5])
        issues.append(StructureIssue(current_path, f"missing rows {details}"))
    if len(extra_rows) > 0:
        details = "; ".join(str(tuple(idx)) for idx in extra_rows[:5])
        issues.append(StructureIssue(current_path, f"unexpected rows {details}"))
    if issues:
        return issues, []

    drifts: list[Drift] = []
    for column in numeric_cols:
        golden_values = golden_indexed[column]
        current_values = current_indexed[column]
        # Align to avoid dtype mismatches
        current_values = current_values.reindex(golden_values.index)
        for key, expected in golden_values.items():
            actual = float(current_values.loc[key])
            if pd.isna(expected) and pd.isna(actual):
                continue
            if pd.isna(expected) or pd.isna(actual):
                drifts.append(
                    Drift(
                        current_path,
                        column,
                        tuple(zip(key_columns, key if isinstance(key, tuple) else (key,))),
                        float(expected) if not pd.isna(expected) else math.nan,
                        actual,
                        math.nan,
                        math.nan,
                    )
                )
                continue
            expected_f = float(expected)
            abs_diff = abs(actual - expected_f)
            scale = max(abs(expected_f), 1e-12)
            rel_diff = abs_diff / scale
            threshold = max(abs_tol, rel_tol * scale)
            if abs_diff > threshold:
                key_tuple: tuple[tuple[str, object], ...]
                if isinstance(key, tuple):
                    key_tuple = tuple(zip(key_columns, key))
                else:
                    key_tuple = ((key_columns[0], key),)
                drifts.append(
                    Drift(
                        current_path,
                        column,
                        key_tuple,
                        expected_f,
                        actual,
                        abs_diff,
                        rel_diff,
                    )
                )
    return [], drifts


def compare_directories(
    golden_dir: Path,
    current_dir: Path,
    *,
    pattern: str = "*.csv",
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
    ignore_columns: Iterable[str] = DEFAULT_IGNORE,
) -> tuple[list[StructureIssue], list[Drift]]:
    issues: list[StructureIssue] = []
    drifts: list[Drift] = []

    golden_dir = golden_dir.resolve()
    current_dir = current_dir.resolve()

    golden_paths = sorted(golden_dir.rglob(pattern))
    seen: set[Path] = set()
    for golden_path in golden_paths:
        relative = golden_path.relative_to(golden_dir)
        current_path = current_dir / relative
        structure, delta = compare_tables(
            golden_path,
            current_path,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            ignore_columns=ignore_columns,
        )
        issues.extend(structure)
        drifts.extend(delta)
        seen.add(relative)

    # Detect stray files in current tables directory
    for current_path in current_dir.rglob(pattern):
        relative = current_path.relative_to(current_dir)
        if relative in seen:
            continue
        issues.append(StructureIssue(current_path, "no corresponding golden reference"))

    return issues, drifts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check generated tables against golden references.")
    parser.add_argument("--goldens", type=Path, default=Path("goldens"), help="Directory with golden CSV files")
    parser.add_argument("--tables", type=Path, default=Path("tables"), help="Directory with current tables")
    parser.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL, help="Relative tolerance for numeric drift")
    parser.add_argument("--abs-tol", type=float, default=DEFAULT_ABS_TOL, help="Absolute tolerance for near-zero values")
    parser.add_argument(
        "--ignore-column",
        dest="ignore_columns",
        action="append",
        default=list(DEFAULT_IGNORE),
        help="Column to ignore during comparison (can be specified multiple times)",
    )
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for files to compare")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ignore = set(args.ignore_columns) if args.ignore_columns else set()
    issues, drifts = compare_directories(
        args.goldens,
        args.tables,
        pattern=args.pattern,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        ignore_columns=ignore,
    )
    problems = [*issues, *drifts]
    if problems:
        for problem in problems:
            if isinstance(problem, Drift):
                print(problem.format())
            else:
                print(problem.format())
        return 1
    print("Golden comparison passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
