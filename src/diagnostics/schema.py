"""Canonical schema helpers for diagnostics parquet exports."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from pandas.api import types as ptypes

SCHEMA_VERSION = "1.0.0"

CANONICAL_COLUMNS = ("env", "split", "seed", "algo", "metric", "value")

_STRING_COLUMNS = ("env", "split", "algo", "metric")
_INTEGER_COLUMNS = ("seed",)
_FLOAT_COLUMNS = ("value",)


@dataclass(frozen=True)
class DiagnosticsSchema:
    columns: tuple[str, ...] = CANONICAL_COLUMNS
    schema_version: str = SCHEMA_VERSION


def normalize_diagnostics_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` coerced to the canonical diagnostics schema."""

    normalized = frame.copy()
    for column in _STRING_COLUMNS:
        normalized[column] = normalized[column].astype("string")
    for column in _INTEGER_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype("int64")
    for column in _FLOAT_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("float64")
    return normalized


def validate_diagnostics_table(frame: pd.DataFrame) -> None:
    """Ensure that ``frame`` satisfies the canonical diagnostics schema."""

    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Diagnostics table missing required columns: {missing}")

    issues: Dict[str, str] = {}
    for column in _STRING_COLUMNS:
        if not ptypes.is_string_dtype(frame[column]):
            issues[column] = str(frame[column].dtype)
    for column in _INTEGER_COLUMNS:
        if not ptypes.is_integer_dtype(frame[column]):
            issues[column] = str(frame[column].dtype)
    for column in _FLOAT_COLUMNS:
        if not ptypes.is_float_dtype(frame[column]):
            issues[column] = str(frame[column].dtype)
    if issues:
        details = ", ".join(f"{col} ({dtype})" for col, dtype in issues.items())
        raise TypeError(f"Diagnostics table has incorrect dtypes: {details}")


__all__ = [
    "CANONICAL_COLUMNS",
    "DiagnosticsSchema",
    "SCHEMA_VERSION",
    "normalize_diagnostics_frame",
    "validate_diagnostics_table",
]
