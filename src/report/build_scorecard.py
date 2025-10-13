"""Utilities for pivoting diagnostics scorecards."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("run_id", "phase", "env", "metric", "value")


def _ensure_dataframe(scorecard: Path | str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(scorecard, pd.DataFrame):
        df = scorecard.copy()
    else:
        df = pd.read_csv(Path(scorecard))
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_fmt = ", ".join(missing)
        raise ValueError(f"Scorecard is missing required columns: {missing_fmt}")
    return df


def build_scorecard(scorecard: Path | str | pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-format diagnostics CSV into a wide table.

    Parameters
    ----------
    scorecard:
        Either a path to a CSV file produced by :mod:`src.diagnostics.export`
        or a :class:`pandas.DataFrame` with the same schema.

    Returns
    -------
    pandas.DataFrame
        A dataframe indexed by run identifier and phase with one column per
        ``(env, metric)`` pair. Column names are flattened as
        ``"{env}__{metric}"`` for ease of downstream consumption.
    """

    df = _ensure_dataframe(scorecard)
    if df.empty:
        return pd.DataFrame(columns=["run_id", "phase"])

    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.sort_values(["run_id", "phase", "env", "metric"], inplace=True)

    pivot = (
        df.set_index(["run_id", "phase", "env", "metric"])["value"]
        .unstack(["env", "metric"])
        .sort_index(axis=1, level=[0, 1])
        .sort_index(axis=0)
    )

    pivot.reset_index(inplace=True)

    def _flatten(column: object) -> str:
        if not isinstance(column, tuple):
            return str(column)
        parts = [str(part) for part in column if part not in (None, "")]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return "__".join(parts)

    pivot.columns = [_flatten(col) for col in pivot.columns]
    return pivot


__all__ = ["build_scorecard"]
