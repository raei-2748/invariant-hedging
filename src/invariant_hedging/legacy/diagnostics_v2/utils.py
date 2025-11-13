"""Shared helpers for diagnostics computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrimmedStats:
    """Container holding trimmed statistics for debugging."""

    raw_values: np.ndarray
    trimmed_values: np.ndarray
    proportion_to_cut: float

    def to_dict(self) -> dict:
        return {
            "raw_values": self.raw_values.tolist(),
            "trimmed_values": self.trimmed_values.tolist(),
            "proportion_to_cut": float(self.proportion_to_cut),
        }


def to_dataframe(data, columns: Sequence[str]) -> pd.DataFrame:
    """Create a DataFrame from arbitrary diagnostics payloads.

    The diagnostics pipeline is flexible about which container type callers use.
    Tests frequently rely on passing dictionaries or already-instantiated DataFrames.
    This helper normalises inputs while preserving column ordering.
    """

    if data is None:
        return pd.DataFrame(columns=columns)
    if isinstance(data, pd.DataFrame):
        missing = [col for col in columns if col not in data.columns]
        if missing:
            raise KeyError(f"Missing columns {missing} in diagnostics input")
        return data.copy()
    if isinstance(data, dict):
        return pd.DataFrame(data, columns=columns)
    if isinstance(data, Iterable):
        return pd.DataFrame(list(data), columns=columns)
    raise TypeError(f"Unsupported data format: {type(data)!r}")


def trimmed_mean(values: Sequence[float], proportion_to_cut: float) -> TrimmedStats:
    """Compute the symmetric trimmed mean while exposing the trimmed sample.

    The paper mandates a 10% trimmed mean, but keeping the helper generic makes it
    easier to write targeted unit tests.  The function never mutates the input
    sequence and always returns a :class:`TrimmedStats` that captures the raw and
    retained samples for debugging or exporting into appendix tables.
    """

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return TrimmedStats(array, array, proportion_to_cut)

    proportion = float(proportion_to_cut)
    if proportion < 0 or proportion >= 0.5:
        raise ValueError("proportion_to_cut must be in [0, 0.5)")

    sorted_vals = np.sort(array)
    k = int(np.floor(proportion * sorted_vals.size))
    if k == 0:
        trimmed = sorted_vals
    else:
        trimmed = sorted_vals[k:-k] if sorted_vals.size > 2 * k else np.array([], dtype=np.float64)
    return TrimmedStats(sorted_vals, trimmed, proportion)


def safe_mean(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(array.mean())


def ensure_array(values) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)
    if isinstance(values, (list, tuple)):
        return np.asarray(values, dtype=np.float64)
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=np.float64, copy=True)
    return np.asarray(list(values), dtype=np.float64)
