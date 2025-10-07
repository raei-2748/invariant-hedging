"""Diagnostic metrics for invariant hedging runs."""
from __future__ import annotations
from typing import Mapping, Sequence

import numpy as np


def _sanitize(values: Mapping[str, float | None] | Sequence[float | None]) -> np.ndarray:
    if isinstance(values, Mapping):
        data = [v for v in values.values() if v is not None]
    else:
        data = [v for v in values if v is not None]
    if not data:
        return np.array([], dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


def invariance_gap(train_env_risks: Mapping[str, float | None] | Sequence[float | None]) -> float:
    """Compute the invariance gap (IG)."""
    arr = _sanitize(train_env_risks)
    if arr.size <= 1:
        return 0.0
    return float(arr.max() - arr.min())


def worst_group_gap(
    train_env_risks: Mapping[str, float | None] | Sequence[float | None],
    test_env_risks: Mapping[str, float | None] | Sequence[float | None],
) -> float:
    """Compute the worst-group gap (WG)."""
    train_arr = _sanitize(train_env_risks)
    test_arr = _sanitize(test_env_risks)
    if test_arr.size == 0:
        return 0.0
    train_worst = train_arr.max() if train_arr.size else 0.0
    return float(test_arr.max() - train_worst)


def mechanism_sensitivity_index(
    s_phi: float | None, s_r: float | None, eps: float = 1e-6
) -> float:
    """Return the mechanism sensitivity index (MSI)."""
    if s_phi is None or s_r is None:
        return 0.0
    return float(s_phi / (s_r + eps))


def invariant_gap(
    per_env_values: Sequence[float | None],
) -> float:  # pragma: no cover - legacy shim
    return invariance_gap(per_env_values)


def worst_group(per_env_values: Sequence[float | None]) -> float:  # pragma: no cover - legacy shim
    filtered = [v for v in per_env_values if v is not None]
    if not filtered:
        return 0.0
    return float(max(filtered))


def mechanistic_sensitivity(
    sensitivities: Sequence[float | None],
) -> float:  # pragma: no cover - legacy shim
    filtered = [v for v in sensitivities if v is not None]
    if not filtered:
        return 0.0
    return float(np.mean(np.abs(filtered)))


__all__ = [
    "invariance_gap",
    "worst_group_gap",
    "mechanism_sensitivity_index",
    "invariant_gap",
    "worst_group",
    "mechanistic_sensitivity",
]
