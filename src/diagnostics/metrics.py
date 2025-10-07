"""Diagnostic metrics for invariant hedging runs."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Union

import numpy as np

Number = Union[int, float]


def _to_values(data: Mapping[str, Number] | Sequence[Number]) -> np.ndarray:
    values = list(data.values()) if isinstance(data, Mapping) else list(data)
    if not values:
        return np.array([], dtype=float)
    return np.asarray(values, dtype=float)


def invariance_gap(train_env_risks: Mapping[str, Number] | Sequence[Number]) -> float:
    """Maximum minus minimum risk across training environments."""
    values = _to_values(train_env_risks)
    if values.size <= 1:
        return 0.0
    return float(values.max() - values.min())


def worst_group_gap(
    train_env_risks: Mapping[str, Number] | Sequence[Number],
    test_env_risks: Mapping[str, Number] | Sequence[Number],
) -> float:
    """Difference between worst test risk and worst train risk."""
    train_values = _to_values(train_env_risks)
    test_values = _to_values(test_env_risks)
    if test_values.size == 0:
        return 0.0
    train_max = float(train_values.max()) if train_values.size else 0.0
    return float(test_values.max() - train_max)


def mechanism_sensitivity_index(s_phi: float, s_r: float, eps: float = 1e-6) -> float:
    """Sensitivity ratio between mechanism and residual components."""
    return float(s_phi / (s_r + eps))


# Backwards-compatibility helpers --------------------------------------------------------------


def invariant_gap(per_env_values: Sequence[Number]) -> float:
    return invariance_gap(per_env_values)


def worst_group(per_env_values: Sequence[Number]) -> float:
    values = _to_values(per_env_values)
    return float(values.max()) if values.size else 0.0


def mechanistic_sensitivity(sensitivities: Iterable[Number]) -> float:
    values = _to_values(list(sensitivities))
    return float(values.mean()) if values.size else 0.0


__all__ = [
    "invariance_gap",
    "worst_group_gap",
    "mechanism_sensitivity_index",
    "invariant_gap",
    "worst_group",
    "mechanistic_sensitivity",
]
