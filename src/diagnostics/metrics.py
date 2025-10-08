"""Diagnostic metrics for invariant hedging experiments."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

NumberCollection = Sequence[float] | Mapping[str, float]


def _values(data: NumberCollection) -> list[float]:
    if isinstance(data, Mapping):
        return [float(v) for v in data.values()]
    return [float(v) for v in data]


def invariance_gap(train_env_risks: NumberCollection) -> float:
    """Return the invariance gap (IG) as max-min across training regimes."""

    values = _values(train_env_risks)
    if not values:
        return 0.0
    return float(max(values) - min(values))


# Backwards-compatible alias used in legacy logging code.
invariant_gap = invariance_gap
IG = invariance_gap


def worst_group_gap(train_env_risks: NumberCollection, test_env_risks: NumberCollection) -> float:
    """Return the worst-group (WG) generalisation gap."""

    train_values = _values(train_env_risks)
    test_values = _values(test_env_risks)
    if not train_values or not test_values:
        return 0.0
    return float(max(test_values) - max(train_values))


# Backwards-compatible helper for existing code paths.
def worst_group(per_env_values: NumberCollection) -> float:
    values = _values(per_env_values)
    if not values:
        return 0.0
    return float(max(values))


WG = worst_group_gap


def mechanism_sensitivity_index(s_phi: float, s_r: float, eps: float = 1e-6) -> float:
    """Return the mechanism sensitivity index (MSI)."""

    return float(s_phi / (s_r + eps))


# Alias retained for older naming.
def mechanistic_sensitivity(sensitivities: Iterable[float]) -> float:
    values = list(float(v) for v in sensitivities)
    if not values:
        return 0.0
    # Treat mean as proxy when individual components are provided.
    mean_phi = sum(values) / len(values)
    return mechanism_sensitivity_index(mean_phi, 1.0)


MSI = mechanism_sensitivity_index


__all__ = [
    "IG",
    "MSI",
    "WG",
    "invariance_gap",
    "invariant_gap",
    "mechanism_sensitivity_index",
    "mechanistic_sensitivity",
    "worst_group",
    "worst_group_gap",
]
