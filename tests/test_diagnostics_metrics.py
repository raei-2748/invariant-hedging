import math

import pytest

from src.diagnostics import metrics as diag_metrics


def test_invariant_gap_zero_when_values_identical():
    assert diag_metrics.invariant_gap([3.0, 3.0, 3.0]) == pytest.approx(0.0)
    assert diag_metrics.invariant_gap([1.0]) == pytest.approx(0.0)


def test_invariant_gap_matches_population_standard_deviation():
    values = [1.0, 2.0, 4.0]
    mean = sum(values) / len(values)
    expected = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    assert diag_metrics.invariant_gap(values) == pytest.approx(expected, rel=1e-7, abs=1e-7)


def test_worst_group_handles_empty_sequences():
    assert diag_metrics.worst_group([]) == 0.0
    assert diag_metrics.worst_group([1.5, -2.0, 0.5]) == pytest.approx(1.5)


def test_mechanistic_sensitivity_returns_mean_value():
    assert diag_metrics.mechanistic_sensitivity([]) == 0.0
    assert diag_metrics.mechanistic_sensitivity([2.0, 4.0, 6.0]) == pytest.approx(4.0)
