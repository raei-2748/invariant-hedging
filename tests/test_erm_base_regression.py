from __future__ import annotations

import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CSV = PROJECT_ROOT / "outputs" / "_baseline_erm_base" / "artifacts" / "ERM_base_crisis.csv"

EXPECTED_MEAN = {
    "test/crisis_cvar": 395.1852,
    "test/crisis_mean_pnl": -7.5203,
    "test/crisis_turnover": 10.7796,
    "test/crisis_sharpe": -0.0788,
    "test/crisis_max_drawdown": 196.0665,
}

EXPECTED_MARGIN = {
    "test/crisis_cvar": 119.4109,
    "test/crisis_mean_pnl": 11.6201,
    "test/crisis_turnover": 1.1327,
    "test/crisis_sharpe": 0.3679,
    "test/crisis_max_drawdown": 15.6910,
}

EXPECTED_SEEDS = {0, 1, 2, 3, 4}


def _load_rows():
    with BASELINE_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
    return rows


def _stat(values):
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    margin = 1.96 * std / math.sqrt(n)
    return mean, margin


def test_erm_base_crisis_metrics_stable():
    assert BASELINE_CSV.exists(), f"Baseline CSV missing: {BASELINE_CSV}"

    rows = _load_rows()
    assert rows, "Baseline CSV is empty"

    seeds = {int(row["seed"]) for row in rows}
    assert seeds == EXPECTED_SEEDS, f"Unexpected seeds: {sorted(seeds)}"

    metrics = list(EXPECTED_MEAN.keys())
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        mean, margin = _stat(values)
        expected_mean = EXPECTED_MEAN[metric]
        expected_margin = EXPECTED_MARGIN[metric]
        assert math.isclose(
            mean, expected_mean, rel_tol=0, abs_tol=1e-4
        ), f"Mean for {metric} changed: {mean:.4f} vs {expected_mean:.4f}"
        assert (
            margin <= expected_margin + 1e-4
        ), f"Spread for {metric} widened: {margin:.4f} vs {expected_margin:.4f}"
