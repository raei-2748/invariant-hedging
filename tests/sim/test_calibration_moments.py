"""Moment calibration tests for calm and crisis regimes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.sim.calibrators import compose_sim_recipe
from src.sim.params import SimRecipe
from src.sim.utils import (
    SampleMoments,
    aggregate_ac1,
    annualize_mean,
    annualize_variance,
    manifest_entry,
)

from .helpers import simulate_heston_qe

pytestmark = pytest.mark.not_heavy

DATA_DIR = Path("configs/sim")
RUNS_DIR = Path("runs/sim_tests")


def _summarise(recipe: SimRecipe, result, steps: int) -> SampleMoments:
    params = recipe.heston
    assert params is not None
    log_returns = result.log_returns.reshape(-1)
    # Work with realised simple returns so jump compensation does not bias the
    # drift check; the multiplicative jump component makes the log + 0.5·var
    # approximation noisier than the exact ratio of spot levels.
    simple_returns = (result.spots[:, 1:] / result.spots[:, :-1] - 1.0).reshape(-1)
    dt = 1.0 / params.year_days
    ann_mean = annualize_mean(simple_returns, params.year_days)
    ann_var = annualize_variance(log_returns, params.year_days)
    var_ac1 = aggregate_ac1(result.variances[:, :-1])
    horizon_years = steps * dt
    total_jumps = result.jump_counts.sum(axis=1)
    jump_rate = float(np.mean(total_jumps) / max(horizon_years, 1e-8))
    sigma_step = float(np.std(log_returns, ddof=0))
    tail = float(np.mean(np.abs(log_returns) > 3.0 * max(sigma_step, 1e-8)))
    return SampleMoments(mean=ann_mean, variance=ann_var, ac1=var_ac1, jump_rate=jump_rate, tail_prob=tail)


def _check_targets(recipe: SimRecipe, moments: SampleMoments, steps: int) -> None:
    params = recipe.heston
    assert params is not None
    target_mean = params.rate - params.div_yield
    mean_tolerance = max(2e-3, 3.5e-3)
    assert abs(moments.mean - target_mean) <= mean_tolerance
    target_var = params.theta
    tolerance = max(0.1 * target_var, 1e-3)
    assert abs(moments.variance - target_var) <= tolerance
    if recipe.merton.enabled:
        expected = recipe.merton.lambda_y
        allowed = max(0.2 * expected, 5e-2)
        assert abs(moments.jump_rate - expected) <= allowed
    else:
        assert moments.jump_rate <= 5e-2


def _run_moment_suite(n_paths: int, steps: int) -> dict:
    calm_recipe = compose_sim_recipe(
        "heston",
        DATA_DIR / "heston_calm.yaml",
        None,
        DATA_DIR / "liquidity_calm.yaml",
        seed=None,
    )
    calm_jumps_recipe = compose_sim_recipe(
        "heston",
        DATA_DIR / "heston_calm.yaml",
        DATA_DIR / "merton_calm.yaml",
        DATA_DIR / "liquidity_calm.yaml",
        seed=None,
    )
    crisis_recipe = compose_sim_recipe(
        "heston",
        DATA_DIR / "heston_crisis.yaml",
        None,
        DATA_DIR / "liquidity_crisis.yaml",
        seed=None,
    )
    crisis_jumps_recipe = compose_sim_recipe(
        "heston",
        DATA_DIR / "heston_crisis.yaml",
        DATA_DIR / "merton_crisis.yaml",
        DATA_DIR / "liquidity_crisis.yaml",
        seed=None,
    )

    scenarios = {
        "calm": calm_recipe,
        "calm_with_jumps": calm_jumps_recipe,
        "crisis": crisis_recipe,
        "crisis_with_jumps": crisis_jumps_recipe,
    }

    manifest = {"n_paths": n_paths, "steps": steps, "scenarios": []}
    summary = {}

    for label, recipe in scenarios.items():
        result = simulate_heston_qe(recipe, n_paths, steps)
        moments = _summarise(recipe, result, steps)
        _check_targets(recipe, moments, steps)
        manifest["scenarios"].append(manifest_entry(label, moments))
        summary[label] = moments

    calm_var = summary["calm"].variance
    crisis_var = summary["crisis"].variance
    calm_jump = summary["calm_with_jumps"].jump_rate
    crisis_jump = summary["crisis_with_jumps"].jump_rate
    calm_tail = summary["calm_with_jumps"].tail_prob
    crisis_tail = summary["crisis_with_jumps"].tail_prob
    calm_ac = summary["calm"].ac1
    crisis_ac = summary["crisis"].ac1

    assert crisis_var > calm_var
    assert crisis_jump >= calm_jump
    assert crisis_tail > calm_tail
    assert crisis_ac >= calm_ac - 5e-2

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RUNS_DIR / "sim_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return summary


def test_moments_calibration_ci_suite():
    _run_moment_suite(n_paths=5_000, steps=252)


@pytest.mark.heavy
def test_moments_calibration_heavy_suite():
    _run_moment_suite(n_paths=50_000, steps=252)
