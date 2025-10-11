"""Helper routines for simulation calibration tests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sim.params import SimRecipe
from src.sim.utils import make_rng


@dataclass
class SimulationResult:
    spots: np.ndarray
    variances: np.ndarray
    log_returns: np.ndarray
    jump_counts: np.ndarray


def simulate_heston_qe(
    recipe: SimRecipe,
    n_paths: int,
    steps: int,
    *,
    seed_offset: int = 0,
) -> SimulationResult:
    if recipe.base_model != "heston" or recipe.heston is None:
        raise ValueError("Heston simulation requested but recipe is not Heston-based")

    params = recipe.heston
    rng = make_rng(recipe.seed + seed_offset)
    dt = 1.0 / float(params.year_days)
    sqrt_dt = np.sqrt(dt)
    kappa = max(params.kappa, 1e-8)
    theta = max(params.theta, 0.0)
    xi = max(params.xi, 1e-8)
    rho = np.clip(params.rho, -0.999, 0.999)

    spots = np.empty((n_paths, steps + 1), dtype=np.float64)
    variances = np.empty_like(spots)
    log_returns = np.empty((n_paths, steps), dtype=np.float64)
    jump_counts = np.zeros((n_paths, steps), dtype=np.int64)

    spots[:, 0] = params.s0
    variances[:, 0] = max(params.v0, 1e-8)
    log_spot = np.full(n_paths, np.log(params.s0), dtype=np.float64)
    jump_compensator = 0.0
    if recipe.merton.enabled:
        jump_compensator = recipe.merton.lambda_y * (
            np.exp(recipe.merton.mu_j + 0.5 * recipe.merton.sigma_j ** 2) - 1.0
        )

    for t in range(steps):
        vt = np.maximum(variances[:, t], 0.0)
        z_v = rng.standard_normal(n_paths)
        z_s = rng.standard_normal(n_paths)
        z_s = rho * z_v + np.sqrt(max(1.0 - rho ** 2, 1e-12)) * z_s

        v_next = vt + kappa * (theta - vt) * dt + xi * np.sqrt(vt) * sqrt_dt * z_v
        v_next += 0.25 * (xi ** 2) * (z_v ** 2 - 1.0) * dt
        v_next = np.clip(v_next, 1e-8, None)

        drift = (params.rate - params.div_yield - 0.5 * vt) * dt
        if jump_compensator:
            drift -= jump_compensator * dt
        diffusion = np.sqrt(vt) * sqrt_dt * z_s
        jump_component = np.zeros(n_paths, dtype=np.float64)
        if recipe.merton.enabled:
            lambda_dt = recipe.merton.lambda_y * dt
            counts = rng.poisson(lambda_dt, size=n_paths)
            jump_counts[:, t] = counts
            if np.any(counts > 0):
                means = counts * recipe.merton.mu_j
                stds = np.sqrt(counts) * recipe.merton.sigma_j
                jump_component = rng.normal(loc=means, scale=stds)
        log_increment = drift + diffusion + jump_component
        log_returns[:, t] = log_increment
        log_spot = log_spot + log_increment
        spots[:, t + 1] = np.exp(log_spot)
        variances[:, t + 1] = v_next

    return SimulationResult(spots=spots, variances=variances, log_returns=log_returns, jump_counts=jump_counts)
