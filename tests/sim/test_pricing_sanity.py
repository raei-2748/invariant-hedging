"""Pricing sanity checks for calm versus crisis regimes."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.modules.markets.pricing import black_scholes_delta, black_scholes_price
from src.modules.sim.calibrators import compose_sim_recipe

from .helpers import simulate_heston_qe

pytestmark = pytest.mark.not_heavy

DATA_DIR = Path("configs/sim")


def _implied_vol(price: float, spot: float, strike: float, rate: float, tau: float) -> float:
    if price <= max(spot - strike * math.exp(-rate * tau), 0.0):
        return 1e-4
    low, high = 1e-4, 5.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        est = float(black_scholes_price(spot, strike, rate, mid, tau))
        if est > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _find_put_strike(spot: float, rate: float, div: float, vol: float, tau: float, target_delta: float = -0.25) -> float:
    low, high = 0.1 * spot, 2.0 * spot
    for _ in range(80):
        mid = 0.5 * (low + high)
        delta = float(black_scholes_delta(spot, mid, rate - div, vol, tau, option_type="put"))
        if delta < target_delta:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _monte_carlo_price(recipe, maturity_days: int, strike: float, n_paths: int = 3000) -> tuple[float, float]:
    result = simulate_heston_qe(recipe, n_paths, maturity_days, seed_offset=maturity_days)
    params = recipe.heston
    assert params is not None
    final_spot = result.spots[:, -1]
    payoff = np.maximum(final_spot - strike, 0.0)
    tau = maturity_days / params.year_days
    discount = math.exp(-params.rate * tau)
    price = float(np.mean(payoff) * discount)
    iv = _implied_vol(price, params.s0, strike, params.rate, tau)
    return price, iv


def _sabr_implied_vol(beta: float, alpha: float, rho: float, nu: float, fwd: float, strike: float, tau: float) -> float:
    if alpha <= 0.0:
        return 0.0
    if abs(fwd - strike) < 1e-12:
        return alpha * (1 + ((2 - 3 * rho ** 2) * nu ** 2 * tau) / 24.0)
    log_fk = math.log(fwd / strike)
    z = (nu / alpha) * log_fk
    x_z = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    adjustment = 1.0 + ((2 - 3 * rho ** 2) * nu ** 2 * tau) / 24.0
    return alpha * (z / x_z) * adjustment


def test_pricing_monotonicity_calm_to_crisis():
    calm = compose_sim_recipe(
        "heston",
        DATA_DIR / "heston_calm.yaml",
        None,
        DATA_DIR / "liquidity_calm.yaml",
        seed=None,
    )
    crisis = compose_sim_recipe(
        "heston",
        DATA_DIR / "heston_crisis.yaml",
        DATA_DIR / "merton_crisis.yaml",
        DATA_DIR / "liquidity_crisis.yaml",
        seed=None,
    )

    params = calm.heston
    assert params is not None
    maturities = [21, 63, 126]
    strikes = [params.s0]
    base_vol = math.sqrt(max(params.theta, 1e-8))
    strikes.append(
        _find_put_strike(params.s0, params.rate, params.div_yield, base_vol, maturities[0] / params.year_days)
    )

    for strike in strikes:
        calm_prices = []
        crisis_prices = []
        for maturity in maturities:
            price_calm, iv_calm = _monte_carlo_price(calm, maturity, strike)
            price_crisis, iv_crisis = _monte_carlo_price(crisis, maturity, strike)
            calm_prices.append(price_calm)
            crisis_prices.append(price_crisis)
            assert iv_crisis >= iv_calm - 5e-3
            assert price_crisis >= price_calm - 5e-3
        for earlier, later in zip(calm_prices, calm_prices[1:]):
            assert later >= earlier - 1e-3
        for earlier, later in zip(crisis_prices, crisis_prices[1:]):
            assert later >= earlier - 1e-3


def test_sabr_crisis_widens_smile():
    calm = compose_sim_recipe(
        "sabr",
        DATA_DIR / "sabr_calm.yaml",
        None,
        DATA_DIR / "liquidity_calm.yaml",
        seed=None,
    )
    crisis = compose_sim_recipe(
        "sabr",
        DATA_DIR / "sabr_crisis.yaml",
        None,
        DATA_DIR / "liquidity_crisis.yaml",
        seed=None,
    )

    calm_params = calm.sabr
    crisis_params = crisis.sabr
    assert calm_params is not None and crisis_params is not None
    tau = 63 / calm_params.year_days
    fwd_calm = calm_params.s0 * math.exp((calm_params.rate - calm_params.div_yield) * tau)
    fwd_crisis = crisis_params.s0 * math.exp((crisis_params.rate - crisis_params.div_yield) * tau)
    strikes = [calm_params.s0, calm_params.s0 * 0.9]

    for strike in strikes:
        iv_calm = _sabr_implied_vol(
            calm_params.beta, calm_params.alpha, calm_params.rho, calm_params.nu, fwd_calm, strike, tau
        )
        iv_crisis = _sabr_implied_vol(
            crisis_params.beta, crisis_params.alpha, crisis_params.rho, crisis_params.nu, fwd_crisis, strike, tau
        )
        price_calm = float(black_scholes_price(calm_params.s0, strike, calm_params.rate, iv_calm, tau))
        price_crisis = float(black_scholes_price(crisis_params.s0, strike, crisis_params.rate, iv_crisis, tau))
        assert iv_crisis >= iv_calm
        assert price_crisis >= price_calm
