import math

import pytest
import torch

from src.markets.pricing import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
)


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def test_black_scholes_price_matches_reference():
    spot = 110.0
    strike = 100.0
    rate = 0.03
    sigma = 0.25
    tau = 0.75

    price = black_scholes_price(
        torch.tensor(spot),
        torch.tensor(strike),
        torch.tensor(rate),
        torch.tensor(sigma),
        torch.tensor(tau),
        option_type="call",
    )

    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    expected = spot * _norm_cdf(d1) - strike * math.exp(-rate * tau) * _norm_cdf(d2)

    assert price.item() == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_black_scholes_delta_matches_autograd():
    spot = torch.tensor(100.0, requires_grad=True)
    strike = torch.tensor(100.0)
    rate = torch.tensor(0.01)
    sigma = torch.tensor(0.2)
    tau = torch.tensor(0.5)
    price = black_scholes_price(spot, strike, rate, sigma, tau)
    price.backward()
    analytic = black_scholes_delta(spot.detach(), strike, rate, sigma, tau)
    assert torch.allclose(spot.grad, analytic, atol=1e-5)


def test_black_scholes_gamma_matches_autograd():
    spot = torch.tensor(100.0, requires_grad=True)
    strike = torch.tensor(95.0)
    rate = torch.tensor(0.01)
    sigma = torch.tensor(0.25)
    tau = torch.tensor(1.0)
    price = black_scholes_price(spot, strike, rate, sigma, tau)
    delta = torch.autograd.grad(price, spot, create_graph=True)[0]
    gamma_auto = torch.autograd.grad(delta, spot)[0]
    analytic = black_scholes_gamma(spot.detach(), strike, rate, sigma, tau)
    assert torch.allclose(gamma_auto, analytic, atol=1e-5)
