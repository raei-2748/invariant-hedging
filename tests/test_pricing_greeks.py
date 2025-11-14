import torch

from invariant_hedging.data.markets.pricing import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_price,
)


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
