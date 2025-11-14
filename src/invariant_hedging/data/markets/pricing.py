"""Black-Scholes pricing and Greeks utilities."""
from __future__ import annotations

import math
from typing import Literal, Tuple

import torch

NORMAL = torch.distributions.Normal(0.0, 1.0)


def _ensure_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x, dtype=torch.get_default_dtype())


def _d1_d2(
    spot, strike, rate, sigma, tau
) -> Tuple[torch.Tensor, torch.Tensor]:
    spot = _ensure_tensor(spot)
    strike = _ensure_tensor(strike)
    rate = _ensure_tensor(rate)
    sigma = torch.clamp(_ensure_tensor(sigma), min=1e-6)
    tau = torch.clamp(_ensure_tensor(tau), min=1e-8)
    log_term = torch.log(spot / strike)
    numerator = log_term + (rate + 0.5 * sigma ** 2) * tau
    denominator = sigma * torch.sqrt(tau)
    d1 = numerator / denominator
    d2 = d1 - sigma * torch.sqrt(tau)
    return d1, d2


def black_scholes_price(
    spot,
    strike,
    rate,
    sigma,
    tau,
    option_type: Literal["call", "put"] = "call",
):
    spot = _ensure_tensor(spot)
    strike = _ensure_tensor(strike)
    rate = _ensure_tensor(rate)
    sigma = _ensure_tensor(sigma)
    tau = _ensure_tensor(tau)
    intrinsic = torch.clamp(spot - strike, min=0.0) if option_type == "call" else torch.clamp(strike - spot, min=0.0)
    if torch.all(tau <= 0):
        return intrinsic
    d1, d2 = _d1_d2(spot, strike, rate, sigma, tau)
    discount = torch.exp(-rate * tau)
    if option_type == "call":
        price = spot * NORMAL.cdf(d1) - strike * discount * NORMAL.cdf(d2)
    else:
        price = strike * discount * NORMAL.cdf(-d2) - spot * NORMAL.cdf(-d1)
    return price


def black_scholes_delta(
    spot,
    strike,
    rate,
    sigma,
    tau,
    option_type: Literal["call", "put"] = "call",
):
    spot = _ensure_tensor(spot)
    strike = _ensure_tensor(strike)
    rate = _ensure_tensor(rate)
    sigma = _ensure_tensor(sigma)
    tau = _ensure_tensor(tau)
    if torch.all(tau <= 0):
        return torch.ones_like(spot) if option_type == "call" else -torch.ones_like(spot)
    d1, _ = _d1_d2(spot, strike, rate, sigma, tau)
    if option_type == "call":
        return NORMAL.cdf(d1)
    return NORMAL.cdf(d1) - 1.0


def black_scholes_gamma(spot, strike, rate, sigma, tau):
    spot = _ensure_tensor(spot)
    strike = _ensure_tensor(strike)
    rate = _ensure_tensor(rate)
    sigma = torch.clamp(_ensure_tensor(sigma), min=1e-6)
    tau = torch.clamp(_ensure_tensor(tau), min=1e-8)
    d1, _ = _d1_d2(spot, strike, rate, sigma, tau)
    pdf = torch.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return pdf / (spot * sigma * torch.sqrt(tau))


def black_scholes_vega(spot, strike, rate, sigma, tau):
    spot = _ensure_tensor(spot)
    strike = _ensure_tensor(strike)
    rate = _ensure_tensor(rate)
    sigma = torch.clamp(_ensure_tensor(sigma), min=1e-6)
    tau = torch.clamp(_ensure_tensor(tau), min=1e-8)
    d1, _ = _d1_d2(spot, strike, rate, sigma, tau)
    pdf = torch.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return spot * torch.sqrt(tau) * pdf
