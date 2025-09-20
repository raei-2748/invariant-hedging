import math
from typing import Tuple

import numpy as np
from scipy.special import erf


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * x * x)


def call_price(spot: np.ndarray, strike: np.ndarray, tau: np.ndarray, sigma: np.ndarray, r: float = 0.0) -> np.ndarray:
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return spot * _norm_cdf(d1) - strike * np.exp(-r * tau) * _norm_cdf(d2)


def call_delta(spot: np.ndarray, strike: np.ndarray, tau: np.ndarray, sigma: np.ndarray, r: float = 0.0) -> np.ndarray:
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return _norm_cdf(d1)


def call_gamma(spot: np.ndarray, strike: np.ndarray, tau: np.ndarray, sigma: np.ndarray, r: float = 0.0) -> np.ndarray:
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return _norm_pdf(d1) / (spot * sigma * np.sqrt(tau))


def call_theta(spot: np.ndarray, strike: np.ndarray, tau: np.ndarray, sigma: np.ndarray, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    first = -spot * _norm_pdf(d1) * sigma / (2.0 * np.sqrt(tau))
    second = q * spot * _norm_cdf(d1)
    third = -r * strike * np.exp(-r * tau) * _norm_cdf(d2)
    return first + second + third


def greeks(spot: np.ndarray, tau: np.ndarray, sigma_imp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    strike = spot
    delta = call_delta(spot, strike, tau, sigma_imp)
    gamma = call_gamma(spot, strike, tau, sigma_imp)
    theta = call_theta(spot, strike, tau, sigma_imp)
    return delta, gamma, theta
