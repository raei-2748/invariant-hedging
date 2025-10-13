"""Parameter containers for synthetic simulation recipes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HestonParams:
    """Heston stochastic volatility parameters."""

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    s0: float = 100.0
    rate: float = 0.0
    div_yield: float = 0.0
    year_days: int = 252
    scheme: str = "QE"
    seed: int = 123


@dataclass(frozen=True)
class MertonParams:
    """Merton jump diffusion overlay parameters."""

    lambda_y: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0
    enabled: bool = False


@dataclass(frozen=True)
class LiquidityParams:
    """Liquidity overlay parameters."""

    base_spread_bps: float = 0.0
    alpha_var_link: float = 0.0
    impact_multiplier: float = 1.0
    enabled: bool = False

    def spread_bps(self, variance: float) -> float:
        """Return the spread in basis points for the provided instantaneous variance."""

        variance = max(float(variance), 0.0)
        if not self.enabled:
            return float(self.base_spread_bps)
        return float(self.base_spread_bps) * (1.0 + float(self.alpha_var_link) * variance)


@dataclass(frozen=True)
class SABRParams:
    """SABR volatility smile parameters."""

    beta: float
    alpha: float
    rho: float
    nu: float
    s0: float = 100.0
    rate: float = 0.0
    div_yield: float = 0.0
    year_days: int = 252
    seed: int = 123


@dataclass(frozen=True)
class SimRecipe:
    """Composable simulation recipe with base diffusion, jumps and liquidity."""

    base_model: str
    heston: Optional[HestonParams]
    sabr: Optional[SABRParams]
    merton: MertonParams
    liquidity: LiquidityParams
    seed: int
