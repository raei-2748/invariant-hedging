"""Calibration utilities for assembling simulation recipes."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .params import HestonParams, LiquidityParams, MertonParams, SABRParams, SimRecipe


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and return a dictionary."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of YAML '{path}'")
    return data


def _require_keys(config: Dict[str, Any], path: str | Path, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing required keys {missing} in '{path}'")


def _validate_rho(rho: float, *, path: str | Path) -> None:
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"Correlation rho must be in (-1, 1); received {rho!r} for '{path}'")


def _validate_non_negative(value: float, *, name: str, path: str | Path) -> None:
    if value < 0.0:
        raise ValueError(f"Parameter '{name}' must be non-negative; received {value!r} for '{path}'")


def load_heston(path: str | Path) -> HestonParams:
    config = load_yaml(path)
    if config.get("base_model") != "heston":
        raise ValueError(f"'{path}' does not describe a Heston configuration")
    _require_keys(config, path, ("kappa", "theta", "xi", "rho", "v0"))
    for key in ("kappa", "theta", "xi", "v0"):
        _validate_non_negative(float(config[key]), name=key, path=path)
    _validate_rho(float(config["rho"]), path=path)
    seed = int(config.get("seed", 123))
    return HestonParams(
        kappa=float(config["kappa"]),
        theta=float(config["theta"]),
        xi=float(config["xi"]),
        rho=float(config["rho"]),
        v0=float(config["v0"]),
        s0=float(config.get("s0", 100.0)),
        rate=float(config.get("rate", 0.0)),
        div_yield=float(config.get("div_yield", 0.0)),
        year_days=int(config.get("year_days", 252)),
        scheme=str(config.get("scheme", "QE")),
        seed=seed,
    )


def load_sabr(path: str | Path) -> SABRParams:
    config = load_yaml(path)
    if config.get("base_model") != "sabr":
        raise ValueError(f"'{path}' does not describe a SABR configuration")
    _require_keys(config, path, ("alpha", "rho", "nu"))
    _validate_non_negative(float(config.get("beta", 1.0)), name="beta", path=path)
    _validate_non_negative(float(config["alpha"]), name="alpha", path=path)
    _validate_non_negative(float(config["nu"]), name="nu", path=path)
    _validate_rho(float(config["rho"]), path=path)
    seed = int(config.get("seed", 123))
    return SABRParams(
        beta=float(config.get("beta", 1.0)),
        alpha=float(config["alpha"]),
        rho=float(config["rho"]),
        nu=float(config["nu"]),
        s0=float(config.get("s0", 100.0)),
        rate=float(config.get("rate", 0.0)),
        div_yield=float(config.get("div_yield", 0.0)),
        year_days=int(config.get("year_days", 252)),
        seed=seed,
    )


def load_merton(path: str | Path) -> MertonParams:
    config = load_yaml(path)
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return MertonParams(enabled=False)
    _require_keys(config, path, ("lambda_y", "mu_j", "sigma_j"))
    _validate_non_negative(float(config["lambda_y"]), name="lambda_y", path=path)
    _validate_non_negative(float(config["sigma_j"]), name="sigma_j", path=path)
    return MertonParams(
        lambda_y=float(config["lambda_y"]),
        mu_j=float(config["mu_j"]),
        sigma_j=float(config["sigma_j"]),
        enabled=True,
    )


def load_liquidity(path: str | Path) -> LiquidityParams:
    config = load_yaml(path)
    enabled = bool(config.get("enabled", False))
    base_spread = float(config.get("base_spread_bps", 0.0))
    alpha = float(config.get("alpha_var_link", 0.0))
    impact = float(config.get("impact_multiplier", 1.0))
    _validate_non_negative(base_spread, name="base_spread_bps", path=path)
    _validate_non_negative(alpha, name="alpha_var_link", path=path)
    _validate_non_negative(impact, name="impact_multiplier", path=path)
    return LiquidityParams(
        base_spread_bps=base_spread,
        alpha_var_link=alpha,
        impact_multiplier=impact,
        enabled=enabled,
    )


def compose_sim_recipe(
    base_model: str,
    base_cfg: str | Path,
    jumps_cfg: Optional[str | Path],
    liq_cfg: Optional[str | Path],
    seed: Optional[int],
) -> SimRecipe:
    """Compose a simulation recipe from base, jump and liquidity configs."""

    base_model = base_model.lower()
    if base_model not in {"heston", "sabr"}:
        raise ValueError(f"Unknown base model '{base_model}'")

    heston_params: Optional[HestonParams] = None
    sabr_params: Optional[SABRParams] = None
    if base_model == "heston":
        heston_params = load_heston(base_cfg)
        sabr_params = None
    else:
        sabr_params = load_sabr(base_cfg)
        heston_params = None

    if base_model == "heston" and heston_params is None:
        raise ValueError("Failed to load Heston parameters")
    if base_model == "sabr" and sabr_params is None:
        raise ValueError("Failed to load SABR parameters")

    if jumps_cfg is not None:
        merton_params = load_merton(jumps_cfg)
    else:
        merton_params = MertonParams(enabled=False)

    if liq_cfg is not None:
        liquidity_params = load_liquidity(liq_cfg)
    else:
        liquidity_params = LiquidityParams(enabled=False)

    base_seed = seed
    if base_model == "heston" and heston_params is not None and base_seed is None:
        base_seed = heston_params.seed
    if base_model == "sabr" and sabr_params is not None and base_seed is None:
        base_seed = sabr_params.seed
    if base_seed is None:
        raise ValueError("Seed could not be determined for simulation recipe")
    base_seed = int(base_seed)

    if base_model == "sabr" and merton_params.enabled:
        raise ValueError("Merton jumps cannot be combined with SABR base model in PR-02")

    return SimRecipe(
        base_model=base_model,
        heston=heston_params,
        sabr=sabr_params,
        merton=merton_params,
        liquidity=liquidity_params,
        seed=base_seed,
    )
