"""Quick integration runner for the crisis stress simulator."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
import yaml
import torch

from src.data.sim import HestonParams, simulate_heston
from src.data.sim.liquidity import LiquidityStressConfig, liquidity_costs
from src.data.sim.merton import JumpSummary, overlay_merton_jumps
from src.envs.registry import SyntheticRegimeRegistry
from src.infra.io import write_sim_params_json, write_stress_summary_json
from src.objectives.cvar import cvar_from_pnl


def _extract_heston(cfg: Mapping[str, object]) -> HestonParams:
    params = cfg.get("data", {}).get("heston", {})  # type: ignore[arg-type]
    if not isinstance(params, Mapping):
        raise TypeError("config.data.heston must be a mapping")
    return HestonParams(
        s0=float(params.get("s0", 100.0)),
        v0=float(params.get("v0", 0.04)),
        mu=float(params.get("mu", 0.0)),
        kappa=float(params.get("kappa", 1.5)),
        theta=float(params.get("theta", 0.04)),
        sigma=float(params.get("sigma", 0.5)),
        rho=float(params.get("rho", -0.7)),
        dt_days=float(params.get("dt_days", 1.0)),
        days=int(params.get("days", 60)),
    )


def _extract_liquidity(cfg: Mapping[str, object]) -> LiquidityStressConfig:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    stress_cfg = data_cfg.get("stress", {}) if isinstance(data_cfg, Mapping) else {}
    liq_cfg = stress_cfg.get("liquidity", {}) if isinstance(stress_cfg, Mapping) else {}
    if not liq_cfg:
        return LiquidityStressConfig(0.0, 0.0, 0.0, 0.0)
    return LiquidityStressConfig(
        base_spread_bps=float(liq_cfg.get("base_spread_bps", 0.0)),
        vol_slope_bps=float(liq_cfg.get("vol_slope_bps", 0.0)),
        size_slope_bps=float(liq_cfg.get("size_slope_bps", 0.0)),
        slippage_coeff=float(liq_cfg.get("slippage_coeff", 0.0)),
    )


def _jump_cfg(cfg: Mapping[str, object]) -> Mapping[str, object]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    stress_cfg = data_cfg.get("stress", {}) if isinstance(data_cfg, Mapping) else {}
    return stress_cfg.get("jump", {}) if isinstance(stress_cfg, Mapping) else {}


def _episode_settings(cfg: Mapping[str, object]) -> Dict[str, int]:
    data_cfg = cfg.get("data", {}) if isinstance(cfg, Mapping) else {}
    episode_cfg = data_cfg.get("episode", {}) if isinstance(data_cfg, Mapping) else {}
    count = int(episode_cfg.get("count_per_regime", 32))
    seed = int(episode_cfg.get("seed", 7))
    return {"count": count, "seed": seed}


def _split_regimes(cfg: Mapping[str, object]) -> Dict[str, Iterable[str]]:
    data_cfg = cfg.get("train", {}) if isinstance(cfg, Mapping) else {}
    return {
        "train": data_cfg.get("envs", []),
        "val": data_cfg.get("val_envs", []),
        "test": data_cfg.get("test_envs", []),
    }


def _run_dir(cfg: Mapping[str, object]) -> Path:
    expname = cfg.get("name", "sim_crisis") if isinstance(cfg, Mapping) else "sim_crisis"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{timestamp}_{expname}"


def _liquidity_inputs(path: pd.DataFrame) -> Dict[str, np.ndarray]:
    returns = path["log_return"].to_numpy()[1:]
    price = path["spot"].to_numpy()
    price_change = np.diff(price, prepend=price[0])
    trade = returns
    variance = path["variance"].to_numpy()[1:]
    return {
        "variance": variance,
        "trade": trade,
        "price_change": price_change[1:],
    }


def _apply_liquidity(
    path: pd.DataFrame,
    config: LiquidityStressConfig,
    enabled: bool,
    notional: float,
    aggregates: Dict[str, List[float]],
) -> float:
    if not enabled:
        return 0.0
    inputs = _liquidity_inputs(path)
    costs, summary = liquidity_costs(
        inputs["variance"],
        inputs["trade"],
        inputs["price_change"],
        notional=notional,
        config=config,
    )
    aggregates["mean_spread_bps"].append(summary["mean_spread_bps"])
    aggregates["mean_slippage"].append(summary["mean_slippage"])
    aggregates["turnover"].append(summary["turnover"])
    return float(np.sum(costs))


def _apply_jump(
    path: pd.DataFrame,
    cfg: Mapping[str, object],
    enabled: bool,
    seed: int,
) -> tuple[pd.DataFrame, JumpSummary]:
    if not enabled:
        result = path.copy()
        result["jump_size"] = 0.0
        return result, JumpSummary(0, 0.0, 0.0)
    lam = float(cfg.get("lam", 0.0))
    mu_j = float(cfg.get("mu_j", 0.0))
    sigma_j = float(cfg.get("sigma_j", 0.1))
    stressed, summary = overlay_merton_jumps(path, lam=lam, mu_j=mu_j, sigma_j=sigma_j, seed_offset=seed)
    return stressed, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to sim crisis yaml config")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    heston_params = _extract_heston(cfg)
    liquidity_cfg = _extract_liquidity(cfg)
    jump_cfg = _jump_cfg(cfg)
    registry = SyntheticRegimeRegistry(cfg)
    episode_cfg = _episode_settings(cfg)
    split_regimes = _split_regimes(cfg)

    run_dir = _run_dir(cfg)

    for seed_idx in range(args.seeds):
        seed = episode_cfg["seed"] + seed_idx
        for split, regimes in split_regimes.items():
            for regime in regimes:
                spec = registry.get(regime)
                episode_pnls: List[float] = []
                aggregates: Dict[str, List[float]] = defaultdict(list)
                jump_count_total = 0
                jump_sum = 0.0
                jump_sq_sum = 0.0
                episodes_processed = 0
                for episode in range(episode_cfg["count"]):
                    episode_seed = seed + spec.seed_offset + episode
                    path = simulate_heston(heston_params, seed=episode_seed)
                    episodes_processed += 1
                    stressed, jump_summary = _apply_jump(
                        path, jump_cfg, spec.stress_jump, episode_seed
                    )
                    if jump_summary.count > 0:
                        jump_count_total += jump_summary.count
                        jump_sum += jump_summary.mean * jump_summary.count
                        jump_sq_sum += (
                            (jump_summary.std ** 2 + jump_summary.mean**2)
                            * jump_summary.count
                        )
                    liquidity_cost = _apply_liquidity(
                        stressed,
                        liquidity_cfg,
                        spec.stress_liquidity,
                        notional=heston_params.s0,
                        aggregates=aggregates,
                    )
                    pnl = float(stressed["spot"].iloc[-1] - stressed["spot"].iloc[0] - liquidity_cost)
                    episode_pnls.append(pnl)
                regime_dir = run_dir / "seeds" / str(seed) / split / regime
                regime_dir.mkdir(parents=True, exist_ok=True)
                pnl_df = pd.DataFrame({"episode": np.arange(len(episode_pnls)), "pnl": episode_pnls})
                pnl_df.to_csv(regime_dir / "pnl.csv", index=False)
                pnl_tensor = torch.tensor(episode_pnls, dtype=torch.float32)
                cvar = float(cvar_from_pnl(pnl_tensor, alpha=0.95))
                with open(regime_dir / "cvar95.json", "w", encoding="utf-8") as handle:
                    json.dump({"cvar95": cvar}, handle, indent=2)
                sim_params = {
                    "heston": asdict(heston_params),
                    "jump": dict(jump_cfg) | {"enabled": spec.stress_jump},
                    "liquidity": liquidity_cfg.to_dict() | {"enabled": spec.stress_liquidity},
                    "seed": seed,
                    "episodes": episode_cfg["count"],
                }
                write_sim_params_json(regime_dir / "sim_params.json", sim_params)
                mean_spread = float(np.mean(aggregates["mean_spread_bps"]) if aggregates["mean_spread_bps"] else 0.0)
                mean_slip = float(np.mean(aggregates["mean_slippage"]) if aggregates["mean_slippage"] else 0.0)
                turnover = float(np.mean(aggregates["turnover"]) if aggregates["turnover"] else 0.0)
                jump_count = int(jump_count_total)
                jump_freq = float(jump_count_total / max(episodes_processed, 1))
                if jump_count_total > 0:
                    jump_mean = float(jump_sum / jump_count_total)
                    second_moment = jump_sq_sum / jump_count_total
                    jump_var = max(second_moment - jump_mean**2, 0.0)
                    jump_std = float(math.sqrt(jump_var))
                else:
                    jump_mean = 0.0
                    jump_std = 0.0
                stress_summary = {
                    "jump_count": jump_count,
                    "jump_frequency": jump_freq,
                    "jump_mean": jump_mean,
                    "jump_std": jump_std,
                    "mean_spread_bps": mean_spread,
                    "mean_slippage": mean_slip,
                    "turnover": turnover,
                }
                write_stress_summary_json(regime_dir / "stress_summary.json", stress_summary)

    print(f"wrote synthetic runs to {run_dir}")


if __name__ == "__main__":
    main()
