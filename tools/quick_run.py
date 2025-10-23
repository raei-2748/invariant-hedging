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
from src.core.losses import cvar_from_pnl


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
"""Minimal smoke-test runner for real anchor configurations.

The utility materialises deterministic real-market anchors and writes tagged
``pnl.csv`` and ``cvar95.json`` summaries into the canonical run directory
structure. It offers a fast way to validate data plumbing without invoking the
full training stack.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.real.anchors import AnchorSpec
from src.data.real.loader import RealAnchorLoader
from src.infra.paths import canonical_run_dir, episode_file_path
from src.infra.tags import extract_episode_tags


@dataclass(frozen=True)
class QuickRunConfig:
    """Resolved configuration payload for the quick run utility."""

    data: Mapping[str, object]
    seed: int
    experiment: str
    output_root: Path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML configuration compatible with configs/examples/real_anchors.yaml.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the run seed")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Compatibility flag; real quick runs use deterministic episodes and do not train.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional experiment name for the run directory (defaults to the config stem).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs",
        help="Root directory for generated outputs (defaults to ./runs).",
    )
    return parser.parse_args(argv)


def _load_config(path: str, seed_override: int | None, experiment: str | None, output_root: str) -> QuickRunConfig:
    cfg = OmegaConf.load(path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):  # pragma: no cover - defensive programming
        raise TypeError("Configuration must resolve to a mapping")

    data_cfg = cfg_dict.get("data", cfg_dict)
    if not isinstance(data_cfg, Mapping):
        raise TypeError("data section of configuration must be a mapping")
    data_cfg = dict(data_cfg)
    train_cfg = cfg_dict.get("train", {})
    seed = seed_override
    if seed is None:
        if isinstance(train_cfg, Mapping) and "seed" in train_cfg:
            seed = int(train_cfg["seed"])
        else:
            seed = int(data_cfg.get("seed", 0))
    data_cfg["seed"] = seed
    experiment_name = experiment or Path(path).stem
    return QuickRunConfig(
        data=data_cfg,
        seed=seed,
        experiment=experiment_name,
        output_root=Path(output_root),
    )


def _ensure_underlying_data(config: Mapping[str, object]) -> Path:
    vendor_cfg = dict(config.get("vendor", {}))
    root = Path(vendor_cfg.get("path_csv_root", "data/real"))
    symbol_cfg = dict(config.get("symbols", {}))
    symbol = symbol_cfg.get("underlying", "SPY")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{symbol}.csv"
    if path.exists():
        return path

    anchors: List[Mapping[str, object]] = list(config.get("anchors", []))
    if anchors:
        start = min(pd.Timestamp(anchor["start"]) for anchor in anchors)
        end = max(pd.Timestamp(anchor["end"]) for anchor in anchors)
    else:  # pragma: no cover - invoked only when misconfigured
        start = pd.Timestamp("2017-01-03")
        end = pd.Timestamp("2019-12-31")
    episode_cfg = dict(config.get("episode", {}))
    window_days = int(episode_cfg.get("days", 60))
    # Ensure we have at least `window_days` business days for the earliest anchor.
    if (end - start).days < window_days:
        end = start + pd.offsets.BDay(window_days)
    dates = pd.bdate_range(start, end)
    steps = np.arange(len(dates), dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "spot": 100.0 + 0.1 * steps,
            "option_price": 10.0 + 0.01 * steps,
            "implied_vol": 0.2 + 0.001 * np.sin(steps / 10.0),
        }
    )
    df.to_csv(path, index=False)
    return path


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _fieldnames(tags: List[Mapping[str, object]]) -> List[str]:
    base = [
        "episode_id",
        "start_date",
        "end_date",
        "split",
        "regime_name",
        "source",
        "symbol_root",
        "seed",
        "mean_pnl",
        "final_pnl",
    ]
    extras = sorted(
        {key for tag in tags for key in tag.keys()} - set(base)
    )
    return base + extras


def _episode_metrics(prices: np.ndarray) -> tuple[float, float]:
    returns = np.diff(prices.astype(np.float64), prepend=prices[0])
    mean_pnl = float(returns.mean())
    final_pnl = float(prices[-1] - prices[0])
    return mean_pnl, final_pnl


def _write_anchor_outputs(
    run_dir: Path,
    episodes: np.ndarray,
    tags: List[Mapping[str, object]],
    *,
    default_seed: int,
) -> None:
    csv_path = episode_file_path(run_dir, tags[0], "pnl.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(tags)
    finals: List[float] = []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode_idx, tag in enumerate(tags):
            prices = np.asarray(episodes[episode_idx], dtype=np.float64)
            mean_pnl, final_pnl = _episode_metrics(prices)
            row = dict(tag)
            row.setdefault("seed", default_seed)
            row["mean_pnl"] = mean_pnl
            row["final_pnl"] = final_pnl
            writer.writerow(row)
            finals.append(final_pnl)
    cvar_path = episode_file_path(run_dir, tags[0], "cvar95.json")
    cvar_path.parent.mkdir(parents=True, exist_ok=True)
    finals_array = np.sort(np.asarray(finals, dtype=np.float64))
    cutoff = max(1, int(np.ceil(0.05 * len(finals_array))))
    tail = finals_array[:cutoff]
    cvar95 = float(tail.mean()) if len(tail) else 0.0
    payload = {
        "cvar95": cvar95,
        "regime_name": tags[0].get("regime_name"),
        "split": tags[0].get("split"),
        "episodes": len(finals),
        "mean_final_pnl": float(finals_array.mean()) if len(finals_array) else 0.0,
    }
    cvar_path.write_text(json.dumps(payload, indent=2))


def _write_missing_anchor_placeholder(
    run_dir: Path,
    loader: RealAnchorLoader,
    anchor: AnchorSpec,
    *,
    default_seed: int,
) -> None:
    underlying = loader._load_underlying()
    mask = (underlying.index >= anchor.start) & (underlying.index <= anchor.end)
    segment = underlying.loc[mask]
    if segment.empty:
        segment = underlying
    prices = segment["spot"].to_numpy(dtype=np.float64)
    if prices.size == 0:
        prices = np.array([0.0, 0.0], dtype=np.float64)
    elif prices.size == 1:
        prices = np.concatenate([prices, prices])
    episodes = prices[np.newaxis, :]
    tag = {
        "episode_id": 0,
        "start_date": anchor.start.strftime("%Y-%m-%d"),
        "end_date": anchor.end.strftime("%Y-%m-%d"),
        "split": anchor.split,
        "regime_name": anchor.name,
        "source": getattr(loader.config, "source", "real"),
        "symbol_root": loader.symbols.underlying,
        "seed": default_seed,
        "anchor_start": anchor.start.strftime("%Y-%m-%d"),
        "anchor_end": anchor.end.strftime("%Y-%m-%d"),
    }
    _write_anchor_outputs(run_dir, episodes, [tag], default_seed=default_seed)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    quick_cfg = _load_config(
        args.config,
        seed_override=args.seed,
        experiment=args.experiment,
        output_root=args.output_root,
    )
    _ensure_underlying_data(quick_cfg.data)
    loader = RealAnchorLoader(quick_cfg.data)
    anchors = loader.load()
    if not anchors:
        raise RuntimeError("No anchors were produced from the configuration")

    timestamp = _timestamp()
    run_dir = canonical_run_dir(timestamp, quick_cfg.experiment, root=quick_cfg.output_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    anchor_lookup = {anchor.name: anchor for anchor in loader.anchors}
    for name, anchor in anchor_lookup.items():
        loaded = anchors.get(name)
        if loaded is not None:
            tag_dicts = [dict(tag) for tag in extract_episode_tags(loaded.batch)]
            if not tag_dicts:
                continue
            episodes = loaded.batch.spot.detach().cpu().numpy()
            for tag in tag_dicts:
                tag.setdefault("seed", quick_cfg.seed)
            _write_anchor_outputs(run_dir, episodes, tag_dicts, default_seed=quick_cfg.seed)
        else:
            _write_missing_anchor_placeholder(
                run_dir, loader, anchor, default_seed=quick_cfg.seed
            )

    print(f"Wrote quick run outputs to {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
"""Minimal driver for the head-only HIRM objective."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from legacy.train.loop import build_config, run_training


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Top-level configuration must be a mapping.")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/examples/hirm_minimal.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    raw = _load_config(args.config)
    train_cfg = raw.setdefault("train", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    config = build_config(raw)
    run_dir = run_training(config, base_dir=args.base_dir)
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
