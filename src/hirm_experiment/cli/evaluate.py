from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hirm_experiment.data.dataset import create_data_module
from hirm_experiment.evaluation.evaluator import EvaluationResult, Evaluator
from hirm_experiment.evaluation.metrics import aggregate_seed_metrics
from hirm_experiment.models.policy import HedgingPolicy, HedgingPolicyConfig
from hirm_experiment.training.features import FeatureBuilder
from hirm_experiment.training.utils import resolve_device, set_seed


def _load_checkpoint(path: Path, device: torch.device) -> Dict:
    state = torch.load(path, map_location=device)
    return state


def _model_from_checkpoint(state: Dict, cfg: Dict) -> HedgingPolicy:
    model_config = state.get("model_config")
    if model_config is None:
        model_config = cfg.get("model", {})
    config = HedgingPolicyConfig(
        input_dim=int(model_config.get("input_dim", cfg["model"]["input_dim"])),
        hidden_dims=list(model_config.get("hidden_dims", cfg["model"]["hidden_dims"])),
        dropout=float(model_config.get("dropout", cfg["model"].get("dropout", 0.0))),
        activation=model_config.get("activation", cfg["model"].get("activation", "relu")),
        output_dim=int(model_config.get("output_dim", cfg["model"].get("output_dim", 1))),
        bounded_output=bool(model_config.get("bounded_output", cfg["model"].get("bounded_output", True))),
        output_scale=float(model_config.get("output_scale", cfg["model"].get("output_scale", 1.0))),
    )
    model = HedgingPolicy(config)
    model.load_state_dict(state["model_state"])
    return model


def _load_seeds(evaluation_cfg: Dict, root: Path) -> List[int]:
    override = evaluation_cfg.get("override_seed")
    if override is not None:
        return [int(override)]
    seeds_file = evaluation_cfg.get("seeds_file")
    if seeds_file is None:
        return [0]
    path = Path(seeds_file)
    if not path.is_absolute():
        path = root / path
    with path.open() as f:
        return [int(line.strip()) for line in f if line.strip()]


def _aggregate_results(per_seed: Dict[int, EvaluationResult]) -> Dict[str, Dict[str, Dict[str, float]]]:
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    env_names = set()
    for result in per_seed.values():
        env_names.update(result.env_metrics.keys())
    for env in env_names:
        seed_metrics = [res.env_metrics[env] for res in per_seed.values() if env in res.env_metrics]
        aggregated.setdefault(env, {})
        aggregated[env]["metrics"] = aggregate_seed_metrics(seed_metrics)
    return aggregated


def _coverage_snapshot(per_seed: Dict[int, EvaluationResult]) -> Dict[str, Dict[str, Any]]:
    if not per_seed:
        return {}
    first = next(iter(per_seed.values()))
    return first.coverage


def _spread_snapshot(per_seed: Dict[int, EvaluationResult]) -> Dict[str, Dict[str, float]]:
    spreads: Dict[str, Dict[str, float]] = {}
    for result in per_seed.values():
        for env, curve in result.spread_sensitivity.items():
            env_curve = spreads.setdefault(env, {})
            for spread, value in curve.items():
                env_curve.setdefault(spread, []).append(value)
    averaged: Dict[str, Dict[str, float]] = {}
    for env, curve in spreads.items():
        averaged[env] = {spread: float(torch.tensor(values).mean()) for spread, values in curve.items()}
    return averaged


@hydra.main(version_base=None, config_path="../../../configs", config_name="experiment_eval")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    root = Path(get_original_cwd())
    checkpoint_path = cfg_dict["evaluation"].get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("evaluation.checkpoint_path must be provided")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    device = resolve_device(cfg_dict["training"].get("device", "auto"))
    seeds = _load_seeds(cfg_dict["evaluation"], root)
    per_seed_results: Dict[int, EvaluationResult] = {}
    for seed in seeds:
        set_seed(seed)
        data_module = create_data_module(cfg_dict["data"], seed, device)
        invariants = cfg_dict["data"]["feature_set"]["invariants"]
        spurious = cfg_dict["data"]["feature_set"].get("spurious", [])
        builder = FeatureBuilder(data_module.get_feature_stats(), invariants, spurious)
        state = _load_checkpoint(checkpoint, device)
        model = _model_from_checkpoint(state, cfg_dict).to(device)
        evaluator = Evaluator(builder, cfg_dict["evaluation"], device)
        per_seed_results[seed] = evaluator.evaluate_module(model, data_module)
    aggregates = _aggregate_results(per_seed_results)
    coverage = _coverage_snapshot(per_seed_results)
    spread = _spread_snapshot(per_seed_results)
    print("Aggregated Metrics:")
    for env, metrics in aggregates.items():
        print(f"Environment: {env}")
        for metric, summary in metrics["metrics"].items():
            print(f"  {metric}: mean={summary['mean']:.4f}, ci=({summary['ci_low']:.4f}, {summary['ci_high']:.4f})")
    if coverage:
        print("\nCoverage Snapshot:")
        for env, stats in coverage.items():
            print(f"  {env}: {stats}")
    if spread:
        print("\nSpread Sensitivity (CVaR-95):")
        for env, curve in spread.items():
            formatted = ", ".join(f"{k}: {v:.4f}" for k, v in curve.items())
            print(f"  {env}: {formatted}")


if __name__ == "__main__":
    main()
