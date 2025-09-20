from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from hirm_experiment.data.dataset import EpisodeBatch, RegimeDataset
from hirm_experiment.evaluation.metrics import compute_metrics
from hirm_experiment.training.features import FeatureBuilder
from hirm_experiment.training.pnl import compute_pnl, rollout_policy


@dataclass
class EvaluationResult:
    env_metrics: Dict[str, Dict[str, float]]
    spread_sensitivity: Dict[str, Dict[str, float]]
    coverage: Dict[str, Dict[str, Any]]


class Evaluator:
    def __init__(self, feature_builder: FeatureBuilder, config: Dict, device: torch.device) -> None:
        self.builder = feature_builder
        self.config = config
        self.device = device
        self.alpha = float(config.get("confidence", 0.95))
        self.spreads = list(config.get("stress_tests", {}).get("spreads", []))

    def evaluate_dataset(self, model: torch.nn.Module, dataset: RegimeDataset) -> tuple[Dict[str, float], Dict[str, float]]:
        batch = dataset.full(self.device)
        with torch.no_grad():
            outputs = rollout_policy(model, batch, self.builder)
        metrics = compute_metrics(outputs, alpha=self.alpha)
        sensitivity = self._spread_curve(outputs.actions, batch)
        return metrics, sensitivity

    def coverage_stats(self, dataset: RegimeDataset) -> Dict[str, Any]:
        realized = float(dataset.realized_vol.mean().item())
        tx_linear = float(dataset.tx_linear.mean().item())
        tx_quadratic = float(dataset.tx_quadratic.mean().item())
        stats: Dict[str, Any] = {
            "episodes": float(dataset.episodes),
            "horizon": float(dataset.horizon),
            "avg_realized_vol": realized,
            "tx_linear": tx_linear,
            "tx_quadratic": tx_quadratic,
        }
        metadata = dataset.metadata
        stats["sample_origin"] = metadata.get("source", "synthetic")
        for field in [
            "window_name",
            "window_type",
            "window_start",
            "window_end",
            "episode_count",
            "seed",
            "base_env",
            "variant",
        ]:
            value = metadata.get(field)
            if value is not None:
                stats[field] = value
        if metadata.get("crisis") is not None:
            stats["crisis"] = bool(metadata.get("crisis"))
        return stats

    def _spread_curve(self, actions: torch.Tensor, batch: EpisodeBatch) -> Dict[str, float]:
        curve: Dict[str, float] = {}
        if not self.spreads:
            return curve
        for spread in self.spreads:
            stressed = EpisodeBatch(
                env=batch.env,
                spot=batch.spot,
                delta=batch.delta,
                gamma=batch.gamma,
                theta=batch.theta,
                realized_vol=batch.realized_vol,
                implied_vol=batch.implied_vol,
                option_price=batch.option_price,
                tx_linear=torch.full_like(batch.tx_linear, float(spread)),
                tx_quadratic=batch.tx_quadratic,
            )
            stressed = stressed.to(self.device)
            outputs = compute_pnl(actions, stressed)
            metrics = compute_metrics(outputs, alpha=self.alpha)
            curve[str(spread)] = metrics["cvar_95"]
        return curve

    def evaluate_module(self, model: torch.nn.Module, data_module) -> EvaluationResult:
        model.eval()
        env_metrics: Dict[str, Dict[str, float]] = {}
        spread_metrics: Dict[str, Dict[str, float]] = {}
        coverage: Dict[str, Dict[str, Any]] = {}
        for env, dataset in data_module.test_sets().items():
            metrics, sensitivity = self.evaluate_dataset(model, dataset)
            env_metrics[env] = metrics
            coverage[env] = self.coverage_stats(dataset)
            if sensitivity:
                spread_metrics[env] = sensitivity
        return EvaluationResult(env_metrics, spread_metrics, coverage)

