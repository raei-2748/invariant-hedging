from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from hirm_experiment.algorithms import build_algorithm
from hirm_experiment.data.dataset import VolatilityRegimeDataModule, create_data_module
from hirm_experiment.evaluation.metrics import compute_metrics
from hirm_experiment.models.policy import HedgingPolicy, HedgingPolicyConfig
from hirm_experiment.training.features import FeatureBuilder
from hirm_experiment.training.logger import ExperimentLogger
from hirm_experiment.training.pnl import rollout_policy
from hirm_experiment.training.schedulers import build_scheduler
from hirm_experiment.training.utils import resolve_device, set_seed


class TrainingEngine:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.project_cfg = cfg.get("project", {})
        self.training_cfg = cfg.get("training", {})
        self.model_cfg = cfg.get("model", {})
        self.algorithm_cfg = cfg.get("algorithm", {})
        self.data_cfg = cfg.get("data", {})
        self.logging_cfg = cfg.get("logging", {})

        set_seed(int(self.project_cfg.get("seed", 42)))
        self.device = resolve_device(self.training_cfg.get("device", "auto"))
        self.output_dir = Path(self.project_cfg.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_module = create_data_module(self.data_cfg, int(self.project_cfg.get("seed", 42)), self.device)
        invariants = self.data_cfg.get("feature_set", {}).get("invariants", [])
        spurious = self.data_cfg.get("feature_set", {}).get("spurious", [])
        feature_stats = self.data_module.get_feature_stats()
        self.feature_builder = FeatureBuilder(feature_stats, invariants, spurious)

        self.model_config = self._build_model_config(len(invariants) + len(spurious))
        self.model = HedgingPolicy(self.model_config).to(self.device)

        self.algorithm = build_algorithm(self.algorithm_cfg, self.feature_builder, self.device, self.data_module.train_envs)

        weight_decay = float(self.algorithm_cfg.get("regularization", {}).get("weight_decay", 0.0))
        lr = float(self.training_cfg.get("optimizer", {}).get("lr", 1e-4))
        betas = tuple(self.training_cfg.get("optimizer", {}).get("betas", [0.9, 0.999]))
        eps = float(self.training_cfg.get("optimizer", {}).get("eps", 1e-8))
        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.scheduler = build_scheduler(self.optimizer, self.training_cfg.get("scheduler", {}), int(self.training_cfg.get("total_steps", 1)))

        self.logger = ExperimentLogger(self.logging_cfg, self.project_cfg)
        self.checkpointing = cfg.get("checkpointing", {"enabled": False})
        self.best_val_metric: Optional[float] = None

    def _build_model_config(self, input_dim: int) -> HedgingPolicyConfig:
        hidden_dims = list(self.model_cfg.get("hidden_dims", [128, 128, 64]))
        dropout = float(self.model_cfg.get("dropout", 0.0))
        if self.algorithm_cfg.get("name") == "erm_reg":
            dropout = float(self.algorithm_cfg.get("regularization", {}).get("dropout", dropout))
        return HedgingPolicyConfig(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=self.model_cfg.get("activation", "relu"),
            output_dim=int(self.model_cfg.get("output_dim", 1)),
            bounded_output=bool(self.model_cfg.get("bounded_output", True)),
            output_scale=float(self.model_cfg.get("output_scale", 1.0)),
        )

    def train(self) -> None:
        total_steps = int(self.training_cfg.get("total_steps", 1))
        batch_size = int(self.training_cfg.get("batch_size", 128))
        log_interval = int(self.training_cfg.get("log_interval", 200))
        eval_interval = int(self.training_cfg.get("eval_interval", 5000))
        save_interval = int(self.training_cfg.get("save_interval", 25000))
        clip_norm = float(self.training_cfg.get("clip_grad_norm", 1.0))

        for step in range(total_steps):
            self.model.train()
            batches = self.data_module.sample_train_batches(batch_size)
            loss_output = self.algorithm.compute_loss(self.model, batches, step)
            self.optimizer.zero_grad(set_to_none=True)
            loss_output.loss.backward()
            if clip_norm > 0:
                clip_grad_norm_(self.model.parameters(), clip_norm)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if step % log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                metrics = {"train/lr": lr}
                metrics.update({f"train/{k}": float(v.cpu().item()) for k, v in loss_output.logs.items()})
                self.logger.log(metrics, step=step)

            if eval_interval and step > 0 and step % eval_interval == 0:
                self.evaluate(step)

            if self.checkpointing.get("enabled", False) and save_interval and step > 0 and step % save_interval == 0:
                self._save_checkpoint(step)

        self.logger.finish()

    def evaluate(self, step: int) -> None:
        self.model.eval()
        eval_logs: Dict[str, float] = {}
        with torch.no_grad():
            val_batches = self.data_module.sample_validation()
            for env, batch in val_batches.items():
                outputs = rollout_policy(self.model, batch, self.feature_builder)
                metrics = compute_metrics(outputs)
                for key, value in metrics.items():
                    eval_logs[f"val/{env}/{key}"] = value
            test_env = self.data_cfg.get("test_environment")
            for name, test_dataset in self.data_module.test_sets().items():
                batch = test_dataset.full(self.device)
                outputs = rollout_policy(self.model, batch, self.feature_builder)
                metrics = compute_metrics(outputs)
                env_name = test_dataset.env if test_dataset.env else name
                for key, value in metrics.items():
                    eval_logs[f"test/{env_name}/{key}"] = value
                if test_env and env_name == test_env:
                    crisis_cvar = metrics.get("cvar_95")
                    if crisis_cvar is not None:
                        if self.best_val_metric is None or crisis_cvar < self.best_val_metric:
                            self.best_val_metric = crisis_cvar
                            if self.checkpointing.get("enabled", False):
                                self._save_checkpoint(step)
        if eval_logs:
            self.logger.log({f"eval/{k}": v for k, v in eval_logs.items()}, step=step)

    def _save_checkpoint(self, step: int) -> None:
        ckpt_dir = Path(self.checkpointing.get("dir", self.output_dir / "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": step,
            "model_config": self.model_config.__dict__,
            "project_config": self.project_cfg,
            "training_step": step,
        }
        ckpt_path = ckpt_dir / f"checkpoint_{step:06d}.pt"
        torch.save(state, ckpt_path)


def run_training(cfg: Dict) -> None:
    engine = TrainingEngine(cfg)
    engine.train()
