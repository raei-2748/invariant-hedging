"""High-level training pipeline orchestrating data, models, and algorithms."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from omegaconf import DictConfig, OmegaConf

from ..algorithms import build_algorithm
from ..algorithms.common import EnvLoss, TrainBatch
from ..algorithms.hirm import HIRMAlgorithm
from ..data.features import FeatureEngineer
from ..envs.single_asset import SingleAssetHedgingEnv
from ..irm.head_grads import freeze_backbone
from ..models import PolicyMLP
from ..objectives import cvar as cvar_obj
from ..utils import checkpoints, logging as log_utils, seed as seed_utils, stats
from ..utils.configs import build_envs, prepare_data_module, unwrap_experiment_config


@dataclass
class RunResult:
    """Summary of a single seeded training run."""

    seed: int
    run_dir: Path
    final_metrics: Dict[str, float]
    checkpoints: List[Path]


@dataclass
class Aggregates:
    """Aggregate statistics over multiple runs."""

    seeds: List[int]
    metrics: Dict[str, Dict[str, float]]


def _device(runtime_cfg: DictConfig | None) -> torch.device:
    device_str = runtime_cfg.get("device", "auto") if runtime_cfg else "auto"
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _init_policy(cfg: DictConfig, feature_dim: int, num_envs: int) -> PolicyMLP:
    return PolicyMLP(
        feature_dim=feature_dim,
        num_envs=num_envs,
        hidden_width=cfg.model.hidden_width,
        hidden_depth=cfg.model.hidden_depth,
        dropout=cfg.model.dropout,
        layer_norm=cfg.model.layer_norm,
        representation_dim=cfg.model.representation_dim,
        adapter_hidden=cfg.model.adapter_hidden,
        max_position=cfg.model.max_position,
        head_name=cfg.model.get("head_name"),
    )


def _setup_optimizer(policy: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    name = str(cfg.optimizer.name).lower()
    params = [p for p in policy.parameters() if p.requires_grad]
    lr = float(cfg.optimizer.lr)
    weight_decay = float(cfg.optimizer.get("weight_decay", 0.0))
    if str(cfg.model.get("name", "")) == "erm_reg":
        regularization = cfg.model.get("regularization", {})
        weight_decay = float(regularization.get("weight_decay", weight_decay))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{cfg.optimizer.name}'.")


def _setup_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    name = str(cfg.scheduler.name).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.train.steps), eta_min=float(cfg.optimizer.lr) * 0.1
        )
    if name == "none":
        return None
    raise ValueError(f"Unsupported scheduler '{cfg.scheduler.name}'.")


def _evaluate(policy, envs: Dict[str, SingleAssetHedgingEnv], device: torch.device, alpha: float):
    policy.eval()
    results: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for name, env in envs.items():
            indices = torch.arange(env.batch.spot.shape[0])
            sim = env.simulate(policy, indices, device)
            pnl = sim.pnl
            step_pnl = sim.step_pnl
            turnover = sim.turnover
            cvar = cvar_obj.cvar_from_pnl(pnl, alpha)
            results[name] = {
                "cvar": float(cvar.item()),
                "mean_pnl": float(pnl.mean().item()),
                "turnover": float(turnover.mean().item()),
                "max_drawdown": float(stats.max_drawdown(step_pnl).mean().item()),
                "sharpe": float(stats.sharpe_ratio(step_pnl).mean().item()),
            }
    policy.train()
    return results


def _collect_env_metrics(
    *,
    env_name: str,
    sim_output,
    pnl: torch.Tensor,
    alpha: float,
    logger: log_utils.RunLogger,
    step: int,
    max_trade_warn: float,
    spike_alerted_envs: set[str],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        f"train/{env_name}_mean_pnl": float(pnl.mean().item()),
        f"train/{env_name}_cvar": float(cvar_obj.cvar_from_pnl(pnl, alpha).item()),
        f"train/{env_name}_turnover": float(sim_output.turnover.mean().item()),
    }
    trades = sim_output.positions[:, 1:] - sim_output.positions[:, :-1]
    final_trade = -sim_output.positions[:, -1:]
    all_trades = torch.cat([trades, final_trade], dim=1)
    metrics[f"train/{env_name}_mean_abs_trade"] = float(torch.abs(all_trades).mean().item())
    max_trade_val = float(sim_output.max_trade.max().item())
    metrics[f"train/{env_name}_max_trade"] = max_trade_val
    if max_trade_warn > 0 and max_trade_val > max_trade_warn and env_name not in spike_alerted_envs:
        print(
            f"[warn] large trade magnitude detected: {max_trade_val:.2f} "
            f"(threshold {max_trade_warn:.2f}) in env '{env_name}' at step {step}"
        )
        spike_alerted_envs.add(env_name)
    if sim_output.probe:
        logger.log_probe(env_name, step, sim_output.probe)
    return metrics


def _algo_name(cfg: DictConfig) -> str:
    algo_cfg = cfg.get("algorithm")
    if algo_cfg and "name" in algo_cfg:
        return str(algo_cfg.name)
    model_obj = cfg.model.get("objective", cfg.model.get("name", "erm"))
    return str(model_obj)


def _algorithm_config(cfg: DictConfig):
    algo_cfg = cfg.get("algorithm")
    if algo_cfg is not None:
        data = OmegaConf.to_container(algo_cfg, resolve=True)
        if not isinstance(data, dict):
            data = dict()
        if "warmup_steps" not in data and cfg.train.get("irm_ramp_steps") is not None:
            data["warmup_steps"] = int(cfg.train.irm_ramp_steps)
        if "delay_steps" not in data and cfg.train.get("pretrain_steps") is not None:
            data["delay_steps"] = int(cfg.train.pretrain_steps)
        return OmegaConf.create(data)
    return cfg.model


def train_one_seed(cfg: DictConfig) -> RunResult:
    cfg = unwrap_experiment_config(cfg)
    seed = int(cfg.train.seed)
    generator = seed_utils.seed_everything(seed)
    runtime_cfg = cfg.get("runtime")
    device = _device(runtime_cfg)

    data_ctx = prepare_data_module(cfg, seed=seed)
    train_batches = data_ctx.data_module.prepare("train", cfg.envs.train)
    val_batches = data_ctx.data_module.prepare("val", cfg.envs.val)
    test_batches = data_ctx.data_module.prepare("test", cfg.envs.test)

    feature_engineer = FeatureEngineer()
    feature_engineer.fit(train_batches.values())
    if feature_engineer.scaler is None:
        raise RuntimeError("Feature scaler must be initialised before training.")

    train_envs = build_envs(train_batches, feature_engineer, data_ctx.name_to_index)
    val_envs = build_envs(val_batches, feature_engineer, data_ctx.name_to_index)
    test_envs = build_envs(test_batches, feature_engineer, data_ctx.name_to_index)

    policy = _init_policy(cfg, feature_dim=len(feature_engineer.feature_names), num_envs=len(data_ctx.env_order))
    policy.to(device)

    optimizer = _setup_optimizer(policy, cfg)
    scheduler = _setup_scheduler(optimizer, cfg)

    use_amp = device.type == "cuda" and bool(getattr(runtime_cfg, "mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    algo_name = _algo_name(cfg).lower()
    algorithm = build_algorithm(
        algo_name,
        model=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        grad_clip=float(cfg.train.get("grad_clip", 0.0)),
        config=_algorithm_config(cfg),
    )

    if isinstance(algorithm, HIRMAlgorithm) and algorithm.irm_config.freeze_backbone:
        freeze_backbone(policy)

    logging_cfg = cfg.get("logging")
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    logging_container = (
        OmegaConf.to_container(logging_cfg, resolve=True) if logging_cfg is not None else {}
    )
    run_logger = log_utils.RunLogger(logging_container, resolved_cfg)
    checkpoint_dir = Path(run_logger.checkpoint_dir)
    ckpt_manager = checkpoints.CheckpointManager(checkpoint_dir, top_k=int(cfg.train.checkpoint_topk))

    per_env_batch = max(1, int(cfg.train.batch_size) // max(len(cfg.envs.train), 1))
    alpha = float(cfg.loss.cvar_alpha)
    log_interval = int(getattr(logging_cfg, "log_interval", cfg.train.get("log_interval", 100)))
    eval_interval = int(getattr(logging_cfg, "eval_interval", cfg.train.get("eval_interval", 1000)))
    warning_factor = float(cfg.train.get("max_trade_warning_factor", 0.0))
    max_position = float(cfg.model.max_position)
    max_trade_warn = warning_factor * max_position if warning_factor > 0 else 0.0
    spike_alerted_envs: set[str] = set()

    label_smoothing = 0.0
    algo_cfg = _algorithm_config(cfg)
    if hasattr(algo_cfg, "label_smoothing"):
        label_smoothing = float(getattr(algo_cfg, "label_smoothing"))
    elif cfg.model.get("name") == "erm_reg":
        reg_cfg = cfg.model.get("regularization", {})
        label_smoothing = float(reg_cfg.get("label_smoothing", 0.0))

    policy.train()

    for step in range(1, int(cfg.train.steps) + 1):
        env_losses: List[EnvLoss] = []
        metrics_to_log: Dict[str, float] = {}
        representation_scale = getattr(algorithm, "representation_scale", None)

        detach_for_penalty = isinstance(algorithm, HIRMAlgorithm) and getattr(
            algorithm, "detach_features", False
        )

        for env_name in cfg.envs.train:
            env = train_envs[env_name]
            indices = env.sample_indices(per_env_batch, generator)
            sim = env.simulate(policy, indices, device, representation_scale=representation_scale)
            pnl = sim.pnl
            if label_smoothing > 0:
                pnl = (1 - label_smoothing) * pnl + label_smoothing * pnl.mean()
            loss = cvar_obj.differentiable_cvar(-pnl, alpha)
            irm_loss = loss
            if detach_for_penalty:
                with policy.alignment_context(detach_features=True):
                    sim_pen = env.simulate(
                        policy,
                        indices,
                        device,
                        representation_scale=representation_scale,
                    )
                pnl_pen = sim_pen.pnl
                irm_loss = cvar_obj.differentiable_cvar(-pnl_pen, alpha)
            env_losses.append(
                EnvLoss(
                    name=env_name,
                    loss=loss,
                    payload={"loss": loss, "irm_loss": irm_loss, "name": env_name},
                )
            )
            metrics_to_log.update(
                _collect_env_metrics(
                    env_name=env_name,
                    sim_output=sim,
                    pnl=pnl,
                    alpha=alpha,
                    logger=run_logger,
                    step=step,
                    max_trade_warn=max_trade_warn,
                    spike_alerted_envs=spike_alerted_envs,
                )
            )

        batch = TrainBatch(step=step, env_losses=env_losses)
        algo_logs = algorithm.step(batch)
        metrics_to_log.update(algo_logs)
        total_loss = metrics_to_log.get("train/loss", 0.0) + metrics_to_log.get("train/penalty", 0.0)
        metrics_to_log.setdefault("train/loss_total", total_loss)
        metrics_to_log["train/lr"] = float(optimizer.param_groups[0]["lr"])

        if step % log_interval == 0 or step == 1:
            run_logger.log_metrics(metrics_to_log, step=step)

        if step % eval_interval == 0 or step == int(cfg.train.steps):
            val_metrics = _evaluate(policy, val_envs, device, alpha)
            val_logs = {f"val/{env}_{metric}": value for env, metrics in val_metrics.items() for metric, value in metrics.items()}
            run_logger.log_metrics(val_logs, step=step)
            primary_env = cfg.envs.val[0]
            score = -val_metrics[primary_env]["cvar"]
            algo_state = algorithm.state_dict()
            checkpoint_payload = {
                "model": policy.state_dict(),
                "step": step,
                "score": score,
                "feature_scaler": {
                    "mean": feature_engineer.scaler.mean.cpu(),
                    "std": feature_engineer.scaler.std.cpu(),
                },
                "optimizer": algo_state.get("optimizer"),
            }
            if "scheduler" in algo_state:
                checkpoint_payload["scheduler"] = algo_state["scheduler"]
            if "scaler" in algo_state:
                checkpoint_payload["amp_scaler"] = algo_state["scaler"]
            ckpt_manager.save(step, score, checkpoint_payload)

    test_metrics = _evaluate(policy, test_envs, device, alpha)
    final_payload = {f"test/{env}_{metric}": value for env, metrics in test_metrics.items() for metric, value in metrics.items()}
    run_logger.log_final(final_payload)

    final_metrics_path = Path(run_logger.final_metrics_path)
    if final_metrics_path.exists():
        final_metrics = json.loads(final_metrics_path.read_text())
    else:
        final_metrics = final_payload

    checkpoints_sorted = [entry.path for entry in sorted(ckpt_manager.heap, reverse=True)]

    run_logger.close()

    return RunResult(
        seed=seed,
        run_dir=Path(run_logger.base_dir),
        final_metrics=final_metrics,
        checkpoints=checkpoints_sorted,
    )


def aggregate_multi_seed(results: Iterable[RunResult]) -> Aggregates:
    seeds: List[int] = []
    metric_store: Dict[str, List[float]] = {}
    for result in results:
        seeds.append(result.seed)
        for key, value in result.final_metrics.items():
            if isinstance(value, (int, float)):
                metric_store.setdefault(key, []).append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for key, values in metric_store.items():
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float32)
        summary[key] = {
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0),
        }

    return Aggregates(seeds=seeds, metrics=summary)


__all__ = ["train_one_seed", "aggregate_multi_seed", "RunResult", "Aggregates"]
