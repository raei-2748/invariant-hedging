"""Lightweight training loop using the head-only HIRM objective."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from hirm.infra.io import write_alignment_csv
from hirm.models import PolicyMLP
from hirm.train.objectives.hirm_head import (
    EnvLossPayload,
    HIRMHeadConfig,
    HIRMHeadLoss,
    hirm_head_loss,
)


@dataclass
class ModelConfig:
    freeze_phi: bool = False
    head_name: str = "decision_head"
    hidden_width: int = 64
    hidden_depth: int = 2
    dropout: float = 0.0
    layer_norm: bool = False
    representation_dim: int = 32
    adapter_hidden: int = 16
    max_position: float = 5.0


@dataclass
class TrainConfig:
    envs: list[str]
    seed: int = 0
    epochs: int = 1
    log_interval: int = 1
    batch_size: int = 64
    steps_per_epoch: int = 10
    feature_dim: int = 8


@dataclass
class ExperimentConfig:
    objective: str
    hirm: HIRMHeadConfig
    model: ModelConfig
    train: TrainConfig


def _ensure_list(value: Iterable[str] | str) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def build_config(raw: Mapping[str, object]) -> ExperimentConfig:
    objective = str(raw.get("objective", "erm"))
    hirm_raw = raw.get("hirm", {})
    if not isinstance(hirm_raw, Mapping):
        raise TypeError("Config field 'hirm' must be a mapping.")
    hirm_cfg = HIRMHeadConfig(
        lambda_align=float(hirm_raw.get("lambda_align", 1.0)),
        normalize_grad=bool(hirm_raw.get("normalize_grad", True)),
        pairwise_mode=str(hirm_raw.get("pairwise_mode", "cosine")),
        detach_features=bool(hirm_raw.get("detach_features", False)),
    )

    model_raw = raw.get("model", {})
    if not isinstance(model_raw, Mapping):
        raise TypeError("Config field 'model' must be a mapping.")
    model_cfg = ModelConfig(
        freeze_phi=bool(model_raw.get("freeze_phi", False)),
        head_name=str(model_raw.get("head_name", "decision_head")),
        hidden_width=int(model_raw.get("hidden_width", 64)),
        hidden_depth=int(model_raw.get("hidden_depth", 2)),
        dropout=float(model_raw.get("dropout", 0.0)),
        layer_norm=bool(model_raw.get("layer_norm", False)),
        representation_dim=int(model_raw.get("representation_dim", 32)),
        adapter_hidden=int(model_raw.get("adapter_hidden", 16)),
        max_position=float(model_raw.get("max_position", 5.0)),
    )

    train_raw = raw.get("train", {})
    if not isinstance(train_raw, Mapping):
        raise TypeError("Config field 'train' must be a mapping.")
    envs = train_raw.get("envs", [])
    if not envs:
        raise ValueError("Training config must specify at least one environment.")
    train_cfg = TrainConfig(
        envs=_ensure_list(envs),
        seed=int(train_raw.get("seed", 0)),
        epochs=int(train_raw.get("epochs", 1)),
        log_interval=int(train_raw.get("log_interval", 1)),
        batch_size=int(train_raw.get("batch_size", 64)),
        steps_per_epoch=int(train_raw.get("steps_per_epoch", 10)),
        feature_dim=int(train_raw.get("feature_dim", 8)),
    )

    return ExperimentConfig(
        objective=objective,
        hirm=hirm_cfg,
        model=model_cfg,
        train=train_cfg,
    )


def _make_grad_closure(
    model: PolicyMLP,
    features: torch.Tensor,
    targets: torch.Tensor,
    env_index: int,
) -> Callable[[bool], torch.Tensor]:
    feat = features.detach()
    targ = targets.detach()

    def _closure(detach: bool) -> torch.Tensor:
        with model.alignment_context(detach_features=detach):
            out = model(feat, env_index=env_index)
            preds = out["action"].squeeze(-1)
            return F.mse_loss(preds, targ)

    return _closure


def _alignment_row(
    step: int,
    epoch: int,
    config: ExperimentConfig,
    loss: HIRMHeadLoss,
    env_names: Sequence[str],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "step": step,
        "epoch": epoch,
        "envs": "|".join(env_names),
        "align_mode": config.hirm.pairwise_mode,
        "lambda_align": config.hirm.lambda_align,
        "normed": bool(config.hirm.normalize_grad),
        "penalty_value": float(loss.penalty.detach().item()),
        "avg_risk": float(loss.avg_risk.detach().item()),
    }
    mode = config.hirm.pairwise_mode.lower()
    if mode == "cosine":
        for idx, value in enumerate(loss.pairwise):
            row[f"pair({idx})"] = float(value.detach().item())
    elif mode == "var":
        if loss.var_total is not None:
            row["var_total"] = float(loss.var_total.detach().item())
    return row


def run_training(config: ExperimentConfig, *, base_dir: Path | None = None) -> Path:
    if config.objective != "hirm_head":
        raise ValueError("This training loop only supports objective='hirm_head'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.train.seed)

    model = PolicyMLP(
        feature_dim=config.train.feature_dim,
        num_envs=len(config.train.envs),
        hidden_width=config.model.hidden_width,
        hidden_depth=config.model.hidden_depth,
        dropout=config.model.dropout,
        layer_norm=config.model.layer_norm,
        representation_dim=config.model.representation_dim,
        adapter_hidden=config.model.adapter_hidden,
        max_position=config.model.max_position,
        head_name=config.model.head_name,
    ).to(device)
    model.train()

    if config.model.freeze_phi:
        model.freeze_phi()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = base_dir if base_dir is not None else Path("runs")
    run_dir = run_root / f"{timestamp}_hirm_head"
    seed_dir = run_dir / "seeds" / str(config.train.seed)
    train_dir = seed_dir / "train"
    alignment_path = train_dir / "alignment_head.csv"

    env_weights: Dict[str, torch.Tensor] = {}
    env_generators: Dict[str, torch.Generator] = {}
    for idx, name in enumerate(config.train.envs):
        gen = torch.Generator(device=device)
        gen.manual_seed(config.train.seed + idx + 1)
        weight = torch.randn(config.train.feature_dim, generator=gen, device=device)
        env_weights[name] = weight
        env_generators[name] = gen

    global_step = 0
    for epoch in range(config.train.epochs):
        for _ in range(config.train.steps_per_epoch):
            global_step += 1
            optimizer.zero_grad()
            payloads: list[EnvLossPayload] = []
            for env_index, env_name in enumerate(config.train.envs):
                generator = env_generators[env_name]
                features = torch.randn(
                    config.train.batch_size,
                    config.train.feature_dim,
                    generator=generator,
                    device=device,
                )
                weight = env_weights[env_name]
                noise = 0.05 * torch.randn(
                    config.train.batch_size, 1, generator=generator, device=device
                )
                targets = (features @ weight.unsqueeze(-1)) + noise
                targets = targets.squeeze(-1)
                outputs = model(features, env_index=env_index)
                preds = outputs["action"].squeeze(-1)
                base_loss = F.mse_loss(preds, targets)
                closure = _make_grad_closure(model, features, targets, env_index)
                payloads.append(EnvLossPayload(env_name, base_loss, closure))

            loss = hirm_head_loss(model, payloads, config.hirm)
            loss.total.backward()
            optimizer.step()

            row = _alignment_row(global_step, epoch, config, loss, config.train.envs)
            write_alignment_csv(alignment_path, [row])

    return seed_dir
