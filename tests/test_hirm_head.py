from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from infra.io import write_alignment_csv
from legacy.train.loop import build_config, run_training
from src.modules.head_invariance import EnvLossPayload, HIRMHeadConfig, hirm_head_loss
from src.modules.models import Policy


class TinyPolicy(Policy):
    def __init__(self) -> None:
        super().__init__(head_name="decision_head")
        self.backbone = nn.Linear(2, 2, bias=False)
        self.decision_head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(2))
            self.decision_head.weight.fill_(0.5)

    def forward(self, features: torch.Tensor, env_index: int = 0, **_) -> dict[str, torch.Tensor]:
        rep = self.backbone(features)
        if self.should_detach_features():
            rep = rep.detach()
        out = self.decision_head(rep)
        return {"action": out}


class HeadOnlyPolicy(Policy):
    def __init__(self) -> None:
        super().__init__(head_name="decision_head")
        self.decision_head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.decision_head.weight.zero_()

    def forward(self, features: torch.Tensor, env_index: int = 0, **_) -> dict[str, torch.Tensor]:
        out = self.decision_head(features)
        return {"action": out}


def _make_payload(
    model: Policy,
    features: torch.Tensor,
    targets: torch.Tensor,
    env_index: int,
    name: str,
) -> EnvLossPayload:
    outputs = model(features, env_index=env_index)
    preds = outputs["action"].squeeze(-1)
    base_loss = F.mse_loss(preds, targets)
    features_detached = features.detach().clone()
    targets_detached = targets.detach().clone()

    def closure(detach: bool) -> torch.Tensor:
        with model.alignment_context(detach_features=detach):
            out = model(features_detached, env_index=env_index)
            pred = out["action"].squeeze(-1)
            return F.mse_loss(pred, targets_detached)

    return EnvLossPayload(name=name, loss=base_loss, grad_closure=closure)


def test_freeze_phi_blocks_gradients() -> None:
    cfg = HIRMHeadConfig(lambda_align=1.0, normalize_grad=True, pairwise_mode="cosine")
    features = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([1.0])
    other_features = torch.tensor([[0.5, -1.0]])
    other_targets = torch.tensor([0.0])

    # Without freezing phi gradients should flow to both phi and psi parameters.
    model = TinyPolicy()
    payloads = [
        _make_payload(model, features, targets, 0, "low"),
        _make_payload(model, other_features, other_targets, 1, "med"),
    ]
    loss = hirm_head_loss(model, payloads, cfg)
    loss.total.backward()
    phi_grads = [param.grad.clone() for param in model.backbone.parameters() if param.grad is not None]
    psi_grads = [param.grad.clone() for param in model.decision_head.parameters() if param.grad is not None]
    assert phi_grads, "phi gradients should be populated when not frozen"
    assert psi_grads and any(torch.any(g.abs() > 0) for g in psi_grads)

    # Freezing phi should zero-out backbone gradients but still update psi.
    model_frozen = TinyPolicy()
    model_frozen.freeze_phi()
    payloads_frozen = [
        _make_payload(model_frozen, features, targets, 0, "low"),
        _make_payload(model_frozen, other_features, other_targets, 1, "med"),
    ]
    frozen_loss = hirm_head_loss(model_frozen, payloads_frozen, cfg)
    frozen_loss.total.backward()
    phi_zero = [param.grad for param in model_frozen.backbone.parameters()]
    psi_nonzero = [param.grad for param in model_frozen.decision_head.parameters()]
    assert all(g is None or torch.allclose(g, torch.zeros_like(g)) for g in phi_zero)
    assert any(g is not None and torch.any(g.abs() > 0) for g in psi_nonzero)


def test_lambda_zero_matches_mean_risk() -> None:
    cfg = HIRMHeadConfig(lambda_align=0.0, normalize_grad=True, pairwise_mode="cosine")
    model = TinyPolicy()
    feats = torch.tensor([[1.0, -1.0]])
    targets = torch.tensor([0.5])
    payloads = [
        _make_payload(model, feats, targets, 0, "env_a"),
        _make_payload(model, feats * 0.5, targets * 0.5, 1, "env_b"),
    ]
    result = hirm_head_loss(model, payloads, cfg)
    assert torch.allclose(result.total, result.avg_risk, atol=1e-7)


def test_cosine_penalty_values() -> None:
    cfg = HIRMHeadConfig(lambda_align=1.0, normalize_grad=True, pairwise_mode="cosine")
    features = torch.ones(1, 1)
    # Identical gradients -> cosine 1 -> penalty ~ 0
    model_same = HeadOnlyPolicy()
    payloads_same = [
        _make_payload(model_same, features, torch.full((1,), -1.0), 0, "env0"),
        _make_payload(model_same, features, torch.full((1,), -1.0), 1, "env1"),
    ]
    result_same = hirm_head_loss(model_same, payloads_same, cfg)
    assert pytest.approx(result_same.penalty.item(), abs=1e-6) == 0.0
    assert result_same.pairwise and pytest.approx(result_same.pairwise[0].item(), abs=1e-6) == 1.0

    # Opposite gradients -> cosine -1 -> penalty ~ 2
    model_opp = HeadOnlyPolicy()
    payloads_opp = [
        _make_payload(model_opp, features, torch.full((1,), -1.0), 0, "env0"),
        _make_payload(model_opp, features, torch.full((1,), 1.0), 1, "env1"),
    ]
    result_opp = hirm_head_loss(model_opp, payloads_opp, cfg)
    assert pytest.approx(result_opp.pairwise[0].item(), abs=1e-6) == -1.0
    assert pytest.approx(result_opp.penalty.item(), abs=1e-6) == 2.0


def test_normalization_invariance() -> None:
    cfg = HIRMHeadConfig(lambda_align=1.0, normalize_grad=True, pairwise_mode="cosine")
    features = torch.ones(1, 1)

    model_base = HeadOnlyPolicy()
    payloads_base = [
        _make_payload(model_base, features, torch.full((1,), -1.0), 0, "env0"),
        _make_payload(model_base, features, torch.full((1,), 0.5), 1, "env1"),
    ]
    penalty_base = hirm_head_loss(model_base, payloads_base, cfg).penalty.item()

    model_scaled = HeadOnlyPolicy()
    payloads_scaled = [
        _make_payload(model_scaled, features * 2.0, torch.full((1,), -2.0), 0, "env0"),
        _make_payload(model_scaled, features, torch.full((1,), 0.5), 1, "env1"),
    ]
    penalty_scaled = hirm_head_loss(model_scaled, payloads_scaled, cfg).penalty.item()

    assert pytest.approx(penalty_base, abs=1e-6) == penalty_scaled


def test_alignment_logging_creates_csv(tmp_path: Path) -> None:
    raw_cfg = {
        "objective": "hirm_head",
        "hirm": {"lambda_align": 1.0, "normalize_grad": True, "pairwise_mode": "cosine"},
        "model": {"freeze_phi": False, "head_name": "decision_head"},
        "train": {
            "envs": ["env0", "env1"],
            "seed": 3,
            "epochs": 1,
            "log_interval": 1,
            "batch_size": 8,
            "steps_per_epoch": 1,
            "feature_dim": 2,
        },
    }
    config = build_config(raw_cfg)
    run_dir = run_training(config, base_dir=tmp_path)
    csv_path = run_dir / "train" / "alignment_head.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows, "alignment CSV should contain at least one row"
    header = reader.fieldnames or []
    required = {"step", "epoch", "envs", "align_mode", "lambda_align", "normed", "penalty_value", "avg_risk"}
    assert required.issubset(set(header))


def test_alignment_writer_appends(tmp_path: Path) -> None:
    path = tmp_path / "alignment.csv"
    rows = [
        {"step": 1, "epoch": 0, "envs": "a|b", "align_mode": "cosine", "lambda_align": 1.0, "normed": True, "penalty_value": 0.1, "avg_risk": 0.5, "pair(0)": 0.9},
        {"step": 2, "epoch": 0, "envs": "a|b", "align_mode": "cosine", "lambda_align": 1.0, "normed": True, "penalty_value": 0.2, "avg_risk": 0.6, "pair(0)": 0.8},
    ]
    write_alignment_csv(path, rows[:1])
    write_alignment_csv(path, rows[1:])
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        loaded = list(reader)
    assert len(loaded) == 2
    assert float(loaded[1]["penalty_value"]) == pytest.approx(0.2)
