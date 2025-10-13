import pytest
import torch
from torch import nn

from hirm.irm.head_grads import compute_env_head_grads, freeze_backbone
from hirm.irm.penalties import cosine_alignment_penalty, varnorm_penalty


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 1)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(2))
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        return self.head(hidden)


def mse_risk(model: ToyModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    preds = model(batch["features"])
    return torch.nn.functional.mse_loss(preds, batch["targets"])


def test_freeze_backbone_only_affects_phi():
    model = ToyModel()
    freeze_backbone(model)
    for param in model.backbone.parameters():
        assert not param.requires_grad
    for param in model.head.parameters():
        assert param.requires_grad


def test_compute_env_head_grads_returns_flat_vectors():
    model = ToyModel()
    batches = [
        {"features": torch.ones(4, 2), "targets": torch.zeros(4, 1)},
        {"features": torch.full((4, 2), 2.0), "targets": torch.ones(4, 1)},
    ]
    grads = compute_env_head_grads(model, mse_risk, batches, create_graph=True)
    assert len(grads) == len(batches)
    head_dim = sum(param.numel() for param in model.head.parameters())
    for grad in grads:
        assert grad.shape == (head_dim,)


def test_penalties_reflect_alignment_behaviour():
    model = ToyModel()
    identical_batches = [
        {"features": torch.ones(8, 2), "targets": torch.ones(8, 1)},
        {"features": torch.ones(8, 2), "targets": torch.ones(8, 1)},
    ]
    grads_identical = compute_env_head_grads(model, mse_risk, identical_batches, create_graph=True)
    cos_identical = cosine_alignment_penalty(grads_identical)
    var_identical = varnorm_penalty(grads_identical)
    assert cos_identical.item() == pytest.approx(0.0, abs=1e-6)
    assert var_identical.item() == pytest.approx(0.0, abs=1e-6)

    opposing_batches = [
        {"features": torch.ones(8, 2), "targets": torch.ones(8, 1)},
        {"features": torch.ones(8, 2), "targets": -torch.ones(8, 1)},
    ]
    grads_opposing = compute_env_head_grads(model, mse_risk, opposing_batches, create_graph=True)
    cos_opposing = cosine_alignment_penalty(grads_opposing)
    var_opposing = varnorm_penalty(grads_opposing)
    assert cos_opposing.item() > 1.0
    assert var_opposing.item() > 0.0


def test_training_step_combines_base_and_penalty():
    model = ToyModel()
    freeze_backbone(model)
    batches = [
        {"features": torch.ones(6, 2), "targets": torch.ones(6, 1)},
        {"features": torch.full((6, 2), 0.5), "targets": torch.zeros(6, 1)},
    ]
    env_losses = [mse_risk(model, batch) for batch in batches]
    base_loss = torch.stack(env_losses).mean()
    grads = compute_env_head_grads(model, mse_risk, batches, create_graph=True)
    irm_penalty = cosine_alignment_penalty(grads)
    assert irm_penalty.item() > 0.0
    lambda_weight = 0.5
    total_loss = base_loss + lambda_weight * irm_penalty

    model.zero_grad(set_to_none=True)
    total_loss.backward()

    assert total_loss.item() == pytest.approx(
        base_loss.item() + lambda_weight * irm_penalty.item(), abs=1e-6
    )
    for param in model.backbone.parameters():
        assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))
    for param in model.head.parameters():
        assert param.grad is not None
        assert torch.linalg.norm(param.grad).item() > 0.0
