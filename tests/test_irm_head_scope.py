import torch
from torch import nn

from src.models.heads import RiskHead
from src.objectives.invariance import irm_penalty


def test_irm_penalty_affects_head_only():
    torch.manual_seed(0)
    encoder = nn.Linear(4, 4)
    head = RiskHead(4, hidden_dim=0)
    input_tensor = torch.randn(8, 4)
    features = encoder(input_tensor)
    for param in list(encoder.parameters()) + list(head.parameters()):
        if param.grad is not None:
            param.grad.zero_()

    env_features = [features.detach(), features.detach() * 1.1]

    def loss_fn(module, feat, target, dummy):
        return (module(feat) * dummy).mean()

    penalty = irm_penalty(head, env_features, None, loss_fn, scope="head")
    penalty.backward()

    head_grads = [param.grad for param in head.parameters() if param.grad is not None]
    assert head_grads, "Head parameters should receive gradients from IRM penalty"
    for grad in head_grads:
        assert torch.any(grad != 0), "Head gradients should be non-zero"

    for param in encoder.parameters():
        assert param.grad is None or torch.all(param.grad == 0), "Encoder should remain unaffected"
