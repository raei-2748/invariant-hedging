import torch

from src.models.two_head_policy import TwoHeadPolicy


def test_hybrid_gate_is_scalar_and_shared():
    policy = TwoHeadPolicy(
        feature_dim=5,
        num_envs=3,
        hidden_width=8,
        hidden_depth=2,
        dropout=0.0,
        layer_norm=False,
        representation_dim=4,
        adapter_hidden=4,
        max_position=2.0,
        risk_hidden=4,
        alpha_init=0.3,
    )

    assert policy.alpha.shape == torch.Size([]), "Alpha parameter should be a scalar"

    gate_before = policy.gate_value().item()
    features_env0 = torch.randn(6, 5)
    features_env1 = torch.randn(6, 5)
    _ = policy(features_env0, env_index=0)
    gate_after_env0 = policy.gate_value().item()
    _ = policy(features_env1, env_index=2)
    gate_after_env1 = policy.gate_value().item()

    assert gate_before == gate_after_env0 == gate_after_env1, "Gate should not depend on environment"
