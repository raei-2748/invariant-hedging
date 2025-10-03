import torch
from torch import nn

from src.models.hirm_hybrid import HIRMHybrid
from src.models.heads import RiskHead


def test_hybrid_gate_is_scalar_and_shared():
    risk_inv = RiskHead(4, 4)
    risk_adapt = RiskHead(4, 4)
    hybrid = HIRMHybrid(nn.Identity(), risk_inv, risk_adapt, alpha_init=0.3)

    assert hybrid.alpha.shape == torch.Size([]), "Alpha parameter should be a scalar"

    gate_before = hybrid.gate_value().item()
    features_env0 = torch.randn(6, 4)
    features_env1 = torch.randn(6, 4)
    out0 = hybrid(features_env0)
    out1 = hybrid(features_env1)

    assert torch.isclose(out0[-1], out1[-1]), "Gate outputs should match across calls"
    assert gate_before == hybrid.gate_value().item(), "Gate parameter should not change during forward"
