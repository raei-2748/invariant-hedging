import torch

from hirm.objectives import cvar as cvar_obj


def test_cvar_matches_numpy_quantile():
    torch.manual_seed(0)
    pnl = torch.randn(1000)
    alpha = 0.95
    est = cvar_obj.cvar_from_pnl(pnl, alpha)
    losses = -pnl.numpy()
    q = torch.quantile(-pnl, alpha).item()
    tail = losses[losses >= q]
    reference = tail.mean()
    assert abs(est.item() - reference) < 1e-3


def test_bootstrap_ci_has_width():
    pnl = torch.linspace(-1, 1, 200)
    result = cvar_obj.bootstrap_cvar_ci(pnl, 0.9, num_samples=200, seed=123)
    assert result.upper > result.lower



def test_cvar_scales_down_with_losses():
    losses = torch.linspace(-2.0, 2.0, 500)
    scaled = losses * 0.5
    alpha = 0.9
    base = cvar_obj.cvar_loss(losses, alpha)
    scaled_val = cvar_obj.cvar_loss(scaled, alpha)
    assert scaled_val < base
