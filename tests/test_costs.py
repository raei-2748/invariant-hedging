import torch

from src.markets.costs import execution_cost


def test_execution_cost_linear_quadratic():
    trade = torch.tensor([1.0, -0.5])
    spot = torch.tensor([100.0, 105.0])
    cfg = {"linear_bps": 10, "quadratic": 0.2, "slippage_multiplier": 1.0}
    cost = execution_cost(trade, spot, cfg)
    expected_linear = torch.abs(trade) * spot * 10e-4
    expected_quad = trade ** 2 * 0.2
    expected = expected_linear + expected_quad
    assert torch.allclose(cost, expected)
