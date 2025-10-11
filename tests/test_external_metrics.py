import pytest
import torch

from src.diagnostics.external import (
    compute_expected_risk,
    compute_ig,
    compute_tail_risk,
    compute_variation_ratio,
    compute_wg,
)


def test_invariant_gap_with_trim_and_clamp():
    values = [0.2, 0.25, 0.3, 4.0]
    raw = compute_ig(values)
    clamped = compute_ig(values, clamp=(0.0, 0.5))
    trimmed = compute_ig(values, trim_fraction=0.25)
    assert raw > clamped
    assert trimmed < raw


def test_worst_group_and_expected_risk():
    train = [0.1, 0.2, 0.15]
    test = [0.25, 0.3, 0.35]
    wg = compute_wg(train, test)
    er = compute_expected_risk(test)
    assert pytest.approx(wg, rel=1e-5) == 0.15
    assert pytest.approx(er, rel=1e-5) == sum(test) / len(test)


def test_variation_ratio_and_tail_risk():
    tight = torch.tensor([[0.3, 0.1], [0.31, 0.11], [0.29, 0.09]])
    spread = torch.tensor([[0.2, 0.05], [0.45, 0.18], [0.55, 0.25]])
    tight_vr = compute_variation_ratio(tight)
    spread_vr = compute_variation_ratio(spread)
    assert spread_vr > tight_vr

    test_values = [0.25, 0.35, 0.4]
    tail = compute_tail_risk(test_values, quantile=0.9)
    expected_quantile = torch.quantile(torch.tensor(test_values, dtype=torch.float32), torch.tensor(0.9))
    assert tail >= expected_quantile.item()
