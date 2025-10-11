import pytest
import torch

from src.diagnostics.isi import compute_C1, compute_C3, compute_ISI


def test_trim_and_clamp_behavior():
    data = torch.tensor([[10.0, 9.0], [1.0, 1.2], [1.1, 0.9]])
    trimmed = compute_ISI(data, trim_fraction=0.34)
    norms = torch.linalg.vector_norm(data, dim=1)
    sorted_idx = torch.argsort(norms)
    trim_count = int(torch.floor(torch.tensor(data.shape[0] * 0.34)).item())
    keep_idx = sorted_idx[trim_count : data.shape[0] - trim_count]
    expected_matrix = data.index_select(0, keep_idx)
    expected = compute_ISI(expected_matrix, trim_fraction=0.0)
    for key in ("C1", "C2", "C3", "ISI"):
        if trimmed[key] is None:
            assert expected[key] is None
        else:
            assert trimmed[key] == pytest.approx(expected[key], abs=1e-2)

    clamped = compute_ISI(data, clamp=(0.0, 2.0))
    clamped_matrix = torch.clamp(data, 0.0, 2.0)
    manual = compute_ISI(clamped_matrix)
    for key in ("C1", "C2", "C3", "ISI"):
        if clamped[key] is None:
            assert manual[key] is None
        else:
            assert clamped[key] == pytest.approx(manual[key], abs=1e-6)


def test_alignment_monotonicity():
    aligned = torch.tensor([[1.0, 1.0], [1.1, 0.9], [0.9, 1.05]])
    misaligned = torch.tensor([[1.0, -1.0], [-1.1, 0.9], [0.9, -1.05]])
    aligned_c1 = compute_C1(aligned, alignment="cosine")
    misaligned_c1 = compute_C1(misaligned, alignment="cosine")
    assert aligned_c1 > misaligned_c1


def test_trace_vs_frobenius_modes():
    matrix = torch.tensor([[1.0, 2.0], [2.0, -1.0], [0.5, 0.5]])
    trace_val = compute_C3(matrix, covariance_dispersion="trace")
    fro_val = compute_C3(matrix, covariance_dispersion="frobenius")
    assert fro_val != trace_val
