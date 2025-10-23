from __future__ import annotations

import torch

from src.modules.head_invariance import cosine_alignment_penalty


def test_cosine_alignment_penalty_is_zero_for_aligned_vectors() -> None:
    grads = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])]
    penalty = cosine_alignment_penalty(grads)
    assert torch.isclose(penalty, torch.tensor(0.0))


def test_cosine_alignment_penalty_is_positive_for_opposing_vectors() -> None:
    grads = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    penalty = cosine_alignment_penalty(grads)
    assert penalty > 0
