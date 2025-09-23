import torch

from src.utils import seed


def test_seed_everything_deterministic():
    gen1 = seed.seed_everything(1234)
    rand1 = torch.randn(5, generator=gen1)
    gen2 = seed.seed_everything(1234)
    rand2 = torch.randn(5, generator=gen2)
    assert torch.allclose(rand1, rand2)
