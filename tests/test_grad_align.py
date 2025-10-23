import torch

from src.core.losses import env_variance, normalized_head_grads, pairwise_cosine


def test_normalized_head_grads_preserve_shape_and_scale():
    torch.manual_seed(0)
    grad_blocks = [torch.randn(2, 3), torch.randn(4)]
    env0 = [block.clone() for block in grad_blocks]
    env1 = [2.5 * block for block in grad_blocks]
    env2 = [torch.zeros_like(block) for block in grad_blocks]

    normals = normalized_head_grads([env0, env1, env2])

    expected_dim = sum(block.numel() for block in grad_blocks)
    assert normals.shape == (3, expected_dim)
    assert torch.allclose(normals[0], normals[1], atol=1e-6)
    assert torch.allclose(normals[0].norm(), torch.tensor(1.0))
    assert torch.count_nonzero(normals[2]) == 0


def test_pairwise_cosine_alignment_monotonicity():
    aligned = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    misaligned = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    aligned_cos = pairwise_cosine(aligned)
    misaligned_cos = pairwise_cosine(misaligned)

    assert aligned_cos.shape == (3,)
    assert misaligned_cos.shape == (3,)
    assert torch.all(aligned_cos <= 1.0)
    assert torch.all(misaligned_cos >= -1.0)
    assert aligned_cos.mean() > misaligned_cos.mean()


def test_env_variance_matches_tensor_var():
    values = [torch.tensor(1.5), torch.tensor(3.5), torch.tensor(5.5)]
    high_dim = [torch.ones(4, 2), 2 * torch.ones(4, 2)]

    expected = torch.tensor(values).var(unbiased=False)
    computed = env_variance(values)
    assert torch.allclose(computed, expected)

    high_dim_expected = torch.stack([tensor.mean() for tensor in high_dim]).var(unbiased=False)
    assert torch.allclose(env_variance(high_dim), high_dim_expected)

    assert env_variance([torch.tensor(7.0)]).item() == 0.0
