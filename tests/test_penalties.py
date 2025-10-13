import torch

from hirm.objectives import penalties


def _env_losses(param: torch.Tensor) -> torch.Tensor:
    return torch.stack([(param - 2.0) ** 2, (param + 1.0) ** 2])


def test_irm_penalty_reduces_variance():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.SGD([param], lr=1e-4)

    initial_var = _env_losses(param.detach()).var(unbiased=False).item()
    for _ in range(200):
        optimizer.zero_grad()
        dummy = torch.tensor(1.0, requires_grad=True)
        env_losses = [((param - 2.0) * dummy) ** 2, ((param + 1.0) * dummy) ** 2]
        mean_loss = torch.stack(env_losses).mean()
        irm_pen = penalties.irm_penalty(env_losses, dummy)
        total = mean_loss + 1e-3 * irm_pen
        total.backward()
        optimizer.step()

    final_var = _env_losses(param.detach()).var(unbiased=False).item()
    assert final_var < initial_var


def test_vrex_penalty_reduces_variance():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.SGD([param], lr=0.02)

    initial_var = _env_losses(param.detach()).var(unbiased=False).item()
    for _ in range(200):
        optimizer.zero_grad()
        env_losses = _env_losses(param)
        mean_loss = env_losses.mean()
        vrex_pen = penalties.vrex_penalty(env_losses)
        total = mean_loss + 1.0 * vrex_pen
        total.backward()
        optimizer.step()

    final_var = _env_losses(param.detach()).var(unbiased=False).item()
    assert final_var < initial_var
