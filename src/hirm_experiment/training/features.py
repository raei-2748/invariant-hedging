from __future__ import annotations

from typing import Dict, List

import torch

from hirm_experiment.data.dataset import EpisodeBatch, FeatureStats


class FeatureBuilder:
    def __init__(self, stats: Dict[str, FeatureStats], invariants: List[str], spurious: List[str]) -> None:
        self.stats = stats
        self.invariants = invariants
        self.spurious = spurious
        self.tau_value = torch.tensor(60.0 / 252.0)

    def step_features(self, batch: EpisodeBatch, t: int, inventory: torch.Tensor) -> torch.Tensor:
        components: List[torch.Tensor] = []
        for name in self.invariants + self.spurious:
            components.append(self._feature(name, batch, t, inventory).unsqueeze(-1))
        return torch.cat(components, dim=-1)

    def _feature(self, name: str, batch: EpisodeBatch, t: int, inventory: torch.Tensor) -> torch.Tensor:
        if name == "delta":
            return self.stats["delta"].normalize(batch.delta[:, t])
        if name == "gamma":
            return self.stats["gamma"].normalize(batch.gamma[:, t])
        if name in {"time_to_maturity", "time_to_maturity_days"}:
            raw = torch.full_like(batch.delta[:, t], self.tau_value.item())
            return self.stats["tau"].normalize(raw)
        if name == "inventory":
            return self.stats["inventory"].normalize(inventory)
        if name in {"realized_volatility", "realized_vol"}:
            return self.stats["realized_vol"].normalize(batch.realized_vol[:, t])
        raise ValueError(f"Unsupported feature: {name}")
