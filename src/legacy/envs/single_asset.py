"""Single-asset hedging environment with daily rebalancing."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import torch

from src.data.features import FeatureEngineer
from src.data.types import EpisodeBatch
from src.markets.costs import execution_cost


@dataclass
class SimulationOutput:
    pnl: torch.Tensor
    turnover: torch.Tensor
    underlying_pnl: torch.Tensor
    option_pnl: torch.Tensor
    costs: torch.Tensor
    positions: torch.Tensor
    step_pnl: torch.Tensor
    max_trade: torch.Tensor
    probe: Optional[List[dict[str, float]]] = None
    representations: Optional[torch.Tensor] = None


class SingleAssetHedgingEnv:
    def __init__(self, env_index: int, batch: EpisodeBatch, feature_engineer: FeatureEngineer):
        self.env_index = env_index
        self.batch = batch
        self.feature_engineer = feature_engineer
        self.cost_config = {
            "linear_bps": batch.meta.get("linear_bps", 0.0),
            "quadratic": batch.meta.get("quadratic", 0.0),
            "slippage_multiplier": batch.meta.get("slippage_multiplier", 1.0),
        }
        self.notional = batch.meta.get("notional", 1.0)
        debug_flag = batch.meta.get("debug_probe")
        if debug_flag is None:
            debug_flag = os.getenv("HIRM_DEBUG_PROBE", "0") not in {"", "0", "false", "False"}
        self._debug_probe_enabled = bool(debug_flag)
        self._debug_episode_logged = False

    def sample_indices(self, batch_size: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        num = self.batch.spot.shape[0]
        if generator is None:
            return torch.randint(0, num, (batch_size,))
        return torch.randint(0, num, (batch_size,), generator=generator)

    def simulate(
        self,
        policy,
        indices: torch.Tensor,
        device: torch.device,
        representation_scale: Optional[torch.Tensor] = None,
        collect_representation: bool = False,
    ) -> SimulationOutput:
        sub_batch = self.batch.subset(indices.tolist()).to(device)
        steps = sub_batch.steps
        batch_size = sub_batch.spot.shape[0]
        spot = sub_batch.spot
        option = sub_batch.option_price
        base_features = self.feature_engineer.base_features(sub_batch).to(device)
        scaler = self.feature_engineer.scaler
        if scaler is None:
            raise RuntimeError("Feature engineer must be fit before simulation.")
        mean = scaler.mean.to(device)
        std = torch.clamp(scaler.std.to(device), min=1e-6)

        positions = torch.zeros(batch_size, steps + 1, device=device)
        pnl = torch.zeros(batch_size, device=device)
        underlying_pnl = torch.zeros_like(pnl)
        option_pnl = torch.zeros_like(pnl)
        costs = torch.zeros_like(pnl)
        turnover = torch.zeros_like(pnl)
        step_pnl = torch.zeros(batch_size, steps, device=device)
        max_trade = torch.zeros(batch_size, device=device)

        probe_records: List[dict[str, float]] = []
        representations: Optional[List[torch.Tensor]] = [] if collect_representation else None

        for t in range(steps):
            inv = positions[:, t] / self.notional
            feat_t = torch.cat([base_features[:, t, :], inv.unsqueeze(-1)], dim=-1)
            feat_t = (feat_t - mean) / std
            out = policy(feat_t, env_index=self.env_index, representation_scale=representation_scale)
            action = out["action"].squeeze(-1)
            raw_action = out.get("raw_action")
            if raw_action is not None:
                raw_action = raw_action.squeeze(-1)
            if collect_representation and representations is not None and "representation" in out:
                representations.append(out["representation"])
            positions[:, t + 1] = action
            trade = positions[:, t + 1] - positions[:, t]
            cost_t = execution_cost(trade, spot[:, t], self.cost_config)
            costs += cost_t
            turnover += torch.abs(trade)
            max_trade = torch.maximum(max_trade, torch.abs(trade))
            underlying_step = positions[:, t] * (spot[:, t + 1] - spot[:, t])
            option_step = -(option[:, t + 1] - option[:, t])
            underlying_pnl += underlying_step
            option_pnl += option_step
            step_value = underlying_step + option_step - cost_t
            step_pnl[:, t] = step_value
            pnl += step_value

            if (
                self._debug_probe_enabled
                and not self._debug_episode_logged
                and positions.shape[0] > 0
                and t < 20
            ):
                episode_idx = 0
                price = spot[episode_idx, t]
                prev_position = positions[episode_idx, t]
                target_position = positions[episode_idx, t + 1]
                trade_val = trade[episode_idx]
                cost_val = cost_t[episode_idx]
                if raw_action is not None:
                    raw_val = raw_action[episode_idx]
                else:
                    raw_val = target_position
                probe_records.append(
                    {
                        "t": int(t),
                        "price": float(price.item()),
                        "inventory": float((prev_position / max(self.notional, 1e-6)).item()),
                        "raw_output": float(raw_val.item()),
                        "position": float(target_position.item()),
                        "trade": float(trade_val.item()),
                        "cost": float(cost_val.item()),
                    }
                )
                if t == 19:
                    self._debug_episode_logged = True

        # Liquidate residual position at final spot price with same cost structure
        final_trade = -positions[:, -1]
        liquidation_cost = execution_cost(final_trade, spot[:, -1], self.cost_config)
        costs += liquidation_cost
        turnover += torch.abs(final_trade)
        max_trade = torch.maximum(max_trade, torch.abs(final_trade))
        pnl -= liquidation_cost
        step_pnl = torch.cat([step_pnl, (-liquidation_cost).unsqueeze(-1)], dim=1)

        if self._debug_probe_enabled and not self._debug_episode_logged:
            self._debug_episode_logged = True

        if self._debug_probe_enabled and probe_records:
            self._debug_episode_logged = True

        rep_tensor: Optional[torch.Tensor] = None
        if collect_representation and representations:
            rep_tensor = torch.stack(representations, dim=1)

        return SimulationOutput(
            pnl=pnl,
            turnover=turnover,
            underlying_pnl=underlying_pnl,
            option_pnl=option_pnl,
            costs=costs,
            positions=positions,
            step_pnl=step_pnl,
            max_trade=max_trade,
            probe=probe_records if probe_records else None,
            representations=rep_tensor,
        )
