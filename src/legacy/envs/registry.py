"""Environment registry for synthetic crisis regimes."""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from src.core.infra.tags import build_episode_tags


def _seed_offset(label: str) -> int:
    return zlib.adler32(label.encode("utf-8")) & 0xFFFFFFFF


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    vix_min: float | None
    vix_max: float | None
    stress_jump: bool
    stress_liquidity: bool
    seed_offset: int


class SyntheticRegimeRegistry:
    """Registry describing synthetic regimes and their stress toggles."""

    def __init__(self, config: Mapping[str, object]):
        data_cfg = config.get("data", {}) if isinstance(config, Mapping) else {}
        regimes_cfg = data_cfg.get("regimes", {}) if isinstance(data_cfg, Mapping) else {}
        bands: Iterable[Mapping[str, object]] = regimes_cfg.get("bands", [])  # type: ignore[assignment]

        stress_cfg = data_cfg.get("stress", {}) if isinstance(data_cfg, Mapping) else {}
        jump_cfg = stress_cfg.get("jump", {}) if isinstance(stress_cfg, Mapping) else {}
        liquidity_cfg = stress_cfg.get("liquidity", {}) if isinstance(stress_cfg, Mapping) else {}

        jump_enabled = bool(jump_cfg.get("enabled", False))
        jump_regimes = set(jump_cfg.get("apply_to", [])) if jump_enabled else set()

        liquidity_enabled = bool(liquidity_cfg.get("enabled", False))
        liquidity_regimes = set(liquidity_cfg.get("apply_to", [])) if liquidity_enabled else set()

        self._specs: Dict[str, RegimeSpec] = {}
        for band in bands or []:
            name = str(band.get("name"))
            if not name:
                continue
            spec = RegimeSpec(
                name=name,
                vix_min=float(band.get("vix_min")) if band.get("vix_min") is not None else None,
                vix_max=float(band.get("vix_max")) if band.get("vix_max") is not None else None,
                stress_jump=name in jump_regimes,
                stress_liquidity=name in liquidity_regimes,
                seed_offset=_seed_offset(name),
            )
            self._specs[name] = spec

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __iter__(self):
        return iter(self._specs.values())

    def get(self, name: str) -> RegimeSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown synthetic regime '{name}'")
        return self._specs[name]

    def tags_for(self, *, name: str, split: str, seed: int) -> Dict[str, str | int | bool]:
        spec = self.get(name)
        return build_episode_tags(
            source="sim",
            split=split,
            regime=name,
            seed=seed,
            stress_jump=spec.stress_jump,
            stress_liquidity=spec.stress_liquidity,
        )

    @property
    def names(self) -> List[str]:
        return list(self._specs.keys())
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

from src.modules.data.real.loader import RealAnchorLoader
from src.modules.data.types import EpisodeBatch
from src.core.infra.tags import extract_episode_tags


@dataclass(frozen=True)
class EnvironmentSpec:
    """Descriptor for a prepared environment anchored to a regime."""

    name: str
    split: str
    regime_name: str
    batch: EpisodeBatch
    tags: List[Mapping[str, object]]


def register_real_anchors(
    config: Mapping[str, object],
    *,
    include: Optional[Iterable[str]] = None,
) -> List[EnvironmentSpec]:
    """Materialise anchor environments from a configuration mapping.

    Parameters
    ----------
    config:
        Mapping compatible with ``config/real_anchors.yaml``.
    include:
        Optional iterable of anchor names to restrict the registry to.
    """

    loader = RealAnchorLoader(config)
    loaded = loader.load()
    if include is not None:
        include = list(include)
        missing = [name for name in include if name not in loaded]
        if missing:
            raise KeyError(f"Unknown anchor names requested: {missing}")
        order = include
    else:
        order = [anchor.name for anchor in loader.anchors if anchor.name in loaded]
    specs: List[EnvironmentSpec] = []
    for name in order:
        item = loaded[name]
        tags = extract_episode_tags(item.batch)
        specs.append(
            EnvironmentSpec(
                name=name,
                split=item.anchor.split,
                regime_name=item.anchor.name,
                batch=item.batch,
                tags=tags,
            )
        )
    return specs
