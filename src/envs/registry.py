"""Registry helpers for deterministic real-market anchors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

from ..data.real.loader import RealAnchorLoader
from ..data.types import EpisodeBatch
from ..infra.tags import extract_episode_tags


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
