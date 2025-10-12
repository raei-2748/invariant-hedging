"""Canonical tagging helpers for dataset provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

from ..data.types import EpisodeBatch


def build_episode_tags(
    *,
    source: str,
    split: str,
    regime: str,
    seed: int,
    stress_jump: bool,
    stress_liquidity: bool,
) -> Dict[str, str | int | bool]:
    """Return a canonical set of tags for synthetic simulation artefacts."""

    return {
        "source": str(source),
        "split": str(split),
        "regime_name": str(regime),
        "seed": int(seed),
        "stress_jump": bool(stress_jump),
        "stress_liquidity": bool(stress_liquidity),
    }


@dataclass(frozen=True)
class EpisodeTag:
    """Structured metadata describing a single market episode."""

    episode_id: int
    regime_name: str
    split: str
    source: str
    start_date: str
    end_date: str
    symbol_root: str
    seed: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dictionary representation of the tag."""

        payload: Dict[str, Any] = {
            "episode_id": int(self.episode_id),
            "regime_name": self.regime_name,
            "split": self.split,
            "source": self.source,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "symbol_root": self.symbol_root,
        }
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        if self.extra:
            payload.update(dict(self.extra))
        return payload


def build_episode_tag(
    *,
    episode_id: int,
    regime_name: str,
    split: str,
    source: str,
    start_date: str,
    end_date: str,
    symbol_root: str,
    seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> EpisodeTag:
    """Factory helper used by loaders when emitting episode tags."""

    return EpisodeTag(
        episode_id=episode_id,
        regime_name=regime_name,
        split=split,
        source=source,
        start_date=start_date,
        end_date=end_date,
        symbol_root=symbol_root,
        seed=seed,
        extra=extra or {},
    )


def attach_tags(batch: EpisodeBatch, tags: Iterable[EpisodeTag]) -> EpisodeBatch:
    """Return a copy of ``batch`` with canonical tag metadata injected."""

    tag_payload: List[Dict[str, Any]] = [tag.to_dict() for tag in tags]
    meta: MutableMapping[str, Any] = dict(batch.meta)
    meta["episode_tags"] = tag_payload
    return EpisodeBatch(
        spot=batch.spot,
        option_price=batch.option_price,
        implied_vol=batch.implied_vol,
        time_to_maturity=batch.time_to_maturity,
        rate=batch.rate,
        env_name=batch.env_name,
        meta=dict(meta),
    )


def extract_episode_tags(batch: EpisodeBatch) -> List[Dict[str, Any]]:
    """Retrieve the canonical tag metadata from a batch if present."""

    tags = batch.meta.get("episode_tags", [])
    if isinstance(tags, list):
        return list(tags)
    return []


__all__ = [
    "EpisodeTag",
    "attach_tags",
    "build_episode_tag",
    "build_episode_tags",
    "extract_episode_tags",
]
