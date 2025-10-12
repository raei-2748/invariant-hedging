"""Definitions for deterministic real-market anchor windows."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Mapping, Sequence

import pandas as pd

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for older Python
    from pytz import timezone as ZoneInfo  # type: ignore

NY_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class AnchorSpec:
    """Canonical real-market regime definition."""

    name: str
    split: str
    start: datetime
    end: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)

    @staticmethod
    def from_config(cfg: Mapping[str, object]) -> "AnchorSpec":
        required = {"name", "split", "start", "end"}
        missing = required - set(cfg.keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise KeyError(f"Anchor configuration missing fields: {missing_str}")
        start = _parse_date(cfg["start"])
        end = _parse_date(cfg["end"])
        if end < start:
            raise ValueError(f"Anchor '{cfg['name']}' has end before start")
        metadata = {k: v for k, v in cfg.items() if k not in required}
        return AnchorSpec(
            name=str(cfg["name"]),
            split=str(cfg["split"]),
            start=start,
            end=end,
            metadata=metadata,
        )


@dataclass(frozen=True)
class EpisodeWindow:
    """Single rolling episode generated from an anchor."""

    anchor: AnchorSpec
    episode_id: int
    start: datetime
    end: datetime
    start_index: int
    end_index: int

    def as_dict(self) -> Mapping[str, object]:
        return {
            "anchor": self.anchor.name,
            "split": self.anchor.split,
            "episode_id": self.episode_id,
            "start_date": self.start.strftime("%Y-%m-%d"),
            "end_date": self.end.strftime("%Y-%m-%d"),
            "start_index": self.start_index,
            "end_index": self.end_index,
        }


def _parse_date(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=NY_TZ)
        return value.astimezone(NY_TZ)
    parsed = pd.Timestamp(value).to_pydatetime()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=NY_TZ)
    else:
        parsed = parsed.astimezone(NY_TZ)
    return parsed


def parse_anchor_specs(config: Sequence[Mapping[str, object]]) -> List[AnchorSpec]:
    """Parse raw mappings into :class:`AnchorSpec` objects."""

    return [AnchorSpec.from_config(item) for item in config]


def validate_non_overlapping(anchors: Iterable[AnchorSpec]) -> None:
    """Ensure anchors from different splits do not overlap in time."""

    anchors_list = list(anchors)
    for i, left in enumerate(anchors_list):
        for right in anchors_list[i + 1 :]:
            if left.split == right.split:
                continue
            if _overlap(left.start, left.end, right.start, right.end):
                raise ValueError(
                    f"Anchors '{left.name}' ({left.split}) and '{right.name}' ({right.split}) overlap"
                )


def _overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    latest_start = max(a_start, b_start)
    earliest_end = min(a_end, b_end)
    return latest_start <= earliest_end


def generate_episode_windows(
    anchor: AnchorSpec,
    trading_days: pd.DatetimeIndex,
    episode_days: int,
    stride_days: int,
) -> List[EpisodeWindow]:
    """Produce rolling windows for an anchor using length/stride parameters."""

    if episode_days <= 0:
        raise ValueError("episode_days must be positive")
    if stride_days <= 0:
        raise ValueError("stride_days must be positive")
    mask = (trading_days >= anchor.start) & (trading_days <= anchor.end)
    subset = trading_days[mask]
    windows: List[EpisodeWindow] = []
    if len(subset) < episode_days:
        return windows
    start_positions = range(0, len(subset) - episode_days + 1, stride_days)
    for episode_index, start_offset in enumerate(start_positions):
        end_offset = start_offset + episode_days - 1
        start_ts = subset[start_offset]
        end_ts = subset[end_offset]
        start_idx = trading_days.get_loc(start_ts)
        end_idx = trading_days.get_loc(end_ts)
        windows.append(
            EpisodeWindow(
                anchor=anchor,
                episode_id=episode_index,
                start=start_ts.to_pydatetime(),
                end=end_ts.to_pydatetime(),
                start_index=start_idx,
                end_index=end_idx,
            )
        )
    return windows


def episode_index_frame(
    anchors: Sequence[AnchorSpec],
    trading_days: pd.DatetimeIndex,
    episode_days: int,
    stride_days: int,
) -> List[EpisodeWindow]:
    """Convenience helper returning all windows for all anchors."""

    output: List[EpisodeWindow] = []
    for anchor in anchors:
        output.extend(generate_episode_windows(anchor, trading_days, episode_days, stride_days))
    return output
