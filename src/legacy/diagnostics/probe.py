"""Utilities for constructing held-out diagnostic probe batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Mapping, MutableMapping, Sequence

import torch


ProbeBatch = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class ProbeConfig:
    batch_size: int
    n_batches: int
    envs: Sequence[str]
    seed: int = 0


def _default_sampler(
    dataset: Sequence[ProbeBatch],
    batch_size: int,
    n_batches: int,
    generator: torch.Generator,
) -> Iterator[ProbeBatch]:
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    cursor = 0
    for _ in range(n_batches):
        if cursor >= len(indices):
            break
        batch_indices = indices[cursor : cursor + batch_size]
        cursor += batch_size
        selected = [dataset[idx] for idx in batch_indices]
        if not selected:
            continue
        merged: MutableMapping[str, List[torch.Tensor]] = {}
        for sample in selected:
            for key, value in sample.items():
                merged.setdefault(key, []).append(value)
        yield {key: torch.stack(values) for key, values in merged.items()}


def get_diagnostic_batches(
    split_cfg: ProbeConfig,
    env_sources: Mapping[str, object],
    sampler: Callable[[Sequence[ProbeBatch], int, int, torch.Generator], Iterator[ProbeBatch]]
    | None = None,
) -> Dict[str, List[ProbeBatch]]:
    """Build deterministic diagnostic batches for each environment.

    The function is intentionally permissive: ``env_sources`` may contain
    sequences of already materialised samples, callables that yield batches, or
    objects exposing a ``diagnostic_probe`` method.  The caller can also supply
    a custom sampler when working with bespoke data structures.
    """

    sampler = sampler or _default_sampler
    generator = torch.Generator()
    generator.manual_seed(int(split_cfg.seed))

    batches: Dict[str, List[ProbeBatch]] = {}
    for env_id in split_cfg.envs:
        source = env_sources.get(env_id)
        if source is None:
            batches[env_id] = []
            continue

        if hasattr(source, "diagnostic_probe"):
            probe_callable = getattr(source, "diagnostic_probe")
            collected = list(
                probe_callable(
                    batch_size=split_cfg.batch_size,
                    n_batches=split_cfg.n_batches,
                    generator=generator,
                )
            )
            batches[env_id] = collected
            continue

        if callable(source):
            collected = list(
                source(
                    batch_size=split_cfg.batch_size,
                    n_batches=split_cfg.n_batches,
                    generator=generator,
                )
            )
            batches[env_id] = collected
            continue

        if isinstance(source, Sequence):
            collected = list(
                sampler(source, split_cfg.batch_size, split_cfg.n_batches, generator)
            )
            batches[env_id] = collected
            continue

        raise TypeError(f"Unsupported diagnostic source for env '{env_id}'")

    return batches


__all__ = ["ProbeConfig", "get_diagnostic_batches", "ProbeBatch"]

