"""Utility helpers for safe diagnostic evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypeVar

import torch

_T = TypeVar("_T")


def _detach_value(value: Any) -> Any:
    """Return a detached version of ``value`` suitable for logging."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.numel() == 1:
            return float(tensor.item())
        return tensor
    if isinstance(value, Mapping):
        return {key: _detach_value(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_value(val) for val in value)
    if isinstance(value, list):
        return [_detach_value(val) for val in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_detach_value(val) for val in value]
    return value


def detach_diagnostics(payload: _T) -> _T:
    """Detach a diagnostics payload from the autograd graph.

    The helper supports arbitrarily nested mappings and sequences whose leaves
    may be ``torch.Tensor`` instances.  Scalars are converted to Python ``float``
    objects when possible so downstream logging utilities can consume the
    diagnostics without triggering autograd or dtype conversions.
    """

    return _detach_value(payload)


def safe_eval_metric(
    metric_fn: Callable[..., _T], *metric_args: Any, **metric_kwargs: Any
) -> _T:
    """Evaluate ``metric_fn`` on detached arguments.

    All tensor arguments (including those nested in sequences or mappings) are
    first detached from the autograd graph and then passed to ``metric_fn``.  The
    computation itself is wrapped in ``torch.no_grad()`` to guarantee the metric
    evaluation never contributes to gradient history.  The return value is
    similarly detached via :func:`detach_diagnostics` which converts scalar
    tensors to Python floats for convenience.
    """

    detached_args = [_detach_value(arg) for arg in metric_args]
    detached_kwargs = {key: _detach_value(val) for key, val in metric_kwargs.items()}
    with torch.no_grad():
        result = metric_fn(*detached_args, **detached_kwargs)
    return detach_diagnostics(result)


__all__ = ["detach_diagnostics", "safe_eval_metric"]
