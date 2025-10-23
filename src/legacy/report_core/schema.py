"""Schema and validation utilities for ``final_metrics.json`` outputs."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping


SCHEMA_VERSION = "1.0.0"


class FinalMetricsValidationError(ValueError):
    """Raised when a ``final_metrics.json`` payload fails validation."""


@dataclass(frozen=True)
class FinalMetricsPayload:
    """Canonical representation of a ``final_metrics.json`` payload."""

    schema_version: str
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any]


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid metrics")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError(f"Metric value '{value}' is not numeric") from exc
    else:
        raise TypeError(f"Unsupported metric value type: {type(value)!r}")
    if not math.isfinite(result):
        raise ValueError("Metric values must be finite numbers")
    return result


def _extract_metrics(mapping: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    metrics: dict[str, float] = {}
    metadata: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in {"metrics", "schema_version"}:
            continue
        if isinstance(value, Mapping):
            # Nested metadata
            metadata[key] = value
            continue
        try:
            metrics[key] = _coerce_float(value)
        except (TypeError, ValueError):
            metadata[key] = value
    if not metrics:
        raise FinalMetricsValidationError("No scalar metrics detected in payload")
    return metrics, metadata


def validate_final_metrics(payload: Mapping[str, Any]) -> FinalMetricsPayload:
    """Validate and normalise a ``final_metrics.json`` payload."""

    if not isinstance(payload, Mapping):
        raise FinalMetricsValidationError("Final metrics payload must be a mapping")

    schema_version = str(payload.get("schema_version", SCHEMA_VERSION))
    metrics_section = payload.get("metrics")

    if isinstance(metrics_section, Mapping):
        metrics: dict[str, float] = {}
        for name, raw_value in metrics_section.items():
            if isinstance(raw_value, Mapping):
                if "value" not in raw_value:
                    raise FinalMetricsValidationError(
                        f"Metric '{name}' object must provide a 'value' field"
                    )
                value = raw_value["value"]
            else:
                value = raw_value
            try:
                metrics[name] = _coerce_float(value)
            except (TypeError, ValueError) as exc:
                raise FinalMetricsValidationError(str(exc)) from exc
        metadata: MutableMapping[str, Any] = {
            key: value
            for key, value in payload.items()
            if key not in {"metrics", "schema_version"}
        }
        if not metrics:
            raise FinalMetricsValidationError("Metrics section must not be empty")
        return FinalMetricsPayload(schema_version=schema_version, metrics=metrics, metadata=dict(metadata))

    metrics, metadata = _extract_metrics(payload)
    return FinalMetricsPayload(schema_version=schema_version, metrics=metrics, metadata=metadata)


def load_final_metrics(path: Path) -> FinalMetricsPayload:
    """Load and validate a ``final_metrics.json`` payload from ``path``."""

    with open(path, "r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise FinalMetricsValidationError(f"Invalid JSON in {path}") from exc
    return validate_final_metrics(payload)


__all__ = [
    "FinalMetricsPayload",
    "FinalMetricsValidationError",
    "SCHEMA_VERSION",
    "load_final_metrics",
    "validate_final_metrics",
]
