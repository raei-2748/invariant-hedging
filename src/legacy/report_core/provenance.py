"""Provenance utilities for aggregation outputs."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

from .aggregate import AggregateResult, hash_config_section


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - git optional in tests
        return "unknown"


def build_manifest(
    result: AggregateResult,
    report_config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_hash": _git_hash(),
        "config_hash": hash_config_section(report_config),
        "regimes": result.regimes,
        "ci_level": report_config.get("confidence_level"),
        "seed_files": [str(sel.diagnostics_path) for sel in result.selected_seeds],
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


__all__ = ["build_manifest", "write_manifest"]
