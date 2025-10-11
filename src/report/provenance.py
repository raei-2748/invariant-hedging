"""Provenance tracking for aggregated reporting artefacts."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _config_hash(config: Mapping[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="utf-8").strip()
    except Exception:  # pragma: no cover - guard for environments outside git repos
        return "unknown"


def write_manifest(
    output_dir: Path,
    *,
    config: Mapping[str, object],
    runs: Sequence[Path],
    regimes_order: Sequence[str],
    confidence_level: float,
    ire_metadata: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
        "git_hash": _git_hash(),
        "config_hash": _config_hash(config),
        "runs": [str(p) for p in runs],
        "regimes_order": list(regimes_order),
        "confidence_level": confidence_level,
    }
    if ire_metadata:
        manifest["ire3d"] = dict(ire_metadata)

    manifest_path = output_dir / "aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
