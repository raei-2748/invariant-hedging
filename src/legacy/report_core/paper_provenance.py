"""Utilities for capturing reproducibility metadata for paper artifacts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional

try:  # pragma: no cover - torch optional at runtime
    import torch
except Exception:  # pragma: no cover - defer torch import failures
    torch = None  # type: ignore


@dataclass
class FileDigest:
    """Digest information for a tracked file."""

    path: Path
    sha256: str
    size: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size": self.size,
        }


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:  # pragma: no cover - git not guaranteed in tests
        return "unknown"


def _hash_file(path: Path) -> FileDigest:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return FileDigest(path=path, sha256=hasher.hexdigest(), size=path.stat().st_size)


def _pip_freeze() -> List[str]:
    packages: set[str] = set()
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        packages.add(f"{name}=={dist.version}")
    return sorted(packages)


def _torch_metadata() -> Mapping[str, object]:
    info: MutableMapping[str, object] = {
        "available": False,
        "cuda_available": False,
        "mps_built": False,
        "mps_available": False,
        "version": None,
        "cuda_version": None,
        "device_count": 0,
    }
    if torch is None:
        info["error"] = "torch not installed"
        return info
    info["version"] = torch.__version__
    info["available"] = bool(torch.cuda.is_available())
    info["cuda_available"] = info["available"]
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    if info["available"]:
        try:
            count = torch.cuda.device_count()
            info["device_count"] = count
            info["devices"] = [torch.cuda.get_device_name(i) for i in range(count)]
        except Exception as exc:  # pragma: no cover - cuda optional
            info["device_error"] = str(exc)
    backend = getattr(torch.backends, "mps", None)
    if backend is not None:
        try:
            info["mps_built"] = bool(getattr(backend, "is_built", lambda: True)())
        except Exception:  # pragma: no cover - defensive
            info["mps_built"] = False
        try:
            info["mps_available"] = bool(backend.is_available())
        except Exception as exc:  # pragma: no cover - defensive
            info["mps_error"] = str(exc)
    return info


def _describe_run(run_dir: Path) -> Mapping[str, object]:
    details: MutableMapping[str, object] = {
        "path": str(run_dir),
        "exists": run_dir.exists(),
    }
    if not run_dir.exists():
        return details
    tracked: Dict[str, FileDigest] = {}
    for candidate in (
        "config.yaml",
        "metrics.jsonl",
        "metadata.json",
        "provenance.json",
    ):
        path = run_dir / candidate
        if path.exists():
            tracked[candidate] = _hash_file(path)
    if tracked:
        details["files"] = {name: digest.to_dict() for name, digest in tracked.items()}
    checkpoints = run_dir / "checkpoints"
    if checkpoints.exists():
        ckpts: List[Mapping[str, object]] = []
        for ckpt in sorted(checkpoints.glob("*.pt")):
            digest = _hash_file(ckpt)
            ckpts.append(digest.to_dict())
        if ckpts:
            details["checkpoints"] = ckpts
    return details


def collect_provenance(run_dir: Optional[Path] = None) -> Mapping[str, object]:
    """Collect metadata required to reproduce paper experiments."""

    now_utc = dt.datetime.now(dt.timezone.utc)
    payload: MutableMapping[str, object] = {
        "generated_at": now_utc.isoformat(timespec="seconds"),
        "git_hash": _git_hash(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "pip": _pip_freeze(),
        "torch": _torch_metadata(),
    }
    if run_dir is not None:
        payload["run"] = _describe_run(run_dir)
    return payload


def write_provenance(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = ["collect_provenance", "write_provenance"]
