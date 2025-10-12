"""Dual logging to Weights & Biases and local JSON/CSV mirrors."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional

import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


class RunLogger:
    def __init__(self, config: Dict, resolved_config: Dict):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        local_cfg = config.get("local_mirror", {})
        base_dir = Path(local_cfg.get("base_dir", "runs")) / timestamp
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.base_dir / local_cfg.get("metrics_file", "metrics.jsonl")
        self.final_metrics_path = self.base_dir / local_cfg.get("final_metrics_file", "final_metrics.json")
        self.config_path = self.base_dir / local_cfg.get("config_file", "config.yaml")
        self.checkpoint_dir = self.base_dir / local_cfg.get("checkpoints_dir", "checkpoints")
        self.artifacts_dir = self.base_dir / local_cfg.get("artifacts_dir", "artifacts")
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(resolved_config, f)
        self.metadata_path = self.base_dir / "metadata.json"
        system_info = self._system_info()
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(system_info, f, indent=2)
        self.provenance_path = self.base_dir / "run_provenance.json"
        self._registry_conn: Optional[sqlite3.Connection] = None
        try:
            provenance = _build_run_provenance(
                resolved_config,
                base_dir=self.base_dir,
                system_info=system_info,
            )
            with open(self.provenance_path, "w", encoding="utf-8") as handle:
                json.dump(provenance, handle, indent=2, sort_keys=True)
            self._registry_conn = _append_run_registry(
                self.artifacts_dir,
                provenance,
                self.provenance_path,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            warnings.warn(f"Failed to record run provenance: {exc}")
        self.wandb_run = None
        wandb_cfg = config.get("wandb", {})
        if wandb_cfg.get("enabled", False) and wandb is not None:
            mode = "offline" if wandb_cfg.get("offline_ok", False) and os.getenv("WANDB_MODE") == "offline" else None
            self.wandb_run = wandb.init(
                project=wandb_cfg.get("project", "invariant-hedging"),
                entity=wandb_cfg.get("entity"),
                config=resolved_config,
                mode=mode,
                dir=wandb_cfg.get("dir"),
            )
        self.metrics_file = open(self.metrics_path, "a", encoding="utf-8")

    def log_metrics(self, metrics: Dict, step: Optional[int] = None) -> None:
        record = {"step": step, **metrics}
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def log_probe(self, env_name: str, step: int, records: List[Dict[str, float]]) -> None:
        if not records:
            return
        probe_dir = self.artifacts_dir / "train" / f"{env_name}_probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        path = probe_dir / f"step_{step:06d}.json"
        payload = {"step": step, "env": env_name, "records": records}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def log_final(self, metrics: Dict) -> None:
        with open(self.final_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if self.wandb_run is not None:
            self.wandb_run.log({f"final/{k}": v for k, v in metrics.items()})

    def save_artifact(self, path: Path, name: Optional[str] = None) -> None:
        target = self.artifacts_dir / (name or path.name)
        if path.is_dir():
            if target.exists():
                return
            target.mkdir(parents=True, exist_ok=True)
        else:
            if path.resolve() == target.resolve():
                return
            target.write_bytes(path.read_bytes())
        if self.wandb_run is not None and wandb is not None:
            artifact = wandb.Artifact(name or path.name, type="file")
            artifact.add_file(str(path))
            self.wandb_run.log_artifact(artifact)

    def close(self) -> None:
        self.metrics_file.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
        if self._registry_conn is not None:
            try:
                self._registry_conn.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    def info(self) -> Dict[str, str]:
        return {"base_dir": str(self.base_dir)}

    def _system_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "git_commit": _get_git_commit(),
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": platform.platform(),
        }
        git_status = _get_git_status_clean()
        if git_status is not None:
            info["git_status_clean"] = git_status
        if torch is not None:
            info["torch_version"] = torch.__version__
            info["cuda_available"] = bool(torch.cuda.is_available())
        return info


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _get_git_status_clean() -> Optional[bool]:
    try:
        output = subprocess.check_output(["git", "status", "--short"], text=True).strip()
        return output == ""
    except Exception:  # pragma: no cover
        return None


def _jsonify(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(key): _jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _collect_hardware_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        info["memory_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:  # pragma: no cover - psutil optional
        pass
    return {key: value for key, value in info.items() if value is not None}


def _collect_cuda_info() -> Dict[str, object]:
    data: Dict[str, object] = {"available": bool(torch is not None and torch.cuda.is_available())}
    if not data["available"] or torch is None:
        return data
    try:
        data["version"] = getattr(torch.version, "cuda", None)
    except Exception:  # pragma: no cover - defensive
        data["version"] = None
    try:
        data["device_count"] = torch.cuda.device_count()
        devices: List[Dict[str, object]] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": getattr(props, "name", None),
                    "total_memory": getattr(props, "total_memory", None),
                    "compute_capability": f"{getattr(props, 'major', '?')}.{getattr(props, 'minor', '?')}",
                }
            )
        if devices:
            data["devices"] = devices
    except Exception:  # pragma: no cover - optional GPU metadata
        pass
    try:
        cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
        if cudnn_version:
            data["cudnn_version"] = cudnn_version
    except Exception:  # pragma: no cover
        pass
    return data


def _collect_seed_values(resolved_config: Mapping[str, object]) -> Dict[str, object]:
    seeds: Dict[str, object] = {}

    def _visit(node: object, path: List[str]) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_str = str(key)
                new_path = path + [key_str]
                lowered = key_str.lower()
                if "seed" in lowered and key_str not in ("seed_everything",):
                    seeds[".".join(new_path)] = _jsonify(value)
                _visit(value, new_path)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                _visit(value, path + [str(idx)])

    _visit(resolved_config, [])
    return seeds


def _discover_dataset_paths(data_cfg: Mapping[str, object]) -> List[str]:
    candidates: set[str] = set()

    def _walk(node: object, key_hint: str | None = None) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                lower = str(key).lower()
                if isinstance(value, str):
                    if any(token in lower for token in ("path", "dir", "root", "file")) or any(
                        sep in value for sep in ("/", "\\")
                    ):
                        candidates.add(value)
                _walk(value, str(key))
        elif isinstance(node, list):
            for value in node:
                _walk(value, key_hint)

    _walk(data_cfg)
    return sorted(candidates)


def _resolve_candidate_path(raw_path: str) -> Optional[Path]:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    potential = (repo_root / candidate).resolve()
    if potential.exists():
        return potential
    hydra_candidate = (Path.cwd() / candidate).resolve()
    if hydra_candidate.exists():
        return hydra_candidate
    return None


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_dataset_fingerprint(resolved_config: Mapping[str, object]) -> Dict[str, object]:
    data_cfg_raw = resolved_config.get("data") if isinstance(resolved_config, Mapping) else None
    if not isinstance(data_cfg_raw, Mapping):
        return {}
    data_cfg = _jsonify(data_cfg_raw)
    file_hashes: MutableMapping[str, str] = {}
    for raw_path in _discover_dataset_paths(data_cfg_raw):
        resolved = _resolve_candidate_path(raw_path)
        if resolved is None or not resolved.is_file():
            continue
        try:
            file_hashes[str(resolved)] = _hash_file(resolved)
        except OSError:
            continue
    payload = {
        "config": data_cfg,
        "files": dict(sorted(file_hashes.items())),
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload["hash"] = hashlib.sha256(serialized).hexdigest()
    return payload


def _config_sha(resolved_config: Mapping[str, object]) -> str:
    serializable = _jsonify(resolved_config)
    payload = json.dumps(serializable, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_run_provenance(
    resolved_config: Mapping[str, object],
    *,
    base_dir: Path,
    system_info: Mapping[str, object],
) -> Dict[str, object]:
    dataset = _build_dataset_fingerprint(resolved_config)
    provenance: Dict[str, object] = {
        "generated_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_dir": str(base_dir.resolve()),
        "git": {
            "commit": system_info.get("git_commit"),
            "status_clean": system_info.get("git_status_clean"),
        },
        "python": system_info.get("python"),
        "platform": system_info.get("platform"),
        "hardware": _collect_hardware_info(),
        "cuda": _collect_cuda_info(),
        "seeds": _collect_seed_values(resolved_config),
        "dataset": dataset,
        "config_hash": _config_sha(resolved_config),
    }
    return provenance


def _append_run_registry(
    artifacts_dir: Path,
    provenance: Mapping[str, object],
    provenance_path: Path,
) -> Optional[sqlite3.Connection]:
    try:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        db_path = artifacts_dir / "runs.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                git_commit TEXT,
                dataset_hash TEXT,
                seeds_json TEXT,
                provenance_path TEXT NOT NULL
            )
            """
        )
        dataset_hash = None
        dataset_section = provenance.get("dataset")
        if isinstance(dataset_section, Mapping):
            dataset_hash = dataset_section.get("hash")
        seeds_json = json.dumps(provenance.get("seeds", {}), sort_keys=True, default=str)
        conn.execute(
            """
            INSERT INTO runs (created_at, run_dir, git_commit, dataset_hash, seeds_json, provenance_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                provenance.get("generated_at"),
                provenance.get("run_dir"),
                provenance.get("git", {}).get("commit") if isinstance(provenance.get("git"), Mapping) else None,
                dataset_hash,
                seeds_json,
                str(provenance_path),
            ),
        )
        conn.commit()
        return conn
    except Exception as exc:  # pragma: no cover - registry best effort
        warnings.warn(f"Failed to update runs.sqlite registry: {exc}")
        return None


__all__ = ["RunLogger"]
