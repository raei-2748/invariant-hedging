"""Dual logging to Weights & Biases and local JSON/CSV mirrors."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from . import seed as seed_utils

REPO_ROOT = Path(__file__).resolve().parents[2]

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
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._system_info(), f, indent=2)
        self.provenance_path = self.base_dir / "run_provenance.json"
        with open(self.provenance_path, "w", encoding="utf-8") as f:
            json.dump(_build_run_provenance(timestamp, resolved_config), f, indent=2)
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
            if hasattr(torch.version, "cuda"):
                info["cuda_version"] = getattr(torch.version, "cuda")
        return info


def _build_run_provenance(timestamp: str, resolved_config: Dict) -> Dict[str, object]:
    provenance: Dict[str, object] = {
        "timestamp": timestamp,
        "git": {
            "commit": _get_git_commit(),
            "clean": _get_git_status_clean(),
        },
        "environment": _environment_snapshot(),
        "python_packages": _pip_freeze(),
        "config": resolved_config,
        "inputs": {
            "environment_yml": _read_optional(REPO_ROOT / "environment.yml"),
            "requirements_txt": _read_optional(REPO_ROOT / "requirements.txt"),
        },
    }
    conda_export = _conda_export()
    if conda_export is not None:
        provenance["conda_environment"] = conda_export
    determinism_state = _determinism_state()
    if determinism_state is not None:
        provenance["determinism"] = determinism_state
    return provenance




def _environment_snapshot() -> Dict[str, object]:
    info: Dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    if torch is not None:
        info["torch_version"] = torch.__version__
        if hasattr(torch.version, "cuda"):
            info["torch_cuda"] = getattr(torch.version, "cuda")
        info["cuda_available"] = bool(torch.cuda.is_available())
    return info


def _pip_freeze() -> List[str]:
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as exc:  # pragma: no cover - depends on external tooling
        return [f"<unavailable: {exc}>"]
    packages = [line.strip() for line in output.splitlines() if line.strip()]
    return sorted(packages)


def _conda_export() -> Optional[Dict[str, str]]:
    commands = []
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        commands.append((["micromamba", "env", "export", "-n", env_name], "micromamba"))
        commands.append((["conda", "env", "export", "-n", env_name, "--no-builds"], "conda"))
    commands.append((["conda", "env", "export", "--no-builds"], "conda"))
    commands.append((["micromamba", "env", "export"], "micromamba"))
    for cmd, name in commands:
        try:
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            return {"tool": name, "command": " ".join(cmd), "spec": output}
        except Exception:
            continue
    return None


def _determinism_state() -> Optional[Dict[str, int]]:
    try:
        state = seed_utils.last_state()
    except Exception:
        return None
    return state.to_dict()


def _read_optional(path: Path) -> Optional[str]:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


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
