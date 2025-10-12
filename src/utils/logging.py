"""Dual logging to Weights & Biases and local JSON/CSV mirrors."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

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
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._system_info(), f, indent=2)
        self.wandb_run = None
        wandb_cfg = config.get("wandb", {})
        if wandb_cfg.get("enabled", False):
            if wandb is None:
                warnings.warn(
                    "Weights & Biases logging requested but the 'wandb' package is not installed. "
                    "Continuing with local logging only.",
                    RuntimeWarning,
                )
            else:
                mode = wandb_cfg.get("mode")
                if (
                    mode is None
                    and wandb_cfg.get("offline_ok", False)
                    and os.getenv("WANDB_MODE") == "offline"
                ):
                    mode = "offline"
                init_kwargs = {
                    "project": wandb_cfg.get("project", "invariant-hedging"),
                    "entity": wandb_cfg.get("entity"),
                    "config": resolved_config,
                    "dir": wandb_cfg.get("dir"),
                }
                if mode is not None:
                    init_kwargs["mode"] = mode
                if wandb_cfg.get("group") is not None:
                    init_kwargs["group"] = wandb_cfg.get("group")
                if wandb_cfg.get("tags") is not None:
                    init_kwargs["tags"] = wandb_cfg.get("tags")
                try:
                    self.wandb_run = wandb.init(**init_kwargs)
                except Exception as exc:  # pragma: no cover - warn and continue
                    warnings.warn(
                        f"Failed to initialize Weights & Biases logging ({exc!s}). "
                        "Continuing with local logging only.",
                        RuntimeWarning,
                    )
                    self.wandb_run = None
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
        source = path.resolve()
        destination = target.resolve()
        if source == destination:
            return
        if path.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if self.wandb_run is not None and wandb is not None:
            artifact_type = "directory" if path.is_dir() else "file"
            artifact = wandb.Artifact(name or path.name, type=artifact_type)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
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
