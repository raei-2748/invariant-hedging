"""Dual logging to Weights & Biases and local JSON/CSV mirrors."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import yaml

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

    def _system_info(self) -> Dict[str, str]:
        info = {}
        try:
            import subprocess

            info["git_commit"] = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            )
            info["git_status_clean"] = (
                subprocess.check_output(["git", "status", "--short"], text=True).strip() == ""
            )
        except Exception:  # pragma: no cover
            info["git_commit"] = "unknown"
        info["python"] = f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        return info
