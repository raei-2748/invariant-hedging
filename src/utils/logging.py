"""Dual logging to Weights & Biases and local JSON/CSV mirrors."""
from __future__ import annotations

import csv
import json
import os
import platform
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

    def log_diagnostics_row(self, row: Dict[str, object]) -> None:
        fieldnames = [
            "seed",
            "method",
            "env",
            "window",
            "CVaR95",
            "mean_pnl",
            "sortino",
            "turnover",
            "IG",
            "WG",
            "VR",
            "ER",
            "TR",
            "C1",
            "C2",
            "C3",
            "ISI",
        ]
        diagnostics_dir = self.base_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        path = diagnostics_dir / "per_seed.csv"
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            ordered = {key: row.get(key) for key in fieldnames}
            writer.writerow(ordered)

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
