from __future__ import annotations

import logging
from typing import Dict, Optional

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


class ExperimentLogger:
    def __init__(self, config: Dict, project_config: Dict) -> None:
        self.config = config
        self.project_config = project_config
        self._wandb_run = None
        self._fallback = logging.getLogger("hirm_experiment")
        self._fallback.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        if not self._fallback.handlers:
            self._fallback.addHandler(handler)
        self._init_wandb()

    def _init_wandb(self) -> None:
        if not self.config.get("enabled", True):
            return
        if wandb is None:
            self._fallback.info("wandb not available; falling back to stdout logging")
            return
        mode = self.config.get("mode", "offline")
        wandb_settings = {"mode": mode}
        params = {
            "project": self.config.get("project", self.project_config.get("name", "hirm_experiment")),
            "entity": self.config.get("entity", None),
            "config": self.project_config,
            "settings": wandb.Settings(**wandb_settings),
        }
        group = self.config.get("group")
        if group:
            params["group"] = group
        tags = self.config.get("tags")
        if tags:
            params["tags"] = tags
        self._wandb_run = wandb.init(**params)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._wandb_run is not None:
            wandb.log(metrics, step=step)
        msg = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        prefix = f"step={step} " if step is not None else ""
        self._fallback.info("%s%s", prefix, msg)

    def finish(self) -> None:
        if self._wandb_run is not None:
            wandb.finish()
