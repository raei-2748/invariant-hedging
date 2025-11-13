"""Hydra entrypoint for the reporting aggregation pipeline."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from invariant_hedging import get_repo_root

CONFIG_DIR = Path(get_repo_root()) / "configs"
_SCRIPT = Path(get_repo_root()) / "tools" / "scripts" / "aggregate.py"


@hydra.main(config_path=str(CONFIG_DIR), config_name="report/default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Render reporting assets using the legacy aggregation harness."""

    if not _SCRIPT.exists():  # pragma: no cover - defensive
        raise FileNotFoundError(f"Reporting harness missing: {_SCRIPT}")
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as handle:
        OmegaConf.save(cfg=cfg, f=handle.name)
        config_path = Path(handle.name)
    try:
        subprocess.run(["python", str(_SCRIPT), "--config", str(config_path)], check=True)
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover
    main()
