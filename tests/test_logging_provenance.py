import json
from pathlib import Path

from src.utils import logging as log_utils
from src.utils import seed


def test_run_logger_writes_provenance(tmp_path, monkeypatch):
    seed.seed_everything(42)
    monkeypatch.setattr(log_utils.time, "strftime", lambda _: "20240101_010203")
    config = {"local_mirror": {"base_dir": str(tmp_path)}, "wandb": {"enabled": False}}
    resolved = {"foo": "bar"}
    logger = log_utils.RunLogger(config, resolved)
    try:
        run_dir = Path(logger.base_dir)
        provenance_path = run_dir / "run_provenance.json"
        assert provenance_path.exists()
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        assert payload["determinism"]["seed"] == 42
        assert payload["config"] == resolved
        assert "python_packages" in payload and payload["python_packages"]
        assert payload["git"]["commit"]
    finally:
        logger.close()
