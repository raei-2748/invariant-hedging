"""Tests for the RunLogger utility."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils import logging as logging_utils


@pytest.mark.parametrize("use_wandb", [False, True])
def test_run_logger_local_outputs(tmp_path, monkeypatch, use_wandb):
    """Ensure local mirrors are identical regardless of the W&B toggle."""

    def _stub_wandb():
        class _DummyRun:
            def log(self, *_args, **_kwargs):
                pass

            def log_artifact(self, _artifact):
                pass

            def finish(self):
                pass

        class _DummyArtifact:
            def __init__(self, name: str, type: str = "file") -> None:  # noqa: A003 - match wandb signature
                self.name = name
                self.type = type

            def add_file(self, _path: str) -> None:
                pass

            def add_dir(self, _path: str) -> None:
                pass

        return SimpleNamespace(init=lambda **_kwargs: _DummyRun(), Artifact=_DummyArtifact)

    config = {
        "local_mirror": {
            "base_dir": str(tmp_path / "runs"),
            "metrics_file": "metrics.jsonl",
            "final_metrics_file": "final_metrics.json",
            "config_file": "config.yaml",
            "artifacts_dir": "artifacts",
        },
        "wandb": {
            "enabled": use_wandb,
            "project": "test-project",
            "entity": None,
            "offline_ok": True,
            "dir": str(tmp_path / "wandb"),
            "group": "pytest",
            "tags": ["unit"],
        },
    }
    resolved_config = {"foo": "bar"}

    if use_wandb:
        monkeypatch.setattr(logging_utils, "wandb", _stub_wandb())
    else:
        monkeypatch.setattr(logging_utils, "wandb", None)

    run_logger = logging_utils.RunLogger(config, resolved_config)
    run_logger.log_metrics({"metric": 1.0}, step=1)
    run_logger.log_final({"metric": 2.0})

    artifact_file = tmp_path / "artifact.txt"
    artifact_file.write_text("payload", encoding="utf-8")
    run_logger.save_artifact(artifact_file)

    artifact_dir = tmp_path / "artifact_dir"
    artifact_dir.mkdir()
    (artifact_dir / "nested.json").write_text("{}", encoding="utf-8")
    run_logger.save_artifact(artifact_dir)

    run_logger.close()

    run_dir = Path(run_logger.base_dir)
    final_metrics_path = run_dir / "final_metrics.json"
    metadata_path = run_dir / "metadata.json"
    mirrored_file = run_dir / "artifacts" / artifact_file.name
    mirrored_dir_file = run_dir / "artifacts" / artifact_dir.name / "nested.json"

    assert json.loads(final_metrics_path.read_text(encoding="utf-8")) == {"metric": 2.0}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["git_commit"]
    assert mirrored_file.read_text(encoding="utf-8") == "payload"
    assert mirrored_dir_file.read_text(encoding="utf-8") == "{}"


def test_run_logger_warns_when_wandb_unavailable(tmp_path, monkeypatch):
    """A clear warning is emitted if W&B is requested but not installed."""

    monkeypatch.setattr(logging_utils, "wandb", None)

    config = {
        "local_mirror": {"base_dir": str(tmp_path / "runs")},
        "wandb": {"enabled": True},
    }

    with pytest.warns(RuntimeWarning):
        run_logger = logging_utils.RunLogger(config, {"foo": "bar"})
        run_logger.close()


def test_run_logger_auto_offline_without_api_key(tmp_path, monkeypatch):
    """If no W&B credentials are configured we automatically fall back to offline mode."""

    class _Recorder:
        def __init__(self) -> None:
            self.init_calls = []

        def init(self, **kwargs):  # type: ignore[no-untyped-def]
            self.init_calls.append(kwargs)
            return SimpleNamespace(log=lambda *_a, **_k: None, finish=lambda: None, log_artifact=lambda *_a, **_k: None)

        def Artifact(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(add_file=lambda *_a, **_k: None, add_dir=lambda *_a, **_k: None)

    recorder = _Recorder()
    monkeypatch.setattr(logging_utils, "wandb", recorder)
    monkeypatch.setattr(logging_utils, "_has_wandb_credentials", lambda: False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    config = {
        "local_mirror": {"base_dir": str(tmp_path / "runs")},
        "wandb": {"enabled": True, "offline_ok": True},
    }

    with pytest.warns(RuntimeWarning):
        logger = logging_utils.RunLogger(config, {"foo": "bar"})
        logger.close()

    assert recorder.init_calls, "wandb.init should have been invoked"
    assert recorder.init_calls[0].get("mode") == "offline"
