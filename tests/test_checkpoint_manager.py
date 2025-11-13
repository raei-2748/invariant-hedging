from __future__ import annotations

import json
from pathlib import Path

from invariant_hedging.core.utils import checkpoints


def _collect_manifest(path: Path) -> list[dict]:
    with (path / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_checkpoint_manager_restores_existing_entries(tmp_path):
    manager = checkpoints.CheckpointManager(tmp_path, top_k=2)
    manager.save(1, score=0.5, state={"step": 1})
    manager.save(2, score=1.0, state={"step": 2})

    restored = checkpoints.CheckpointManager(tmp_path, top_k=2)
    scores = sorted(entry.score for entry in restored.heap)
    assert scores == [0.5, 1.0]


def test_checkpoint_manager_prunes_missing_and_excess_checkpoints(tmp_path):
    manager = checkpoints.CheckpointManager(tmp_path, top_k=3)
    for idx, score in enumerate([0.2, 0.4, 0.6], start=1):
        manager.save(idx, score=score, state={"step": idx})

    # Remove one checkpoint file to simulate manual deletion
    missing_path = tmp_path / "checkpoint_1.pt"
    missing_path.unlink()

    restored = checkpoints.CheckpointManager(tmp_path, top_k=2)
    scores = sorted(entry.score for entry in restored.heap)
    assert scores == [0.4, 0.6]

    manifest = _collect_manifest(tmp_path)
    assert len(manifest) == 2
