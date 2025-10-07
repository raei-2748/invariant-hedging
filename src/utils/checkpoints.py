"""Checkpoint management utilities."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(order=True)
class CheckpointEntry:
    score: float
    path: Path = field(compare=False)


class CheckpointManager:
    def __init__(self, directory: Path, top_k: int):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.heap: list[CheckpointEntry] = []
        self._load_manifest()

    def save(self, step: int, score: float, state: dict) -> Path:
        path = self.directory / f"checkpoint_{step}.pt"
        torch.save(state, path)
        entry = CheckpointEntry(score=score, path=path)
        heapq.heappush(self.heap, entry)
        if len(self.heap) > self.top_k:
            removed = heapq.heappop(self.heap)
            if removed.path.exists():
                removed.path.unlink()
        self._write_manifest()
        return path

    def _write_manifest(self) -> None:
        manifest = [
            {"score": entry.score, "path": entry.path.name}
            for entry in sorted(self.heap, reverse=True)
        ]
        with open(self.directory / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _load_manifest(self) -> None:
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.exists():
            return

        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                entries = json.load(handle)
        except (json.JSONDecodeError, OSError):
            # Corrupted manifest – start from a clean slate.
            self.heap.clear()
            return

        changed = False
        for record in entries:
            path_name = record.get("path")
            score = record.get("score")
            if path_name is None or score is None:
                changed = True
                continue
            path = self.directory / path_name
            if not path.exists():
                changed = True
                continue
            heapq.heappush(self.heap, CheckpointEntry(score=float(score), path=path))

        while len(self.heap) > self.top_k:
            removed = heapq.heappop(self.heap)
            if removed.path.exists():
                removed.path.unlink()
            changed = True

        if changed:
            self._write_manifest()


def load_checkpoint(path: Path, map_location: str | torch.device | None = None):
    return torch.load(path, map_location=map_location)
