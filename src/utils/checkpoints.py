"""Checkpoint management utilities."""
from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
        self.heap: List[CheckpointEntry] = []

    def save(self, step: int, score: float, state: Dict) -> Path:
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


def load_checkpoint(path: Path, map_location: str | torch.device | None = None):
    return torch.load(path, map_location=map_location)
