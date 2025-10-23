"""CLI entrypoint for running crisis diagnostics and evaluations."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from src.evaluation.evaluate_crisis import main as evaluate_main


if __name__ == "__main__":
    evaluate_main()
