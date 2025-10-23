"""CLI entrypoint for running crisis diagnostics and evaluations."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate_crisis import main as evaluate_main


if __name__ == "__main__":
    evaluate_main()
