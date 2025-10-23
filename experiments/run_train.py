"""CLI wrapper for launching the invariant hedging training engine."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.engine import main as train_main


if __name__ == "__main__":
    train_main()
