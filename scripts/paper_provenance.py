#!/usr/bin/env python3
"""CLI entry-point to capture paper reproducibility metadata."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_provenance_functions() -> Tuple[Callable[..., object], Callable[[Path, object], None]]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.report.paper_provenance import collect_provenance, write_provenance

    return collect_provenance, write_provenance


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run directory whose artifacts should be fingerprinted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to write the provenance manifest to.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output (applies to stdout and --output).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    collect_provenance, write_provenance = _resolve_provenance_functions()
    data = collect_provenance(args.run_dir)
    indent = 2 if args.pretty or args.output else None
    text = json.dumps(data, indent=indent, sort_keys=True)
    if args.output is not None:
        write_provenance(args.output, data)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
