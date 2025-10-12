"""IO helpers for writing alignment diagnostics."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def write_alignment_csv(path: Path | str, rows: Sequence[Mapping[str, object]]) -> None:
    """Append alignment diagnostics to a CSV file.

    Parameters
    ----------
    path:
        Target CSV path. Parent directories are created automatically.
    rows:
        Iterable of dictionaries representing rows. Keys determine the
        header order; this function preserves the insertion order of the
        first row.
    """

    if not rows:
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: Iterable[str] | None = None
    # Preserve header order using the first row's keys.
    first_row = rows[0]
    fieldnames = list(first_row.keys())
    needs_header = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
