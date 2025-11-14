"""Entry point for generating diagnostics reports via the unified CLI."""
from __future__ import annotations

from typing import Iterable

from invariant_hedging.reporting import cli as reporting_cli


def main(argv: Iterable[str] | None = None) -> None:
    """Delegate to the reporting CLI module."""

    raise SystemExit(reporting_cli.main(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
