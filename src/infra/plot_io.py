"""Utilities for figure IO and Matplotlib configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl

LOGGER = logging.getLogger(__name__)


def ensure_out_dir(run_dir: Path | str, out_dir: Path | str | None) -> Path:
    """Return the directory where figures should be stored.

    Parameters
    ----------
    run_dir:
        Root directory for the run (``runs/<timestamp>_<expname>``).
    out_dir:
        Optional explicit output directory.  When ``None`` the default is
        ``<run_dir>/figures``.
    """

    run_dir = Path(run_dir)
    resolved = Path(out_dir) if out_dir is not None else run_dir / "figures"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_formats(format_arg: str | Sequence[str]) -> tuple[str, ...]:
    """Normalise a format specification into a tuple of extensions."""

    if isinstance(format_arg, str):
        tokens = [token.strip() for token in format_arg.split(",") if token.strip()]
    else:
        tokens = [str(token).strip() for token in format_arg if str(token).strip()]
    if not tokens:
        raise ValueError("At least one image format must be provided")
    normalised: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower not in {"png", "pdf"}:
            raise ValueError(f"Unsupported image format: {token}")
        if lower not in normalised:
            normalised.append(lower)
    return tuple(normalised)


def apply_style(style: str) -> None:
    """Apply a repo-standard Matplotlib style profile."""

    style = style.lower()
    if style not in {"journal", "poster"}:
        raise ValueError(f"Unknown style '{style}' (expected 'journal' or 'poster')")

    base_font = 11 if style == "journal" else 13
    legend_font = 12 if style == "journal" else 14
    title_font = base_font + 1

    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": base_font,
            "axes.titlesize": title_font,
            "axes.labelsize": base_font,
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#d0d0d0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.fontsize": legend_font,
            "legend.frameon": False,
            "xtick.labelsize": base_font - 1,
            "ytick.labelsize": base_font - 1,
        }
    )


def save_figure(
    figure: mpl.figure.Figure,
    out_dir: Path | str,
    base_name: str,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Persist a Matplotlib figure in the requested formats.

    Returns the list of written paths.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fmt in formats:
        suffix = fmt.lower()
        target = out_path / f"{base_name}.{suffix}"
        figure.savefig(target, dpi=dpi, bbox_inches="tight")
        written.append(target)
        LOGGER.info("Saved figure %s", target)
    return written


def append_manifest(out_dir: Path | str, entry: dict) -> Path:
    """Append or update a manifest entry for the generated figure."""

    if "name" not in entry:
        raise ValueError("Manifest entry must include a 'name' field")

    manifest_path = Path(out_dir) / "manifest.json"
    data: list[dict]
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text())
            if not isinstance(data, list):
                LOGGER.warning("Manifest at %s is not a list; reinitialising", manifest_path)
                data = []
        except json.JSONDecodeError:
            LOGGER.warning("Manifest at %s is invalid JSON; reinitialising", manifest_path)
            data = []
    else:
        data = []

    replaced = False
    for idx, existing in enumerate(data):
        if existing.get("name") == entry["name"]:
            data[idx] = entry
            replaced = True
            break
    if not replaced:
        data.append(entry)

    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    LOGGER.info("Updated manifest entry '%s'", entry["name"])
    return manifest_path

