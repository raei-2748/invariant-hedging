"""Runtime device detection utilities with Apple MPS support."""
from __future__ import annotations

import platform
from typing import Optional

import torch


def _mps_backend_available() -> bool:
    """Return ``True`` when the PyTorch MPS backend is built and ready."""

    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_available())
    except Exception:  # pragma: no cover - defensive for older torch builds
        return False


def _mps_backend_built() -> bool:
    """Return ``True`` if the PyTorch binary was compiled with MPS support."""

    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_built())
    except AttributeError:  # pragma: no cover - method missing on older builds
        # Older releases expose ``is_available`` but not ``is_built``; treat as built
        return _mps_backend_available()


def resolve_device(device_str: Optional[str] = None) -> torch.device:
    """Resolve the execution device preferring MPS over CUDA when available."""

    selection = (device_str or "auto").strip().lower()
    if selection == "auto":
        if _mps_backend_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(selection)


def log_device_diagnostics(device: torch.device) -> None:
    """Print startup diagnostics for the resolved device."""

    mps_available = _mps_backend_available()
    cuda_available = torch.cuda.is_available()
    print(f"Device: {device}")
    print(f"MPS available: {mps_available}")
    print(f"CUDA available: {cuda_available}")
    if device.type == "cpu" and platform.system() == "Darwin":
        if not _mps_backend_built():
            print(
                "[warn] Running on CPU because the current PyTorch build lacks MPS support."
            )
        elif not mps_available:
            print(
                "[warn] Running on CPU because the MPS backend is unavailable. "
                "Install an MPS-enabled PyTorch wheel to enable Apple GPU acceleration."
            )


def should_enable_mixed_precision(device: torch.device, requested: bool) -> bool:
    """Return whether AMP should be enabled for the resolved device."""

    if device.type != "cuda":
        return False
    return bool(requested)


__all__ = [
    "log_device_diagnostics",
    "resolve_device",
    "should_enable_mixed_precision",
]
