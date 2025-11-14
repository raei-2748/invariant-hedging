"""Device resolution helpers with MPS/CUDA fallbacks."""
from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import torch
from omegaconf import DictConfig

RuntimeCfg = Optional[Union[DictConfig, Mapping[str, object]]]


def _cfg_get(cfg: RuntimeCfg, key: str, default: object) -> object:
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return default


def _mps_status() -> Tuple[bool, bool]:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False, False
    try:
        built = bool(getattr(backend, "is_built", lambda: True)())
    except Exception:  # pragma: no cover - defensive
        built = False
    try:
        available = bool(backend.is_available())
    except Exception:  # pragma: no cover - defensive
        available = False
    return built, available


@dataclass(frozen=True)
class DeviceSetup:
    device: torch.device
    use_mixed_precision: bool


def resolve_device(runtime_cfg: RuntimeCfg, *, context: str = "runtime") -> DeviceSetup:
    """Resolve the torch device with preference order MPS -> CUDA -> CPU."""

    requested = str(_cfg_get(runtime_cfg, "device", "auto")).strip().lower()
    allow_amp = bool(_cfg_get(runtime_cfg, "mixed_precision", True))

    mps_built, mps_available = _mps_status()
    cuda_available = torch.cuda.is_available()

    if requested == "auto":
        if mps_available:
            device = torch.device("mps")
        elif cuda_available:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)
        if device.type == "mps" and not mps_available:
            raise RuntimeError("runtime.device=mps requested but MPS backend is unavailable.")
        if device.type == "cuda" and not cuda_available:
            raise RuntimeError("runtime.device=cuda requested but CUDA is unavailable.")

    if device.type == "mps" and allow_amp:
        print(f"[device] ({context}) Disabling mixed precision for MPS backend.")
        allow_amp = False

    print(f"[device] ({context}) Device: {device}")
    print(f"[device] ({context}) MPS built: {mps_built}, available: {mps_available}")
    print(f"[device] ({context}) CUDA available: {cuda_available}")

    if device.type == "cpu" and platform.system() == "Darwin":
        print(
            f"[device] ({context}) WARNING: running on CPU on macOS — install the MPS-enabled PyTorch wheel to "
            "use the Apple GPU."
        )

    return DeviceSetup(device=device, use_mixed_precision=allow_amp)
