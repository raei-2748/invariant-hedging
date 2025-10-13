"""Policy utilities separating backbone (phi) and head (psi) parameters."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional

from torch import nn


class Policy(nn.Module):
    """Base policy providing phi/psi parameter views and alignment context."""

    def __init__(self, *, head_name: Optional[str] = None) -> None:
        super().__init__()
        self._psi_module_name = head_name or "decision_head"
        self._alignment_detach = False

    @property
    def head_name(self) -> str:
        return self._psi_module_name

    def set_head_name(self, name: str) -> None:
        self._psi_module_name = name

    # ------------------------------------------------------------------
    # Parameter helpers
    def _psi_module(self) -> nn.Module:
        module = getattr(self, self._psi_module_name, None)
        if module is None:
            raise AttributeError(
                f"Policy is missing head module '{self._psi_module_name}'."
            )
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Attribute '{self._psi_module_name}' is not an nn.Module instance."
            )
        return module

    def psi_params(self) -> List[nn.Parameter]:
        head = self._psi_module()
        return [param for param in head.parameters() if param.requires_grad]

    def phi_params(self) -> List[nn.Parameter]:
        head_ids = {id(param) for param in self.psi_params()}
        backbone: List[nn.Parameter] = []
        for param in self.parameters():
            if id(param) in head_ids:
                continue
            if param.requires_grad:
                backbone.append(param)
        return backbone

    def freeze_phi(self) -> None:
        for param in self.phi_params():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Alignment helpers
    def should_detach_features(self) -> bool:
        return self._alignment_detach

    @contextmanager
    def alignment_context(self, *, detach_features: bool) -> Iterator[None]:
        prev = self._alignment_detach
        self._alignment_detach = detach_features
        try:
            yield
        finally:
            self._alignment_detach = prev


__all__ = ["Policy"]
