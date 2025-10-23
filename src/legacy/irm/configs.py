"""Configuration helpers for IRM penalty settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class IRMLoggingConfig:
    log_irm_grads: bool = False


@dataclass(frozen=True)
class IRMConfig:
    enabled: bool = False
    penalty_type: str = "cosine"
    lambda_weight: float = 0.0
    normalize: str = "l2"
    eps: float = 1.0e-12
    env_min: int = 2
    freeze_backbone: bool = True
    logging: IRMLoggingConfig = field(default_factory=IRMLoggingConfig)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        penalty = self.penalty_type.lower()
        if penalty not in {"cosine", "varnorm"}:
            raise ValueError(f"Unsupported IRM penalty type '{self.penalty_type}'.")
        if self.normalize.lower() != "l2":
            raise ValueError("Only L2 normalisation is supported for IRM penalties.")
        if self.lambda_weight < 0:
            raise ValueError("IRM lambda must be non-negative.")
        if self.env_min < 2:
            raise ValueError("IRM penalties require at least two environments (env_min >= 2).")

    @property
    def type(self) -> str:
        return self.penalty_type

    @classmethod
    def from_config(cls, cfg: Optional[Mapping[str, Any] | DictConfig]) -> "IRMConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
        mapping: MutableMapping[str, Any] = dict(cfg)  # type: ignore[arg-type]
        if "mode" in mapping:
            mode = str(mapping["mode"]).lower()
            if mode == "hybrid":
                raise ValueError("The hybrid HIRM mode has been removed; choose 'cosine' or 'varnorm'.")
            raise ValueError("'irm.mode' is no longer supported; use 'irm.type'.")
        penalty_type = str(mapping.get("type", "cosine")).lower()
        if penalty_type == "hybrid":
            raise ValueError("Hybrid penalties are deprecated; valid options are 'cosine' and 'varnorm'.")
        lambda_value = mapping.get("lambda", mapping.get("lambda_weight", 0.0))
        logging_cfg = mapping.get("logging", {})
        if not isinstance(logging_cfg, Mapping):
            raise TypeError("irm.logging must be a mapping if provided.")
        logging = IRMLoggingConfig(log_irm_grads=bool(logging_cfg.get("log_irm_grads", False)))
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            penalty_type=penalty_type,
            lambda_weight=float(lambda_value),
            normalize=str(mapping.get("normalize", "l2")),
            eps=float(mapping.get("eps", 1.0e-12)),
            env_min=int(mapping.get("env_min", 2)),
            freeze_backbone=bool(mapping.get("freeze_backbone", True)),
            logging=logging,
        )


__all__ = ["IRMConfig", "IRMLoggingConfig"]

