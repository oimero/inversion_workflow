"""Configuration of simultaneous PIAI training and unfiltered residual output."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping
import math


def _construct(cls, payload: Mapping[str, Any] | None):
    raw = dict(payload or {})
    unknown = set(raw) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**raw)


@dataclass(frozen=True)
class NetworkConfig:
    sample_count: int
    wavelet_samples: int
    tcn_channels: tuple[int, ...] = (16, 16, 16)
    hidden_channels: int = 32
    kernel_size: int = 3
    dilation: int = 2
    gru_layers: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tcn_channels", tuple(self.tcn_channels))
        integers = (self.sample_count, self.wavelet_samples, self.hidden_channels,
                    self.kernel_size, self.dilation, self.gru_layers, *self.tcn_channels)
        if any(isinstance(v, bool) or int(v) != v or v <= 0 for v in integers):
            raise ValueError("Network dimensions must be positive integers.")
        if self.sample_count < 2 or self.wavelet_samples < 3 or self.wavelet_samples % 2 != 1:
            raise ValueError("Use at least two trace samples and an odd wavelet length >= 3.")
        if not self.tcn_channels or self.hidden_channels % 2 or self.kernel_size % 2 != 1:
            raise ValueError("PIAI requires TCN channels, even hidden width, and an odd kernel.")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]):
        return _construct(cls, payload)


@dataclass(frozen=True)
class LossWeights:
    independent: float = 1.0
    physics: float = 1.0
    cross: float = 1.0

    def __post_init__(self) -> None:
        values = (self.independent, self.physics, self.cross)
        if any(not math.isfinite(v) or v < 0.0 for v in values) or not any(values):
            raise ValueError("I/P/C weights must be nonnegative, finite, and not all zero.")


@dataclass(frozen=True)
class TrainingConfig:
    updates: int = 1000
    labeled_batch_size: int = 6
    unlabeled_batch_size: int = 32
    learning_rate: float = 0.004
    weight_decay: float = 0.01
    seed: int = 20261004
    validate_every: int = 100
    log_every: int = 20
    max_train_traces: int = 4096
    validation_traces: int = 128
    validation_gap_m: float = 300.0
    min_support_samples: int = 8
    device: str = "cuda"
    loss_weights: LossWeights = field(default_factory=LossWeights)

    def __post_init__(self) -> None:
        names = ("updates", "labeled_batch_size", "unlabeled_batch_size", "validate_every",
                 "log_every", "max_train_traces", "validation_traces", "min_support_samples")
        if any(isinstance(getattr(self, name), bool) or int(getattr(self, name)) != getattr(self, name)
               or getattr(self, name) <= 0 for name in names):
            raise ValueError("Training counts must be positive integers.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be nonnegative and finite.")
        if not math.isfinite(self.validation_gap_m) or self.validation_gap_m < 0.0:
            raise ValueError("validation_gap_m must be nonnegative and finite.")
        if not isinstance(self.loss_weights, LossWeights):
            raise TypeError("loss_weights must be LossWeights.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None):
        raw = dict(payload or {})
        raw["loss_weights"] = _construct(LossWeights, raw.get("loss_weights"))
        return _construct(cls, raw)


@dataclass(frozen=True)
class InferenceConfig:
    batch_size: int = 128
    min_support_samples: int = 8

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.min_support_samples < 2:
            raise ValueError("Inference batch and finite-support lengths must be positive.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None):
        return _construct(cls, payload)

