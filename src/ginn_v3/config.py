"""Configuration of simultaneous PIAI training and unfiltered residual output."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping
import math

import numpy as np


def _construct(cls, payload: Mapping[str, Any] | None):
    raw = dict(payload or {})
    unknown = set(raw) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**raw)


@dataclass(frozen=True)
class KernelConfig:
    """An independently configured, zero-centered temporal kernel grid."""

    dt_s: float
    duration_s: float | None = None
    samples: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.dt_s, bool) or not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("kernel.dt_s must be finite and positive seconds.")
        if (self.duration_s is None) == (self.samples is None):
            raise ValueError("Specify exactly one of duration_s and samples in kernel.")
        if self.samples is not None:
            if (isinstance(self.samples, bool) or int(self.samples) != self.samples
                    or self.samples < 3 or self.samples % 2 == 0):
                raise ValueError("kernel.samples must be an odd integer of at least three samples.")
        elif (isinstance(self.duration_s, bool) or not math.isfinite(self.duration_s)
              or self.duration_s < 2.0 * self.dt_s):
            raise ValueError("kernel.duration_s must cover at least two kernel sample intervals.")

    @property
    def sample_count(self) -> int:
        if self.samples is not None:
            return int(self.samples)
        half = int(math.floor(self.duration_s / (2.0 * self.dt_s) + 1e-9))
        return 2 * half + 1

    def time_axis_s(self) -> np.ndarray:
        center = self.sample_count // 2
        return (np.arange(self.sample_count, dtype=np.float64) - center) * self.dt_s

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "KernelConfig":
        if not isinstance(payload, Mapping) or "dt_s" not in payload:
            raise ValueError("kernel requires an explicit dt_s and either duration_s or samples.")
        return _construct(cls, payload)


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
    device: str = "cpu"

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or int(value) != value for value in
               (self.batch_size, self.min_support_samples)):
            raise ValueError("Inference batch and support lengths must be integers.")
        if self.batch_size <= 0 or self.min_support_samples < 2:
            raise ValueError("Inference batch and finite-support lengths must be positive.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Inference device must be explicit and non-empty.")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None):
        return _construct(cls, payload)

