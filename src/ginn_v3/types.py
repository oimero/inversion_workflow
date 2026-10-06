"""Typed trace, observation, and prediction values for independent PIAI inversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from cup.seismic.geometry import SampleAxis


@dataclass(frozen=True, order=True)
class TraceKey:
    inline_index: int
    xline_index: int

    def __post_init__(self) -> None:
        for value in (self.inline_index, self.xline_index):
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError("Trace keys require nonnegative integer array indices.")


@dataclass(frozen=True)
class Normalization:
    seismic_mean: float
    seismic_std: float
    lfm_mean: float
    lfm_std: float
    impedance_mean: float
    impedance_std: float

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("Normalization values must be finite.")
        if min(self.seismic_std, self.lfm_std, self.impedance_std) <= 0.0:
            raise ValueError("Normalization standard deviations must be positive.")


@dataclass(frozen=True)
class ObservationBatch:
    keys: tuple[TraceKey, ...]
    sample_axis: SampleAxis
    features: Tensor
    initial_log_ai: Tensor
    lfm_mask: Tensor
    observed_seismic: Tensor
    observed_mask: Tensor
    velocity_mps: Tensor | None = None

    def __post_init__(self) -> None:
        expected = (len(self.keys), self.sample_axis.values.size)
        if self.features.shape != (expected[0], 2, expected[1]):
            raise ValueError("PIAI features must have shape (batch, 2, samples).")
        for name in ("initial_log_ai", "observed_seismic", "lfm_mask", "observed_mask"):
            if getattr(self, name).shape != expected:
                raise ValueError(f"ObservationBatch.{name} differs from the common trace shape.")
        if self.lfm_mask.dtype != torch.bool or self.observed_mask.dtype != torch.bool:
            raise TypeError("Observation masks must be boolean.")
        if self.velocity_mps is not None and self.velocity_mps.shape != expected:
            raise ValueError("Fixed velocity must match the common trace shape.")


@dataclass(frozen=True)
class WellTarget:
    well_name: str
    key: TraceKey
    log_ai: np.ndarray
    valid_mask: np.ndarray
    evaluation_mask: np.ndarray
    trace_weights: tuple[tuple[TraceKey, float], ...] = ()


@dataclass(frozen=True)
class WellBatch:
    observations: ObservationBatch
    target_log_ai: Tensor
    target_mask: Tensor
    well_names: tuple[str, ...]


@dataclass(frozen=True)
class Prediction:
    log_ai: Tensor
    raw_correction: Tensor
    wavelets: Tensor


@dataclass(frozen=True)
class ForwardResult:
    seismic: Tensor
    valid_mask: Tensor


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    selected_checkpoint: Path
    last_checkpoint: Path
    updates_completed: int


@dataclass(frozen=True)
class TracePrediction:
    keys: tuple[TraceKey, ...]
    log_ai: np.ndarray
    valid_mask: np.ndarray
    wavelet_mean_normalized: np.ndarray

