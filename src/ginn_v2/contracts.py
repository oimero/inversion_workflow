"""Small shared interfaces used by both domain adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from cup.lfm.artifacts import LfmInput
from cup.seismic.geometry import SampleAxis


def _trace_batch(value: Tensor, *, name: str, batch: int | None = None, samples: int | None = None) -> Tensor:
    if not isinstance(value, Tensor) or not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating torch.Tensor.")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (batch, samples).")
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f"{name} batch dimension differs from observed_seismic.")
    if samples is not None and value.shape[1] != samples:
        raise ValueError(f"{name} sample dimension differs from SampleAxis.")
    if not bool(torch.all(torch.isfinite(value)).item()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


@dataclass(frozen=True)
class CommonObservationBatch:
    """Domain-neutral center-trace batch passed through the shared workflow.

    Domain-specific arrays such as ``velocity_mps`` live in ``domain_extras``.
    The trace arrays all use ``(batch, samples)`` so training code has no domain
    branches and no implicit channel convention.
    """

    sample_axis: SampleAxis
    observed_seismic: Tensor
    observed_valid_mask: Tensor
    lfm_log_ai: Tensor
    lfm_valid_mask: Tensor
    xy_m: Tensor
    domain_extras: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_axis, SampleAxis):
            raise TypeError("sample_axis must be cup.seismic.geometry.SampleAxis.")
        observed = _trace_batch(
            self.observed_seismic,
            name="observed_seismic",
            samples=self.sample_axis.values.size,
        )
        batch, samples = observed.shape
        _trace_batch(self.lfm_log_ai, name="lfm_log_ai", batch=batch, samples=samples)
        for name, value in (
            ("observed_valid_mask", self.observed_valid_mask),
            ("lfm_valid_mask", self.lfm_valid_mask),
        ):
            if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != observed.shape:
                raise ValueError(f"{name} must be a bool tensor matching observed_seismic.")
        if not isinstance(self.xy_m, Tensor) or not torch.is_floating_point(self.xy_m):
            raise TypeError("xy_m must be a floating torch.Tensor.")
        if self.xy_m.shape != (batch, 2) or not bool(torch.all(torch.isfinite(self.xy_m)).item()):
            raise ValueError("xy_m must contain finite actual metre coordinates with shape (batch, 2).")
        if not isinstance(self.domain_extras, Mapping):
            raise TypeError("domain_extras must be a mapping.")


@dataclass(frozen=True)
class ForwardClosureResult:
    """Body-scale prediction and its frozen-forward reconstruction."""

    body_log_ai: Tensor
    synthetic_seismic: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        if self.body_log_ai.shape != self.synthetic_seismic.shape:
            raise ValueError("body_log_ai and synthetic_seismic must have matching shapes.")
        if self.valid_mask.dtype != torch.bool or self.valid_mask.shape != self.body_log_ai.shape:
            raise ValueError("valid_mask must be boolean and match the closure outputs.")


@dataclass(frozen=True)
class GinnLfmInputs:
    """Primary and sensitivity LFM roles consumed by body inversion."""

    primary: LfmInput
    sensitivity: LfmInput

    def __post_init__(self) -> None:
        if self.primary.baseline_method != "proportional_kriging":
            raise ValueError("GINN V2 primary LFM must use proportional_kriging.")
        if self.sensitivity.baseline_method != "trend":
            raise ValueError("GINN V2 sensitivity LFM must use trend.")
        if self.primary.variant.run_dir.resolve() != self.sensitivity.variant.run_dir.resolve():
            raise ValueError("Primary and sensitivity LFM must come from the same Step-7 run.")
        for name in ("sample_axis", "ilines", "xlines"):
            left = getattr(self.primary, name)
            right = getattr(self.sensitivity, name)
            if name == "sample_axis":
                matches = (
                    left.domain == right.domain
                    and left.unit == right.unit
                    and left.depth_basis == right.depth_basis
                    and np.array_equal(left.values, right.values)
                )
            else:
                matches = np.array_equal(left, right)
            if not matches:
                raise ValueError(f"Primary and sensitivity LFM {name} differ.")
        if self.primary.lowpass_unit != self.sensitivity.lowpass_unit or not np.isclose(
            self.primary.lowpass_value,
            self.sensitivity.lowpass_value,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("Primary and sensitivity LFM low-pass scales differ.")


__all__ = ["CommonObservationBatch", "ForwardClosureResult", "GinnLfmInputs"]
