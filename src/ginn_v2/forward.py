"""The Structured GINN V2 NumPy/Torch physics forward seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cup.physics.calibration import AIVelocityRelation
from cup.physics.numpy_backend import (
    forward_depth as _numpy_forward_depth,
    forward_time as _numpy_forward_time,
    velocity_from_ai as _numpy_velocity_from_ai,
)
from cup.synthetic.core.records import SampleAxis


def _wavelet_arrays(
    wavelet_time_s: Any,
    wavelet_amplitude: Any,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(wavelet_time_s, dtype=np.float64).reshape(-1)
    amplitude = np.asarray(wavelet_amplitude, dtype=np.float64).reshape(-1)
    if time.size != amplitude.size or time.size < 3 or time.size % 2 == 0:
        raise ValueError("wavelet_time_s and wavelet_amplitude must share an odd length >= 3.")
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(amplitude)):
        raise ValueError("wavelet_time_s and wavelet_amplitude must be finite.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("wavelet_time_s must be strictly increasing.")
    if not np.allclose(np.diff(time), time[1] - time[0], rtol=1e-6, atol=1e-12):
        raise ValueError("wavelet_time_s must be regularly sampled.")
    center = time.size // 2
    if not np.isclose(
        time[center],
        0.0,
        rtol=0.0,
        atol=max(1e-12, abs(time[1] - time[0]) * 1e-6),
    ):
        raise ValueError("wavelet_time_s center sample must be zero seconds.")
    return time, amplitude


def _relation(
    value: AIVelocityRelation | Mapping[str, Any] | None,
) -> AIVelocityRelation | None:
    if value is None or isinstance(value, AIVelocityRelation):
        return value
    if isinstance(value, Mapping):
        return AIVelocityRelation.from_mapping(value)
    raise TypeError("ai_velocity_relation must be AIVelocityRelation, mapping, or None.")


@dataclass(frozen=True)
class ForwardContext:
    """All inputs required by both forward implementations.

    The interface owns domain, units, axis identity, wavelet identity, and the
    frozen depth AI--Vp relation.  Inline/xline values are geometry metadata;
    no forward calculation uses them as physical distances.
    """

    sample_axis: SampleAxis
    wavelet_time_s: np.ndarray
    wavelet_amplitude: np.ndarray
    ai_velocity_relation: AIVelocityRelation | Mapping[str, Any] | None = None
    output_chunk_size: int = 64
    lateral_m: float | None = None
    inline: float | None = None
    xline: float | None = None
    xline_step: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_axis, SampleAxis):
            raise TypeError(
                "ForwardContext.sample_axis must be cup.synthetic.core.records.SampleAxis."
            )
        time, amplitude = _wavelet_arrays(self.wavelet_time_s, self.wavelet_amplitude)
        if isinstance(self.output_chunk_size, bool) or int(self.output_chunk_size) <= 0:
            raise ValueError("ForwardContext.output_chunk_size must be positive.")
        relation = _relation(self.ai_velocity_relation)
        if self.sample_axis.sample_domain == "depth" and relation is None:
            raise ValueError("Depth ForwardContext requires ai_velocity_relation.")
        if self.sample_axis.sample_domain == "time" and relation is not None:
            raise ValueError("Time ForwardContext must not declare ai_velocity_relation.")
        for name in ("lateral_m", "inline", "xline", "xline_step"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"ForwardContext.{name} must be finite when supplied.")
        if self.xline_step is not None and float(self.xline_step) == 0.0:
            raise ValueError("ForwardContext.xline_step must be non-zero when supplied.")
        object.__setattr__(self, "wavelet_time_s", time)
        object.__setattr__(self, "wavelet_amplitude", amplitude)
        object.__setattr__(self, "ai_velocity_relation", relation)

    @property
    def sample_domain(self) -> str:
        return self.sample_axis.sample_domain

    @property
    def sample_unit(self) -> str:
        return self.sample_axis.unit

    @property
    def depth_basis(self) -> str | None:
        return self.sample_axis.depth_basis

    @property
    def manifest_fields(self) -> dict[str, Any]:
        relation = (
            None
            if self.ai_velocity_relation is None
            else self.ai_velocity_relation.to_mapping()
        )
        return {
            "sample_domain": self.sample_domain,
            "sample_unit": self.sample_unit,
            "depth_basis": self.depth_basis,
            "sample_interval": float(self.sample_axis.sample_interval),
            "wavelet_num_samples": int(self.wavelet_time_s.size),
            "wavelet_dt_s": float(self.wavelet_time_s[1] - self.wavelet_time_s[0]),
            "ai_velocity_relation": relation,
            "lateral_m": self.lateral_m,
            "inline": self.inline,
            "xline": self.xline,
            "xline_step": self.xline_step,
        }


def _validate_log_ai_numpy(log_ai: Any, *, expected_samples: int) -> np.ndarray:
    values = np.asarray(log_ai)
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("log_ai must have a floating dtype.")
    if values.ndim < 1 or values.shape[-1] != expected_samples:
        raise ValueError(
            f"log_ai final dimension must match sample axis length {expected_samples}, "
            f"got {values.shape}."
        )
    return values


def forward_numpy(context: ForwardContext, log_ai: Any) -> np.ndarray:
    """Run the strict NumPy forward operator described by context."""
    values = _validate_log_ai_numpy(
        log_ai,
        expected_samples=context.sample_axis.coordinates.size,
    )
    if context.sample_domain == "time":
        return _numpy_forward_time(
            values,
            context.wavelet_time_s,
            context.wavelet_amplitude,
        )
    assert context.ai_velocity_relation is not None
    velocity = _numpy_velocity_from_ai(
        np.exp(values),
        a=context.ai_velocity_relation.a,
        b=context.ai_velocity_relation.b,
    )
    return _numpy_forward_depth(
        values,
        velocity,
        context.sample_axis.coordinates,
        context.wavelet_time_s,
        context.wavelet_amplitude,
        output_chunk_size=int(context.output_chunk_size),
    )


def forward_torch(context: ForwardContext, log_ai: Any):
    """Run the differentiable Torch forward operator described by context."""
    import torch

    if not isinstance(log_ai, torch.Tensor):
        raise TypeError("Torch forward log_ai must be a torch.Tensor.")
    if not torch.is_floating_point(log_ai):
        raise TypeError("Torch forward log_ai must have a floating dtype.")
    if log_ai.ndim < 1 or log_ai.shape[-1] != context.sample_axis.coordinates.size:
        raise ValueError(
            "Torch forward log_ai final dimension must match the sample axis length."
        )
    wavelet_time = torch.as_tensor(
        context.wavelet_time_s,
        dtype=log_ai.dtype,
        device=log_ai.device,
    )
    wavelet_amplitude = torch.as_tensor(
        context.wavelet_amplitude,
        dtype=log_ai.dtype,
        device=log_ai.device,
    )
    if context.sample_domain == "time":
        from cup.physics.torch_backend import forward_time as torch_forward_time

        return torch_forward_time(log_ai, wavelet_time, wavelet_amplitude)
    assert context.ai_velocity_relation is not None
    from cup.physics.torch_backend import (
        forward_depth as torch_forward_depth,
        velocity_from_ai as torch_velocity_from_ai,
    )

    depth = torch.as_tensor(
        context.sample_axis.coordinates,
        dtype=log_ai.dtype,
        device=log_ai.device,
    )
    velocity = torch_velocity_from_ai(
        torch.exp(log_ai),
        a=context.ai_velocity_relation.a,
        b=context.ai_velocity_relation.b,
    )
    return torch_forward_depth(
        log_ai,
        velocity,
        depth,
        wavelet_time,
        wavelet_amplitude,
        output_chunk_size=int(context.output_chunk_size),
    )


__all__ = ["ForwardContext", "forward_numpy", "forward_torch"]
