"""Domain-neutral observations and time/depth physics adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping

import torch
from torch import Tensor

from cup.physics.torch_backend import forward_depth, forward_time
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
        if self.xy_m.shape != (batch, 2):
            raise ValueError("xy_m must contain actual metre coordinates with shape (batch, 2).")
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


def depth_coordinates_from_twt(velocity_mps: Tensor, twt_s: Tensor) -> Tensor:
    """Integrate fixed velocity along TWT to physical depth coordinates."""
    if velocity_mps.ndim != 2 or twt_s.ndim != 1 or velocity_mps.shape[-1] != twt_s.numel():
        raise ValueError("velocity_mps/twt_s must have shapes (batch, samples)/(samples,).")
    if not bool(torch.all(torch.isfinite(velocity_mps)).item()) or not bool(
        torch.all(torch.isfinite(twt_s)).item()
    ):
        raise ValueError("velocity and TWT must contain only finite values.")
    if bool(torch.any(velocity_mps <= 0.0).item()) or bool(torch.any(torch.diff(twt_s) <= 0.0).item()):
        raise ValueError("velocity must be positive and TWT must be strictly increasing.")
    dz = 0.25 * (velocity_mps[:, :-1] + velocity_mps[:, 1:]) * torch.diff(twt_s)[None, :]
    return torch.cat((torch.zeros_like(velocity_mps[:, :1]), torch.cumsum(dz, dim=-1)), dim=-1)


class DomainAdapter(ABC):
    """Domain seam used by shared training and inference code."""

    sample_domain: str
    adapter_id: str

    def __init__(self, wavelet_time_s: Tensor, wavelet_amplitude: Tensor) -> None:
        self.wavelet_time_s = wavelet_time_s
        self.wavelet_amplitude = wavelet_amplitude

    def _require_domain(self, axis: SampleAxis) -> None:
        if axis.domain != self.sample_domain:
            raise ValueError(f"{self.adapter_id} requires a {self.sample_domain} SampleAxis.")

    @abstractmethod
    def vertical_coordinates_m(self, batch: CommonObservationBatch) -> Tensor: ...

    @abstractmethod
    def forward(self, body_log_ai: Tensor, batch: CommonObservationBatch) -> Tensor: ...

    def close_body(
        self,
        body_log_ai: Tensor,
        batch: CommonObservationBatch,
    ) -> ForwardClosureResult:
        """Forward an already body-scale log-AI trace through the frozen physics."""
        self._require_domain(batch.sample_axis)
        if not isinstance(body_log_ai, Tensor) or not torch.is_floating_point(body_log_ai):
            raise TypeError("body_log_ai must be a floating torch.Tensor.")
        if body_log_ai.shape != batch.observed_seismic.shape:
            raise ValueError("body_log_ai must match the common batch trace shape.")
        if not bool(torch.all(torch.isfinite(body_log_ai)).item()):
            raise ValueError("body_log_ai must contain only finite values.")
        synthetic = self.forward(body_log_ai, batch)
        return ForwardClosureResult(body_log_ai, synthetic, batch.observed_valid_mask)

    def _forward_supported(
        self,
        body_log_ai: Tensor,
        batch: CommonObservationBatch,
        forward_segment: Callable[[int, int, int], Tensor],
    ) -> Tensor:
        """Forward each finite LFM/domain run without crossing support gaps."""

        support = batch.lfm_valid_mask & torch.isfinite(body_log_ai)
        velocity = batch.domain_extras.get("velocity_mps")
        if velocity is not None:
            if velocity.shape != body_log_ai.shape:
                raise ValueError("velocity_mps must match body_log_ai when supplied.")
            support &= torch.isfinite(velocity)
        output = torch.zeros_like(body_log_ai)
        for row in range(body_log_ai.shape[0]):
            finite = support[row]
            padded = torch.cat(
                (
                    torch.zeros(1, dtype=torch.bool, device=finite.device),
                    finite,
                    torch.zeros(1, dtype=torch.bool, device=finite.device),
                )
            )
            changes = torch.nonzero(padded[1:] != padded[:-1], as_tuple=False).reshape(-1, 2)
            for start, stop in changes.tolist():
                if stop - start < 2:
                    raise ValueError("Forward-model finite support contains a run shorter than two samples.")
                segment = forward_segment(row, int(start), int(stop))
                if segment.shape != body_log_ai[row, start:stop].shape:
                    raise ValueError("Forward-model segment returned an unexpected shape.")
                if not bool(torch.all(torch.isfinite(segment)).item()):
                    raise ValueError("Forward-model segment contains non-finite values.")
                output[row, start:stop] = segment
        if not bool(torch.any(support).item()):
            raise ValueError("Forward-model support is empty.")
        return output


class TimeDomainAdapter(DomainAdapter):
    sample_domain = "time"
    adapter_id = "time_twt_stationary_v1"

    def vertical_coordinates_m(self, batch: CommonObservationBatch) -> Tensor:
        self._require_domain(batch.sample_axis)
        explicit = batch.domain_extras.get("depth_by_sample_m")
        if explicit is not None:
            if explicit.shape != batch.observed_seismic.shape:
                raise ValueError("depth_by_sample_m must match the common trace shape.")
            return explicit
        velocity = batch.domain_extras.get("velocity_mps")
        if velocity is None:
            raise ValueError("Time adapter requires depth_by_sample_m or velocity_mps for metre smoothing.")
        return depth_coordinates_from_twt(
            velocity,
            torch.as_tensor(
                batch.sample_axis.values,
                device=velocity.device,
                dtype=velocity.dtype,
            ),
        )

    def forward(self, body_log_ai: Tensor, batch: CommonObservationBatch) -> Tensor:
        self._require_domain(batch.sample_axis)
        wavelet_time = self.wavelet_time_s.to(device=body_log_ai.device, dtype=body_log_ai.dtype)
        wavelet_amplitude = self.wavelet_amplitude.to(device=body_log_ai.device, dtype=body_log_ai.dtype)
        return self._forward_supported(
            body_log_ai,
            batch,
            lambda row, start, stop: forward_time(
                body_log_ai[row, start:stop],
                wavelet_time,
                wavelet_amplitude,
                sample_step_s=float(batch.sample_axis.step),
            ),
        )


class DepthDomainAdapter(DomainAdapter):
    sample_domain = "depth"
    adapter_id = "depth_tvdss_nonstationary_v1"

    def vertical_coordinates_m(self, batch: CommonObservationBatch) -> Tensor:
        self._require_domain(batch.sample_axis)
        axis = torch.as_tensor(
            batch.sample_axis.values,
            device=batch.observed_seismic.device,
            dtype=batch.observed_seismic.dtype,
        )
        return axis

    def forward(self, body_log_ai: Tensor, batch: CommonObservationBatch) -> Tensor:
        self._require_domain(batch.sample_axis)
        velocity = batch.domain_extras.get("velocity_mps")
        if velocity is None or velocity.shape != body_log_ai.shape:
            raise ValueError("Depth adapter requires velocity_mps matching body_log_ai.")
        if not torch.is_floating_point(velocity) or bool(torch.any(torch.isinf(velocity)).item()):
            raise ValueError("Depth adapter velocity_mps must be floating without infinite values.")
        depth = torch.as_tensor(
            batch.sample_axis.values,
            device=body_log_ai.device,
            dtype=body_log_ai.dtype,
        )
        velocity = velocity.to(device=body_log_ai.device, dtype=body_log_ai.dtype)
        wavelet_time = self.wavelet_time_s.to(device=body_log_ai.device, dtype=body_log_ai.dtype)
        wavelet_amplitude = self.wavelet_amplitude.to(device=body_log_ai.device, dtype=body_log_ai.dtype)
        return self._forward_supported(
            body_log_ai,
            batch,
            lambda row, start, stop: forward_depth(
                body_log_ai[row, start:stop],
                velocity[row, start:stop],
                depth[start:stop],
                wavelet_time,
                wavelet_amplitude,
            ),
        )


__all__ = [
    "CommonObservationBatch",
    "DepthDomainAdapter",
    "DomainAdapter",
    "ForwardClosureResult",
    "TimeDomainAdapter",
    "depth_coordinates_from_twt",
]
