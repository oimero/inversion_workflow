"""Time/depth adapters behind one GINN V2 forward-and-scale interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from cup.physics.torch_backend import forward_depth, forward_time
from cup.seismic.geometry import SampleAxis
from ginn_v2.contracts import CommonObservationBatch, ForwardClosureResult
from ginn_v2.scales import BODY_SMOOTHING_FWHM_M, depth_coordinates_from_twt, gaussian_smooth_torch


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
        raw_log_ai: Tensor,
        batch: CommonObservationBatch,
        *,
        body_smoothing_fwhm_m: float = BODY_SMOOTHING_FWHM_M,
    ) -> ForwardClosureResult:
        """Apply the fixed body scale then the frozen acoustic forward."""
        self._require_domain(batch.sample_axis)
        if raw_log_ai.shape != batch.observed_seismic.shape:
            raise ValueError("raw_log_ai must match the common batch trace shape.")
        body = gaussian_smooth_torch(
            raw_log_ai,
            self.vertical_coordinates_m(batch),
            fwhm_m=body_smoothing_fwhm_m,
        )
        synthetic = self.forward(body, batch)
        return ForwardClosureResult(body, synthetic, batch.observed_valid_mask)


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
        return forward_time(
            body_log_ai,
            self.wavelet_time_s.to(device=body_log_ai.device, dtype=body_log_ai.dtype),
            self.wavelet_amplitude.to(device=body_log_ai.device, dtype=body_log_ai.dtype),
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
        depth = torch.as_tensor(
            batch.sample_axis.values,
            device=body_log_ai.device,
            dtype=body_log_ai.dtype,
        )
        return forward_depth(
            body_log_ai,
            velocity.to(device=body_log_ai.device, dtype=body_log_ai.dtype),
            depth,
            self.wavelet_time_s.to(device=body_log_ai.device, dtype=body_log_ai.dtype),
            self.wavelet_amplitude.to(device=body_log_ai.device, dtype=body_log_ai.dtype),
        )


__all__ = ["DepthDomainAdapter", "DomainAdapter", "TimeDomainAdapter"]
