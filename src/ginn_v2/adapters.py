"""Time/depth adapters behind one GINN V2 forward-and-scale interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from cup.physics.torch_backend import forward_depth, forward_time
from cup.seismic.geometry import SampleAxis
from ginn_v2.contracts import CommonObservationBatch, ForwardClosureResult
from ginn_v2.scales import depth_coordinates_from_twt


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
        if not torch.is_floating_point(velocity) or bool(torch.any(torch.isinf(velocity)).item()):
            raise ValueError("Depth adapter velocity_mps must be floating without infinite values.")
        depth = torch.as_tensor(
            batch.sample_axis.values,
            device=body_log_ai.device,
            dtype=body_log_ai.dtype,
        )
        velocity = velocity.to(device=body_log_ai.device, dtype=body_log_ai.dtype)
        output = torch.zeros_like(body_log_ai)
        for row in range(body_log_ai.shape[0]):
            finite = torch.isfinite(velocity[row])
            padded = torch.cat(
                (
                    torch.zeros(1, dtype=torch.bool, device=finite.device),
                    finite,
                    torch.zeros(1, dtype=torch.bool, device=finite.device),
                )
            )
            changes = torch.nonzero(padded[1:] != padded[:-1], as_tuple=False).reshape(-1, 2)
            for start, stop in changes.tolist():
                if not bool(finite[start].item()):
                    continue
                if stop - start < 2:
                    raise ValueError("Depth adapter finite velocity support contains a run shorter than two samples.")
                output[row, start:stop] = forward_depth(
                    body_log_ai[row, start:stop],
                    velocity[row, start:stop],
                    depth[start:stop],
                    self.wavelet_time_s.to(device=body_log_ai.device, dtype=body_log_ai.dtype),
                    self.wavelet_amplitude.to(device=body_log_ai.device, dtype=body_log_ai.dtype),
                )
        if not bool(torch.any(torch.isfinite(velocity)).item()):
            raise ValueError("Depth adapter velocity_mps has no finite support.")
        return output


__all__ = ["DepthDomainAdapter", "DomainAdapter", "TimeDomainAdapter"]
