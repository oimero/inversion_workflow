"""Differentiable time/depth acoustics with fixed kinematics and learned wavelets."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from cup.physics.torch_backend import forward_time
from cup.seismic.geometry import SampleAxis
from cup.utils.masks import true_runs
from ginn_v3.types import ForwardResult, ObservationBatch


def _depth_forward(log_ai: Tensor, velocity: Tensor, depth: Tensor,
                   wavelet_time: Tensor, wavelet: Tensor) -> Tensor:
    """Evaluate the canonical depth operator only at nonzero interpolation pairs.

    Removing out-of-window pairs before indexing the wavelet avoids backward
    scatter of millions of zero contributions into its endpoint coefficients.
    The travel-time and interface definitions match cup.physics.torch_backend.
    """
    b, n = log_ai.shape
    dz = depth[1:] - depth[:-1]
    interval = 2.0 * dz[None, :] * 0.5 * (
        velocity[:, :-1].reciprocal() + velocity[:, 1:].reciprocal()
    )
    sample_twt = torch.cat((torch.zeros_like(interval[:, :1]), interval.cumsum(dim=-1)), dim=-1)
    interface_twt = 0.5 * (sample_twt[:, :-1] + sample_twt[:, 1:])
    tau = sample_twt[:, :, None] - interface_twt[:, None, :]
    pairs = torch.nonzero((tau >= wavelet_time[0]) & (tau <= wavelet_time[-1]), as_tuple=False)
    if pairs.numel() == 0:
        return torch.zeros_like(log_ai) + 0.0 * (log_ai.sum() + wavelet.sum())
    row, output_index, interface_index = pairs.unbind(dim=1)
    selected_tau = tau[row, output_index, interface_index]
    right = torch.searchsorted(wavelet_time, selected_tau.contiguous()).clamp(1, wavelet_time.numel() - 1)
    left = right - 1
    alpha = (selected_tau - wavelet_time[left]) / (wavelet_time[right] - wavelet_time[left])
    weights = (1.0 - alpha) * wavelet[left] + alpha * wavelet[right]
    reflectivity = torch.tanh(0.5 * (log_ai[:, 1:] - log_ai[:, :-1]))
    events = weights * reflectivity[row, interface_index]
    flat = torch.zeros(b * n, dtype=log_ai.dtype, device=log_ai.device)
    return flat.index_add(0, row * n + output_index, events).reshape(b, n)


class AcousticPhysics:
    """Shared time/depth interface; only impedance and wavelet carry gradients."""

    def __init__(self, sample_axis: SampleAxis, wavelet_time_s: np.ndarray | Tensor):
        self.sample_axis = sample_axis
        if isinstance(wavelet_time_s, Tensor):
            times = wavelet_time_s.detach().cpu().numpy().astype(np.float64)
        else:
            times = np.asarray(wavelet_time_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 3 or times.size % 2 != 1:
            raise ValueError("The wavelet needs an odd one-dimensional time grid of at least three samples.")
        if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
            raise ValueError("Wavelet times must be finite and increasing in seconds.")
        if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-6, atol=1e-12):
            raise ValueError("Wavelet times must be regularly sampled.")
        if not np.isclose(times[times.size // 2], 0.0, rtol=0.0, atol=1e-10):
            raise ValueError("The wavelet centre must be zero seconds.")
        if sample_axis.values.size < 2:
            raise ValueError("Acoustic inversion needs at least two samples.")
        if sample_axis.domain == "time" and not np.isclose(
                np.diff(times)[0], sample_axis.step, rtol=1e-6, atol=1e-12):
            raise ValueError("Time-domain wavelet spacing must equal the seismic sample interval.")
        self.wavelet_time_s = times.copy()

    def forward(self, log_ai: Tensor, batch: ObservationBatch, wavelet_amplitude: Tensor,
                support_mask: Tensor | None = None) -> ForwardResult:
        if batch.sample_axis.domain != self.sample_axis.domain or not np.array_equal(
                batch.sample_axis.values, self.sample_axis.values):
            raise ValueError("Observation and acoustic sample axes differ.")
        if log_ai.shape != batch.initial_log_ai.shape or not torch.is_floating_point(log_ai):
            raise ValueError("Acoustic log-AI must match the floating batch trace shape.")
        if wavelet_amplitude.ndim != 1 or wavelet_amplitude.numel() != self.wavelet_time_s.size:
            raise ValueError("Acoustic forward uses one batch-mean wavelet on its fixed time grid.")
        if not bool(torch.isfinite(wavelet_amplitude).all().item()):
            raise ValueError("Learned wavelet contains nonfinite coefficients.")
        support = batch.lfm_mask.clone()
        if support_mask is not None:
            if support_mask.shape != support.shape or support_mask.dtype != torch.bool:
                raise ValueError("Explicit acoustic support must be a matching boolean mask.")
            support &= support_mask
        if bool((support & ~torch.isfinite(log_ai)).any().item()):
            raise ValueError("Supported predicted impedance contains nonfinite values.")
        velocity = batch.velocity_mps
        if self.sample_axis.domain == "depth":
            if velocity is None:
                raise ValueError("Depth acoustics requires fixed velocity in m/s.")
            if velocity.requires_grad:
                raise ValueError("GINN v3 first-version velocity is fixed during training.")
            if bool((support & (~torch.isfinite(velocity) | (velocity <= 0.0))).any().item()):
                raise ValueError("Supported depth velocity must be finite and positive.")
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for row, mask in enumerate(support.detach().cpu().numpy()):
            for start, stop in true_runs(mask):
                if stop - start >= 2:
                    grouped[(int(start), int(stop))].append(row)
        output = torch.zeros_like(log_ai)
        valid = torch.zeros_like(support)
        # Keep the long regular seconds grid in float64 for the time backend's
        # spacing validation; float32 subtraction at +/-0.6 s can lose its
        # regularity. Time convolution still uses the impedance/kernel dtype.
        time_dtype = torch.float64 if self.sample_axis.domain == "time" else log_ai.dtype
        times = torch.as_tensor(self.wavelet_time_s, device=log_ai.device, dtype=time_dtype)
        amplitude = wavelet_amplitude.to(device=log_ai.device, dtype=log_ai.dtype)
        for (start, stop), rows in grouped.items():
            selected = torch.as_tensor(rows, device=log_ai.device, dtype=torch.long)
            values = log_ai[selected, start:stop]
            if self.sample_axis.domain == "time":
                synthetic = forward_time(values, times, amplitude, sample_step_s=self.sample_axis.step)
            else:
                coordinates = torch.as_tensor(self.sample_axis.values[start:stop].copy(),
                                              device=log_ai.device, dtype=log_ai.dtype)
                synthetic = _depth_forward(values, velocity[selected, start:stop].to(log_ai.dtype),
                                           coordinates, times, amplitude)
            output[selected, start:stop] = synthetic
            valid[selected, start:stop] = True
        if not bool(valid.any().item()):
            raise ValueError("Acoustic forward has no continuous supported run of at least two samples.")
        return ForwardResult(output, valid)

