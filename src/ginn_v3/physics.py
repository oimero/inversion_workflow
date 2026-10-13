"""Differentiable time/depth acoustics with fixed kinematics and learned wavelets."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from cup.physics.torch_backend import forward_at_interfaces, relative_twt_from_depth
from cup.seismic.geometry import SampleAxis
from cup.utils.masks import true_runs
from ginn_v3.types import ForwardResult, ObservationBatch


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
        times = torch.as_tensor(self.wavelet_time_s, device=log_ai.device, dtype=torch.float64)
        for (start, stop), rows in grouped.items():
            selected = torch.as_tensor(rows, device=log_ai.device, dtype=torch.long)
            values = log_ai[selected, start:stop]
            if self.sample_axis.domain == "time":
                sample_times = torch.as_tensor(
                    self.sample_axis.values[start:stop].copy(),
                    device=log_ai.device,
                    dtype=torch.float64,
                )
                sample_times = sample_times - sample_times[0]
            else:
                coordinates = torch.as_tensor(
                    self.sample_axis.values[start:stop].copy(),
                    device=log_ai.device,
                    dtype=torch.float64,
                )
                sample_times, _ = relative_twt_from_depth(
                    velocity[selected, start:stop],
                    coordinates,
                )
            synthetic = forward_at_interfaces(
                values,
                sample_times,
                times,
                wavelet_amplitude,
            )
            output[selected, start:stop] = synthetic
            valid[selected, start:stop] = True
        if not bool(valid.any().item()):
            raise ValueError("Acoustic forward has no continuous supported run of at least two samples.")
        return ForwardResult(output, valid)

