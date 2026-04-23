"""Depth-domain nonstationary wavelet forward model.

The core operator is a batch-built matrix ``W_depth``:

    d_syn[l] = sum_j W_depth[l, j] * r[j]

where ``r[j]`` is reflectivity at the interface between impedance samples
``j`` and ``j+1``.  The weights are sampled from a time-domain wavelet using
the two-way-time difference induced by a fixed depth-domain velocity trace.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _as_trace_batch(x: Tensor, name: str) -> Tensor:
    """Normalize ``(N,)``, ``(B, N)``, or ``(B, 1, N)`` to ``(B, N)``."""
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :]
    raise ValueError(f"{name} must have shape (N,), (B, N), or (B, 1, N), got {tuple(x.shape)}.")


def reflectivity(impedance: Tensor, eps: float = 1e-10) -> Tensor:
    """Compute normal-incidence acoustic reflectivity from impedance.

    Parameters
    ----------
    impedance
        Acoustic impedance with shape ``(N,)``, ``(B, N)``, or ``(B, 1, N)``.

    Returns
    -------
    Tensor
        Reflectivity with shape ``(B, N-1)``.
    """
    ai = _as_trace_batch(impedance, "impedance")
    ai_upper = ai[:, :-1]
    ai_lower = ai[:, 1:]
    return (ai_lower - ai_upper) / (ai_lower + ai_upper + float(eps))


class DepthWaveletMatrixBuilder(nn.Module):
    """Build batch ``W_depth`` matrices from depth-domain velocity traces.

    Parameters
    ----------
    wavelet_time_s, wavelet_amp
        Time-domain wavelet samples. ``wavelet_time_s`` must be sorted in
        ascending seconds and can include negative lags for zero-phase wavelets.
    depth_axis_m
        Optional fixed depth sample axis. If omitted, ``forward`` must receive
        ``depth_axis_m``.
    amplitude_threshold
        Optional absolute threshold below which weights are set to zero.

    Notes
    -----
    The output matrix has shape ``(B, N, N-1)``. It maps reflectivity at
    impedance interfaces to seismic samples at depth sample centers.
    """

    def __init__(
        self,
        wavelet_time_s: np.ndarray | Tensor,
        wavelet_amp: np.ndarray | Tensor,
        depth_axis_m: np.ndarray | Tensor | None = None,
        *,
        amplitude_threshold: float = 0.0,
    ) -> None:
        super().__init__()

        wavelet_time = torch.as_tensor(wavelet_time_s, dtype=torch.float32).flatten()
        wavelet_values = torch.as_tensor(wavelet_amp, dtype=torch.float32).flatten()
        if wavelet_time.numel() != wavelet_values.numel():
            raise ValueError("wavelet_time_s and wavelet_amp must have the same number of samples.")
        if wavelet_time.numel() < 2:
            raise ValueError("wavelet must contain at least two samples.")
        if torch.any(wavelet_time[1:] <= wavelet_time[:-1]):
            raise ValueError("wavelet_time_s must be strictly increasing.")

        self.register_buffer("wavelet_time_s", wavelet_time)
        self.register_buffer("wavelet_amp", wavelet_values)

        if depth_axis_m is None:
            self.register_buffer("depth_axis_m", torch.empty(0, dtype=torch.float32))
        else:
            depth_axis = torch.as_tensor(depth_axis_m, dtype=torch.float32).flatten()
            self._validate_depth_axis(depth_axis)
            self.register_buffer("depth_axis_m", depth_axis)

        if amplitude_threshold < 0.0:
            raise ValueError(f"amplitude_threshold must be non-negative, got {amplitude_threshold}.")
        self.amplitude_threshold = float(amplitude_threshold)

    @staticmethod
    def _validate_depth_axis(depth_axis_m: Tensor) -> None:
        if depth_axis_m.ndim != 1:
            raise ValueError("depth_axis_m must be 1D.")
        if depth_axis_m.numel() < 2:
            raise ValueError("depth_axis_m must contain at least two samples.")
        if torch.any(depth_axis_m[1:] <= depth_axis_m[:-1]):
            raise ValueError("depth_axis_m must be strictly increasing.")

    def _resolve_depth_axis(
        self, depth_axis_m: np.ndarray | Tensor | None, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if depth_axis_m is None:
            if self.depth_axis_m.numel() == 0:  # type: ignore
                raise ValueError("depth_axis_m was not provided at init time or call time.")
            depth_axis = self.depth_axis_m
        else:
            depth_axis = torch.as_tensor(depth_axis_m, dtype=torch.float32)
            self._validate_depth_axis(depth_axis.flatten())
        return depth_axis.to(device=device, dtype=dtype).flatten()  # type: ignore

    @staticmethod
    def compute_twt_axes(depth_axis_m: Tensor, velocity_mps: Tensor) -> tuple[Tensor, Tensor]:
        """Compute TWT at sample centers and impedance interfaces.

        ``velocity_mps`` must have shape ``(B, N)`` and ``depth_axis_m`` must
        have shape ``(N,)``.
        """
        if velocity_mps.ndim != 2:
            raise ValueError(f"velocity_mps must have shape (B, N), got {tuple(velocity_mps.shape)}.")
        if depth_axis_m.ndim != 1:
            raise ValueError("depth_axis_m must be 1D.")
        if velocity_mps.shape[-1] != depth_axis_m.numel():
            raise ValueError(
                f"velocity/depth length mismatch: velocity N={velocity_mps.shape[-1]}, depth N={depth_axis_m.numel()}"
            )
        if torch.any(~torch.isfinite(velocity_mps)) or torch.any(velocity_mps <= 0.0):
            raise ValueError("velocity_mps must be finite and positive everywhere.")

        dz = depth_axis_m[1:] - depth_axis_m[:-1]
        inv_v_mid = 0.5 * (velocity_mps[:, :-1].reciprocal() + velocity_mps[:, 1:].reciprocal())
        dtwt = 2.0 * dz.unsqueeze(0) * inv_v_mid
        twt_sample = torch.cat(
            [
                torch.zeros((velocity_mps.shape[0], 1), device=velocity_mps.device, dtype=velocity_mps.dtype),
                dtwt.cumsum(dim=1),
            ],
            dim=1,
        )
        twt_interface = 0.5 * (twt_sample[:, :-1] + twt_sample[:, 1:])
        return twt_sample, twt_interface

    def _interp_wavelet(self, tau_s: Tensor) -> Tensor:
        wavelet_time = self.wavelet_time_s.to(device=tau_s.device, dtype=tau_s.dtype)
        wavelet_amp = self.wavelet_amp.to(device=tau_s.device, dtype=tau_s.dtype)

        outside = (tau_s < wavelet_time[0]) | (tau_s > wavelet_time[-1])  # type: ignore
        idx_right = torch.searchsorted(wavelet_time, tau_s.contiguous(), right=False)  # type: ignore
        idx_right = idx_right.clamp(min=1, max=wavelet_time.numel() - 1)  # type: ignore
        idx_left = idx_right - 1

        t0 = wavelet_time[idx_left]  # type: ignore
        t1 = wavelet_time[idx_right]  # type: ignore
        a0 = wavelet_amp[idx_left]  # type: ignore
        a1 = wavelet_amp[idx_right]  # type: ignore
        alpha = (tau_s - t0) / (t1 - t0).clamp_min(torch.finfo(tau_s.dtype).eps)
        values = a0 + alpha * (a1 - a0)
        values = torch.where(outside, torch.zeros_like(values), values)

        if self.amplitude_threshold > 0.0:
            values = torch.where(values.abs() < self.amplitude_threshold, torch.zeros_like(values), values)
        return values

    def forward(self, velocity_mps: Tensor, depth_axis_m: np.ndarray | Tensor | None = None) -> Tensor:
        """Return ``W_depth`` with shape ``(B, N, N-1)``."""
        velocity = _as_trace_batch(velocity_mps, "velocity_mps")
        depth_axis = self._resolve_depth_axis(depth_axis_m, device=velocity.device, dtype=velocity.dtype)
        twt_sample, twt_interface = self.compute_twt_axes(depth_axis, velocity)
        tau = twt_sample[:, :, None] - twt_interface[:, None, :]
        return self._interp_wavelet(tau)


class DepthForwardModel(nn.Module):
    """Depth-domain physics forward model: ``AI -> r -> W_depth @ r``."""

    def __init__(
        self,
        wavelet_time_s: np.ndarray | Tensor,
        wavelet_amp: np.ndarray | Tensor,
        depth_axis_m: np.ndarray | Tensor | None = None,
        *,
        amplitude_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.matrix_builder = DepthWaveletMatrixBuilder(
            wavelet_time_s,
            wavelet_amp,
            depth_axis_m=depth_axis_m,
            amplitude_threshold=amplitude_threshold,
        )

    def forward(
        self,
        impedance: Tensor,
        velocity_mps: Tensor,
        depth_axis_m: np.ndarray | Tensor | None = None,
        *,
        return_matrix: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Synthesize depth-domain seismic from impedance and fixed velocity.

        Parameters
        ----------
        impedance
            Acoustic impedance with shape ``(B, 1, N)`` or ``(B, N)``.
        velocity_mps
            Depth-domain velocity with shape ``(B, 1, N)`` or ``(B, N)``.
        depth_axis_m
            Optional call-time depth axis if the model was not initialized with
            one.
        return_matrix
            If ``True``, return ``(d_syn, W_depth)``.
        """
        r = reflectivity(impedance)
        W_depth = self.matrix_builder(velocity_mps, depth_axis_m=depth_axis_m)
        if W_depth.shape[0] != r.shape[0] or W_depth.shape[-1] != r.shape[-1]:
            raise ValueError(f"W/r shape mismatch: W={tuple(W_depth.shape)}, r={tuple(r.shape)}")
        d_syn = torch.bmm(W_depth, r.unsqueeze(-1)).squeeze(-1).unsqueeze(1)
        if return_matrix:
            return d_syn, W_depth
        return d_syn
