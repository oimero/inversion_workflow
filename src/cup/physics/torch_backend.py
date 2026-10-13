"""Strict differentiable PyTorch acoustic forward models.

Import this module explicitly when PyTorch support is needed.  The package
``cup.physics`` does not import PyTorch at module import time.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_OUTPUT_CHUNK_SIZE = 64


def _floating_tensor(value: Any, *, name: str, ndim: int | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating dtype, got {value.dtype}.")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(value.shape)}.")
    if not bool(torch.all(torch.isfinite(value)).item()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _validate_output_chunk_size(output_chunk_size: int) -> int:
    if isinstance(output_chunk_size, bool) or not isinstance(output_chunk_size, int):
        raise TypeError("output_chunk_size must be a positive integer.")
    if output_chunk_size <= 0:
        raise ValueError("output_chunk_size must be a positive integer.")
    return output_chunk_size


def _validate_wavelet(
    wavelet_time_s: Any,
    wavelet_amp: Any,
) -> tuple[Tensor, Tensor]:
    time = _floating_tensor(wavelet_time_s, name="wavelet_time_s", ndim=1)
    amplitude = _floating_tensor(wavelet_amp, name="wavelet_amp", ndim=1)
    if time.shape != amplitude.shape:
        raise ValueError(
            "wavelet_time_s and wavelet_amp must have exactly matching shapes."
        )
    if time.numel() < 3 or time.numel() % 2 == 0:
        raise ValueError("wavelet must have odd length >= 3.")
    differences = time[1:] - time[:-1]
    if bool(torch.any(differences <= 0.0).item()):
        raise ValueError("wavelet_time_s must be strictly increasing.")
    dt_s = differences[0]
    regular_rtol = max(1e-6, 64.0 * torch.finfo(time.dtype).eps)
    if not torch.allclose(
        differences,
        dt_s.expand_as(differences),
        rtol=regular_rtol,
        atol=1e-12,
    ):
        raise ValueError("wavelet_time_s must be regularly sampled.")
    center = time.numel() // 2
    center_tolerance = max(1e-12, abs(float(dt_s.detach().cpu())) * 1e-6)
    if not math.isclose(
        float(time[center].detach().cpu()),
        0.0,
        rel_tol=0.0,
        abs_tol=center_tolerance,
    ):
        raise ValueError("wavelet_time_s center sample must be zero seconds.")
    return time, amplitude


def _validate_log_ai(log_ai: Any) -> Tensor:
    values = _floating_tensor(log_ai, name="log_ai")
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("log_ai must have shape [..., N] with N >= 2.")
    return values


def _validate_velocity_depth(
    velocity_mps: Any,
    depth_m: Any,
) -> tuple[Tensor, Tensor]:
    velocity = _floating_tensor(velocity_mps, name="velocity_mps")
    if velocity.ndim < 1 or velocity.shape[-1] < 2:
        raise ValueError("velocity_mps must have shape [..., N] with N >= 2.")
    if bool(torch.any(velocity <= 0.0).item()):
        raise ValueError("velocity_mps must be positive everywhere.")
    depth = _floating_tensor(depth_m, name="depth_m", ndim=1)
    if depth.numel() != velocity.shape[-1]:
        raise ValueError(
            "depth_m length must match the final velocity_mps dimension, "
            f"got {depth.numel()} and {velocity.shape[-1]}."
        )
    if bool(torch.any(depth[1:] <= depth[:-1]).item()):
        raise ValueError("depth_m must be strictly increasing.")
    return velocity, depth


def _validate_depth_inputs(
    log_ai: Any,
    velocity_mps: Any,
    depth_m: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    log_values = _validate_log_ai(log_ai)
    velocity, depth = _validate_velocity_depth(velocity_mps, depth_m)
    if velocity.shape != log_values.shape:
        raise ValueError(
            "velocity_mps and log_ai must have exactly matching shapes, "
            f"got {tuple(velocity.shape)} and {tuple(log_values.shape)}."
        )
    return log_values, velocity, depth


def _promoted_dtype(*tensors: Tensor) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return dtype


def reflectivity_from_log_ai(log_ai: Any) -> Tensor:
    """Return lower-interface acoustic reflectivity with shape ``[..., N-1]``."""
    values = _validate_log_ai(log_ai)
    return torch.tanh(0.5 * (values[..., 1:] - values[..., :-1]))


def _convolve_sample_aligned(reflectivity: Tensor, wavelet_amp: Tensor) -> Tensor:
    """Return ``N`` sample-aligned values from ``N-1`` lower-sample events."""
    interfaces = reflectivity.shape[-1]
    output_samples = interfaces + 1
    flattened = reflectivity.reshape((-1, 1, interfaces))
    kernel = torch.flip(wavelet_amp, dims=[-1]).reshape((1, 1, -1))
    full = F.conv1d(flattened, kernel, padding=wavelet_amp.numel() - 1)
    start = wavelet_amp.numel() // 2 - 1
    output = full[..., start : start + output_samples]
    return output.reshape((*reflectivity.shape[:-1], output_samples))


def forward_time(
    log_ai: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
    *,
    sample_step_s: float | None = None,
) -> Tensor:
    """Apply Robinson forward modeling on an explicitly sampled time axis.

    ``sample_step_s`` is optional for low-level callers without an axis.  When
    supplied, it is checked against the wavelet spacing so a wavelet cannot be
    silently interpreted with the wrong seismic sampling interval.
    """
    values = _validate_log_ai(log_ai)
    wavelet_time, amplitude = _validate_wavelet(wavelet_time_s, wavelet_amp)
    if sample_step_s is not None:
        step = float(sample_step_s)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("sample_step_s must be finite and positive.")
        wavelet_step = wavelet_time[1] - wavelet_time[0]
        tolerance = max(1.0e-12, abs(step) * 1.0e-6)
        if not torch.allclose(
            wavelet_step,
            torch.as_tensor(step, device=wavelet_step.device, dtype=wavelet_step.dtype),
            rtol=1.0e-6,
            atol=tolerance,
        ):
            raise ValueError(
                "wavelet sample spacing differs from the seismic sample_step_s."
            )
    dtype = _promoted_dtype(values, amplitude)
    values_cast = values.to(dtype=dtype)
    amplitude_cast = amplitude.to(device=values.device, dtype=dtype)
    reflectivity = torch.tanh(
        0.5 * (values_cast[..., 1:] - values_cast[..., :-1])
    )
    return _convolve_sample_aligned(reflectivity, amplitude_cast)


def _relative_twt_axes(
    velocity_flat: Tensor,
    depth_m: Tensor,
) -> tuple[Tensor, Tensor]:
    dz = depth_m[1:] - depth_m[:-1]
    inverse_velocity_midpoint = 0.5 * (
        velocity_flat[:, :-1].reciprocal()
        + velocity_flat[:, 1:].reciprocal()
    )
    interval_twt = 2.0 * dz.unsqueeze(0) * inverse_velocity_midpoint
    sample_twt = torch.cat(
        [
            torch.zeros(
                (velocity_flat.shape[0], 1),
                device=velocity_flat.device,
                dtype=velocity_flat.dtype,
            ),
            torch.cumsum(interval_twt, dim=-1),
        ],
        dim=-1,
    )
    interface_twt = 0.5 * (sample_twt[:, :-1] + sample_twt[:, 1:])
    return sample_twt, interface_twt


def relative_twt_from_depth(
    velocity_mps: Any,
    depth_m: Any,
) -> tuple[Tensor, Tensor]:
    """Return float64 sample and midpoint-interface TWT axes.

    ``velocity_mps`` may have shape ``[..., N]`` and ``depth_m`` has shape
    ``(N,)``.  The returned arrays keep the leading velocity dimensions and
    have shapes ``[..., N]`` and ``[..., N - 1]`` respectively.  The
    kinematic calculation is intentionally promoted to float64 while the
    legacy ``forward_depth`` entry point retains its existing dtype policy.
    """

    velocity, depth = _validate_velocity_depth(velocity_mps, depth_m)
    original_shape = tuple(velocity.shape)
    n_samples = int(original_shape[-1])
    velocity_flat = velocity.to(dtype=torch.float64).reshape((-1, n_samples))
    depth_float64 = depth.to(device=velocity.device, dtype=torch.float64)
    sample_twt, interface_twt = _relative_twt_axes(velocity_flat, depth_float64)
    return (
        sample_twt.reshape(original_shape),
        interface_twt.reshape((*original_shape[:-1], n_samples - 1)),
    )


def _wavelet_weights(
    sample_twt: Tensor,
    interface_twt: Tensor,
    wavelet_time_s: Tensor,
    wavelet_amp: Tensor,
    *,
    start: int,
    stop: int,
) -> Tensor:
    tau_s = sample_twt[:, start:stop, None] - interface_twt[:, None, :]
    outside = (tau_s < wavelet_time_s[0]) | (tau_s > wavelet_time_s[-1])
    right = torch.searchsorted(wavelet_time_s, tau_s.contiguous(), right=False)
    right = right.clamp(min=1, max=wavelet_time_s.numel() - 1)
    left = right - 1
    t0 = wavelet_time_s[left]
    t1 = wavelet_time_s[right]
    a0 = wavelet_amp[left]
    a1 = wavelet_amp[right]
    alpha = (tau_s - t0) / (t1 - t0)
    values = a0 + alpha * (a1 - a0)
    return torch.where(outside, torch.zeros_like(values), values)


def _validate_sample_times(
    sample_time_s: Any,
    *,
    batch_size: int,
    n_samples: int,
    device: torch.device,
) -> Tensor:
    """Validate and broadcast a common or per-trace sample-time axis."""

    values = _floating_tensor(sample_time_s, name="sample_time_s")
    if values.ndim == 1:
        if values.numel() != n_samples:
            raise ValueError(
                "sample_time_s length must match the final log_ai dimension, "
                f"got {values.numel()} and {n_samples}."
            )
        values = values.reshape(1, n_samples).expand(batch_size, -1)
    elif values.ndim == 2:
        if values.shape != (batch_size, n_samples):
            raise ValueError(
                "A per-trace sample_time_s axis must match the flattened log_ai "
                f"shape {(batch_size, n_samples)}, got {tuple(values.shape)}."
            )
    else:
        raise ValueError("sample_time_s must be one-dimensional or have shape (B, N).")
    values = values.to(device=device, dtype=torch.float64)
    if bool(torch.any(values[:, 1:] <= values[:, :-1]).item()):
        raise ValueError("sample_time_s must be strictly increasing on every trace.")
    return values


def forward_at_interfaces(
    log_ai: Any,
    sample_time_s: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
    *,
    output_chunk_size: int = DEFAULT_OUTPUT_CHUNK_SIZE,
) -> Tensor:
    """Forward model on sample-midpoint interfaces and arbitrary time axes.

    ``sample_time_s`` is either one common ``(N,)`` relative-time axis or a
    per-trace ``(B, N)`` axis.  The operator forms reflectivity between
    adjacent impedance samples, places each event at the midpoint sample time,
    linearly interpolates the kernel in seconds, and omits kernel-outside
    pairs.  Kinematic times are float64; impedance, kernel, and output retain
    their promoted input dtype.
    """

    values = _validate_log_ai(log_ai)
    wavelet_time, amplitude = _validate_wavelet(wavelet_time_s, wavelet_amp)
    chunk_size = _validate_output_chunk_size(output_chunk_size)
    n_samples = int(values.shape[-1])
    batch_size = math.prod(values.shape[:-1]) if values.ndim > 1 else 1
    output_shape = tuple(values.shape)
    device = values.device
    sample_times = _validate_sample_times(
        sample_time_s,
        batch_size=batch_size,
        n_samples=n_samples,
        device=device,
    )
    dtype = _promoted_dtype(values, amplitude)
    values_cast = values.to(dtype=dtype).reshape(batch_size, n_samples)
    amplitude_cast = amplitude.to(device=device, dtype=dtype)
    wavelet_times = wavelet_time.to(device=device, dtype=torch.float64)
    reflectivity = torch.tanh(
        0.5 * (values_cast[:, 1:] - values_cast[:, :-1])
    )
    interface_times = 0.5 * (sample_times[:, :-1] + sample_times[:, 1:])
    outputs: list[Tensor] = []
    wavelet_min = wavelet_times[0]
    wavelet_max = wavelet_times[-1]
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        tau = sample_times[:, start:stop, None] - interface_times[:, None, :]
        pairs = torch.nonzero(
            (tau >= wavelet_min) & (tau <= wavelet_max),
            as_tuple=False,
        )
        if pairs.numel() == 0:
            zero = (values_cast.sum() + amplitude_cast.sum()) * 0.0
            outputs.append(torch.zeros(
                (batch_size, stop - start),
                dtype=dtype,
                device=device,
            ) + zero)
            continue
        row, output_index, interface_index = pairs.unbind(dim=1)
        selected_tau = tau[row, output_index, interface_index]
        right = torch.searchsorted(
            wavelet_times,
            selected_tau.contiguous(),
            right=False,
        ).clamp(1, wavelet_times.numel() - 1)
        left = right - 1
        alpha = (
            (selected_tau - wavelet_times[left])
            / (wavelet_times[right] - wavelet_times[left])
        ).to(dtype=dtype)
        weights = (1.0 - alpha) * amplitude_cast[left] + alpha * amplitude_cast[right]
        events = weights * reflectivity[row, interface_index]
        flat = torch.zeros(
            batch_size * (stop - start),
            dtype=dtype,
            device=device,
        ).index_add(
            0,
            row * (stop - start) + output_index,
            events,
        )
        outputs.append(flat.reshape(batch_size, stop - start))
    return torch.cat(outputs, dim=1).reshape(output_shape)


def forward_depth(
    log_ai: Any,
    velocity_mps: Any,
    depth_m: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
    *,
    output_chunk_size: int = DEFAULT_OUTPUT_CHUNK_SIZE,
) -> Tensor:
    """Synthesize ``N`` TVDSS samples without materializing ``W_depth`` by default."""
    log_values, velocity, depth = _validate_depth_inputs(log_ai, velocity_mps, depth_m)
    wavelet_time, amplitude = _validate_wavelet(wavelet_time_s, wavelet_amp)
    chunk_size = _validate_output_chunk_size(output_chunk_size)
    dtype = _promoted_dtype(log_values, velocity, depth, wavelet_time, amplitude)
    device = log_values.device
    n_samples = log_values.shape[-1]
    log_flat = log_values.to(dtype=dtype).reshape((-1, n_samples))
    velocity_flat = velocity.to(device=device, dtype=dtype).reshape((-1, n_samples))
    depth_cast = depth.to(device=device, dtype=dtype)
    wavelet_time_cast = wavelet_time.to(device=device, dtype=dtype)
    amplitude_cast = amplitude.to(device=device, dtype=dtype)
    reflectivity = torch.tanh(0.5 * (log_flat[:, 1:] - log_flat[:, :-1]))
    sample_twt, interface_twt = _relative_twt_axes(velocity_flat, depth_cast)
    output_chunks = []
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        weights = _wavelet_weights(
            sample_twt,
            interface_twt,
            wavelet_time_cast,
            amplitude_cast,
            start=start,
            stop=stop,
        )
        output_chunks.append(torch.bmm(weights, reflectivity.unsqueeze(-1)).squeeze(-1))
    output = torch.cat(output_chunks, dim=-1).reshape(log_values.shape)
    return output


__all__ = [
    "DEFAULT_OUTPUT_CHUNK_SIZE",
    "forward_at_interfaces",
    "forward_depth",
    "forward_time",
    "relative_twt_from_depth",
    "reflectivity_from_log_ai",
]
