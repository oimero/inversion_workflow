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
    if not torch.allclose(
        differences,
        dt_s.expand_as(differences),
        rtol=1e-6,
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


def _convolve_same_length(reflectivity: Tensor, wavelet_amp: Tensor) -> Tensor:
    samples = reflectivity.shape[-1]
    flattened = reflectivity.reshape((-1, 1, samples))
    kernel = torch.flip(wavelet_amp, dims=[-1]).reshape((1, 1, -1))
    full = F.conv1d(flattened, kernel, padding=wavelet_amp.numel() - 1)
    minimum = min(samples, wavelet_amp.numel())
    maximum = max(samples, wavelet_amp.numel())
    start = (minimum - 1) // 2
    if wavelet_amp.numel() > samples:
        start += (maximum - samples) // 2
    output = full[..., start : start + samples]
    return output.reshape(reflectivity.shape)


def forward_time(
    log_ai: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
) -> Tensor:
    """Apply the time-domain Robinson forward model and return ``[..., N-1]``."""
    values = _validate_log_ai(log_ai)
    _, amplitude = _validate_wavelet(wavelet_time_s, wavelet_amp)
    dtype = _promoted_dtype(values, amplitude)
    values_cast = values.to(dtype=dtype)
    amplitude_cast = amplitude.to(device=values.device, dtype=dtype)
    reflectivity = torch.tanh(
        0.5 * (values_cast[..., 1:] - values_cast[..., :-1])
    )
    return _convolve_same_length(reflectivity, amplitude_cast)


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


def build_depth_operator(
    velocity_mps: Any,
    depth_m: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
    *,
    output_chunk_size: int = DEFAULT_OUTPUT_CHUNK_SIZE,
) -> Tensor:
    """Build and return ``W_depth[..., N, N-1]`` in output-depth chunks."""
    velocity, depth = _validate_velocity_depth(velocity_mps, depth_m)
    wavelet_time, amplitude = _validate_wavelet(wavelet_time_s, wavelet_amp)
    chunk_size = _validate_output_chunk_size(output_chunk_size)
    dtype = _promoted_dtype(velocity, depth, wavelet_time, amplitude)
    device = velocity.device
    n_samples = velocity.shape[-1]
    velocity_flat = velocity.to(dtype=dtype).reshape((-1, n_samples))
    depth_cast = depth.to(device=device, dtype=dtype)
    wavelet_time_cast = wavelet_time.to(device=device, dtype=dtype)
    amplitude_cast = amplitude.to(device=device, dtype=dtype)
    sample_twt, interface_twt = _relative_twt_axes(velocity_flat, depth_cast)
    chunks = []
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        chunks.append(
            _wavelet_weights(
                sample_twt,
                interface_twt,
                wavelet_time_cast,
                amplitude_cast,
                start=start,
                stop=stop,
            )
        )
    operator = torch.cat(chunks, dim=1)
    return operator.reshape((*velocity.shape[:-1], n_samples, n_samples - 1))


def forward_depth(
    log_ai: Any,
    velocity_mps: Any,
    depth_m: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
    *,
    output_chunk_size: int = DEFAULT_OUTPUT_CHUNK_SIZE,
    return_operator: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Synthesize ``N`` TVDSS samples without materializing ``W_depth`` by default."""
    if not isinstance(return_operator, bool):
        raise TypeError("return_operator must be boolean.")
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
    operator_chunks = [] if return_operator else None
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
        if operator_chunks is not None:
            operator_chunks.append(weights)
    output = torch.cat(output_chunks, dim=-1).reshape(log_values.shape)
    if operator_chunks is None:
        return output
    operator = torch.cat(operator_chunks, dim=1).reshape(
        (*log_values.shape[:-1], n_samples, n_samples - 1)
    )
    return output, operator


def _validate_relation_coefficients(*, a: float, b: float) -> tuple[float, float]:
    if isinstance(a, bool) or isinstance(b, bool):
        raise TypeError("a and b must be finite real scalars.")
    a_value = float(a)
    b_value = float(b)
    if not math.isfinite(a_value) or not math.isfinite(b_value):
        raise ValueError("a and b must be finite.")
    if a_value <= 0.0:
        raise ValueError("a must be positive.")
    return a_value, b_value


def ai_from_velocity(velocity_mps: Any, *, a: float, b: float) -> Tensor:
    """Apply ``AI = a * Vp + b`` with strict physical validation."""
    velocity = _floating_tensor(velocity_mps, name="velocity_mps")
    if bool(torch.any(velocity <= 0.0).item()):
        raise ValueError("velocity_mps must be positive everywhere.")
    a_value, b_value = _validate_relation_coefficients(a=a, b=b)
    impedance = a_value * velocity + b_value
    if bool(torch.any(~torch.isfinite(impedance)).item()) or bool(
        torch.any(impedance <= 0.0).item()
    ):
        raise ValueError("AI derived from velocity_mps must be finite and positive.")
    return impedance


def velocity_from_ai(ai: Any, *, a: float, b: float) -> Tensor:
    """Apply ``Vp = (AI - b) / a`` without clipping invalid velocities."""
    impedance = _floating_tensor(ai, name="ai")
    if bool(torch.any(impedance <= 0.0).item()):
        raise ValueError("ai must be positive everywhere.")
    a_value, b_value = _validate_relation_coefficients(a=a, b=b)
    velocity = (impedance - b_value) / a_value
    if bool(torch.any(~torch.isfinite(velocity)).item()) or bool(
        torch.any(velocity <= 0.0).item()
    ):
        raise ValueError("velocity derived from ai must be finite and positive.")
    return velocity


__all__ = [
    "DEFAULT_OUTPUT_CHUNK_SIZE",
    "ai_from_velocity",
    "build_depth_operator",
    "forward_depth",
    "forward_time",
    "reflectivity_from_log_ai",
    "velocity_from_ai",
]
