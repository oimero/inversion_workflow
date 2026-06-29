"""Strict NumPy acoustic forward models for time and depth sample domains.

The functions in this module are numerical kernels.  They do not fill missing
values, infer units, normalize amplitudes, or apply gain.
"""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_OUTPUT_CHUNK_SIZE = 64


def _floating_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype, got {array.dtype}.")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_output_chunk_size(output_chunk_size: int) -> int:
    if isinstance(output_chunk_size, bool) or not isinstance(output_chunk_size, (int, np.integer)):
        raise TypeError("output_chunk_size must be a positive integer.")
    size = int(output_chunk_size)
    if size <= 0:
        raise ValueError("output_chunk_size must be a positive integer.")
    return size


def _validate_wavelet(
    wavelet_time_s: Any,
    wavelet_amp: Any,
) -> tuple[np.ndarray, np.ndarray]:
    time = _floating_array(wavelet_time_s, name="wavelet_time_s", ndim=1)
    amplitude = _floating_array(wavelet_amp, name="wavelet_amp", ndim=1)
    if time.shape != amplitude.shape:
        raise ValueError(
            "wavelet_time_s and wavelet_amp must have exactly matching shapes."
        )
    if time.size < 3 or time.size % 2 == 0:
        raise ValueError("wavelet must have odd length >= 3.")
    differences = np.diff(time)
    if np.any(differences <= 0.0):
        raise ValueError("wavelet_time_s must be strictly increasing.")
    dt_s = float(differences[0])
    if not np.allclose(differences, dt_s, rtol=1e-6, atol=1e-12):
        raise ValueError("wavelet_time_s must be regularly sampled.")
    center = time.size // 2
    center_tolerance = max(1e-12, abs(dt_s) * 1e-6)
    if not np.isclose(time[center], 0.0, rtol=0.0, atol=center_tolerance):
        raise ValueError("wavelet_time_s center sample must be zero seconds.")
    return time, amplitude


def _validate_log_ai(log_ai: Any) -> np.ndarray:
    values = _floating_array(log_ai, name="log_ai")
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("log_ai must have shape [..., N] with N >= 2.")
    return values


def _validate_depth_inputs(
    log_ai: Any,
    velocity_mps: Any,
    depth_m: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_values = _validate_log_ai(log_ai)
    velocity = _floating_array(velocity_mps, name="velocity_mps")
    if velocity.shape != log_values.shape:
        raise ValueError(
            "velocity_mps and log_ai must have exactly matching shapes, "
            f"got {velocity.shape} and {log_values.shape}."
        )
    if np.any(velocity <= 0.0):
        raise ValueError("velocity_mps must be positive everywhere.")
    depth = _floating_array(depth_m, name="depth_m", ndim=1)
    if depth.size != log_values.shape[-1]:
        raise ValueError(
            "depth_m length must match the final log_ai dimension, "
            f"got {depth.size} and {log_values.shape[-1]}."
        )
    if np.any(np.diff(depth) <= 0.0):
        raise ValueError("depth_m must be strictly increasing.")
    return log_values, velocity, depth


def _validate_velocity_depth(
    velocity_mps: Any,
    depth_m: Any,
) -> tuple[np.ndarray, np.ndarray]:
    velocity = _floating_array(velocity_mps, name="velocity_mps")
    if velocity.ndim < 1 or velocity.shape[-1] < 2:
        raise ValueError("velocity_mps must have shape [..., N] with N >= 2.")
    if np.any(velocity <= 0.0):
        raise ValueError("velocity_mps must be positive everywhere.")
    depth = _floating_array(depth_m, name="depth_m", ndim=1)
    if depth.size != velocity.shape[-1]:
        raise ValueError(
            "depth_m length must match the final velocity_mps dimension, "
            f"got {depth.size} and {velocity.shape[-1]}."
        )
    if np.any(np.diff(depth) <= 0.0):
        raise ValueError("depth_m must be strictly increasing.")
    return velocity, depth


def reflectivity_from_log_ai(log_ai: Any) -> np.ndarray:
    """Return lower-interface acoustic reflectivity with shape ``[..., N-1]``."""
    values = _validate_log_ai(log_ai)
    return np.tanh(0.5 * np.diff(values, axis=-1))


def _convolve_sample_aligned(reflectivity: np.ndarray, wavelet_amp: np.ndarray) -> np.ndarray:
    """Return ``N`` sample-aligned values from ``N-1`` lower-sample events.

    With an odd wavelet of half-width ``c``, output sample ``l`` is full
    convolution sample ``c + l - 1``.  Consequently ``output[..., 1:]`` is
    exactly the repository's legacy Robinson ``same`` result, including the
    historical crop behavior when the wavelet is longer than the trace.
    """
    interfaces = reflectivity.shape[-1]
    output_samples = interfaces + 1
    flattened = reflectivity.reshape((-1, interfaces))
    output = np.empty(
        (flattened.shape[0], output_samples),
        dtype=np.result_type(reflectivity, wavelet_amp),
    )
    start = wavelet_amp.size // 2 - 1
    for index, trace in enumerate(flattened):
        full = np.convolve(wavelet_amp, trace, mode="full")
        output[index] = full[start : start + output_samples]
    return output.reshape((*reflectivity.shape[:-1], output_samples))


def forward_time(
    log_ai: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
) -> np.ndarray:
    """Apply Robinson forward modeling and return ``N`` input-sample values."""
    values = _validate_log_ai(log_ai)
    _, amplitude = _validate_wavelet(wavelet_time_s, wavelet_amp)
    reflectivity = reflectivity_from_log_ai(values)
    return _convolve_sample_aligned(reflectivity, amplitude)


def _relative_twt_axes(
    velocity_flat: np.ndarray,
    depth_m: np.ndarray,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    dz = np.diff(depth_m).astype(dtype, copy=False)
    inverse_velocity_midpoint = 0.5 * (
        np.reciprocal(velocity_flat[:, :-1])
        + np.reciprocal(velocity_flat[:, 1:])
    )
    interval_twt = 2.0 * dz[None, :] * inverse_velocity_midpoint
    sample_twt = np.empty(
        (velocity_flat.shape[0], velocity_flat.shape[1]),
        dtype=dtype,
    )
    sample_twt[:, 0] = 0.0
    sample_twt[:, 1:] = np.cumsum(interval_twt, axis=-1)
    interface_twt = 0.5 * (sample_twt[:, :-1] + sample_twt[:, 1:])
    return sample_twt, interface_twt


def _wavelet_weights(
    sample_twt: np.ndarray,
    interface_twt: np.ndarray,
    wavelet_time_s: np.ndarray,
    wavelet_amp: np.ndarray,
    *,
    start: int,
    stop: int,
    dtype: np.dtype,
) -> np.ndarray:
    tau_s = sample_twt[:, start:stop, None] - interface_twt[:, None, :]
    interpolated = np.interp(
        tau_s.reshape(-1),
        wavelet_time_s,
        wavelet_amp,
        left=0.0,
        right=0.0,
    )
    return interpolated.reshape(tau_s.shape).astype(dtype, copy=False)


def build_depth_operator(
    velocity_mps: Any,
    depth_m: Any,
    wavelet_time_s: Any,
    wavelet_amp: Any,
    *,
    output_chunk_size: int = DEFAULT_OUTPUT_CHUNK_SIZE,
) -> np.ndarray:
    """Build and return ``W_depth[..., N, N-1]`` in output-depth chunks."""
    velocity, depth = _validate_velocity_depth(velocity_mps, depth_m)
    wavelet_time, amplitude = _validate_wavelet(wavelet_time_s, wavelet_amp)
    chunk_size = _validate_output_chunk_size(output_chunk_size)
    dtype = np.dtype(np.result_type(velocity.dtype, depth.dtype, amplitude.dtype))
    n_samples = velocity.shape[-1]
    velocity_flat = velocity.astype(dtype, copy=False).reshape((-1, n_samples))
    sample_twt, interface_twt = _relative_twt_axes(
        velocity_flat,
        depth.astype(dtype, copy=False),
        dtype=dtype,
    )
    operator = np.empty(
        (velocity_flat.shape[0], n_samples, n_samples - 1),
        dtype=dtype,
    )
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        operator[:, start:stop, :] = _wavelet_weights(
            sample_twt,
            interface_twt,
            wavelet_time.astype(dtype, copy=False),
            amplitude.astype(dtype, copy=False),
            start=start,
            stop=stop,
            dtype=dtype,
        )
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
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Synthesize ``N`` TVDSS samples without materializing ``W_depth`` by default."""
    if not isinstance(return_operator, (bool, np.bool_)):
        raise TypeError("return_operator must be boolean.")
    log_values, velocity, depth = _validate_depth_inputs(log_ai, velocity_mps, depth_m)
    wavelet_time, amplitude = _validate_wavelet(wavelet_time_s, wavelet_amp)
    chunk_size = _validate_output_chunk_size(output_chunk_size)
    dtype = np.dtype(
        np.result_type(log_values.dtype, velocity.dtype, depth.dtype, amplitude.dtype)
    )
    n_samples = log_values.shape[-1]
    batch_size = int(np.prod(log_values.shape[:-1], dtype=np.int64)) or 1
    log_flat = log_values.astype(dtype, copy=False).reshape((batch_size, n_samples))
    velocity_flat = velocity.astype(dtype, copy=False).reshape((batch_size, n_samples))
    reflectivity = np.tanh(0.5 * np.diff(log_flat, axis=-1))
    sample_twt, interface_twt = _relative_twt_axes(
        velocity_flat,
        depth.astype(dtype, copy=False),
        dtype=dtype,
    )
    output = np.zeros((batch_size, n_samples), dtype=dtype)
    operator = (
        np.empty((batch_size, n_samples, n_samples - 1), dtype=dtype)
        if bool(return_operator)
        else None
    )
    wavelet_time_cast = wavelet_time.astype(dtype, copy=False)
    amplitude_cast = amplitude.astype(dtype, copy=False)
    if operator is not None:
        for start in range(0, n_samples, chunk_size):
            stop = min(start + chunk_size, n_samples)
            weights = _wavelet_weights(
                sample_twt,
                interface_twt,
                wavelet_time_cast,
                amplitude_cast,
                start=start,
                stop=stop,
                dtype=dtype,
            )
            output[:, start:stop] = np.einsum(
                "bij,bj->bi",
                weights,
                reflectivity,
                optimize=True,
            )
            operator[:, start:stop, :] = weights
    else:
        # Exact finite-support banded path.  searchsorted removes only weights
        # that the dense interpolation defines as exactly zero.
        wavelet_min = float(wavelet_time_cast[0])
        wavelet_max = float(wavelet_time_cast[-1])
        for output_start in range(0, n_samples, chunk_size):
            output_stop = min(output_start + chunk_size, n_samples)
            for batch_index in range(batch_size):
                events = interface_twt[batch_index]
                for sample_index in range(output_start, output_stop):
                    sample_time = sample_twt[batch_index, sample_index]
                    first = int(np.searchsorted(events, sample_time - wavelet_max, side="left"))
                    last = int(np.searchsorted(events, sample_time - wavelet_min, side="right"))
                    if first >= last:
                        continue
                    tau = sample_time - events[first:last]
                    weights = np.interp(
                        tau,
                        wavelet_time_cast,
                        amplitude_cast,
                        left=0.0,
                        right=0.0,
                    ).astype(dtype, copy=False)
                    output[batch_index, sample_index] = np.dot(
                        weights,
                        reflectivity[batch_index, first:last],
                    )
    output_shaped = output.reshape(log_values.shape)
    if operator is None:
        return output_shaped
    operator_shaped = operator.reshape(
        (*log_values.shape[:-1], n_samples, n_samples - 1)
    )
    return output_shaped, operator_shaped


def _validate_relation_coefficients(*, a: float, b: float) -> tuple[float, float]:
    if isinstance(a, bool) or isinstance(b, bool):
        raise TypeError("a and b must be finite real scalars.")
    a_value = float(a)
    b_value = float(b)
    if not np.isfinite(a_value) or not np.isfinite(b_value):
        raise ValueError("a and b must be finite.")
    if a_value <= 0.0:
        raise ValueError("a must be positive.")
    return a_value, b_value


def ai_from_velocity(velocity_mps: Any, *, a: float, b: float) -> np.ndarray:
    """Apply ``AI = a * Vp + b`` with strict physical validation."""
    velocity = _floating_array(velocity_mps, name="velocity_mps")
    if np.any(velocity <= 0.0):
        raise ValueError("velocity_mps must be positive everywhere.")
    a_value, b_value = _validate_relation_coefficients(a=a, b=b)
    impedance = a_value * velocity + b_value
    if np.any(~np.isfinite(impedance)) or np.any(impedance <= 0.0):
        raise ValueError("AI derived from velocity_mps must be finite and positive.")
    return impedance


def velocity_from_ai(ai: Any, *, a: float, b: float) -> np.ndarray:
    """Apply ``Vp = (AI - b) / a`` without clipping invalid velocities."""
    impedance = _floating_array(ai, name="ai")
    if np.any(impedance <= 0.0):
        raise ValueError("ai must be positive everywhere.")
    a_value, b_value = _validate_relation_coefficients(a=a, b=b)
    velocity = (impedance - b_value) / a_value
    if np.any(~np.isfinite(velocity)) or np.any(velocity <= 0.0):
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
