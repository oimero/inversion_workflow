"""Support masks for finite-run acoustic forward modeling.

The numerical forward operators keep each finite run independent.  A sample
next to an internal gap can nevertheless contain a partial response: the
missing midpoint interface would contribute inside the wavelet support.  This
module computes the complete-response mask used by losses and QC while
leaving the synthesized values untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cup.utils.masks import true_runs


def longest_true_run(mask: Any, *, minimum_samples: int = 1) -> tuple[int, int]:
    """Return the deterministic longest half-open true interval in ``mask``."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional.")
    if isinstance(minimum_samples, bool) or int(minimum_samples) != minimum_samples:
        raise TypeError("minimum_samples must be an integer.")
    minimum = int(minimum_samples)
    if minimum < 1:
        raise ValueError("minimum_samples must be positive.")
    runs = true_runs(values)
    if not runs:
        raise ValueError("mask contains no true samples.")
    best = max(runs, key=lambda run: (run[1] - run[0], -run[0]))
    if best[1] - best[0] < minimum:
        raise ValueError(
            f"mask has no true run with at least {minimum} samples; "
            f"the longest run has {best[1] - best[0]}."
        )
    return best


def _wavelet_times(wavelet_time_s: Any) -> np.ndarray:
    values = np.asarray(wavelet_time_s, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or values.size % 2 == 0:
        raise ValueError("wavelet_time_s must be an odd one-dimensional axis of at least three samples.")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError("wavelet_time_s must be finite and strictly increasing.")
    if not np.isclose(values[values.size // 2], 0.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("wavelet_time_s must be centred at zero seconds.")
    return values


def complete_response_mask(
    sample_time_s: Any,
    support_mask: Any,
    wavelet_time_s: Any,
) -> np.ndarray:
    """Return samples whose finite-run response is complete around gaps.

    ``sample_time_s`` is a common ``(N,)`` or per-trace ``(B, N)`` seconds
    axis.  ``support_mask`` has the same shape (or is one-dimensional for a
    common axis).  For every internal false run, its first and last missing
    reflection interfaces are placed at the adjacent sample midpoints.
    Samples on the left are
    excluded when their wavelet window reaches the negative kernel endpoint;
    samples on the right are excluded when it reaches the positive endpoint.

    Only internal gap edges are eroded.  The first and last sample of a
    calculation window retain the existing finite-run truncation convention.
    The returned mask never marks an unsupported sample as valid.
    """

    times = np.asarray(sample_time_s, dtype=np.float64)
    support = np.asarray(support_mask, dtype=bool)
    return_1d = times.ndim == 1 and support.ndim == 1
    if times.ndim == 1:
        times = times[None, :]
    elif times.ndim != 2:
        raise ValueError("sample_time_s must be one-dimensional or have shape (B, N).")
    if support.ndim == 1:
        support = support[None, :]
    elif support.ndim != 2:
        raise ValueError("support_mask must be one-dimensional or have shape (B, N).")
    if (
        times.shape != support.shape
        and times.shape[0] == 1
        and support.ndim == 2
        and times.shape[1] == support.shape[1]
    ):
        times = np.broadcast_to(times, support.shape).copy()
    elif (
        times.shape != support.shape
        and support.shape[0] == 1
        and times.ndim == 2
        and support.shape[1] == times.shape[1]
    ):
        support = np.broadcast_to(support, times.shape).copy()
    if times.shape != support.shape:
        raise ValueError(
            "sample_time_s and support_mask must have matching shapes, "
            f"got {times.shape} and {support.shape}."
        )
    if times.shape[1] < 1:
        raise ValueError("sample_time_s must contain at least one sample.")
    wavelet = _wavelet_times(wavelet_time_s)
    if np.any(support & ~np.isfinite(times)):
        raise ValueError("sample_time_s must be finite on supported samples.")

    result = support.copy()
    wavelet_min = float(wavelet[0])
    wavelet_max = float(wavelet[-1])
    for row in range(times.shape[0]):
        runs = true_runs(support[row])
        for (_left_start, left_stop), (right_start, _right_stop) in zip(runs, runs[1:]):
            left_index = left_stop - 1
            right_index = right_start
            first_gap = left_stop
            last_gap = right_start - 1
            boundary_times = times[row, [left_index, first_gap, last_gap, right_index]]
            if not np.all(np.isfinite(boundary_times)):
                raise ValueError("sample_time_s must be finite around every internal support gap.")
            left_interface = 0.5 * (boundary_times[0] + boundary_times[1])
            right_interface = 0.5 * (boundary_times[2] + boundary_times[3])
            if not right_interface > left_interface:
                raise ValueError("sample_time_s must increase across every internal support gap.")
            left_cut = left_interface + wavelet_min
            right_cut = right_interface + wavelet_max
            left_indices = np.arange(_left_start, left_stop)
            right_indices = np.arange(right_start, _right_stop)
            result[row, left_indices[times[row, left_indices] >= left_cut]] = False
            result[row, right_indices[times[row, right_indices] <= right_cut]] = False
    if return_1d:
        return result[0]
    return result


def relative_twt_with_gaps(
    velocity_mps: Any,
    depth_m: Any,
    support_mask: Any,
) -> np.ndarray:
    """Build per-trace relative TWT axes without crossing impedance gaps.

    When the fixed velocity is finite across the supported span, that span is
    integrated with the same midpoint-velocity rule as the depth forward
    operator.  Otherwise, valid runs use that rule independently and the
    interval from the last valid depth to the first valid depth is represented
    by one kinematic interval using the two finite endpoint velocities.  This
    gives the missing midpoint interfaces physical seconds positions while
    preserving the impedance gap itself.  Unsupported samples receive
    interpolated kinematic times only; callers must continue to use
    ``support_mask`` for signal operations.
    """

    velocity = np.asarray(velocity_mps, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    support = np.asarray(support_mask, dtype=bool)
    return_1d = velocity.ndim == 1 and support.ndim == 1
    if velocity.ndim == 1:
        velocity = velocity[None, :]
    elif velocity.ndim != 2:
        raise ValueError("velocity_mps must be one-dimensional or have shape (B, N).")
    if depth.ndim != 1 or depth.size != velocity.shape[1]:
        raise ValueError("depth_m must be a one-dimensional axis matching velocity_mps.")
    if support.ndim == 1:
        support = support[None, :]
    elif support.ndim != 2:
        raise ValueError("support_mask must be one-dimensional or have shape (B, N).")
    if support.ndim == 2 and velocity.shape[0] == 1 and support.shape[1] == velocity.shape[1]:
        velocity = np.broadcast_to(velocity, support.shape).copy()
    elif support.ndim == 2 and support.shape[0] == 1 and velocity.shape[1] == support.shape[1]:
        support = np.broadcast_to(support, velocity.shape).copy()
    elif support.ndim == 2 and support.shape != velocity.shape:
        raise ValueError("support_mask must match velocity_mps.")
    if support.shape != velocity.shape:
        raise ValueError("support_mask must match velocity_mps.")
    if not np.all(np.isfinite(depth)) or np.any(np.diff(depth) <= 0.0):
        raise ValueError("depth_m must be finite and strictly increasing.")
    if np.any(support & (~np.isfinite(velocity) | (velocity <= 0.0))):
        raise ValueError("velocity_mps must be finite and positive on supported samples.")

    result = np.full(velocity.shape, np.nan, dtype=np.float64)
    for row in range(velocity.shape[0]):
        runs = true_runs(support[row])
        if not runs:
            continue
        span_start, span_stop = runs[0][0], runs[-1][1]
        span_velocity = velocity[row, span_start:span_stop]
        if np.all(np.isfinite(span_velocity)) and np.all(span_velocity > 0.0):
            dz = np.diff(depth[span_start:span_stop])
            inv_midpoint = 0.5 * (
                1.0 / span_velocity[:-1] + 1.0 / span_velocity[1:]
            )
            result[row, span_start] = 0.0
            result[row, span_start + 1:span_stop] = np.cumsum(2.0 * dz * inv_midpoint)
            continue
        previous_stop: int | None = None
        for start, stop in runs:
            if previous_stop is None:
                origin = 0.0
            else:
                left = previous_stop - 1
                right = start
                gap_twt = 2.0 * (depth[right] - depth[left]) * 0.5 * (
                    1.0 / velocity[row, left] + 1.0 / velocity[row, right]
                )
                origin = float(result[row, left] + gap_twt)
                result[row, previous_stop:right] = np.linspace(
                    result[row, left], origin, right - left + 1,
                )[1:-1]
            result[row, start] = origin
            if stop - start > 1:
                dz = np.diff(depth[start:stop])
                inv_midpoint = 0.5 * (
                    1.0 / velocity[row, start:stop - 1]
                    + 1.0 / velocity[row, start + 1:stop]
                )
                result[row, start + 1:stop] = origin + np.cumsum(2.0 * dz * inv_midpoint)
            previous_stop = stop
    if return_1d:
        return result[0]
    return result


__all__ = ["complete_response_mask", "longest_true_run", "relative_twt_with_gaps"]
