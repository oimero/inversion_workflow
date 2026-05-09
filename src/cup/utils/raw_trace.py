"""Basic preprocessing helpers for raw 1D and 2D traces."""

from __future__ import annotations

import numpy as np


def zscore_trace(values: np.ndarray, *, nan_method: str = "linear") -> np.ndarray:
    """Interpolate missing samples and z-score a full 1D trace."""
    from wtie.processing.logs import interpolate_nans

    x = interpolate_nans(values, method=nan_method)
    mean = float(np.mean(x))
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Trace has non-positive standard deviation.")
    return (x - mean) / scale


def zscore_traces_axis(values: np.ndarray) -> np.ndarray:
    """Z-score every trace (row) of a 2-D array independently (axis=1).

    NaN positions are excluded from mean/std computation.
    """
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1, keepdims=True)
    if np.any(counts <= 0):
        raise ValueError("At least one trace has no finite samples.")
    means = np.where(finite, values, 0.0).sum(axis=1, keepdims=True) / counts
    centered = np.where(finite, values - means, 0.0)
    stds = np.sqrt((centered ** 2).sum(axis=1, keepdims=True) / counts)
    if np.any(~np.isfinite(stds) | (stds <= 0.0)):
        raise ValueError("At least one trace has non-positive standard deviation.")
    return centered / stds


# ── Centered moving window ──


def _window_radius(window: int) -> tuple[int, int]:
    w = int(window)
    if w < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    left = w // 2
    right = w - 1 - left
    return left, right


def centered_moving_sum(values: np.ndarray, window: int) -> np.ndarray:
    """1-D centered moving sum with zero-padding at edges."""
    values = np.asarray(values, dtype=float)
    left, right = _window_radius(window)
    padded = np.pad(values, (left, right), mode="constant", constant_values=0.0)
    cumsum = np.cumsum(np.insert(padded, 0, 0.0))
    return cumsum[window:] - cumsum[:-window]


def centered_moving_sum_axis(values: np.ndarray, window: int) -> np.ndarray:
    """2-D centered moving sum along axis=1 (batch of 1-D traces)."""
    values = np.asarray(values, dtype=np.float32)
    left, right = _window_radius(window)
    padded = np.pad(values, ((0, 0), (left, right)), mode="constant", constant_values=0.0)
    cumsum = np.cumsum(
        np.concatenate([np.zeros((values.shape[0], 1), dtype=np.float64), padded], axis=1),
        axis=1, dtype=np.float64,
    )
    return cumsum[:, window:] - cumsum[:, :-window]


def centered_moving_rms(values: np.ndarray, window: int) -> np.ndarray:
    """1-D centered moving RMS, treating NaN as missing data."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    numerator = centered_moving_sum(np.where(valid, values ** 2, 0.0), window)
    denominator = centered_moving_sum(valid.astype(float), window)
    out = np.full(values.shape, np.nan, dtype=float)
    positive = denominator > 0.0
    out[positive] = np.sqrt(numerator[positive] / denominator[positive])
    return out


def centered_moving_rms_axis(values: np.ndarray, window: int) -> np.ndarray:
    """2-D centered moving RMS along axis=1, treating NaN as missing data."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    numerator = centered_moving_sum_axis(np.where(valid, values ** 2, 0.0), window)
    denominator = centered_moving_sum_axis(valid.astype(np.float32), window)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    positive = denominator > 0.0
    out[positive] = np.sqrt(numerator[positive] / denominator[positive]).astype(np.float32)
    return out


# ── Unit conversion ──


def meters_to_odd_samples(
    window_m: float, sample_step_m: float, *, min_samples: int = 3,
) -> int:
    """Convert a physical window length (m) to the nearest odd sample count."""
    n = int(round(float(window_m) / float(sample_step_m)))
    n = max(n, int(min_samples))
    if n % 2 == 0:
        n += 1
    return n
