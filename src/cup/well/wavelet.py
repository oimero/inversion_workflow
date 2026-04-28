"""Shared wavelet loading and generation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ACTIVE_SUPPORT_THRESHOLD = 0.05


def make_wavelet(
    wavelet_type: str,
    freq: float,
    dt: float,
    length: int,
    gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a time-domain wavelet and return ``(time_s, amplitude)``."""
    if wavelet_type != "ricker":
        raise ValueError(f"Unsupported wavelet_type: {wavelet_type}")
    if freq <= 0.0:
        raise ValueError(f"wavelet_freq must be positive, got {freq}.")
    if dt <= 0.0:
        raise ValueError(f"wavelet_dt must be positive, got {dt}.")
    if length < 2:
        raise ValueError(f"wavelet_length must be at least 2, got {length}.")

    from wtie.modeling.wavelet import ricker

    time_s, amplitude = ricker(freq, dt, length)
    return np.asarray(time_s, dtype=np.float64), (np.asarray(amplitude, dtype=np.float64) * float(gain))


def load_wavelet_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a time-domain wavelet CSV with columns ``time_s`` and ``amplitude``."""
    path = Path(path)
    df = pd.read_csv(path)
    required = {"time_s", "amplitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"wavelet CSV is missing columns: {sorted(missing)}")

    time_s = df["time_s"].to_numpy(dtype=np.float64)
    amplitude = df["amplitude"].to_numpy(dtype=np.float64)
    finite = np.isfinite(time_s) & np.isfinite(amplitude)
    if np.count_nonzero(finite) < 2:
        raise ValueError(f"wavelet CSV does not contain enough finite samples: {path}")

    time_s = time_s[finite]
    amplitude = amplitude[finite]
    order = np.argsort(time_s)
    time_s = time_s[order]
    amplitude = amplitude[order]
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("wavelet time_s samples must be strictly increasing after sorting.")
    return time_s, amplitude


def infer_wavelet_dt(time_s: np.ndarray) -> float:
    """Return the regular sampling interval of a wavelet time axis."""
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if time_s.size < 2:
        raise ValueError("wavelet time_s must contain at least two samples.")
    deltas = np.diff(time_s)
    if np.any(deltas <= 0.0):
        raise ValueError("wavelet time_s samples must be strictly increasing.")

    dt = float(np.median(deltas))
    if not np.allclose(deltas, dt, rtol=1e-5, atol=1e-9):
        raise ValueError("wavelet time_s samples must be regularly sampled.")
    return dt


def compute_wavelet_active_half_support_s(
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
    *,
    active_threshold: float = DEFAULT_ACTIVE_SUPPORT_THRESHOLD,
) -> float:
    """Estimate wavelet active half-support in seconds.

    The active support is where ``abs(wavelet)`` is at least
    ``active_threshold`` times the wavelet peak amplitude. The returned value
    is the largest active time offset from the wavelet peak.
    """
    if not 0.0 < active_threshold <= 1.0:
        raise ValueError(f"active_threshold must be within (0, 1], got {active_threshold}.")

    wavelet_time_s = np.asarray(wavelet_time_s, dtype=np.float64).reshape(-1)
    wavelet = np.asarray(wavelet, dtype=np.float64).reshape(-1)
    if wavelet_time_s.shape != wavelet.shape:
        raise ValueError(
            f"wavelet_time_s shape {wavelet_time_s.shape} does not match wavelet shape {wavelet.shape}."
        )
    if wavelet.size == 0:
        raise ValueError("Cannot compute active half-support from an empty wavelet.")

    abs_wavelet = np.abs(wavelet)
    peak = float(abs_wavelet.max())
    if peak <= 0.0:
        raise ValueError("Cannot compute active half-support because wavelet peak amplitude is zero.")

    peak_index = int(abs_wavelet.argmax())
    active = abs_wavelet >= peak * float(active_threshold)
    return float(np.abs(wavelet_time_s[active] - wavelet_time_s[peak_index]).max())


def validate_wavelet_dt(time_s: np.ndarray, expected_dt_s: float) -> float:
    """Validate a file wavelet sampling interval against an expected seismic dt."""
    expected_dt = float(expected_dt_s)
    if expected_dt <= 0.0:
        raise ValueError(f"expected_dt_s must be positive, got {expected_dt_s}.")

    wavelet_dt = infer_wavelet_dt(time_s)
    if not np.isclose(wavelet_dt, expected_dt, rtol=1e-5, atol=1e-9):
        raise ValueError(
            "precomputed wavelet dt does not match seismic sample interval: "
            f"wavelet_dt={wavelet_dt}, seismic_dt={expected_dt}."
        )
    return wavelet_dt
