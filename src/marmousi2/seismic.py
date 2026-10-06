"""Read AGL processed time sections and fit an effective training-well wavelet."""

from pathlib import Path
import struct

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cup.physics.numpy_backend import forward_time
from cup.seismic.wavelet import wavelet_l2_normalize, wavelet_spectrum_features
from marmousi2.raw import _read_selected_segy


def read_processed_time_segy(path: Path):
    """Return amplitude[x,t], metre positions, TWT seconds and file information.

    The AGL text header explicitly documents its unusual coordinates multiplied
    by 1000. Its depth sections use the same binary-header field as time sections,
    so the declared domain must be read before interpreting that field as dt.
    """
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(3600)
    text = raw[:3200].decode("cp500")
    if "MARMOUSI" not in text.upper():
        text = raw[:3200].decode("ascii", errors="replace")
    if "IN DEPTH" in text.upper():
        raise ValueError(f"{path.name} is a depth section; use the corresponding *_time.segy file.")
    values, info = _read_selected_segy(path, trace_stride=1, depth_stride=1)
    dt = info["binary_dt_us"] * 1e-6
    if dt <= 0:
        raise ValueError("The time section needs a positive SEG-Y sample interval.")
    record_bytes = info["record_bytes"]
    records = np.memmap(path, mode="r", offset=3600, dtype=np.uint8)
    x = np.ndarray((values.shape[0],), dtype=">i4", buffer=records,
                   offset=72, strides=(record_bytes,)).astype(float)
    scalars = np.ndarray((values.shape[0],), dtype=">i2", buffer=records,
                         offset=70, strides=(record_bytes,)).astype(float)
    delay_ms = struct.unpack_from(">h", records, 108)[0]
    del records
    if "MULTIPLIED BY 1000" in text.upper():
        x /= 1000.0
    else:
        x *= np.where(scalars > 0, scalars, np.where(scalars < 0, 1 / np.maximum(abs(scalars), 1), 1))
    if np.any(np.diff(x) <= 0):
        raise ValueError("Processed section coordinates must increase along the profile.")
    times = delay_ms * 0.001 + np.arange(values.shape[1]) * dt
    info.update(sample_domain="time", dt_s=dt, x_range_m=[float(x[0]), float(x[-1])],
                dx_m=float(np.median(np.diff(x))), time_range_s=[float(times[0]), float(times[-1])],
                text_header=[text[i:i + 80].strip() for i in range(0, 1120, 80)])
    return values, x, times, info


def resample_processed_seismic(values, source_x, source_t, target_x, target_t):
    """Interpolate the observed section onto the model's common supported grid."""
    target_x, target_t = np.asarray(target_x), np.asarray(target_t)
    mesh_x, mesh_t = np.meshgrid(target_x, target_t, indexing="ij")
    points = np.column_stack((mesh_x.ravel(), mesh_t.ravel()))
    interpolator = RegularGridInterpolator((source_x, source_t), values, bounds_error=True)
    return interpolator(points).reshape(mesh_x.shape).astype(np.float32)


def estimate_training_wavelet(log_ai, observed, times, training_indices, *, duration_s=0.16,
                             ridge_fraction=0.01, fit_start_s=0.65):
    """Fit one finite stationary wavelet using only the training pseudo-wells.

    The returned phase includes the common well/seismic timing offset. This is
    an effective convolutional approximation to the processed elastic data,
    rather than a claim to recover the original source signature.
    """
    times = np.asarray(times, dtype=float)
    dt = float(times[1] - times[0])
    half = max(2, round(duration_s / (2 * dt)))
    wavelet_times = np.arange(-half, half + 1) * dt
    count = wavelet_times.size
    n = times.size
    indices = np.arange(n)[:, None] + half - 1 - np.arange(count)[None, :]
    support = (indices >= 0) & (indices < n - 1)
    fit = (times >= max(fit_start_s, times[0] + half * dt)) & (times <= times[-1] - half * dt)
    gram, rhs = np.zeros((count, count)), np.zeros(count)
    matrices, targets = [], []
    for trace in training_indices:
        reflection = np.tanh(0.5 * np.diff(np.asarray(log_ai[trace], dtype=float)))
        design = np.where(support, reflection[np.clip(indices, 0, n - 2)], 0.0)
        target = np.asarray(observed[trace], dtype=float)
        target = target - np.mean(target[fit])
        rms = float(np.sqrt(np.mean(target[fit] ** 2)))
        if rms <= 0:
            raise ValueError(f"Training pseudo-well {trace} has no seismic signal in the fit window.")
        target /= rms
        gram += design[fit].T @ design[fit]
        rhs += design[fit].T @ target[fit]
        matrices.append(design)
        targets.append(target)
    # Ridge, a gentle second-difference penalty, and zero DC stabilise a short
    # wavelet without changing the impedance labels or seismic amplitudes.
    penalty = ridge_fraction * float(np.trace(gram)) / count
    difference = np.diff(np.eye(count), n=2, axis=0)
    system = gram + penalty * (np.eye(count) + difference.T @ difference)
    constraint = np.ones(count)
    augmented = np.block([[system, constraint[:, None]], [constraint[None, :], np.zeros((1, 1))]])
    fitted = np.linalg.solve(augmented, np.r_[rhs, 0])[:-1]
    wavelet, amplitude_scale = wavelet_l2_normalize(fitted)
    correlations = []
    for matrix, target in zip(matrices, targets):
        prediction = matrix @ wavelet
        correlations.append(float(np.corrcoef(prediction[fit], target[fit])[0, 1]))
    spectrum = wavelet_spectrum_features(wavelet_times, wavelet)
    return wavelet_times, wavelet, {
        "method": "joint_training_well_ridge", "training_profile_indices": list(map(int, training_indices)),
        "duration_s": float(2 * half * dt), "ridge_fraction": ridge_fraction,
        "fit_time_range_s": [float(times[fit][0]), float(times[fit][-1])],
        "training_well_correlations": correlations, "l2_normalized": True,
        "fitted_amplitude_scale": amplitude_scale, "spectrum": spectrum.to_row(),
        "interpretation": "effective stationary wavelet; only training-well AI used; no time warping",
    }


def training_well_correlations(log_ai, observed, times, wavelet_times, wavelet, indices, start_s=0.65, stop_s=None):
    predicted = forward_time(np.asarray(log_ai)[indices], wavelet_times, wavelet,
                             sample_step_s=float(times[1] - times[0]))
    mask = np.asarray(times) >= start_s
    if stop_s is not None:
        mask &= np.asarray(times) <= stop_s
    return [float(np.corrcoef(p[mask], s[mask])[0, 1]) for p, s in zip(predicted, np.asarray(observed)[indices])]
