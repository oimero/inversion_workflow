"""High-resolution forward-model QC for synthoseis-lite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import firwin, resample_poly

from cup.seismic.observability import forward_log_ai


def antialias_taps(
    factor: int,
    *,
    taps_per_factor: int = 32,
    cutoff_output_nyquist_fraction: float = 0.9,
    kaiser_beta: float = 8.6,
) -> np.ndarray:
    if factor < 1:
        raise ValueError("factor must be positive.")
    return firwin(
        taps_per_factor * factor + 1,
        cutoff_output_nyquist_fraction / factor,
        window=("kaiser", kaiser_beta),
        scale=True,
    ).astype(np.float64)


def downsample_continuous(values: np.ndarray, factor: int, taps: np.ndarray) -> np.ndarray:
    return np.asarray(
        resample_poly(values, up=1, down=factor, axis=-1, window=taps, padtype="line"),
        dtype=np.float64,
    )


def categorical_model_grids(
    state_highres: np.ndarray,
    object_highres: np.ndarray,
    zone_highres: np.ndarray,
    boundary_highres: np.ndarray,
    factor: int,
    n_model: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_lateral, n_highres = state_highres.shape
    fractions = np.zeros((n_lateral, n_model, 3), dtype=np.float32)
    dominant = np.full((n_lateral, n_model), -1, dtype=np.int32)
    zone_model = np.full((n_lateral, n_model), -1, dtype=np.int16)
    boundary_fraction = np.zeros((n_lateral, n_model), dtype=np.float32)
    valid = np.zeros((n_lateral, n_model), dtype=bool)
    for model_index in range(n_model):
        center = model_index * factor
        start = max(0, center - factor // 2)
        end = min(n_highres, center + factor // 2 + 1)
        for lateral_index in range(n_lateral):
            states = state_highres[lateral_index, start:end]
            objects = object_highres[lateral_index, start:end]
            zones = zone_highres[lateral_index, start:end]
            finite = states >= 0
            if not np.any(finite):
                continue
            valid[lateral_index, model_index] = True
            for state in range(3):
                fractions[lateral_index, model_index, state] = np.mean(states[finite] == state)
            object_values, object_counts = np.unique(objects[finite], return_counts=True)
            dominant[lateral_index, model_index] = int(object_values[np.argmax(object_counts)])
            zone_values, zone_counts = np.unique(zones[finite], return_counts=True)
            zone_model[lateral_index, model_index] = int(zone_values[np.argmax(zone_counts)])
            boundary_fraction[lateral_index, model_index] = float(
                np.mean(boundary_highres[lateral_index, start:end])
            )
    return fractions, dominant, zone_model, boundary_fraction, valid


@dataclass(frozen=True)
class HighresWavelet:
    time_s: np.ndarray
    amplitude: np.ndarray
    factor: int
    filter_taps: np.ndarray


@dataclass(frozen=True)
class HighresForwardResult:
    seismic_model_grid: np.ndarray
    qc: dict[str, Any]


def resample_wavelet_to_highres(
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
    *,
    factor: int,
) -> HighresWavelet:
    """Polyphase-upsample a centered odd wavelet while preserving duration."""
    time = np.asarray(wavelet_time_s, dtype=np.float64).reshape(-1)
    amplitude = np.asarray(wavelet, dtype=np.float64).reshape(-1)
    if time.shape != amplitude.shape or time.size < 3 or time.size % 2 == 0:
        raise ValueError("invalid_wavelet_for_highres_forward")
    if int(factor) < 1:
        raise ValueError("highres wavelet factor must be positive.")
    factor = int(factor)
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=1e-5, atol=1e-9):
        raise ValueError("highres wavelet source axis must be regular.")
    taps = antialias_taps(factor)
    upsampled = resample_poly(
        amplitude,
        up=factor,
        down=1,
        window=taps,
        padtype="constant",
    )
    expected_size = (amplitude.size - 1) * factor + 1
    upsampled = np.asarray(upsampled[:expected_size], dtype=np.float64)
    norm = float(np.linalg.norm(upsampled))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid_highres_wavelet_energy")
    upsampled /= norm
    high_dt = dt / factor
    high_time = time[0] + np.arange(expected_size, dtype=np.float64) * high_dt
    center = expected_size // 2
    if abs(float(high_time[center])) > 1e-9:
        raise ValueError("invalid_highres_wavelet_center")
    return HighresWavelet(
        time_s=high_time,
        amplitude=upsampled,
        factor=factor,
        filter_taps=taps,
    )


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array))) if array.size else float("nan")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else float("nan")


def _spectral_shape_error(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    test = np.asarray(candidate, dtype=np.float64)
    ref_spectrum = np.mean(np.abs(np.fft.rfft(ref, axis=-1)), axis=0)
    test_spectrum = np.mean(np.abs(np.fft.rfft(test, axis=-1)), axis=0)
    ref_norm = float(np.linalg.norm(ref_spectrum))
    test_norm = float(np.linalg.norm(test_spectrum))
    if ref_norm <= 0.0 or test_norm <= 0.0:
        return float("nan")
    return _rms(ref_spectrum / ref_norm - test_spectrum / test_norm)


def highres_forward_to_model_grid(
    truth_log_ai_highres: np.ndarray,
    seismic_model_consistent: np.ndarray,
    *,
    highres_wavelet: HighresWavelet,
    forward_valid_mask_model: np.ndarray,
) -> HighresForwardResult:
    """Forward high-resolution truth and anti-alias it onto the model axis."""
    truth = np.asarray(truth_log_ai_highres, dtype=np.float64)
    reference = np.asarray(seismic_model_consistent, dtype=np.float64)
    valid = np.asarray(forward_valid_mask_model, dtype=bool)
    if truth.ndim != 2 or reference.ndim != 2 or valid.shape != reference.shape:
        raise ValueError("invalid_highres_forward_shapes")
    factor = int(highres_wavelet.factor)
    highres = np.stack(
        [forward_log_ai(trace, highres_wavelet.amplitude) for trace in truth],
        axis=0,
    )
    # High-resolution reflectivity sample 0 hangs on truth sample 1. The first
    # model-grid reflectivity hangs on truth sample ``factor``.
    aligned = highres[:, factor - 1 :]
    downsampled = downsample_continuous(
        aligned,
        factor,
        highres_wavelet.filter_taps,
    )[..., : reference.shape[-1]]
    if downsampled.shape != reference.shape:
        raise ValueError("invalid_highres_forward_alignment")
    mask = valid & np.isfinite(reference) & np.isfinite(downsampled)
    if np.count_nonzero(mask) < 3:
        raise ValueError("insufficient_highres_forward_qc_samples")
    ref_values = reference[mask]
    test_values = downsampled[mask]
    difference = test_values - ref_values
    reference_rms = _rms(ref_values)
    test_rms = _rms(test_values)
    denominator = float(np.dot(test_values, test_values))
    scale = (
        float(np.dot(ref_values, test_values) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    aligned_difference = scale * test_values - ref_values
    status = (
        "zero_energy_reference"
        if reference_rms <= 1e-15 and test_rms <= 1e-15
        else "ok"
    )
    qc = {
        "highres_forward_status": status,
        "highres_forward_valid_samples": int(np.count_nonzero(mask)),
        "highres_forward_rms": test_rms,
        "model_grid_forward_rms": reference_rms,
        "highres_forward_raw_difference_rms": _rms(difference),
        "highres_forward_raw_nrmse": (
            _rms(difference) / reference_rms if reference_rms > 0.0 else float("nan")
        ),
        "highres_forward_corr": _correlation(ref_values, test_values),
        "highres_forward_amplitude_scale_to_model": scale,
        "highres_forward_scale_aligned_difference_rms": _rms(aligned_difference),
        "highres_forward_scale_aligned_nrmse": (
            _rms(aligned_difference) / reference_rms
            if reference_rms > 0.0
            else float("nan")
        ),
        "highres_forward_spectral_shape_error": _spectral_shape_error(
            reference,
            downsampled,
        ),
        "highres_wavelet_dt_s": float(
            highres_wavelet.time_s[1] - highres_wavelet.time_s[0]
        ),
        "highres_wavelet_n_samples": int(highres_wavelet.amplitude.size),
        "highres_wavelet_l2_energy": float(
            np.linalg.norm(highres_wavelet.amplitude)
        ),
    }
    return HighresForwardResult(seismic_model_grid=downsampled, qc=qc)


def model_grid_closure_qc(
    model_target_log_ai: np.ndarray,
    seismic_model_consistent: np.ndarray,
    wavelet: np.ndarray,
) -> dict[str, Any]:
    """Verify the primary 2 ms forward path closes exactly."""
    target = np.asarray(model_target_log_ai, dtype=np.float64)
    reference = np.asarray(seismic_model_consistent, dtype=np.float64)
    recomputed = np.stack(
        [forward_log_ai(trace, wavelet) for trace in target],
        axis=0,
    )
    if recomputed.shape != reference.shape:
        raise ValueError("invalid_model_grid_forward_alignment")
    difference = recomputed - reference
    return {
        "model_grid_closure_status": "ok",
        "model_grid_closure_max_abs": float(np.max(np.abs(difference))),
        "model_grid_closure_rms": _rms(difference),
    }
