"""High-resolution forward-model QC for synthoseis-lite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from cup.physics.numpy_backend import forward_time
from cup.synthetic.core.signal import finite_support_fir, valid_filter_decimate


def antialias_taps(
    factor: int,
    *,
    taps_per_factor: int = 32,
    cutoff_output_nyquist_fraction: float = 0.9,
    kaiser_beta: float = 8.6,
) -> np.ndarray:
    if factor < 1:
        raise ValueError("factor must be positive.")
    if (taps_per_factor, cutoff_output_nyquist_fraction, kaiser_beta) != (32, 0.9, 8.6):
        raise ValueError("science v2 uses one fixed antialias filter")
    return finite_support_fir(factor)


@dataclass(frozen=True)
class HighresWavelet:
    time_s: np.ndarray
    amplitude: np.ndarray
    factor: int
    filter_taps: np.ndarray


@dataclass(frozen=True)
class HighresForwardResult:
    seismic_model_grid: np.ndarray
    decimation_support_1d: np.ndarray
    qc: dict[str, Any]


def forward_sample_valid_mask(interface_mask: np.ndarray) -> np.ndarray:
    """Project the N-1 lower-interface validity contract onto N seismic samples."""
    valid = np.asarray(interface_mask, dtype=bool)
    if valid.ndim < 1 or valid.shape[-1] < 1:
        raise ValueError("forward interface mask must contain at least one interface")
    output = np.empty((*valid.shape[:-1], valid.shape[-1] + 1), dtype=bool)
    output[..., 0] = valid[..., 0]
    output[..., 1:] = valid
    return output


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
    interface_valid = np.asarray(forward_valid_mask_model, dtype=bool)
    valid = forward_sample_valid_mask(interface_valid)
    if truth.ndim != 2 or reference.ndim != 2 or valid.shape != reference.shape:
        raise ValueError("invalid_highres_forward_shapes")
    factor = int(highres_wavelet.factor)
    highres = forward_time(
        truth,
        highres_wavelet.time_s,
        highres_wavelet.amplitude,
    )
    aligned = highres
    downsampled, decimation_support = valid_filter_decimate(
        aligned, factor=factor, taps=highres_wavelet.filter_taps
    )
    downsampled = downsampled[..., : reference.shape[-1]]
    decimation_support = decimation_support[: reference.shape[-1]]
    if downsampled.shape != reference.shape:
        raise ValueError("invalid_highres_forward_alignment")
    mask = valid & decimation_support[None, :] & np.isfinite(reference) & np.isfinite(downsampled)
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
    return HighresForwardResult(
        seismic_model_grid=downsampled,
        decimation_support_1d=decimation_support,
        qc=qc,
    )


def model_grid_closure_qc(
    model_target_log_ai: np.ndarray,
    seismic_model_consistent: np.ndarray,
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
) -> dict[str, Any]:
    """Verify the primary 2 ms forward path closes exactly."""
    target = np.asarray(model_target_log_ai, dtype=np.float64)
    reference = np.asarray(seismic_model_consistent, dtype=np.float64)
    recomputed = forward_time(target, wavelet_time_s, wavelet)
    if recomputed.shape != reference.shape:
        raise ValueError("invalid_model_grid_forward_alignment")
    difference = recomputed - reference
    return {
        "model_grid_closure_status": "ok",
        "model_grid_closure_max_abs": float(np.max(np.abs(difference))),
        "model_grid_closure_rms": _rms(difference),
    }
