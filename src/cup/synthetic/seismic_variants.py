"""Seismic mismatch variants for synthoseis-lite generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.signal import hilbert

from cup.synthetic.random import ar1_irregular, named_rng
from cup.synthetic.stats import centered_rms


@dataclass(frozen=True)
class SeismicVariantResult:
    variant_id: str
    mismatch_family: str
    seismic_observed: np.ndarray
    positive_gain: np.ndarray
    additive_noise: np.ndarray
    qc: dict[str, Any]


def _rng(
    *,
    global_seed: int,
    generator_family: str,
    realization_id: str,
    variant_id: str,
    coefficient_name: str,
) -> np.random.Generator:
    return named_rng(
        global_seed=global_seed,
        benchmark_version="synthoseis_lite_v1",
        generator_family=generator_family,
        stream_purpose="seismic_mismatch",
        realization_id=realization_id,
        coefficient_name=coefficient_name,
        variant_id=variant_id,
    )


def _normalize_to_rms(values: np.ndarray, mask: np.ndarray, rms: float) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return np.zeros_like(output)
    output[~valid] = 0.0
    output[valid] -= float(np.mean(output[valid]))
    current = float(np.sqrt(np.mean(output[valid] * output[valid])))
    if current <= 0.0 or not np.isfinite(current):
        return np.zeros_like(output)
    return output * (float(rms) / current)


def _target_noise_rms(seismic: np.ndarray, mask: np.ndarray, config: Mapping[str, Any], fraction_key: str) -> float:
    signal = centered_rms(seismic, mask)
    floor = float(config["noise"]["absolute_noise_rms_floor"])
    fraction = float(config["noise"][fraction_key])
    if np.isfinite(signal) and signal > 0.0:
        return max(floor, fraction * signal)
    return floor


def _white_noise(shape: tuple[int, int], *, rng: np.random.Generator, mask: np.ndarray, rms: float) -> np.ndarray:
    raw = rng.normal(size=shape)
    return _normalize_to_rms(raw, mask, rms)


def _time_ar1_noise(
    shape: tuple[int, int],
    *,
    rng: np.random.Generator,
    mask: np.ndarray,
    correlation_samples: float,
    rms: float,
) -> np.ndarray:
    n_lateral, n_time = shape
    rho = float(np.exp(-1.0 / max(float(correlation_samples), np.finfo(np.float64).eps)))
    raw = np.empty(shape, dtype=np.float64)
    raw[:, 0] = rng.normal(size=n_lateral)
    scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    for index in range(1, n_time):
        raw[:, index] = rho * raw[:, index - 1] + scale * rng.normal(size=n_lateral)
    return _normalize_to_rms(raw, mask, rms)


def _phase_rotate(seismic: np.ndarray, degrees: float) -> np.ndarray:
    analytic = hilbert(np.asarray(seismic, dtype=np.float64), axis=-1)
    phase = np.deg2rad(float(degrees))
    return np.real(analytic * np.exp(1j * phase))


def _fractional_shift(seismic: np.ndarray, samples: float) -> np.ndarray:
    values = np.asarray(seismic, dtype=np.float64)
    grid = np.arange(values.shape[-1], dtype=np.float64)
    shifted = np.empty_like(values)
    for lateral_index in range(values.shape[0]):
        shifted[lateral_index] = np.interp(
            grid - float(samples),
            grid,
            values[lateral_index],
            left=values[lateral_index, 0],
            right=values[lateral_index, -1],
        )
    return shifted


def _tracewise_gain(
    *,
    lateral_m: np.ndarray,
    shape: tuple[int, int],
    log_sigma: float,
    correlation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    lateral = np.asarray(lateral_m, dtype=np.float64).reshape(-1)
    if lateral.size < 2 or float(log_sigma) <= 0.0:
        gain = np.ones(shape, dtype=np.float64)
        return gain, {
            "requested_correlation_length_m": float("nan"),
            "effective_correlation_length_m": float("nan"),
            "empirical_correlation_length_m": float("nan"),
        }
    requested = float(correlation_fraction) * max(float(lateral[-1] - lateral[0]), np.finfo(np.float64).eps)
    spacing = float(np.median(np.diff(lateral)))
    effective = max(requested, 4.0 * spacing)
    field, field_qc = ar1_irregular(lateral, correlation_length_m=effective, rng=rng)
    gain = np.exp(float(log_sigma) * field)[:, None] * np.ones(shape, dtype=np.float64)
    return gain, {
        "requested_correlation_length_m": requested,
        "effective_correlation_length_m": effective,
        "empirical_correlation_length_m": float(field_qc.get("empirical_correlation_length_m", float("nan"))),
    }


def _regular_ar1(size: int, *, rng: np.random.Generator, correlation_fraction: float) -> np.ndarray:
    if size < 2:
        return np.zeros(size, dtype=np.float64)
    corr = max(float(correlation_fraction) * size, 1.0)
    rho = float(np.exp(-1.0 / corr))
    raw = np.empty(size, dtype=np.float64)
    raw[0] = rng.normal()
    scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    for index in range(1, size):
        raw[index] = rho * raw[index - 1] + scale * rng.normal()
    clipped = np.clip(raw, -3.0, 3.0)
    centered = clipped - float(np.mean(clipped))
    rms = float(np.sqrt(np.mean(centered * centered)))
    return centered / rms if rms > 0.0 else np.zeros_like(centered)


def _time_lateral_gain(
    *,
    lateral_m: np.ndarray,
    shape: tuple[int, int],
    log_sigma: float,
    lateral_correlation_fraction: float,
    time_correlation_fraction: float,
    rng_lateral: np.random.Generator,
    rng_time: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    trace_gain, lateral_qc = _tracewise_gain(
        lateral_m=lateral_m,
        shape=shape,
        log_sigma=1.0,
        correlation_fraction=lateral_correlation_fraction,
        rng=rng_lateral,
    )
    lateral_field = np.log(trace_gain[:, 0])
    time_field = _regular_ar1(shape[1], rng=rng_time, correlation_fraction=time_correlation_fraction)
    raw = lateral_field[:, None] + time_field[None, :]
    raw -= float(np.mean(raw))
    raw_rms = float(np.sqrt(np.mean(raw * raw)))
    if raw_rms > 0.0:
        raw /= raw_rms
    return np.exp(float(log_sigma) * raw), lateral_qc


def _build_result(
    *,
    variant_id: str,
    mismatch_family: str,
    seismic_convolved: np.ndarray,
    mask: np.ndarray,
    gain: np.ndarray,
    noise: np.ndarray,
    extra_qc: Mapping[str, Any],
) -> SeismicVariantResult:
    observed = gain * seismic_convolved + noise
    valid = np.asarray(mask, dtype=bool) & np.isfinite(observed)
    if not np.any(valid):
        raise ValueError("invalid_seismic_variant:no_valid_samples")
    gain_valid = gain[valid]
    noise_valid = noise[valid]
    qc = {
        "seismic_variant_status": "ok",
        "seismic_variant_id": variant_id,
        "seismic_mismatch_family": mismatch_family,
        "seismic_observed_rms": centered_rms(observed, valid),
        "seismic_convolved_rms": centered_rms(seismic_convolved, valid),
        "additive_noise_rms": float(np.sqrt(np.mean(noise_valid * noise_valid))),
        "positive_gain_min": float(np.min(gain_valid)),
        "positive_gain_max": float(np.max(gain_valid)),
        "positive_gain_mean": float(np.mean(gain_valid)),
        **dict(extra_qc),
    }
    return SeismicVariantResult(
        variant_id=variant_id,
        mismatch_family=mismatch_family,
        seismic_observed=observed,
        positive_gain=gain,
        additive_noise=noise,
        qc=qc,
    )


def generate_seismic_variants(
    *,
    seismic_model_consistent: np.ndarray,
    forward_valid_mask: np.ndarray,
    lateral_m: np.ndarray,
    config: Mapping[str, Any],
    global_seed: int,
    generator_family: str,
    realization_id: str,
    source_variant_id: str = "",
) -> list[SeismicVariantResult]:
    """Generate a finite suite of named observed-seismic mismatch variants."""
    if not bool(config.get("enabled", True)):
        return []
    seismic = np.asarray(seismic_model_consistent, dtype=np.float64)
    mask = np.asarray(forward_valid_mask, dtype=bool) & np.isfinite(seismic)
    shape = seismic.shape
    ones = np.ones(shape, dtype=np.float64)
    zeros = np.zeros(shape, dtype=np.float64)
    results: list[SeismicVariantResult] = []

    white_id = "white_noise"
    white_rng = _rng(
        global_seed=global_seed,
        generator_family=generator_family,
        realization_id=realization_id,
        variant_id=f"{source_variant_id}:{white_id}",
        coefficient_name="white_noise",
    )
    white_rms = _target_noise_rms(seismic, mask, config, "white_noise_rms_fraction")
    results.append(
        _build_result(
            variant_id=white_id,
            mismatch_family="white_noise",
            seismic_convolved=seismic,
            mask=mask,
            gain=ones,
            noise=_white_noise(shape, rng=white_rng, mask=mask, rms=white_rms),
            extra_qc={"requested_noise_rms": white_rms},
        )
    )

    colored_id = "colored_noise"
    colored_rng = _rng(
        global_seed=global_seed,
        generator_family=generator_family,
        realization_id=realization_id,
        variant_id=f"{source_variant_id}:{colored_id}",
        coefficient_name="colored_noise",
    )
    colored_rms = _target_noise_rms(seismic, mask, config, "colored_noise_rms_fraction")
    results.append(
        _build_result(
            variant_id=colored_id,
            mismatch_family="colored_noise",
            seismic_convolved=seismic,
            mask=mask,
            gain=ones,
            noise=_time_ar1_noise(
                shape,
                rng=colored_rng,
                mask=mask,
                correlation_samples=float(config["noise"]["colored_time_correlation_samples"]),
                rms=colored_rms,
            ),
            extra_qc={
                "requested_noise_rms": colored_rms,
                "colored_time_correlation_samples": float(config["noise"]["colored_time_correlation_samples"]),
            },
        )
    )

    gain_cfg = config["gain"]
    global_id = "global_scalar_gain"
    global_rng = _rng(
        global_seed=global_seed,
        generator_family=generator_family,
        realization_id=realization_id,
        variant_id=f"{source_variant_id}:{global_id}",
        coefficient_name="global_scalar_gain",
    )
    scalar_gain = float(np.exp(float(gain_cfg["global_log_sigma"]) * global_rng.normal()))
    results.append(
        _build_result(
            variant_id=global_id,
            mismatch_family="global_scalar_gain",
            seismic_convolved=seismic,
            mask=mask,
            gain=np.full(shape, scalar_gain, dtype=np.float64),
            noise=zeros,
            extra_qc={"global_scalar_gain": scalar_gain},
        )
    )

    trace_id = "tracewise_lateral_smooth_gain"
    trace_gain, trace_qc = _tracewise_gain(
        lateral_m=lateral_m,
        shape=shape,
        log_sigma=float(gain_cfg["tracewise_log_sigma"]),
        correlation_fraction=float(gain_cfg["lateral_correlation_fraction"]),
        rng=_rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
            variant_id=f"{source_variant_id}:{trace_id}",
            coefficient_name="tracewise_lateral_smooth_gain",
        ),
    )
    results.append(
        _build_result(
            variant_id=trace_id,
            mismatch_family="tracewise_lateral_smooth_gain",
            seismic_convolved=seismic,
            mask=mask,
            gain=trace_gain,
            noise=zeros,
            extra_qc={f"gain_{key}": value for key, value in trace_qc.items()},
        )
    )

    tl_id = "time_lateral_smooth_gain"
    tl_gain, tl_qc = _time_lateral_gain(
        lateral_m=lateral_m,
        shape=shape,
        log_sigma=float(gain_cfg["time_lateral_log_sigma"]),
        lateral_correlation_fraction=float(gain_cfg["lateral_correlation_fraction"]),
        time_correlation_fraction=float(gain_cfg["time_correlation_fraction"]),
        rng_lateral=_rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
            variant_id=f"{source_variant_id}:{tl_id}",
            coefficient_name="time_lateral_smooth_gain:lateral",
        ),
        rng_time=_rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
            variant_id=f"{source_variant_id}:{tl_id}",
            coefficient_name="time_lateral_smooth_gain:time",
        ),
    )
    results.append(
        _build_result(
            variant_id=tl_id,
            mismatch_family="time_lateral_smooth_gain",
            seismic_convolved=seismic,
            mask=mask,
            gain=tl_gain,
            noise=zeros,
            extra_qc={f"gain_{key}": value for key, value in tl_qc.items()},
        )
    )

    for phase in config["wavelet"]["phase_rotation_degrees"]:
        value = float(phase)
        variant_id = f"phase_rotation_{value:+g}deg".replace("+", "p").replace("-", "m")
        results.append(
            _build_result(
                variant_id=variant_id,
                mismatch_family="constant_phase_rotation",
                seismic_convolved=_phase_rotate(seismic, value),
                mask=mask,
                gain=ones,
                noise=zeros,
                extra_qc={"phase_rotation_degrees": value},
            )
        )
    for shift in config["wavelet"]["time_shift_samples"]:
        value = float(shift)
        variant_id = f"fractional_time_shift_{value:+g}samples".replace("+", "p").replace("-", "m").replace(".", "p")
        results.append(
            _build_result(
                variant_id=variant_id,
                mismatch_family="fractional_time_shift",
                seismic_convolved=_fractional_shift(seismic, value),
                mask=mask,
                gain=ones,
                noise=zeros,
                extra_qc={"time_shift_samples": value},
            )
        )
    combined_cfg = config["combined"]
    if bool(combined_cfg["enabled"]):
        combined_id = "combined_moderate"
        shifted = _fractional_shift(
            _phase_rotate(seismic, float(combined_cfg["phase_rotation_degrees"])),
            float(combined_cfg["time_shift_samples"]),
        )
        combined_gain, combined_gain_qc = _tracewise_gain(
            lateral_m=lateral_m,
            shape=shape,
            log_sigma=float(combined_cfg["gain_log_sigma"]),
            correlation_fraction=float(gain_cfg["lateral_correlation_fraction"]),
            rng=_rng(
                global_seed=global_seed,
                generator_family=generator_family,
                realization_id=realization_id,
                variant_id=f"{source_variant_id}:{combined_id}",
                coefficient_name="combined_gain",
            ),
        )
        combined_rms = _target_noise_rms(
            shifted,
            mask,
            {"noise": {**config["noise"], "combined_noise_rms_fraction": combined_cfg["noise_rms_fraction"]}},
            "combined_noise_rms_fraction",
        )
        combined_noise = _time_ar1_noise(
            shape,
            rng=_rng(
                global_seed=global_seed,
                generator_family=generator_family,
                realization_id=realization_id,
                variant_id=f"{source_variant_id}:{combined_id}",
                coefficient_name="combined_noise",
            ),
            mask=mask,
            correlation_samples=float(config["noise"]["colored_time_correlation_samples"]),
            rms=combined_rms,
        )
        results.append(
            _build_result(
                variant_id=combined_id,
                mismatch_family="combined_phase_shift_gain_noise",
                seismic_convolved=shifted,
                mask=mask,
                gain=combined_gain,
                noise=combined_noise,
                extra_qc={
                    "phase_rotation_degrees": float(combined_cfg["phase_rotation_degrees"]),
                    "time_shift_samples": float(combined_cfg["time_shift_samples"]),
                    "requested_noise_rms": combined_rms,
                    **{f"gain_{key}": value for key, value in combined_gain_qc.items()},
                },
            )
        )
    return results
