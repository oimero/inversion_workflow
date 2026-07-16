"""Domain-neutral seismic-variant runner for Synthoseis science v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from cup.synthetic.schemas import (
    BENCHMARK_SCHEMA_VERSION,
    RANDOM_STREAM_CONTRACT_VERSION,
    SCIENCE_REVISION,
)
from cup.synthetic.core.random import ar1_irregular, named_rng
from cup.utils.statistics import centered_rms


@dataclass(frozen=True)
class SeismicVariantResult:
    variant_id: str
    mismatch_family: str
    operator_source: str
    seismic_observed: np.ndarray
    positive_gain: np.ndarray
    additive_noise: np.ndarray
    parameters: dict[str, Any]
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
        benchmark_version=BENCHMARK_SCHEMA_VERSION,
        science_revision=SCIENCE_REVISION,
        random_stream_contract_version=RANDOM_STREAM_CONTRACT_VERSION,
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
    fraction = float(config["noise"][fraction_key])
    if not np.isfinite(signal) or signal <= 0.0:
        raise ValueError("invalid_seismic_variant:zero_energy_roi")
    return fraction * signal


def _white_noise(shape: tuple[int, int], *, rng: np.random.Generator, mask: np.ndarray, rms: float) -> np.ndarray:
    raw = rng.normal(size=shape)
    return _normalize_to_rms(raw, mask, rms)


def _axis_ar1_noise(
    shape: tuple[int, int],
    *,
    rng: np.random.Generator,
    mask: np.ndarray,
    sample_axis: np.ndarray,
    correlation_length: float,
    rms: float,
) -> np.ndarray:
    coordinate = np.asarray(sample_axis, dtype=np.float64).reshape(-1)
    if coordinate.size != shape[1] or np.any(~np.isfinite(coordinate)):
        raise ValueError("invalid_seismic_variant:sample_axis")
    increments = np.diff(coordinate)
    if np.any(increments <= 0.0):
        raise ValueError("invalid_seismic_variant:sample_axis_not_increasing")
    length = float(correlation_length)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("invalid_seismic_variant:colored_axis_correlation_length")
    n_lateral, n_axis = shape
    raw = np.empty(shape, dtype=np.float64)
    raw[:, 0] = rng.normal(size=n_lateral)
    for index in range(1, n_axis):
        rho = float(np.exp(-increments[index - 1] / length))
        scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
        raw[:, index] = rho * raw[:, index - 1] + scale * rng.normal(size=n_lateral)
    return _normalize_to_rms(raw, mask, rms)


def _axis_static(
    seismic: np.ndarray, axis: np.ndarray, shift: float, source_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(seismic, dtype=np.float64)
    coordinate = np.asarray(axis, dtype=np.float64)
    output = np.full_like(values, np.nan)
    support = np.zeros_like(values, dtype=bool)
    for trace in range(values.shape[0]):
        valid = np.asarray(source_mask[trace], dtype=bool) & np.isfinite(values[trace])
        if np.count_nonzero(valid) < 2:
            continue
        source_axis = coordinate[valid]
        sample_at = coordinate - float(shift)
        inside = (sample_at >= source_axis[0]) & (sample_at <= source_axis[-1])
        output[trace, inside] = np.interp(sample_at[inside], source_axis, values[trace, valid])
        support[trace, inside] = True
    return output, support


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


def _axis_lateral_gain(
    *,
    lateral_m: np.ndarray,
    shape: tuple[int, int],
    log_sigma: float,
    lateral_correlation_fraction: float,
    axis_correlation_fraction: float,
    rng_lateral: np.random.Generator,
    rng_axis: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    trace_gain, lateral_qc = _tracewise_gain(
        lateral_m=lateral_m,
        shape=shape,
        log_sigma=1.0,
        correlation_fraction=lateral_correlation_fraction,
        rng=rng_lateral,
    )
    lateral_field = np.log(trace_gain[:, 0])
    axis_field = _regular_ar1(
        shape[1], rng=rng_axis, correlation_fraction=axis_correlation_fraction
    )
    raw = lateral_field[:, None] + axis_field[None, :]
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
    operator_source: str = "observed_base",
) -> SeismicVariantResult:
    observed = gain * seismic_convolved + noise
    valid = np.asarray(mask, dtype=bool)
    if np.any(valid & ~np.isfinite(observed)):
        raise ValueError("invalid_seismic_variant:roi_support_not_finite")
    if not np.any(valid):
        raise ValueError("invalid_seismic_variant:no_valid_samples")
    gain_valid = gain[valid]
    noise_valid = noise[valid]
    qc = {
        "seismic_variant_status": "ok",
        "seismic_variant_id": variant_id,
        "seismic_mismatch_family": mismatch_family,
        "seismic_variant_operator_source": operator_source,
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
        operator_source=operator_source,
        seismic_observed=observed,
        positive_gain=gain,
        additive_noise=noise,
        parameters=dict(extra_qc),
        qc=qc,
    )


def generate_seismic_variants(
    *,
    seismic_input: np.ndarray,
    valid_mask: np.ndarray,
    lateral_m: np.ndarray,
    sample_axis: np.ndarray,
    config: Mapping[str, Any],
    global_seed: int,
    generator_family: str,
    realization_id: str,
    perturbed_wavelet_forward: Callable[[float, float], tuple[np.ndarray, np.ndarray]] | None = None,
    axis_static_shifts: tuple[float, ...] = (),
    combined_axis_static_shift: float = 0.0,
    base_operator_support: np.ndarray | None = None,
) -> list[SeismicVariantResult]:
    """Generate a finite suite of named observed-seismic mismatch variants."""
    if not bool(config.get("enabled", True)):
        return []
    seismic = np.asarray(seismic_input, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.shape != seismic.shape:
        raise ValueError("invalid_seismic_variant:valid_mask_shape")
    if np.any(mask & ~np.isfinite(seismic)):
        raise ValueError("invalid_seismic_variant:base_support_not_finite")
    shape = seismic.shape
    operator_support = (
        np.isfinite(seismic)
        if base_operator_support is None
        else np.asarray(base_operator_support, dtype=bool)
    )
    if operator_support.shape != shape or np.any(mask & ~operator_support):
        raise ValueError("invalid_seismic_variant:base_operator_support")
    ones = np.ones(shape, dtype=np.float64)
    zeros = np.zeros(shape, dtype=np.float64)
    results: list[SeismicVariantResult] = []

    white_id = "white_noise"
    white_rng = _rng(
        global_seed=global_seed,
        generator_family=generator_family,
        realization_id=realization_id,
        variant_id=white_id,
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
        variant_id=colored_id,
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
            noise=_axis_ar1_noise(
                shape,
                rng=colored_rng,
                mask=mask,
                sample_axis=sample_axis,
                correlation_length=float(config["noise"]["colored_axis_correlation_length"]),
                rms=colored_rms,
            ),
            extra_qc={
                "requested_noise_rms": colored_rms,
                "colored_axis_correlation_length": float(config["noise"]["colored_axis_correlation_length"]),
            },
        )
    )

    gain_cfg = config["gain"]
    global_id = "global_gain"
    global_rng = _rng(
        global_seed=global_seed,
        generator_family=generator_family,
        realization_id=realization_id,
        variant_id=global_id,
        coefficient_name="global_gain",
    )
    scalar_gain = float(np.exp(float(gain_cfg["global_log_sigma"]) * global_rng.normal()))
    results.append(
        _build_result(
            variant_id=global_id,
            mismatch_family="global_gain",
            seismic_convolved=seismic,
            mask=mask,
            gain=np.full(shape, scalar_gain, dtype=np.float64),
            noise=zeros,
            extra_qc={"global_gain": scalar_gain},
        )
    )

    trace_id = "tracewise_gain"
    trace_gain, trace_qc = _tracewise_gain(
        lateral_m=lateral_m,
        shape=shape,
        log_sigma=float(gain_cfg["tracewise_log_sigma"]),
        correlation_fraction=float(gain_cfg["lateral_correlation_fraction"]),
        rng=_rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
        variant_id=trace_id,
            coefficient_name="tracewise_gain",
        ),
    )
    results.append(
        _build_result(
            variant_id=trace_id,
            mismatch_family="tracewise_gain",
            seismic_convolved=seismic,
            mask=mask,
            gain=trace_gain,
            noise=zeros,
            extra_qc={f"gain_{key}": value for key, value in trace_qc.items()},
        )
    )

    tl_id = "axis_lateral_gain"
    tl_gain, tl_qc = _axis_lateral_gain(
        lateral_m=lateral_m,
        shape=shape,
        log_sigma=float(gain_cfg["axis_lateral_log_sigma"]),
        lateral_correlation_fraction=float(gain_cfg["lateral_correlation_fraction"]),
        axis_correlation_fraction=float(gain_cfg["axis_correlation_fraction"]),
        rng_lateral=_rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
        variant_id=tl_id,
            coefficient_name="axis_lateral_gain:lateral",
        ),
        rng_axis=_rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
        variant_id=tl_id,
            coefficient_name="axis_lateral_gain:axis",
        ),
    )
    results.append(
        _build_result(
            variant_id=tl_id,
            mismatch_family="axis_lateral_gain",
            seismic_convolved=seismic,
            mask=mask,
            gain=tl_gain,
            noise=zeros,
            extra_qc={f"gain_{key}": value for key, value in tl_qc.items()},
        )
    )

    for phase in config["wavelet"]["phase_rotation_degrees"]:
        value = float(phase)
        if perturbed_wavelet_forward is None:
            raise ValueError("wavelet variants require a Forward Adapter")
        perturbed_seismic, perturbed_mask = perturbed_wavelet_forward(value, 0.0)
        if not np.array_equal(perturbed_mask, mask):
            raise ValueError("invalid_seismic_variant:wavelet_mask_differs_from_base")
        variant_id = f"phase_rotation_{value:+g}deg".replace("+", "p").replace("-", "m")
        results.append(
            _build_result(
                variant_id=variant_id,
                mismatch_family="wavelet_phase",
                seismic_convolved=perturbed_seismic,
                mask=mask,
                gain=ones,
                noise=zeros,
                extra_qc={"phase_rotation_degrees": value},
                operator_source="truth_highres_forward",
            )
        )
    for shift in config["wavelet"]["time_shift_s"]:
        value_s = float(shift)
        if perturbed_wavelet_forward is None:
            raise ValueError("wavelet variants require a Forward Adapter")
        perturbed_seismic, perturbed_mask = perturbed_wavelet_forward(0.0, value_s)
        if not np.array_equal(perturbed_mask, mask):
            raise ValueError("invalid_seismic_variant:wavelet_mask_differs_from_base")
        variant_id = f"wavelet_time_shift_{value_s:+g}s".replace("+", "p").replace("-", "m").replace(".", "p")
        results.append(
            _build_result(
                variant_id=variant_id,
                mismatch_family="wavelet_time_shift",
                seismic_convolved=perturbed_seismic,
                mask=mask,
                gain=ones,
                noise=zeros,
                extra_qc={"wavelet_time_shift_s": value_s},
                operator_source="truth_highres_forward",
            )
        )
    if axis_static_shifts:
        for shift in axis_static_shifts:
            shifted, shifted_support = _axis_static(
                seismic, sample_axis, float(shift), operator_support
            )
            if not np.array_equal(mask & shifted_support, mask):
                raise ValueError("invalid_seismic_variant:axis_static_mask_differs_from_base")
            token = f"{float(shift):+g}".replace("+", "p").replace("-", "m").replace(".", "p")
            results.append(_build_result(
                variant_id=f"axis_static_{token}", mismatch_family="axis_static",
                seismic_convolved=shifted, mask=mask, gain=ones, noise=zeros,
                extra_qc={"axis_static": float(shift)},
            ))
    combined_cfg = config["combined"]
    if bool(combined_cfg["enabled"]):
        combined_id = "combined_moderate"
        if perturbed_wavelet_forward is None:
            raise ValueError("combined wavelet variant requires a Forward Adapter")
        shifted, combined_mask = perturbed_wavelet_forward(
            float(combined_cfg["phase_rotation_degrees"]),
            float(combined_cfg["time_shift_s"]),
        )
        if not np.array_equal(combined_mask, mask):
            raise ValueError("invalid_seismic_variant:combined_mask_differs_from_base")
        if float(combined_axis_static_shift) != 0.0:
            shifted, static_support = _axis_static(
                shifted, sample_axis, float(combined_axis_static_shift), np.isfinite(shifted)
            )
            if not np.array_equal(mask & static_support, mask):
                raise ValueError("invalid_seismic_variant:combined_static_mask_differs_from_base")
        combined_gain, combined_gain_qc = _tracewise_gain(
            lateral_m=lateral_m,
            shape=shape,
            log_sigma=float(combined_cfg["gain_log_sigma"]),
            correlation_fraction=float(gain_cfg["lateral_correlation_fraction"]),
            rng=_rng(
                global_seed=global_seed,
                generator_family=generator_family,
                realization_id=realization_id,
        variant_id=combined_id,
                coefficient_name="combined_gain",
            ),
        )
        combined_rms = _target_noise_rms(
            shifted,
            mask,
            {"noise": {**config["noise"], "combined_noise_rms_fraction": combined_cfg["noise_rms_fraction"]}},
            "combined_noise_rms_fraction",
        )
        combined_noise = _axis_ar1_noise(
            shape,
            rng=_rng(
                global_seed=global_seed,
                generator_family=generator_family,
                realization_id=realization_id,
        variant_id=combined_id,
                coefficient_name="combined_noise",
            ),
            mask=mask,
            sample_axis=sample_axis,
            correlation_length=float(config["noise"]["colored_axis_correlation_length"]),
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
                    "wavelet_time_shift_s": float(combined_cfg["time_shift_s"]),
                    "axis_static": float(combined_axis_static_shift),
                    "requested_noise_rms": combined_rms,
                    **{f"gain_{key}": value for key, value in combined_gain_qc.items()},
                },
                operator_source="truth_highres_forward",
            )
        )
    return results
