"""Low-frequency priors and residual targets for synthoseis-lite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.signal import firwin

from cup.synthetic.random import ar1_irregular, named_rng
from cup.utils.statistics import centered_rms


@dataclass(frozen=True)
class LfmResult:
    lfm_ideal: np.ndarray
    lfm_controlled_degraded: np.ndarray
    residual_vs_lfm_ideal: np.ndarray
    residual_vs_lfm_controlled_degraded: np.ndarray
    qc: dict[str, Any]


def _rng(
    *,
    global_seed: int,
    generator_family: str,
    realization_id: str,
    stream_purpose: str,
    coefficient_name: str = "",
    variant_id: str = "",
) -> np.random.Generator:
    return named_rng(
        global_seed=global_seed,
        benchmark_version="synthoseis_lite_v3",
        generator_family=generator_family,
        stream_purpose=stream_purpose,
        realization_id=realization_id,
        coefficient_name=coefficient_name,
        variant_id=variant_id,
    )


def _fill_trace(trace: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(values)
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(values)
    if np.count_nonzero(mask) == 1:
        return np.full_like(values, float(values[mask][0]))
    index = np.arange(values.size, dtype=np.float64)
    return np.interp(index, index[mask], values[mask])


def lowpass_model_grid(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    dt_s: float,
    cutoff_hz: float,
    numtaps: int,
    kaiser_beta: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a deterministic zero-phase FIR lowpass along model-grid TWT."""
    data = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(data)
    if data.ndim != 2:
        raise ValueError("lowpass_model_grid expects a [lateral, twt] array.")
    nyquist = 0.5 / float(dt_s)
    cutoff = float(cutoff_hz)
    if not np.isfinite(cutoff) or cutoff <= 0.0 or cutoff >= nyquist:
        raise ValueError("lfm cutoff_hz must be finite and within (0, Nyquist).")
    taps_count = int(numtaps)
    if taps_count < 3:
        raise ValueError("lfm numtaps must be >= 3.")
    if taps_count % 2 == 0:
        taps_count += 1
    taps = firwin(
        taps_count,
        cutoff / nyquist,
        window=("kaiser", float(kaiser_beta)),
        scale=True,
    ).astype(np.float64)
    pad = taps_count // 2
    output = np.full(data.shape, np.nan, dtype=np.float64)
    for lateral_index in range(data.shape[0]):
        filled = _fill_trace(data[lateral_index], valid[lateral_index])
        padded = np.pad(filled, pad_width=pad, mode="edge")
        filtered = np.convolve(padded, taps, mode="valid")
        output[lateral_index, valid[lateral_index]] = filtered[valid[lateral_index]]
    return output, {
        "cutoff_hz": cutoff,
        "nyquist_hz": nyquist,
        "numtaps": taps_count,
        "kaiser_beta": float(kaiser_beta),
        "filter_family": "zero_phase_fir_kaiser_edge_padded",
    }


def _normalized_twt(twt_s: np.ndarray) -> np.ndarray:
    twt = np.asarray(twt_s, dtype=np.float64).reshape(-1)
    span = float(twt[-1] - twt[0]) if twt.size > 1 else 0.0
    if span <= 0.0:
        return np.zeros_like(twt)
    return 2.0 * (twt - float(twt[0])) / span - 1.0


def _windowed_bump(size: int, *, center: float, width: float) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=np.float64)
    x = np.linspace(0.0, 1.0, size, dtype=np.float64)
    half = max(float(width), np.finfo(np.float64).eps) * 0.5
    distance = np.abs(x - float(center)) / half
    window = np.zeros_like(x)
    inside = distance < 1.0
    window[inside] = 0.5 * (1.0 + np.cos(np.pi * distance[inside]))
    return window


def _add_valid(target: np.ndarray, values: np.ndarray, valid: np.ndarray) -> None:
    target[valid] = target[valid] + values[valid]


def derive_lfm_priors(
    section: Any,
    *,
    config: Mapping[str, Any],
    global_seed: int,
    generator_family: str,
    model_target_log_ai: np.ndarray | None = None,
    degradation_variant_id: str | None = None,
) -> LfmResult:
    """Derive ideal and controlled-degraded LFM arrays from one model-grid target."""
    enabled = bool(config.get("enabled", True))
    target = np.asarray(
        section.model_target_log_ai if model_target_log_ai is None else model_target_log_ai,
        dtype=np.float64,
    )
    valid = np.asarray(section.valid_mask_model, dtype=bool) & np.isfinite(target)
    dt_s = float(section.twt_model_s[1] - section.twt_model_s[0])
    if not enabled:
        nan = np.full_like(target, np.nan, dtype=np.float64)
        return LfmResult(
            lfm_ideal=nan,
            lfm_controlled_degraded=nan,
            residual_vs_lfm_ideal=nan,
            residual_vs_lfm_controlled_degraded=nan,
            qc={"lfm_status": "disabled"},
        )

    ideal_config = dict(config.get("ideal") or {})
    degraded_config = dict(config.get("controlled_degraded") or {})
    ideal, ideal_qc = lowpass_model_grid(
        target,
        valid_mask=valid,
        dt_s=dt_s,
        cutoff_hz=float(ideal_config.get("cutoff_hz", 10.0)),
        numtaps=int(ideal_config.get("numtaps", 129)),
        kaiser_beta=float(ideal_config.get("kaiser_beta", 8.6)),
    )
    degraded = ideal.copy()
    variant_id = (
        str(degradation_variant_id)
        if degradation_variant_id is not None
        else str(section.realization_id)
    )
    realization_id = str(section.realization_id)
    components: dict[str, float] = {}

    constant_sigma = float(degraded_config.get("constant_bias_sigma_log_ai", 0.02))
    constant_rng = _rng(
        global_seed=global_seed,
        generator_family=generator_family,
        realization_id=realization_id,
        stream_purpose="lfm_degradation",
        coefficient_name="constant_log_ai_bias",
        variant_id=variant_id,
    )
    constant_bias = constant_sigma * float(constant_rng.normal())
    _add_valid(degraded, np.full_like(degraded, constant_bias), valid)
    components["constant_log_ai_bias"] = constant_bias

    trend_sigma = float(degraded_config.get("linear_twt_trend_sigma_log_ai", 0.02))
    trend_rng = _rng(
        global_seed=global_seed,
        generator_family=generator_family,
        realization_id=realization_id,
        stream_purpose="lfm_degradation",
        coefficient_name="linear_twt_trend_bias",
        variant_id=variant_id,
    )
    trend = _normalized_twt(section.twt_model_s)[None, :]
    trend_bias = trend_sigma * float(trend_rng.normal()) * np.broadcast_to(
        trend,
        target.shape,
    )
    _add_valid(degraded, trend_bias, valid)
    components["linear_twt_trend_bias_rms"] = centered_rms(trend_bias, valid)

    zone_sigma = float(degraded_config.get("zonewise_bias_sigma_log_ai", 0.03))
    zone_bias = np.zeros_like(target)
    zone_model = np.asarray(section.zone_id_model)
    for zone_value in sorted(int(value) for value in np.unique(zone_model[zone_model >= 0])):
        zone_rng = _rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
            stream_purpose="lfm_degradation",
            coefficient_name=f"zonewise_bias:{zone_value}",
            variant_id=variant_id,
        )
        zone_bias[zone_model == zone_value] = zone_sigma * float(zone_rng.normal())
    _add_valid(degraded, zone_bias, valid)
    components["zonewise_bias_rms"] = centered_rms(zone_bias, valid)

    lateral_sigma = float(degraded_config.get("lateral_smooth_bias_sigma_log_ai", 0.02))
    lateral = np.asarray(section.lateral_m, dtype=np.float64)
    if lateral.size >= 2 and lateral_sigma > 0.0:
        requested_lx = float(degraded_config.get("lateral_correlation_fraction", 0.3)) * max(
            float(lateral[-1] - lateral[0]),
            np.finfo(np.float64).eps,
        )
        spacing = float(np.median(np.diff(lateral)))
        effective_lx = max(requested_lx, 4.0 * spacing)
        lateral_rng = _rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
            stream_purpose="lfm_degradation",
            coefficient_name="lateral_smooth_bias_field",
            variant_id=variant_id,
        )
        field, field_qc = ar1_irregular(
            lateral,
            correlation_length_m=effective_lx,
            rng=lateral_rng,
        )
        lateral_bias = lateral_sigma * field[:, None] * np.ones_like(target)
    else:
        requested_lx = float("nan")
        effective_lx = float("nan")
        field_qc = {}
        lateral_bias = np.zeros_like(target)
    _add_valid(degraded, lateral_bias, valid)
    components["lateral_smooth_bias_rms"] = centered_rms(lateral_bias, valid)

    scale_sigma = float(degraded_config.get("amplitude_scale_sigma", 0.05))
    scale_rng = _rng(
        global_seed=global_seed,
        generator_family=generator_family,
        realization_id=realization_id,
        stream_purpose="lfm_degradation",
        coefficient_name="amplitude_scale_bias",
        variant_id=variant_id,
    )
    amplitude_scale = float(np.exp(scale_sigma * float(scale_rng.normal())))
    for lateral_index in range(degraded.shape[0]):
        row_valid = valid[lateral_index] & np.isfinite(degraded[lateral_index])
        if not np.any(row_valid):
            continue
        mean = float(np.mean(degraded[lateral_index, row_valid]))
        degraded[lateral_index, row_valid] = (
            mean + amplitude_scale * (degraded[lateral_index, row_valid] - mean)
        )

    smoothing = dict(degraded_config.get("over_smoothing") or {})
    over_cutoff = float(smoothing.get("cutoff_hz", 6.0))
    blend = float(smoothing.get("blend", 1.0))
    over_smoothed, over_qc = lowpass_model_grid(
        degraded,
        valid_mask=valid,
        dt_s=dt_s,
        cutoff_hz=over_cutoff,
        numtaps=int(smoothing.get("numtaps", ideal_qc["numtaps"])),
        kaiser_beta=float(smoothing.get("kaiser_beta", ideal_qc["kaiser_beta"])),
    )
    degraded[valid] = (1.0 - blend) * degraded[valid] + blend * over_smoothed[valid]

    missing = dict(degraded_config.get("local_missing_control_bias") or {})
    if bool(missing.get("enabled", True)):
        missing_rng = _rng(
            global_seed=global_seed,
            generator_family=generator_family,
            realization_id=realization_id,
            stream_purpose="lfm_degradation",
            coefficient_name="local_missing_control_bias",
            variant_id=variant_id,
        )
        lateral_center = float(missing_rng.uniform(0.25, 0.75))
        twt_center = float(missing_rng.uniform(0.25, 0.75))
        lateral_window = _windowed_bump(
            target.shape[0],
            center=lateral_center,
            width=float(missing.get("lateral_width_fraction", 0.30)),
        )
        twt_window = _windowed_bump(
            target.shape[1],
            center=twt_center,
            width=float(missing.get("twt_width_fraction", 0.30)),
        )
        amplitude = float(missing.get("max_abs_log_ai", 0.04)) * float(
            np.clip(missing_rng.normal(), -1.0, 1.0)
        )
        missing_bias = amplitude * lateral_window[:, None] * twt_window[None, :]
        _add_valid(degraded, missing_bias, valid)
        components["local_missing_control_bias_peak"] = amplitude
        components["local_missing_control_bias_rms"] = centered_rms(missing_bias, valid)

    degraded[~valid] = np.nan
    ideal[~valid] = np.nan
    residual_ideal = target - ideal
    residual_degraded = target - degraded
    residual_ideal[~valid] = np.nan
    residual_degraded[~valid] = np.nan
    degradation = degraded - ideal
    qc = {
        "lfm_status": "ok",
        "lfm_ideal_cutoff_hz": float(ideal_qc["cutoff_hz"]),
        "lfm_ideal_numtaps": int(ideal_qc["numtaps"]),
        "lfm_ideal_filter_family": ideal_qc["filter_family"],
        "lfm_controlled_degraded_over_smoothing_cutoff_hz": over_cutoff,
        "lfm_controlled_degraded_over_smoothing_blend": blend,
        "lfm_amplitude_scale_bias": amplitude_scale,
        "lfm_lateral_requested_correlation_length_m": requested_lx,
        "lfm_lateral_effective_correlation_length_m": effective_lx,
        "lfm_lateral_empirical_correlation_length_m": float(
            field_qc.get("empirical_correlation_length_m", float("nan"))
        ),
        "lfm_valid_sample_count": int(np.count_nonzero(valid)),
        "lfm_ideal_rms": centered_rms(ideal, valid),
        "lfm_controlled_degraded_rms": centered_rms(degraded, valid),
        "lfm_degradation_rms": centered_rms(degradation, valid),
        "residual_vs_lfm_ideal_rms": centered_rms(residual_ideal, valid),
        "residual_vs_lfm_controlled_degraded_rms": centered_rms(
            residual_degraded,
            valid,
        ),
        **{f"lfm_component_{key}": value for key, value in components.items()},
    }
    return LfmResult(
        lfm_ideal=ideal,
        lfm_controlled_degraded=degraded,
        residual_vs_lfm_ideal=residual_ideal,
        residual_vs_lfm_controlled_degraded=residual_degraded,
        qc=qc,
    )
