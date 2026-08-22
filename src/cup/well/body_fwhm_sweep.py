"""GINN body/residual boundary sweep on native wells and model-grid forwards."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import fftconvolve

from cup.seismic.target_zone import TargetZone
from cup.well.real_field_control_qc import (
    forward_depth_finite_runs,
    horizon_markers_along_control,
    sample_seismic_along_control,
)
from cup.well.real_field_controls import WellControl, WellControlSet
from cup.well.scale_separation import gaussian_smooth_finite_runs_numpy


SCHEMA_VERSION = "ginn_v2_body_fwhm_sweep_v1"


@dataclass(frozen=True)
class BodyFwhmSweepPolicy:
    """Physical scales and fixed event-selection settings for one sweep."""

    fwhm_values_m: tuple[float, ...]
    reference_fwhm_m: float
    real_event_threshold_fraction: float
    residual_lobe_threshold_fraction: float
    max_real_events_per_well: int
    minimum_event_samples: int
    event_context_min_m: float
    event_context_width_multiple: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BodyFwhmSweepPolicy":
        expected = {
            "fwhm_values_m",
            "reference_fwhm_m",
            "real_event_threshold_fraction",
            "residual_lobe_threshold_fraction",
            "max_real_events_per_well",
            "minimum_event_samples",
            "event_context_min_m",
            "event_context_width_multiple",
        }
        if set(raw) != expected:
            raise ValueError(f"body FWHM sweep settings must contain exactly {sorted(expected)}.")
        values = tuple(float(value) for value in raw["fwhm_values_m"])
        policy = cls(
            fwhm_values_m=values,
            reference_fwhm_m=float(raw["reference_fwhm_m"]),
            real_event_threshold_fraction=float(raw["real_event_threshold_fraction"]),
            residual_lobe_threshold_fraction=float(raw["residual_lobe_threshold_fraction"]),
            max_real_events_per_well=int(raw["max_real_events_per_well"]),
            minimum_event_samples=int(raw["minimum_event_samples"]),
            event_context_min_m=float(raw["event_context_min_m"]),
            event_context_width_multiple=float(raw["event_context_width_multiple"]),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        values = self.fwhm_values_m
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("fwhm_values_m must contain finite positive values.")
        if any(right <= left for left, right in zip(values[:-1], values[1:])):
            raise ValueError("fwhm_values_m must be unique and strictly increasing.")
        if self.reference_fwhm_m not in values:
            raise ValueError("reference_fwhm_m must be one of fwhm_values_m.")
        for name, value in (
            ("real_event_threshold_fraction", self.real_event_threshold_fraction),
            ("residual_lobe_threshold_fraction", self.residual_lobe_threshold_fraction),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and within (0, 1].")
        if self.max_real_events_per_well < 1 or self.minimum_event_samples < 2:
            raise ValueError("event counts and minimum_event_samples are invalid.")
        if (
            not math.isfinite(self.event_context_min_m)
            or self.event_context_min_m <= 0.0
            or not math.isfinite(self.event_context_width_multiple)
            or self.event_context_width_multiple <= 0.0
        ):
            raise ValueError("event context settings must be finite and positive.")


@dataclass(frozen=True)
class RealEventWindow:
    event_rank: int
    top_m: float
    bottom_m: float
    polarity: int
    peak_abs: float

    @property
    def width_m(self) -> float:
        return float(self.bottom_m - self.top_m)


@dataclass(frozen=True)
class CandidateSweepResult:
    fwhm_m: float
    native_body_log_ai: np.ndarray
    native_residual_log_ai: np.ndarray
    native_negative_curvature: np.ndarray
    model_body_log_ai: np.ndarray
    model_residual_log_ai: np.ndarray
    model_sharpening_template: np.ndarray
    body_forward: np.ndarray
    curvature_fit_gain: float
    curvature_fit_intercept: float
    sharpening_fit_gain: float
    sharpening_fit_intercept: float


@dataclass(frozen=True)
class WellBodyFwhmSweep:
    well_name: str
    native_axis_m: np.ndarray
    native_full_log_ai: np.ndarray
    native_target_support: np.ndarray
    model_axis_m: np.ndarray
    model_full_log_ai: np.ndarray
    model_target_support: np.ndarray
    real_seismic: np.ndarray
    full_forward: np.ndarray
    horizon_markers: tuple[tuple[float, str], ...]
    events: tuple[RealEventWindow, ...]
    candidates: tuple[CandidateSweepResult, ...]


@dataclass(frozen=True)
class BodyFwhmSweepResult:
    policy: BodyFwhmSweepPolicy
    wells: tuple[WellBodyFwhmSweep, ...]
    candidate_metrics: tuple[Mapping[str, Any], ...]
    event_metrics: tuple[Mapping[str, Any], ...]


def _finite_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    return tuple(
        (int(start), int(stop))
        for start, stop in np.flatnonzero(padded[1:] != padded[:-1]).reshape((-1, 2))
    )


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.sqrt(np.mean(np.square(finite)))) if finite.size else float("nan")


def _safe_corr(left: np.ndarray, right: np.ndarray, support: np.ndarray) -> float:
    selected = (
        np.asarray(support, dtype=bool)
        & np.isfinite(left)
        & np.isfinite(right)
    )
    if np.count_nonzero(selected) < 3:
        return float("nan")
    first = np.asarray(left, dtype=np.float64)[selected]
    second = np.asarray(right, dtype=np.float64)[selected]
    if np.std(first) <= np.finfo(np.float64).tiny or np.std(second) <= np.finfo(np.float64).tiny:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _template_fit(
    values: np.ndarray,
    template: np.ndarray,
    support: np.ndarray,
) -> tuple[float, float, float, float]:
    selected = (
        np.asarray(support, dtype=bool)
        & np.isfinite(values)
        & np.isfinite(template)
    )
    if np.count_nonzero(selected) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    target = np.asarray(values, dtype=np.float64)[selected]
    source = np.asarray(template, dtype=np.float64)[selected]
    source_centered = source - float(np.mean(source))
    denominator = float(np.dot(source_centered, source_centered))
    if denominator <= np.finfo(np.float64).tiny or np.std(target) <= np.finfo(np.float64).tiny:
        return float("nan"), float("nan"), float("nan"), float("nan")
    gain = float(np.dot(source_centered, target - float(np.mean(target))) / denominator)
    intercept = float(np.mean(target) - gain * np.mean(source))
    correlation = float(np.corrcoef(target, source)[0, 1])
    return correlation, correlation * correlation, gain, intercept


def _interpolate_finite_runs(
    source_axis: np.ndarray,
    source_values: np.ndarray,
    target_axis: np.ndarray,
) -> np.ndarray:
    output = np.full(np.asarray(target_axis).shape, np.nan, dtype=np.float64)
    for start, stop in _finite_runs(np.isfinite(source_values)):
        if stop - start < 2:
            continue
        inside = (target_axis >= source_axis[start]) & (target_axis <= source_axis[stop - 1])
        output[inside] = np.interp(
            target_axis[inside],
            source_axis[start:stop],
            source_values[start:stop],
        )
    return output


def _gaussian_smooth_for_sweep(
    values: np.ndarray,
    axis: np.ndarray,
    *,
    fwhm_m: float,
) -> np.ndarray:
    """Use the shared physical Gaussian, with an exact regular-axis fast path."""

    array = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(axis, dtype=np.float64)
    output = np.full(array.shape, np.nan, dtype=np.float64)
    sigma = float(fwhm_m) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    for start, stop in _finite_runs(np.isfinite(array)):
        local = array[start:stop]
        local_axis = coordinates[start:stop]
        if local.size < 2:
            output[start:stop] = local
            continue
        steps = np.diff(local_axis)
        step = float(np.median(steps))
        if not np.allclose(steps, step, rtol=1.0e-6, atol=1.0e-9):
            output[start:stop] = gaussian_smooth_finite_runs_numpy(
                local,
                local_axis,
                fwhm_m=fwhm_m,
            )
            continue
        radius_samples = int(math.floor((8.0 * sigma) / step + 1.0e-12))
        offsets = np.arange(-radius_samples, radius_samples + 1, dtype=np.float64) * step
        kernel = np.exp(-0.5 * np.square(offsets / sigma))
        numerator = fftconvolve(local, kernel, mode="same")
        denominator = fftconvolve(np.ones(local.shape, dtype=np.float64), kernel, mode="same")
        output[start:stop] = numerator / denominator
    return output


def _negative_curvature(
    axis: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    output = np.full(np.asarray(values).shape, np.nan, dtype=np.float64)
    selected = np.asarray(support, dtype=bool) & np.isfinite(values)
    for start, stop in _finite_runs(selected):
        if stop - start < 5:
            continue
        first = np.gradient(values[start:stop], axis[start:stop], edge_order=2)
        output[start:stop] = -np.gradient(first, axis[start:stop], edge_order=2)
    return output


def _zero_crossing_widths(
    axis: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    widths: list[float] = []
    selected = np.asarray(support, dtype=bool) & np.isfinite(values)
    for start, stop in _finite_runs(selected):
        local = np.asarray(values[start:stop], dtype=np.float64)
        local_axis = np.asarray(axis[start:stop], dtype=np.float64)
        if local.size < 2:
            continue
        changes = np.flatnonzero((local[1:] >= 0.0) != (local[:-1] >= 0.0)) + 1
        edges = np.r_[0, changes, local.size]
        for left, right in zip(edges[:-1], edges[1:]):
            if right <= left:
                continue
            local_step = float(np.median(np.diff(local_axis)))
            widths.append(float(local_axis[right - 1] - local_axis[left] + local_step))
    return np.asarray(widths, dtype=np.float64)


def _major_same_sign_widths(
    axis: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
    *,
    threshold_fraction: float,
    amplitude_scale: float | None = None,
) -> np.ndarray:
    selected = np.asarray(support, dtype=bool) & np.isfinite(values)
    if np.count_nonzero(selected) < 3:
        return np.empty(0, dtype=np.float64)
    centered = np.asarray(values, dtype=np.float64) - float(np.median(values[selected]))
    scale = (
        float(np.percentile(np.abs(centered[selected]), 95.0))
        if amplitude_scale is None
        else float(amplitude_scale)
    )
    if not math.isfinite(scale) or scale <= 0.0:
        return np.empty(0, dtype=np.float64)
    widths: list[float] = []
    for run_start, run_stop in _finite_runs(selected):
        local = centered[run_start:run_stop]
        local_axis = np.asarray(axis[run_start:run_stop], dtype=np.float64)
        if local.size < 1:
            continue
        changes = np.flatnonzero((local[1:] >= 0.0) != (local[:-1] >= 0.0)) + 1
        edges = np.r_[0, changes, local.size]
        local_step = (
            float(np.median(np.diff(local_axis)))
            if local_axis.size > 1
            else float(np.median(np.diff(axis)))
        )
        for left, right in zip(edges[:-1], edges[1:]):
            if right <= left or float(np.max(np.abs(local[left:right]))) < threshold_fraction * scale:
                continue
            widths.append(float(local_axis[right - 1] - local_axis[left] + local_step))
    return np.asarray(widths, dtype=np.float64)


def _autocorrelation_half_width(
    axis: np.ndarray,
    values: np.ndarray,
    support: np.ndarray,
) -> float:
    runs = _finite_runs(np.asarray(support, dtype=bool) & np.isfinite(values))
    if not runs:
        return float("nan")
    start, stop = max(runs, key=lambda item: item[1] - item[0])
    local_axis = np.asarray(axis[start:stop], dtype=np.float64)
    local = np.asarray(values[start:stop], dtype=np.float64)
    if local.size < 8:
        return float("nan")
    step = float(np.median(np.diff(local_axis)))
    if not np.allclose(np.diff(local_axis), step, rtol=1.0e-4, atol=1.0e-6):
        regular_axis = np.arange(local_axis[0], local_axis[-1] + 0.5 * step, step)
        local = np.interp(regular_axis, local_axis, local)
    local = local - float(np.mean(local))
    denominator = float(np.dot(local, local))
    if denominator <= np.finfo(np.float64).tiny:
        return float("nan")
    fft_size = 1 << (2 * local.size - 1).bit_length()
    spectrum = np.fft.rfft(local, n=fft_size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[: local.size]
    correlation = correlation / denominator
    crossings = np.flatnonzero(correlation <= 0.5)
    return float(crossings[0] * step) if crossings.size else float((local.size - 1) * step)


def _real_event_windows(
    axis: np.ndarray,
    seismic: np.ndarray,
    support: np.ndarray,
    *,
    threshold_fraction: float,
    maximum: int,
    minimum_samples: int,
) -> tuple[RealEventWindow, ...]:
    selected = np.asarray(support, dtype=bool) & np.isfinite(seismic)
    if np.count_nonzero(selected) < minimum_samples:
        return ()
    centered = np.asarray(seismic, dtype=np.float64) - float(np.median(seismic[selected]))
    scale = float(np.percentile(np.abs(centered[selected]), 95.0))
    records: list[tuple[float, float, float, int]] = []
    for run_start, run_stop in _finite_runs(selected):
        local = centered[run_start:run_stop]
        local_axis = np.asarray(axis[run_start:run_stop], dtype=np.float64)
        if local.size < minimum_samples:
            continue
        changes = np.flatnonzero((local[1:] >= 0.0) != (local[:-1] >= 0.0)) + 1
        edges = np.r_[0, changes, local.size]
        step = float(np.median(np.diff(local_axis)))
        for left, right in zip(edges[:-1], edges[1:]):
            if right - left < minimum_samples:
                continue
            values = local[left:right]
            peak = float(np.max(np.abs(values)))
            if peak < threshold_fraction * scale:
                continue
            top = float(local_axis[left] - 0.5 * step)
            bottom = float(local_axis[right - 1] + 0.5 * step)
            polarity = 1 if float(values[int(np.argmax(np.abs(values)))]) >= 0.0 else -1
            records.append((peak, top, bottom, polarity))
    strongest = sorted(records, key=lambda item: item[0], reverse=True)[:maximum]
    ranked = [
        RealEventWindow(rank, top, bottom, polarity, peak)
        for rank, (peak, top, bottom, polarity) in enumerate(strongest, start=1)
    ]
    return tuple(sorted(ranked, key=lambda item: item.top_m))


def _forward_metrics(
    full: np.ndarray,
    body: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(support, dtype=bool) & np.isfinite(full) & np.isfinite(body)
    if np.count_nonzero(selected) < 3:
        return {
            "forward_corr": float("nan"),
            "forward_difference_rms_ratio": float("nan"),
            "body_forward_rms_ratio": float("nan"),
            "body_to_full_gain": float("nan"),
            "gain_aligned_forward_nrmse": float("nan"),
        }
    full_values = np.asarray(full, dtype=np.float64)[selected]
    body_values = np.asarray(body, dtype=np.float64)[selected]
    full_rms = _rms(full_values)
    denominator = float(np.dot(body_values, body_values))
    gain = float(np.dot(full_values, body_values) / denominator) if denominator > 0.0 else float("nan")
    return {
        "forward_corr": _safe_corr(full, body, selected),
        "forward_difference_rms_ratio": _rms(full_values - body_values) / full_rms,
        "body_forward_rms_ratio": _rms(body_values) / full_rms,
        "body_to_full_gain": gain,
        "gain_aligned_forward_nrmse": _rms(full_values - gain * body_values) / full_rms,
    }


def _peak_shift_m(
    axis: np.ndarray,
    full: np.ndarray,
    body: np.ndarray,
    support: np.ndarray,
) -> float:
    selected = np.flatnonzero(
        np.asarray(support, dtype=bool) & np.isfinite(full) & np.isfinite(body)
    )
    if selected.size < 2:
        return float("nan")
    full_index = int(selected[int(np.argmax(np.abs(full[selected])))])
    body_index = int(selected[int(np.argmax(np.abs(body[selected])))])
    return float(axis[body_index] - axis[full_index])


def _candidate_for_well(
    control: WellControl,
    *,
    fwhm_m: float,
    native_target: np.ndarray,
    model_target: np.ndarray,
    full_forward: np.ndarray,
    real_seismic: np.ndarray,
    events: tuple[RealEventWindow, ...],
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    relation_a: float,
    relation_b: float,
    policy: BodyFwhmSweepPolicy,
) -> tuple[CandidateSweepResult, dict[str, Any], list[dict[str, Any]]]:
    native_axis = np.asarray(control.native.coordinates, dtype=np.float64)
    native_full = np.asarray(control.native.full_log_ai, dtype=np.float64)
    native_body = _gaussian_smooth_for_sweep(
        native_full,
        native_axis,
        fwhm_m=fwhm_m,
    )
    native_residual = native_full - native_body
    native_support = native_target & np.isfinite(native_body) & np.isfinite(native_residual)
    negative_curvature = _negative_curvature(native_axis, native_body, native_support)
    curvature_corr, curvature_r2, curvature_gain, curvature_intercept = _template_fit(
        native_residual,
        negative_curvature,
        native_support,
    )

    model_axis = np.asarray(control.sample_axis.values, dtype=np.float64)
    model_full = np.asarray(control.log_ai.values, dtype=np.float64)
    model_body = _interpolate_finite_runs(native_axis, native_body, model_axis)
    model_residual = model_full - model_body
    twice_smoothed_body = _gaussian_smooth_for_sweep(
        model_body,
        model_axis,
        fwhm_m=fwhm_m,
    )
    sharpening_template = model_body - twice_smoothed_body
    model_support = (
        model_target
        & np.isfinite(model_full)
        & np.isfinite(model_body)
        & np.isfinite(model_residual)
        & np.isfinite(sharpening_template)
    )
    sharpening_corr, sharpening_r2, sharpening_gain, sharpening_intercept = _template_fit(
        model_residual,
        sharpening_template,
        model_support,
    )
    body_forward = forward_depth_finite_runs(
        model_body,
        model_axis,
        wavelet_time_s=wavelet_time_s,
        wavelet_amplitude=wavelet_amplitude,
        relation_a=relation_a,
        relation_b=relation_b,
    )
    forward_support = model_target & np.isfinite(full_forward) & np.isfinite(body_forward)
    forward = _forward_metrics(full_forward, body_forward, forward_support)

    zero_widths = _zero_crossing_widths(native_axis, native_residual, native_support)
    centered_residual = native_residual - float(np.median(native_residual[native_support]))
    residual_scale = float(np.percentile(np.abs(centered_residual[native_support]), 95.0))
    major_widths = _major_same_sign_widths(
        native_axis,
        native_residual,
        native_support,
        threshold_fraction=policy.residual_lobe_threshold_fraction,
        amplitude_scale=residual_scale,
    )
    row: dict[str, Any] = {
        "well_name": control.well_name,
        "candidate": f"F{fwhm_m:g}",
        "fwhm_m": float(fwhm_m),
        "native_target_samples": int(np.count_nonzero(native_support)),
        "model_target_samples": int(np.count_nonzero(model_support)),
        "native_residual_rms": _rms(native_residual[native_support]),
        "native_residual_zero_crossing_p50_m": (
            float(np.median(zero_widths)) if zero_widths.size else float("nan")
        ),
        "native_residual_zero_crossing_p90_m": (
            float(np.quantile(zero_widths, 0.90)) if zero_widths.size else float("nan")
        ),
        "native_residual_major_interval_count": int(major_widths.size),
        "native_residual_major_interval_width_p50_m": (
            float(np.median(major_widths)) if major_widths.size else float("nan")
        ),
        "native_residual_major_interval_width_p90_m": (
            float(np.quantile(major_widths, 0.90)) if major_widths.size else float("nan")
        ),
        "native_residual_autocorrelation_half_width_m": _autocorrelation_half_width(
            native_axis,
            native_residual,
            native_support,
        ),
        "native_residual_negative_curvature_corr": curvature_corr,
        "native_residual_negative_curvature_r2": curvature_r2,
        "model_residual_rms": _rms(model_residual[model_support]),
        "model_residual_unsharp_corr": sharpening_corr,
        "model_residual_unsharp_r2": sharpening_r2,
        "model_body_full_corr": _safe_corr(model_full, model_body, model_support),
        "full_real_forward_corr": _safe_corr(
            real_seismic,
            full_forward,
            model_target & np.isfinite(real_seismic) & np.isfinite(full_forward),
        ),
        "body_real_forward_corr": _safe_corr(
            real_seismic,
            body_forward,
            model_target & np.isfinite(real_seismic) & np.isfinite(body_forward),
        ),
        **forward,
    }

    event_rows: list[dict[str, Any]] = []
    for event in events:
        native_event = (
            native_support
            & (native_axis >= event.top_m)
            & (native_axis <= event.bottom_m)
        )
        event_widths = _major_same_sign_widths(
            native_axis,
            native_residual,
            native_event,
            threshold_fraction=policy.residual_lobe_threshold_fraction,
            amplitude_scale=residual_scale,
        )
        event_curvature_corr, event_curvature_r2, _gain, _intercept = _template_fit(
            native_residual,
            negative_curvature,
            native_event,
        )
        model_event = (
            model_target
            & (model_axis >= event.top_m)
            & (model_axis <= event.bottom_m)
            & np.isfinite(full_forward)
            & np.isfinite(body_forward)
        )
        event_forward = _forward_metrics(full_forward, body_forward, model_event)
        event_rows.append(
            {
                "well_name": control.well_name,
                "event_rank": int(event.event_rank),
                "event_top_m": float(event.top_m),
                "event_bottom_m": float(event.bottom_m),
                "event_width_m": float(event.width_m),
                "event_polarity": int(event.polarity),
                "event_peak_abs": float(event.peak_abs),
                "candidate": f"F{fwhm_m:g}",
                "fwhm_m": float(fwhm_m),
                "residual_major_interval_count": int(event_widths.size),
                "residual_major_interval_width_p50_m": (
                    float(np.median(event_widths)) if event_widths.size else float("nan")
                ),
                "residual_major_interval_width_p90_m": (
                    float(np.quantile(event_widths, 0.90)) if event_widths.size else float("nan")
                ),
                "residual_negative_curvature_corr": event_curvature_corr,
                "residual_negative_curvature_r2": event_curvature_r2,
                "forward_peak_shift_m": _peak_shift_m(
                    model_axis,
                    full_forward,
                    body_forward,
                    model_event,
                ),
                **event_forward,
            }
        )

    candidate = CandidateSweepResult(
        fwhm_m=float(fwhm_m),
        native_body_log_ai=native_body,
        native_residual_log_ai=native_residual,
        native_negative_curvature=negative_curvature,
        model_body_log_ai=model_body,
        model_residual_log_ai=model_residual,
        model_sharpening_template=sharpening_template,
        body_forward=body_forward,
        curvature_fit_gain=curvature_gain,
        curvature_fit_intercept=curvature_intercept,
        sharpening_fit_gain=sharpening_gain,
        sharpening_fit_intercept=sharpening_intercept,
    )
    return candidate, row, event_rows


def run_body_fwhm_sweep(
    controls: WellControlSet,
    *,
    trusted_well_names: Sequence[str],
    survey: Any,
    target_zone: TargetZone,
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    relation_a: float,
    relation_b: float,
    policy: BodyFwhmSweepPolicy,
    logger: logging.Logger | None = None,
) -> BodyFwhmSweepResult:
    """Run the fixed-well, fixed-event FWHM sweep."""

    policy.validate()
    if controls.sample_domain != "depth" or controls.sample_unit != "m" or controls.depth_basis != "tvdss":
        raise ValueError("The body FWHM sweep requires depth/TVDSS well controls.")
    names = tuple(str(name).strip() for name in trusted_well_names)
    if not names or any(not name for name in names) or len({name.casefold() for name in names}) != len(names):
        raise ValueError("trusted_well_names must be non-empty and unique.")
    control_by_name = {control.well_name.casefold(): control for control in controls.controls}
    selected_controls: list[WellControl] = []
    for name in names:
        control = control_by_name.get(name.casefold())
        if control is None:
            raise ValueError(f"Trusted well is absent from WellControlSet: {name}")
        selected_controls.append(control)

    well_results: list[WellBodyFwhmSweep] = []
    candidate_rows: list[Mapping[str, Any]] = []
    event_rows: list[Mapping[str, Any]] = []
    for control in selected_controls:
        markers = tuple(horizon_markers_along_control(control, target_zone))
        target_top = float(markers[0][0])
        target_bottom = float(markers[-1][0])
        native_axis = np.asarray(control.native.coordinates, dtype=np.float64)
        native_full = np.asarray(control.native.full_log_ai, dtype=np.float64)
        native_target = (
            np.asarray(control.native.valid_mask, dtype=bool)
            & (native_axis >= target_top)
            & (native_axis <= target_bottom)
        )
        model_axis = np.asarray(control.sample_axis.values, dtype=np.float64)
        model_full = np.asarray(control.log_ai.values, dtype=np.float64)
        model_target = (
            np.asarray(control.observed_valid_mask, dtype=bool)
            & (model_axis >= target_top)
            & (model_axis <= target_bottom)
        )
        real_seismic = sample_seismic_along_control(control, survey)
        full_forward = forward_depth_finite_runs(
            model_full,
            model_axis,
            wavelet_time_s=wavelet_time_s,
            wavelet_amplitude=wavelet_amplitude,
            relation_a=relation_a,
            relation_b=relation_b,
        )
        events = _real_event_windows(
            model_axis,
            real_seismic,
            model_target,
            threshold_fraction=policy.real_event_threshold_fraction,
            maximum=policy.max_real_events_per_well,
            minimum_samples=policy.minimum_event_samples,
        )
        if not events:
            raise ValueError(f"{control.well_name}: no fixed real-seismic event window passed the sweep threshold.")
        candidates: list[CandidateSweepResult] = []
        for fwhm_m in policy.fwhm_values_m:
            candidate, candidate_row, local_event_rows = _candidate_for_well(
                control,
                fwhm_m=fwhm_m,
                native_target=native_target,
                model_target=model_target,
                full_forward=full_forward,
                real_seismic=real_seismic,
                events=events,
                wavelet_time_s=wavelet_time_s,
                wavelet_amplitude=wavelet_amplitude,
                relation_a=relation_a,
                relation_b=relation_b,
                policy=policy,
            )
            candidates.append(candidate)
            candidate_rows.append(candidate_row)
            event_rows.extend(local_event_rows)
        well_results.append(
            WellBodyFwhmSweep(
                well_name=control.well_name,
                native_axis_m=native_axis,
                native_full_log_ai=native_full,
                native_target_support=native_target,
                model_axis_m=model_axis,
                model_full_log_ai=model_full,
                model_target_support=model_target,
                real_seismic=real_seismic,
                full_forward=full_forward,
                horizon_markers=markers,
                events=events,
                candidates=tuple(candidates),
            )
        )
        if logger is not None:
            logger.info(
                "well sweep complete | well=%s | candidates=%d | events=%d",
                control.well_name,
                len(candidates),
                len(events),
            )
    return BodyFwhmSweepResult(
        policy=policy,
        wells=tuple(well_results),
        candidate_metrics=tuple(candidate_rows),
        event_metrics=tuple(event_rows),
    )


__all__ = [
    "BodyFwhmSweepPolicy",
    "BodyFwhmSweepResult",
    "CandidateSweepResult",
    "RealEventWindow",
    "SCHEMA_VERSION",
    "WellBodyFwhmSweep",
    "run_body_fwhm_sweep",
]
