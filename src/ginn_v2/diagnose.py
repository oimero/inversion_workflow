"""GINN training QC, body-scale diagnosis, and diagnostic artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import torch

from cup.seismic.geometry import SampleAxis
from cup.seismic.target_zone import TargetZone
from cup.seismic.viz import plot_well_waveform_qc
from cup.utils.io import repo_relative_path, sanitize_filename, write_json
from cup.utils.masks import true_runs as _finite_runs
from cup.well.controls import (
    WellControl,
    WellControlSet,
    _waveform_objects,
    forward_depth_finite_runs,
    horizon_markers_along_control,
    sample_seismic_along_control,
)
from cup.well.scale import gaussian_smooth_finite_runs_numpy
from ginn_v2.physics import CommonObservationBatch
from wtie.processing import grid


def _longest_contiguous_run(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional integer array.")
    if np.any(np.diff(values) <= 0):
        raise ValueError("indices must be strictly increasing.")
    breaks = np.flatnonzero(np.diff(values) > 1) + 1
    bounds = np.column_stack((np.r_[0, breaks], np.r_[breaks, values.size]))
    start, stop = max(bounds, key=lambda item: int(item[1] - item[0]))
    return values[int(start) : int(stop)]


def _forward_well_curve(
    trainer: Any,
    *,
    axis: SampleAxis,
    body_log_ai: np.ndarray,
    observed_seismic: np.ndarray,
    observed_valid_mask: np.ndarray,
    lfm_log_ai: np.ndarray,
    lfm_valid_mask: np.ndarray,
    xy_m: np.ndarray,
    domain_extras: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Forward one assembled well curve through the same adapter as training."""

    body = torch.as_tensor(body_log_ai, device=trainer.device, dtype=torch.float32)[None, :]
    common = CommonObservationBatch(
        sample_axis=axis,
        observed_seismic=torch.as_tensor(observed_seismic, device=trainer.device, dtype=torch.float32)[None, :],
        observed_valid_mask=torch.as_tensor(observed_valid_mask, device=trainer.device, dtype=torch.bool)[None, :],
        lfm_log_ai=torch.as_tensor(lfm_log_ai, device=trainer.device, dtype=torch.float32)[None, :],
        lfm_valid_mask=torch.as_tensor(lfm_valid_mask, device=trainer.device, dtype=torch.bool)[None, :],
        xy_m=torch.as_tensor(xy_m, device=trainer.device, dtype=torch.float32)[None, :],
        domain_extras={
            name: torch.as_tensor(value, device=trainer.device, dtype=torch.float32)[None, :]
            for name, value in domain_extras.items()
        },
    )
    with torch.no_grad():
        closure = trainer.adapter.close_body(body, common)
    return closure.synthetic_seismic[0].cpu().numpy(), common.observed_seismic[0].cpu().numpy()


def write_well_waveform_qc(
    trainer: Any,
    model: Any,
    qc_dir: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    """Write one waveform QC figure and metrics table per trusted well."""

    import matplotlib.pyplot as plt

    if trainer.data.reader.sample_axis.domain != "depth" or trainer.data.reader.sample_axis.depth_basis != "tvdss":
        raise ValueError("GINN V2 well waveform QC currently requires a depth/TVDSS SampleAxis.")

    predictions: dict[str, dict[int, dict[str, list[float] | dict[str, list[float]]]]] = {}
    with torch.no_grad():
        items = trainer.data.trusted_well_patches
        for start in range(0, len(items), trainer.config.batch_size):
            local = items[start : start + trainer.config.batch_size]
            batch = trainer.data.reader.batch(
                tuple(item.patch_key for item in local),
                center_visible=True,
                device=trainer.device,
            )
            body, _synthetic, common = trainer._predict(model, batch)
            for row, item in enumerate(local):
                observed = batch.observed_seismic[row].cpu().numpy()
                observed_mask = common.observed_valid_mask[row].cpu().numpy()
                body_values = body[row].cpu().numpy()
                lfm_values = batch.lfm_log_ai[row].cpu().numpy()
                lfm_mask = batch.lfm_valid_mask[row].cpu().numpy()
                xy = batch.xy_m[row].cpu().numpy()
                domain_values = {
                    name: value[row].cpu().numpy()
                    for name, value in batch.domain_extras.items()
                }
                for sample_index in np.flatnonzero(item.target_mask):
                    index = int(sample_index)
                    sample = predictions.setdefault(item.well_name, {}).setdefault(
                        index,
                        {
                            "body_log_ai": [],
                            "observed": [],
                            "lfm_log_ai": [],
                            "xy_m": [],
                            "domain_extras": {},
                        },
                    )
                    sample["body_log_ai"].append(float(body_values[index]))  # type: ignore[union-attr]
                    if observed_mask[index]:
                        sample["observed"].append(float(observed[index]))  # type: ignore[union-attr]
                    if lfm_mask[index]:
                        sample["lfm_log_ai"].append(float(lfm_values[index]))  # type: ignore[union-attr]
                    sample["xy_m"].append(xy.tolist())  # type: ignore[union-attr]
                    extras = sample["domain_extras"]  # type: ignore[assignment]
                    for name, values in domain_values.items():
                        if np.isfinite(values[index]):
                            extras.setdefault(name, []).append(float(values[index]))

    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    axis = np.asarray(trainer.data.reader.sample_axis.values, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    figures: list[str] = []
    for well_name in sorted(predictions):
        by_sample = predictions[well_name]
        available = np.asarray(
            [
                index
                for index in sorted(by_sample)
                if by_sample[index]["body_log_ai"]
                and by_sample[index]["observed"]
                and by_sample[index]["lfm_log_ai"]
                and all(by_sample[index]["domain_extras"].get(name) for name in trainer.data.reader.domain_extras)
            ],
            dtype=np.int64,
        )
        indices = _longest_contiguous_run(available)
        if indices.size < 8:
            raise ValueError(f"{well_name}: longest predicted well QC support run has fewer than eight samples.")

        record = lambda index: by_sample[int(index)]
        predicted_log_ai = np.asarray(
            [np.mean(record(index)["body_log_ai"]) for index in indices],
            dtype=np.float64,
        )
        observed = np.asarray(
            [np.mean(record(index)["observed"]) for index in indices],
            dtype=np.float64,
        )
        lfm_values = np.asarray(
            [np.mean(record(index)["lfm_log_ai"]) for index in indices],
            dtype=np.float64,
        )
        lfm_mask = np.ones(indices.size, dtype=bool)
        xy_values = np.asarray(
            [np.mean(np.asarray(record(index)["xy_m"], dtype=np.float64), axis=0) for index in indices],
            dtype=np.float64,
        )
        xy_m = np.mean(xy_values, axis=0)
        domain_extras = {
            name: np.asarray(
                [np.mean(record(index)["domain_extras"][name]) for index in indices],
                dtype=np.float64,
            )
            for name in trainer.data.reader.domain_extras
        }
        target = trainer.data.well_targets[well_name]
        reference_log_ai = np.asarray(target.model_axis_target[indices], dtype=np.float64)
        observed_valid_mask = np.ones(indices.size, dtype=bool)
        if any(
            np.any(~np.isfinite(values))
            for values in (reference_log_ai, predicted_log_ai, observed, lfm_values, xy_m, *domain_extras.values())
        ):
            raise ValueError(f"{well_name}: assembled well QC arrays contain non-finite values.")

        local_axis = np.asarray(axis[indices], dtype=np.float64)
        local_sample_axis = SampleAxis(
            local_axis,
            trainer.data.reader.sample_axis.domain,
            trainer.data.reader.sample_axis.unit,
            trainer.data.reader.sample_axis.depth_basis,
        )
        synthetic, observed = _forward_well_curve(
            trainer,
            axis=local_sample_axis,
            body_log_ai=predicted_log_ai,
            observed_seismic=observed,
            observed_valid_mask=observed_valid_mask,
            lfm_log_ai=lfm_values,
            lfm_valid_mask=lfm_mask,
            xy_m=xy_m,
            domain_extras=domain_extras,
        )

        observed_centered = observed - float(np.mean(observed))
        observed_scale = float(np.std(observed_centered))
        if observed_scale <= 0.0 or not np.isfinite(observed_scale):
            raise ValueError(f"{well_name}: observed well seismic has zero variance in the QC interval.")
        observed_normalized = observed_centered / observed_scale
        synthetic_denominator = float(np.dot(synthetic, synthetic))
        signed_gain = (
            float(np.dot(observed_normalized, synthetic) / synthetic_denominator)
            if synthetic_denominator > 0.0
            else 1.0
        )
        gain = abs(signed_gain)
        synthetic_scaled = gain * synthetic
        correlation = float(np.corrcoef(observed_normalized, synthetic_scaled)[0, 1])
        if not np.isfinite(correlation):
            raise ValueError(f"{well_name}: predicted well waveform correlation is non-finite.")

        predicted_objects = _waveform_objects(
            axis=local_axis,
            log_ai=predicted_log_ai,
            synthetic=synthetic_scaled,
            real=observed_normalized,
            dynamic_window_m=float(trainer.config.waveform_qc_dynamic_window_m),
            name="GINN V2 predicted body",
        )
        body_reference = grid.Log(
            np.exp(reference_log_ai),
            local_axis,
            "tvdss",
            name=f"{trainer.config.body_smoothing_fwhm_m:g} m body reference",
            unit="m/s*g/cm3",
        )
        figure, axes = plot_well_waveform_qc(
            [body_reference, predicted_objects[0]],
            predicted_objects[1],
            predicted_objects[2],
            predicted_objects[3],
            predicted_objects[4],
            predicted_objects[5],
            figsize=(13.0, 7.5),
            synthetic_ai=predicted_objects[0],
            title=f"GINN V2 well-curve forward QC | {well_name} | corr={correlation:.3f}",
        )
        for line in axes[0].lines:
            if line.get_label() == body_reference.name:
                line.set_color("gray")
                line.set_linewidth(1.3)
                line.set_alpha(0.85)
                line.set_zorder(2)
        legend = axes[0].get_legend()
        if legend is not None:
            handles = getattr(legend, "legend_handles", getattr(legend, "legendHandles", ()))
            for handle, label in zip(handles, legend.get_texts()):
                if label.get_text() == body_reference.name:
                    handle.set_color("gray")
                    handle.set_linewidth(1.3)
                    handle.set_alpha(0.85)
        well_dir = qc_dir / sanitize_filename(well_name)
        well_dir.mkdir(parents=True, exist_ok=True)
        figure_path = well_dir / "waveform_qc.png"
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        figures.append(repo_relative_path(figure_path, root=root))

        body_residual = predicted_log_ai - reference_log_ai
        rows.append(
            {
                "well_name": well_name,
                "support_start_m": float(local_axis[0]),
                "support_stop_m": float(local_axis[-1]),
                "support_samples": int(indices.size),
                "signed_forward_gain": signed_gain,
                "well_curve_forward_corr": correlation,
                "predicted_vs_body_rmse_log_ai": float(np.sqrt(np.mean(np.square(body_residual)))),
                "predicted_vs_body_corr": float(np.corrcoef(reference_log_ai, predicted_log_ai)[0, 1]),
                "figure": repo_relative_path(figure_path, root=root),
            }
        )

    metrics_path = qc_dir / "metrics.csv"
    fieldnames = list(rows[0]) if rows else ["well_name", "figure"]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "ok",
        "plot_function": "cup.seismic.viz.plot_well_waveform_qc",
        "correlation_metric": "well_curve_forward_corr",
        "correlation_definition": "Assemble the predicted AI curve on the well support, then forward it with the same domain adapter used by training.",
        "dynamic_correlation_window_m": float(trainer.config.waveform_qc_dynamic_window_m),
        "figures": figures,
        "metrics": repo_relative_path(metrics_path, root=root),
    }
    write_json(qc_dir / "manifest.json", manifest)
    return manifest


SCHEMA_VERSION = "ginn_v2_body_fwhm_sweep_v2"


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
    native_filtered_log_ai: np.ndarray
    native_target_support: np.ndarray
    model_axis_m: np.ndarray
    model_grid_filtered_log_ai: np.ndarray
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
    native_filtered = np.asarray(control.native.native_filtered_log_ai, dtype=np.float64)
    native_body = _gaussian_smooth_for_sweep(
        native_filtered,
        native_axis,
        fwhm_m=fwhm_m,
    )
    native_residual = native_filtered - native_body
    native_support = native_target & np.isfinite(native_body) & np.isfinite(native_residual)
    negative_curvature = _negative_curvature(native_axis, native_body, native_support)
    curvature_corr, curvature_r2, curvature_gain, curvature_intercept = _template_fit(
        native_residual,
        negative_curvature,
        native_support,
    )

    model_axis = np.asarray(control.sample_axis.values, dtype=np.float64)
    model_grid_filtered = np.asarray(control.model_grid_filtered_log_ai.values, dtype=np.float64)
    model_body = _interpolate_finite_runs(native_axis, native_body, model_axis)
    model_residual = model_grid_filtered - model_body
    twice_smoothed_body = _gaussian_smooth_for_sweep(
        model_body,
        model_axis,
        fwhm_m=fwhm_m,
    )
    sharpening_template = model_body - twice_smoothed_body
    model_support = (
        model_target
        & np.isfinite(model_grid_filtered)
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
        "model_body_full_corr": _safe_corr(model_grid_filtered, model_body, model_support),
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
        native_filtered = np.asarray(control.native.native_filtered_log_ai, dtype=np.float64)
        native_target = (
            np.asarray(control.native.valid_mask, dtype=bool)
            & (native_axis >= target_top)
            & (native_axis <= target_bottom)
        )
        model_axis = np.asarray(control.sample_axis.values, dtype=np.float64)
        model_grid_filtered = np.asarray(control.model_grid_filtered_log_ai.values, dtype=np.float64)
        model_target = (
            np.asarray(control.observed_valid_mask, dtype=bool)
            & (model_axis >= target_top)
            & (model_axis <= target_bottom)
        )
        real_seismic = sample_seismic_along_control(control, survey)
        full_forward = forward_depth_finite_runs(
            model_grid_filtered,
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
                native_filtered_log_ai=native_filtered,
                native_target_support=native_target,
                model_axis_m=model_axis,
                model_grid_filtered_log_ai=model_grid_filtered,
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


_SUMMARY_METRICS = (
    "forward_corr",
    "forward_difference_rms_ratio",
    "gain_aligned_forward_nrmse",
    "native_residual_negative_curvature_r2",
    "model_residual_unsharp_r2",
    "native_residual_major_interval_width_p50_m",
    "native_residual_autocorrelation_half_width_m",
)


def _fwhm_key(value: float) -> str:
    return f"fwhm_{value:g}".replace(".", "p")


def _finite_scale(values: list[np.ndarray], *, quantile: float = 0.99) -> float:
    finite = [np.abs(item[np.isfinite(item)]) for item in values]
    finite = [item for item in finite if item.size]
    if not finite:
        return 1.0
    scale = float(np.quantile(np.concatenate(finite), quantile))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _normalized(values: np.ndarray, scale: float) -> np.ndarray:
    output = np.full(np.asarray(values).shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    output[finite] = np.asarray(values, dtype=np.float64)[finite] / float(scale)
    return output


def _plot_window_comparison(
    well: WellBodyFwhmSweep,
    *,
    top_m: float,
    bottom_m: float,
    event_top_m: float | None,
    event_bottom_m: float | None,
    output_path: Path,
    title: str,
    reference_fwhm_m: float,
) -> None:
    native_view = (
        (well.native_axis_m >= top_m)
        & (well.native_axis_m <= bottom_m)
        & well.native_target_support
    )
    model_view = (
        (well.model_axis_m >= top_m)
        & (well.model_axis_m <= bottom_m)
        & well.model_target_support
    )
    if not np.any(native_view) or not np.any(model_view):
        raise ValueError(f"{well.well_name}: comparison window has no native/model support.")
    body_scale = _finite_scale(
        [well.native_filtered_log_ai[native_view]]
        + [item.native_body_log_ai[native_view] for item in well.candidates]
    )
    body_values = np.concatenate(
        [well.native_filtered_log_ai[native_view]]
        + [item.native_body_log_ai[native_view] for item in well.candidates]
    )
    body_finite = body_values[np.isfinite(body_values)]
    body_min = float(np.quantile(body_finite, 0.01))
    body_max = float(np.quantile(body_finite, 0.99))
    body_pad = max(0.03 * (body_max - body_min), 1.0e-4 * body_scale)
    residual_limit = _finite_scale(
        [item.native_residual_log_ai[native_view] for item in well.candidates]
        + [
            item.curvature_fit_intercept
            + item.curvature_fit_gain * item.native_negative_curvature[native_view]
            for item in well.candidates
        ]
    )
    forward_scale = _finite_scale(
        [well.full_forward[model_view]]
        + [item.body_forward[model_view] for item in well.candidates],
        quantile=0.95,
    )
    seismic_scale = _finite_scale([well.real_seismic[model_view]], quantile=0.95)

    figure, axes = plt.subplots(
        len(well.candidates),
        4,
        figsize=(13.5, max(8.0, 2.0 * len(well.candidates))),
        squeeze=False,
        constrained_layout=True,
    )
    for row, candidate in enumerate(well.candidates):
        axes[row, 0].plot(
            _normalized(well.real_seismic, seismic_scale),
            well.model_axis_m,
            color="black",
            linewidth=1.0,
        )
        axes[row, 1].plot(
            well.native_filtered_log_ai,
            well.native_axis_m,
            color="0.25",
            linewidth=0.65,
            label="full well log",
        )
        axes[row, 1].plot(
            candidate.native_body_log_ai,
            well.native_axis_m,
            color="#d62728",
            linewidth=1.15,
            label=f"{candidate.fwhm_m:g} m body",
        )
        fitted_curvature = (
            candidate.curvature_fit_intercept
            + candidate.curvature_fit_gain * candidate.native_negative_curvature
        )
        axes[row, 2].plot(
            candidate.native_residual_log_ai,
            well.native_axis_m,
            color="#9467bd",
            linewidth=0.75,
            label="full − body",
        )
        axes[row, 2].plot(
            fitted_curvature,
            well.native_axis_m,
            color="0.2",
            linewidth=0.85,
            linestyle="--",
            label="fitted sharpening",
        )
        axes[row, 2].axvline(0.0, color="0.6", linewidth=0.5)
        axes[row, 3].plot(
            _normalized(well.full_forward, forward_scale),
            well.model_axis_m,
            color="black",
            linewidth=1.0,
            label="full forward",
        )
        axes[row, 3].plot(
            _normalized(candidate.body_forward, forward_scale),
            well.model_axis_m,
            color="#ff7f0e",
            linewidth=1.05,
            label="body forward",
        )
        axes[row, 0].set_xlim(-1.2, 1.2)
        axes[row, 1].set_xlim(body_min - body_pad, body_max + body_pad)
        axes[row, 2].set_xlim(-residual_limit, residual_limit)
        axes[row, 3].set_xlim(-1.2, 1.2)
        for column in range(4):
            panel = axes[row, column]
            panel.set_ylim(bottom_m, top_m)
            panel.grid(alpha=0.20)
            if event_top_m is not None and event_bottom_m is not None:
                panel.axhspan(event_top_m, event_bottom_m, color="#1f77b4", alpha=0.10)
        marker = " ← current" if candidate.fwhm_m == reference_fwhm_m else ""
        axes[row, 0].set_ylabel(f"F={candidate.fwhm_m:g} m{marker}\nTVDSS [m]")
    for column, name in enumerate(
        ("real seismic", "full / body log-AI", "residual / sharpening fit", "full / body forward")
    ):
        axes[0, column].set_title(name)
    axes[0, 1].legend(fontsize=7)
    axes[0, 2].legend(fontsize=7)
    axes[0, 3].legend(fontsize=7)
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_well_artifact(well: WellBodyFwhmSweep, path: Path) -> None:
    payload: dict[str, np.ndarray] = {
        "native_axis_m": well.native_axis_m,
        "native_filtered_log_ai": well.native_filtered_log_ai,
        "native_target_support": well.native_target_support,
        "model_axis_m": well.model_axis_m,
        "model_grid_filtered_log_ai": well.model_grid_filtered_log_ai,
        "model_target_support": well.model_target_support,
        "real_seismic": well.real_seismic,
        "full_forward": well.full_forward,
        "horizon_depth_m": np.asarray([item[0] for item in well.horizon_markers], dtype=np.float64),
        "horizon_name": np.asarray([item[1] for item in well.horizon_markers], dtype="U64"),
        "event_rank": np.asarray([item.event_rank for item in well.events], dtype=np.int32),
        "event_top_m": np.asarray([item.top_m for item in well.events], dtype=np.float64),
        "event_bottom_m": np.asarray([item.bottom_m for item in well.events], dtype=np.float64),
        "event_polarity": np.asarray([item.polarity for item in well.events], dtype=np.int8),
        "event_peak_abs": np.asarray([item.peak_abs for item in well.events], dtype=np.float64),
    }
    for candidate in well.candidates:
        prefix = _fwhm_key(candidate.fwhm_m)
        payload[f"{prefix}_native_body_log_ai"] = candidate.native_body_log_ai
        payload[f"{prefix}_native_residual_log_ai"] = candidate.native_residual_log_ai
        payload[f"{prefix}_native_negative_curvature"] = candidate.native_negative_curvature
        payload[f"{prefix}_model_body_log_ai"] = candidate.model_body_log_ai
        payload[f"{prefix}_model_residual_log_ai"] = candidate.model_residual_log_ai
        payload[f"{prefix}_model_sharpening_template"] = candidate.model_sharpening_template
        payload[f"{prefix}_body_forward"] = candidate.body_forward
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _summary_frame(candidate_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fwhm_m, group in candidate_metrics.groupby("fwhm_m", sort=True):
        row: dict[str, Any] = {
            "candidate": f"F{float(fwhm_m):g}",
            "fwhm_m": float(fwhm_m),
            "well_count": int(group["well_name"].nunique()),
        }
        for metric in _SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_p10"] = float(values.quantile(0.10))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_p90"] = float(values.quantile(0.90))
        records.append(row)
    return pd.DataFrame.from_records(records)


def _plot_summary(candidate_metrics: pd.DataFrame, output_path: Path) -> None:
    panels = (
        ("forward_corr", "Full/body forward correlation"),
        ("forward_difference_rms_ratio", "Forward difference RMS / full RMS"),
        ("native_residual_negative_curvature_r2", "Native residual explained by curvature"),
        ("model_residual_unsharp_r2", "5 m residual explained by unsharp body"),
        ("native_residual_major_interval_width_p50_m", "Major residual interval P50 [m]"),
        ("native_residual_autocorrelation_half_width_m", "Residual autocorrelation half-width [m]"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    for panel, (metric, title) in zip(axes.ravel(), panels):
        for _well_name, group in candidate_metrics.groupby("well_name", sort=False):
            ordered = group.sort_values("fwhm_m")
            panel.plot(
                ordered["fwhm_m"],
                ordered[metric],
                color="0.72",
                linewidth=0.8,
                marker="o",
                markersize=2.5,
            )
        median = candidate_metrics.groupby("fwhm_m", sort=True)[metric].median()
        panel.plot(
            median.index,
            median.values,
            color="#d62728",
            linewidth=2.0,
            marker="o",
            label="well median",
        )
        panel.set_title(title)
        panel.set_xlabel("Body smoothing FWHM [m]")
        panel.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("GINN V2 body/residual FWHM sweep")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_body_fwhm_sweep_artifacts(
    result: BodyFwhmSweepResult,
    *,
    output_dir: Path,
    repo_root: Path,
    resolved_config: Mapping[str, Any],
    inputs: Mapping[str, str],
    horizon_sources: list[dict[str, str]],
) -> dict[str, Any]:
    """Write one complete sweep result into a new output directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    wells_dir = output_dir / "wells"
    figures_dir.mkdir()
    wells_dir.mkdir()

    candidate_metrics = pd.DataFrame.from_records(result.candidate_metrics).sort_values(
        ["well_name", "fwhm_m"]
    )
    event_metrics = pd.DataFrame.from_records(result.event_metrics).sort_values(
        ["well_name", "event_rank", "fwhm_m"]
    )
    event_rows = [
        {
            "well_name": well.well_name,
            "event_rank": event.event_rank,
            "event_top_m": event.top_m,
            "event_bottom_m": event.bottom_m,
            "event_width_m": event.width_m,
            "event_polarity": event.polarity,
            "event_peak_abs": event.peak_abs,
        }
        for well in result.wells
        for event in well.events
    ]
    event_windows = pd.DataFrame.from_records(event_rows).sort_values(["well_name", "event_rank"])
    summary = _summary_frame(candidate_metrics)

    candidate_path = output_dir / "candidate_metrics.csv"
    event_metrics_path = output_dir / "event_metrics.csv"
    event_windows_path = output_dir / "event_windows.csv"
    summary_path = output_dir / "fwhm_summary.csv"
    candidate_metrics.to_csv(candidate_path, index=False)
    event_metrics.to_csv(event_metrics_path, index=False)
    event_windows.to_csv(event_windows_path, index=False)
    summary.to_csv(summary_path, index=False)

    summary_figure = figures_dir / "fwhm_summary.png"
    _plot_summary(candidate_metrics, summary_figure)
    well_artifacts: dict[str, str] = {}
    well_figures: dict[str, Any] = {}
    for well in result.wells:
        safe_name = sanitize_filename(well.well_name)
        artifact_path = wells_dir / f"{safe_name}.npz"
        _write_well_artifact(well, artifact_path)
        well_artifacts[well.well_name] = repo_relative_path(artifact_path, root=repo_root)
        well_figure_dir = figures_dir / safe_name
        target_top = float(well.horizon_markers[0][0])
        target_bottom = float(well.horizon_markers[-1][0])
        overview_path = well_figure_dir / "target_interval_sweep.png"
        _plot_window_comparison(
            well,
            top_m=target_top,
            bottom_m=target_bottom,
            event_top_m=None,
            event_bottom_m=None,
            output_path=overview_path,
            title=f"{well.well_name} | complete target interval",
            reference_fwhm_m=result.policy.reference_fwhm_m,
        )
        event_paths: list[str] = []
        for event in well.events:
            context = max(
                result.policy.event_context_min_m,
                result.policy.event_context_width_multiple * event.width_m,
            )
            path = well_figure_dir / f"event_{event.event_rank:02d}_sweep.png"
            _plot_window_comparison(
                well,
                top_m=event.top_m - context,
                bottom_m=event.bottom_m + context,
                event_top_m=event.top_m,
                event_bottom_m=event.bottom_m,
                output_path=path,
                title=(
                    f"{well.well_name} | fixed real-seismic event {event.event_rank} | "
                    f"{event.top_m:g}–{event.bottom_m:g} m"
                ),
                reference_fwhm_m=result.policy.reference_fwhm_m,
            )
            event_paths.append(repo_relative_path(path, root=repo_root))
        well_figures[well.well_name] = {
            "target_interval": repo_relative_path(overview_path, root=repo_root),
            "events": event_paths,
        }

    manifest = {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "sample_domain": "depth",
        "sample_unit": "m",
        "depth_basis": "tvdss",
        "inputs": dict(inputs),
        "horizon_sources": horizon_sources,
        "resolved_config": dict(resolved_config),
        "well_names": [well.well_name for well in result.wells],
        "fwhm_values_m": list(result.policy.fwhm_values_m),
        "reference_fwhm_m": result.policy.reference_fwhm_m,
        "tables": {
            "candidate_metrics": repo_relative_path(candidate_path, root=repo_root),
            "event_metrics": repo_relative_path(event_metrics_path, root=repo_root),
            "event_windows": repo_relative_path(event_windows_path, root=repo_root),
            "fwhm_summary": repo_relative_path(summary_path, root=repo_root),
        },
        "figures": {
            "summary": repo_relative_path(summary_figure, root=repo_root),
            "wells": well_figures,
        },
        "well_artifacts": well_artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = [
    "BodyFwhmSweepPolicy",
    "BodyFwhmSweepResult",
    "CandidateSweepResult",
    "RealEventWindow",
    "SCHEMA_VERSION",
    "WellBodyFwhmSweep",
    "run_body_fwhm_sweep",
    "write_body_fwhm_sweep_artifacts",
    "write_well_waveform_qc",
]
