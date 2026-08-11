"""Depth-domain forward QC for canonical real-field well controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cup.physics.numpy_backend import forward_depth, reflectivity_from_log_ai
from cup.seismic.target_zone import TargetZone
from cup.seismic.viz import plot_well_waveform_qc
from cup.utils.io import repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.well.real_field_controls import WellControl, WellControlSet
from cup.well.scale_separation import gaussian_smooth_finite_runs_numpy
from wtie.optimize.similarity import normalized_xcorr
from wtie.processing import grid


QC_SCHEMA_VERSION = "real_field_well_control_qc_v1"


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    return [tuple(item) for item in np.flatnonzero(padded[1:] != padded[:-1]).reshape((-1, 2))]


def _safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 3 or np.std(a[valid]) <= 0.0 or np.std(b[valid]) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _load_depth_forward_inputs(
    run_dir: Path,
    *,
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, float, float, Path]:
    path = run_dir / "forward_model_inputs.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema") != "forward_model_inputs_v3"
        or payload.get("sample_domain") != "depth"
        or payload.get("depth_basis") != "tvdss"
    ):
        raise ValueError("Step 6 depth QC requires TVDSS forward_model_inputs_v3.")
    wavelet_path = resolve_relative_path(str(payload["wavelet"]["path"]), root=repo_root)
    if not wavelet_path.is_file():
        raise FileNotFoundError(wavelet_path)
    frame = pd.read_csv(wavelet_path)
    if set(frame.columns) != {"time_s", "amplitude"}:
        raise ValueError(f"Unexpected wavelet columns: {wavelet_path}")
    time_s = frame["time_s"].to_numpy(dtype=np.float64)
    amplitude = frame["amplitude"].to_numpy(dtype=np.float64)
    relation = dict(payload["ai_velocity_relation"])
    if relation.get("formula") != "AI = a * Vp + b":
        raise ValueError("Unsupported AI-Vp relation in depth forward inputs.")
    a = float(relation["a"])
    b = float(relation["b"])
    if not np.isfinite(a) or a <= 0.0 or not np.isfinite(b):
        raise ValueError("AI-Vp relation coefficients are invalid.")
    return time_s, amplitude, a, b, path


def _forward_finite_runs(
    log_ai: np.ndarray,
    depth_m: np.ndarray,
    *,
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    relation_a: float,
    relation_b: float,
) -> np.ndarray:
    values = np.asarray(log_ai, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for start, stop in _finite_runs(np.isfinite(values)):
        if stop - start < 2:
            continue
        local_log_ai = values[start:stop]
        velocity = (np.exp(local_log_ai) - relation_b) / relation_a
        if np.any(~np.isfinite(velocity)) or np.any(velocity <= 0.0):
            raise ValueError("AI-Vp relation produced invalid velocity in a well-control run.")
        output[start:stop] = forward_depth(
            local_log_ai,
            velocity,
            depth[start:stop],
            wavelet_time_s,
            wavelet_amplitude,
        )
    return output


def _sample_seismic_along_control(control: WellControl, survey: Any) -> np.ndarray:
    """Bilinearly sample one seismic value at every well-path/sample intersection."""

    needed: set[tuple[int, int]] = set()
    plans: dict[int, list[tuple[tuple[int, int], float]]] = {}
    for sample_index in np.flatnonzero(
        np.isfinite(control.inline_by_sample) & np.isfinite(control.xline_by_sample)
    ):
        i_float, j_float = survey.line_geometry.line_to_index(
            float(control.inline_by_sample[sample_index]),
            float(control.xline_by_sample[sample_index]),
        )
        i0, i1 = int(np.floor(i_float)), int(np.ceil(i_float))
        j0, j1 = int(np.floor(j_float)), int(np.ceil(j_float))
        wi, wj = float(i_float - i0), float(j_float - j0)
        local: dict[tuple[int, int], float] = {}
        for key, weight in (
            ((i0, j0), (1.0 - wi) * (1.0 - wj)),
            ((i0, j1), (1.0 - wi) * wj),
            ((i1, j0), wi * (1.0 - wj)),
            ((i1, j1), wi * wj),
        ):
            if weight <= 0.0:
                continue
            if survey.trace_flat_index(*key) < 0:
                raise ValueError(f"{control.well_name}: trajectory intersects a missing seismic trace.")
            local[key] = local.get(key, 0.0) + weight
            needed.add(key)
        plans[int(sample_index)] = sorted(local.items())
    if not needed:
        raise ValueError(f"{control.well_name}: no valid seismic sampling positions.")

    traces = survey.read_traces_at_indices(sorted(needed), domain=control.sample_axis.domain)
    output = np.full(control.sample_axis.values.shape, np.nan, dtype=np.float64)
    for sample_index, weighted_indices in plans.items():
        value = 0.0
        for key, weight in weighted_indices:
            trace = traces[key]
            if not np.array_equal(
                np.asarray(trace.basis, dtype=np.float64),
                control.sample_axis.values,
            ):
                raise ValueError("Survey trace axis differs from the canonical well-control axis.")
            value += weight * float(trace.values[sample_index])
        output[sample_index] = value
    return output


def _horizon_markers(control: WellControl, target_zone: TargetZone) -> list[tuple[float, str]]:
    axis = control.sample_axis.values
    position_valid = np.isfinite(control.inline_by_sample) & np.isfinite(control.xline_by_sample)
    markers: list[tuple[float, str]] = []
    for name in target_zone.horizon_names:
        local_horizon = np.full(axis.shape, np.nan, dtype=np.float64)
        for index in np.flatnonzero(position_valid):
            local_horizon[index] = target_zone.get_horizon_interpretation_at_location(
                name,
                float(control.inline_by_sample[index]),
                float(control.xline_by_sample[index]),
            )
        valid = np.isfinite(local_horizon)
        if not np.any(valid):
            raise ValueError(f"{control.well_name}: horizon {name!r} has no well-path support.")
        candidates = np.flatnonzero(valid)
        selected = int(candidates[np.argmin(np.abs(axis[candidates] - local_horizon[candidates]))])
        markers.append((float(axis[selected]), name))
    if any(markers[index + 1][0] <= markers[index][0] for index in range(len(markers) - 1)):
        raise ValueError(f"{control.well_name}: target horizons are not ordered along the well path.")
    return markers


def _target_support_slice(
    axis: np.ndarray,
    valid: np.ndarray,
    markers: list[tuple[float, str]],
) -> tuple[slice, float]:
    target = (axis >= markers[0][0]) & (axis <= markers[-1][0])
    target_count = int(np.count_nonzero(target))
    runs = _finite_runs(valid & target)
    if not runs:
        raise ValueError("No common full/body/seismic support inside the target interval.")
    start, stop = max(runs, key=lambda item: item[1] - item[0])
    if stop - start < 8:
        raise ValueError("Fewer than eight common samples inside the target interval.")
    return slice(start, stop), float((stop - start) / target_count)


def _dynamic_xcorr(
    real: grid.Seismic,
    synthetic: grid.Seismic,
    *,
    window_axis_units: float,
) -> grid.DynamicXCorr:
    step = float(real.sampling_rate)
    half = max(2, int(round(float(window_axis_units) / step)) // 2)
    first = np.pad(np.asarray(real.values, dtype=np.float64), half, mode="reflect")
    second = np.pad(np.asarray(synthetic.values, dtype=np.float64), half, mode="reflect")
    rows = []
    for index in range(real.size):
        rows.append(
            normalized_xcorr(
                first[index : index + 2 * half],
                second[index : index + 2 * half],
            )
        )
    return grid.DynamicXCorr(
        np.asarray(rows, dtype=np.float64),
        np.asarray(real.basis, dtype=np.float64),
        "tvdss",
        name="Local lag [m]",
    )


def _waveform_objects(
    *,
    axis: np.ndarray,
    log_ai: np.ndarray,
    synthetic: np.ndarray,
    real: np.ndarray,
    dynamic_window_m: float,
    name: str,
) -> tuple[grid.Log, grid.Log, grid.Seismic, grid.Seismic, grid.XCorr, grid.DynamicXCorr]:
    linear_ai = grid.Log(
        np.exp(log_ai),
        axis,
        "tvdss",
        name=name,
        unit="m/s*g/cm3",
    )
    reflectivity_values = np.r_[0.0, reflectivity_from_log_ai(log_ai)]
    reflectivity = grid.Log(
        reflectivity_values,
        axis,
        "tvdss",
        name="Reflectivity",
    )
    synthetic_trace = grid.Seismic(synthetic, axis, "tvdss", name="Synthetic")
    real_trace = grid.Seismic(real, axis, "tvdss", name="Seismic")
    xcorr_values = normalized_xcorr(real, synthetic)
    lags = float(axis[1] - axis[0]) * np.arange(-(axis.size - 1), axis.size)
    xcorr = grid.XCorr(xcorr_values, lags, "zlag", name="XCorr")
    dynamic = _dynamic_xcorr(
        real_trace,
        synthetic_trace,
        window_axis_units=dynamic_window_m,
    )
    return linear_ai, reflectivity, synthetic_trace, real_trace, xcorr, dynamic


def _event_windows(
    seismic: np.ndarray,
    *,
    threshold_fraction: float,
    maximum: int,
) -> list[tuple[int, int]]:
    values = np.asarray(seismic, dtype=np.float64)
    centered = values - np.median(values)
    sign = np.sign(centered)
    for index in range(1, sign.size):
        if sign[index] == 0.0:
            sign[index] = sign[index - 1]
    if sign[0] == 0.0:
        nonzero = np.flatnonzero(sign)
        sign[: nonzero[0] if nonzero.size else sign.size] = sign[nonzero[0]] if nonzero.size else 1.0
    changes = np.r_[True, sign[1:] != sign[:-1], True]
    runs = np.flatnonzero(changes)
    threshold = float(threshold_fraction) * float(np.percentile(np.abs(centered), 95.0))
    scored: list[tuple[float, int, int]] = []
    for start, stop in zip(runs[:-1], runs[1:]):
        if stop - start < 2:
            continue
        score = float(np.max(np.abs(centered[start:stop])))
        if score >= threshold:
            scored.append((score, int(start), int(stop)))
    selected = sorted(scored, reverse=True)[: int(maximum)]
    return sorted((start, stop) for _score, start, stop in selected)


def _plot_event_comparison(
    *,
    output_path: Path,
    well_name: str,
    axis: np.ndarray,
    full_log_ai: np.ndarray,
    body_log_ai: np.ndarray,
    real: np.ndarray,
    full_synthetic: np.ndarray,
    body_synthetic: np.ndarray,
    threshold_fraction: float,
    maximum_events: int,
) -> int:
    events = _event_windows(
        real,
        threshold_fraction=threshold_fraction,
        maximum=maximum_events,
    )
    if not events:
        raise ValueError(f"{well_name}: no target-interval waveform events passed the threshold.")
    fig, axes = plt.subplots(
        len(events),
        3,
        figsize=(12.0, max(3.2, 2.6 * len(events))),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (event_start, event_stop) in enumerate(events):
        width = event_stop - event_start
        pad = max(3, width)
        start = max(0, event_start - pad)
        stop = min(axis.size, event_stop + pad)
        local = slice(start, stop)
        local_axis = axis[local]

        axes[row, 0].plot(real[local], local_axis, color="black", lw=1.5)
        axes[row, 0].axhspan(axis[event_start], axis[event_stop - 1], color="tab:blue", alpha=0.12)
        axes[row, 0].set_title("Real seismic" if row == 0 else "")

        axes[row, 1].plot(full_log_ai[local], local_axis, color="black", lw=1.1, label="full")
        axes[row, 1].plot(body_log_ai[local], local_axis, color="tab:red", lw=1.5, label="15 m body")
        axes[row, 1].set_title("log-AI" if row == 0 else "")
        if row == 0:
            axes[row, 1].legend(fontsize=8)

        axes[row, 2].plot(real[local], local_axis, color="black", lw=1.5, label="real")
        axes[row, 2].plot(full_synthetic[local], local_axis, color="tab:blue", lw=1.2, label="full forward")
        axes[row, 2].plot(body_synthetic[local], local_axis, color="tab:orange", lw=1.2, label="body forward")
        axes[row, 2].set_title("Shared-gain forward" if row == 0 else "")
        if row == 0:
            axes[row, 2].legend(fontsize=8)

        for column in range(3):
            axes[row, column].set_ylim(float(local_axis[-1]), float(local_axis[0]))
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_ylabel("TVDSS [m]" if column == 0 else "")
    fig.suptitle(f"{well_name} | target-interval full/body waveform slices")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return len(events)


def write_depth_well_control_qc(
    control_set: WellControlSet,
    *,
    survey: Any,
    target_zone: TargetZone,
    forward_inputs_run_dir: Path,
    output_dir: Path,
    repo_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Write three target-interval QC figures for every successful Step 6 well."""

    expected = {
        "body_smoothing_fwhm_m",
        "dynamic_correlation_window_m",
        "event_threshold_fraction",
        "max_event_windows_per_well",
    }
    if set(config) != expected:
        raise ValueError(f"real_field_well_controls_qc must contain exactly {sorted(expected)}.")
    if control_set.sample_domain != "depth" or control_set.depth_basis != "tvdss":
        raise ValueError("Current Step 6 forward QC requires depth/TVDSS well controls.")
    body_fwhm = float(config["body_smoothing_fwhm_m"])
    dynamic_window = float(config["dynamic_correlation_window_m"])
    threshold_fraction = float(config["event_threshold_fraction"])
    maximum_events = int(config["max_event_windows_per_well"])
    if body_fwhm <= 0.0 or dynamic_window <= 0.0 or not 0.0 < threshold_fraction <= 1.0 or maximum_events < 1:
        raise ValueError("Step 6 QC numeric settings are invalid.")
    wavelet_time, wavelet_amp, relation_a, relation_b, forward_inputs_path = (
        _load_depth_forward_inputs(forward_inputs_run_dir, repo_root=repo_root)
    )

    figures_root = output_dir / "figures"
    figures_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for control in control_set.controls:
        well_dir = figures_root / sanitize_filename(control.well_name)
        well_dir.mkdir()
        axis = control.sample_axis.values
        full_log_ai = np.asarray(control.log_ai.values, dtype=np.float64)
        body_log_ai = gaussian_smooth_finite_runs_numpy(
            full_log_ai,
            axis,
            fwhm_m=body_fwhm,
        )
        real = _sample_seismic_along_control(control, survey)
        full_forward = _forward_finite_runs(
            full_log_ai,
            axis,
            wavelet_time_s=wavelet_time,
            wavelet_amplitude=wavelet_amp,
            relation_a=relation_a,
            relation_b=relation_b,
        )
        body_forward = _forward_finite_runs(
            body_log_ai,
            axis,
            wavelet_time_s=wavelet_time,
            wavelet_amplitude=wavelet_amp,
            relation_a=relation_a,
            relation_b=relation_b,
        )
        markers = _horizon_markers(control, target_zone)
        common = (
            control.valid_mask
            & np.isfinite(real)
            & np.isfinite(full_forward)
            & np.isfinite(body_forward)
        )
        selected, support_fraction = _target_support_slice(axis, common, markers)
        local_axis = axis[selected]
        local_real = real[selected]
        real_std = float(np.std(local_real))
        if not np.isfinite(real_std) or real_std <= 0.0:
            raise ValueError(f"{control.well_name}: target seismic has zero variance.")
        local_real = (local_real - float(np.mean(local_real))) / real_std
        local_full_forward = full_forward[selected]
        local_body_forward = body_forward[selected]
        denominator = float(np.dot(local_full_forward, local_full_forward))
        signed_gain = (
            float(np.dot(local_real, local_full_forward) / denominator)
            if denominator > 0.0
            else 1.0
        )
        gain = abs(signed_gain)
        local_full_forward = gain * local_full_forward
        local_body_forward = gain * local_body_forward
        local_markers = [item for item in markers if local_axis[0] <= item[0] <= local_axis[-1]]

        full_objects = _waveform_objects(
            axis=local_axis,
            log_ai=full_log_ai[selected],
            synthetic=local_full_forward,
            real=local_real,
            dynamic_window_m=dynamic_window,
            name="Full AI",
        )
        full_corr = _safe_corr(local_real, local_full_forward)
        fig, _ = plot_well_waveform_qc(
            [full_objects[0]],
            full_objects[1],
            full_objects[2],
            full_objects[3],
            full_objects[4],
            full_objects[5],
            figsize=(13.0, 7.5),
            synthetic_ai=full_objects[0],
            title=f"Step 6 full forward QC | {control.well_name} | corr={full_corr:.3f}",
            horizon_markers=local_markers,
        )
        full_path = well_dir / "full_waveform_qc.png"
        fig.savefig(full_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        body_objects = _waveform_objects(
            axis=local_axis,
            log_ai=body_log_ai[selected],
            synthetic=local_body_forward,
            real=local_real,
            dynamic_window_m=dynamic_window,
            name=f"{body_fwhm:g} m body AI",
        )
        body_corr = _safe_corr(local_real, local_body_forward)
        fig, _ = plot_well_waveform_qc(
            [body_objects[0]],
            body_objects[1],
            body_objects[2],
            body_objects[3],
            body_objects[4],
            body_objects[5],
            figsize=(13.0, 7.5),
            synthetic_ai=body_objects[0],
            title=f"Step 6 {body_fwhm:g} m body forward QC | {control.well_name} | corr={body_corr:.3f}",
            horizon_markers=local_markers,
        )
        body_path = well_dir / "body_waveform_qc.png"
        fig.savefig(body_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        comparison_path = well_dir / "event_waveform_comparison.png"
        event_count = _plot_event_comparison(
            output_path=comparison_path,
            well_name=control.well_name,
            axis=local_axis,
            full_log_ai=full_log_ai[selected],
            body_log_ai=body_log_ai[selected],
            real=local_real,
            full_synthetic=local_full_forward,
            body_synthetic=local_body_forward,
            threshold_fraction=threshold_fraction,
            maximum_events=maximum_events,
        )
        rows.append(
            {
                "well_name": control.well_name,
                "target_support_fraction": support_fraction,
                "shared_forward_gain": gain,
                "signed_forward_gain": signed_gain,
                "full_forward_correlation": full_corr,
                "body_forward_correlation": body_corr,
                "event_window_count": event_count,
                "full_waveform_qc": repo_relative_path(full_path, root=repo_root),
                "body_waveform_qc": repo_relative_path(body_path, root=repo_root),
                "event_waveform_comparison": repo_relative_path(comparison_path, root=repo_root),
            }
        )

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame.from_records(rows).to_csv(metrics_path, index=False)
    manifest = {
        "schema_version": QC_SCHEMA_VERSION,
        "status": "ok",
        "sample_domain": "depth",
        "depth_basis": "tvdss",
        "config": dict(config),
        "forward_model_inputs": repo_relative_path(forward_inputs_path, root=repo_root),
        "well_count": len(rows),
        "outputs": {
            "metrics_csv": repo_relative_path(metrics_path, root=repo_root),
            "figures_dir": repo_relative_path(figures_root, root=repo_root),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = ["QC_SCHEMA_VERSION", "write_depth_well_control_qc"]
