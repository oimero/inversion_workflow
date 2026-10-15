"""Well and volume-section QC using the inversion's physical forward operator."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cup.physics.numpy_backend import reflectivity_from_log_ai
from cup.physics.support import longest_true_run
from cup.seismic.geometry import SampleAxis
from cup.seismic.viz import plot_seismic_closure_qc, plot_well_waveform_qc, waveform_qc_metrics
from cup.utils.io import sanitize_filename, write_json
from ginn_v3.infer import validate_volume_section_requests
from ginn_v3.types import TraceKey
from wtie.optimize.similarity import normalized_xcorr
from wtie.processing import grid


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    x, y = a - np.mean(a), b - np.mean(b)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator == 0.0 else float(np.dot(x, y) / denominator)


def _roughness(values: np.ndarray) -> dict[str, float | int]:
    difference = np.diff(values)
    signs = np.sign(difference[np.abs(difference) > 1e-4])
    return {"tv": float(np.abs(difference).sum()),
            "turns": int(np.count_nonzero(signs[1:] * signs[:-1] < 0.0))}


def write_wavelet_csv(path: Path, time_s: np.ndarray, amplitude_normalized: np.ndarray,
                      seismic_std: float) -> None:
    """Export learned kernel coefficients without a DC offset or energy rescaling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(time_s, dtype=np.float64)
    amplitude = np.asarray(amplitude_normalized, dtype=np.float64)
    if times.shape != amplitude.shape or not np.all(np.isfinite(amplitude)):
        raise ValueError("Learned wavelet and seconds axis must have matching finite samples.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "amplitude_normalized", "amplitude_raw"))
        writer.writerows(zip(times, amplitude, amplitude * seismic_std))


def _dynamic_xcorr(
    real: grid.Seismic,
    synthetic: grid.Seismic,
    *,
    window_axis_units: float,
) -> grid.DynamicXCorr:
    """Build the local correlation panel on a physical sample window."""

    step = float(real.sampling_rate)
    half = max(2, int(round(float(window_axis_units) / step)) // 2)
    first = np.pad(np.asarray(real.values, dtype=np.float64), half, mode="reflect")
    second = np.pad(np.asarray(synthetic.values, dtype=np.float64), half, mode="reflect")
    rows = [
        normalized_xcorr(
            first[index : index + 2 * half],
            second[index : index + 2 * half],
        )
        for index in range(real.size)
    ]
    return grid.DynamicXCorr(
        np.asarray(rows, dtype=np.float64),
        np.asarray(real.basis, dtype=np.float64),
        "twt" if real.is_twt else "tvdss",
        name="Local lag [s]" if real.is_twt else "Local lag [m]",
    )


def _waveform_objects(
    *,
    axis: np.ndarray,
    log_ai: np.ndarray,
    synthetic: np.ndarray,
    real: np.ndarray,
    dynamic_window_axis_units: float,
    basis_type: str,
    name: str,
) -> tuple[grid.Log, grid.Log, grid.Seismic, grid.Seismic, grid.XCorr, grid.DynamicXCorr]:
    """Create the same six-panel inputs used by the shared well-QC plot."""

    if basis_type not in {"twt", "tvdss"}:
        raise ValueError("GINN waveform QC requires a TWT or TVDSS basis.")
    axis_values = np.asarray(axis, dtype=np.float64)
    log_values = np.asarray(log_ai, dtype=np.float64)
    synthetic_values = np.asarray(synthetic, dtype=np.float64)
    real_values = np.asarray(real, dtype=np.float64)
    if axis_values.ndim != 1 or axis_values.size < 2:
        raise ValueError("GINN waveform QC requires at least two samples.")
    if not (log_values.shape == synthetic_values.shape == real_values.shape == axis_values.shape):
        raise ValueError("GINN waveform QC traces must share one sample axis.")
    if not np.all(np.isfinite(axis_values)) or np.any(np.diff(axis_values) <= 0.0):
        raise ValueError("GINN waveform QC axis must be finite and strictly increasing.")
    if not np.all(np.isfinite(log_values)):
        raise ValueError("GINN waveform QC impedance must be finite on the plotted support.")
    if not np.all(np.isfinite(synthetic_values)) or not np.all(np.isfinite(real_values)):
        raise ValueError("GINN waveform QC seismic must be finite on the plotted support.")
    if not np.isfinite(float(dynamic_window_axis_units)) or float(dynamic_window_axis_units) <= 0.0:
        raise ValueError("dynamic_window_axis_units must be finite and positive.")

    predicted_ai = grid.Log(
        np.exp(log_values),
        axis_values,
        basis_type,
        name=name,
        unit="m/s*g/cm3",
        allow_nan=False,
    )
    reflectivity = grid.Log(
        reflectivity_from_log_ai(log_values),
        0.5 * (axis_values[:-1] + axis_values[1:]),
        basis_type,
        name="Reflectivity",
        allow_nan=False,
    )
    synthetic_trace = grid.Seismic(
        synthetic_values, axis_values, basis_type, name="Synthetic", allow_nan=False
    )
    real_trace = grid.Seismic(
        real_values, axis_values, basis_type, name="Seismic", allow_nan=False
    )
    xcorr_values = normalized_xcorr(real_values, synthetic_values)
    lags = float(axis_values[1] - axis_values[0]) * np.arange(
        -(axis_values.size - 1), axis_values.size
    )
    xcorr = grid.XCorr(
        xcorr_values,
        lags,
        "tlag" if basis_type == "twt" else "zlag",
        name="XCorr",
    )
    dynamic = _dynamic_xcorr(
        real_trace,
        synthetic_trace,
        window_axis_units=float(dynamic_window_axis_units),
    )
    return predicted_ai, reflectivity, synthetic_trace, real_trace, xcorr, dynamic


def _write_waveform_qc_figure(
    folder: Path,
    *,
    axis: np.ndarray,
    predicted_log_ai: np.ndarray,
    reference_log_ai: np.ndarray,
    synthetic: np.ndarray,
    real: np.ndarray,
    sample_domain: str,
    well_name: str,
    correlation: float | None,
) -> tuple[Path, Path]:
    """Write an amplitude-preserving six-panel QC figure for one well."""

    basis_type = "twt" if str(sample_domain).casefold() == "time" else "tvdss"
    dynamic_window = 0.070 if basis_type == "twt" else 75.0
    predicted, reflectivity, synthetic_trace, real_trace, xcorr, dynamic = _waveform_objects(
        axis=axis,
        log_ai=predicted_log_ai,
        synthetic=synthetic,
        real=real,
        dynamic_window_axis_units=dynamic_window,
        basis_type=basis_type,
        name="GINN AI",
    )
    reference = grid.Log(
        np.exp(np.asarray(reference_log_ai, dtype=np.float64)),
        np.asarray(axis, dtype=np.float64),
        basis_type,
        name="Reference AI",
        unit="m/s*g/cm3",
        allow_nan=False,
    )
    figure, _axes = plot_well_waveform_qc(
        [predicted, reference],
        reflectivity,
        synthetic_trace,
        real_trace,
        xcorr,
        dynamic,
        figsize=(13.0, 7.5),
        synthetic_ai=predicted,
        reference_ai=reference,
        title=(
            f"GINN well waveform QC | {well_name}"
            if correlation is None
            else f"GINN well waveform QC | {well_name} | corr={correlation:.3f}"
        ),
    )
    folder.mkdir(parents=True, exist_ok=True)
    png_path = folder / "waveform_qc.png"
    svg_path = folder / "waveform_qc.svg"
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return png_path, svg_path


def write_well_qc(model, data, physics, mean_wavelet_normalized,
                  output_dir, device, *,
                  wavelet_cohort: str = "saved_training_trace_mean") -> dict[str, Any]:
    """Score all evaluation wells; use a single global learned wavelet for every well."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    mean = torch.as_tensor(mean_wavelet_normalized, dtype=torch.float32, device=resolved_device)
    axis = data.reader.sample_axis
    norm = data.reader.normalization
    write_wavelet_csv(output / "learned_wavelet.csv", physics.wavelet_time_s,
                      mean.detach().cpu().numpy(), norm.seismic_std)
    trusted = {well.well_name for well in data.train_wells}
    records = []
    for well in data.evaluation_wells:
        with torch.no_grad():
            batch = data.well_batch((well,), resolved_device)
            obs = batch.observations
            prediction = model(obs.features, obs.initial_log_ai)
            learned = physics.forward(prediction.log_ai, obs, mean)
        support = np.asarray(well.evaluation_mask, dtype=bool)
        if support.shape != axis.values.shape or np.count_nonzero(support) < 8:
            raise ValueError(f"{well.well_name}: Step-6 evaluation support is invalid.")
        available = (learned.valid_mask[0] & obs.observed_mask[0]).cpu().numpy()
        if np.any(support & ~well.valid_mask):
            raise ValueError(
                f"{well.well_name}: fixed Step-6 impedance support contains unsupported target samples."
            )
        effective_waveform = support & available
        effective_count = int(np.count_nonzero(effective_waveform))
        try:
            waveform_start, waveform_stop = longest_true_run(
                effective_waveform,
                minimum_samples=8,
            )
        except ValueError as exc:
            fixed_count = int(np.count_nonzero(support))
            raise ValueError(
                f"{well.well_name}: complete forward response covers only {effective_count} of "
                f"{fixed_count} fixed Step-6 samples; the learned kernel support cannot evaluate "
                "the fixed waveform interval."
            ) from exc
        waveform_support = np.zeros_like(support)
        waveform_support[waveform_start:waveform_stop] = True
        waveform_count = waveform_stop - waveform_start
        predicted = prediction.log_ai[0].cpu().numpy().astype(np.float64)
        baseline = obs.initial_log_ai[0].cpu().numpy().astype(np.float64)
        observed = obs.observed_seismic[0].cpu().numpy().astype(np.float64)
        synthesized = learned.seismic[0].cpu().numpy().astype(np.float64) * norm.seismic_std + norm.seismic_mean
        truth = np.asarray(well.log_ai, dtype=np.float64)
        for values in (predicted, baseline, truth):
            if not np.all(np.isfinite(values[support])):
                raise ValueError(f"{well.well_name}: nonfinite impedance value on fixed evaluation support.")
        for values in (observed, synthesized):
            if not np.all(np.isfinite(values[waveform_support])):
                raise ValueError(f"{well.well_name}: nonfinite waveform value on complete forward support.")
        selected = np.flatnonzero(support)
        selected_waveform = np.flatnonzero(waveform_support)
        record = {
            "well_name": well.well_name,
            "role": "training" if well.well_name in trusted else "held_out",
            "sample_domain": axis.domain, "sample_unit": axis.unit,
            "support_start_index": int(selected[0]), "support_stop_index": int(selected[-1]) + 1,
            "support_start": float(axis.values[selected[0]]), "support_stop": float(axis.values[selected[-1]]),
            "support_samples": int(selected.size),
            "waveform_support_start_index": int(selected_waveform[0]),
            "waveform_support_stop_index": int(selected_waveform[-1]) + 1,
            "waveform_support_start": float(axis.values[selected_waveform[0]]),
            "waveform_support_stop": float(axis.values[selected_waveform[-1]]),
            "waveform_support_samples": waveform_count,
            "log_ai_rmse": float(np.sqrt(np.mean((predicted[support] - truth[support]) ** 2))),
            "lfm_log_ai_rmse": float(np.sqrt(np.mean((baseline[support] - truth[support]) ** 2))),
            "learned_global_correlation": _correlation(observed[waveform_support], synthesized[waveform_support]),
            "prediction_tv": _roughness(predicted[support])["tv"],
            "truth_tv": _roughness(truth[support])["tv"],
            "prediction_turns": _roughness(predicted[support])["turns"],
            "truth_turns": _roughness(truth[support])["turns"],
        }
        records.append(record)
        folder = output / sanitize_filename(well.well_name)
        folder.mkdir(parents=True, exist_ok=True)
        observed_for_csv = np.where(waveform_support, observed, np.nan)
        synthesized_for_csv = np.where(waveform_support, synthesized, np.nan)
        waveform_axis = axis.values[waveform_support]
        waveform_predicted = predicted[waveform_support]
        waveform_reference = truth[waveform_support]
        waveform_synthetic = synthesized[waveform_support]
        waveform_observed = observed[waveform_support]
        png_path, svg_path = _write_waveform_qc_figure(
            folder,
            axis=waveform_axis,
            predicted_log_ai=waveform_predicted,
            reference_log_ai=waveform_reference,
            synthetic=waveform_synthetic,
            real=waveform_observed,
            sample_domain=axis.domain,
            well_name=well.well_name,
            correlation=record["learned_global_correlation"],
        )
        record["waveform_qc_png"] = str(png_path.relative_to(output))
        record["waveform_qc_svg"] = str(svg_path.relative_to(output))
        columns = np.column_stack((
            axis.values[support],
            truth[support],
            baseline[support],
            predicted[support],
            observed_for_csv[support],
            synthesized_for_csv[support],
        ))
        np.savetxt(folder / "curves.csv", columns, delimiter=",", comments="",
                   header=(
                       "sample,truth_log_ai,lfm_log_ai,prediction_log_ai,"
                       "observed_raw,learned_synthetic_raw"
                   ))
    if not records:
        raise ValueError("Well QC requires at least one evaluation well.")
    with (output / "well_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = {
        "schema": "ginn_v3_well_qc_v3",
        "wavelet_cohort": wavelet_cohort,
        "support_source": "Step-6 fixed intervals for impedance; complete forward response for waveform",
        "wells": records,
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def write_volume_closure_qc(
    model, reader, physics, mean_wavelet_normalized,
    *,
    log_ai_path: str | Path,
    valid_mask_path: str | Path,
    ilines: Sequence[float],
    xlines: Sequence[float],
    output_dir: str | Path,
    device: str | torch.device,
    inline_numbers: Sequence[float] = (),
    xline_numbers: Sequence[float] = (),
    batch_size: int = 128,
) -> dict[str, Any]:
    """Forward saved target predictions with replayed network context.

    Only the requested sections are replayed. Saved log-AI is authoritative
    inside the published target; network predictions supply the surrounding
    calculation support. The same fixed kinematics and one saved volume-mean
    kernel are used for every trace, without fitting an amplitude gain.
    """

    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("Closure QC batch_size must be a positive integer.")
    axis = reader.sample_axis
    geometry = reader.geometry
    inline_axis = np.asarray(ilines, dtype=np.float64)
    xline_axis = np.asarray(xlines, dtype=np.float64)
    requested = validate_volume_section_requests(
        ilines=inline_axis, xlines=xline_axis,
        inline_numbers=inline_numbers, xline_numbers=xline_numbers,
    )
    # An inference tile has local stored-array indices and global reader keys.
    source_indices = validate_volume_section_requests(
        ilines=geometry.inline_axis.values(), xlines=geometry.xline_axis.values(),
        inline_numbers=inline_axis, xline_numbers=xline_axis,
    )
    stored = np.load(log_ai_path, mmap_mode="r", allow_pickle=False)
    stored_valid = np.load(valid_mask_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (inline_axis.size, xline_axis.size, axis.values.size)
    if stored.shape != expected_shape or stored_valid.shape != expected_shape:
        raise ValueError("Saved closure volume and mask must match the explicit line and sample axes.")
    if stored_valid.dtype != np.bool_:
        raise TypeError("Saved closure volume mask must be boolean.")
    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    wavelet = torch.as_tensor(mean_wavelet_normalized, dtype=torch.float32, device=resolved_device)
    normalization = reader.normalization
    sections = []
    for direction, fixed_indices in requested.items():
        for fixed_index in fixed_indices:
            if direction == "inline":
                line = float(inline_axis[fixed_index])
                local_i = np.full(xline_axis.size, fixed_index, dtype=int)
                local_j = np.arange(xline_axis.size)
                x_m, y_m = geometry.trace_xy_grids(inline_axis[fixed_index:fixed_index + 1], xline_axis)
                lateral_lines = xline_axis
            else:
                line = float(xline_axis[fixed_index])
                local_i = np.arange(inline_axis.size)
                local_j = np.full(inline_axis.size, fixed_index, dtype=int)
                x_m, y_m = geometry.trace_xy_grids(inline_axis, xline_axis[fixed_index:fixed_index + 1])
                lateral_lines = inline_axis
            keys = tuple(
                TraceKey(source_indices["inline"][int(i)], source_indices["xline"][int(j)])
                for i, j in zip(local_i, local_j)
            )
            shape = (len(keys), axis.values.size)
            synthetic = np.full(shape, np.nan, dtype=np.float64)
            observed = np.full(shape, np.nan, dtype=np.float64)
            valid = np.zeros(shape, dtype=bool)
            target = np.zeros(shape, dtype=bool)
            for start in range(0, len(keys), batch_size):
                stop = min(start + batch_size, len(keys))
                saved_values = np.asarray(stored[local_i[start:stop], local_j[start:stop]], dtype=np.float32)
                saved_mask = np.asarray(stored_valid[local_i[start:stop], local_j[start:stop]], dtype=bool)
                with torch.no_grad():
                    batch = reader.batch(keys[start:stop], resolved_device)
                    if not np.array_equal(saved_mask, batch.output_mask.cpu().numpy()):
                        raise ValueError("Saved prediction support differs from the current target support.")
                    if np.any(saved_mask & ~np.isfinite(saved_values)):
                        raise ValueError("Saved log-AI is nonfinite inside its prediction mask.")
                    prediction = model(batch.features, batch.initial_log_ai)
                    full_log_ai = prediction.log_ai.clone()
                    selected = torch.as_tensor(saved_mask, device=resolved_device)
                    saved_tensor = torch.as_tensor(saved_values, device=resolved_device)
                    full_log_ai[selected] = saved_tensor[selected]
                    response = physics.forward(full_log_ai, batch, wavelet)
                raw_forward = (
                    response.seismic.cpu().numpy().astype(np.float64) * normalization.seismic_std
                    + normalization.seismic_mean
                )
                raw_observed = batch.observed_seismic.cpu().numpy().astype(np.float64)
                observed_mask = batch.observed_mask.cpu().numpy()
                response_mask = response.valid_mask.cpu().numpy()
                evaluated = saved_mask & observed_mask & response_mask
                if np.any(evaluated & (~np.isfinite(raw_forward) | ~np.isfinite(raw_observed))):
                    raise ValueError("Closure waveforms are nonfinite on evaluation support.")
                synthetic[start:stop] = np.where(evaluated, raw_forward, np.nan)
                observed[start:stop] = np.where(observed_mask, raw_observed, np.nan)
                valid[start:stop] = evaluated
                target[start:stop] = saved_mask
            if not np.any(valid):
                raise ValueError(f"{direction} {line:g} has no valid target waveform response.")
            distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(x_m.ravel()), np.diff(y_m.ravel())))]
            sections.append({
                "direction": direction, "line": line, "index": int(fixed_index),
                "distance": distance, "lateral_lines": lateral_lines,
                "synthetic": synthetic, "observed": observed, "valid": valid, "target": target,
            })

    target_samples = np.concatenate([np.flatnonzero(np.any(item["target"], axis=0)) for item in sections])
    sample_start, sample_stop = int(target_samples.min()), int(target_samples.max()) + 1
    display_axis = SampleAxis(
        values=axis.values[sample_start:sample_stop], domain=axis.domain,
        unit=axis.unit, depth_basis=axis.depth_basis,
    )
    amplitudes = np.concatenate([
        np.abs(item[key][item["valid"]])
        for item in sections for key in ("synthetic", "observed")
    ])
    amplitude_limit = max(float(np.percentile(amplitudes, 99.0)), float(np.finfo(np.float32).tiny))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in sections:
        direction, line = item["direction"], item["line"]
        synthetic = item["synthetic"][:, sample_start:sample_stop]
        observed = item["observed"][:, sample_start:sample_stop]
        valid = item["valid"][:, sample_start:sample_stop]
        target = item["target"][:, sample_start:sample_stop]
        residual = np.where(valid, observed - synthetic, np.nan)
        metrics = waveform_qc_metrics(observed, synthetic, valid)
        figure, _ = plot_seismic_closure_qc(
            synthetic=synthetic, observed=observed, valid_mask=valid,
            lateral_coordinates_m=item["distance"], sample_axis=display_axis,
            amplitude_limit=amplitude_limit, title=f"{direction.title()} {line:g}",
            correlation=metrics["corr"],
        )
        stem = sanitize_filename(f"{direction}_{line:g}_physical_closure")
        png_path, svg_path, data_path = (output / f"{stem}.{suffix}" for suffix in ("png", "svg", "npz"))
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        figure.savefig(svg_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        np.savez_compressed(
            data_path, sample_values=display_axis.values,
            sample_domain=np.asarray(axis.domain), sample_unit=np.asarray(axis.unit),
            depth_basis=np.asarray("" if axis.depth_basis is None else axis.depth_basis),
            lateral_distance_m=item["distance"], lateral_line_numbers=item["lateral_lines"],
            synthetic_raw=synthetic, observed_raw=observed, residual_raw=residual,
            valid_mask=valid, target_mask=target,
        )
        records.append({
            "direction": direction, "line": line, "index": item["index"],
            **metrics, "target_samples": int(np.count_nonzero(target)),
            "sample_start": float(display_axis.values[0]), "sample_stop": float(display_axis.values[-1]),
            "lateral_distance_max_m": float(item["distance"][-1]),
            "png": png_path.name, "svg": svg_path.name, "data_npz": data_path.name,
        })
    manifest = {
        "status": "ok", "sample_domain": axis.domain, "sample_unit": axis.unit,
        "wavelet_cohort": "predicted_volume_trace_mean",
        "target_impedance_source": "saved_volume_prediction",
        "context_source": "same_checkpoint_full_trace_network_prediction",
        "physical_amplitude": "normalized forward * seismic_std + seismic_mean",
        "residual_definition": "observed_raw - synthetic_raw on target and complete-response support",
        "color_scale": {"quantity": "raw_seismic_amplitude", "vmin": -amplitude_limit, "vmax": amplitude_limit,
                        "method": "shared 99th percentile of absolute target observed and forward amplitudes"},
        "sections": records,
    }
    write_json(output / "manifest.json", manifest)
    return manifest
