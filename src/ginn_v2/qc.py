"""Waveform QC artifacts for trusted well body predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cup.seismic.geometry import SampleAxis
from cup.seismic.viz import plot_well_waveform_qc
from cup.utils.io import repo_relative_path, sanitize_filename, write_json
from cup.well.real_field_control_qc import _waveform_objects
from ginn_v2.contracts import CommonObservationBatch
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


__all__ = ["write_well_waveform_qc"]
