"""Waveform QC artifacts for trusted well body predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cup.seismic.viz import plot_well_waveform_qc
from cup.utils.io import repo_relative_path, sanitize_filename, write_json
from cup.well.real_field_control_qc import _waveform_objects
from wtie.processing import grid


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

    predictions: dict[str, dict[int, dict[str, list[float]]]] = {}
    with torch.no_grad():
        items = trainer.data.trusted_well_patches
        for start in range(0, len(items), trainer.config.batch_size):
            local = items[start : start + trainer.config.batch_size]
            batch = trainer.data.reader.batch(
                tuple(item.patch_key for item in local),
                center_visible=True,
                device=trainer.device,
            )
            body, synthetic, common = trainer._predict(model, batch)
            for row, item in enumerate(local):
                observed = batch.observed_seismic[row].cpu().numpy()
                observed_mask = common.observed_valid_mask[row].cpu().numpy()
                body_values = body[row].cpu().numpy()
                synthetic_values = synthetic[row].cpu().numpy()
                for sample_index in np.flatnonzero(item.target_mask & observed_mask):
                    index = int(sample_index)
                    sample = predictions.setdefault(item.well_name, {}).setdefault(
                        index,
                        {"body_log_ai": [], "synthetic": [], "observed": []},
                    )
                    sample["body_log_ai"].append(float(body_values[index]))
                    sample["synthetic"].append(float(synthetic_values[index]))
                    sample["observed"].append(float(observed[index]))

    qc_dir = Path(qc_dir)
    qc_dir.mkdir(parents=True, exist_ok=True)
    axis = np.asarray(trainer.data.reader.sample_axis.values, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    figures: list[str] = []
    for well_name in sorted(predictions):
        by_sample = predictions[well_name]
        indices = np.asarray(sorted(by_sample), dtype=np.int64)
        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        runs = np.column_stack((np.r_[0, breaks], np.r_[breaks, indices.size]))
        start, stop = max(runs, key=lambda item: int(item[1] - item[0]))
        indices = indices[start:stop]
        if indices.size < 8:
            raise ValueError(f"{well_name}: longest predicted well QC support run has fewer than eight samples.")
        target = trainer.data.well_targets[well_name]
        reference_log_ai = np.asarray(target.model_axis_target[indices], dtype=np.float64)
        predicted_log_ai = np.asarray(
            [np.mean(by_sample[index]["body_log_ai"]) for index in indices],
            dtype=np.float64,
        )
        observed = np.asarray(
            [np.mean(by_sample[index]["observed"]) for index in indices],
            dtype=np.float64,
        )
        synthetic = np.asarray(
            [np.mean(by_sample[index]["synthetic"]) for index in indices],
            dtype=np.float64,
        )
        if any(
            np.any(~np.isfinite(values))
            for values in (reference_log_ai, predicted_log_ai, observed, synthetic)
        ):
            raise ValueError(f"{well_name}: predicted well QC arrays contain non-finite values.")

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

        local_axis = axis[indices]
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
            title=f"GINN V2 predicted body forward QC | {well_name} | corr={correlation:.3f}",
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
                "corr": correlation,
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
        "dynamic_correlation_window_m": float(trainer.config.waveform_qc_dynamic_window_m),
        "figures": figures,
        "metrics": repo_relative_path(metrics_path, root=root),
    }
    write_json(qc_dir / "manifest.json", manifest)
    return manifest


__all__ = ["write_well_waveform_qc"]
