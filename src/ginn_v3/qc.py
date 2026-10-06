"""Well metrics and raw curves on the fixed Step-6 evaluation intervals."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cup.utils.io import sanitize_filename, write_json


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


def write_well_qc(model, data, physics, mean_wavelet_normalized,
                  reference_wavelet_amplitude, output_dir, device, *,
                  wavelet_cohort: str = "saved_training_trace_mean") -> dict[str, Any]:
    """Score all evaluation wells; use a single global learned wavelet for every well."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    model.to(resolved_device).eval()
    mean = torch.as_tensor(mean_wavelet_normalized, dtype=torch.float32, device=resolved_device)
    reference = torch.as_tensor(reference_wavelet_amplitude, dtype=torch.float32, device=resolved_device)
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
            fixed = physics.forward(prediction.log_ai, obs, reference)
        support = np.asarray(well.evaluation_mask, dtype=bool)
        if support.shape != axis.values.shape or np.count_nonzero(support) < 8:
            raise ValueError(f"{well.well_name}: Step-6 evaluation support is invalid.")
        available = (learned.valid_mask[0] & fixed.valid_mask[0] & obs.observed_mask[0]).cpu().numpy()
        if np.any(support & ~available) or np.any(support & ~well.valid_mask):
            raise ValueError(f"{well.well_name}: prediction does not cover the complete fixed Step-6 support.")
        predicted = prediction.log_ai[0].cpu().numpy().astype(np.float64)
        baseline = obs.initial_log_ai[0].cpu().numpy().astype(np.float64)
        observed = obs.observed_seismic[0].cpu().numpy().astype(np.float64)
        synthesized = learned.seismic[0].cpu().numpy().astype(np.float64) * norm.seismic_std + norm.seismic_mean
        reference_synthetic = fixed.seismic[0].cpu().numpy().astype(np.float64)
        truth = np.asarray(well.log_ai, dtype=np.float64)
        for values in (predicted, baseline, observed, synthesized, reference_synthetic, truth):
            if not np.all(np.isfinite(values[support])):
                raise ValueError(f"{well.well_name}: nonfinite value on fixed evaluation support.")
        selected = np.flatnonzero(support)
        record = {
            "well_name": well.well_name,
            "role": "training" if well.well_name in trusted else "held_out",
            "sample_domain": axis.domain, "sample_unit": axis.unit,
            "support_start_index": int(selected[0]), "support_stop_index": int(selected[-1]) + 1,
            "support_start": float(axis.values[selected[0]]), "support_stop": float(axis.values[selected[-1]]),
            "support_samples": int(selected.size),
            "log_ai_rmse": float(np.sqrt(np.mean((predicted[support] - truth[support]) ** 2))),
            "lfm_log_ai_rmse": float(np.sqrt(np.mean((baseline[support] - truth[support]) ** 2))),
            "learned_global_correlation": _correlation(observed[support], synthesized[support]),
            "reference_correlation": _correlation(observed[support], reference_synthetic[support]),
            "prediction_tv": _roughness(predicted[support])["tv"],
            "truth_tv": _roughness(truth[support])["tv"],
            "prediction_turns": _roughness(predicted[support])["turns"],
            "truth_turns": _roughness(truth[support])["turns"],
        }
        records.append(record)
        folder = output / sanitize_filename(well.well_name)
        folder.mkdir(parents=True, exist_ok=True)
        columns = np.column_stack((axis.values[support], truth[support], baseline[support], predicted[support],
                                   observed[support], synthesized[support], reference_synthetic[support]))
        np.savetxt(folder / "curves.csv", columns, delimiter=",", comments="",
                   header="sample,truth_log_ai,lfm_log_ai,prediction_log_ai,observed_raw,learned_synthetic_raw,reference_synthetic_raw")
        figure, panels = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        x = axis.values[support]
        panels[0].plot(x, truth[support], label="Well", color="black", linewidth=1)
        panels[0].plot(x, baseline[support], label="LFM", linewidth=1)
        panels[0].plot(x, predicted[support], label="PIAI", marker=".", markersize=2, linewidth=1)
        panels[0].set_ylabel("log AI")
        panels[0].legend()
        panels[0].set_title(f"{well.well_name} | {record['role']} | fixed Step-6 support")
        panels[1].plot(x, (observed[support] - norm.seismic_mean) / norm.seismic_std, label="Observed", color="black", linewidth=1)
        panels[1].plot(x, (synthesized[support] - norm.seismic_mean) / norm.seismic_std, label="Learned global wavelet", linewidth=1)
        panels[1].plot(x, (reference_synthetic[support] - norm.seismic_mean) / norm.seismic_std, label="Reference wavelet", linewidth=1, alpha=.65)
        panels[1].set_ylabel("Shared seismic scale")
        panels[1].set_xlabel(f"{axis.domain} ({axis.unit})")
        panels[1].legend()
        figure.tight_layout()
        figure.savefig(folder / "waveform_qc.png", dpi=150)
        plt.close(figure)
    if not records:
        raise ValueError("Well QC requires at least one evaluation well.")
    with (output / "well_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = {"schema": "ginn_v3_well_qc_v1", "wavelet_cohort": wavelet_cohort,
               "support_source": "Step-6 fixed intervals", "wells": records}
    write_json(output / "metrics.json", metrics)
    return metrics
