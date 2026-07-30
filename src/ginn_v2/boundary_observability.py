"""Clean-synthetic boundary observability audit for Structured GINN V2.

The audit stratifies canonical HSMM truth boundaries by wavelet tuning scale
and a deterministic local forward counterfactual.  Its labels are deliberately
scoped to clean model-consistent synthetic data; they are not claims about
real-field seismic resolution.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cup.physics.numpy_backend import velocity_from_ai
from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from ginn_v2.data import ParentSplitManifest
from ginn_v2.forward import ForwardContext, forward_numpy
from ginn_v2.hsmm import HsmmPrior, exact_hsmm
from ginn_v2.lateral import (
    LateralPatchDataModule,
    LateralStructuredModel,
    center_trace_batch,
    lateral_patch_to_torch,
)
from ginn_v2.lateral_training import Stage1Step3Config
from ginn_v2.oracle import (
    forward_context_from_sample,
    project_log_ai_to_model_grid,
)
from ginn_v2.runtime import configure_training_logger, resolve_device
from ginn_v2.structure import truth_hsmm_segments
from ginn_v2.truth import StructuredTruthAdapter
from ginn_v2.undersegmentation import (
    _AuditJournal,
    _file_sha256,
    _fingerprint,
    _stratified_tuning_parents,
)


BOUNDARY_OBSERVABILITY_AUDIT_SCHEMA = (
    "structured_ginn_v2_boundary_observability_audit_v1"
)
BOUNDARY_OBSERVABILITY_PLAN_SCHEMA = (
    "structured_ginn_v2_boundary_observability_plan_v1"
)
BOUNDARY_OBSERVABILITY_PROGRESS_SCHEMA = (
    "structured_ginn_v2_boundary_observability_progress_v1"
)
OBSERVABILITY_CLASSES = (
    "clean_forward_sensitive_isolated",
    "clean_forward_sensitive_tuned",
    "clean_forward_weak",
)


@dataclass(frozen=True)
class BoundaryObservabilityAuditOptions:
    maximum_parents: int = 16
    condition: str = "clean"
    boundary_tolerance_model_samples: tuple[int, ...] = (1, 2, 4, 8)
    counterfactual_half_width_model_samples: int = 1
    response_window_tuning_multiples: float = 1.0
    clean_sensitivity_threshold: float = 0.05
    progress_every_parents: int = 1
    resume: bool = False

    def __post_init__(self) -> None:
        if int(self.maximum_parents) <= 0:
            raise ValueError("maximum_parents must be positive.")
        if self.condition not in {"clean", "dirty"}:
            raise ValueError("condition must be clean or dirty.")
        tolerances = tuple(
            int(value) for value in self.boundary_tolerance_model_samples
        )
        if (
            not tolerances
            or any(value <= 0 for value in tolerances)
            or len(set(tolerances)) != len(tolerances)
        ):
            raise ValueError(
                "boundary_tolerance_model_samples must be unique and positive."
            )
        if int(self.counterfactual_half_width_model_samples) <= 0:
            raise ValueError(
                "counterfactual_half_width_model_samples must be positive."
            )
        if (
            not np.isfinite(self.response_window_tuning_multiples)
            or self.response_window_tuning_multiples <= 0.0
        ):
            raise ValueError(
                "response_window_tuning_multiples must be positive."
            )
        if (
            not np.isfinite(self.clean_sensitivity_threshold)
            or self.clean_sensitivity_threshold <= 0.0
        ):
            raise ValueError("clean_sensitivity_threshold must be positive.")
        if int(self.progress_every_parents) <= 0:
            raise ValueError("progress_every_parents must be positive.")
        object.__setattr__(
            self,
            "boundary_tolerance_model_samples",
            tolerances,
        )


def wavelet_power_centroid_hz(context: ForwardContext) -> float:
    """Return the positive-frequency power centroid of the frozen wavelet."""
    amplitude = np.asarray(context.wavelet_amplitude, dtype=np.float64)
    dt = float(context.wavelet_time_s[1] - context.wavelet_time_s[0])
    frequency = np.fft.rfftfreq(amplitude.size, dt)
    power = np.square(np.abs(np.fft.rfft(amplitude)))
    positive = frequency > 0.0
    denominator = float(np.sum(power[positive]))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("wavelet has no positive-frequency power.")
    centroid = float(
        np.sum(frequency[positive] * power[positive]) / denominator
    )
    if not np.isfinite(centroid) or centroid <= 0.0:
        raise ValueError("wavelet power centroid is invalid.")
    return centroid


def tuning_scale_for_sample(
    context: ForwardContext,
    log_ai: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float | None, float]:
    """Return quarter-wavelength tuning scale in the sample-axis unit."""
    centroid = wavelet_power_centroid_hz(context)
    mask = np.asarray(valid, dtype=bool)
    values = np.asarray(log_ai, dtype=np.float64)
    if values.shape != mask.shape or not np.any(mask):
        raise ValueError("tuning-scale log AI and validity masks differ.")
    if context.sample_domain == "time":
        return 1.0 / (2.0 * centroid), None, centroid
    relation = context.ai_velocity_relation
    if relation is None:
        raise ValueError("depth tuning scale requires an AI--Vp relation.")
    velocity = velocity_from_ai(
        np.exp(values[mask]),
        a=relation.a,
        b=relation.b,
    )
    median_vp = float(np.median(velocity))
    if not np.isfinite(median_vp) or median_vp <= 0.0:
        raise ValueError("depth tuning scale produced invalid median velocity.")
    return median_vp / (4.0 * centroid), median_vp, centroid


def bridge_boundary_counterfactual(
    log_ai: np.ndarray,
    *,
    boundary_index: int,
    zone_start: int,
    zone_stop: int,
    half_width_samples: int,
) -> np.ndarray:
    """Remove one local discontinuity with a continuous linear bridge."""
    values = np.asarray(log_ai, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("counterfactual log AI must be one-dimensional.")
    boundary_index = int(boundary_index)
    zone_start = int(zone_start)
    zone_stop = int(zone_stop)
    half_width_samples = int(half_width_samples)
    if not 0 <= zone_start < boundary_index < zone_stop <= values.size:
        raise ValueError("counterfactual boundary is outside its zone.")
    if half_width_samples <= 0:
        raise ValueError("counterfactual half width must be positive.")
    replace_start = max(zone_start + 1, boundary_index - half_width_samples)
    replace_stop = min(zone_stop - 1, boundary_index + half_width_samples)
    if replace_start >= replace_stop:
        raise ValueError("counterfactual bridge has no replaceable support.")
    left_anchor = replace_start - 1
    right_anchor = replace_stop
    if not (
        np.isfinite(values[left_anchor])
        and np.isfinite(values[right_anchor])
    ):
        raise ValueError("counterfactual bridge anchors are non-finite.")
    output = np.array(values, copy=True)
    count = replace_stop - replace_start
    fractions = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
    output[replace_start:replace_stop] = (
        (1.0 - fractions) * values[left_anchor]
        + fractions * values[right_anchor]
    )
    return output


def _matched_truth_indices(
    predicted: Sequence[float],
    truth: Sequence[float],
    *,
    tolerance: float,
) -> set[int]:
    candidates = [
        (abs(float(predicted_value) - float(truth_value)), p_index, t_index)
        for p_index, predicted_value in enumerate(predicted)
        for t_index, truth_value in enumerate(truth)
        if abs(float(predicted_value) - float(truth_value))
        <= float(tolerance) + 1.0e-12
    ]
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    for _, predicted_index, truth_index in sorted(candidates):
        if (
            predicted_index in used_predicted
            or truth_index in used_truth
        ):
            continue
        used_predicted.add(predicted_index)
        used_truth.add(truth_index)
    return used_truth


def _classify(
    *,
    sensitivity: float,
    nearest_boundary_distance: float | None,
    tuning_scale: float,
    threshold: float,
) -> str:
    if sensitivity < threshold:
        return "clean_forward_weak"
    if (
        nearest_boundary_distance is not None
        and nearest_boundary_distance < tuning_scale
    ):
        return "clean_forward_sensitive_tuned"
    return "clean_forward_sensitive_isolated"


def _sample_rows(
    parent: Any,
    center: Any,
    batch_index: int,
    predicted_segments: Sequence[Any],
    options: BoundaryObservabilityAuditOptions,
) -> list[dict[str, Any]]:
    key_parts = str(center.sample_keys[batch_index]).rsplit("|", 2)
    if len(key_parts) != 3:
        raise ValueError("center sample key is not parent|lateral|zone.")
    parent_id, lateral_text, zone_id = key_parts
    if parent_id != parent.identity.realization_id:
        raise ValueError("center sample key parent differs from loaded parent.")
    lateral_index = int(lateral_text)
    sample = StructuredTruthAdapter.from_structured_parent(
        parent,
        zone_id=zone_id,
        lateral_index=lateral_index,
    )
    context = forward_context_from_sample(sample)
    zone = torch.nonzero(
        center.zone_valid[batch_index],
        as_tuple=False,
    ).flatten()
    zone_start = int(zone[0].item())
    zone_stop = int(zone[-1].item()) + 1
    truth_segments = truth_hsmm_segments(center, batch_index)
    boundary_indices = [
        zone_start + int(segment.start) for segment in truth_segments[1:]
    ]
    if not boundary_indices:
        return []
    axis = np.asarray(sample.latent.latent_axis.coordinates, dtype=np.float64)
    model_axis = np.asarray(
        sample.observed.sample_axis.coordinates,
        dtype=np.float64,
    )
    tuning_scale, median_vp, centroid = tuning_scale_for_sample(
        context,
        sample.latent.log_ai_highres_truth,
        sample.zone.zone_valid,
    )
    highres_interval = float(sample.latent.latent_axis.sample_interval)
    model_interval = float(sample.observed.sample_axis.sample_interval)
    half_width = int(
        round(
            options.counterfactual_half_width_model_samples
            * model_interval
            / highres_interval
        )
    )
    counterfactuals = np.stack(
        [
            bridge_boundary_counterfactual(
                sample.latent.log_ai_highres_truth,
                boundary_index=index,
                zone_start=zone_start,
                zone_stop=zone_stop,
                half_width_samples=half_width,
            )
            for index in boundary_indices
        ],
        axis=0,
    )
    projection = project_log_ai_to_model_grid(
        counterfactuals,
        sample.latent.latent_axis,
        sample.observed.sample_axis,
    )
    counterfactual_seismic = np.asarray(
        forward_numpy(context, projection.model_log_ai),
        dtype=np.float64,
    )
    truth_seismic = np.asarray(
        sample.observed.model_consistent_seismic,
        dtype=np.float64,
    )
    predicted_coordinates = [
        float(axis[zone_start + int(segment.start)])
        for segment in predicted_segments[1:]
    ]
    truth_coordinates = [float(axis[index]) for index in boundary_indices]
    tolerance_values = [
        int(value) * model_interval
        for value in options.boundary_tolerance_model_samples
    ]
    matches = {
        tolerance: _matched_truth_indices(
            predicted_coordinates,
            truth_coordinates,
            tolerance=tolerance,
        )
        for tolerance in tolerance_values
    }
    rows: list[dict[str, Any]] = []
    for boundary_order, boundary_index in enumerate(boundary_indices):
        coordinate = float(axis[boundary_index])
        other = [
            abs(coordinate - value)
            for index, value in enumerate(truth_coordinates)
            if index != boundary_order
        ]
        nearest = min(other) if other else None
        response_half_width = (
            options.response_window_tuning_multiples * tuning_scale
        )
        response_mask = (
            np.abs(model_axis - coordinate) <= response_half_width
        ) & np.asarray(sample.observed.observed_valid, dtype=bool)
        response_mask &= np.asarray(
            projection.support_model[boundary_order],
            dtype=bool,
        )
        response_mask &= np.isfinite(truth_seismic)
        response_mask &= np.isfinite(
            counterfactual_seismic[boundary_order]
        )
        if np.count_nonzero(response_mask) < 3:
            raise ValueError(
                "counterfactual response window has fewer than three samples."
            )
        delta = (
            counterfactual_seismic[boundary_order, response_mask]
            - truth_seismic[response_mask]
        )
        baseline_rms = float(
            np.sqrt(np.mean(np.square(truth_seismic[response_mask])))
        )
        delta_rms = float(np.sqrt(np.mean(np.square(delta))))
        sensitivity = delta_rms / max(
            baseline_rms,
            np.finfo(np.float64).eps,
        )
        jump = float(
            sample.latent.log_ai_highres_truth[boundary_index]
            - sample.latent.log_ai_highres_truth[boundary_index - 1]
        )
        label = _classify(
            sensitivity=sensitivity,
            nearest_boundary_distance=nearest,
            tuning_scale=tuning_scale,
            threshold=options.clean_sensitivity_threshold,
        )
        row: dict[str, Any] = {
            "parent_id": parent_id,
            "lateral_index": lateral_index,
            "zone_id": zone_id,
            "boundary_order": boundary_order,
            "boundary_coordinate": coordinate,
            "sample_unit": sample.observed.sample_axis.unit,
            "interface_jump_log_ai": jump,
            "absolute_interface_jump_log_ai": abs(jump),
            "nearest_boundary_distance": nearest,
            "wavelet_power_centroid_hz": centroid,
            "median_vp_mps": median_vp,
            "tuning_scale": tuning_scale,
            "counterfactual_delta_rms": delta_rms,
            "counterfactual_baseline_rms": baseline_rms,
            "clean_forward_sensitivity": sensitivity,
            "observability_class": label,
        }
        for tolerance, matched in matches.items():
            row[f"matched_{tolerance:g}{sample.observed.sample_axis.unit}"] = (
                boundary_order in matched
            )
        rows.append(row)
    return rows


def _audit_parent(
    model: LateralStructuredModel,
    data: LateralPatchDataModule,
    prior: HsmmPrior,
    config: Stage1Step3Config,
    options: BoundaryObservabilityAuditOptions,
    *,
    parent_id: str,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    parent = data.benchmark.read_parent(parent_id)
    numpy_batch = data.build_parent_batch(
        parent,
        split="tuning_validation",
        seed=config.training.base.seed + 700_000,
        condition=options.condition,
        samples_per_zone_per_parent=1,
    )
    patch = lateral_patch_to_torch(numpy_batch, device=device)
    center = center_trace_batch(patch)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        evidence = model.encode_patch(patch)
        for batch_index, zone_id in enumerate(center.zone_ids):
            zone = torch.nonzero(
                center.zone_valid[batch_index],
                as_tuple=False,
            ).flatten()
            result = exact_hsmm(
                evidence.emission_log_potential[batch_index, zone],
                torch.zeros_like(
                    evidence.boundary_log_potential[batch_index, zone]
                ),
                prior.zone(zone_id),
            )
            rows.extend(
                _sample_rows(
                    parent,
                    center,
                    batch_index,
                    result.consensus_segments,
                    options,
                )
            )
    return {
        "status": "complete",
        "condition": options.condition,
        "boundary_rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    tolerance_labels: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label in (*OBSERVABILITY_CLASSES, "all"):
        selected = (
            list(rows)
            if label == "all"
            else [
                row for row in rows
                if row["observability_class"] == label
            ]
        )
        metrics: dict[str, Any] = {
            "boundary_count": len(selected),
            "median_clean_forward_sensitivity": (
                None
                if not selected
                else float(
                    np.median(
                        [
                            row["clean_forward_sensitivity"]
                            for row in selected
                        ]
                    )
                )
            ),
        }
        for tolerance_label in tolerance_labels:
            matched = sum(
                bool(row[f"matched_{tolerance_label}"])
                for row in selected
            )
            metrics[f"recall_{tolerance_label}"] = (
                None if not selected else matched / len(selected)
            )
        output[label] = metrics
    return output


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("boundary observability audit produced no rows.")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError("boundary observability rows have inconsistent columns.")
    temporary = path.with_name(f".{path.name}.staging")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_boundary_observability_audit(
    config: Stage1Step3Config,
    *,
    checkpoint_path: str | Path,
    split_manifest_path: str | Path,
    output_dir: str | Path,
    input_mode: str,
    options: BoundaryObservabilityAuditOptions | None = None,
) -> dict[str, Any]:
    """Run clean-synthetic observability stratification on tuning parents."""
    audit_options = options or BoundaryObservabilityAuditOptions()
    if input_mode not in {"full", "no-seismic"}:
        raise ValueError("input_mode must be full or no-seismic.")
    checkpoint_path = Path(checkpoint_path)
    split_manifest_path = Path(split_manifest_path)
    manifest = ParentSplitManifest.from_dict(
        json.loads(split_manifest_path.read_text(encoding="utf-8"))
    )
    split_fingerprint = str(
        manifest.to_dict()["fingerprint_sha256"]
    )
    benchmark = StructuredSyntheticBenchmark(config.benchmark_dir)
    parent_ids = _stratified_tuning_parents(
        benchmark,
        manifest,
        maximum_parents=audit_options.maximum_parents,
        seed=config.training.base.seed + 601,
    )
    plan_without_fingerprint = {
        "schema": BOUNDARY_OBSERVABILITY_PLAN_SCHEMA,
        "input_mode": input_mode,
        "split": "tuning_validation",
        "condition": audit_options.condition,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "split_manifest": str(split_manifest_path.resolve()),
        "split_manifest_fingerprint": split_fingerprint,
        "parent_ids": list(parent_ids),
        "boundary_tolerance_model_samples": list(
            audit_options.boundary_tolerance_model_samples
        ),
        "counterfactual_half_width_model_samples": int(
            audit_options.counterfactual_half_width_model_samples
        ),
        "response_window_tuning_multiples": float(
            audit_options.response_window_tuning_multiples
        ),
        "clean_sensitivity_threshold": float(
            audit_options.clean_sensitivity_threshold
        ),
        "interpretation_scope": (
            "clean model-consistent synthetic upper bound; clean sensitivity "
            "is necessary but not sufficient for real-field visibility"
        ),
    }
    plan = {
        **plan_without_fingerprint,
        "fingerprint_sha256": _fingerprint(plan_without_fingerprint),
    }
    journal = _AuditJournal(
        Path(output_dir),
        plan,
        resume=audit_options.resume,
        progress_schema=BOUNDARY_OBSERVABILITY_PROGRESS_SCHEMA,
    )
    logger = configure_training_logger(journal.output_dir)
    device, runtime = resolve_device(config.training.base.device)
    model, checkpoint = LateralStructuredModel.from_step3_checkpoint(
        checkpoint_path,
        device=device,
    )
    if str(checkpoint.get("input_mode", "")) != input_mode:
        raise ValueError("audit checkpoint input mode differs from request.")
    if str(checkpoint.get("split_manifest_fingerprint", "")) != (
        split_fingerprint
    ):
        raise ValueError("audit checkpoint and split manifest differ.")
    prior = HsmmPrior.from_dict(checkpoint["hsmm_prior"])
    data = LateralPatchDataModule(
        config.benchmark_dir,
        config.impedance_calibration,
        manifest,
        patch_width=int(model.config.patch_width),
        augmentation_profile=config.augmentation,
        dirty_probability=config.dirty_probability,
        condition_limit=config.training.base.condition_limit,
    )
    model.eval()
    pending = journal.pending()
    existing = len(parent_ids) - len(pending)
    started = time.perf_counter()
    logger.info(
        "boundary observability audit start | device=%s | parents=%d | "
        "condition=%s | resume=%s",
        device,
        len(parent_ids),
        audit_options.condition,
        audit_options.resume,
    )
    for completed_in_call, (index, parent_id) in enumerate(
        pending,
        start=1,
    ):
        try:
            payload = _audit_parent(
                model,
                data,
                prior,
                config,
                audit_options,
                parent_id=parent_id,
                device=device,
            )
        except KeyboardInterrupt as error:
            journal.fail(
                parent_id=parent_id,
                error=error,
                recoverable=True,
            )
            raise
        except Exception as error:
            journal.fail(
                parent_id=parent_id,
                error=error,
                recoverable=False,
            )
            raise
        journal.commit(index, parent_id, payload)
        completed = existing + completed_in_call
        if (
            completed_in_call == 1
            or completed % audit_options.progress_every_parents == 0
            or completed == len(parent_ids)
        ):
            logger.info(
                "boundary observability audit | parents=%d/%d | "
                "boundaries=%d | last_parent=%.1fs | elapsed=%.1fs",
                completed,
                len(parent_ids),
                len(payload["boundary_rows"]),
                float(payload["elapsed_seconds"]),
                time.perf_counter() - started,
            )
    payloads = journal.payloads()
    rows = [
        row
        for payload in payloads
        for row in payload["boundary_rows"]
    ]
    sample_units = {str(row["sample_unit"]) for row in rows}
    if len(sample_units) != 1:
        raise ValueError("observability rows do not share one sample unit.")
    tolerance_labels = [
        key.removeprefix("matched_")
        for key in rows[0]
        if key.startswith("matched_")
    ]
    _write_rows(journal.output_dir / "boundary_rows.csv", rows)
    report = {
        "schema": BOUNDARY_OBSERVABILITY_AUDIT_SCHEMA,
        "status": "complete",
        "input_mode": input_mode,
        "condition": audit_options.condition,
        "device": str(device),
        "runtime": runtime,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "split_manifest": str(split_manifest_path),
        "split_manifest_fingerprint": split_fingerprint,
        "plan_fingerprint_sha256": plan["fingerprint_sha256"],
        "parent_count": len(parent_ids),
        "boundary_count": len(rows),
        "sample_unit": next(iter(sample_units)),
        "interpretation_scope": plan["interpretation_scope"],
        "thresholds": {
            "clean_sensitivity_threshold": float(
                audit_options.clean_sensitivity_threshold
            ),
            "boundary_tolerance_model_samples": list(
                audit_options.boundary_tolerance_model_samples
            ),
            "counterfactual_half_width_model_samples": int(
                audit_options.counterfactual_half_width_model_samples
            ),
            "response_window_tuning_multiples": float(
                audit_options.response_window_tuning_multiples
            ),
        },
        "tuning_scale": {
            "median": float(
                np.median([row["tuning_scale"] for row in rows])
            ),
            "minimum": float(
                np.min([row["tuning_scale"] for row in rows])
            ),
            "maximum": float(
                np.max([row["tuning_scale"] for row in rows])
            ),
            "wavelet_power_centroid_hz": float(
                np.median(
                    [row["wavelet_power_centroid_hz"] for row in rows]
                )
            ),
        },
        "metrics_by_observability_class": _aggregate(
            rows,
            tolerance_labels=tolerance_labels,
        ),
        "artifacts": {
            "boundary_rows_csv": "boundary_rows.csv",
        },
        "elapsed_seconds": sum(
            float(payload["elapsed_seconds"]) for payload in payloads
        ),
    }
    journal.complete(report)
    logger.info(
        "boundary observability audit complete | parents=%d | "
        "boundaries=%d | tuning_median=%.3f%s",
        len(parent_ids),
        len(rows),
        report["tuning_scale"]["median"],
        report["sample_unit"],
    )
    return report


__all__ = [
    "BOUNDARY_OBSERVABILITY_AUDIT_SCHEMA",
    "BoundaryObservabilityAuditOptions",
    "bridge_boundary_counterfactual",
    "run_boundary_observability_audit",
    "tuning_scale_for_sample",
    "wavelet_power_centroid_hz",
]
