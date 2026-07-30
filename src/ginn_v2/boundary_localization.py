"""Truth-count-matched audit of Stage 1 boundary localization evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from ginn_v2.data import ParentSplitManifest
from ginn_v2.hsmm import (
    HsmmPrior,
    HsmmSegment,
    count_targeted_viterbi_segments,
)
from ginn_v2.lateral import (
    LateralPatchDataModule,
    LateralStructuredModel,
    center_trace_batch,
    lateral_patch_to_torch,
)
from ginn_v2.lateral_training import Stage1Step3Config
from ginn_v2.runtime import configure_training_logger, resolve_device
from ginn_v2.structure import truth_hsmm_segments
from ginn_v2.undersegmentation import (
    BOUNDARY_TOLERANCES_M,
    _AuditJournal,
    _file_sha256,
    _fingerprint,
    _match_boundaries,
    _merge_totals,
    _stratified_tuning_parents,
)


BOUNDARY_LOCALIZATION_AUDIT_SCHEMA = (
    "structured_ginn_v2_boundary_localization_audit_v1"
)
BOUNDARY_LOCALIZATION_PLAN_SCHEMA = (
    "structured_ginn_v2_boundary_localization_plan_v1"
)
BOUNDARY_LOCALIZATION_PROGRESS_SCHEMA = (
    "structured_ginn_v2_boundary_localization_progress_v1"
)
DEFAULT_BOUNDARY_SCALES = (-1.0, 0.0, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class BoundaryLocalizationAuditOptions:
    maximum_parents: int = 16
    condition: str = "clean"
    boundary_scales: tuple[float, ...] = DEFAULT_BOUNDARY_SCALES
    minimum_f1_improvement: float = 0.02
    progress_every_parents: int = 2
    resume: bool = False

    def __post_init__(self) -> None:
        if int(self.maximum_parents) <= 0:
            raise ValueError("maximum_parents must be positive.")
        if self.condition not in {"clean", "dirty"}:
            raise ValueError("audit condition must be clean or dirty.")
        scales = tuple(float(value) for value in self.boundary_scales)
        if not scales or not all(np.isfinite(value) for value in scales):
            raise ValueError("boundary_scales must be finite and non-empty.")
        if len(set(scales)) != len(scales):
            raise ValueError("boundary_scales must be unique.")
        if 0.0 not in scales:
            raise ValueError("boundary_scales must contain the scale=0 baseline.")
        if not any(value > 0.0 for value in scales):
            raise ValueError("boundary_scales need at least one positive scale.")
        if (
            not np.isfinite(self.minimum_f1_improvement)
            or self.minimum_f1_improvement < 0.0
        ):
            raise ValueError(
                "minimum_f1_improvement must be finite and non-negative."
            )
        if int(self.progress_every_parents) <= 0:
            raise ValueError("progress_every_parents must be positive.")
        object.__setattr__(self, "boundary_scales", scales)


def _scale_key(scale: float) -> str:
    return format(float(scale), ".12g")


def _structure_metric_totals(
    center: Any,
    segmentations: Sequence[Sequence[HsmmSegment]],
    *,
    sample_interval_m: float,
) -> dict[str, float]:
    totals: dict[str, float] = {
        "state_correct": 0.0,
        "state_count": 0.0,
        "segment_iou_sum": 0.0,
        "segment_iou_count": 0.0,
        "duration_error_samples_sum": 0.0,
        "duration_error_count": 0.0,
        "predicted_segment_count": 0.0,
        "truth_segment_count": 0.0,
        "sample_count": 0.0,
        "exact_target_count": 0.0,
    }
    for state_id in range(3):
        totals[f"state_{state_id}_correct"] = 0.0
        totals[f"state_{state_id}_count"] = 0.0
    for tolerance in BOUNDARY_TOLERANCES_M:
        label = f"{tolerance:g}m"
        totals[f"boundary_{label}_tp"] = 0.0
        totals[f"boundary_{label}_fp"] = 0.0
        totals[f"boundary_{label}_fn"] = 0.0
    for batch_index, predicted in enumerate(segmentations):
        zone = torch.nonzero(
            center.zone_valid[batch_index],
            as_tuple=False,
        ).flatten()
        first = int(zone[0].item())
        truth = truth_hsmm_segments(center, batch_index)
        truth_state = center.truth_state_highres[batch_index, zone]
        predicted_state = torch.full_like(truth_state, -1)
        for segment in predicted:
            predicted_state[segment.start : segment.stop] = segment.state_id
        totals["state_correct"] += float(
            torch.count_nonzero(predicted_state == truth_state).item()
        )
        totals["state_count"] += float(zone.numel())
        for state_id in range(3):
            state_mask = truth_state == state_id
            totals[f"state_{state_id}_count"] += float(
                torch.count_nonzero(state_mask).item()
            )
            totals[f"state_{state_id}_correct"] += float(
                torch.count_nonzero(
                    state_mask & (predicted_state == state_id)
                ).item()
            )
        truth_boundaries = [first + item.start for item in truth[1:]]
        predicted_boundaries = [
            first + item.start for item in predicted[1:]
        ]
        for tolerance in BOUNDARY_TOLERANCES_M:
            tp, fp, fn = _match_boundaries(
                predicted_boundaries,
                truth_boundaries,
                tolerance_m=tolerance,
                sample_interval_m=sample_interval_m,
            )
            label = f"{tolerance:g}m"
            totals[f"boundary_{label}_tp"] += float(tp)
            totals[f"boundary_{label}_fp"] += float(fp)
            totals[f"boundary_{label}_fn"] += float(fn)
        totals["predicted_segment_count"] += float(len(predicted))
        totals["truth_segment_count"] += float(len(truth))
        totals["sample_count"] += 1.0
        totals["exact_target_count"] += float(len(predicted) == len(truth))
        for truth_segment in truth:
            truth_indices = set(
                range(truth_segment.start, truth_segment.stop)
            )
            best_iou = 0.0
            best_duration_error: float | None = None
            for predicted_segment in predicted:
                if predicted_segment.state_id != truth_segment.state_id:
                    continue
                predicted_indices = set(
                    range(predicted_segment.start, predicted_segment.stop)
                )
                intersection = len(truth_indices & predicted_indices)
                if intersection == 0:
                    continue
                union = len(truth_indices | predicted_indices)
                best_iou = max(best_iou, intersection / union)
                error = abs(
                    predicted_segment.duration_samples
                    - truth_segment.duration_samples
                )
                if (
                    best_duration_error is None
                    or error < best_duration_error
                ):
                    best_duration_error = float(error)
            totals["segment_iou_sum"] += best_iou
            totals["segment_iou_count"] += 1.0
            totals["duration_error_samples_sum"] += (
                float(truth_segment.duration_samples)
                if best_duration_error is None
                else best_duration_error
            )
            totals["duration_error_count"] += 1.0
    return totals


def _finalize_structure_metrics(
    values: Mapping[str, float],
    *,
    sample_interval_m: float,
) -> dict[str, Any]:
    recalls = [
        values[f"state_{state_id}_correct"]
        / max(values[f"state_{state_id}_count"], 1.0)
        for state_id in range(3)
    ]
    boundaries: dict[str, Any] = {}
    for tolerance in BOUNDARY_TOLERANCES_M:
        label = f"{tolerance:g}m"
        tp = values[f"boundary_{label}_tp"]
        fp = values[f"boundary_{label}_fp"]
        fn = values[f"boundary_{label}_fn"]
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        boundaries[label] = {
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0
                * precision
                * recall
                / max(precision + recall, 1.0e-12)
            ),
        }
    sample_count = max(values["sample_count"], 1.0)
    duration_error_samples = (
        values["duration_error_samples_sum"]
        / max(values["duration_error_count"], 1.0)
    )
    return {
        "sample_count": int(values["sample_count"]),
        "exact_target_count_rate": (
            values["exact_target_count"] / sample_count
        ),
        "state_accuracy": (
            values["state_correct"] / max(values["state_count"], 1.0)
        ),
        "state_balanced_accuracy": float(np.mean(recalls)),
        "state_recall_by_class": recalls,
        "boundary_metrics": boundaries,
        "segment_iou": (
            values["segment_iou_sum"]
            / max(values["segment_iou_count"], 1.0)
        ),
        "duration_error_samples": duration_error_samples,
        "duration_error_m": (
            duration_error_samples * float(sample_interval_m)
        ),
        "mean_predicted_segment_count": (
            values["predicted_segment_count"] / sample_count
        ),
        "mean_truth_segment_count": (
            values["truth_segment_count"] / sample_count
        ),
        "segment_count_bias": (
            values["predicted_segment_count"]
            - values["truth_segment_count"]
        )
        / sample_count,
    }


def _raw_score_diagnostics(
    scores: torch.Tensor,
    truth: Sequence[HsmmSegment],
) -> dict[str, float]:
    boundary_mask = torch.zeros_like(scores, dtype=torch.bool)
    for segment in truth[1:]:
        boundary_mask[segment.start] = True
    support = torch.ones_like(boundary_mask)
    support[0] = False
    positive = scores[boundary_mask]
    negative = scores[support & ~boundary_mask]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("boundary score audit requires both classes.")
    comparison = positive[:, None] - negative[None, :]
    wins = torch.count_nonzero(comparison > 0.0)
    ties = torch.count_nonzero(comparison == 0.0)
    return {
        "positive_score_sum": float(torch.sum(positive).item()),
        "positive_count": float(positive.numel()),
        "negative_score_sum": float(torch.sum(negative).item()),
        "negative_count": float(negative.numel()),
        "auc_credit_sum": float((wins + 0.5 * ties).item()),
        "auc_pair_count": float(comparison.numel()),
    }


def _audit_parent(
    model: LateralStructuredModel,
    data: LateralPatchDataModule,
    prior: HsmmPrior,
    config: Stage1Step3Config,
    options: BoundaryLocalizationAuditOptions,
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
    segmentations: dict[str, list[tuple[HsmmSegment, ...]]] = {
        _scale_key(scale): [] for scale in options.boundary_scales
    }
    raw_totals: dict[str, float] = {}
    with torch.no_grad():
        evidence = model.encode_patch(patch)
        for batch_index, zone_id in enumerate(center.zone_ids):
            zone = torch.nonzero(
                center.zone_valid[batch_index],
                as_tuple=False,
            ).flatten()
            local_emission = evidence.emission_log_potential[
                batch_index, zone
            ]
            local_boundary = evidence.boundary_log_potential[
                batch_index, zone
            ]
            truth = truth_hsmm_segments(center, batch_index)
            _merge_totals(
                raw_totals,
                _raw_score_diagnostics(local_boundary, truth),
            )
            for scale in options.boundary_scales:
                segmentations[_scale_key(scale)].append(
                    count_targeted_viterbi_segments(
                        local_emission,
                        float(scale) * local_boundary,
                        prior.zone(zone_id),
                        target_count=len(truth),
                    )
                )
    sample_interval_m = float(parent.highres_axis.sample_interval)
    return {
        "status": "complete",
        "condition": options.condition,
        "highres_sample_interval_m": sample_interval_m,
        "scale_metric_totals": {
            key: _structure_metric_totals(
                center,
                values,
                sample_interval_m=sample_interval_m,
            )
            for key, values in segmentations.items()
        },
        "raw_score_totals": raw_totals,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _ranking_decision(
    results: Sequence[Mapping[str, Any]],
    *,
    minimum_f1_improvement: float,
) -> dict[str, Any]:
    by_scale = {float(item["scale"]): item for item in results}
    baseline = by_scale[0.0]

    def rank(item: Mapping[str, Any]) -> tuple[float, float, float]:
        metrics = item["metrics"]
        return (
            float(metrics["boundary_metrics"]["5m"]["f1"]),
            float(metrics["boundary_metrics"]["3m"]["f1"]),
            float(metrics["segment_iou"]),
        )

    positive = max(
        (item for item in results if float(item["scale"]) > 0.0),
        key=rank,
    )
    negative_items = [
        item for item in results if float(item["scale"]) < 0.0
    ]
    negative = max(negative_items, key=rank) if negative_items else None

    def contrast(item: Mapping[str, Any]) -> dict[str, float]:
        metrics = item["metrics"]
        return {
            "boundary_f1_5m_delta": (
                metrics["boundary_metrics"]["5m"]["f1"]
                - baseline["metrics"]["boundary_metrics"]["5m"]["f1"]
            ),
            "boundary_f1_3m_delta": (
                metrics["boundary_metrics"]["3m"]["f1"]
                - baseline["metrics"]["boundary_metrics"]["3m"]["f1"]
            ),
            "segment_iou_delta": (
                metrics["segment_iou"] - baseline["metrics"]["segment_iou"]
            ),
        }

    positive_contrast = contrast(positive)
    negative_contrast = contrast(negative) if negative is not None else None
    if (
        negative_contrast is not None
        and negative_contrast["boundary_f1_5m_delta"]
        >= minimum_f1_improvement
        and negative_contrast["boundary_f1_5m_delta"]
        > positive_contrast["boundary_f1_5m_delta"]
    ):
        decision = "boundary_polarity_inverted"
    elif (
        positive_contrast["boundary_f1_5m_delta"]
        >= minimum_f1_improvement
    ):
        decision = "positive_boundary_ranking_signal_present"
    else:
        decision = "boundary_ranking_signal_not_demonstrated"
    return {
        "decision": decision,
        "minimum_f1_improvement": float(minimum_f1_improvement),
        "baseline_scale": 0.0,
        "best_positive_scale": float(positive["scale"]),
        "best_positive_contrast": positive_contrast,
        "best_negative_scale": (
            None if negative is None else float(negative["scale"])
        ),
        "best_negative_contrast": negative_contrast,
    }


def run_boundary_localization_audit(
    config: Stage1Step3Config,
    *,
    checkpoint_path: str | Path,
    split_manifest_path: str | Path,
    output_dir: str | Path,
    input_mode: str,
    options: BoundaryLocalizationAuditOptions | None = None,
) -> dict[str, Any]:
    """Compare raw boundary scales at identical truth segment counts."""
    audit_options = options or BoundaryLocalizationAuditOptions()
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
        "schema": BOUNDARY_LOCALIZATION_PLAN_SCHEMA,
        "input_mode": input_mode,
        "split": "tuning_validation",
        "condition": audit_options.condition,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "split_manifest": str(split_manifest_path.resolve()),
        "split_manifest_fingerprint": split_fingerprint,
        "parent_ids": list(parent_ids),
        "boundary_scales": list(audit_options.boundary_scales),
        "boundary_tolerances_m": list(BOUNDARY_TOLERANCES_M),
        "count_contract": "truth_segment_count_per_center_trace",
        "minimum_f1_improvement": float(
            audit_options.minimum_f1_improvement
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
        progress_schema=BOUNDARY_LOCALIZATION_PROGRESS_SCHEMA,
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
        "boundary localization audit start | device=%s | condition=%s | "
        "parents=%d | scales=%s | resume=%s",
        device,
        audit_options.condition,
        len(parent_ids),
        ",".join(_scale_key(value) for value in audit_options.boundary_scales),
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
                "boundary localization audit | parents=%d/%d | "
                "last_parent=%.1fs | elapsed=%.1fs",
                completed,
                len(parent_ids),
                float(payload["elapsed_seconds"]),
                time.perf_counter() - started,
            )
    payloads = journal.payloads()
    sample_intervals = {
        float(payload["highres_sample_interval_m"])
        for payload in payloads
    }
    if len(sample_intervals) != 1:
        raise ValueError(
            "audit parents do not share one highres sample interval."
        )
    sample_interval_m = next(iter(sample_intervals))
    aggregate: dict[str, dict[str, float]] = {
        _scale_key(scale): {} for scale in audit_options.boundary_scales
    }
    raw_totals: dict[str, float] = {}
    for payload in payloads:
        for key in aggregate:
            _merge_totals(
                aggregate[key],
                payload["scale_metric_totals"][key],
            )
        _merge_totals(raw_totals, payload["raw_score_totals"])
    scale_results = [
        {
            "scale": float(scale),
            "metrics": _finalize_structure_metrics(
                aggregate[_scale_key(scale)],
                sample_interval_m=sample_interval_m,
            ),
        }
        for scale in audit_options.boundary_scales
    ]
    raw_score_report = {
        "truth_boundary_mean": (
            raw_totals["positive_score_sum"]
            / max(raw_totals["positive_count"], 1.0)
        ),
        "non_boundary_mean": (
            raw_totals["negative_score_sum"]
            / max(raw_totals["negative_count"], 1.0)
        ),
        "pairwise_auc": (
            raw_totals["auc_credit_sum"]
            / max(raw_totals["auc_pair_count"], 1.0)
        ),
        "truth_boundary_count": int(raw_totals["positive_count"]),
        "non_boundary_count": int(raw_totals["negative_count"]),
    }
    report = {
        "schema": BOUNDARY_LOCALIZATION_AUDIT_SCHEMA,
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
        "parent_ids": list(parent_ids),
        "highres_sample_interval_m": sample_interval_m,
        "boundary_tolerances_m": list(BOUNDARY_TOLERANCES_M),
        "count_contract": plan["count_contract"],
        "raw_score_diagnostics": raw_score_report,
        "scale_results": scale_results,
        "ranking_decision": _ranking_decision(
            scale_results,
            minimum_f1_improvement=(
                audit_options.minimum_f1_improvement
            ),
        ),
        "elapsed_seconds": sum(
            float(payload["elapsed_seconds"]) for payload in payloads
        ),
    }
    journal.complete(report)
    logger.info(
        "boundary localization audit complete | parents=%d | "
        "decision=%s | raw_auc=%.4f",
        len(parent_ids),
        report["ranking_decision"]["decision"],
        raw_score_report["pairwise_auc"],
    )
    return report


__all__ = [
    "BOUNDARY_LOCALIZATION_AUDIT_SCHEMA",
    "BoundaryLocalizationAuditOptions",
    "run_boundary_localization_audit",
]
