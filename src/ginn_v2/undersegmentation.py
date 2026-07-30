"""Frozen-checkpoint audit for Structured GINN V2 under-segmentation.

The module diagnoses the single-trace evidence/HSMM seam before any further
training.  It only reads the tuning split and compares controlled decoding
interventions on identical parent patches.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
    ZoneHsmmPrior,
    exact_hsmm,
)
from ginn_v2.lateral import (
    LateralPatchDataModule,
    LateralStructuredModel,
    center_trace_batch,
    lateral_patch_to_torch,
    parameterize_lateral_segments,
)
from ginn_v2.lateral_training import Stage1Step3Config
from ginn_v2.model import DirectionalEvidence, TeacherForcingOutput
from ginn_v2.runtime import configure_training_logger, resolve_device
from ginn_v2.structure import truth_hsmm_segments


UNDERSEGMENTATION_AUDIT_SCHEMA = (
    "structured_ginn_v2_undersegmentation_audit_v1"
)
UNDERSEGMENTATION_PLAN_SCHEMA = (
    "structured_ginn_v2_undersegmentation_plan_v1"
)
UNDERSEGMENTATION_PROGRESS_SCHEMA = (
    "structured_ginn_v2_undersegmentation_progress_v1"
)

AUDIT_VARIANTS = (
    "current_consensus",
    "current_viterbi_map",
    "raw_boundary_consensus",
    "flat_prior_consensus",
    "truth_boundary_predicted_state",
    "truth_state_consensus",
)
BOUNDARY_TOLERANCES_M = (0.0, 1.0, 3.0, 5.0)


@dataclass(frozen=True)
class UndersegmentationAuditOptions:
    maximum_parents: int = 16
    condition: str = "clean"
    oracle_logit: float = 12.0
    progress_every_parents: int = 2
    resume: bool = False

    def __post_init__(self) -> None:
        if int(self.maximum_parents) <= 0:
            raise ValueError("maximum_parents must be positive.")
        if self.condition not in {"clean", "dirty"}:
            raise ValueError("audit condition must be clean or dirty.")
        if not np.isfinite(self.oracle_logit) or float(self.oracle_logit) <= 0:
            raise ValueError("oracle_logit must be finite and positive.")
        if int(self.progress_every_parents) <= 0:
            raise ValueError("progress_every_parents must be positive.")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging")
    temporary.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratified_tuning_parents(
    benchmark: StructuredSyntheticBenchmark,
    manifest: ParentSplitManifest,
    *,
    maximum_parents: int,
    seed: int,
) -> tuple[str, ...]:
    parent_ids = tuple(manifest.tuning_validation)
    if int(maximum_parents) >= len(parent_ids):
        return parent_ids
    frame = benchmark.index.loc[
        benchmark.index["realization_id"].isin(parent_ids)
    ].copy()
    strata = [
        name
        for name in ("section_id", "duration_mode", "geometry_family")
        if name in frame.columns
    ]
    groups: list[list[str]] = []
    grouped = (
        frame.groupby(strata, sort=True, dropna=False)
        if strata
        else [(("all",), frame)]
    )
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        digest = hashlib.sha256(
            (
                f"{int(seed)}|"
                + "|".join(str(value) for value in key_values)
            ).encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        values = sorted(str(value) for value in group["realization_id"])
        order = rng.permutation(len(values))
        groups.append([values[int(index)] for index in order])
    selected: list[str] = []
    offset = 0
    while len(selected) < int(maximum_parents):
        added = False
        for group in groups:
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == int(maximum_parents):
                    break
        if not added:
            break
        offset += 1
    if len(selected) != int(maximum_parents):
        raise RuntimeError("audit stratification produced too few tuning parents.")
    return tuple(selected)


class _AuditJournal:
    def __init__(
        self,
        output_dir: Path,
        plan: Mapping[str, Any],
        *,
        resume: bool,
        progress_schema: str = UNDERSEGMENTATION_PROGRESS_SCHEMA,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.plan = dict(plan)
        self.plan_path = self.output_dir / "audit_plan.json"
        self.progress_path = self.output_dir / "audit_progress.json"
        self.progress_schema = str(progress_schema)
        if self.output_dir.exists():
            if not resume:
                raise FileExistsError(
                    f"audit output already exists; pass --resume: {self.output_dir}"
                )
            if not self.plan_path.is_file():
                raise ValueError("audit output has no resumable audit_plan.json.")
            existing = json.loads(self.plan_path.read_text(encoding="utf-8"))
            if existing != self.plan:
                raise ValueError("resume audit plan differs from existing plan.")
        else:
            if resume:
                raise FileNotFoundError(
                    f"resume audit output does not exist: {self.output_dir}"
                )
            self.output_dir.mkdir(parents=True)
            _atomic_json(self.plan_path, self.plan)
        self.parent_ids = tuple(str(value) for value in self.plan["parent_ids"])
        self.completed = self._scan_completed()
        self._write_progress("running", None)

    def _shard_path(self, index: int) -> Path:
        return self.output_dir / "parent_shards" / f"{int(index):04d}.json"

    def _scan_completed(self) -> int:
        count = 0
        for index, parent_id in enumerate(self.parent_ids):
            path = self._shard_path(index)
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("parent_id") != parent_id:
                raise ValueError(f"audit shard identity mismatch: {path}")
            count += 1
        return count

    def _write_progress(
        self,
        status: str,
        issue: Mapping[str, Any] | None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema": self.progress_schema,
            "plan_fingerprint_sha256": self.plan["fingerprint_sha256"],
            "status": str(status),
            "completed_parent_count": int(self.completed),
            "total_parent_count": len(self.parent_ids),
            "updated_unix_time": time.time(),
        }
        if issue is not None:
            payload["issue"] = dict(issue)
        _atomic_json(self.progress_path, payload)

    def pending(self) -> list[tuple[int, str]]:
        output: list[tuple[int, str]] = []
        for index, parent_id in enumerate(self.parent_ids):
            path = self._shard_path(index)
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("parent_id") != parent_id:
                    raise ValueError(f"audit shard identity mismatch: {path}")
                continue
            output.append((index, parent_id))
        return output

    def commit(
        self,
        index: int,
        parent_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        path = self._shard_path(index)
        existed = path.is_file()
        _atomic_json(
            path,
            {
                "parent_id": str(parent_id),
                **dict(payload),
            },
        )
        if not existed:
            self.completed += 1
        self._write_progress("running", None)

    def payloads(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for index, parent_id in enumerate(self.parent_ids):
            path = self._shard_path(index)
            if not path.is_file():
                raise RuntimeError(
                    f"audit is incomplete at parent {parent_id!r}."
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("parent_id") != parent_id:
                raise ValueError(f"audit shard identity mismatch: {path}")
            output.append(payload)
        return output

    def fail(
        self,
        *,
        parent_id: str | None,
        error: BaseException,
        recoverable: bool,
    ) -> None:
        issue = {
            "category": (
                "interrupted" if isinstance(error, KeyboardInterrupt)
                else "audit_runtime"
            ),
            "severity": "recoverable" if recoverable else "blocking",
            "parent_id": parent_id,
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        _atomic_json(self.output_dir / "audit_failure.json", issue)
        self._write_progress(
            "interrupted" if recoverable else "failed",
            issue,
        )

    def complete(self, report: Mapping[str, Any]) -> None:
        _atomic_json(self.output_dir / "audit_report.json", report)
        self._write_progress("complete", None)


def _flat_prior(prior: ZoneHsmmPrior) -> ZoneHsmmPrior:
    transition = np.full((3, 3), np.log(0.5), dtype=np.float64)
    np.fill_diagonal(transition, -np.inf)
    return ZoneHsmmPrior(
        zone_id=prior.zone_id,
        initial_log_probability=np.full(3, -np.log(3.0), dtype=np.float64),
        transition_log_probability=transition,
        duration_fraction_edges=np.array(
            prior.duration_fraction_edges,
            copy=True,
        ),
        duration_log_probability=np.zeros_like(
            prior.duration_log_probability,
            dtype=np.float64,
        ),
    )


def _truth_state_emission(
    truth_state: torch.Tensor,
    *,
    oracle_logit: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    if truth_state.ndim != 1:
        raise ValueError("truth state oracle requires one state sequence.")
    if not bool(torch.all((truth_state >= 0) & (truth_state <= 2)).item()):
        raise ValueError("truth state oracle contains invalid state ids.")
    output = torch.full(
        (truth_state.numel(), 3),
        -float(oracle_logit),
        dtype=dtype,
        device=truth_state.device,
    )
    return output.scatter(
        1,
        truth_state.to(dtype=torch.long).unsqueeze(1),
        float(oracle_logit),
    )


def _fixed_boundary_viterbi(
    emission: torch.Tensor,
    prior: ZoneHsmmPrior,
    extents: Sequence[HsmmSegment],
) -> tuple[HsmmSegment, ...]:
    """Choose states while keeping externally supplied segment extents fixed."""
    if not extents or extents[0].start != 0:
        raise ValueError("fixed-boundary path must start at zero.")
    if extents[-1].stop != int(emission.shape[0]):
        raise ValueError("fixed-boundary path must cover the complete zone.")
    initial = torch.as_tensor(
        prior.initial_log_probability,
        dtype=emission.dtype,
        device=emission.device,
    )
    transition = torch.as_tensor(
        prior.transition_log_probability,
        dtype=emission.dtype,
        device=emission.device,
    )
    duration = prior.duration_scores(
        int(emission.shape[0]),
        device=emission.device,
        dtype=emission.dtype,
    )
    local_scores: list[torch.Tensor] = []
    previous_stop = 0
    for extent in extents:
        if extent.start != previous_stop:
            raise ValueError("fixed-boundary extents must be contiguous.")
        length = extent.duration_samples
        local_scores.append(
            torch.sum(emission[extent.start : extent.stop], dim=0)
            + duration[:, length - 1]
        )
        previous_stop = extent.stop
    scores = initial + local_scores[0]
    backpointers: list[torch.Tensor] = []
    for local in local_scores[1:]:
        candidates = scores.unsqueeze(1) + transition
        best, back = torch.max(candidates, dim=0)
        scores = best + local
        backpointers.append(back)
    state = int(torch.argmax(scores).item())
    states = [state]
    for back in reversed(backpointers):
        state = int(back[state].item())
        states.append(state)
    states.reverse()
    return tuple(
        HsmmSegment(
            state_id=state_id,
            start=extent.start,
            stop=extent.stop,
        )
        for state_id, extent in zip(states, extents, strict=True)
    )


def _run_variants(
    evidence: DirectionalEvidence,
    center: Any,
    prior: HsmmPrior,
    *,
    oracle_logit: float,
) -> tuple[
    dict[str, tuple[tuple[HsmmSegment, ...], ...]],
    dict[str, float],
]:
    segmentations: dict[str, list[tuple[HsmmSegment, ...]]] = {
        name: [] for name in AUDIT_VARIANTS
    }
    expected_counts = {
        "current_posterior_expected_count_sum": 0.0,
        "raw_boundary_posterior_expected_count_sum": 0.0,
        "flat_prior_posterior_expected_count_sum": 0.0,
        "truth_state_posterior_expected_count_sum": 0.0,
        "truth_segment_count_sum": 0.0,
        "sample_count": 0.0,
    }
    for batch_index, zone_id in enumerate(center.zone_ids):
        zone = torch.nonzero(
            center.zone_valid[batch_index],
            as_tuple=False,
        ).flatten()
        if zone.numel() == 0:
            raise ValueError("undersegmentation audit received an empty zone.")
        local_emission = evidence.emission_log_potential[batch_index, zone]
        local_boundary = evidence.boundary_log_potential[batch_index, zone]
        zero_boundary = torch.zeros_like(local_boundary)
        zone_prior = prior.zone(zone_id)
        current = exact_hsmm(local_emission, zero_boundary, zone_prior)
        raw_boundary = exact_hsmm(
            local_emission,
            local_boundary,
            zone_prior,
        )
        flat = exact_hsmm(
            local_emission,
            zero_boundary,
            _flat_prior(zone_prior),
        )
        truth_states = center.truth_state_highres[batch_index, zone]
        truth_state_result = exact_hsmm(
            _truth_state_emission(
                truth_states,
                oracle_logit=oracle_logit,
                dtype=local_emission.dtype,
            ),
            zero_boundary,
            zone_prior,
        )
        truth_segments = truth_hsmm_segments(center, batch_index)
        fixed_boundary = _fixed_boundary_viterbi(
            local_emission,
            zone_prior,
            truth_segments,
        )
        segmentations["current_consensus"].append(
            current.consensus_segments
        )
        segmentations["current_viterbi_map"].append(current.map_segments)
        segmentations["raw_boundary_consensus"].append(
            raw_boundary.consensus_segments
        )
        segmentations["flat_prior_consensus"].append(
            flat.consensus_segments
        )
        segmentations["truth_boundary_predicted_state"].append(
            fixed_boundary
        )
        segmentations["truth_state_consensus"].append(
            truth_state_result.consensus_segments
        )
        expected_counts["current_posterior_expected_count_sum"] += float(
            torch.sum(current.boundary_marginal).item()
        )
        expected_counts[
            "raw_boundary_posterior_expected_count_sum"
        ] += float(torch.sum(raw_boundary.boundary_marginal).item())
        expected_counts[
            "flat_prior_posterior_expected_count_sum"
        ] += float(torch.sum(flat.boundary_marginal).item())
        expected_counts[
            "truth_state_posterior_expected_count_sum"
        ] += float(torch.sum(truth_state_result.boundary_marginal).item())
        expected_counts["truth_segment_count_sum"] += float(
            len(truth_segments)
        )
        expected_counts["sample_count"] += 1.0
    return (
        {
            name: tuple(items)
            for name, items in segmentations.items()
        },
        expected_counts,
    )


def _match_boundaries(
    predicted: Sequence[int],
    truth: Sequence[int],
    *,
    tolerance_m: float,
    sample_interval_m: float,
) -> tuple[int, int, int]:
    unmatched = list(sorted(int(value) for value in truth))
    matched = 0
    for candidate in sorted(int(value) for value in predicted):
        choices = [
            (abs(candidate - target), index)
            for index, target in enumerate(unmatched)
            if abs(candidate - target) * float(sample_interval_m)
            <= float(tolerance_m) + 1.0e-9
        ]
        if not choices:
            continue
        _, selected = min(choices)
        unmatched.pop(selected)
        matched += 1
    return matched, len(predicted) - matched, len(unmatched)


def _segmentation_metric_totals(
    center: Any,
    output: TeacherForcingOutput,
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
    projection_mask = (
        output.projection_support & center.projected_support
    )
    projected_error = (
        output.projected_log_ai - center.projected_truth
    ).square()
    totals["projected_squared_error"] = float(
        torch.sum(projected_error[projection_mask]).detach().cpu()
    )
    totals["projected_count"] = float(
        torch.count_nonzero(projection_mask).item()
    )
    highres_error = (
        output.decoded_highres - center.truth_highres
    ).square()
    totals["highres_squared_error"] = float(
        torch.sum(highres_error[center.zone_valid]).detach().cpu()
    )
    totals["highres_count"] = float(
        torch.count_nonzero(center.zone_valid).item()
    )
    return totals


def _merge_totals(
    target: dict[str, float],
    source: Mapping[str, float],
) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def _finalize_variant(
    values: Mapping[str, float],
    *,
    sample_interval_m: float,
) -> dict[str, Any]:
    state_count = max(values["state_count"], 1.0)
    state_accuracy = values["state_correct"] / state_count
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
    return {
        "sample_count": int(values["sample_count"]),
        "highres_sample_interval_m": float(sample_interval_m),
        "state_accuracy": state_accuracy,
        "state_balanced_accuracy": float(np.mean(recalls)),
        "state_recall_by_class": recalls,
        "boundary_metrics": boundaries,
        "segment_iou": values["segment_iou_sum"]
        / max(values["segment_iou_count"], 1.0),
        "duration_error_samples": values["duration_error_samples_sum"]
        / max(values["duration_error_count"], 1.0),
        "duration_error_m": (
            values["duration_error_samples_sum"]
            / max(values["duration_error_count"], 1.0)
            * float(sample_interval_m)
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
        "projected_rmse": float(
            np.sqrt(
                values["projected_squared_error"]
                / max(values["projected_count"], 1.0)
            )
        ),
        "highres_rmse": float(
            np.sqrt(
                values["highres_squared_error"]
                / max(values["highres_count"], 1.0)
            )
        ),
    }


def _audit_parent(
    model: LateralStructuredModel,
    data: LateralPatchDataModule,
    prior: HsmmPrior,
    config: Stage1Step3Config,
    *,
    parent_id: str,
    condition: str,
    oracle_logit: float,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    parent = data.benchmark.read_parent(parent_id)
    numpy_batch = data.build_parent_batch(
        parent,
        split="tuning_validation",
        seed=config.training.base.seed + 700_000,
        condition=condition,
        samples_per_zone_per_parent=1,
    )
    patch = lateral_patch_to_torch(numpy_batch, device=device)
    center = center_trace_batch(patch)
    with torch.no_grad():
        evidence = model.encode_patch(patch)
        segmentations, expected_counts = _run_variants(
            evidence,
            center,
            prior,
            oracle_logit=oracle_logit,
        )
        variant_totals: dict[str, dict[str, float]] = {}
        for name in AUDIT_VARIANTS:
            _, output = parameterize_lateral_segments(
                model,
                patch,
                evidence,
                segmentations[name],
            )
            variant_totals[name] = _segmentation_metric_totals(
                center,
                output,
                segmentations[name],
                sample_interval_m=float(
                    parent.highres_axis.sample_interval
                ),
            )
    return {
        "status": "complete",
        "condition": condition,
        "highres_sample_interval_m": float(
            parent.highres_axis.sample_interval
        ),
        "variant_metric_totals": variant_totals,
        "posterior_expected_count_totals": expected_counts,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _diagnostic_contrasts(
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    current = variants["current_consensus"]

    def compare(name: str) -> dict[str, float]:
        candidate = variants[name]
        return {
            "segment_count_bias_delta": (
                candidate["segment_count_bias"]
                - current["segment_count_bias"]
            ),
            "segment_iou_delta": (
                candidate["segment_iou"] - current["segment_iou"]
            ),
            "boundary_f1_5m_delta": (
                candidate["boundary_metrics"]["5m"]["f1"]
                - current["boundary_metrics"]["5m"]["f1"]
            ),
            "projected_rmse_reduction": (
                current["projected_rmse"]
                - candidate["projected_rmse"]
            ),
            "highres_rmse_reduction": (
                current["highres_rmse"]
                - candidate["highres_rmse"]
            ),
        }

    return {
        "consensus_vs_viterbi": compare("current_viterbi_map"),
        "raw_boundary_intervention": compare(
            "raw_boundary_consensus"
        ),
        "flat_prior_intervention": compare("flat_prior_consensus"),
        "truth_boundary_ceiling": compare(
            "truth_boundary_predicted_state"
        ),
        "truth_state_intervention": compare("truth_state_consensus"),
    }


def run_undersegmentation_audit(
    config: Stage1Step3Config,
    *,
    checkpoint_path: str | Path,
    split_manifest_path: str | Path,
    output_dir: str | Path,
    input_mode: str,
    options: UndersegmentationAuditOptions | None = None,
) -> dict[str, Any]:
    """Run the six-way frozen tuning audit and publish a resumable report."""
    audit_options = options or UndersegmentationAuditOptions()
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
        "schema": UNDERSEGMENTATION_PLAN_SCHEMA,
        "input_mode": input_mode,
        "split": "tuning_validation",
        "condition": audit_options.condition,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "split_manifest": str(split_manifest_path.resolve()),
        "split_manifest_fingerprint": split_fingerprint,
        "parent_ids": list(parent_ids),
        "variants": list(AUDIT_VARIANTS),
        "boundary_tolerances_m": list(BOUNDARY_TOLERANCES_M),
        "oracle_logit": float(audit_options.oracle_logit),
    }
    plan = {
        **plan_without_fingerprint,
        "fingerprint_sha256": _fingerprint(plan_without_fingerprint),
    }
    journal = _AuditJournal(
        Path(output_dir),
        plan,
        resume=audit_options.resume,
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
        "undersegmentation audit start | device=%s | condition=%s | "
        "parents=%d | variants=%d | resume=%s",
        device,
        audit_options.condition,
        len(parent_ids),
        len(AUDIT_VARIANTS),
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
                parent_id=parent_id,
                condition=audit_options.condition,
                oracle_logit=audit_options.oracle_logit,
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
                "undersegmentation audit | parents=%d/%d | "
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
    aggregate_totals: dict[str, dict[str, float]] = {
        name: {} for name in AUDIT_VARIANTS
    }
    expected_totals: dict[str, float] = {}
    for payload in payloads:
        for name in AUDIT_VARIANTS:
            _merge_totals(
                aggregate_totals[name],
                payload["variant_metric_totals"][name],
            )
        _merge_totals(
            expected_totals,
            payload["posterior_expected_count_totals"],
        )
    variants = {
        name: _finalize_variant(
            aggregate_totals[name],
            sample_interval_m=sample_interval_m,
        )
        for name in AUDIT_VARIANTS
    }
    expected_sample_count = max(expected_totals["sample_count"], 1.0)
    truth_count_sum = expected_totals["truth_segment_count_sum"]
    expected_report = {
        "sample_count": int(expected_totals["sample_count"]),
        "mean_truth_segment_count": (
            truth_count_sum / expected_sample_count
        ),
    }
    for prefix in (
        "current",
        "raw_boundary",
        "flat_prior",
        "truth_state",
    ):
        key = f"{prefix}_posterior_expected_count_sum"
        mean = expected_totals[key] / expected_sample_count
        expected_report[f"{prefix}_mean_expected_segment_count"] = mean
        expected_report[f"{prefix}_expected_segment_count_bias"] = (
            expected_totals[key] - truth_count_sum
        ) / expected_sample_count
    report = {
        "schema": UNDERSEGMENTATION_AUDIT_SCHEMA,
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
        "variants": variants,
        "posterior_expected_segment_counts": expected_report,
        "diagnostic_contrasts": _diagnostic_contrasts(variants),
        "elapsed_seconds": sum(
            float(payload["elapsed_seconds"]) for payload in payloads
        ),
    }
    journal.complete(report)
    logger.info(
        "undersegmentation audit complete | parents=%d | "
        "current_count_bias=%.4f | current_iou=%.4f",
        len(parent_ids),
        variants["current_consensus"]["segment_count_bias"],
        variants["current_consensus"]["segment_iou"],
    )
    return report


__all__ = [
    "AUDIT_VARIANTS",
    "BOUNDARY_TOLERANCES_M",
    "UNDERSEGMENTATION_AUDIT_SCHEMA",
    "UndersegmentationAuditOptions",
    "run_undersegmentation_audit",
]
