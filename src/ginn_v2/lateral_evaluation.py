"""Resumable, preflight-first evaluation for Stage 1 Step 3.

The evaluator treats geometry holdout as an explicit final gate.  Its default
diagnostic scope evaluates small, stratified development subsets while a
model-free preflight scans every evaluation parent for late contract failures.
Every evaluated parent is committed atomically, so interruption never discards
completed inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cup.synthetic.readers.structured import (
    StructuredParent,
    StructuredSyntheticBenchmark,
)
from ginn_v2.data import ParentSplitManifest
from ginn_v2.hsmm import HsmmPrior, HsmmSegment, hsmm_path_score
from ginn_v2.lateral import (
    LATERAL_RUN_SCHEMA,
    LateralPatchDataModule,
    LateralStructuredModel,
    center_trace_batch,
    infer_lateral_patch,
    lateral_patch_to_torch,
    lateral_structured_training_loss,
    validate_parent_event_identity,
)
from ginn_v2.lateral_training import Stage1Step3Config
from ginn_v2.runtime import configure_training_logger, resolve_device
from ginn_v2.structure import truth_hsmm_segments
from ginn_v2.structure_training import (
    finalize_predicted_metrics,
    merge_metric_totals,
    predicted_metric_totals,
)


EVALUATION_SCHEMA = "structured_ginn_v2_stage1_step3_evaluation_v2"
EVALUATION_PLAN_SCHEMA = "structured_ginn_v2_stage1_step3_evaluation_plan_v1"
EVALUATION_PROGRESS_SCHEMA = "structured_ginn_v2_stage1_step3_evaluation_progress_v1"
PREFLIGHT_SCHEMA = "structured_ginn_v2_stage1_step3_preflight_v1"


@dataclass(frozen=True)
class Step3EvaluationOptions:
    """Small evaluator interface; persistence and sharding remain internal."""

    scope: str = "diagnostic"
    diagnostic_parents_per_split: int = 8
    progress_every_parents: int = 5
    preflight_only: bool = False
    resume: bool = False
    preflight_parent_limit_per_split: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"diagnostic", "final"}:
            raise ValueError("evaluation scope must be diagnostic or final.")
        if int(self.diagnostic_parents_per_split) <= 0:
            raise ValueError("diagnostic_parents_per_split must be positive.")
        if int(self.progress_every_parents) <= 0:
            raise ValueError("progress_every_parents must be positive.")
        if (
            self.preflight_parent_limit_per_split is not None
            and int(self.preflight_parent_limit_per_split) <= 0
        ):
            raise ValueError(
                "preflight_parent_limit_per_split must be positive when supplied."
            )


@dataclass(frozen=True)
class EvaluationStage:
    name: str
    split: str
    condition: str
    parent_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "split": self.split,
            "condition": self.condition,
            "parent_ids": list(self.parent_ids),
        }


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


def _issue(
    *,
    category: str,
    phase: str,
    error: BaseException,
    parent_id: str | None = None,
    split: str | None = None,
    condition: str | None = None,
    severity: str = "blocking",
) -> dict[str, Any]:
    return {
        "category": str(category),
        "severity": str(severity),
        "phase": str(phase),
        "parent_id": parent_id,
        "split": split,
        "condition": condition,
        "exception_type": type(error).__name__,
        "message": str(error),
    }


def _stratified_parent_ids(
    benchmark: StructuredSyntheticBenchmark,
    parent_ids: Sequence[str],
    *,
    limit: int,
    seed: int,
) -> tuple[str, ...]:
    values = tuple(str(value) for value in parent_ids)
    if int(limit) >= len(values):
        return values
    frame = benchmark.index.loc[
        benchmark.index["realization_id"].isin(values)
    ].copy()
    strata = [
        name
        for name in ("section_id", "duration_mode", "geometry_family")
        if name in frame.columns
    ]
    if not strata:
        return tuple(sorted(values)[: int(limit)])
    groups: list[list[str]] = []
    for key, group in frame.groupby(strata, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        digest = hashlib.sha256(
            (
                f"{int(seed)}|"
                + "|".join(str(value) for value in key_values)
            ).encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        group_ids = sorted(str(value) for value in group["realization_id"])
        order = rng.permutation(len(group_ids))
        groups.append([group_ids[int(index)] for index in order])
    selected: list[str] = []
    offset = 0
    while len(selected) < int(limit):
        added = False
        for group in groups:
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == int(limit):
                    break
        if not added:
            break
        offset += 1
    if len(selected) != int(limit):
        raise RuntimeError("stratified diagnostic selection produced too few parents.")
    return tuple(selected)


def _evaluation_stages(
    benchmark: StructuredSyntheticBenchmark,
    manifest: ParentSplitManifest,
    options: Step3EvaluationOptions,
    *,
    seed: int,
) -> tuple[EvaluationStage, ...]:
    if options.scope == "final":
        tuning = manifest.tuning_validation
        calibration = manifest.calibration
    else:
        tuning = _stratified_parent_ids(
            benchmark,
            manifest.tuning_validation,
            limit=options.diagnostic_parents_per_split,
            seed=seed + 101,
        )
        calibration = _stratified_parent_ids(
            benchmark,
            manifest.calibration,
            limit=options.diagnostic_parents_per_split,
            seed=seed + 211,
        )
    stages = [
        EvaluationStage(
            "tuning_validation_clean",
            "tuning_validation",
            "clean",
            tuple(tuning),
        ),
        EvaluationStage(
            "tuning_validation_dirty",
            "tuning_validation",
            "dirty",
            tuple(tuning),
        ),
        EvaluationStage(
            "calibration_clean",
            "calibration",
            "clean",
            tuple(calibration),
        ),
        EvaluationStage(
            "calibration_dirty",
            "calibration",
            "dirty",
            tuple(calibration),
        ),
    ]
    if options.scope == "final":
        stages.append(
            EvaluationStage(
                "geometry_holdout_clean",
                "geometry_holdout",
                "clean",
                tuple(manifest.geometry_holdout),
            )
        )
    return tuple(stages)


class _EvaluationJournal:
    """Atomic per-parent journal hidden behind one evaluation-run seam."""

    def __init__(
        self,
        output_dir: Path,
        plan: Mapping[str, Any],
        *,
        resume: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.plan = dict(plan)
        self.plan_path = self.output_dir / "evaluation_plan.json"
        self.progress_path = self.output_dir / "evaluation_progress.json"
        if self.output_dir.exists():
            if not resume:
                raise FileExistsError(
                    f"output directory already exists; pass --resume to continue: "
                    f"{self.output_dir}"
                )
            if not self.plan_path.is_file():
                raise ValueError(
                    "existing output has no resumable evaluation_plan.json"
                )
            existing = json.loads(self.plan_path.read_text(encoding="utf-8"))
            if existing != self.plan:
                raise ValueError("resume evaluation plan differs from existing plan.")
        else:
            if resume:
                raise FileNotFoundError(
                    f"resume output directory does not exist: {self.output_dir}"
                )
            self.output_dir.mkdir(parents=True)
            _atomic_json(self.plan_path, self.plan)
        self._phase_parent_ids = {
            "preflight": tuple(
                str(value) for value in self.plan["preflight_parent_ids"]
            ),
            **{
                str(stage["name"]): tuple(
                    str(value) for value in stage["parent_ids"]
                )
                for stage in self.plan["stages"]
            },
        }
        self._completed_counts = {
            phase: self._completed_count(phase, parent_ids)
            for phase, parent_ids in self._phase_parent_ids.items()
        }
        self._write_progress(status="running", current_phase="initializing")

    def _shard_path(self, phase: str, index: int) -> Path:
        return self.output_dir / "shards" / str(phase) / f"{int(index):05d}.json"

    def _completed_count(self, phase: str, parent_ids: Sequence[str]) -> int:
        completed = 0
        for index, parent_id in enumerate(parent_ids):
            path = self._shard_path(phase, index)
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("parent_id") != str(parent_id):
                raise ValueError(
                    f"evaluation shard identity mismatch: {path}"
                )
            completed += 1
        return completed

    def _progress_counts(self) -> dict[str, Any]:
        return {
            phase: {
                "completed": int(self._completed_counts[phase]),
                "total": len(parent_ids),
            }
            for phase, parent_ids in self._phase_parent_ids.items()
        }

    def _write_progress(
        self,
        *,
        status: str,
        current_phase: str,
        issue: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema": EVALUATION_PROGRESS_SCHEMA,
            "plan_fingerprint_sha256": self.plan["fingerprint_sha256"],
            "status": str(status),
            "current_phase": str(current_phase),
            "phases": self._progress_counts(),
            "updated_unix_time": time.time(),
        }
        if issue is not None:
            payload["issue"] = dict(issue)
        _atomic_json(self.progress_path, payload)

    def pending(
        self,
        phase: str,
        parent_ids: Sequence[str],
    ) -> list[tuple[int, str]]:
        output: list[tuple[int, str]] = []
        for index, parent_id in enumerate(parent_ids):
            path = self._shard_path(phase, index)
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("parent_id") != str(parent_id):
                    raise ValueError(
                        f"evaluation shard identity mismatch: {path}"
                    )
                continue
            output.append((index, str(parent_id)))
        return output

    def commit(
        self,
        phase: str,
        index: int,
        parent_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        shard = {
            "phase": str(phase),
            "parent_id": str(parent_id),
            **dict(payload),
        }
        path = self._shard_path(phase, index)
        existed = path.is_file()
        _atomic_json(path, shard)
        if not existed:
            self._completed_counts[str(phase)] += 1
        self._write_progress(status="running", current_phase=phase)

    def phase_payloads(
        self,
        phase: str,
        parent_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for index, parent_id in enumerate(parent_ids):
            path = self._shard_path(phase, index)
            if not path.is_file():
                raise RuntimeError(
                    f"evaluation phase {phase!r} is incomplete at parent "
                    f"{parent_id!r}."
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("parent_id") != str(parent_id):
                raise ValueError(f"evaluation shard identity mismatch: {path}")
            payloads.append(payload)
        return payloads

    def fail(self, issue: Mapping[str, Any]) -> None:
        _atomic_json(self.output_dir / "evaluation_failure.json", dict(issue))
        status = (
            "interrupted"
            if issue.get("category") == "interrupted"
            else "failed"
        )
        self._write_progress(
            status=status,
            current_phase=str(issue.get("phase", "unknown")),
            issue=issue,
        )

    def complete(self, report: Mapping[str, Any]) -> None:
        _atomic_json(self.output_dir / "evaluation_report.json", report)
        self._write_progress(status="complete", current_phase="complete")

    def preflight_complete(self) -> None:
        self._write_progress(
            status="preflight_complete",
            current_phase="preflight",
        )


def _load_prior(
    checkpoint: Mapping[str, Any],
    split_fingerprint: str,
) -> HsmmPrior:
    if str(checkpoint.get("split_manifest_fingerprint")) != str(
        split_fingerprint
    ):
        raise ValueError("Step-3 checkpoint and split manifest differ.")
    prior = HsmmPrior.from_dict(checkpoint["hsmm_prior"])
    if prior.split_manifest_fingerprint != str(split_fingerprint):
        raise ValueError("Step-3 HSMM prior and split manifest differ.")
    return prior


def _validate_parent_hsmm_tables(
    parent: StructuredParent,
    prior: HsmmPrior,
) -> int:
    """Validate every active truth path without constructing model tensors."""
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in parent.segments:
        if (
            float(row["bottom"]) <= float(row["top"])
            or float(row["duration_fraction"]) <= 0.0
            or int(row.get("duration_samples", 0)) <= 0
        ):
            continue
        key = (int(row["lateral_index"]), str(row["zone_id"]))
        grouped.setdefault(key, []).append(row)
    if not grouped:
        raise ValueError("parent has no active HSMM truth paths.")
    path_count = 0
    # A pinched-out producer event may retain a sub-sample geometric extent
    # while publishing duration_samples=0.  It has no HSMM support and is
    # legitimately bridged by the neighbouring sampled runs.
    endpoint_tolerance = max(
        abs(float(parent.highres_axis.sample_interval)) * (1.0 + 1.0e-6),
        1.0e-8,
    )
    for (_, zone_id), rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda item: (
                float(item["top"]),
                int(item["object_id"]),
            ),
        )
        canonical_states: list[int] = []
        canonical_durations: list[int] = []
        previous_bottom: float | None = None
        for row in ordered:
            top = float(row["top"])
            bottom = float(row["bottom"])
            if (
                previous_bottom is not None
                and not np.isclose(
                    top,
                    previous_bottom,
                    rtol=0.0,
                    atol=endpoint_tolerance,
                )
            ):
                raise ValueError(
                    f"zone {zone_id!r} segment endpoints are not contiguous."
                )
            state_id = int(row["state_id"])
            if state_id not in {0, 1, 2}:
                raise ValueError(
                    f"zone {zone_id!r} contains invalid state {state_id}."
                )
            duration = int(row["duration_samples"])
            if canonical_states and canonical_states[-1] == state_id:
                canonical_durations[-1] += duration
            else:
                canonical_states.append(state_id)
                canonical_durations.append(duration)
            previous_bottom = bottom
        segments: list[HsmmSegment] = []
        start = 0
        for state_id, duration in zip(
            canonical_states,
            canonical_durations,
            strict=True,
        ):
            segments.append(
                HsmmSegment(
                    state_id=state_id,
                    start=start,
                    stop=start + duration,
                )
            )
            start += duration
        emission = torch.zeros((start, 3), dtype=torch.float32)
        boundary = torch.zeros((start,), dtype=torch.float32)
        score = hsmm_path_score(
            emission,
            boundary,
            prior.zone(zone_id),
            segments,
        )
        if not bool(torch.isfinite(score).item()):
            raise FloatingPointError("truth HSMM table path score is non-finite.")
        path_count += 1
    return path_count


def _preflight_parent(
    data: LateralPatchDataModule,
    prior: HsmmPrior,
    *,
    parent_id: str,
    split: str,
    seed: int,
    deep: bool,
) -> dict[str, Any]:
    try:
        parent = data.benchmark.read_parent(parent_id)
    except Exception as error:
        return {
            "status": "error",
            "issue": _issue(
                category="artifact_contract",
                phase="preflight",
                error=error,
                parent_id=parent_id,
                split=split,
            ),
        }
    try:
        identity = validate_parent_event_identity(parent)
    except Exception as error:
        return {
            "status": "error",
            "issue": _issue(
                category="topology_contract",
                phase="preflight",
                error=error,
                parent_id=parent_id,
                split=split,
            ),
        }
    try:
        path_count = _validate_parent_hsmm_tables(parent, prior)
    except Exception as error:
        return {
            "status": "error",
            "issue": _issue(
                category="hsmm_contract",
                phase="preflight",
                error=error,
                parent_id=parent_id,
                split=split,
            ),
        }
    patch_count = 0
    if deep:
        try:
            numpy_batch = data.build_parent_batch(
                parent,
                split=split,
                seed=seed,
                condition="clean",
                samples_per_zone_per_parent=1,
                validate_event_identity=False,
            )
            patch = lateral_patch_to_torch(
                numpy_batch,
                device=torch.device("cpu"),
            )
            center = center_trace_batch(patch)
            for batch_index, zone_id in enumerate(center.zone_ids):
                sample_count = int(
                    torch.count_nonzero(center.zone_valid[batch_index]).item()
                )
                emission = torch.zeros(
                    (sample_count, 3),
                    dtype=torch.float32,
                )
                boundary = torch.zeros(
                    (sample_count,),
                    dtype=torch.float32,
                )
                score = hsmm_path_score(
                    emission,
                    boundary,
                    prior.zone(zone_id),
                    truth_hsmm_segments(center, batch_index),
                )
                if not bool(torch.isfinite(score).item()):
                    raise FloatingPointError(
                        "truth HSMM path score is non-finite."
                    )
            patch_count = int(numpy_batch.batch_size)
        except Exception as error:
            return {
                "status": "error",
                "issue": _issue(
                    category="patch_contract",
                    phase="preflight",
                    error=error,
                    parent_id=parent_id,
                    split=split,
                ),
            }
    return {
        "status": "valid",
        "deep_patch_validation": bool(deep),
        "patch_count": patch_count,
        "hsmm_path_count": int(path_count),
        "lateral_count": int(identity["lateral_count"]),
    }


def _run_preflight(
    journal: _EvaluationJournal,
    data: LateralPatchDataModule,
    prior: HsmmPrior,
    manifest: ParentSplitManifest,
    *,
    seed: int,
    logger: logging.Logger,
    progress_every: int,
) -> dict[str, Any]:
    parent_ids = tuple(str(value) for value in journal.plan["preflight_parent_ids"])
    deep_parent_ids = {
        str(value) for value in journal.plan["preflight_deep_parent_ids"]
    }
    split_by_parent = {
        parent_id: split
        for split in ("tuning_validation", "calibration", "geometry_holdout")
        for parent_id in manifest.parent_ids(split)
    }
    pending = journal.pending("preflight", parent_ids)
    started = time.perf_counter()
    for completed_in_call, (index, parent_id) in enumerate(pending, start=1):
        split = split_by_parent[parent_id]
        payload = _preflight_parent(
            data,
            prior,
            parent_id=parent_id,
            split=split,
            seed=seed,
            deep=parent_id in deep_parent_ids,
        )
        journal.commit("preflight", index, parent_id, payload)
        total_completed = len(parent_ids) - len(pending) + completed_in_call
        if (
            completed_in_call == 1
            or total_completed % int(progress_every) == 0
            or total_completed == len(parent_ids)
        ):
            logger.info(
                "evaluation preflight | parents=%d/%d | errors=%d | elapsed=%.1fs",
                total_completed,
                len(parent_ids),
                sum(
                    item.get("status") == "error"
                    for item in journal.phase_payloads(
                        "preflight",
                        parent_ids[:total_completed],
                    )
                ),
                time.perf_counter() - started,
            )
    payloads = journal.phase_payloads("preflight", parent_ids)
    issues = [
        dict(payload["issue"])
        for payload in payloads
        if payload.get("status") == "error"
    ]
    category_counts: dict[str, int] = {}
    for issue in issues:
        category = str(issue["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "failed" if issues else "complete",
        "plan_fingerprint_sha256": journal.plan["fingerprint_sha256"],
        "parent_count": len(parent_ids),
        "deep_parent_count": len(deep_parent_ids),
        "valid_parent_count": len(parent_ids) - len(issues),
        "error_count": len(issues),
        "error_category_counts": category_counts,
        "issues": issues,
    }
    _atomic_json(journal.output_dir / "preflight_report.json", report)
    if issues:
        issue = {
            "category": "preflight_contract",
            "severity": "blocking",
            "phase": "preflight",
            "exception_type": "EvaluationPreflightError",
            "message": (
                f"preflight found {len(issues)} invalid parents; "
                "see preflight_report.json"
            ),
        }
        journal.fail(issue)
        raise RuntimeError(issue["message"])
    return report


_LOSS_KEYS = (
    "loss",
    "structure_loss",
    "emission",
    "boundary",
    "hsmm_nll",
    "teacher",
    "feature_consistency",
    "state_consistency",
)


def _evaluate_parent(
    model: LateralStructuredModel,
    data: LateralPatchDataModule,
    prior: HsmmPrior,
    config: Stage1Step3Config,
    stage: EvaluationStage,
    *,
    parent_id: str,
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    parent = data.benchmark.read_parent(parent_id)
    numpy_batch = data.build_parent_batch(
        parent,
        split=stage.split,
        seed=config.training.base.seed + 500_000,
        condition=stage.condition,
        samples_per_zone_per_parent=config.training.base.samples_per_zone_per_parent,
        maximum_patches_per_parent=config.training.base.maximum_samples_per_parent,
    )
    patch = lateral_patch_to_torch(numpy_batch, device=device)
    with torch.no_grad():
        losses, evidence, _ = lateral_structured_training_loss(
            model,
            patch,
            prior,
            config.loss,
        )
        if not bool(torch.isfinite(losses.total).item()):
            raise FloatingPointError("Stage-1 Step-3 loss became non-finite.")
        posterior = infer_lateral_patch(model, patch, prior, evidence=evidence)
        predicted = predicted_metric_totals(
            posterior,
            center_trace_batch(patch),
        )
    loss_totals = {
        "loss": float(losses.total.detach().cpu()),
        "structure_loss": float(losses.structure.total.detach().cpu()),
        "emission": float(
            losses.structure.emission_cross_entropy.detach().cpu()
        ),
        "boundary": float(
            losses.structure.boundary_binary_cross_entropy.detach().cpu()
        ),
        "hsmm_nll": float(
            losses.structure.hsmm_negative_log_likelihood.detach().cpu()
        ),
        "teacher": float(
            losses.structure.teacher_forcing.total.detach().cpu()
        ),
        "feature_consistency": float(
            losses.feature_consistency.detach().cpu()
        ),
        "state_consistency": float(
            losses.state_consistency.detach().cpu()
        ),
    }
    return {
        "status": "complete",
        "condition": stage.condition,
        "loss_totals": loss_totals,
        "predicted_metric_totals": predicted,
        "batch_count": 1,
        "patch_count": int(numpy_batch.batch_size),
        "trace_count": int(numpy_batch.trace_batch.batch_size),
        "topology_sample_count": int(
            torch.count_nonzero(patch.topology_mask).item()
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _aggregate_stage(
    stage: EvaluationStage,
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not payloads:
        raise ValueError(f"evaluation stage {stage.name!r} has no payloads.")
    loss_totals = {key: 0.0 for key in _LOSS_KEYS}
    predicted_totals: dict[str, float] = {}
    batch_count = 0
    patch_count = 0
    trace_count = 0
    topology_count = 0
    elapsed = 0.0
    for payload in payloads:
        for key in _LOSS_KEYS:
            loss_totals[key] += float(payload["loss_totals"][key])
        merge_metric_totals(
            predicted_totals,
            payload["predicted_metric_totals"],
        )
        batch_count += int(payload["batch_count"])
        patch_count += int(payload["patch_count"])
        trace_count += int(payload["trace_count"])
        topology_count += int(payload["topology_sample_count"])
        elapsed += float(payload["elapsed_seconds"])
    if batch_count <= 0:
        raise ValueError(f"evaluation stage {stage.name!r} has no batches.")
    return {
        "condition": stage.condition,
        "loss": loss_totals["loss"] / batch_count,
        "structure_loss": loss_totals["structure_loss"] / batch_count,
        "emission_cross_entropy": loss_totals["emission"] / batch_count,
        "boundary_binary_cross_entropy": loss_totals["boundary"] / batch_count,
        "hsmm_negative_log_likelihood": loss_totals["hsmm_nll"] / batch_count,
        "teacher_forcing_loss": loss_totals["teacher"] / batch_count,
        "feature_consistency": loss_totals["feature_consistency"] / batch_count,
        "state_consistency": loss_totals["state_consistency"] / batch_count,
        "topology_sample_count": topology_count,
        "batch_count": batch_count,
        "patch_count": patch_count,
        "trace_count": trace_count,
        "parent_count": len(payloads),
        "elapsed_seconds": elapsed,
        "predicted_map": finalize_predicted_metrics(predicted_totals),
    }


def evaluate_stage1_step3_checkpoint(
    config: Stage1Step3Config,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split_manifest_path: str | Path,
    input_mode: str,
    options: Step3EvaluationOptions | None = None,
) -> dict[str, Any]:
    """Preflight and evaluate a frozen checkpoint with atomic parent shards."""
    evaluation_options = options or Step3EvaluationOptions()
    if input_mode not in {"full", "no-seismic"}:
        raise ValueError("input_mode must be full or no-seismic.")
    checkpoint_path = Path(checkpoint_path)
    split_manifest_path = Path(split_manifest_path)
    manifest = ParentSplitManifest.from_dict(
        json.loads(split_manifest_path.read_text(encoding="utf-8"))
    )
    split_payload = manifest.to_dict()
    split_fingerprint = str(split_payload["fingerprint_sha256"])
    benchmark = StructuredSyntheticBenchmark(config.benchmark_dir)
    stages = _evaluation_stages(
        benchmark,
        manifest,
        evaluation_options,
        seed=config.training.base.seed,
    )
    preflight_parent_ids_list: list[str] = []
    for split_index, split in enumerate(
        ("tuning_validation", "calibration", "geometry_holdout")
    ):
        split_parent_ids = manifest.parent_ids(split)
        if evaluation_options.preflight_parent_limit_per_split is not None:
            split_parent_ids = _stratified_parent_ids(
                benchmark,
                split_parent_ids,
                limit=evaluation_options.preflight_parent_limit_per_split,
                seed=config.training.base.seed + 307 + split_index,
            )
        preflight_parent_ids_list.extend(split_parent_ids)
    preflight_parent_ids = tuple(preflight_parent_ids_list)
    if evaluation_options.scope == "final":
        preflight_deep_parent_ids = preflight_parent_ids
    else:
        deep_parent_ids = {
            parent_id
            for stage in stages
            for parent_id in stage.parent_ids
        }
        geometry_parent_set = set(manifest.geometry_holdout)
        geometry_candidates = [
            parent_id
            for parent_id in preflight_parent_ids
            if parent_id in geometry_parent_set
        ]
        if geometry_candidates:
            geometry_deep = _stratified_parent_ids(
                benchmark,
                geometry_candidates,
                limit=min(
                    evaluation_options.diagnostic_parents_per_split,
                    len(geometry_candidates),
                ),
                seed=config.training.base.seed + 419,
            )
            deep_parent_ids.update(geometry_deep)
        preflight_deep_parent_ids = tuple(
            parent_id
            for parent_id in preflight_parent_ids
            if parent_id in deep_parent_ids
        )
    plan_without_fingerprint = {
        "schema": EVALUATION_PLAN_SCHEMA,
        "scope": evaluation_options.scope,
        "input_mode": input_mode,
        "benchmark_dir": str(config.benchmark_dir),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "split_manifest": str(split_manifest_path.resolve()),
        "split_manifest_fingerprint": split_fingerprint,
        "augmentation_profile": config.augmentation.to_dict(),
        "preflight_contract_version": 2,
        "preflight_parent_limit_per_split": (
            evaluation_options.preflight_parent_limit_per_split
        ),
        "preflight_parent_ids": list(preflight_parent_ids),
        "preflight_deep_parent_ids": list(preflight_deep_parent_ids),
        "stages": [stage.to_dict() for stage in stages],
    }
    plan = {
        **plan_without_fingerprint,
        "fingerprint_sha256": _fingerprint(plan_without_fingerprint),
    }
    journal = _EvaluationJournal(
        Path(output_dir),
        plan,
        resume=evaluation_options.resume,
    )
    logger = configure_training_logger(journal.output_dir)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint_mode = str(checkpoint.get("input_mode", ""))
    if checkpoint_mode != input_mode:
        issue = _issue(
            category="checkpoint_contract",
            phase="initialization",
            error=ValueError(
                f"checkpoint input mode {checkpoint_mode!r} differs from "
                f"requested {input_mode!r}."
            ),
        )
        journal.fail(issue)
        raise ValueError(issue["message"])
    prior = _load_prior(checkpoint, split_fingerprint)
    checkpoint_profile = checkpoint.get("augmentation_profile")
    if (
        checkpoint_profile is not None
        and checkpoint_profile != config.augmentation.to_dict()
    ):
        issue = _issue(
            category="checkpoint_contract",
            phase="initialization",
            error=ValueError(
                "evaluation augmentation profile differs from checkpoint profile."
            ),
        )
        journal.fail(issue)
        raise ValueError(issue["message"])
    data = LateralPatchDataModule(
        config.benchmark_dir,
        config.impedance_calibration,
        manifest,
        patch_width=int(checkpoint["model_config"]["patch_width"]),
        augmentation_profile=config.augmentation,
        dirty_probability=config.dirty_probability,
        condition_limit=config.training.base.condition_limit,
    )
    logger.info(
        "stage1 step3 evaluation preflight start | scope=%s | parents=%d | "
        "resume=%s",
        evaluation_options.scope,
        len(preflight_parent_ids),
        evaluation_options.resume,
    )
    preflight = _run_preflight(
        journal,
        data,
        prior,
        manifest,
        seed=config.training.base.seed + 400_000,
        logger=logger,
        progress_every=evaluation_options.progress_every_parents,
    )
    if evaluation_options.preflight_only:
        journal.preflight_complete()
        logger.info(
            "stage1 step3 preflight complete | parents=%d | errors=0",
            preflight["parent_count"],
        )
        return preflight

    device, runtime = resolve_device(config.training.base.device)
    model, loaded_checkpoint = LateralStructuredModel.from_step3_checkpoint(
        checkpoint_path,
        device=device,
    )
    model.eval()
    logger.info(
        "stage1 step3 evaluation start | device=%s | input_mode=%s | "
        "checkpoint_epoch=%s | scope=%s | stages=%d",
        device,
        input_mode,
        loaded_checkpoint.get("epoch", "unknown"),
        evaluation_options.scope,
        len(stages),
    )
    stage_reports: dict[str, Any] = {}
    for stage in stages:
        pending = journal.pending(stage.name, stage.parent_ids)
        existing_count = len(stage.parent_ids) - len(pending)
        stage_started = time.perf_counter()
        for completed_in_call, (index, parent_id) in enumerate(pending, start=1):
            try:
                payload = _evaluate_parent(
                    model,
                    data,
                    prior,
                    config,
                    stage,
                    parent_id=parent_id,
                    device=device,
                )
            except KeyboardInterrupt as error:
                issue = _issue(
                    category="interrupted",
                    severity="recoverable",
                    phase=stage.name,
                    error=error,
                    parent_id=parent_id,
                    split=stage.split,
                    condition=stage.condition,
                )
                journal.fail(issue)
                raise
            except FloatingPointError as error:
                issue = _issue(
                    category="model_numerical",
                    phase=stage.name,
                    error=error,
                    parent_id=parent_id,
                    split=stage.split,
                    condition=stage.condition,
                )
                journal.fail(issue)
                raise
            except Exception as error:
                issue = _issue(
                    category="evaluation_runtime",
                    severity="recoverable",
                    phase=stage.name,
                    error=error,
                    parent_id=parent_id,
                    split=stage.split,
                    condition=stage.condition,
                )
                journal.fail(issue)
                raise
            journal.commit(stage.name, index, parent_id, payload)
            total_completed = existing_count + completed_in_call
            if (
                completed_in_call == 1
                or total_completed % evaluation_options.progress_every_parents
                == 0
                or total_completed == len(stage.parent_ids)
            ):
                logger.info(
                    "evaluation stage=%s | parents=%d/%d | last_parent=%.1fs | "
                    "elapsed=%.1fs",
                    stage.name,
                    total_completed,
                    len(stage.parent_ids),
                    float(payload["elapsed_seconds"]),
                    time.perf_counter() - stage_started,
                )
        stage_reports[stage.name] = _aggregate_stage(
            stage,
            journal.phase_payloads(stage.name, stage.parent_ids),
        )
    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "model_schema": LATERAL_RUN_SCHEMA,
        "status": "complete",
        "evaluation_only": True,
        "scope": evaluation_options.scope,
        "input_mode": input_mode,
        "device": str(device),
        "runtime": runtime,
        "benchmark_dir": str(config.benchmark_dir),
        "impedance_calibration": str(config.impedance_calibration),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "checkpoint_epoch": int(loaded_checkpoint.get("epoch", -1)),
        "split_manifest": str(split_manifest_path),
        "split_manifest_fingerprint": split_fingerprint,
        "plan_fingerprint_sha256": plan["fingerprint_sha256"],
        "preflight": preflight,
        "model": model.config.to_dict(),
        "training": asdict(config.training),
        "augmentation_profile": config.augmentation.to_dict(),
        "split_counts": {
            name: len(manifest.parent_ids(name))
            for name in (
                "training",
                "tuning_validation",
                "calibration",
                "geometry_holdout",
            )
        },
        "evaluated_parent_counts": {
            stage.name: len(stage.parent_ids) for stage in stages
        },
        "hsmm_prior": {
            "duration_bin_count": prior.duration_bin_count,
            "smoothing": prior.smoothing,
            "zones": sorted(prior.zones),
        },
        **stage_reports,
    }
    journal.complete(report)
    logger.info(
        "stage1 step3 evaluation complete | scope=%s | stages=%d",
        evaluation_options.scope,
        len(stages),
    )
    return report


__all__ = [
    "EVALUATION_SCHEMA",
    "Step3EvaluationOptions",
    "evaluate_stage1_step3_checkpoint",
]
