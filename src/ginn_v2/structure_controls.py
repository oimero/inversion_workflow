"""Paired evidence controls for Stage 1 Step 2 structured checkpoints."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from cup.utils.logging import configure_run_logger
from ginn_v2.data import TeacherForcingDataModule
from ginn_v2.hsmm import HsmmPrior
from ginn_v2.model import (
    SingleTraceStructuredModel,
    TeacherForcingModelConfig,
    TorchTeacherForcingBatch,
    batch_to_torch,
)
from ginn_v2.structure import infer_center_trace
from ginn_v2.structure_training import (
    RUN_SCHEMA as STEP2_RUN_SCHEMA,
    finalize_predicted_metrics,
    merge_metric_totals,
    predicted_metric_totals,
)


CONTROL_REPORT_SCHEMA = "structured_ginn_v2_stage1_step2_controls_v1"

_HIGHER_IS_BETTER = (
    "state_accuracy",
    "state_balanced_accuracy",
    "state_recall_0",
    "state_recall_1",
    "state_recall_2",
    "boundary_f1",
    "boundary_tolerant_f1",
    "segment_iou",
)
_LOWER_IS_BETTER = (
    "duration_error_samples",
    "absolute_segment_count_bias",
    "projected_rmse",
    "highres_rmse",
)


def _load_structured_model(
    run_dir: Path,
    *,
    expected_mode: str,
    device: torch.device,
) -> tuple[SingleTraceStructuredModel, Mapping[str, Any], Mapping[str, Any]]:
    checkpoint_path = run_dir / "stage1_step2_checkpoint.pt"
    report_path = run_dir / "training_report.json"
    if not checkpoint_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"incomplete Stage 1 Step 2 run: {run_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != STEP2_RUN_SCHEMA or report.get("status") != "complete":
        raise ValueError(f"unsupported or incomplete Step 2 report: {run_dir}")
    if report.get("input_mode") != expected_mode:
        raise ValueError(
            f"run {run_dir} has mode {report.get('input_mode')!r}, "
            f"expected {expected_mode!r}."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schema") != STEP2_RUN_SCHEMA:
        raise ValueError(f"unsupported Step 2 checkpoint: {checkpoint_path}")
    if checkpoint.get("input_mode") != expected_mode:
        raise ValueError("checkpoint/report input modes differ.")
    config = TeacherForcingModelConfig.from_mapping(checkpoint["model_config"])
    if config.use_seismic != (expected_mode == "full"):
        raise ValueError("checkpoint use_seismic flag differs from its input mode.")
    model = SingleTraceStructuredModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, report, checkpoint


def _validate_paired_runs(
    full_report: Mapping[str, Any],
    full_checkpoint: Mapping[str, Any],
    no_report: Mapping[str, Any],
    no_checkpoint: Mapping[str, Any],
) -> HsmmPrior:
    full_architecture = dict(full_report["model"])
    no_architecture = dict(no_report["model"])
    full_architecture.pop("use_seismic", None)
    no_architecture.pop("use_seismic", None)
    if full_architecture != no_architecture:
        raise ValueError("full/no-seismic Step 2 architectures differ.")
    for field in ("benchmark_dir", "training", "loss"):
        if full_report[field] != no_report[field]:
            raise ValueError(f"full/no-seismic {field} contracts differ.")
    full_fingerprint = str(full_checkpoint["split_manifest_fingerprint"])
    no_fingerprint = str(no_checkpoint["split_manifest_fingerprint"])
    if full_fingerprint != no_fingerprint:
        raise ValueError("full/no-seismic split fingerprints differ.")
    if full_checkpoint["hsmm_prior"] != no_checkpoint["hsmm_prior"]:
        raise ValueError("full/no-seismic HSMM priors differ.")
    prior = HsmmPrior.from_dict(full_checkpoint["hsmm_prior"])
    if prior.split_manifest_fingerprint != full_fingerprint:
        raise ValueError("HSMM prior and checkpoint split fingerprints differ.")
    return prior


def matched_parent_zone_seismic_shuffle(
    batch: TorchTeacherForcingBatch,
) -> tuple[TorchTeacherForcingBatch, dict[str, Any]]:
    """Rotate donor seismic among laterals within each parent-zone group."""
    parent_ids = [key.rsplit("|", 2)[0] for key in batch.sample_keys]
    if len(set(parent_ids)) != 1:
        raise ValueError("matched seismic intervention requires one complete parent.")
    shuffled = batch.seismic.clone()
    donor_index = torch.arange(
        batch.seismic.shape[0],
        dtype=torch.long,
        device=batch.seismic.device,
    )
    group_sizes: dict[str, int] = {}
    for zone_id in sorted(set(batch.zone_ids)):
        indices = [
            index
            for index, value in enumerate(batch.zone_ids)
            if value == zone_id
        ]
        if len(indices) < 2:
            raise ValueError(
                f"matched seismic intervention requires at least two traces "
                f"for zone {zone_id!r}."
            )
        target = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=batch.seismic.device,
        )
        donor = torch.roll(target, shifts=1)
        shuffled[target] = batch.seismic[donor]
        donor_index[target] = donor
        group_sizes[zone_id] = len(indices)
    if bool(torch.any(donor_index == torch.arange(
        donor_index.numel(),
        device=donor_index.device,
    )).item()):
        raise RuntimeError("matched seismic intervention left a sample unchanged.")
    return (
        replace(batch, seismic=shuffled),
        {
            "parent_id": parent_ids[0],
            "sample_count": int(batch.seismic.shape[0]),
            "zone_group_sizes": group_sizes,
        },
    )


def _metric_view(metrics: Mapping[str, Any]) -> dict[str, float]:
    recalls = metrics["state_recall_by_class"]
    return {
        "state_accuracy": float(metrics["state_accuracy"]),
        "state_balanced_accuracy": float(metrics["state_balanced_accuracy"]),
        "state_recall_0": float(recalls[0]),
        "state_recall_1": float(recalls[1]),
        "state_recall_2": float(recalls[2]),
        "boundary_f1": float(metrics["boundary_f1"]),
        "boundary_tolerant_f1": float(metrics["boundary_tolerant_f1"]),
        "segment_iou": float(metrics["segment_iou"]),
        "duration_error_samples": float(metrics["duration_error_samples"]),
        "absolute_segment_count_bias": abs(float(metrics["segment_count_bias"])),
        "projected_rmse": float(metrics["projected_rmse"]),
        "highres_rmse": float(metrics["highres_rmse"]),
    }


def _paired_bootstrap(
    parent_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    reference: str,
    candidate: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    parents = sorted(parent_metrics)
    if not parents:
        raise ValueError("paired bootstrap received no parents.")
    rng = np.random.default_rng(int(seed))
    output: dict[str, Any] = {}
    for metric in (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER):
        if metric in _HIGHER_IS_BETTER:
            differences = np.asarray(
                [
                    parent_metrics[parent][candidate][metric]
                    - parent_metrics[parent][reference][metric]
                    for parent in parents
                ],
                dtype=np.float64,
            )
            direction = "candidate_minus_reference"
        else:
            differences = np.asarray(
                [
                    parent_metrics[parent][reference][metric]
                    - parent_metrics[parent][candidate][metric]
                    for parent in parents
                ],
                dtype=np.float64,
            )
            direction = "reference_minus_candidate_error"
        draws = rng.integers(
            0,
            differences.size,
            size=(int(samples), differences.size),
        )
        bootstrap_means = np.mean(differences[draws], axis=1)
        output[metric] = {
            "direction": direction,
            "positive_favors": candidate,
            "mean_improvement": float(np.mean(differences)),
            "ci95": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "positive_parent_fraction": float(np.mean(differences > 0.0)),
        }
    return output


def run_stage1_step2_controls(
    *,
    data: TeacherForcingDataModule,
    full_run_dir: str | Path,
    no_seismic_run_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    seed: int,
    samples_per_zone_per_parent: int = 2,
    bootstrap_samples: int = 2000,
    maximum_parents: int | None = None,
    maximum_samples_per_parent: int | None = None,
) -> dict[str, Any]:
    """Evaluate paired structured inference and a matched seismic intervention."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"control output directory already exists: {output}")
    if int(samples_per_zone_per_parent) < 2:
        raise ValueError("matched controls require at least two samples per zone.")
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    full_path = Path(full_run_dir)
    no_path = Path(no_seismic_run_dir)
    full_model, full_report, full_checkpoint = _load_structured_model(
        full_path,
        expected_mode="full",
        device=device,
    )
    no_model, no_report, no_checkpoint = _load_structured_model(
        no_path,
        expected_mode="no-seismic",
        device=device,
    )
    prior = _validate_paired_runs(
        full_report,
        full_checkpoint,
        no_report,
        no_checkpoint,
    )
    expected_fingerprint = str(full_checkpoint["split_manifest_fingerprint"])
    actual_fingerprint = str(
        data.split_manifest.to_dict()["fingerprint_sha256"]
    )
    if actual_fingerprint != expected_fingerprint:
        raise ValueError("control data uses a different parent split.")

    output.mkdir(parents=True)
    logger = configure_run_logger(
        output,
        logger_name="ginn_v2_step2_controls",
        file_name="controls.log",
    )
    totals: dict[str, dict[str, float]] = {
        "full": {},
        "no_seismic": {},
        "matched_seismic_shuffle": {},
    }
    parent_metrics: dict[str, dict[str, dict[str, float]]] = {}
    intervention_groups: dict[str, Any] = {}
    started = time.perf_counter()
    with torch.no_grad():
        for parent_index, numpy_batch in enumerate(
            data.iter_parent_batches(
                "tuning_validation",
                seed=int(seed),
                maximum_parents=maximum_parents,
                samples_per_zone_per_parent=samples_per_zone_per_parent,
                maximum_samples_per_parent=maximum_samples_per_parent,
            ),
            start=1,
        ):
            batch = batch_to_torch(numpy_batch, device=device)
            shuffled_batch, intervention = matched_parent_zone_seismic_shuffle(
                batch
            )
            parent_id = str(intervention["parent_id"])
            if parent_id in parent_metrics:
                raise ValueError(f"duplicate control parent {parent_id!r}.")
            logger.info(
                "paired Step 2 controls | parent=%d | id=%s | samples=%d | start",
                parent_index,
                parent_id,
                batch.seismic.shape[0],
            )
            posteriors = {}
            for name, model, model_batch in (
                ("full", full_model, batch),
                ("no_seismic", no_model, batch),
                (
                    "matched_seismic_shuffle",
                    full_model,
                    shuffled_batch,
                ),
            ):
                condition_started = time.perf_counter()
                posteriors[name] = infer_center_trace(
                    model,
                    model_batch,
                    prior,
                )
                logger.info(
                    "paired Step 2 controls | parent=%d | condition=%s | "
                    "elapsed=%.1fs",
                    parent_index,
                    name,
                    time.perf_counter() - condition_started,
                )
            parent_metrics[parent_id] = {}
            for name, posterior in posteriors.items():
                values = predicted_metric_totals(posterior, batch)
                merge_metric_totals(totals[name], values)
                parent_metrics[parent_id][name] = _metric_view(
                    finalize_predicted_metrics(values)
                )
            intervention_groups[parent_id] = intervention["zone_group_sizes"]
            logger.info(
                "paired Step 2 controls | parents=%d | samples=%d | elapsed=%.1fs",
                parent_index,
                int(totals["full"].get("sample_count", 0.0)),
                time.perf_counter() - started,
            )
    if not parent_metrics:
        raise ValueError("Step 2 controls produced no parent metrics.")
    aggregate = {
        name: finalize_predicted_metrics(values)
        for name, values in totals.items()
    }
    paired = {
        "full_vs_no_seismic": _paired_bootstrap(
            parent_metrics,
            reference="no_seismic",
            candidate="full",
            seed=seed,
            samples=bootstrap_samples,
        ),
        "full_vs_matched_seismic_shuffle": _paired_bootstrap(
            parent_metrics,
            reference="matched_seismic_shuffle",
            candidate="full",
            seed=seed + 1,
            samples=bootstrap_samples,
        ),
    }
    report = {
        "schema": CONTROL_REPORT_SCHEMA,
        "status": "complete",
        "full_run_dir": str(full_path.resolve()),
        "no_seismic_run_dir": str(no_path.resolve()),
        "source_best_epochs": {
            "full": int(full_report["best_epoch"]),
            "no_seismic": int(no_report["best_epoch"]),
        },
        "split_manifest_fingerprint": expected_fingerprint,
        "parent_count": len(parent_metrics),
        "samples_per_zone_per_parent": int(samples_per_zone_per_parent),
        "sample_count": int(aggregate["full"]["sample_count"]),
        "metrics": aggregate,
        "paired_bootstrap": paired,
        "parent_metrics": parent_metrics,
        "seismic_intervention": {
            "name": "matched_seismic_shuffle",
            "semantics": (
                "cyclic donor reassignment among lateral traces from the same "
                "parent and zone; LFM, truth, axis and support remain fixed"
            ),
            "zone_group_sizes_by_parent": intervention_groups,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary = output / ".control_report.json.staging"
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(output / "control_report.json")
    logger.info(
        "paired Step 2 controls complete | parents=%d | samples=%d",
        len(parent_metrics),
        aggregate["full"]["sample_count"],
    )
    return report


__all__ = [
    "CONTROL_REPORT_SCHEMA",
    "matched_parent_zone_seismic_shuffle",
    "run_stage1_step2_controls",
]
