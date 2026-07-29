"""Paired controls for the Stage 1 Step 3 lateral slice."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from ginn_v2.hsmm import HsmmPrior
from ginn_v2.lateral import (
    LateralPatchDataModule,
    LateralStructuredModel,
    TorchLateralPatchBatch,
    center_trace_batch,
    infer_lateral_patch,
    lateral_patch_to_torch,
)
from ginn_v2.model import SingleTraceStructuredModel, TeacherForcingModelConfig
from ginn_v2.structure import infer_center_trace
from ginn_v2.structure_training import (
    finalize_predicted_metrics,
    merge_metric_totals,
    predicted_metric_totals,
)


CONTROL_REPORT_SCHEMA = "structured_ginn_v2_stage1_step3_controls_v1"

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


def _replace_seismic(
    patch: TorchLateralPatchBatch,
    seismic: torch.Tensor,
) -> TorchLateralPatchBatch:
    if seismic.shape != patch.trace_batch.seismic.shape:
        raise ValueError("intervention seismic shape differs from the patch.")
    return replace(
        patch,
        trace_batch=replace(patch.trace_batch, seismic=seismic),
    )


def matched_center_seismic_shuffle(
    patch: TorchLateralPatchBatch,
) -> TorchLateralPatchBatch:
    """Replace only center seismic by a same-patch lateral donor."""
    width = patch.patch_width
    if width < 3:
        raise ValueError("matched center shuffle requires at least three patch traces.")
    seismic = patch.trace_batch.seismic.clone()
    for batch_index in range(patch.batch_size):
        valid = torch.nonzero(patch.lateral_valid[batch_index], as_tuple=False).flatten()
        donors = valid[valid != int(patch.center_index)]
        if donors.numel() == 0:
            raise ValueError("matched center shuffle found no lateral donor.")
        donor = donors[torch.argmin(torch.abs(donors - int(patch.center_index)))]
        center_row = batch_index * width + int(patch.center_index)
        donor_row = batch_index * width + int(donor.item())
        seismic[center_row] = patch.trace_batch.seismic[donor_row]
    return _replace_seismic(patch, seismic)


def neighbor_seismic_shuffle(
    patch: TorchLateralPatchBatch,
) -> TorchLateralPatchBatch:
    """Keep center seismic fixed and rotate only the lateral donors."""
    width = patch.patch_width
    seismic = patch.trace_batch.seismic.clone()
    for batch_index in range(patch.batch_size):
        valid = torch.nonzero(patch.lateral_valid[batch_index], as_tuple=False).flatten()
        neighbors = valid[valid != int(patch.center_index)]
        if neighbors.numel() < 2:
            continue
        rows = batch_index * width + neighbors
        seismic[rows] = torch.roll(patch.trace_batch.seismic[rows], shifts=1, dims=0)
    return _replace_seismic(patch, seismic)


def parent_zone_seismic_shuffle(
    patch: TorchLateralPatchBatch,
    donor_by_zone: Mapping[str, torch.Tensor],
) -> TorchLateralPatchBatch:
    """Replace each patch's seismic with another parent's same-zone patch."""
    width = patch.patch_width
    seismic = patch.trace_batch.seismic.clone()
    for batch_index, zone_id in enumerate(patch.zone_ids):
        if zone_id not in donor_by_zone:
            raise KeyError(f"no parent-shuffle donor is available for zone {zone_id!r}.")
        donor = torch.as_tensor(
            donor_by_zone[zone_id],
            dtype=seismic.dtype,
            device=seismic.device,
        )
        if donor.shape != (width, seismic.shape[-1]):
            raise ValueError("parent-shuffle donor shape differs from current patch.")
        start = batch_index * width
        seismic[start : start + width] = donor
    return _replace_seismic(patch, seismic)


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
    parents = sorted(
        parent
        for parent, values in parent_metrics.items()
        if reference in values and candidate in values
    )
    if not parents:
        raise ValueError("lateral paired bootstrap has no common parents.")
    rng = np.random.default_rng(int(seed))
    result: dict[str, Any] = {}
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
        draws = rng.integers(0, differences.size, size=(int(samples), differences.size))
        means = np.mean(differences[draws], axis=1)
        result[metric] = {
            "direction": direction,
            "positive_favors": candidate,
            "mean_improvement": float(np.mean(differences)),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "positive_parent_fraction": float(np.mean(differences > 0.0)),
            "parent_count": len(parents),
        }
    return result


def _load_single_trace_model(
    run_dir: str | Path,
    *,
    device: torch.device,
    expected_mode: str,
) -> tuple[SingleTraceStructuredModel, Mapping[str, Any], Mapping[str, Any]]:
    root = Path(run_dir)
    report = json.loads((root / "training_report.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(root / "stage1_step2_checkpoint.pt", map_location=device, weights_only=True)
    config = TeacherForcingModelConfig.from_mapping(checkpoint["model_config"])
    if config.use_seismic != (expected_mode == "full"):
        raise ValueError("single-trace checkpoint input mode differs from expected mode.")
    model = SingleTraceStructuredModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, report, checkpoint


def run_stage1_step3_controls(
    *,
    data: LateralPatchDataModule,
    full_run_dir: str | Path,
    no_seismic_run_dir: str | Path,
    single_trace_full_run_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    seed: int,
    samples_per_zone_per_parent: int = 1,
    bootstrap_samples: int = 2000,
    maximum_parents: int | None = None,
) -> dict[str, Any]:
    """Evaluate lateral, single-trace, seismic and parent-shuffle controls."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"control output directory already exists: {output}")
    if int(samples_per_zone_per_parent) <= 0 or int(bootstrap_samples) <= 0:
        raise ValueError("control sample counts must be positive.")
    lateral_full, full_checkpoint = LateralStructuredModel.from_step3_checkpoint(
        Path(full_run_dir) / "stage1_step3_checkpoint.pt",
        device=device,
    )
    lateral_no, no_checkpoint = LateralStructuredModel.from_step3_checkpoint(
        Path(no_seismic_run_dir) / "stage1_step3_checkpoint.pt",
        device=device,
    )
    if lateral_full.config.to_dict()["base"]["use_seismic"] is not True:
        raise ValueError("full lateral checkpoint does not use seismic.")
    if lateral_no.config.to_dict()["base"]["use_seismic"] is not False:
        raise ValueError("no-seismic lateral checkpoint uses seismic.")
    full_architecture = lateral_full.config.to_dict()
    no_architecture = lateral_no.config.to_dict()
    full_architecture["base"].pop("use_seismic", None)
    no_architecture["base"].pop("use_seismic", None)
    if full_architecture != no_architecture:
        raise ValueError("full/no-seismic lateral architectures differ.")
    if full_checkpoint["split_manifest_fingerprint"] != no_checkpoint["split_manifest_fingerprint"]:
        raise ValueError("full/no-seismic lateral split manifests differ.")
    prior = HsmmPrior.from_dict(full_checkpoint["hsmm_prior"])
    single_model, single_report, single_checkpoint = _load_single_trace_model(
        single_trace_full_run_dir,
        device=device,
        expected_mode="full",
    )
    if single_checkpoint["split_manifest_fingerprint"] != full_checkpoint["split_manifest_fingerprint"]:
        raise ValueError("single-trace and lateral split manifests differ.")

    conditions = (
        "lateral_full",
        "single_trace_full",
        "lateral_no_seismic",
        "matched_center_seismic_shuffle",
        "neighbor_shuffle",
        "parent_shuffle",
    )
    totals: dict[str, dict[str, float]] = {name: {} for name in conditions}
    parent_metrics: dict[str, dict[str, dict[str, float]]] = {}
    donor_by_zone: dict[str, torch.Tensor] = {}
    output.mkdir(parents=True)
    started = time.perf_counter()
    with torch.no_grad():
        for parent_index, numpy_batch in enumerate(
            data.iter_parent_batches(
                "tuning_validation",
                seed=int(seed),
                condition="clean",
                samples_per_zone_per_parent=samples_per_zone_per_parent,
                maximum_parents=maximum_parents,
            ),
            start=1,
        ):
            patch = lateral_patch_to_torch(
                numpy_batch,
                device=device,
            )
            parent_id = numpy_batch.parent_ids[0]
            shuffled_center = matched_center_seismic_shuffle(patch)
            shuffled_neighbors = neighbor_seismic_shuffle(patch)
            parent_condition = None
            if all(zone_id in donor_by_zone for zone_id in patch.zone_ids):
                parent_condition = parent_zone_seismic_shuffle(patch, donor_by_zone)
            for batch_index, zone_id in enumerate(patch.zone_ids):
                start = batch_index * patch.patch_width
                stop = start + patch.patch_width
                donor_by_zone[zone_id] = patch.trace_batch.seismic[start:stop].detach().clone()
            center = center_trace_batch(patch)
            posteriors = {
                "lateral_full": (lateral_full, patch, infer_lateral_patch(lateral_full, patch, prior)),
                "single_trace_full": (single_model, center, infer_center_trace(single_model, center, prior)),
                "lateral_no_seismic": (lateral_no, patch, infer_lateral_patch(lateral_no, patch, prior)),
                "matched_center_seismic_shuffle": (lateral_full, shuffled_center, infer_lateral_patch(lateral_full, shuffled_center, prior)),
                "neighbor_shuffle": (lateral_full, shuffled_neighbors, infer_lateral_patch(lateral_full, shuffled_neighbors, prior)),
            }
            if parent_condition is not None:
                posteriors["parent_shuffle"] = (
                    lateral_full,
                    parent_condition,
                    infer_lateral_patch(lateral_full, parent_condition, prior),
                )
            parent_values: dict[str, dict[str, float]] = {}
            for name, (_, condition_batch, posterior) in posteriors.items():
                target = center_trace_batch(condition_batch) if isinstance(condition_batch, TorchLateralPatchBatch) else condition_batch
                values = predicted_metric_totals(posterior, target)
                merge_metric_totals(totals[name], values)
                parent_values[name] = _metric_view(finalize_predicted_metrics(values))
            parent_metrics[parent_id] = parent_values
            if parent_index == 1 or parent_index % 10 == 0:
                print(
                    f"Step-3 controls | parents={parent_index} | id={parent_id} | "
                    f"conditions={len(parent_values)} | elapsed={time.perf_counter() - started:.1f}s"
                )
    if not parent_metrics:
        raise ValueError("Step-3 controls produced no parent metrics.")
    aggregate = {
        name: finalize_predicted_metrics(values)
        for name, values in totals.items()
        if values
    }
    paired = {
        "lateral_vs_single_trace": _paired_bootstrap(
            parent_metrics,
            reference="single_trace_full",
            candidate="lateral_full",
            seed=seed,
            samples=bootstrap_samples,
        ),
        "lateral_vs_no_seismic": _paired_bootstrap(
            parent_metrics,
            reference="lateral_no_seismic",
            candidate="lateral_full",
            seed=seed + 1,
            samples=bootstrap_samples,
        ),
        "lateral_vs_matched_center_shuffle": _paired_bootstrap(
            parent_metrics,
            reference="matched_center_seismic_shuffle",
            candidate="lateral_full",
            seed=seed + 2,
            samples=bootstrap_samples,
        ),
        "lateral_vs_neighbor_shuffle": _paired_bootstrap(
            parent_metrics,
            reference="neighbor_shuffle",
            candidate="lateral_full",
            seed=seed + 3,
            samples=bootstrap_samples,
        ),
    }
    if any("parent_shuffle" in values for values in parent_metrics.values()):
        paired["lateral_vs_parent_shuffle"] = _paired_bootstrap(
            parent_metrics,
            reference="parent_shuffle",
            candidate="lateral_full",
            seed=seed + 4,
            samples=bootstrap_samples,
        )
    report = {
        "schema": CONTROL_REPORT_SCHEMA,
        "status": "complete",
        "full_run_dir": str(Path(full_run_dir).resolve()),
        "no_seismic_run_dir": str(Path(no_seismic_run_dir).resolve()),
        "single_trace_full_run_dir": str(Path(single_trace_full_run_dir).resolve()),
        "split_manifest_fingerprint": full_checkpoint["split_manifest_fingerprint"],
        "parent_count": len(parent_metrics),
        "metrics": aggregate,
        "paired_bootstrap": paired,
        "parent_metrics": parent_metrics,
        "interventions": {
            "matched_center_seismic_shuffle": "center seismic replaced by a same-patch lateral donor; LFM/truth/support fixed",
            "neighbor_shuffle": "center seismic fixed; non-center valid traces cyclically rotated",
            "parent_shuffle": "all patch seismic replaced by another parent same-zone donor; LFM/truth/support fixed",
        },
        "source_best_epochs": {
            "lateral_full": int(full_checkpoint["epoch"]),
            "lateral_no_seismic": int(no_checkpoint["epoch"]),
            "single_trace_full": int(single_checkpoint["epoch"]),
            "single_trace_report_best_epoch": int(single_report["best_epoch"]),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "control_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return report


__all__ = [
    "CONTROL_REPORT_SCHEMA",
    "matched_center_seismic_shuffle",
    "neighbor_seismic_shuffle",
    "parent_zone_seismic_shuffle",
    "run_stage1_step3_controls",
]
