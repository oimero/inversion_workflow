"""Training orchestration for Stage 1 step 2: evidence plus exact HSMM."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from ginn_v2.data import TeacherForcingDataModule, freeze_parent_split_manifest
from ginn_v2.hsmm import HsmmPrior, freeze_hsmm_prior
from ginn_v2.model import (
    SingleTraceStructuredModel,
    TeacherForcingLossConfig,
    TeacherForcingModelConfig,
    batch_to_torch,
)
from ginn_v2.runtime import configure_training_logger, resolve_device
from ginn_v2.structure import (
    StructuredLossConfig,
    infer_center_trace,
    structured_training_loss,
    truth_hsmm_segments,
)
from ginn_v2.training import TeacherForcingTrainingConfig


RUN_SCHEMA = "structured_ginn_v2_stage1_step2_v3"


@dataclass(frozen=True)
class Step2TrainingConfig:
    """Step-2 optimizer controls plus bounded exact posterior evaluation."""

    base: TeacherForcingTrainingConfig
    exact_validation_batches_per_epoch: int = 8
    final_exact_validation_batches: int | None = 32

    def __post_init__(self) -> None:
        if int(self.exact_validation_batches_per_epoch) <= 0:
            raise ValueError("exact_validation_batches_per_epoch must be positive.")
        if (
            self.final_exact_validation_batches is not None
            and int(self.final_exact_validation_batches) <= 0
        ):
            raise ValueError("final_exact_validation_batches must be positive.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Step2TrainingConfig":
        required = {"base"}
        optional = {
            "exact_validation_batches_per_epoch",
            "final_exact_validation_batches",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(required | optional))
        if missing or unknown:
            raise ValueError(
                f"step-2 training mismatch; missing={missing}, unknown={unknown}"
            )
        return cls(
            base=TeacherForcingTrainingConfig.from_mapping(value["base"]),
            exact_validation_batches_per_epoch=int(
                value.get("exact_validation_batches_per_epoch", 8)
            ),
            final_exact_validation_batches=(
                None
                if value.get("final_exact_validation_batches", 32) is None
                else int(value.get("final_exact_validation_batches", 32))
            ),
        )


@dataclass(frozen=True)
class Stage1Step2Config:
    benchmark_dir: Path
    impedance_calibration: Path
    split_seed: int
    initial_checkpoint_full: Path
    initial_checkpoint_no_seismic: Path
    duration_bin_count: int
    prior_smoothing: float
    loss: StructuredLossConfig
    training: Step2TrainingConfig

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        root: str | Path,
    ) -> "Stage1Step2Config":
        required = {
            "benchmark_dir",
            "impedance_calibration",
            "split_seed",
            "initial_checkpoint_full",
            "initial_checkpoint_no_seismic",
            "duration_bin_count",
            "prior_smoothing",
            "loss",
            "training",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(required))
        if missing or unknown:
            raise ValueError(
                f"Stage 1 step 2 config mismatch; missing={missing}, unknown={unknown}"
            )
        base = Path(root)

        def resolve(item: object) -> Path:
            path = Path(str(item))
            return path if path.is_absolute() else (base / path).resolve()

        loss_value = value["loss"]
        if not isinstance(loss_value, Mapping):
            raise TypeError("stage1 step2 loss must be a mapping.")
        required_loss = {
            "emission_weight",
            "boundary_weight",
            "hsmm_nll_weight",
            "teacher_forcing_weight",
            "teacher_forcing",
        }
        loss_missing = sorted(required_loss.difference(loss_value))
        loss_unknown = sorted(set(loss_value).difference(required_loss))
        if loss_missing or loss_unknown:
            raise ValueError(
                f"step-2 loss mismatch; missing={loss_missing}, unknown={loss_unknown}"
            )
        return cls(
            benchmark_dir=resolve(value["benchmark_dir"]),
            impedance_calibration=resolve(value["impedance_calibration"]),
            split_seed=int(value["split_seed"]),
            initial_checkpoint_full=resolve(value["initial_checkpoint_full"]),
            initial_checkpoint_no_seismic=resolve(
                value["initial_checkpoint_no_seismic"]
            ),
            duration_bin_count=int(value["duration_bin_count"]),
            prior_smoothing=float(value["prior_smoothing"]),
            loss=StructuredLossConfig(
                emission_weight=float(loss_value["emission_weight"]),
                boundary_weight=float(loss_value["boundary_weight"]),
                hsmm_nll_weight=float(loss_value["hsmm_nll_weight"]),
                teacher_forcing_weight=float(
                    loss_value["teacher_forcing_weight"]
                ),
                teacher_forcing=TeacherForcingLossConfig.from_mapping(
                    loss_value["teacher_forcing"]
                ),
            ),
            training=Step2TrainingConfig.from_mapping(value["training"]),
        )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_initial_model(
    checkpoint_path: Path,
    *,
    input_mode: str,
    device: torch.device,
    split_fingerprint: str,
) -> tuple[SingleTraceStructuredModel, Mapping[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("split_manifest_fingerprint") != split_fingerprint:
        raise ValueError("Step-1 checkpoint uses a different parent split.")
    model_config = TeacherForcingModelConfig.from_mapping(
        checkpoint["model_config"]
    )
    expected_seismic = input_mode == "full"
    if model_config.use_seismic != expected_seismic:
        raise ValueError(
            f"Step-1 checkpoint use_seismic={model_config.use_seismic} "
            f"does not match input mode {input_mode!r}."
        )
    model = SingleTraceStructuredModel(model_config).to(device)
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )
    allowed_prefixes = ("structure_encoder.", "emission_head.", "boundary_head.")
    invalid_missing = [
        name for name in missing if not name.startswith(allowed_prefixes)
    ]
    if invalid_missing or unexpected:
        raise ValueError(
            f"Step-1 checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    model.freeze_teacher_forcing_modules()
    return model, checkpoint


def _truth_boundaries(
    batch: Any,
    batch_index: int,
) -> set[int]:
    zone = torch.nonzero(
        batch.zone_valid[batch_index],
        as_tuple=False,
    ).flatten()
    first = int(zone[0].item())
    return {
        first + segment.start
        for segment in truth_hsmm_segments(batch, batch_index)[1:]
    }


def predicted_metric_totals(
    posterior: Any,
    batch: Any,
) -> dict[str, float]:
    state_correct = 0
    state_count = 0
    state_correct_by_class = [0, 0, 0]
    state_count_by_class = [0, 0, 0]
    boundary_tp = 0
    boundary_fp = 0
    boundary_fn = 0
    tolerant_tp = 0
    tolerant_fp = 0
    tolerant_fn = 0
    segment_iou_sum = 0.0
    segment_iou_count = 0
    duration_error_sum = 0.0
    duration_error_count = 0
    predicted_segment_count = 0
    truth_segment_count = 0
    for batch_index, result in enumerate(posterior.hsmm_results):
        zone = torch.nonzero(
            batch.zone_valid[batch_index],
            as_tuple=False,
        ).flatten()
        truth_state = batch.truth_state_highres[batch_index, zone]
        predicted_state = posterior.map_state_highres[batch_index, zone]
        state_correct += int(torch.count_nonzero(truth_state == predicted_state).item())
        state_count += int(zone.numel())
        for state_id in range(3):
            state_mask = truth_state == state_id
            state_count_by_class[state_id] += int(
                torch.count_nonzero(state_mask).item()
            )
            state_correct_by_class[state_id] += int(
                torch.count_nonzero(state_mask & (predicted_state == state_id)).item()
            )
        truth_boundary = _truth_boundaries(batch, batch_index)
        first = int(zone[0].item())
        predicted_boundary = {
            first + segment.start for segment in result.consensus_segments[1:]
        }
        boundary_tp += len(truth_boundary & predicted_boundary)
        boundary_fp += len(predicted_boundary - truth_boundary)
        boundary_fn += len(truth_boundary - predicted_boundary)
        unmatched_truth = list(sorted(truth_boundary))
        matched = 0
        for predicted in sorted(predicted_boundary):
            candidates = [
                (abs(predicted - target), index)
                for index, target in enumerate(unmatched_truth)
                if abs(predicted - target) <= int(batch.projection_factor)
            ]
            if not candidates:
                continue
            _, selected = min(candidates)
            unmatched_truth.pop(selected)
            matched += 1
        tolerant_tp += matched
        tolerant_fp += len(predicted_boundary) - matched
        tolerant_fn += len(unmatched_truth)
        truth_segments = truth_hsmm_segments(batch, batch_index)
        predicted_segment_count += len(result.consensus_segments)
        truth_segment_count += len(truth_segments)
        for truth in truth_segments:
            best_iou = 0.0
            best_duration_error: float | None = None
            truth_set = set(range(truth.start, truth.stop))
            for predicted in result.consensus_segments:
                if predicted.state_id != truth.state_id:
                    continue
                predicted_set = set(range(predicted.start, predicted.stop))
                intersection = len(truth_set & predicted_set)
                if intersection == 0:
                    continue
                union = len(truth_set | predicted_set)
                best_iou = max(best_iou, intersection / union)
                error = abs(predicted.duration_samples - truth.duration_samples)
                if best_duration_error is None or error < best_duration_error:
                    best_duration_error = float(error)
            segment_iou_sum += best_iou
            segment_iou_count += 1
            duration_error_sum += (
                float(truth.duration_samples)
                if best_duration_error is None
                else best_duration_error
            )
            duration_error_count += 1
    projection_mask = (
        posterior.predicted_profile.projection_support
        & batch.projected_support
    )
    projected_error = (
        posterior.predicted_profile.projected_log_ai - batch.projected_truth
    ).square()
    projected_squared_error = float(
        torch.sum(projected_error[projection_mask]).detach().cpu()
    )
    projected_count = int(torch.count_nonzero(projection_mask).item())
    highres_error = (
        posterior.predicted_profile.decoded_highres - batch.truth_highres
    ).square()
    highres_mask = batch.zone_valid
    metrics = {
        "state_correct": float(state_correct),
        "state_count": float(state_count),
        "boundary_tp": float(boundary_tp),
        "boundary_fp": float(boundary_fp),
        "boundary_fn": float(boundary_fn),
        "tolerant_boundary_tp": float(tolerant_tp),
        "tolerant_boundary_fp": float(tolerant_fp),
        "tolerant_boundary_fn": float(tolerant_fn),
        "segment_iou_sum": segment_iou_sum,
        "segment_iou_count": float(segment_iou_count),
        "duration_error_sum": duration_error_sum,
        "duration_error_count": float(duration_error_count),
        "predicted_segment_count": float(predicted_segment_count),
        "truth_segment_count": float(truth_segment_count),
        "projected_squared_error": projected_squared_error,
        "projected_count": float(projected_count),
        "highres_squared_error": float(
            torch.sum(highres_error[highres_mask]).detach().cpu()
        ),
        "highres_count": float(torch.count_nonzero(highres_mask).item()),
        "sample_count": float(batch.seismic.shape[0]),
    }
    for state_id in range(3):
        metrics[f"state_{state_id}_correct"] = float(
            state_correct_by_class[state_id]
        )
        metrics[f"state_{state_id}_count"] = float(state_count_by_class[state_id])
    return metrics


def merge_metric_totals(
    target: dict[str, float],
    source: Mapping[str, float],
) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def finalize_predicted_metrics(values: Mapping[str, float]) -> dict[str, Any]:
    if not values:
        return {"sample_count": 0}
    tp = values["boundary_tp"]
    fp = values["boundary_fp"]
    fn = values["boundary_fn"]
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    tolerant_tp = values["tolerant_boundary_tp"]
    tolerant_fp = values["tolerant_boundary_fp"]
    tolerant_fn = values["tolerant_boundary_fn"]
    tolerant_precision = tolerant_tp / max(tolerant_tp + tolerant_fp, 1.0)
    tolerant_recall = tolerant_tp / max(tolerant_tp + tolerant_fn, 1.0)
    state_recalls = [
        values[f"state_{state_id}_correct"]
        / max(values[f"state_{state_id}_count"], 1.0)
        for state_id in range(3)
    ]
    majority_baseline = max(
        values[f"state_{state_id}_count"] for state_id in range(3)
    ) / max(values["state_count"], 1.0)
    state_accuracy = values["state_correct"] / max(values["state_count"], 1.0)
    return {
        "sample_count": int(values["sample_count"]),
        "state_accuracy": state_accuracy,
        "state_balanced_accuracy": float(np.mean(state_recalls)),
        "state_recall_by_class": state_recalls,
        "state_majority_baseline": majority_baseline,
        "state_accuracy_over_majority": state_accuracy - majority_baseline,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "boundary_tolerance_samples": 5,
        "boundary_tolerant_precision": tolerant_precision,
        "boundary_tolerant_recall": tolerant_recall,
        "boundary_tolerant_f1": (
            2.0
            * tolerant_precision
            * tolerant_recall
            / max(tolerant_precision + tolerant_recall, 1e-12)
        ),
        "segment_iou": values["segment_iou_sum"]
        / max(values["segment_iou_count"], 1.0),
        "duration_error_samples": values["duration_error_sum"]
        / max(values["duration_error_count"], 1.0),
        "segment_count_bias": (
            values["predicted_segment_count"] - values["truth_segment_count"]
        )
        / max(values["sample_count"], 1.0),
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


def _run_epoch(
    model: SingleTraceStructuredModel,
    data: TeacherForcingDataModule,
    prior: HsmmPrior,
    config: Stage1Step2Config,
    *,
    split: str,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    logger: logging.Logger,
    exact_batch_limit: int | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    options = config.training.base
    maximum_parents = (
        options.maximum_training_parents
        if training
        else options.maximum_validation_parents
    )
    maximum_batches = (
        options.maximum_training_batches
        if training
        else options.maximum_validation_batches
    )
    iterator = data.iter_batches(
        split,
        batch_size=options.batch_size,
        shuffle=training,
        seed=options.seed + (1009 * epoch if training else 500_000),
        maximum_parents=maximum_parents,
        samples_per_zone_per_parent=options.samples_per_zone_per_parent,
        maximum_samples_per_parent=options.maximum_samples_per_parent,
        boundary_jitter_samples=(
            options.boundary_jitter_samples if training else 0
        ),
    )
    totals = {
        "loss": 0.0,
        "emission": 0.0,
        "boundary": 0.0,
        "hsmm_nll": 0.0,
        "teacher": 0.0,
        "batch_count": 0.0,
        "sample_count": 0.0,
    }
    predicted_totals: dict[str, float] = {}
    started = time.perf_counter()
    previous_parent: str | None = None
    completed_parents = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, numpy_batch in enumerate(iterator):
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            parent_id = numpy_batch.sample_keys[0].rsplit("|", 2)[0]
            if previous_parent is None:
                previous_parent = parent_id
            elif parent_id != previous_parent:
                completed_parents += 1
                previous_parent = parent_id
                if completed_parents % options.progress_log_every_parents == 0:
                    logger.info(
                        "%s epoch %d | parents=%d | batches=%d | samples=%d | "
                        "elapsed=%.1fs",
                        "training" if training else "tuning",
                        epoch + 1,
                        completed_parents,
                        int(totals["batch_count"]),
                        int(totals["sample_count"]),
                        time.perf_counter() - started,
                    )
            batch = batch_to_torch(numpy_batch, device=device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses, evidence, _ = structured_training_loss(
                model,
                batch,
                prior,
                config.loss,
            )
            if not bool(torch.isfinite(losses.total).item()):
                raise FloatingPointError("Step-2 structured loss became non-finite.")
            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    options.gradient_clip_norm,
                )
                optimizer.step()
            totals["loss"] += float(losses.total.detach().cpu())
            totals["emission"] += float(
                losses.emission_cross_entropy.detach().cpu()
            )
            totals["boundary"] += float(
                losses.boundary_binary_cross_entropy.detach().cpu()
            )
            totals["hsmm_nll"] += float(
                losses.hsmm_negative_log_likelihood.detach().cpu()
            )
            totals["teacher"] += float(
                losses.teacher_forcing.total.detach().cpu()
            )
            totals["batch_count"] += 1.0
            totals["sample_count"] += float(numpy_batch.batch_size)
            if (
                not training
                and (
                    exact_batch_limit is None
                    or batch_index < int(exact_batch_limit)
                )
            ):
                posterior = infer_center_trace(
                    model,
                    batch,
                    prior,
                    evidence=evidence,
                )
                merge_metric_totals(
                    predicted_totals,
                    predicted_metric_totals(posterior, batch),
                )
    if previous_parent is not None:
        completed_parents += 1
    if totals["batch_count"] == 0:
        raise ValueError(f"split {split!r} produced no Step-2 batches.")
    denominator = totals["batch_count"]
    return {
        "loss": totals["loss"] / denominator,
        "emission_cross_entropy": totals["emission"] / denominator,
        "boundary_binary_cross_entropy": totals["boundary"] / denominator,
        "hsmm_negative_log_likelihood": totals["hsmm_nll"] / denominator,
        "teacher_forcing_loss": totals["teacher"] / denominator,
        "batch_count": int(totals["batch_count"]),
        "sample_count": int(totals["sample_count"]),
        "parent_count": completed_parents,
        "elapsed_seconds": time.perf_counter() - started,
        "predicted_map": finalize_predicted_metrics(predicted_totals),
    }


def _atomic_torch_save(payload: object, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.staging")
    torch.save(payload, temporary)
    temporary.replace(target)


def run_stage1_step2(
    config: Stage1Step2Config,
    *,
    output_dir: str | Path,
    input_mode: str = "full",
) -> dict[str, Any]:
    """Train and publish the Stage-1 Step-2 single-trace structured model."""
    if input_mode not in {"full", "no-seismic"}:
        raise ValueError("input_mode must be 'full' or 'no-seismic'.")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    logger = configure_training_logger(output)
    options = config.training.base
    _seed_everything(options.seed)
    benchmark = StructuredSyntheticBenchmark(config.benchmark_dir)
    manifest = freeze_parent_split_manifest(
        benchmark,
        output / "parent_split_manifest.json",
        seed=config.split_seed,
    )
    logger.info(
        "HSMM prior calibration start | training_parents=%d | duration_bins=%d",
        len(manifest.training),
        config.duration_bin_count,
    )
    prior = freeze_hsmm_prior(
        benchmark,
        manifest,
        output / "hsmm_prior.json",
        duration_bin_count=config.duration_bin_count,
        smoothing=config.prior_smoothing,
    )
    logger.info(
        "HSMM prior calibration complete | zones=%s",
        sorted(prior.zones),
    )
    data = TeacherForcingDataModule(
        config.benchmark_dir,
        config.impedance_calibration,
        manifest,
        condition_limit=options.condition_limit,
    )
    device, runtime = resolve_device(options.device)
    initial_path = (
        config.initial_checkpoint_full
        if input_mode == "full"
        else config.initial_checkpoint_no_seismic
    )
    model, initial_checkpoint = _load_initial_model(
        initial_path,
        input_mode=input_mode,
        device=device,
        split_fingerprint=str(manifest.to_dict()["fingerprint_sha256"]),
    )
    logger.info(
        "stage1 step2 start | device=%s | input_mode=%s | "
        "training_parents=%d | tuning_parents=%d | initial_epoch=%s",
        device,
        input_mode,
        len(manifest.training),
        len(manifest.tuning_validation),
        initial_checkpoint.get("epoch"),
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    frozen_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )
    logger.info(
        "structure-only optimization | trainable_parameters=%d | "
        "frozen_profile_parameters=%d",
        trainable_count,
        frozen_count,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    checkpoint_path = output / "stage1_step2_checkpoint.pt"
    for epoch in range(options.epochs):
        training_metrics = _run_epoch(
            model,
            data,
            prior,
            config,
            split="training",
            device=device,
            epoch=epoch,
            optimizer=optimizer,
            logger=logger,
            exact_batch_limit=0,
        )
        validation_metrics = _run_epoch(
            model,
            data,
            prior,
            config,
            split="tuning_validation",
            device=device,
            epoch=epoch,
            optimizer=None,
            logger=logger,
            exact_batch_limit=config.training.exact_validation_batches_per_epoch,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "training": training_metrics,
                "tuning_validation": validation_metrics,
            }
        )
        predicted = validation_metrics["predicted_map"]
        logger.info(
            "epoch %d/%d | train_loss=%.6g | tuning_loss=%.6g | "
            "state_acc=%s | balanced_state_acc=%s | vs_majority=%s | "
            "boundary_f1_5m=%s | segment_iou=%s | projected_rmse=%s",
            epoch + 1,
            options.epochs,
            training_metrics["loss"],
            validation_metrics["loss"],
            f"{predicted['state_accuracy']:.4f}"
            if predicted["sample_count"]
            else "n/a",
            f"{predicted['state_balanced_accuracy']:.4f}"
            if predicted["sample_count"]
            else "n/a",
            f"{predicted['state_accuracy_over_majority']:+.4f}"
            if predicted["sample_count"]
            else "n/a",
            f"{predicted['boundary_tolerant_f1']:.4f}"
            if predicted["sample_count"]
            else "n/a",
            f"{predicted['segment_iou']:.4f}"
            if predicted["sample_count"]
            else "n/a",
            f"{predicted['projected_rmse']:.5f}"
            if predicted["sample_count"]
            else "n/a",
        )
        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            _atomic_torch_save(
                {
                    "schema": RUN_SCHEMA,
                    "epoch": best_epoch,
                    "input_mode": input_mode,
                    "model_config": model.config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "split_manifest_fingerprint": manifest.to_dict()[
                        "fingerprint_sha256"
                    ],
                    "hsmm_prior": prior.to_dict(),
                    "initial_checkpoint": str(initial_path),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if (
            epoch + 1 >= options.minimum_epochs
            and epochs_without_improvement >= options.early_stopping_patience
        ):
            stopped_early = True
            logger.info(
                "early stopping | epoch=%d | best_epoch=%d",
                epoch + 1,
                best_epoch,
            )
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    logger.info(
        "final exact tuning evaluation start | best_epoch=%d | max_batches=%s",
        best_epoch,
        config.training.final_exact_validation_batches,
    )
    final_metrics = _run_epoch(
        model,
        data,
        prior,
        config,
        split="tuning_validation",
        device=device,
        epoch=best_epoch - 1,
        optimizer=None,
        logger=logger,
        exact_batch_limit=config.training.final_exact_validation_batches,
    )
    report = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "input_mode": input_mode,
        "device": str(device),
        "runtime": runtime,
        "benchmark_dir": str(config.benchmark_dir),
        "initial_checkpoint": str(initial_path),
        "best_epoch": best_epoch,
        "best_tuning_validation_loss": best_loss,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "history": history,
        "final_tuning_evaluation": final_metrics,
        "model": model.config.to_dict(),
        "loss": {
            "emission_weight": config.loss.emission_weight,
            "boundary_weight": config.loss.boundary_weight,
            "hsmm_nll_weight": config.loss.hsmm_nll_weight,
            "teacher_forcing_weight": config.loss.teacher_forcing_weight,
            "teacher_forcing": asdict(config.loss.teacher_forcing),
        },
        "training": {
            "base": asdict(config.training.base),
            "exact_validation_batches_per_epoch": (
                config.training.exact_validation_batches_per_epoch
            ),
            "final_exact_validation_batches": (
                config.training.final_exact_validation_batches
            ),
        },
        "hsmm_prior": {
            "duration_bin_count": prior.duration_bin_count,
            "smoothing": prior.smoothing,
            "zones": sorted(prior.zones),
        },
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(
        "stage1 step2 complete | best_epoch=%d | final_exact_samples=%d",
        best_epoch,
        final_metrics["predicted_map"]["sample_count"],
    )
    return report


__all__ = [
    "RUN_SCHEMA",
    "Stage1Step2Config",
    "Step2TrainingConfig",
    "finalize_predicted_metrics",
    "merge_metric_totals",
    "predicted_metric_totals",
    "run_stage1_step2",
]
