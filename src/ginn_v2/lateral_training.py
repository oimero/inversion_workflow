"""Training and frozen-split evaluation for Stage 1 Step 3."""

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
from ginn_v2.augmentation import ObservationAugmentationProfile
from ginn_v2.data import ParentSplitManifest, freeze_parent_split_manifest
from ginn_v2.hsmm import HsmmPrior
from ginn_v2.lateral import (
    LATERAL_RUN_SCHEMA,
    LateralLoss,
    LateralLossConfig,
    LateralModelConfig,
    LateralPatchDataModule,
    LateralStructuredModel,
    center_trace_batch,
    infer_lateral_patch,
    lateral_patch_to_torch,
    lateral_structured_training_loss,
)
from ginn_v2.model import TeacherForcingLossConfig
from ginn_v2.runtime import configure_training_logger, resolve_device
from ginn_v2.structure import StructuredLossConfig
from ginn_v2.structure_training import (
    finalize_predicted_metrics,
    merge_metric_totals,
    predicted_metric_totals,
)
from ginn_v2.training import TeacherForcingTrainingConfig


@dataclass(frozen=True)
class Step3ModelOptions:
    patch_width: int = 21
    mixer_hidden_channels: int = 64
    mixer_layers: int = 2
    lateral_distance_scale_m: float = 250.0
    feature_consistency_weight: float = 0.05
    state_consistency_weight: float = 0.10

    def __post_init__(self) -> None:
        if int(self.patch_width) <= 0 or int(self.patch_width) % 2 != 1:
            raise ValueError("patch_width must be a positive odd integer.")
        if int(self.mixer_hidden_channels) <= 0 or int(self.mixer_layers) <= 0:
            raise ValueError("lateral mixer sizes must be positive.")
        if not np.isfinite(self.lateral_distance_scale_m) or self.lateral_distance_scale_m <= 0.0:
            raise ValueError("lateral_distance_scale_m must be positive.")
        for name in ("feature_consistency_weight", "state_consistency_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Step3ModelOptions":
        names = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(names))
        if unknown:
            raise ValueError(f"unknown Stage-1 Step-3 model fields: {unknown}")
        return cls(**dict(value))


@dataclass(frozen=True)
class Step3TrainingConfig:
    base: TeacherForcingTrainingConfig
    exact_validation_batches_per_epoch: int = 8
    final_exact_validation_batches: int | None = 32
    final_dirty_validation_batches: int | None = 32
    training_condition: str = "mixed"
    validation_condition: str = "clean"

    def __post_init__(self) -> None:
        for name in ("exact_validation_batches_per_epoch",):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("final_exact_validation_batches", "final_dirty_validation_batches"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when supplied.")
        if self.training_condition not in {"clean", "dirty", "mixed"}:
            raise ValueError("training_condition must be clean, dirty or mixed.")
        if self.validation_condition not in {"clean", "dirty"}:
            raise ValueError("validation_condition must be clean or dirty.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Step3TrainingConfig":
        required = {"base"}
        allowed = required | {
            "exact_validation_batches_per_epoch",
            "final_exact_validation_batches",
            "final_dirty_validation_batches",
            "training_condition",
            "validation_condition",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(allowed))
        if missing or unknown:
            raise ValueError(f"Stage-1 Step-3 training mismatch; missing={missing}, unknown={unknown}")
        return cls(
            base=TeacherForcingTrainingConfig.from_mapping(value["base"]),
            exact_validation_batches_per_epoch=int(value.get("exact_validation_batches_per_epoch", 8)),
            final_exact_validation_batches=(
                None
                if value.get("final_exact_validation_batches", 32) is None
                else int(value.get("final_exact_validation_batches", 32))
            ),
            final_dirty_validation_batches=(
                None
                if value.get("final_dirty_validation_batches", 32) is None
                else int(value.get("final_dirty_validation_batches", 32))
            ),
            training_condition=str(value.get("training_condition", "mixed")),
            validation_condition=str(value.get("validation_condition", "clean")),
        )


@dataclass(frozen=True)
class Stage1Step3Config:
    benchmark_dir: Path
    impedance_calibration: Path
    split_seed: int
    initial_checkpoint_full: Path
    initial_checkpoint_no_seismic: Path
    model: Step3ModelOptions
    loss: LateralLossConfig
    augmentation: ObservationAugmentationProfile
    dirty_probability: float
    training: Step3TrainingConfig

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.dirty_probability) <= 1.0:
            raise ValueError("dirty_probability must be in [0, 1].")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        root: str | Path,
    ) -> "Stage1Step3Config":
        required = {
            "benchmark_dir",
            "impedance_calibration",
            "split_seed",
            "initial_checkpoint_full",
            "initial_checkpoint_no_seismic",
            "model",
            "loss",
            "augmentation",
            "dirty_probability",
            "training",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(required))
        if missing or unknown:
            raise ValueError(f"Stage-1 Step-3 config mismatch; missing={missing}, unknown={unknown}")
        base = Path(root)

        def resolve(item: object) -> Path:
            path = Path(str(item))
            return path if path.is_absolute() else (base / path).resolve()

        loss_value = value["loss"]
        if not isinstance(loss_value, Mapping):
            raise TypeError("Stage-1 Step-3 loss must be a mapping.")
        required_loss = {
            "emission_weight",
            "boundary_weight",
            "hsmm_nll_weight",
            "teacher_forcing_weight",
            "teacher_forcing",
            "feature_consistency_weight",
            "state_consistency_weight",
        }
        loss_missing = sorted(required_loss.difference(loss_value))
        loss_unknown = sorted(set(loss_value).difference(required_loss))
        if loss_missing or loss_unknown:
            raise ValueError(f"Stage-1 Step-3 loss mismatch; missing={loss_missing}, unknown={loss_unknown}")
        teacher_value = loss_value["teacher_forcing"]
        if not isinstance(teacher_value, Mapping):
            raise TypeError("Stage-1 Step-3 teacher_forcing loss must be a mapping.")
        structure_loss = StructuredLossConfig(
            emission_weight=float(loss_value["emission_weight"]),
            boundary_weight=float(loss_value["boundary_weight"]),
            hsmm_nll_weight=float(loss_value["hsmm_nll_weight"]),
            teacher_forcing_weight=float(loss_value["teacher_forcing_weight"]),
            teacher_forcing=TeacherForcingLossConfig.from_mapping(teacher_value),
        )
        return cls(
            benchmark_dir=resolve(value["benchmark_dir"]),
            impedance_calibration=resolve(value["impedance_calibration"]),
            split_seed=int(value["split_seed"]),
            initial_checkpoint_full=resolve(value["initial_checkpoint_full"]),
            initial_checkpoint_no_seismic=resolve(value["initial_checkpoint_no_seismic"]),
            model=Step3ModelOptions.from_mapping(value["model"]),
            loss=LateralLossConfig(
                structure=structure_loss,
                feature_consistency_weight=float(loss_value["feature_consistency_weight"]),
                state_consistency_weight=float(loss_value["state_consistency_weight"]),
            ),
            augmentation=ObservationAugmentationProfile.from_mapping(value["augmentation"]),
            dirty_probability=float(value["dirty_probability"]),
            training=Step3TrainingConfig.from_mapping(value["training"]),
        )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _atomic_torch_save(payload: object, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.staging")
    torch.save(payload, temporary)
    temporary.replace(target)


def _load_prior(checkpoint: Mapping[str, Any], split_fingerprint: str) -> HsmmPrior:
    if str(checkpoint.get("split_manifest_fingerprint")) != str(split_fingerprint):
        raise ValueError("Step-2 checkpoint and Step-3 split manifest differ.")
    prior = HsmmPrior.from_dict(checkpoint["hsmm_prior"])
    if prior.split_manifest_fingerprint != str(split_fingerprint):
        raise ValueError("Step-2 HSMM prior and split manifest differ.")
    return prior


def _run_epoch(
    model: LateralStructuredModel,
    data: LateralPatchDataModule,
    prior: HsmmPrior,
    config: Stage1Step3Config,
    *,
    split: str,
    condition: str,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    logger: logging.Logger,
    exact_batch_limit: int | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    options = config.training.base
    maximum_parents = options.maximum_training_parents if training else options.maximum_validation_parents
    maximum_batches = options.maximum_training_batches if training else options.maximum_validation_batches
    iterator = data.iter_batches(
        split,
        batch_size=options.batch_size,
        shuffle=training,
        seed=options.seed + (1009 * epoch if training else 500_000),
        condition=condition,
        samples_per_zone_per_parent=options.samples_per_zone_per_parent,
        maximum_parents=maximum_parents,
        maximum_patches_per_parent=options.maximum_samples_per_parent,
        boundary_jitter_samples=options.boundary_jitter_samples if training else 0,
    )
    totals = {
        "loss": 0.0,
        "structure_loss": 0.0,
        "emission": 0.0,
        "boundary": 0.0,
        "hsmm_nll": 0.0,
        "teacher": 0.0,
        "feature_consistency": 0.0,
        "state_consistency": 0.0,
        "batch_count": 0.0,
        "sample_count": 0.0,
        "patch_count": 0.0,
        "topology_count": 0.0,
    }
    predicted_totals: dict[str, float] = {}
    started = time.perf_counter()
    seen_parents: set[str] = set()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, numpy_batch in enumerate(iterator):
            if maximum_batches is not None and batch_index >= int(maximum_batches):
                break
            patch = lateral_patch_to_torch(numpy_batch, device=device)
            seen_parents.update(numpy_batch.parent_ids)
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses, evidence, _ = lateral_structured_training_loss(
                model,
                patch,
                prior,
                config.loss,
            )
            if not bool(torch.isfinite(losses.total).item()):
                raise FloatingPointError("Stage-1 Step-3 loss became non-finite.")
            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(),
                    max_norm=options.gradient_clip_norm,
                )
                optimizer.step()
            totals["loss"] += float(losses.total.detach().cpu())
            totals["structure_loss"] += float(losses.structure.total.detach().cpu())
            totals["emission"] += float(losses.structure.emission_cross_entropy.detach().cpu())
            totals["boundary"] += float(losses.structure.boundary_binary_cross_entropy.detach().cpu())
            totals["hsmm_nll"] += float(losses.structure.hsmm_negative_log_likelihood.detach().cpu())
            totals["teacher"] += float(losses.structure.teacher_forcing.total.detach().cpu())
            totals["feature_consistency"] += float(losses.feature_consistency.detach().cpu())
            totals["state_consistency"] += float(losses.state_consistency.detach().cpu())
            totals["topology_count"] += float(torch.count_nonzero(patch.topology_mask).item())
            totals["batch_count"] += 1.0
            totals["patch_count"] += float(numpy_batch.batch_size)
            totals["sample_count"] += float(numpy_batch.trace_batch.batch_size)
            if not training and (exact_batch_limit is None or batch_index < int(exact_batch_limit)):
                posterior = infer_lateral_patch(model, patch, prior, evidence=evidence)
                merge_metric_totals(
                    predicted_totals,
                    predicted_metric_totals(posterior, center_trace_batch(patch)),
                )
            if batch_index == 0 or (batch_index + 1) % max(int(options.progress_log_every_parents), 1) == 0:
                logger.info(
                    "%s epoch %d | batches=%d | patches=%d | parents=%d | condition=%s | "
                    "loss=%.6g | topology=%d | elapsed=%.1fs",
                    "training" if training else "evaluation",
                    epoch + 1,
                    int(totals["batch_count"]),
                    int(totals["patch_count"]),
                    len(seen_parents),
                    condition,
                    float(losses.total.detach().cpu()),
                    int(totals["topology_count"]),
                    time.perf_counter() - started,
                )
    if totals["batch_count"] == 0:
        raise ValueError(f"split {split!r} produced no Stage-1 Step-3 batches.")
    denominator = totals["batch_count"]
    return {
        "condition": condition,
        "loss": totals["loss"] / denominator,
        "structure_loss": totals["structure_loss"] / denominator,
        "emission_cross_entropy": totals["emission"] / denominator,
        "boundary_binary_cross_entropy": totals["boundary"] / denominator,
        "hsmm_negative_log_likelihood": totals["hsmm_nll"] / denominator,
        "teacher_forcing_loss": totals["teacher"] / denominator,
        "feature_consistency": totals["feature_consistency"] / denominator,
        "state_consistency": totals["state_consistency"] / denominator,
        "topology_sample_count": int(totals["topology_count"]),
        "batch_count": int(totals["batch_count"]),
        "patch_count": int(totals["patch_count"]),
        "trace_count": int(totals["sample_count"]),
        "parent_count": len(seen_parents),
        "elapsed_seconds": time.perf_counter() - started,
        "predicted_map": finalize_predicted_metrics(predicted_totals),
    }


def run_stage1_step3(
    config: Stage1Step3Config,
    *,
    output_dir: str | Path,
    input_mode: str = "full",
) -> dict[str, Any]:
    """Train the lateral patch model and freeze clean/dirty split reports."""
    if input_mode not in {"full", "no-seismic"}:
        raise ValueError("input_mode must be full or no-seismic.")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    logger = configure_training_logger(output)
    _seed_everything(config.training.base.seed)
    benchmark = StructuredSyntheticBenchmark(config.benchmark_dir)
    manifest = freeze_parent_split_manifest(
        benchmark,
        output / "parent_split_manifest.json",
        seed=config.split_seed,
    )
    checkpoint_path = config.initial_checkpoint_full if input_mode == "full" else config.initial_checkpoint_no_seismic
    device, runtime = resolve_device(config.training.base.device)
    model, initial_checkpoint, base_model_config = LateralStructuredModel.from_step2_checkpoint(
        checkpoint_path,
        device=device,
        patch_width=config.model.patch_width,
        mixer_hidden_channels=config.model.mixer_hidden_channels,
        mixer_layers=config.model.mixer_layers,
        lateral_distance_scale_m=config.model.lateral_distance_scale_m,
        feature_consistency_weight=config.model.feature_consistency_weight,
        state_consistency_weight=config.model.state_consistency_weight,
    )
    if bool(initial_checkpoint.get("input_mode") == "full") != (input_mode == "full"):
        raise ValueError("Step-2 initial checkpoint input mode differs from requested Step-3 mode.")
    prior = _load_prior(initial_checkpoint, str(manifest.to_dict()["fingerprint_sha256"]))
    data = LateralPatchDataModule(
        config.benchmark_dir,
        config.impedance_calibration,
        manifest,
        patch_width=config.model.patch_width,
        augmentation_profile=config.augmentation,
        dirty_probability=config.dirty_probability,
        condition_limit=config.training.base.condition_limit,
    )
    logger.info(
        "stage1 step3 start | device=%s | input_mode=%s | patch_width=%d | "
        "training_parents=%d | tuning_parents=%d | training_condition=%s",
        device,
        input_mode,
        config.model.patch_width,
        len(manifest.training),
        len(manifest.tuning_validation),
        config.training.training_condition,
    )
    trainable = model.trainable_parameters()
    logger.info(
        "lateral optimization | trainable_parameters=%d | total_parameters=%d",
        sum(parameter.numel() for parameter in trainable),
        sum(parameter.numel() for parameter in model.parameters()),
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.base.learning_rate,
        weight_decay=config.training.base.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    without_improvement = 0
    stopped_early = False
    checkpoint_output = output / "stage1_step3_checkpoint.pt"
    for epoch in range(config.training.base.epochs):
        training_metrics = _run_epoch(
            model,
            data,
            prior,
            config,
            split="training",
            condition=config.training.training_condition,
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
            condition=config.training.validation_condition,
            device=device,
            epoch=epoch,
            optimizer=None,
            logger=logger,
            exact_batch_limit=config.training.exact_validation_batches_per_epoch,
        )
        history.append({"epoch": epoch + 1, "training": training_metrics, "tuning_validation": validation_metrics})
        predicted = validation_metrics["predicted_map"]
        logger.info(
            "epoch %d/%d | train_loss=%.6g | tuning_loss=%.6g | "
            "state_acc=%s | balanced_state_acc=%s | boundary_f1_5m=%s | "
            "segment_iou=%s | projected_rmse=%s",
            epoch + 1,
            config.training.base.epochs,
            training_metrics["loss"],
            validation_metrics["loss"],
            f"{predicted['state_accuracy']:.4f}" if predicted["sample_count"] else "n/a",
            f"{predicted['state_balanced_accuracy']:.4f}" if predicted["sample_count"] else "n/a",
            f"{predicted['boundary_tolerant_f1']:.4f}" if predicted["sample_count"] else "n/a",
            f"{predicted['segment_iou']:.4f}" if predicted["sample_count"] else "n/a",
            f"{predicted['projected_rmse']:.5f}" if predicted["sample_count"] else "n/a",
        )
        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            without_improvement = 0
            _atomic_torch_save(
                {
                    "schema": LATERAL_RUN_SCHEMA,
                    "epoch": best_epoch,
                    "input_mode": input_mode,
                    "model_config": model.config.to_dict(),
                    "base_model_config": base_model_config,
                    "model_state_dict": model.state_dict(),
                    "split_manifest_fingerprint": manifest.to_dict()["fingerprint_sha256"],
                    "hsmm_prior": prior.to_dict(),
                    "augmentation_profile": config.augmentation.to_dict(),
                    "initial_checkpoint": str(checkpoint_path),
                },
                checkpoint_output,
            )
        else:
            without_improvement += 1
        if epoch + 1 >= config.training.base.minimum_epochs and without_improvement >= config.training.base.early_stopping_patience:
            stopped_early = True
            logger.info("early stopping | epoch=%d | best_epoch=%d", epoch + 1, best_epoch)
            break
    checkpoint = torch.load(checkpoint_output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    final_clean = _run_epoch(
        model,
        data,
        prior,
        config,
        split="tuning_validation",
        condition="clean",
        device=device,
        epoch=max(best_epoch - 1, 0),
        optimizer=None,
        logger=logger,
        exact_batch_limit=config.training.final_exact_validation_batches,
    )
    final_dirty = _run_epoch(
        model,
        data,
        prior,
        config,
        split="tuning_validation",
        condition="dirty",
        device=device,
        epoch=max(best_epoch - 1, 0),
        optimizer=None,
        logger=logger,
        exact_batch_limit=config.training.final_dirty_validation_batches,
    )
    calibration_clean = _run_epoch(
        model,
        data,
        prior,
        config,
        split="calibration",
        condition="clean",
        device=device,
        epoch=max(best_epoch - 1, 0),
        optimizer=None,
        logger=logger,
        exact_batch_limit=None,
    )
    calibration_dirty = _run_epoch(
        model,
        data,
        prior,
        config,
        split="calibration",
        condition="dirty",
        device=device,
        epoch=max(best_epoch - 1, 0),
        optimizer=None,
        logger=logger,
        exact_batch_limit=None,
    )
    geometry_clean = _run_epoch(
        model,
        data,
        prior,
        config,
        split="geometry_holdout",
        condition="clean",
        device=device,
        epoch=max(best_epoch - 1, 0),
        optimizer=None,
        logger=logger,
        exact_batch_limit=None,
    )
    report = {
        "schema": LATERAL_RUN_SCHEMA,
        "status": "complete",
        "input_mode": input_mode,
        "device": str(device),
        "runtime": runtime,
        "benchmark_dir": str(config.benchmark_dir),
        "impedance_calibration": str(config.impedance_calibration),
        "initial_checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_tuning_validation_loss": best_loss,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "history": history,
        "final_tuning_clean": final_clean,
        "final_tuning_dirty": final_dirty,
        "calibration_clean": calibration_clean,
        "calibration_dirty": calibration_dirty,
        "geometry_holdout_clean": geometry_clean,
        "model": model.config.to_dict(),
        "loss": {
            "structure": asdict(config.loss.structure),
            "feature_consistency_weight": config.loss.feature_consistency_weight,
            "state_consistency_weight": config.loss.state_consistency_weight,
        },
        "training": asdict(config.training),
        "augmentation_profile": config.augmentation.to_dict(),
        "split_counts": {
            name: len(manifest.parent_ids(name))
            for name in ("training", "tuning_validation", "calibration", "geometry_holdout")
        },
        "split_manifest_fingerprint": manifest.to_dict()["fingerprint_sha256"],
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
        "stage1 step3 complete | best_epoch=%d | clean_projected_rmse=%.6g | dirty_projected_rmse=%.6g",
        best_epoch,
        final_clean["predicted_map"]["projected_rmse"],
        final_dirty["predicted_map"]["projected_rmse"],
    )
    return report


def evaluate_stage1_step3_checkpoint(
    config: Stage1Step3Config,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    split_manifest_path: str | Path,
    input_mode: str,
) -> dict[str, Any]:
    """Evaluate a frozen Step-3 checkpoint without updating its weights.

    The split manifest is an explicit input so a repaired inference seam can
    re-evaluate an interrupted run without silently generating a new split.
    """
    if input_mode not in {"full", "no-seismic"}:
        raise ValueError("input_mode must be full or no-seismic.")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    logger = configure_training_logger(output)
    _seed_everything(config.training.base.seed)

    benchmark = StructuredSyntheticBenchmark(config.benchmark_dir)
    manifest = ParentSplitManifest.from_dict(
        json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    )
    (output / "parent_split_manifest.json").write_text(
        json.dumps(
            manifest.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    split_fingerprint = manifest.to_dict()["fingerprint_sha256"]
    device, runtime = resolve_device(config.training.base.device)
    model, checkpoint = LateralStructuredModel.from_step3_checkpoint(
        checkpoint_path,
        device=device,
    )
    checkpoint_mode = str(checkpoint.get("input_mode", ""))
    if checkpoint_mode != input_mode:
        raise ValueError(
            f"checkpoint input mode {checkpoint_mode!r} differs from requested {input_mode!r}."
        )
    prior = _load_prior(checkpoint, str(split_fingerprint))
    checkpoint_profile = checkpoint.get("augmentation_profile")
    if checkpoint_profile is not None and checkpoint_profile != config.augmentation.to_dict():
        raise ValueError(
            "evaluation augmentation profile differs from the checkpoint profile."
        )
    data = LateralPatchDataModule(
        config.benchmark_dir,
        config.impedance_calibration,
        manifest,
        patch_width=int(model.config.patch_width),
        augmentation_profile=config.augmentation,
        dirty_probability=config.dirty_probability,
        condition_limit=config.training.base.condition_limit,
    )
    logger.info(
        "stage1 step3 evaluation start | device=%s | input_mode=%s | checkpoint_epoch=%s | "
        "tuning_parents=%d | calibration_parents=%d | geometry_parents=%d",
        device,
        input_mode,
        checkpoint.get("epoch", "unknown"),
        len(manifest.tuning_validation),
        len(manifest.calibration),
        len(manifest.geometry_holdout),
    )
    epoch = max(int(checkpoint.get("epoch", 1)) - 1, 0)
    final_clean = _run_epoch(
        model,
        data,
        prior,
        config,
        split="tuning_validation",
        condition="clean",
        device=device,
        epoch=epoch,
        optimizer=None,
        logger=logger,
        exact_batch_limit=config.training.final_exact_validation_batches,
    )
    final_dirty = _run_epoch(
        model,
        data,
        prior,
        config,
        split="tuning_validation",
        condition="dirty",
        device=device,
        epoch=epoch,
        optimizer=None,
        logger=logger,
        exact_batch_limit=config.training.final_dirty_validation_batches,
    )
    calibration_clean = _run_epoch(
        model,
        data,
        prior,
        config,
        split="calibration",
        condition="clean",
        device=device,
        epoch=epoch,
        optimizer=None,
        logger=logger,
        exact_batch_limit=None,
    )
    calibration_dirty = _run_epoch(
        model,
        data,
        prior,
        config,
        split="calibration",
        condition="dirty",
        device=device,
        epoch=epoch,
        optimizer=None,
        logger=logger,
        exact_batch_limit=None,
    )
    geometry_clean = _run_epoch(
        model,
        data,
        prior,
        config,
        split="geometry_holdout",
        condition="clean",
        device=device,
        epoch=epoch,
        optimizer=None,
        logger=logger,
        exact_batch_limit=None,
    )
    report = {
        "schema": LATERAL_RUN_SCHEMA,
        "status": "complete",
        "evaluation_only": True,
        "input_mode": input_mode,
        "device": str(device),
        "runtime": runtime,
        "benchmark_dir": str(config.benchmark_dir),
        "impedance_calibration": str(config.impedance_calibration),
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split_manifest": str(Path(split_manifest_path)),
        "split_manifest_fingerprint": str(split_fingerprint),
        "final_tuning_clean": final_clean,
        "final_tuning_dirty": final_dirty,
        "calibration_clean": calibration_clean,
        "calibration_dirty": calibration_dirty,
        "geometry_holdout_clean": geometry_clean,
        "model": model.config.to_dict(),
        "training": asdict(config.training),
        "augmentation_profile": config.augmentation.to_dict(),
        "split_counts": {
            name: len(manifest.parent_ids(name))
            for name in ("training", "tuning_validation", "calibration", "geometry_holdout")
        },
        "hsmm_prior": {
            "duration_bin_count": prior.duration_bin_count,
            "smoothing": prior.smoothing,
            "zones": sorted(prior.zones),
        },
    }
    (output / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    logger.info(
        "stage1 step3 evaluation complete | checkpoint_epoch=%d | "
        "clean_projected_rmse=%.6g | dirty_projected_rmse=%.6g",
        int(checkpoint.get("epoch", -1)),
        final_clean["predicted_map"]["projected_rmse"],
        final_dirty["predicted_map"]["projected_rmse"],
    )
    return report


__all__ = [
    "RUN_SCHEMA",
    "Stage1Step3Config",
    "Step3ModelOptions",
    "Step3TrainingConfig",
    "evaluate_stage1_step3_checkpoint",
    "run_stage1_step3",
]


RUN_SCHEMA = LATERAL_RUN_SCHEMA
