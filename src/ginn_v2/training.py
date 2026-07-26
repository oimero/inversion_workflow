"""Training orchestration for Stage 1 implementation step 1."""

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
from ginn_v2.data import (
    ParentSplitManifest,
    TeacherForcingDataModule,
    freeze_parent_split_manifest,
)
from ginn_v2.model import (
    TeacherForcedParameterModel,
    TeacherForcingLossConfig,
    TeacherForcingModelConfig,
    batch_to_torch,
    project_highres_torch,
    teacher_forcing_loss,
)
from ginn_v2.runtime import configure_training_logger, resolve_device


RUN_SCHEMA = "structured_ginn_v2_stage1_step1_v1"


@dataclass(frozen=True)
class TeacherForcingTrainingConfig:
    seed: int = 20260725
    epochs: int = 10
    minimum_epochs: int = 3
    early_stopping_patience: int = 2
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    boundary_jitter_samples: int = 2
    condition_limit: float = 100.0
    samples_per_zone_per_parent: int | None = 8
    progress_log_every_parents: int = 10
    device: str = "auto"
    maximum_training_parents: int | None = None
    maximum_validation_parents: int | None = None
    maximum_samples_per_parent: int | None = None
    maximum_training_batches: int | None = None
    maximum_validation_batches: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "minimum_epochs",
            "early_stopping_patience",
            "batch_size",
            "progress_log_every_parents",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs.")
        if (
            self.samples_per_zone_per_parent is not None
            and int(self.samples_per_zone_per_parent) <= 0
        ):
            raise ValueError("samples_per_zone_per_parent must be positive when supplied.")
        for name in (
            "maximum_training_parents",
            "maximum_validation_parents",
            "maximum_samples_per_parent",
            "maximum_training_batches",
            "maximum_validation_batches",
        ):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when supplied.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer configuration is invalid.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")
        if self.boundary_jitter_samples < 0:
            raise ValueError("boundary_jitter_samples must be non-negative.")
        if self.condition_limit <= 1.0:
            raise ValueError("condition_limit must be greater than one.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TeacherForcingTrainingConfig":
        names = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(names))
        if unknown:
            raise ValueError(f"unknown teacher-forcing training fields: {unknown}")
        return cls(**dict(value))


@dataclass(frozen=True)
class Stage1Step1Config:
    benchmark_dir: Path
    impedance_calibration: Path
    split_seed: int
    model: TeacherForcingModelConfig
    loss: TeacherForcingLossConfig
    training: TeacherForcingTrainingConfig

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        root: str | Path,
    ) -> "Stage1Step1Config":
        required = {
            "benchmark_dir",
            "impedance_calibration",
            "split_seed",
            "model",
            "loss",
            "training",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(required))
        if missing or unknown:
            raise ValueError(
                f"Stage 1 step 1 config mismatch; missing={missing}, unknown={unknown}"
            )
        base = Path(root)

        def resolve(value: object) -> Path:
            path = Path(str(value))
            return path if path.is_absolute() else (base / path).resolve()

        return cls(
            benchmark_dir=resolve(value["benchmark_dir"]),
            impedance_calibration=resolve(value["impedance_calibration"]),
            split_seed=int(value["split_seed"]),
            model=TeacherForcingModelConfig.from_mapping(value["model"]),
            loss=TeacherForcingLossConfig.from_mapping(value["loss"]),
            training=TeacherForcingTrainingConfig.from_mapping(value["training"]),
        )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _batch_limit_reached(index: int, limit: int | None) -> bool:
    return limit is not None and index >= int(limit)


def _run_epoch(
    model: TeacherForcedParameterModel,
    data: TeacherForcingDataModule,
    *,
    split: str,
    config: Stage1Step1Config,
    device: torch.device,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    logger: logging.Logger,
    samples_per_zone_per_parent: int | None,
    phase_name: str,
) -> dict[str, float | int | list[float]]:
    training = optimizer is not None
    model.train(training)
    options = config.training
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
    totals = {
        "loss": 0.0,
        "parameter_nll": 0.0,
        "highres_squared_error": 0.0,
        "projected_squared_error": 0.0,
        "parameter_count": 0,
        "highres_count": 0,
        "projected_count": 0,
        "batch_count": 0,
        "anchor_squared_error": 0.0,
        "anchor_count": 0,
    }
    parameter_targets: list[list[float]] = [[], [], []]
    parameter_predictions: list[list[float]] = [[], [], []]
    fixed_debug_identity = training and options.maximum_training_batches is not None
    available_parents = len(data.split_manifest.parent_ids(split))
    expected_parents = min(
        available_parents,
        available_parents if maximum_parents is None else int(maximum_parents),
    )
    iterator = data.iter_batches(
        split,
        batch_size=options.batch_size,
        shuffle=training,
        seed=(
            options.seed
            if fixed_debug_identity
            else options.seed + 1009 * int(epoch)
            if training
            else options.seed + 500_000
        ),
        maximum_parents=maximum_parents,
        samples_per_zone_per_parent=samples_per_zone_per_parent,
        maximum_samples_per_parent=options.maximum_samples_per_parent,
        boundary_jitter_samples=(
            options.boundary_jitter_samples if training else 0
        ),
    )
    wall_start = time.perf_counter()
    previous_batch_end = wall_start
    data_prepare_seconds = 0.0
    model_step_seconds = 0.0
    completed_parents = 0
    current_parent: str | None = None
    sample_count = 0

    def log_progress(*, force: bool = False) -> None:
        if completed_parents == 0:
            return
        interval = options.progress_log_every_parents
        if not force and completed_parents % interval != 0:
            return
        elapsed = time.perf_counter() - wall_start
        rate = completed_parents / max(elapsed, np.finfo(np.float64).eps)
        remaining = max(expected_parents - completed_parents, 0)
        eta = remaining / max(rate, np.finfo(np.float64).eps)
        logger.info(
            "%s epoch %d | parents=%d/%d | batches=%d | samples=%d | "
            "data=%.1fs | model=%.1fs | elapsed=%.1fs | eta=%.1fs",
            phase_name,
            epoch + 1,
            completed_parents,
            expected_parents,
            totals["batch_count"],
            sample_count,
            data_prepare_seconds,
            model_step_seconds,
            elapsed,
            eta,
        )

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, numpy_batch in enumerate(iterator):
            batch_ready = time.perf_counter()
            data_prepare_seconds += batch_ready - previous_batch_end
            if _batch_limit_reached(batch_index, maximum_batches):
                break
            parent_id = numpy_batch.sample_keys[0].rsplit("|", 2)[0]
            if current_parent is None:
                current_parent = parent_id
            elif parent_id != current_parent:
                completed_parents += 1
                log_progress()
                current_parent = parent_id
            step_start = time.perf_counter()
            batch = batch_to_torch(numpy_batch, device=device)
            sample_count += numpy_batch.batch_size
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = teacher_forcing_loss(output, batch, config.loss)
            if not bool(torch.isfinite(losses.total).item()):
                raise FloatingPointError("teacher-forcing loss is non-finite.")
            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=options.gradient_clip_norm,
                )
                optimizer.step()
            totals["loss"] += float(losses.total.detach().cpu())
            totals["parameter_nll"] += (
                float(losses.parameter_nll.detach().cpu()) * losses.parameter_count
            )
            totals["highres_squared_error"] += (
                float(losses.highres_mse.detach().cpu()) * losses.highres_count
            )
            totals["projected_squared_error"] += (
                float(losses.projected_mse.detach().cpu()) * losses.projected_count
            )
            totals["parameter_count"] += losses.parameter_count
            totals["highres_count"] += losses.highres_count
            totals["projected_count"] += losses.projected_count
            totals["batch_count"] += 1
            anchor_projected, anchor_support = project_highres_torch(
                torch.where(
                    batch.zone_valid,
                    batch.background_highres,
                    torch.full_like(batch.background_highres, float("nan")),
                ),
                batch.zone_valid,
                factor=batch.projection_factor,
            )
            anchor_mask = anchor_support & batch.projected_support
            anchor_error = (anchor_projected - batch.projected_truth).square()
            anchor_count = int(torch.count_nonzero(anchor_mask).item())
            if anchor_count:
                totals["anchor_squared_error"] += float(
                    torch.sum(anchor_error[anchor_mask]).detach().cpu()
                )
                totals["anchor_count"] += anchor_count
            valid_parameters = batch.parameter_supervision_valid
            for coefficient in range(3):
                parameter_targets[coefficient].extend(
                    batch.target_parameters[..., coefficient][valid_parameters]
                    .detach()
                    .cpu()
                    .tolist()
                )
                parameter_predictions[coefficient].extend(
                    output.parameter_mean[..., coefficient][valid_parameters]
                    .detach()
                    .cpu()
                    .tolist()
                )
            model_step_seconds += time.perf_counter() - step_start
            previous_batch_end = time.perf_counter()
    if current_parent is not None:
        completed_parents += 1
        log_progress(force=True)
    if totals["batch_count"] == 0:
        raise ValueError(f"split {split!r} produced no teacher-forcing batches.")
    correlations: list[float] = []
    parameter_mae: list[float] = []
    for targets, predictions in zip(
        parameter_targets, parameter_predictions, strict=True
    ):
        left = np.asarray(targets, dtype=np.float64)
        right = np.asarray(predictions, dtype=np.float64)
        parameter_mae.append(
            float(np.mean(np.abs(left - right))) if left.size else float("nan")
        )
        correlations.append(
            float(np.corrcoef(left, right)[0, 1])
            if left.size >= 2 and np.std(left) > 0.0 and np.std(right) > 0.0
            else float("nan")
        )
    return {
        "loss": totals["loss"] / totals["batch_count"],
        "parameter_nll": totals["parameter_nll"]
        / max(totals["parameter_count"], 1),
        "highres_rmse": float(
            np.sqrt(
                totals["highres_squared_error"] / max(totals["highres_count"], 1)
            )
        ),
        "projected_rmse": float(
            np.sqrt(
                totals["projected_squared_error"]
                / max(totals["projected_count"], 1)
            )
        ),
        "anchor_projected_rmse": float(
            np.sqrt(
                totals["anchor_squared_error"] / max(totals["anchor_count"], 1)
            )
        ),
        "parameter_count": totals["parameter_count"],
        "highres_count": totals["highres_count"],
        "projected_count": totals["projected_count"],
        "batch_count": totals["batch_count"],
        "parent_count": completed_parents,
        "sample_count": sample_count,
        "data_prepare_seconds": data_prepare_seconds,
        "model_step_seconds": model_step_seconds,
        "elapsed_seconds": time.perf_counter() - wall_start,
        "parameter_mae": parameter_mae,
        "parameter_correlation": correlations,
    }


def _atomic_torch_save(payload: object, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.staging")
    torch.save(payload, temporary)
    temporary.replace(target)


def run_stage1_step1(
    config: Stage1Step1Config,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train and publish the first Structured GINN V2 vertical slice."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    logger = configure_training_logger(output)
    _seed_everything(config.training.seed)
    benchmark = StructuredSyntheticBenchmark(config.benchmark_dir)
    split_manifest = freeze_parent_split_manifest(
        benchmark,
        output / "parent_split_manifest.json",
        seed=config.split_seed,
    )
    data = TeacherForcingDataModule(
        config.benchmark_dir,
        config.impedance_calibration,
        split_manifest,
        condition_limit=config.training.condition_limit,
    )
    device, runtime = resolve_device(config.training.device)
    logger.info(
        "stage1 step1 start | device=%s | training_parents=%d | "
        "tuning_parents=%d | use_seismic=%s | "
        "samples_per_zone_parent=%s | max_epochs=%d",
        device,
        len(split_manifest.training),
        len(split_manifest.tuning_validation),
        config.model.use_seismic,
        config.training.samples_per_zone_per_parent,
        config.training.epochs,
    )
    model = TeacherForcedParameterModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False
    checkpoint_path = output / "stage1_step1_checkpoint.pt"
    for epoch in range(config.training.epochs):
        training_metrics = _run_epoch(
            model,
            data,
            split="training",
            config=config,
            device=device,
            epoch=epoch,
            optimizer=optimizer,
            logger=logger,
            samples_per_zone_per_parent=(
                config.training.samples_per_zone_per_parent
            ),
            phase_name="training",
        )
        validation_metrics = _run_epoch(
            model,
            data,
            split="tuning_validation",
            config=config,
            device=device,
            epoch=epoch,
            optimizer=None,
            logger=logger,
            samples_per_zone_per_parent=(
                config.training.samples_per_zone_per_parent
            ),
            phase_name="tuning",
        )
        history.append(
            {
                "epoch": epoch + 1,
                "training": training_metrics,
                "tuning_validation": validation_metrics,
            }
        )
        logger.info(
            "epoch %d/%d | train_loss=%.6g | tuning_loss=%.6g | "
            "tuning_projected_rmse=%.6g | anchor_rmse=%.6g",
            epoch + 1,
            config.training.epochs,
            float(training_metrics["loss"]),
            float(validation_metrics["loss"]),
            float(validation_metrics["projected_rmse"]),
            float(validation_metrics["anchor_projected_rmse"]),
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
                    "model_config": config.model.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "split_manifest_fingerprint": split_manifest.to_dict()[
                        "fingerprint_sha256"
                    ],
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if (
            epoch + 1 >= config.training.minimum_epochs
            and epochs_without_improvement
            >= config.training.early_stopping_patience
        ):
            stopped_early = True
            logger.info(
                "early stopping | epoch=%d | best_epoch=%d | "
                "epochs_without_improvement=%d",
                epoch + 1,
                best_epoch,
                epochs_without_improvement,
            )
            break
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    logger.info(
        "full tuning evaluation start | best_epoch=%d | all lateral traces",
        best_epoch,
    )
    full_tuning_metrics = _run_epoch(
        model,
        data,
        split="tuning_validation",
        config=config,
        device=device,
        epoch=best_epoch - 1,
        optimizer=None,
        logger=logger,
        samples_per_zone_per_parent=None,
        phase_name="tuning-full",
    )
    report = {
        "schema": RUN_SCHEMA,
        "status": "complete",
        "device": str(device),
        "runtime": runtime,
        "benchmark_dir": str(config.benchmark_dir),
        "impedance_calibration": str(config.impedance_calibration),
        "best_epoch": best_epoch,
        "best_tuning_validation_loss": best_loss,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "history": history,
        "full_tuning_evaluation": full_tuning_metrics,
        "model": config.model.to_dict(),
        "loss": asdict(config.loss),
        "training": asdict(config.training),
        "split_counts": {
            name: len(split_manifest.parent_ids(name))
            for name in (
                "training",
                "tuning_validation",
                "calibration",
                "geometry_holdout",
            )
        },
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("stage1 step1 complete | best_epoch=%d", best_epoch)
    return report


__all__ = [
    "RUN_SCHEMA",
    "Stage1Step1Config",
    "TeacherForcingTrainingConfig",
    "run_stage1_step1",
]
