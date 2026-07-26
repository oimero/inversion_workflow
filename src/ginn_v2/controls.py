"""Paired evidence controls for Stage 1 teacher-forcing checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from cup.utils.logging import configure_run_logger
from ginn_v2.anchor import decode_lfm_anchored_torch
from ginn_v2.data import (
    ParentSplitManifest,
    TeacherForcingBatch,
    TeacherForcingDataModule,
)
from ginn_v2.model import (
    TeacherForcedParameterModel,
    TeacherForcingModelConfig,
    TorchTeacherForcingBatch,
    batch_to_torch,
    project_highres_torch,
)


CONTROL_REPORT_SCHEMA = "structured_ginn_v2_stage1_step1_controls_v1"


def _descriptor_matrix(
    state_id: np.ndarray,
    duration: np.ndarray,
    extent: np.ndarray,
) -> np.ndarray:
    midpoint = 0.5 * (extent[..., 0] + extent[..., 1])
    return np.stack(
        (
            np.ones_like(duration),
            duration,
            duration.square() if hasattr(duration, "square") else duration**2,
            midpoint,
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class StateDurationBaseline:
    """State-specific ridge regressions over duration and segment position."""

    coefficients: np.ndarray
    residual_std: np.ndarray
    target_std: np.ndarray
    sample_count: np.ndarray
    ridge: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        residual_std = np.asarray(self.residual_std, dtype=np.float64)
        target_std = np.asarray(self.target_std, dtype=np.float64).reshape(-1)
        counts = np.asarray(self.sample_count, dtype=np.int64).reshape(-1)
        if coefficients.shape != (3, 4, 3):
            raise ValueError("StateDurationBaseline.coefficients must be [3, 4, 3].")
        if residual_std.shape != (3, 3) or np.any(residual_std <= 0.0):
            raise ValueError("StateDurationBaseline.residual_std must be positive [3, 3].")
        if target_std.shape != (3,) or np.any(target_std <= 0.0):
            raise ValueError("StateDurationBaseline.target_std must be positive [3].")
        if counts.shape != (3,) or np.any(counts < 4):
            raise ValueError("StateDurationBaseline requires at least four samples per state.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "residual_std", residual_std)
        object.__setattr__(self, "target_std", target_std)
        object.__setattr__(self, "sample_count", counts)
        object.__setattr__(self, "ridge", float(self.ridge))

    def predict_numpy(
        self,
        state_id: np.ndarray,
        duration: np.ndarray,
        extent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state_id, dtype=np.int64)
        features = _descriptor_matrix(
            state,
            np.asarray(duration, dtype=np.float64),
            np.asarray(extent, dtype=np.float64),
        )
        mean = np.zeros((*state.shape, 3), dtype=np.float64)
        std = np.ones_like(mean)
        for state_value in range(3):
            mask = state == state_value
            mean[mask] = features[mask] @ self.coefficients[state_value]
            std[mask] = self.residual_std[state_value]
        return mean, std

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": self.coefficients.tolist(),
            "residual_std": self.residual_std.tolist(),
            "target_std": self.target_std.tolist(),
            "sample_count": self.sample_count.tolist(),
            "ridge": self.ridge,
            "features": ["intercept", "duration", "duration_squared", "midpoint"],
        }


def fit_state_duration_baseline(
    batches: Iterator[TeacherForcingBatch],
    *,
    ridge: float = 1e-4,
    logger: Any | None = None,
) -> StateDurationBaseline:
    """Fit the analytic control using identifiable training segments only."""
    features_by_state: list[list[np.ndarray]] = [[], [], []]
    targets_by_state: list[list[np.ndarray]] = [[], [], []]
    all_targets: list[np.ndarray] = []
    current_parent: str | None = None
    completed_parents = 0
    for batch in batches:
        parent_id = batch.sample_keys[0].rsplit("|", 2)[0]
        if current_parent is None:
            current_parent = parent_id
        elif parent_id != current_parent:
            completed_parents += 1
            if logger is not None and completed_parents % 10 == 0:
                logger.info(
                    "state-duration fit | parents=%d",
                    completed_parents,
                )
            current_parent = parent_id
        descriptors = _descriptor_matrix(
            batch.state_id,
            batch.duration_fraction,
            batch.extent_fraction,
        )
        valid = batch.parameter_supervision_valid
        all_targets.append(batch.target_parameters[valid].astype(np.float64))
        for state_value in range(3):
            mask = valid & (batch.state_id == state_value)
            if np.any(mask):
                features_by_state[state_value].append(descriptors[mask])
                targets_by_state[state_value].append(
                    batch.target_parameters[mask].astype(np.float64)
                )
    if current_parent is not None:
        completed_parents += 1
    if not all_targets:
        raise ValueError("state-duration baseline received no identifiable parameters.")
    coefficients = np.zeros((3, 4, 3), dtype=np.float64)
    residual_std = np.zeros((3, 3), dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    penalty = np.diag([0.0, float(ridge), float(ridge), float(ridge)])
    for state_value in range(3):
        if not features_by_state[state_value]:
            raise ValueError(f"state-duration baseline lacks state {state_value}.")
        design = np.concatenate(features_by_state[state_value], axis=0)
        targets = np.concatenate(targets_by_state[state_value], axis=0)
        counts[state_value] = design.shape[0]
        coefficients[state_value] = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ targets,
        )
        residual = targets - design @ coefficients[state_value]
        residual_std[state_value] = np.maximum(
            np.sqrt(np.mean(residual**2, axis=0)),
            1e-3,
        )
    target_std = np.std(np.concatenate(all_targets, axis=0), axis=0)
    return StateDurationBaseline(
        coefficients=coefficients,
        residual_std=residual_std,
        target_std=np.maximum(target_std, 1e-6),
        sample_count=counts,
        ridge=ridge,
    )


class _MetricAccumulator:
    def __init__(self) -> None:
        self.parameter_count = np.zeros(3, dtype=np.int64)
        self.absolute_error = np.zeros(3, dtype=np.float64)
        self.squared_error = np.zeros(3, dtype=np.float64)
        self.nll = np.zeros(3, dtype=np.float64)
        self.target_sum = np.zeros(3, dtype=np.float64)
        self.target_squared_sum = np.zeros(3, dtype=np.float64)
        self.prediction_sum = np.zeros(3, dtype=np.float64)
        self.prediction_squared_sum = np.zeros(3, dtype=np.float64)
        self.cross_sum = np.zeros(3, dtype=np.float64)
        self.sign_correct = np.zeros(3, dtype=np.int64)
        self.sign_count = np.zeros(3, dtype=np.int64)
        self.highres_squared_error = 0.0
        self.highres_count = 0
        self.projected_squared_error = 0.0
        self.projected_count = 0

    def update(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        decoded: torch.Tensor,
        projected: torch.Tensor,
        projection_support: torch.Tensor,
        batch: TorchTeacherForcingBatch,
    ) -> None:
        valid = batch.parameter_supervision_valid
        target = batch.target_parameters
        for coefficient in range(3):
            selected_target = target[..., coefficient][valid].detach().cpu().numpy()
            selected_mean = mean[..., coefficient][valid].detach().cpu().numpy()
            selected_std = std[..., coefficient][valid].detach().cpu().numpy()
            if selected_target.size == 0:
                continue
            error = selected_mean - selected_target
            self.parameter_count[coefficient] += selected_target.size
            self.absolute_error[coefficient] += float(np.sum(np.abs(error)))
            self.squared_error[coefficient] += float(np.sum(error**2))
            self.nll[coefficient] += float(
                np.sum(
                    0.5
                    * (
                        error**2 / np.maximum(selected_std**2, 1e-12)
                        + np.log(np.maximum(selected_std**2, 1e-12))
                    )
                )
            )
            self.target_sum[coefficient] += float(np.sum(selected_target))
            self.target_squared_sum[coefficient] += float(np.sum(selected_target**2))
            self.prediction_sum[coefficient] += float(np.sum(selected_mean))
            self.prediction_squared_sum[coefficient] += float(np.sum(selected_mean**2))
            self.cross_sum[coefficient] += float(np.sum(selected_target * selected_mean))
            sign_mask = np.abs(selected_target) > 1e-6
            self.sign_count[coefficient] += int(np.count_nonzero(sign_mask))
            self.sign_correct[coefficient] += int(
                np.count_nonzero(
                    np.sign(selected_target[sign_mask])
                    == np.sign(selected_mean[sign_mask])
                )
            )
        profile_mask = torch.any(
            batch.segment_mask
            & batch.profile_supervision_valid.unsqueeze(-1),
            dim=1,
        ) & batch.zone_valid
        highres_error = (
            torch.nan_to_num(decoded, nan=0.0) - batch.truth_highres
        ).square()
        self.highres_squared_error += float(
            torch.sum(highres_error[profile_mask]).detach().cpu()
        )
        self.highres_count += int(torch.count_nonzero(profile_mask).item())
        projected_mask = projection_support & batch.projected_support
        projected_error = (projected - batch.projected_truth).square()
        self.projected_squared_error += float(
            torch.sum(projected_error[projected_mask]).detach().cpu()
        )
        self.projected_count += int(torch.count_nonzero(projected_mask).item())

    def metrics(self, *, target_std: np.ndarray) -> dict[str, Any]:
        count = np.maximum(self.parameter_count, 1)
        mae = self.absolute_error / count
        target_variance_mass = self.target_squared_sum - (
            self.target_sum**2 / count
        )
        prediction_variance_mass = self.prediction_squared_sum - (
            self.prediction_sum**2 / count
        )
        covariance_mass = self.cross_sum - (
            self.target_sum * self.prediction_sum / count
        )
        correlation = covariance_mass / np.sqrt(
            np.maximum(target_variance_mass * prediction_variance_mass, 1e-24)
        )
        r_squared = 1.0 - self.squared_error / np.maximum(
            target_variance_mass, 1e-24
        )
        return {
            "parameter_count": self.parameter_count.tolist(),
            "parameter_mae": mae.tolist(),
            "parameter_normalized_mae": (
                mae / np.maximum(target_std, 1e-12)
            ).tolist(),
            "parameter_rmse": np.sqrt(self.squared_error / count).tolist(),
            "parameter_r_squared": r_squared.tolist(),
            "parameter_correlation": correlation.tolist(),
            "parameter_sign_accuracy": (
                self.sign_correct / np.maximum(self.sign_count, 1)
            ).tolist(),
            "parameter_nll": (self.nll / count).tolist(),
            "highres_rmse": float(
                np.sqrt(self.highres_squared_error / max(self.highres_count, 1))
            ),
            "projected_rmse": float(
                np.sqrt(
                    self.projected_squared_error / max(self.projected_count, 1)
                )
            ),
            "highres_count": self.highres_count,
            "projected_count": self.projected_count,
        }


def _load_model(run_dir: Path, *, device: torch.device) -> tuple[
    TeacherForcedParameterModel, Mapping[str, Any]
]:
    checkpoint_path = run_dir / "stage1_step1_checkpoint.pt"
    report_path = run_dir / "training_report.json"
    if not checkpoint_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"incomplete Stage 1 step-1 run: {run_dir}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = TeacherForcingModelConfig.from_mapping(checkpoint["model_config"])
    model = TeacherForcedParameterModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, json.loads(report_path.read_text(encoding="utf-8"))


def _within_parent_zone_seismic_shuffle(
    batch: TorchTeacherForcingBatch,
    sample_keys: tuple[str, ...],
) -> TorchTeacherForcingBatch:
    """Rotate seismic among traces from the same parent and zone."""
    shuffled = batch.seismic.clone()
    zone_ids = [key.rsplit("|", 2)[2] for key in sample_keys]
    for zone_id in sorted(set(zone_ids)):
        indices = [
            index for index, value in enumerate(zone_ids) if value == zone_id
        ]
        if len(indices) < 2:
            continue
        target = torch.as_tensor(indices, dtype=torch.long, device=shuffled.device)
        donor = torch.roll(target, shifts=1)
        shuffled[target] = batch.seismic[donor]
    return replace(batch, seismic=shuffled)


def _parent_metric(
    accumulator: _MetricAccumulator,
    *,
    target_std: np.ndarray,
) -> dict[str, float]:
    metrics = accumulator.metrics(target_std=target_std)
    result = {
        "projected_rmse": float(metrics["projected_rmse"]),
        "highres_rmse": float(metrics["highres_rmse"]),
    }
    for index, name in enumerate(("c0", "c1", "c2")):
        result[f"{name}_mae"] = float(metrics["parameter_mae"][index])
    return result


def _paired_bootstrap(
    parent_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    reference: str,
    candidate: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    parents = sorted(parent_metrics)
    metric_names = sorted(parent_metrics[parents[0]][reference])
    rng = np.random.default_rng(int(seed))
    result: dict[str, Any] = {}
    for metric in metric_names:
        differences = np.asarray(
            [
                parent_metrics[parent][reference][metric]
                - parent_metrics[parent][candidate][metric]
                for parent in parents
            ],
            dtype=np.float64,
        )
        draws = rng.integers(0, differences.size, size=(int(samples), differences.size))
        means = np.mean(differences[draws], axis=1)
        result[metric] = {
            "mean_error_reduction": float(np.mean(differences)),
            "ci95": [
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)),
            ],
            "positive_parent_fraction": float(np.mean(differences > 0.0)),
        }
    return result


def run_stage1_step1_controls(
    *,
    data: TeacherForcingDataModule,
    full_run_dir: str | Path,
    no_seismic_run_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
    training_samples_per_zone: int,
    batch_size: int,
    seed: int,
    bootstrap_samples: int = 2000,
    maximum_training_parents: int | None = None,
    maximum_tuning_parents: int | None = None,
    maximum_samples_per_parent: int | None = None,
) -> dict[str, Any]:
    """Fit the analytic baseline and evaluate all controls on identical samples."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"control output directory already exists: {output}")
    full_model, full_report = _load_model(Path(full_run_dir), device=device)
    no_model, no_report = _load_model(Path(no_seismic_run_dir), device=device)
    if not full_model.config.use_seismic:
        raise ValueError("full checkpoint declares use_seismic=false.")
    if no_model.config.use_seismic:
        raise ValueError("no-seismic checkpoint declares use_seismic=true.")
    full_architecture = dict(full_report["model"])
    no_architecture = dict(no_report["model"])
    full_architecture.pop("use_seismic", None)
    no_architecture.pop("use_seismic", None)
    if full_architecture != no_architecture:
        raise ValueError("full/no-seismic model architectures differ.")
    for field in ("training", "loss", "benchmark_dir", "impedance_calibration"):
        if full_report[field] != no_report[field]:
            raise ValueError(f"full/no-seismic {field} contracts differ.")
    full_fingerprint = json.loads(
        (Path(full_run_dir) / "parent_split_manifest.json").read_text(encoding="utf-8")
    )["fingerprint_sha256"]
    no_fingerprint = json.loads(
        (Path(no_seismic_run_dir) / "parent_split_manifest.json").read_text(encoding="utf-8")
    )["fingerprint_sha256"]
    if full_fingerprint != no_fingerprint:
        raise ValueError("full/no-seismic split manifest fingerprints differ.")
    output.mkdir(parents=True)
    logger = configure_run_logger(
        output,
        logger_name="ginn_v2_controls",
        file_name="controls.log",
    )
    logger.info("fit state-duration baseline start")
    fit_start = time.perf_counter()
    baseline = fit_state_duration_baseline(
        data.iter_batches(
            "training",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            maximum_parents=maximum_training_parents,
            samples_per_zone_per_parent=training_samples_per_zone,
            maximum_samples_per_parent=maximum_samples_per_parent,
            boundary_jitter_samples=0,
        ),
        logger=logger,
    )
    baseline_fit_seconds = time.perf_counter() - fit_start
    logger.info(
        "fit state-duration baseline complete | samples=%s | elapsed=%.1fs",
        baseline.sample_count.tolist(),
        baseline_fit_seconds,
    )
    aggregate = {
        "full": _MetricAccumulator(),
        "no_seismic": _MetricAccumulator(),
        "within_parent_zone_shuffle": _MetricAccumulator(),
        "state_duration": _MetricAccumulator(),
    }
    parent_metrics: dict[str, dict[str, dict[str, float]]] = {}
    iterator = data.iter_batches(
        "tuning_validation",
        batch_size=batch_size,
        shuffle=False,
        seed=seed + 500_000,
        maximum_parents=maximum_tuning_parents,
        samples_per_zone_per_parent=None,
        maximum_samples_per_parent=maximum_samples_per_parent,
        boundary_jitter_samples=0,
    )
    current_parent: str | None = None
    current = {name: _MetricAccumulator() for name in aggregate}
    completed = 0
    evaluation_start = time.perf_counter()
    with torch.no_grad():
        for numpy_batch in iterator:
            parent_id = numpy_batch.sample_keys[0].rsplit("|", 2)[0]
            if current_parent is None:
                current_parent = parent_id
            elif parent_id != current_parent:
                parent_metrics[current_parent] = {
                    name: _parent_metric(
                        accumulator,
                        target_std=baseline.target_std,
                    )
                    for name, accumulator in current.items()
                }
                completed += 1
                if completed % 10 == 0:
                    logger.info(
                        "paired tuning | parents=%d | elapsed=%.1fs",
                        completed,
                        time.perf_counter() - evaluation_start,
                    )
                current_parent = parent_id
                current = {name: _MetricAccumulator() for name in aggregate}
            batch = batch_to_torch(numpy_batch, device=device)
            shuffled_batch = _within_parent_zone_seismic_shuffle(
                batch,
                numpy_batch.sample_keys,
            )
            outputs = {
                "full": full_model(batch),
                "no_seismic": no_model(batch),
                "within_parent_zone_shuffle": full_model(shuffled_batch),
            }
            baseline_mean_np, baseline_std_np = baseline.predict_numpy(
                numpy_batch.state_id,
                numpy_batch.duration_fraction,
                numpy_batch.extent_fraction,
            )
            baseline_mean = torch.as_tensor(
                baseline_mean_np,
                dtype=torch.float32,
                device=device,
            )
            baseline_std = torch.as_tensor(
                baseline_std_np,
                dtype=torch.float32,
                device=device,
            )
            baseline_decoded = decode_lfm_anchored_torch(
                batch.background_highres,
                batch.segment_basis,
                batch.segment_mask,
                baseline_mean,
                batch.zone_valid,
                batch.ai_bounds,
            )
            baseline_projected, baseline_support = project_highres_torch(
                baseline_decoded,
                batch.zone_valid,
                factor=batch.projection_factor,
            )
            values = {
                "full": (
                    outputs["full"].parameter_mean,
                    outputs["full"].parameter_std,
                    outputs["full"].decoded_highres,
                    outputs["full"].projected_log_ai,
                    outputs["full"].projection_support,
                ),
                "no_seismic": (
                    outputs["no_seismic"].parameter_mean,
                    outputs["no_seismic"].parameter_std,
                    outputs["no_seismic"].decoded_highres,
                    outputs["no_seismic"].projected_log_ai,
                    outputs["no_seismic"].projection_support,
                ),
                "within_parent_zone_shuffle": (
                    outputs["within_parent_zone_shuffle"].parameter_mean,
                    outputs["within_parent_zone_shuffle"].parameter_std,
                    outputs["within_parent_zone_shuffle"].decoded_highres,
                    outputs["within_parent_zone_shuffle"].projected_log_ai,
                    outputs["within_parent_zone_shuffle"].projection_support,
                ),
                "state_duration": (
                    baseline_mean,
                    baseline_std,
                    baseline_decoded,
                    baseline_projected,
                    baseline_support,
                ),
            }
            for name, tensors in values.items():
                aggregate[name].update(*tensors, batch)
                current[name].update(*tensors, batch)
    if current_parent is not None:
        parent_metrics[current_parent] = {
            name: _parent_metric(accumulator, target_std=baseline.target_std)
            for name, accumulator in current.items()
        }
        completed += 1
    metrics = {
        name: accumulator.metrics(target_std=baseline.target_std)
        for name, accumulator in aggregate.items()
    }
    report = {
        "schema": CONTROL_REPORT_SCHEMA,
        "status": "complete",
        "full_run_dir": str(Path(full_run_dir).resolve()),
        "no_seismic_run_dir": str(Path(no_seismic_run_dir).resolve()),
        "split_manifest_fingerprint": full_fingerprint,
        "parent_count": completed,
        "state_duration_baseline": baseline.to_dict(),
        "metrics": metrics,
        "paired_bootstrap": {
            "full_vs_no_seismic": _paired_bootstrap(
                parent_metrics,
                reference="no_seismic",
                candidate="full",
                seed=seed,
                samples=bootstrap_samples,
            ),
            "full_vs_state_duration": _paired_bootstrap(
                parent_metrics,
                reference="state_duration",
                candidate="full",
                seed=seed + 1,
                samples=bootstrap_samples,
            ),
            "full_vs_within_parent_zone_shuffle": _paired_bootstrap(
                parent_metrics,
                reference="within_parent_zone_shuffle",
                candidate="full",
                seed=seed + 2,
                samples=bootstrap_samples,
            ),
        },
        "seismic_intervention": {
            "name": "within_parent_zone_shuffle",
            "semantics": (
                "cyclic seismic reassignment within each parent/zone batch; "
                "preserves local donor family but is not the cross-parent "
                "matched-donor intervention"
            ),
        },
        "source_best_epochs": {
            "full": int(full_report["best_epoch"]),
            "no_seismic": int(no_report["best_epoch"]),
        },
        "baseline_fit_seconds": baseline_fit_seconds,
        "elapsed_seconds": time.perf_counter() - evaluation_start,
    }
    (output / "state_duration_baseline.json").write_text(
        json.dumps(baseline.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "control_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("paired controls complete | parents=%d", completed)
    return report


__all__ = [
    "CONTROL_REPORT_SCHEMA",
    "StateDurationBaseline",
    "fit_state_duration_baseline",
    "run_stage1_step1_controls",
]
