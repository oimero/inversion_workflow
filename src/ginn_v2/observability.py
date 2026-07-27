"""Truth-boundary forward observability diagnostic for Stage 1.

This module is deliberately separate from model training.  It fixes the
published segmentation, state, LFM anchor, nuisance context, and every zone
outside the selected one.  Only identifiable LFM-relative segment parameters
are optimized against model-consistent seismic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from cup.utils.logging import configure_run_logger
from ginn_v2.anchor import (
    LfmAnchoredStructuredSample,
    anchor_to_lfm,
    decode_lfm_anchored_torch,
    load_zone_ai_bounds,
)
from ginn_v2.data import ParentSplitManifest
from ginn_v2.forward import forward_torch
from ginn_v2.model import project_highres_torch
from ginn_v2.oracle import forward_context_from_sample
from ginn_v2.runtime import resolve_device
from ginn_v2.truth import StructuredTruthAdapter


REPORT_SCHEMA = "structured_ginn_v2_truth_boundary_observability_v1"
DECISION_SCHEMA = "structured_ginn_v2_truth_boundary_decision_v1"


@dataclass(frozen=True)
class TruthBoundaryOracleConfig:
    benchmark_dir: Path
    impedance_calibration: Path
    split_manifest: Path
    split: str = "calibration"
    seed: int = 20260726
    device: str = "auto"
    maximum_parents: int = 4
    samples_per_parent: int = 4
    random_starts: int = 5
    optimization_steps: int = 250
    learning_rate: float = 0.03
    early_stopping_patience: int = 50
    minimum_improvement: float = 1e-7
    condition_limit: float = 100.0
    parameter_limits: tuple[float, float, float] = (0.5, 0.25, 0.25)
    initialization_scales: tuple[float, float, float] = (0.18, 0.06, 0.06)
    seismic_margin_samples: int = 10
    target_energy_floor: float = 1e-8
    near_optimal_relative_tolerance: float = 0.02
    near_optimal_absolute_tolerance: float = 1e-5
    truth_init_nrmse_tolerance: float = 1e-4
    fitted_seismic_nrmse_tolerance: float = 0.10
    profile_rmse_tolerance: float = 0.03
    coefficient_normalized_rmse_tolerance: float = 0.50
    coefficient_dispersion_tolerance: float = 0.50
    optimize_identifiable_only: bool = True
    progress_log_every_samples: int = 1
    progress_log_every_steps: int = 25

    def __post_init__(self) -> None:
        if self.split not in (
            "training",
            "tuning_validation",
            "calibration",
            "geometry_holdout",
        ):
            raise ValueError(f"unsupported Oracle split: {self.split!r}")
        for name in (
            "maximum_parents",
            "samples_per_parent",
            "random_starts",
            "optimization_steps",
            "early_stopping_patience",
            "progress_log_every_samples",
            "progress_log_every_steps",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.minimum_improvement < 0.0:
            raise ValueError("minimum_improvement must be non-negative.")
        if self.condition_limit <= 1.0:
            raise ValueError("condition_limit must be greater than one.")
        if self.seismic_margin_samples < 0:
            raise ValueError("seismic_margin_samples must be non-negative.")
        for name in ("parameter_limits", "initialization_scales"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"{name} must contain three positive finite values.")
            object.__setattr__(self, name, values)
        if any(
            scale >= limit
            for scale, limit in zip(
                self.initialization_scales,
                self.parameter_limits,
                strict=True,
            )
        ):
            raise ValueError("initialization_scales must be smaller than parameter_limits.")
        for name in (
            "target_energy_floor",
            "near_optimal_absolute_tolerance",
            "truth_init_nrmse_tolerance",
            "fitted_seismic_nrmse_tolerance",
            "profile_rmse_tolerance",
            "coefficient_normalized_rmse_tolerance",
            "coefficient_dispersion_tolerance",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.near_optimal_relative_tolerance < 0.0:
            raise ValueError("near_optimal_relative_tolerance must be non-negative.")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        root: str | Path,
    ) -> "TruthBoundaryOracleConfig":
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(fields))
        required = {"benchmark_dir", "impedance_calibration", "split_manifest"}
        missing = sorted(required.difference(value))
        if missing or unknown:
            raise ValueError(
                f"truth-boundary Oracle config mismatch; missing={missing}, unknown={unknown}"
            )
        payload = dict(value)
        base = Path(root)
        for name in required:
            path = Path(str(payload[name]))
            payload[name] = path if path.is_absolute() else (base / path).resolve()
        for name in ("parameter_limits", "initialization_scales"):
            if name in payload:
                payload[name] = tuple(float(item) for item in payload[name])
        return cls(**payload)


@dataclass(frozen=True)
class _OptimizationResult:
    summary: Mapping[str, Any]
    starts: tuple[Mapping[str, Any], ...]
    segment_rows: tuple[Mapping[str, Any], ...]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    actual = np.asarray(left, dtype=np.float64).reshape(-1)
    expected = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(actual) & np.isfinite(expected)
    actual = actual[valid]
    expected = expected[valid]
    if (
        actual.size < 2
        or float(np.std(actual)) <= np.finfo(np.float64).eps
        or float(np.std(expected)) <= np.finfo(np.float64).eps
    ):
        return None
    return float(np.corrcoef(actual, expected)[0, 1])


def _rmse(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool)
    actual = np.asarray(left, dtype=np.float64)[valid]
    expected = np.asarray(right, dtype=np.float64)[valid]
    if actual.size == 0 or np.any(~np.isfinite(actual)) or np.any(~np.isfinite(expected)):
        raise ValueError("Oracle metric mask contains no finite comparison samples.")
    return float(np.sqrt(np.mean(np.square(actual - expected))))


def _selected_sample_keys(
    parent: Any,
    *,
    count: int,
    seed: int,
) -> list[tuple[int, str]]:
    candidates = sorted(
        {
            (int(row["lateral_index"]), str(row["zone_id"]))
            for row in parent.zones
            if bool(row.get("zone_valid", True))
        }
    )
    if not candidates:
        raise ValueError(f"parent {parent.identity.realization_id!r} has no valid zones.")
    if len(candidates) <= int(count):
        return candidates
    rng = np.random.default_rng(int(seed))
    by_zone: dict[str, list[tuple[int, str]]] = {}
    for key in candidates:
        by_zone.setdefault(key[1], []).append(key)
    selected: list[tuple[int, str]] = []
    zones = sorted(by_zone)
    base = int(count) // len(zones)
    remainder = int(count) % len(zones)
    for zone_index, zone_id in enumerate(zones):
        quota = base + int(zone_index < remainder)
        if quota == 0:
            continue
        values = by_zone[zone_id]
        quota = min(quota, len(values))
        phase = float(rng.random())
        positions = np.floor(
            (np.arange(quota, dtype=np.float64) + phase) * len(values) / quota
        ).astype(np.int64)
        selected.extend(values[int(index)] for index in positions)
    if len(selected) < int(count):
        remaining = [key for key in candidates if key not in set(selected)]
        selected.extend(remaining[: int(count) - len(selected)])
    return sorted(selected[: int(count)])


def _selected_parent_ids(
    benchmark: StructuredSyntheticBenchmark,
    candidates: Sequence[str],
    *,
    count: int,
    seed: int,
) -> list[str]:
    """Select a deterministic stratum-balanced subset from one frozen split."""
    candidate_set = {str(value) for value in candidates}
    frame = benchmark.index.loc[
        benchmark.index["realization_id"].isin(candidate_set)
    ].copy()
    if set(frame["realization_id"]) != candidate_set:
        raise ValueError("Oracle parent candidates differ from the benchmark index.")
    strata = ("section_id", "duration_mode", "geometry_family")
    missing = sorted(set(strata).difference(frame.columns))
    if missing:
        raise ValueError(f"benchmark index lacks Oracle strata columns: {missing}")
    queues: list[list[str]] = []
    for key, group in frame.groupby(list(strata), sort=True, dropna=False):
        values = sorted(str(value) for value in group["realization_id"])
        digest = hashlib.sha256(
            f"{int(seed)}|{'|'.join(str(item) for item in key)}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        queues.append([values[int(index)] for index in rng.permutation(len(values))])
    selected: list[str] = []
    offset = 0
    while len(selected) < min(int(count), len(candidate_set)):
        added = False
        for queue in queues:
            if offset < len(queue):
                selected.append(queue[offset])
                added = True
                if len(selected) == min(int(count), len(candidate_set)):
                    break
        if not added:
            break
        offset += 1
    if len(selected) != min(int(count), len(candidate_set)):
        raise RuntimeError("stratum-balanced Oracle parent selection is incomplete.")
    return selected


def _parameter_tensor(
    anchored: LfmAnchoredStructuredSample,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_count = len(anchored.segments)
    highres_count = anchored.source.latent.latent_axis.coordinates.size
    basis = torch.zeros(
        (1, segment_count, highres_count, 3),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros(
        (1, segment_count, highres_count),
        dtype=torch.bool,
        device=device,
    )
    truth = torch.empty((segment_count, 3), dtype=torch.float32, device=device)
    identifiable = torch.zeros(segment_count, dtype=torch.bool, device=device)
    for index, segment in enumerate(anchored.segments):
        indices = torch.as_tensor(segment.sample_indices, dtype=torch.long, device=device)
        basis[0, index, indices] = torch.as_tensor(
            segment.basis,
            dtype=torch.float32,
            device=device,
        )
        mask[0, index, indices] = True
        truth[index] = torch.as_tensor(
            segment.effective_parameters_lfm,
            dtype=torch.float32,
            device=device,
        )
        identifiable[index] = bool(segment.parameter_supervision_valid)
    return basis, mask, truth, identifiable


def _decode_full_trace(
    anchored: LfmAnchoredStructuredSample,
    parameters: torch.Tensor,
    *,
    basis: torch.Tensor,
    segment_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    starts = parameters.shape[0]
    background = torch.as_tensor(
        anchored.background_lfm_highres,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).expand(starts, -1)
    zone_valid = torch.as_tensor(
        anchored.source.zone.zone_valid,
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0).expand(starts, -1)
    bounds = torch.as_tensor(
        anchored.ai_bounds,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).expand(starts, -1)
    zone_trace = decode_lfm_anchored_torch(
        background,
        basis.expand(starts, -1, -1, -1),
        segment_mask.expand(starts, -1, -1),
        parameters,
        zone_valid,
        bounds,
    )
    full_truth = torch.as_tensor(
        anchored.source.latent.log_ai_highres_truth,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).expand(starts, -1)
    return torch.where(zone_valid, torch.nan_to_num(zone_trace), full_truth)


def _forward_candidates(
    anchored: LfmAnchoredStructuredSample,
    parameters: torch.Tensor,
    *,
    basis: torch.Tensor,
    segment_mask: torch.Tensor,
    forward_context: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    highres = _decode_full_trace(
        anchored,
        parameters,
        basis=basis,
        segment_mask=segment_mask,
        device=device,
    )
    valid = torch.as_tensor(
        anchored.source.latent.latent_valid,
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0).expand(parameters.shape[0], -1)
    ratio = (
        anchored.source.observed.sample_axis.sample_interval
        / anchored.source.latent.latent_axis.sample_interval
    )
    factor = int(round(ratio))
    projected, support = project_highres_torch(highres, valid, factor=factor)
    seismic = forward_torch(forward_context, projected)
    return highres, projected, torch.as_tensor(seismic)


def _seismic_mask(
    anchored: LfmAnchoredStructuredSample,
    projection_support: torch.Tensor,
    *,
    margin_samples: int,
    device: torch.device,
) -> torch.Tensor:
    axis = np.asarray(
        anchored.source.observed.sample_axis.coordinates,
        dtype=np.float64,
    )
    margin = float(margin_samples) * anchored.source.observed.sample_axis.sample_interval
    zone = (axis >= anchored.source.zone.top - margin) & (
        axis <= anchored.source.zone.bottom + margin
    )
    target = np.asarray(
        anchored.source.observed.model_consistent_seismic,
        dtype=np.float64,
    )
    mask = (
        zone
        & np.asarray(anchored.source.observed.observed_valid, dtype=bool)
        & np.isfinite(target)
        & projection_support[0].detach().cpu().numpy()
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError("truth-boundary Oracle has fewer than three seismic samples.")
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def _candidate_metrics(
    anchored: LfmAnchoredStructuredSample,
    parameters: np.ndarray,
    highres: np.ndarray,
    projected: np.ndarray,
    seismic: np.ndarray,
    *,
    seismic_mask: np.ndarray,
    projection_support: np.ndarray,
) -> dict[str, Any]:
    source = anchored.source
    truth_parameters = np.stack(
        [segment.effective_parameters_lfm for segment in anchored.segments],
        axis=0,
    )
    identifiable = np.asarray(
        [segment.parameter_supervision_valid for segment in anchored.segments],
        dtype=bool,
    )
    zone_mask = np.asarray(source.zone.zone_valid, dtype=bool)
    model_axis = np.asarray(source.observed.sample_axis.coordinates, dtype=np.float64)
    model_zone = (
        (model_axis >= source.zone.top)
        & (model_axis <= source.zone.bottom)
        & np.asarray(projection_support, dtype=bool)
        & np.asarray(source.observed.observed_valid, dtype=bool)
    )
    truth_highres = np.asarray(source.latent.log_ai_highres_truth, dtype=np.float64)
    from ginn_v2.oracle import project_log_ai_to_model_grid

    projected_truth = project_log_ai_to_model_grid(
        truth_highres,
        source.latent.latent_axis,
        source.observed.sample_axis,
    ).model_log_ai
    target_seismic = np.asarray(source.observed.model_consistent_seismic, dtype=np.float64)
    seismic_rmse = _rmse(seismic, target_seismic, seismic_mask)
    target_rms = float(np.sqrt(np.mean(np.square(target_seismic[seismic_mask]))))
    result: dict[str, Any] = {
        "seismic_rmse": seismic_rmse,
        "seismic_normalized_rmse": seismic_rmse / max(
            target_rms,
            np.finfo(np.float64).eps,
        ),
        "seismic_correlation": _finite_correlation(
            seismic[seismic_mask],
            target_seismic[seismic_mask],
        ),
        "profile_rmse": _rmse(highres, truth_highres, zone_mask),
        "projected_log_ai_rmse": _rmse(projected, projected_truth, model_zone),
        "identifiable_segment_count": int(np.count_nonzero(identifiable)),
    }
    coefficient_names = ("c0", "c1", "c2")
    for coefficient, name in enumerate(coefficient_names):
        predicted = parameters[identifiable, coefficient]
        truth = truth_parameters[identifiable, coefficient]
        if truth.size == 0:
            result[f"{name}_mae"] = None
            result[f"{name}_rmse"] = None
            result[f"{name}_normalized_rmse"] = None
            result[f"{name}_correlation"] = None
            continue
        rmse = float(np.sqrt(np.mean(np.square(predicted - truth))))
        scale = max(float(np.std(truth)), 0.01)
        result[f"{name}_mae"] = float(np.mean(np.abs(predicted - truth)))
        result[f"{name}_rmse"] = rmse
        result[f"{name}_normalized_rmse"] = rmse / scale
        result[f"{name}_correlation"] = _finite_correlation(predicted, truth)
    return result


def _optimize_sample(
    anchored: LfmAnchoredStructuredSample,
    config: TruthBoundaryOracleConfig,
    *,
    device: torch.device,
    seed: int,
    logger: logging.Logger,
    sample_label: str,
) -> _OptimizationResult:
    basis, segment_mask, truth_parameters, identifiable = _parameter_tensor(
        anchored,
        device=device,
    )
    if not config.optimize_identifiable_only:
        identifiable = torch.as_tensor(
            [item.profile_supervision_valid for item in anchored.segments],
            dtype=torch.bool,
            device=device,
        )
    if int(torch.count_nonzero(identifiable).item()) == 0:
        raise ValueError("selected sample has no optimizable structured segments.")
    limits = torch.as_tensor(
        config.parameter_limits,
        dtype=torch.float32,
        device=device,
    )
    if bool(torch.any(torch.abs(truth_parameters[identifiable]) >= limits).item()):
        raise ValueError("truth parameters exceed configured Oracle parameter_limits.")
    forward_context = forward_context_from_sample(anchored.source)

    truth_batch = truth_parameters.unsqueeze(0)
    truth_highres, _, truth_seismic = _forward_candidates(
        anchored,
        truth_batch,
        basis=basis,
        segment_mask=segment_mask,
        forward_context=forward_context,
        device=device,
    )
    full_valid = torch.as_tensor(
        anchored.source.latent.latent_valid,
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    ratio = (
        anchored.source.observed.sample_axis.sample_interval
        / anchored.source.latent.latent_axis.sample_interval
    )
    _, projection_support = project_highres_torch(
        truth_highres,
        full_valid,
        factor=int(round(ratio)),
    )
    score_mask = _seismic_mask(
        anchored,
        projection_support,
        margin_samples=config.seismic_margin_samples,
        device=device,
    )
    target = torch.as_tensor(
        anchored.source.observed.model_consistent_seismic,
        dtype=torch.float32,
        device=device,
    )
    target_energy = torch.clamp(
        torch.mean(torch.square(target[score_mask])),
        min=float(config.target_energy_floor),
    )
    truth_loss = torch.mean(
        torch.square(truth_seismic[0, score_mask] - target[score_mask])
    ) / target_energy

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    desired = torch.zeros(
        (config.random_starts, len(anchored.segments), 3),
        dtype=torch.float32,
    )
    if config.random_starts > 1:
        noise = torch.randn(
            (config.random_starts - 1, len(anchored.segments), 3),
            generator=generator,
            dtype=torch.float32,
        )
        scales = torch.as_tensor(config.initialization_scales, dtype=torch.float32)
        desired[1:] = torch.clamp(
            noise * scales,
            min=-0.95 * torch.as_tensor(config.parameter_limits),
            max=0.95 * torch.as_tensor(config.parameter_limits),
        )
    desired = desired.to(device=device)
    raw = torch.nn.Parameter(
        torch.atanh(torch.clamp(desired / limits, min=-0.999, max=0.999))
    )
    optimizer = torch.optim.Adam([raw], lr=float(config.learning_rate))
    fixed = truth_parameters.unsqueeze(0).expand(config.random_starts, -1, -1)
    optimize_mask = identifiable.view(1, -1, 1)
    with torch.no_grad():
        initial_parameters = torch.where(
            optimize_mask,
            limits * torch.tanh(raw),
            fixed,
        )
        _, _, initial_seismic = _forward_candidates(
            anchored,
            initial_parameters,
            basis=basis,
            segment_mask=segment_mask,
            forward_context=forward_context,
            device=device,
        )
        initial_losses = torch.mean(
            torch.square(
                initial_seismic[:, score_mask] - target[score_mask].unsqueeze(0)
            ),
            dim=1,
        ) / target_energy
    best_loss = torch.full(
        (config.random_starts,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    best_parameters = fixed.detach().clone()
    best_step = torch.zeros(config.random_starts, dtype=torch.long, device=device)
    stale_steps = 0
    completed_steps = 0
    for step in range(config.optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        bounded = limits * torch.tanh(raw)
        parameters = torch.where(optimize_mask, bounded, fixed)
        _, _, seismic = _forward_candidates(
            anchored,
            parameters,
            basis=basis,
            segment_mask=segment_mask,
            forward_context=forward_context,
            device=device,
        )
        losses = torch.mean(
            torch.square(seismic[:, score_mask] - target[score_mask].unsqueeze(0)),
            dim=1,
        ) / target_energy
        mean_loss = torch.mean(losses)
        if not bool(torch.isfinite(mean_loss).item()):
            raise FloatingPointError("truth-boundary Oracle optimization became non-finite.")
        mean_loss.backward()
        optimizer.step()
        completed_steps = step + 1
        with torch.no_grad():
            improved = losses < best_loss - float(config.minimum_improvement)
            if bool(torch.any(improved).item()):
                best_loss = torch.where(improved, losses, best_loss)
                best_parameters[improved] = parameters.detach()[improved]
                best_step[improved] = step + 1
                stale_steps = 0
            else:
                stale_steps += 1
        if (
            completed_steps % config.progress_log_every_steps == 0
            or completed_steps == config.optimization_steps
            or stale_steps >= config.early_stopping_patience
        ):
            logger.info(
                "truth-boundary optimize | sample=%s | step=%d/%d | "
                "best_objective=%.6g | median_objective=%.6g | stale=%d",
                sample_label,
                completed_steps,
                config.optimization_steps,
                float(torch.min(best_loss).item()),
                float(torch.quantile(best_loss, 0.5).item()),
                stale_steps,
            )
        if stale_steps >= config.early_stopping_patience:
            break

    with torch.no_grad():
        best_highres, best_projected, best_seismic = _forward_candidates(
            anchored,
            best_parameters,
            basis=basis,
            segment_mask=segment_mask,
            forward_context=forward_context,
            device=device,
        )
    parameter_values = best_parameters.detach().cpu().numpy()
    highres_values = best_highres.detach().cpu().numpy()
    projected_values = best_projected.detach().cpu().numpy()
    seismic_values = best_seismic.detach().cpu().numpy()
    mask_values = score_mask.detach().cpu().numpy()
    support_values = projection_support[0].detach().cpu().numpy()
    start_rows: list[dict[str, Any]] = []
    for index in range(config.random_starts):
        metrics = _candidate_metrics(
            anchored,
            parameter_values[index],
            highres_values[index],
            projected_values[index],
            seismic_values[index],
            seismic_mask=mask_values,
            projection_support=support_values,
        )
        start_rows.append(
            {
                "start_index": index,
                "initialization": "zero" if index == 0 else "random",
                "initial_objective": float(initial_losses[index].item()),
                "best_step": int(best_step[index].item()),
                "objective": float(best_loss[index].item()),
                **metrics,
            }
        )
    best_index = int(torch.argmin(best_loss).item())
    best_objective = float(best_loss[best_index].item())
    near_threshold = (
        best_objective * (1.0 + config.near_optimal_relative_tolerance)
        + config.near_optimal_absolute_tolerance
    )
    near_indices = [
        index
        for index, row in enumerate(start_rows)
        if float(row["objective"]) <= near_threshold
    ]
    near_parameters = parameter_values[near_indices]
    identifiable_np = identifiable.detach().cpu().numpy()
    dispersion: dict[str, float] = {}
    truth_np = truth_parameters.detach().cpu().numpy()
    for coefficient, name in enumerate(("c0", "c1", "c2")):
        values = near_parameters[:, identifiable_np, coefficient]
        mean_std = float(np.mean(np.std(values, axis=0))) if values.size else 0.0
        scale = max(float(np.std(truth_np[identifiable_np, coefficient])), 0.01)
        dispersion[f"{name}_near_optimal_std"] = mean_std
        dispersion[f"{name}_near_optimal_normalized_std"] = mean_std / scale

    segment_rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(anchored.segments):
        row: dict[str, Any] = {
            "realization_id": anchored.source.realization_id,
            "lateral_index": anchored.source.lateral_index,
            "zone_id": anchored.source.zone.zone_id,
            "object_id": segment.source.object_id,
            "state": segment.source.state,
            "state_id": segment.source.state_id,
            "parameter_supervision_valid": segment.parameter_supervision_valid,
            "parameter_identifiability_rank": segment.parameter_identifiability_rank,
            "parameter_basis_condition": segment.parameter_basis_condition,
        }
        for coefficient, name in enumerate(("c0", "c1", "c2")):
            row[f"{name}_truth"] = float(truth_np[segment_index, coefficient])
            row[f"{name}_best"] = float(
                parameter_values[best_index, segment_index, coefficient]
            )
            row[f"{name}_near_optimal_std"] = float(
                np.std(near_parameters[:, segment_index, coefficient])
            )
        segment_rows.append(row)

    summary = {
        "realization_id": anchored.source.realization_id,
        "lateral_index": anchored.source.lateral_index,
        "lateral_m": anchored.source.lateral_m,
        "inline": anchored.source.inline,
        "xline": anchored.source.xline,
        "xline_step": anchored.source.xline_step,
        "zone_id": anchored.source.zone.zone_id,
        "segment_count": len(anchored.segments),
        "optimized_segment_count": int(np.count_nonzero(identifiable_np)),
        "seismic_score_sample_count": int(np.count_nonzero(mask_values)),
        "optimization_steps_completed": completed_steps,
        "truth_init_objective": float(truth_loss.item()),
        "truth_init_seismic_normalized_rmse": float(np.sqrt(truth_loss.item())),
        "best_start_index": best_index,
        "near_optimal_start_count": len(near_indices),
        "near_optimal_start_indices": near_indices,
        **start_rows[best_index],
        **dispersion,
    }
    return _OptimizationResult(
        summary=summary,
        starts=tuple(start_rows),
        segment_rows=tuple(segment_rows),
    )


def _aggregate(
    samples: Sequence[Mapping[str, Any]],
    config: TruthBoundaryOracleConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not samples:
        raise ValueError("truth-boundary Oracle produced no samples.")

    def median(name: str) -> float:
        values = [
            float(row[name])
            for row in samples
            if row.get(name) is not None and np.isfinite(float(row[name]))
        ]
        if not values:
            raise ValueError(f"truth-boundary Oracle lacks aggregate metric {name!r}.")
        return float(np.median(values))

    metrics = {
        "sample_count": len(samples),
        "truth_init_seismic_normalized_rmse_max": float(
            max(float(row["truth_init_seismic_normalized_rmse"]) for row in samples)
        ),
        "best_seismic_normalized_rmse_median": median("seismic_normalized_rmse"),
        "best_profile_rmse_median": median("profile_rmse"),
        "best_projected_log_ai_rmse_median": median("projected_log_ai_rmse"),
        "near_optimal_start_count_median": median("near_optimal_start_count"),
    }
    for name in ("c0", "c1", "c2"):
        metrics[f"{name}_normalized_rmse_median"] = median(
            f"{name}_normalized_rmse"
        )
        metrics[f"{name}_near_optimal_normalized_std_median"] = median(
            f"{name}_near_optimal_normalized_std"
        )
    truth_closed = (
        metrics["truth_init_seismic_normalized_rmse_max"]
        <= config.truth_init_nrmse_tolerance
    )
    forward_fitted = (
        metrics["best_seismic_normalized_rmse_median"]
        <= config.fitted_seismic_nrmse_tolerance
    )
    profile_recovered = (
        metrics["best_profile_rmse_median"] <= config.profile_rmse_tolerance
    )
    coefficients_recovered = all(
        metrics[f"{name}_normalized_rmse_median"]
        <= config.coefficient_normalized_rmse_tolerance
        for name in ("c0", "c1", "c2")
    )
    dispersion_assessable = metrics["near_optimal_start_count_median"] >= 2.0
    coefficients_stable = dispersion_assessable and all(
        metrics[f"{name}_near_optimal_normalized_std_median"]
        <= config.coefficient_dispersion_tolerance
        for name in ("c0", "c1", "c2")
    )
    if not truth_closed:
        recommendation = "implementation_contract_failed"
    elif forward_fitted and profile_recovered and coefficients_recovered and coefficients_stable:
        recommendation = "retain_three_parameter_target"
    elif forward_fitted:
        recommendation = "profile_primary_coefficients_audit_only"
    else:
        recommendation = "optimizer_unresolved"
    decision = {
        "schema": DECISION_SCHEMA,
        "recommendation": recommendation,
        "automatic_evidence": {
            "truth_init_contract_closed": truth_closed,
            "random_starts_fit_seismic": forward_fitted,
            "decoded_profile_recovered": profile_recovered,
            "coefficients_recovered": coefficients_recovered,
            "near_optimal_dispersion_assessable": dispersion_assessable,
            "near_optimal_coefficients_stable": coefficients_stable,
        },
        "thresholds": {
            "truth_init_nrmse_tolerance": config.truth_init_nrmse_tolerance,
            "fitted_seismic_nrmse_tolerance": config.fitted_seismic_nrmse_tolerance,
            "profile_rmse_tolerance": config.profile_rmse_tolerance,
            "coefficient_normalized_rmse_tolerance": (
                config.coefficient_normalized_rmse_tolerance
            ),
            "coefficient_dispersion_tolerance": config.coefficient_dispersion_tolerance,
        },
        "metrics": metrics,
        "requires_scientific_review": True,
    }
    return metrics, decision


def _write_segment_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot publish an empty Oracle segment table.")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError("Oracle segment rows do not share one schema.")
    temporary = path.with_name(f".{path.name}.staging")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_truth_boundary_oracle(
    config: TruthBoundaryOracleConfig,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run and publish the complete truth-boundary observability diagnostic."""
    if not isinstance(config, TruthBoundaryOracleConfig):
        raise TypeError("config must be TruthBoundaryOracleConfig.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    logger = configure_run_logger(
        destination,
        logger_name="ginn_v2.truth_boundary_oracle",
        file_name="oracle.log",
    )
    device, runtime = resolve_device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    benchmark = StructuredSyntheticBenchmark(config.benchmark_dir)
    split_manifest = ParentSplitManifest.from_dict(
        json.loads(config.split_manifest.read_text(encoding="utf-8"))
    )
    indexed = {item.realization_id for item in benchmark.list_parents()}
    split_ids = list(split_manifest.parent_ids(config.split))
    missing = sorted(set(split_ids).difference(indexed))
    if missing:
        raise ValueError(f"Oracle split contains parents absent from benchmark: {missing[:5]}")
    parent_ids = _selected_parent_ids(
        benchmark,
        split_ids,
        count=config.maximum_parents,
        seed=config.seed,
    )
    ai_bounds = load_zone_ai_bounds(config.impedance_calibration)
    logger.info(
        "truth-boundary Oracle start | device=%s | split=%s | parents=%d | "
        "samples_per_parent=%d | starts=%d | steps=%d",
        device,
        config.split,
        len(parent_ids),
        config.samples_per_parent,
        config.random_starts,
        config.optimization_steps,
    )
    sample_reports: list[dict[str, Any]] = []
    start_reports: list[dict[str, Any]] = []
    segment_rows: list[Mapping[str, Any]] = []
    expected_samples = len(parent_ids) * config.samples_per_parent
    completed = 0
    for parent_index, parent_id in enumerate(parent_ids):
        parent = benchmark.read_parent(parent_id)
        keys = _selected_sample_keys(
            parent,
            count=config.samples_per_parent,
            seed=config.seed + 1009 * parent_index,
        )
        expected_samples -= config.samples_per_parent - len(keys)
        for lateral_index, zone_id in keys:
            sample = StructuredTruthAdapter.from_structured_parent(
                parent,
                zone_id=zone_id,
                lateral_index=lateral_index,
            )
            anchored = anchor_to_lfm(
                sample,
                ai_bounds=ai_bounds,
                condition_limit=config.condition_limit,
            )
            result = _optimize_sample(
                anchored,
                config,
                device=device,
                seed=config.seed + 1_000_003 * parent_index + 101 * lateral_index,
                logger=logger,
                sample_label=f"{parent_id}|{lateral_index}|{zone_id}",
            )
            sample_reports.append(dict(result.summary))
            for row in result.starts:
                start_reports.append(
                    {
                        "realization_id": parent_id,
                        "lateral_index": lateral_index,
                        "zone_id": zone_id,
                        **dict(row),
                    }
                )
            segment_rows.extend(result.segment_rows)
            completed += 1
            if (
                completed % config.progress_log_every_samples == 0
                or completed == expected_samples
            ):
                logger.info(
                    "truth-boundary Oracle %d/%d | parent=%s | lateral=%d | "
                    "zone=%s | seismic_nrmse=%.4f | profile_rmse=%.4f | "
                    "near_optimal=%d",
                    completed,
                    expected_samples,
                    parent_id,
                    lateral_index,
                    zone_id,
                    float(result.summary["seismic_normalized_rmse"]),
                    float(result.summary["profile_rmse"]),
                    int(result.summary["near_optimal_start_count"]),
                )
    aggregate, decision = _aggregate(sample_reports, config)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "runtime": runtime,
        "config": asdict(config),
        "split_manifest_fingerprint": split_manifest.to_dict()["fingerprint_sha256"],
        "parent_ids": parent_ids,
        "aggregate": aggregate,
        "samples": sample_reports,
        "starts": start_reports,
    }
    _atomic_json(destination / "oracle_report.json", report)
    _atomic_json(destination / "decision_manifest.json", decision)
    _write_segment_csv(destination / "segment_recovery.csv", segment_rows)
    logger.info(
        "truth-boundary Oracle complete | recommendation=%s | samples=%d | "
        "seismic_nrmse_median=%.4f | profile_rmse_median=%.4f",
        decision["recommendation"],
        len(sample_reports),
        aggregate["best_seismic_normalized_rmse_median"],
        aggregate["best_profile_rmse_median"],
    )
    return report


__all__ = [
    "DECISION_SCHEMA",
    "REPORT_SCHEMA",
    "TruthBoundaryOracleConfig",
    "run_truth_boundary_oracle",
]
