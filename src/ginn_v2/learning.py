"""Learning and target-preflight seams for Structured GINN V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from cup.physics.numpy_backend import reflectivity_from_log_ai
from ginn_v2.artifacts import (
    Corpus,
    StructuredTrainingTile,
    load_checkpoint,
    load_corpus,
    parent_training_tiles,
    save_checkpoint,
)
from ginn_v2.contracts import ObservationTile
from ginn_v2.evidence import (
    BandlimitedEvidenceNetwork,
    BandlimitedTargets,
    EvidenceNetworkConfig,
    ObservationPerturbationProfile,
    build_bandlimited_targets,
    build_paired_observation_views,
    evidence_loss,
)
from ginn_v2.generator import ConditionalGenerator
from ginn_v2.representation import build_lfm_anchor, lfm_residual_from_anchor
from ginn_v2.runtime import configure_training_logger, resolve_device


TARGET_PREFLIGHT_SCHEMA = "structured_ginn_v2_target_preflight_v1"
TRAINING_REPORT_SCHEMA = "structured_ginn_v2_evidence_training_v1"
EVALUATION_REPORT_SCHEMA = "structured_ginn_v2_evidence_evaluation_v1"


@dataclass(frozen=True)
class LearningConfig:
    epochs: int = 4
    parent_batch_size: int = 4
    training_parent_count: int = 420
    tuning_parent_count: int = 120
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    random_seed: int = 20260806
    dirty_random_identity: int = 2026080601
    log_every_batches: int = 20
    early_stopping_patience: int = 2
    include_peak_poor: bool = True
    increment_loss_weight: float = 1.0
    reflectivity_loss_weight: float = 1.0
    state_loss_weight: float = 1.0
    scale_loss_weight: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "parent_batch_size",
            "training_parent_count",
            "tuning_parent_count",
            "log_every_batches",
            "early_stopping_patience",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "learning_rate",
            "increment_loss_weight",
            "reflectivity_loss_weight",
            "state_loss_weight",
            "scale_loss_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not isinstance(self.include_peak_poor, bool):
            raise TypeError("include_peak_poor must be boolean.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LearningConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "tuning"
    parent_count: int = 120
    parent_batch_size: int = 4
    dirty_random_identity: int = 2026080701
    log_every_batches: int = 20

    def __post_init__(self) -> None:
        if self.split not in {"tuning", "calibration", "section_gate"}:
            raise ValueError("evaluation split must be tuning, calibration, or section_gate.")
        for name in ("parent_count", "parent_batch_size", "log_every_batches"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EvaluationConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class TargetPreflightConfig:
    """Small, fixed budget used before any formal evidence training."""

    parent_count: int = 12
    local_filter_training_parent_count: int = 8
    overfit_parent_count: int = 2
    overfit_steps: int = 120
    learning_rate: float = 3.0e-3
    local_filter_radius_samples: int = 2
    local_filter_ridge: float = 1.0e-3
    random_seed: int = 20260806
    log_every_steps: int = 20
    minimum_overfit_relative_improvement: float = 0.10
    increment_loss_weight: float = 1.0
    reflectivity_loss_weight: float = 1.0
    state_loss_weight: float = 1.0
    scale_loss_weight: float = 0.10

    def __post_init__(self) -> None:
        integer_fields = (
            "parent_count",
            "local_filter_training_parent_count",
            "overfit_parent_count",
            "overfit_steps",
            "local_filter_radius_samples",
            "log_every_steps",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.parent_count < 3:
            raise ValueError("target preflight requires at least three parents.")
        if self.local_filter_training_parent_count >= self.parent_count:
            raise ValueError(
                "local_filter_training_parent_count must leave held-out parents."
            )
        if self.overfit_parent_count > self.parent_count:
            raise ValueError("overfit_parent_count cannot exceed parent_count.")
        positive_fields = (
            "learning_rate",
            "local_filter_ridge",
            "increment_loss_weight",
            "reflectivity_loss_weight",
            "state_loss_weight",
            "scale_loss_weight",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        threshold = float(self.minimum_overfit_relative_improvement)
        if not np.isfinite(threshold) or not 0.0 <= threshold < 1.0:
            raise ValueError(
                "minimum_overfit_relative_improvement must lie in [0, 1)."
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TargetPreflightConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class _PreparedTarget:
    parent_id: str
    geometry_family: str
    zone_id: str
    tile: StructuredTrainingTile
    targets: BandlimitedTargets
    anchor_model: np.ndarray
    lfm_residual: np.ndarray
    input_support: np.ndarray


def validate_observation_contract(
    generator: ConditionalGenerator,
    tiles: Iterable[ObservationTile],
    *,
    vp_model_mps_by_identity: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the cheap evidence seam check over supplied tiles."""

    count = 0
    domains: set[str] = set()
    widths: list[int] = []
    for tile in tiles:
        vp_model_mps = None
        if tile.sample_domain == "depth":
            if vp_model_mps_by_identity is None or tile.identity not in vp_model_mps_by_identity:
                raise ValueError(
                    f"depth tile {tile.identity!r} requires explicit model-grid Vp."
                )
            vp_model_mps = vp_model_mps_by_identity[tile.identity]
        evidence = generator.observe(tile, vp_model_mps=vp_model_mps)
        count += 1
        domains.add(evidence.model_axis.sample_domain)
        widths.append(tile.width)
    if count == 0:
        raise ValueError("contract validation received no tiles.")
    return {
        "tile_count": count,
        "domains": sorted(domains),
        "width_min": min(widths),
        "width_max": max(widths),
    }


def _select_split_parents(
    corpus: Corpus,
    split: str,
    count: int,
) -> tuple[str, ...]:
    if split not in corpus.splits:
        raise ValueError(f"unknown corpus split: {split}")
    if count > len(corpus.splits[split]):
        raise ValueError(
            f"{split} split contains {len(corpus.splits[split])} parents, not {count}."
        )
    index = corpus.benchmark.index
    selected_split = index[index["split_role"].eq(split)]
    family_values: dict[str, list[str]] = {}
    for family in sorted(selected_split["geometry_family"].astype(str).unique()):
        family_values[family] = sorted(
            selected_split.loc[
                selected_split["geometry_family"].astype(str).eq(family),
                "realization_id",
            ]
            .astype(str)
            .tolist()
        )
    selected: list[str] = []
    offset = 0
    while len(selected) < count:
        progressed = False
        for family in sorted(family_values):
            values = family_values[family]
            if offset < len(values):
                selected.append(values[offset])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"{split} split contains fewer than {count} parents.")
        offset += 1
    return tuple(selected)


def _select_training_parents(corpus: Corpus, count: int) -> tuple[str, ...]:
    return _select_split_parents(corpus, "training", count)


def _prepare_target(
    parent_id: str,
    geometry_family: str,
    tile: StructuredTrainingTile,
) -> _PreparedTarget:
    observation = tile.observation
    anchor = build_lfm_anchor(observation)
    categorical = np.asarray(tile.categorical_valid_model, dtype=bool)
    zone_support = np.asarray(tile.model_zone_support, dtype=bool)
    anchor_support = anchor.model_support & zone_support & categorical
    targets = build_bandlimited_targets(
        observation,
        model_log_ai=tile.model_log_ai,
        state_fraction_model=tile.state_fraction_model,
        background_lfm_linear=anchor.model,
        anchor_support=anchor_support,
    )
    input_support = (
        anchor.model_support
        & zone_support
        & observation.observed_valid
        & observation.lateral_valid[:, None]
    )
    residual = np.where(
        input_support,
        lfm_residual_from_anchor(observation, anchor),
        0.0,
    )
    zone_id = observation.identity.rsplit(":", 1)[-1]
    return _PreparedTarget(
        parent_id=parent_id,
        geometry_family=geometry_family,
        zone_id=zone_id,
        tile=tile,
        targets=targets,
        anchor_model=anchor.model,
        lfm_residual=residual,
        input_support=input_support,
    )


def _load_preflight_targets(
    corpus: Corpus,
    parent_ids: Sequence[str],
    logger: logging.Logger,
) -> list[_PreparedTarget]:
    index = corpus.benchmark.index.set_index("realization_id", drop=False)
    prepared: list[_PreparedTarget] = []
    for position, parent_id in enumerate(parent_ids, start=1):
        family = str(index.loc[parent_id, "geometry_family"])
        parent_tiles = parent_training_tiles(corpus, parent_id)
        for tile in parent_tiles:
            prepared.append(_prepare_target(parent_id, family, tile))
        logger.info(
            "target preflight load %d/%d | parent=%s | zones=%d",
            position,
            len(parent_ids),
            parent_id,
            len(parent_tiles),
        )
    return prepared


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0 or np.any(~np.isfinite(data)):
        raise ValueError("target distribution requires finite values.")
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "standard_deviation": float(np.std(data)),
        "rms": float(np.sqrt(np.mean(data**2))),
        "minimum": float(np.min(data)),
        "p01": float(np.quantile(data, 0.01)),
        "p50": float(np.quantile(data, 0.50)),
        "p99": float(np.quantile(data, 0.99)),
        "maximum": float(np.max(data)),
    }


def _target_statistics(examples: Sequence[_PreparedTarget]) -> dict[str, Any]:
    increment = np.concatenate(
        [item.targets.projected_log_ai_increment[item.targets.support] for item in examples]
    )
    reflectivity = np.concatenate(
        [item.targets.signed_reflectivity[item.targets.support] for item in examples]
    )
    state = np.concatenate(
        [item.targets.state_fraction[item.targets.support] for item in examples], axis=0
    )
    collapse_count = int(
        sum(
            np.count_nonzero(
                np.asarray(item.tile.projection_collapse_mask_model, dtype=bool)
                & item.targets.support
            )
            for item in examples
        )
    )
    support_count = int(increment.size)
    state_entropy = -np.sum(state * np.log(np.clip(state, 1.0e-12, 1.0)), axis=-1)
    return {
        "supported_samples": support_count,
        "projected_log_ai_increment": _distribution(increment),
        "signed_reflectivity": _distribution(reflectivity),
        "state_fraction_mean": [float(value) for value in np.mean(state, axis=0)],
        "state_fraction_mixed_sample_fraction": float(
            np.mean(np.max(state, axis=-1) < 1.0 - 1.0e-6)
        ),
        "state_fraction_entropy": _distribution(state_entropy),
        "projection_collapse_supported_samples": collapse_count,
        "projection_collapse_supported_fraction": float(collapse_count / support_count),
        "by_geometry_family": {
            family: int(
                sum(
                    np.count_nonzero(item.targets.support)
                    for item in examples
                    if item.geometry_family == family
                )
            )
            for family in sorted({item.geometry_family for item in examples})
        },
    }


def _paired_target_identity(
    examples: Sequence[_PreparedTarget],
    profile: ObservationPerturbationProfile,
    random_identity: int,
) -> dict[str, Any]:
    checked = 0
    maximum_difference = 0.0
    for item in examples:
        views = build_paired_observation_views(
            item.tile.observation,
            random_identity=random_identity,
            profile=profile,
        )
        for view in (views.dirty, views.peak_poor):
            anchor = build_lfm_anchor(view)
            rebuilt = build_bandlimited_targets(
                view,
                model_log_ai=item.tile.model_log_ai,
                state_fraction_model=item.tile.state_fraction_model,
                background_lfm_linear=anchor.model,
                anchor_support=(
                    anchor.model_support
                    & item.tile.model_zone_support
                    & item.tile.categorical_valid_model.astype(bool)
                ),
            )
            if not np.array_equal(rebuilt.support, item.targets.support):
                raise ValueError("paired observation changed target support.")
            for name in (
                "projected_log_ai_increment",
                "signed_reflectivity",
                "state_fraction",
            ):
                difference = float(
                    np.max(np.abs(getattr(rebuilt, name) - getattr(item.targets, name)))
                )
                maximum_difference = max(maximum_difference, difference)
            checked += 1
    return {
        "view_comparisons": checked,
        "maximum_absolute_target_difference": maximum_difference,
        "identical": bool(maximum_difference == 0.0),
    }


def _reflectivity(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    finite = np.where(np.isfinite(data), data, 0.0)
    result = np.zeros_like(finite)
    result[:, 1:] = reflectivity_from_log_ai(finite)
    return result


def _regression_metrics(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    supports: Sequence[np.ndarray],
) -> dict[str, float]:
    prediction = np.concatenate(
        [np.asarray(value)[support] for value, support in zip(predictions, supports, strict=True)]
    ).astype(np.float64)
    truth = np.concatenate(
        [np.asarray(value)[support] for value, support in zip(targets, supports, strict=True)]
    ).astype(np.float64)
    residual = prediction - truth
    correlation = 0.0
    if prediction.size > 1 and np.std(prediction) > 0.0 and np.std(truth) > 0.0:
        correlation = float(np.corrcoef(prediction, truth)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "correlation": correlation,
    }


def _state_metrics(
    predictions: Sequence[np.ndarray],
    examples: Sequence[_PreparedTarget],
) -> dict[str, float]:
    predicted = np.concatenate(
        [value[item.targets.support] for value, item in zip(predictions, examples, strict=True)],
        axis=0,
    )
    truth = np.concatenate(
        [item.targets.state_fraction[item.targets.support] for item in examples], axis=0
    )
    probabilities = np.clip(predicted, 1.0e-8, None)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return {
        "brier": float(np.mean(np.sum((probabilities - truth) ** 2, axis=-1))),
        "cross_entropy": float(
            np.mean(-np.sum(truth * np.log(probabilities), axis=-1))
        ),
    }


def _baseline_report(
    name: str,
    examples: Sequence[_PreparedTarget],
    increment: Sequence[np.ndarray],
    reflectivity: Sequence[np.ndarray],
    state: Sequence[np.ndarray],
) -> dict[str, Any]:
    supports = [item.targets.support for item in examples]
    return {
        "name": name,
        "projected_log_ai_increment": _regression_metrics(
            increment,
            [item.targets.projected_log_ai_increment for item in examples],
            supports,
        ),
        "signed_reflectivity": _regression_metrics(
            reflectivity,
            [item.targets.signed_reflectivity for item in examples],
            supports,
        ),
        "state_fraction": _state_metrics(state, examples),
    }


def _shifted(values: np.ndarray, lag: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    if lag == 0:
        result[:] = values
    elif lag > 0:
        result[:, lag:] = values[:, :-lag]
    else:
        result[:, :lag] = values[:, -lag:]
    return result


def _local_features(
    example: _PreparedTarget,
    radius: int,
    *,
    seismic: np.ndarray | None = None,
) -> np.ndarray:
    source = example.tile.observation.seismic if seismic is None else seismic
    columns = [_shifted(source, lag) for lag in range(-radius, radius + 1)]
    columns.extend((example.lfm_residual, np.ones_like(source)))
    return np.stack(columns, axis=-1)[example.targets.support]


def _fit_local_filter(
    examples: Sequence[_PreparedTarget],
    *,
    radius: int,
    ridge: float,
) -> np.ndarray:
    dimension = 2 * radius + 3
    xtx = np.zeros((dimension, dimension), dtype=np.float64)
    xty = np.zeros((dimension, 5), dtype=np.float64)
    for item in examples:
        features = _local_features(item, radius)
        target = np.column_stack(
            (
                item.targets.projected_log_ai_increment[item.targets.support],
                item.targets.signed_reflectivity[item.targets.support],
                item.targets.state_fraction[item.targets.support],
            )
        )
        xtx += features.T @ features
        xty += features.T @ target
    regularization = np.eye(dimension, dtype=np.float64)
    regularization[-1, -1] = 0.0
    scale = max(float(np.trace(xtx) / dimension), 1.0e-12)
    return np.linalg.solve(xtx + ridge * scale * regularization, xty)


def _matched_donor_seismic(
    recipient: _PreparedTarget,
    donor: _PreparedTarget,
) -> np.ndarray:
    target = recipient.tile.observation.seismic
    source = donor.tile.observation.seismic
    if source.shape != target.shape:
        raise ValueError("matched seismic donors must share observation shape.")
    result = np.zeros_like(target, dtype=np.float64)
    for trace in range(target.shape[0]):
        valid = recipient.input_support[trace]
        if not np.any(valid):
            continue
        donor_values = source[trace, valid]
        recipient_values = target[trace, valid]
        donor_rms = float(np.sqrt(np.mean(donor_values**2)))
        recipient_rms = float(np.sqrt(np.mean(recipient_values**2)))
        if donor_rms <= 0.0 or not np.isfinite(donor_rms):
            raise ValueError("matched seismic donor has zero or invalid RMS.")
        result[trace, valid] = donor_values * (recipient_rms / donor_rms)
    return result


def _different_parent_donors(
    examples: Sequence[_PreparedTarget],
) -> list[_PreparedTarget]:
    """Match every zone to the same zone from a different parent."""

    by_zone: dict[str, list[_PreparedTarget]] = {}
    for item in examples:
        by_zone.setdefault(item.zone_id, []).append(item)
    donor_by_identity: dict[tuple[str, str], _PreparedTarget] = {}
    for zone_id, values in by_zone.items():
        if len({item.parent_id for item in values}) < 2:
            raise ValueError(
                f"matched seismic shuffle needs two parents for zone {zone_id!r}."
            )
        for index, item in enumerate(values):
            donor = values[(index + 1) % len(values)]
            if donor.parent_id == item.parent_id:
                raise ValueError("matched seismic donor must come from another parent.")
            donor_by_identity[(item.parent_id, item.zone_id)] = donor
    return [donor_by_identity[(item.parent_id, item.zone_id)] for item in examples]


def _local_filter_predictions(
    examples: Sequence[_PreparedTarget],
    coefficients: np.ndarray,
    radius: int,
    *,
    donor_examples: Sequence[_PreparedTarget] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    increment: list[np.ndarray] = []
    reflectivity: list[np.ndarray] = []
    state: list[np.ndarray] = []
    for index, item in enumerate(examples):
        donor_seismic = None
        if donor_examples is not None:
            donor_seismic = _matched_donor_seismic(item, donor_examples[index])
        values = _local_features(item, radius, seismic=donor_seismic) @ coefficients
        shape = item.targets.support.shape
        increment_value = np.zeros(shape, dtype=np.float64)
        reflectivity_value = np.zeros(shape, dtype=np.float64)
        state_value = np.full(shape + (3,), 1.0 / 3.0, dtype=np.float64)
        increment_value[item.targets.support] = values[:, 0]
        reflectivity_value[item.targets.support] = values[:, 1]
        probabilities = np.clip(values[:, 2:], 1.0e-8, None)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        state_value[item.targets.support] = probabilities
        increment.append(increment_value)
        reflectivity.append(reflectivity_value)
        state.append(state_value)
    return increment, reflectivity, state


def _simple_baselines(
    examples: Sequence[_PreparedTarget],
    training_parent_count: int,
    radius: int,
    ridge: float,
) -> dict[str, Any]:
    parent_order = list(dict.fromkeys(item.parent_id for item in examples))
    training_ids = set(parent_order[:training_parent_count])
    training = [item for item in examples if item.parent_id in training_ids]
    validation = [item for item in examples if item.parent_id not in training_ids]
    if not training or len(validation) < 2:
        raise ValueError("local-filter baseline needs train and at least two validation parents.")

    increment_mean = float(
        np.mean(
            np.concatenate(
                [item.targets.projected_log_ai_increment[item.targets.support] for item in training]
            )
        )
    )
    reflectivity_mean = float(
        np.mean(
            np.concatenate(
                [item.targets.signed_reflectivity[item.targets.support] for item in training]
            )
        )
    )
    state_mean = np.mean(
        np.concatenate(
            [item.targets.state_fraction[item.targets.support] for item in training], axis=0
        ),
        axis=0,
    )
    constant = _baseline_report(
        "training_mean",
        validation,
        [np.full_like(item.targets.projected_log_ai_increment, increment_mean) for item in validation],
        [np.full_like(item.targets.signed_reflectivity, reflectivity_mean) for item in validation],
        [np.broadcast_to(state_mean, item.targets.state_fraction.shape).copy() for item in validation],
    )
    anchor = _baseline_report(
        "zone_linear_anchor",
        validation,
        [np.zeros_like(item.targets.projected_log_ai_increment) for item in validation],
        [_reflectivity(item.anchor_model) for item in validation],
        [np.broadcast_to(state_mean, item.targets.state_fraction.shape).copy() for item in validation],
    )
    full_lfm = _baseline_report(
        "full_lfm",
        validation,
        [item.lfm_residual for item in validation],
        [_reflectivity(item.tile.observation.lfm) for item in validation],
        [np.broadcast_to(state_mean, item.targets.state_fraction.shape).copy() for item in validation],
    )
    coefficients = _fit_local_filter(training, radius=radius, ridge=ridge)
    local_values = _local_filter_predictions(validation, coefficients, radius)
    local = _baseline_report("local_vertical_filter", validation, *local_values)
    donors = _different_parent_donors(validation)
    shuffled_values = _local_filter_predictions(
        validation,
        coefficients,
        radius,
        donor_examples=donors,
    )
    shuffled = _baseline_report(
        "local_vertical_filter_matched_seismic_shuffle",
        validation,
        *shuffled_values,
    )
    return {
        "training_parents": sorted(training_ids),
        "validation_parents": list(dict.fromkeys(item.parent_id for item in validation)),
        "training_mean": constant,
        "zone_linear_anchor": anchor,
        "full_lfm": full_lfm,
        "local_vertical_filter": local,
        "local_vertical_filter_matched_seismic_shuffle": shuffled,
        "local_filter_coefficients": coefficients.tolist(),
    }


def _rms(values: Sequence[np.ndarray], supports: Sequence[np.ndarray]) -> float:
    represented = np.concatenate(
        [value[support] for value, support in zip(values, supports, strict=True)]
    ).astype(np.float64)
    return max(float(np.sqrt(np.mean(represented**2))), 1.0e-6)


def _normalized_network_config(
    base: EvidenceNetworkConfig,
    examples: Sequence[_PreparedTarget],
    *,
    input_mode: str,
) -> EvidenceNetworkConfig:
    target_support = [item.targets.support for item in examples]
    input_support = [item.input_support for item in examples]
    return replace(
        base,
        input_mode=input_mode,
        seismic_scale=_rms(
            [item.tile.observation.seismic for item in examples], input_support
        ),
        lfm_residual_scale=_rms(
            [item.lfm_residual for item in examples], input_support
        ),
        projected_log_ai_increment_scale=_rms(
            [item.targets.projected_log_ai_increment for item in examples],
            target_support,
        ),
        signed_reflectivity_scale=_rms(
            [item.targets.signed_reflectivity for item in examples], target_support
        ),
    )


def _torch_example(
    item: _PreparedTarget,
    device: torch.device,
    *,
    seismic: np.ndarray | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    observation = item.tile.observation
    source = observation.seismic if seismic is None else seismic
    inputs = {
        "seismic": torch.as_tensor(source[None], dtype=torch.float32, device=device),
        "lfm_residual": torch.as_tensor(
            item.lfm_residual[None], dtype=torch.float32, device=device
        ),
        "observed_valid": torch.as_tensor(
            item.input_support[None], dtype=torch.bool, device=device
        ),
        "lateral_m": torch.as_tensor(
            observation.lateral_m[None], dtype=torch.float32, device=device
        ),
        "lateral_valid": torch.as_tensor(
            observation.lateral_valid[None], dtype=torch.bool, device=device
        ),
    }
    targets = {
        "projected_log_ai_increment": torch.as_tensor(
            item.targets.projected_log_ai_increment[None],
            dtype=torch.float32,
            device=device,
        ),
        "signed_reflectivity": torch.as_tensor(
            item.targets.signed_reflectivity[None], dtype=torch.float32, device=device
        ),
        "state_fraction": torch.as_tensor(
            item.targets.state_fraction[None], dtype=torch.float32, device=device
        ),
        "support": torch.as_tensor(
            item.targets.support[None], dtype=torch.bool, device=device
        ),
    }
    return inputs, targets


def _network_loss(
    network: BandlimitedEvidenceNetwork,
    item: _PreparedTarget,
    device: torch.device,
    config: TargetPreflightConfig,
    *,
    seismic: np.ndarray | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs, targets = _torch_example(item, device, seismic=seismic)
    output = network(**inputs)
    losses = evidence_loss(
        output,
        targets,
        config=network.config,
        increment_weight=config.increment_loss_weight,
        reflectivity_weight=config.reflectivity_loss_weight,
        state_weight=config.state_loss_weight,
        scale_weight=config.scale_loss_weight,
    )
    return losses, output


def _evaluate_network(
    network: BandlimitedEvidenceNetwork,
    examples: Sequence[_PreparedTarget],
    device: torch.device,
    config: TargetPreflightConfig,
    *,
    donors: Sequence[_PreparedTarget] | None = None,
) -> dict[str, float]:
    loss_values: dict[str, list[float]] = {}
    increment_prediction: list[np.ndarray] = []
    reflectivity_prediction: list[np.ndarray] = []
    state_prediction: list[np.ndarray] = []
    network.eval()
    with torch.no_grad():
        for index, item in enumerate(examples):
            seismic = None
            if donors is not None:
                seismic = _matched_donor_seismic(item, donors[index])
            losses, output = _network_loss(
                network, item, device, config, seismic=seismic
            )
            for name, value in losses.items():
                loss_values.setdefault(name, []).append(float(value.detach().cpu()))
            increment_prediction.append(
                output["projected_log_ai_increment_mean"][0].detach().cpu().numpy()
            )
            reflectivity_prediction.append(
                output["signed_reflectivity_mean"][0].detach().cpu().numpy()
            )
            state_prediction.append(
                torch.softmax(output["state_logits"][0], dim=-1).detach().cpu().numpy()
            )
    support = [item.targets.support for item in examples]
    increment = _regression_metrics(
        increment_prediction,
        [item.targets.projected_log_ai_increment for item in examples],
        support,
    )
    reflectivity = _regression_metrics(
        reflectivity_prediction,
        [item.targets.signed_reflectivity for item in examples],
        support,
    )
    state = _state_metrics(state_prediction, examples)
    result = {name: float(np.mean(values)) for name, values in loss_values.items()}
    result.update(
        {
            "increment_rmse": increment["rmse"],
            "increment_correlation": increment["correlation"],
            "reflectivity_rmse": reflectivity["rmse"],
            "reflectivity_correlation": reflectivity["correlation"],
            "state_brier": state["brier"],
        }
    )
    return result


def _mini_overfit(
    examples: Sequence[_PreparedTarget],
    base_network_config: EvidenceNetworkConfig,
    preflight_config: TargetPreflightConfig,
    device: torch.device,
    logger: logging.Logger,
    *,
    input_mode: str,
) -> dict[str, Any]:
    network_config = _normalized_network_config(
        base_network_config, examples, input_mode=input_mode
    )
    torch.manual_seed(preflight_config.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(preflight_config.random_seed)
    network = BandlimitedEvidenceNetwork(network_config).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=preflight_config.learning_rate)
    initial = _evaluate_network(network, examples, device, preflight_config)
    network.train()
    for step in range(1, preflight_config.overfit_steps + 1):
        item = examples[(step - 1) % len(examples)]
        optimizer.zero_grad(set_to_none=True)
        losses, _ = _network_loss(network, item, device, preflight_config)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
        optimizer.step()
        if step == 1 or step % preflight_config.log_every_steps == 0:
            logger.info(
                "mini overfit | mode=%s | step=%d/%d | loss=%.6f",
                input_mode,
                step,
                preflight_config.overfit_steps,
                float(losses["loss"].detach().cpu()),
            )
    final = _evaluate_network(network, examples, device, preflight_config)
    initial_loss = float(initial["loss"])
    final_loss = float(final["loss"])
    relative_improvement = (initial_loss - final_loss) / max(abs(initial_loss), 1.0e-12)
    result: dict[str, Any] = {
        "input_mode": input_mode,
        "network_config": asdict(network_config),
        "initial": initial,
        "final": final,
        "relative_loss_improvement": float(relative_improvement),
    }
    if input_mode == "full" and len(examples) > 1:
        donors = _different_parent_donors(examples)
        result["matched_seismic_shuffle"] = _evaluate_network(
            network,
            examples,
            device,
            preflight_config,
            donors=donors,
        )
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".staging")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def preflight_evidence_targets(
    corpus_root: str | Path,
    output_dir: str | Path,
    *,
    evidence_config: EvidenceNetworkConfig,
    config: TargetPreflightConfig = TargetPreflightConfig(),
    perturbation_profile: ObservationPerturbationProfile = ObservationPerturbationProfile(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Audit evidence targets and cheap learnability before formal training."""

    started = time.perf_counter()
    output = Path(output_dir)
    report_path = output / "target_preflight_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(output, file_name="target_preflight.log")
    device, runtime = resolve_device(device_name)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    logger.info(
        "target preflight start | corpus=%s | device=%s | parents=%d | overfit_steps=%d",
        Path(corpus_root),
        device,
        config.parent_count,
        config.overfit_steps,
    )

    corpus = load_corpus(corpus_root)
    parent_ids = _select_training_parents(corpus, config.parent_count)
    examples = _load_preflight_targets(corpus, parent_ids, logger)
    statistics = _target_statistics(examples)
    paired_identity = _paired_target_identity(
        examples,
        perturbation_profile,
        config.random_seed,
    )
    baselines = _simple_baselines(
        examples,
        config.local_filter_training_parent_count,
        config.local_filter_radius_samples,
        config.local_filter_ridge,
    )
    overfit_parent_ids = set(parent_ids[: config.overfit_parent_count])
    overfit_examples = [item for item in examples if item.parent_id in overfit_parent_ids]
    full_overfit = _mini_overfit(
        overfit_examples,
        evidence_config,
        config,
        device,
        logger,
        input_mode="full",
    )
    no_seismic_overfit = _mini_overfit(
        overfit_examples,
        evidence_config,
        config,
        device,
        logger,
        input_mode="no_seismic",
    )

    local = baselines["local_vertical_filter"]
    shuffled = baselines["local_vertical_filter_matched_seismic_shuffle"]
    gates = {
        "supported_samples_present": statistics["supported_samples"] > 0,
        "projected_increment_non_degenerate": (
            statistics["projected_log_ai_increment"]["standard_deviation"] > 1.0e-8
        ),
        "signed_reflectivity_non_degenerate": (
            statistics["signed_reflectivity"]["standard_deviation"] > 1.0e-8
        ),
        "all_states_represented": min(statistics["state_fraction_mean"]) > 0.0,
        "paired_observations_preserve_targets": paired_identity["identical"],
        "fixed_mini_corpus_learns": (
            full_overfit["relative_loss_improvement"]
            >= config.minimum_overfit_relative_improvement
        ),
        "matched_seismic_shuffle_degrades_reflectivity": (
            shuffled["signed_reflectivity"]["rmse"]
            > local["signed_reflectivity"]["rmse"]
        ),
    }
    status = "passed" if all(gates.values()) else "failed"
    report = {
        "schema": TARGET_PREFLIGHT_SCHEMA,
        "status": status,
        "corpus": {
            "root": Path(corpus_root).as_posix(),
            "schema": corpus.manifest.get("schema_version"),
            "publication_status": corpus.manifest.get("status"),
            "sample_domain": corpus.benchmark.sample_domain,
            "sample_unit": corpus.benchmark.sample_unit,
            "depth_basis": corpus.benchmark.depth_basis,
        },
        "runtime": runtime,
        "config": asdict(config),
        "perturbation_profile": asdict(perturbation_profile),
        "selected_parent_ids": list(parent_ids),
        "selected_zone_count": len(examples),
        "overfit_parent_ids": sorted(overfit_parent_ids),
        "target_statistics": statistics,
        "paired_target_identity": paired_identity,
        "baselines": baselines,
        "mini_corpus_overfit": {
            "full": full_overfit,
            "no_seismic": no_seismic_overfit,
        },
        "gates": gates,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _atomic_json(report_path, report)
    logger.info(
        "target preflight finished | status=%s | elapsed=%.1fs | report=%s",
        status,
        report["elapsed_seconds"],
        report_path,
    )
    return report


def _read_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"report root must be an object: {source}")
    return payload


def _validate_preflight_report(
    report: Mapping[str, Any],
    corpus: Corpus,
) -> tuple[str, ...]:
    if report.get("schema") != TARGET_PREFLIGHT_SCHEMA:
        raise ValueError("target preflight report uses an incompatible schema.")
    if report.get("status") != "passed":
        raise ValueError("target preflight report has not passed.")
    corpus_value = report.get("corpus")
    if not isinstance(corpus_value, Mapping):
        raise ValueError("target preflight report lacks corpus semantics.")
    expected = {
        "schema": corpus.manifest.get("schema_version"),
        "sample_domain": corpus.benchmark.sample_domain,
        "sample_unit": corpus.benchmark.sample_unit,
        "depth_basis": corpus.benchmark.depth_basis,
    }
    for name, value in expected.items():
        if corpus_value.get(name) != value:
            raise ValueError(f"target preflight corpus {name} differs from the input corpus.")
    parent_value = report.get("selected_parent_ids")
    if not isinstance(parent_value, list) or not parent_value:
        raise ValueError("target preflight report lacks selected_parent_ids.")
    parents = tuple(str(value) for value in parent_value)
    if not set(parents).issubset(corpus.splits["training"]):
        raise ValueError("target preflight parents are outside the training split.")
    return parents


def _network_config_for_mode(
    base: EvidenceNetworkConfig,
    normalization_examples: Sequence[_PreparedTarget],
    mode: str,
) -> EvidenceNetworkConfig:
    if mode not in {"full", "no_seismic", "single_trace"}:
        raise ValueError("input mode must be full, no_seismic, or single_trace.")
    normalized = _normalized_network_config(
        base,
        normalization_examples,
        input_mode="no_seismic" if mode == "no_seismic" else "full",
    )
    return replace(
        normalized,
        use_lateral_context=(False if mode == "single_trace" else base.use_lateral_context),
    )


def _prepare_parent_batch(
    corpus: Corpus,
    parent_ids: Sequence[str],
) -> list[_PreparedTarget]:
    index = corpus.benchmark.index.set_index("realization_id", drop=False)
    prepared: list[_PreparedTarget] = []
    for parent_id in parent_ids:
        family = str(index.loc[parent_id, "geometry_family"])
        for tile in parent_training_tiles(corpus, parent_id):
            prepared.append(_prepare_target(parent_id, family, tile))
    return prepared


def _observation_seismic(
    item: _PreparedTarget,
    view: str,
    *,
    random_identity: int,
    perturbation_profile: ObservationPerturbationProfile,
    donor: _PreparedTarget | None,
    paired_views: Any | None = None,
) -> np.ndarray:
    if view == "clean":
        return item.tile.observation.seismic
    if view == "matched_seismic_shuffle":
        if donor is None:
            raise ValueError("matched seismic shuffle requires a donor.")
        return _matched_donor_seismic(item, donor)
    views = paired_views
    if views is None:
        views = build_paired_observation_views(
            item.tile.observation,
            random_identity=random_identity,
            profile=perturbation_profile,
        )
    if view == "dirty":
        return views.dirty.seismic
    if view == "peak_poor":
        return views.peak_poor.seismic
    raise ValueError(f"unknown observation view: {view}")


def _stack_evidence_batch(
    examples: Sequence[_PreparedTarget],
    views: Sequence[str],
    *,
    random_identity: int,
    perturbation_profile: ObservationPerturbationProfile,
    device: torch.device,
    donors: Sequence[_PreparedTarget] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not examples or not views:
        raise ValueError("evidence batch requires examples and observation views.")
    if donors is not None and len(donors) != len(examples):
        raise ValueError("matched donors must align with examples.")
    shapes = {item.tile.observation.seismic.shape for item in examples}
    if len(shapes) != 1:
        raise ValueError("one evidence batch requires a common observation shape.")

    seismic: list[np.ndarray] = []
    lfm_residual: list[np.ndarray] = []
    observed_valid: list[np.ndarray] = []
    lateral_m: list[np.ndarray] = []
    lateral_valid: list[np.ndarray] = []
    increment: list[np.ndarray] = []
    reflectivity: list[np.ndarray] = []
    state_fraction: list[np.ndarray] = []
    support: list[np.ndarray] = []
    paired_cache: dict[int, Any] = {}
    if any(view in {"dirty", "peak_poor"} for view in views):
        for index, item in enumerate(examples):
            paired_cache[index] = build_paired_observation_views(
                item.tile.observation,
                random_identity=random_identity,
                profile=perturbation_profile,
            )
    for view in views:
        for index, item in enumerate(examples):
            donor = None if donors is None else donors[index]
            seismic.append(
                _observation_seismic(
                    item,
                    view,
                    random_identity=random_identity,
                    perturbation_profile=perturbation_profile,
                    donor=donor,
                    paired_views=paired_cache.get(index),
                )
            )
            lfm_residual.append(item.lfm_residual)
            observed_valid.append(item.input_support)
            lateral_m.append(item.tile.observation.lateral_m)
            lateral_valid.append(item.tile.observation.lateral_valid)
            increment.append(item.targets.projected_log_ai_increment)
            reflectivity.append(item.targets.signed_reflectivity)
            state_fraction.append(item.targets.state_fraction)
            support.append(item.targets.support)
    inputs = {
        "seismic": torch.as_tensor(
            np.stack(seismic), dtype=torch.float32, device=device
        ),
        "lfm_residual": torch.as_tensor(
            np.stack(lfm_residual), dtype=torch.float32, device=device
        ),
        "observed_valid": torch.as_tensor(
            np.stack(observed_valid), dtype=torch.bool, device=device
        ),
        "lateral_m": torch.as_tensor(
            np.stack(lateral_m), dtype=torch.float32, device=device
        ),
        "lateral_valid": torch.as_tensor(
            np.stack(lateral_valid), dtype=torch.bool, device=device
        ),
    }
    targets = {
        "projected_log_ai_increment": torch.as_tensor(
            np.stack(increment), dtype=torch.float32, device=device
        ),
        "signed_reflectivity": torch.as_tensor(
            np.stack(reflectivity), dtype=torch.float32, device=device
        ),
        "state_fraction": torch.as_tensor(
            np.stack(state_fraction), dtype=torch.float32, device=device
        ),
        "support": torch.as_tensor(
            np.stack(support), dtype=torch.bool, device=device
        ),
    }
    return inputs, targets


def _configured_evidence_loss(
    network: BandlimitedEvidenceNetwork,
    output: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    config: LearningConfig,
) -> dict[str, torch.Tensor]:
    return evidence_loss(
        output,
        targets,
        config=network.config,
        increment_weight=config.increment_loss_weight,
        reflectivity_weight=config.reflectivity_loss_weight,
        state_weight=config.state_loss_weight,
        scale_weight=config.scale_loss_weight,
    )


class _EvidenceMetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.loss_weight = 0
        self.loss_sums: dict[str, float] = {}
        self.regression = {
            "projected_log_ai_increment": np.zeros(7, dtype=np.float64),
            "signed_reflectivity": np.zeros(7, dtype=np.float64),
        }
        self.state_brier_sum = 0.0
        self.state_cross_entropy_sum = 0.0
        self.increment_coverage_1_sum = 0
        self.increment_coverage_2_sum = 0
        self.reflectivity_coverage_1_sum = 0
        self.reflectivity_coverage_2_sum = 0

    @staticmethod
    def _update_regression(
        accumulator: np.ndarray,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        predicted = prediction.detach().double()
        truth = target.detach().double()
        residual = predicted - truth
        accumulator += np.asarray(
            [
                predicted.numel(),
                float(torch.sum(predicted).cpu()),
                float(torch.sum(truth).cpu()),
                float(torch.sum(predicted**2).cpu()),
                float(torch.sum(truth**2).cpu()),
                float(torch.sum(predicted * truth).cpu()),
                float(torch.sum(residual**2).cpu()),
            ],
            dtype=np.float64,
        )

    def update(
        self,
        output: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        losses: Mapping[str, torch.Tensor],
    ) -> None:
        support = targets["support"].bool() & output["support"].bool()
        count = int(torch.count_nonzero(support).item())
        if count <= 0:
            raise ValueError("metric batch has no supported samples.")
        self.count += count
        self.loss_weight += count
        for name, value in losses.items():
            self.loss_sums[name] = self.loss_sums.get(name, 0.0) + (
                float(value.detach().cpu()) * count
            )
        increment_prediction = output["projected_log_ai_increment_mean"][support]
        increment_target = targets["projected_log_ai_increment"][support]
        reflectivity_prediction = output["signed_reflectivity_mean"][support]
        reflectivity_target = targets["signed_reflectivity"][support]
        self._update_regression(
            self.regression["projected_log_ai_increment"],
            increment_prediction,
            increment_target,
        )
        self._update_regression(
            self.regression["signed_reflectivity"],
            reflectivity_prediction,
            reflectivity_target,
        )
        state_truth = targets["state_fraction"][support].detach().double()
        state_probability = (
            torch.softmax(output["state_logits"][support], dim=-1).detach().double()
        )
        self.state_brier_sum += float(
            torch.sum(torch.sum((state_probability - state_truth) ** 2, dim=-1)).cpu()
        )
        self.state_cross_entropy_sum += float(
            torch.sum(
                -torch.sum(
                    state_truth * torch.log(state_probability.clamp_min(1.0e-12)),
                    dim=-1,
                )
            ).cpu()
        )
        increment_error = torch.abs(increment_prediction - increment_target).detach()
        increment_scale = output["projected_log_ai_increment_scale"][support].detach()
        reflectivity_error = torch.abs(
            reflectivity_prediction - reflectivity_target
        ).detach()
        reflectivity_scale = output["signed_reflectivity_scale"][support].detach()
        self.increment_coverage_1_sum += int(
            torch.count_nonzero(increment_error <= increment_scale).item()
        )
        self.increment_coverage_2_sum += int(
            torch.count_nonzero(increment_error <= 2.0 * increment_scale).item()
        )
        self.reflectivity_coverage_1_sum += int(
            torch.count_nonzero(reflectivity_error <= reflectivity_scale).item()
        )
        self.reflectivity_coverage_2_sum += int(
            torch.count_nonzero(reflectivity_error <= 2.0 * reflectivity_scale).item()
        )

    @staticmethod
    def _regression_metrics(values: np.ndarray) -> dict[str, float]:
        count, sum_p, sum_t, sum_p2, sum_t2, sum_pt, sum_error2 = values
        if count <= 0.0:
            raise ValueError("regression metric has no samples.")
        covariance = sum_pt - sum_p * sum_t / count
        variance_p = max(sum_p2 - sum_p * sum_p / count, 0.0)
        variance_t = max(sum_t2 - sum_t * sum_t / count, 0.0)
        denominator = np.sqrt(variance_p * variance_t)
        return {
            "rmse": float(np.sqrt(sum_error2 / count)),
            "correlation": float(covariance / denominator) if denominator > 0.0 else 0.0,
        }

    def finalize(self) -> dict[str, Any]:
        if self.count <= 0 or self.loss_weight <= 0:
            raise ValueError("cannot finalize empty evidence metrics.")
        return {
            "supported_samples": self.count,
            "loss": {
                name: float(value / self.loss_weight)
                for name, value in sorted(self.loss_sums.items())
            },
            "projected_log_ai_increment": {
                **self._regression_metrics(
                    self.regression["projected_log_ai_increment"]
                ),
                "coverage_1_scale": float(self.increment_coverage_1_sum / self.count),
                "coverage_2_scale": float(self.increment_coverage_2_sum / self.count),
            },
            "signed_reflectivity": {
                **self._regression_metrics(self.regression["signed_reflectivity"]),
                "coverage_1_scale": float(
                    self.reflectivity_coverage_1_sum / self.count
                ),
                "coverage_2_scale": float(
                    self.reflectivity_coverage_2_sum / self.count
                ),
            },
            "state_fraction": {
                "brier": float(self.state_brier_sum / self.count),
                "cross_entropy": float(self.state_cross_entropy_sum / self.count),
            },
        }


def _parent_chunks(
    parent_ids: Sequence[str],
    batch_size: int,
    *,
    require_donor: bool,
) -> list[tuple[str, ...]]:
    chunks = [
        tuple(parent_ids[start : start + batch_size])
        for start in range(0, len(parent_ids), batch_size)
    ]
    if require_donor:
        if len(parent_ids) < 2 or batch_size < 2:
            raise ValueError("matched seismic evaluation requires batches of two parents.")
        if len(chunks[-1]) == 1:
            if len(chunks) < 2 or len(chunks[-2]) <= 2:
                raise ValueError("cannot form a non-singleton matched donor batch.")
            donor_parent = chunks[-2][-1]
            chunks[-2] = chunks[-2][:-1]
            chunks[-1] = (donor_parent,) + chunks[-1]
    return chunks


def _evaluate_network_split(
    network: BandlimitedEvidenceNetwork,
    corpus: Corpus,
    parent_ids: Sequence[str],
    *,
    parent_batch_size: int,
    random_identity: int,
    perturbation_profile: ObservationPerturbationProfile,
    learning_config: LearningConfig,
    device: torch.device,
    conditions: Sequence[str],
    logger: logging.Logger,
    log_every_batches: int,
    label: str,
) -> dict[str, Any]:
    accumulators = {condition: _EvidenceMetricAccumulator() for condition in conditions}
    network.eval()
    chunks = _parent_chunks(
        parent_ids,
        parent_batch_size,
        require_donor="matched_seismic_shuffle" in conditions,
    )
    batch_count = len(chunks)
    started = time.perf_counter()
    with torch.no_grad():
        processed = 0
        for batch_index, current_ids in enumerate(chunks, start=1):
            examples = _prepare_parent_batch(corpus, current_ids)
            donors = (
                _different_parent_donors(examples)
                if "matched_seismic_shuffle" in conditions
                else None
            )
            for condition in conditions:
                inputs, targets = _stack_evidence_batch(
                    examples,
                    (condition,),
                    random_identity=random_identity,
                    perturbation_profile=perturbation_profile,
                    device=device,
                    donors=donors if condition == "matched_seismic_shuffle" else None,
                )
                output = network(**inputs)
                losses = _configured_evidence_loss(
                    network, output, targets, learning_config
                )
                accumulators[condition].update(output, targets, losses)
            processed += len(current_ids)
            if (
                batch_index == 1
                or batch_index % log_every_batches == 0
                or batch_index == batch_count
            ):
                logger.info(
                    "%s | batch=%d/%d | parents=%d/%d | elapsed=%.1fs",
                    label,
                    batch_index,
                    batch_count,
                    processed,
                    len(parent_ids),
                    time.perf_counter() - started,
                )
    return {name: accumulator.finalize() for name, accumulator in accumulators.items()}


def _checkpoint_metadata(
    *,
    mode: str,
    epoch: int,
    selection_loss: float,
    corpus: Corpus,
    preflight_path: Path,
    training_parent_ids: Sequence[str],
    tuning_parent_ids: Sequence[str],
    learning_config: LearningConfig,
    perturbation_profile: ObservationPerturbationProfile,
) -> dict[str, Any]:
    return {
        "training_mode": mode,
        "epoch": int(epoch),
        "selection_loss": float(selection_loss),
        "corpus_schema": corpus.manifest.get("schema_version"),
        "sample_domain": corpus.benchmark.sample_domain,
        "sample_unit": corpus.benchmark.sample_unit,
        "depth_basis": corpus.benchmark.depth_basis,
        "preflight_report": preflight_path.as_posix(),
        "training_parent_ids": list(training_parent_ids),
        "tuning_parent_ids": list(tuning_parent_ids),
        "learning_config": asdict(learning_config),
        "perturbation_profile": asdict(perturbation_profile),
    }


def train_generator(
    corpus_root: str | Path,
    preflight_report: str | Path,
    output_dir: str | Path,
    *,
    evidence_config: EvidenceNetworkConfig,
    config: LearningConfig = LearningConfig(),
    perturbation_profile: ObservationPerturbationProfile = ObservationPerturbationProfile(),
    input_mode: str = "full",
    device_name: str = "auto",
) -> dict[str, Any]:
    """Train one evidence model through the canonical parent-atomic seam."""

    started = time.perf_counter()
    output = Path(output_dir)
    report_path = output / "training_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(output)
    device, runtime = resolve_device(device_name)
    corpus = load_corpus(corpus_root)
    preflight_path = Path(preflight_report)
    preflight = _read_report(preflight_path)
    normalization_parent_ids = _validate_preflight_report(preflight, corpus)
    normalization_examples = _prepare_parent_batch(corpus, normalization_parent_ids)
    network_config = _network_config_for_mode(
        evidence_config, normalization_examples, input_mode
    )
    training_parent_ids = _select_split_parents(
        corpus, "training", config.training_parent_count
    )
    tuning_parent_ids = _select_split_parents(
        corpus, "tuning", config.tuning_parent_count
    )
    if set(training_parent_ids).intersection(tuning_parent_ids):
        raise ValueError("training and tuning parents overlap.")

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.random_seed)
    network = BandlimitedEvidenceNetwork(network_config).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    training_views = ["clean", "dirty"]
    if config.include_peak_poor:
        training_views.append("peak_poor")
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    status = "running"
    report: dict[str, Any] = {
        "schema": TRAINING_REPORT_SCHEMA,
        "status": status,
        "input_mode": input_mode,
        "runtime": runtime,
        "corpus": {
            "root": Path(corpus_root).as_posix(),
            "schema": corpus.manifest.get("schema_version"),
            "sample_domain": corpus.benchmark.sample_domain,
            "sample_unit": corpus.benchmark.sample_unit,
            "depth_basis": corpus.benchmark.depth_basis,
        },
        "preflight_report": preflight_path.as_posix(),
        "network_config": asdict(network_config),
        "learning_config": asdict(config),
        "perturbation_profile": asdict(perturbation_profile),
        "training_parent_ids": list(training_parent_ids),
        "tuning_parent_ids": list(tuning_parent_ids),
        "history": history,
        "best_epoch": best_epoch,
        "best_selection_loss": None,
    }
    _atomic_json(report_path, report)
    logger.info(
        "evidence training start | mode=%s | device=%s | training_parents=%d | tuning_parents=%d | epochs=%d",
        input_mode,
        device,
        len(training_parent_ids),
        len(tuning_parent_ids),
        config.epochs,
    )

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        order = np.asarray(training_parent_ids, dtype=object)
        rng = np.random.Generator(np.random.PCG64DXSM(config.random_seed + epoch))
        rng.shuffle(order)
        train_metrics = _EvidenceMetricAccumulator()
        network.train()
        batch_count = (len(order) + config.parent_batch_size - 1) // config.parent_batch_size
        for batch_index, start in enumerate(
            range(0, len(order), config.parent_batch_size), start=1
        ):
            current_ids = tuple(str(value) for value in order[start : start + config.parent_batch_size])
            examples = _prepare_parent_batch(corpus, current_ids)
            inputs, targets = _stack_evidence_batch(
                examples,
                training_views,
                random_identity=config.dirty_random_identity + epoch,
                perturbation_profile=perturbation_profile,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            output_values = network(**inputs)
            losses = _configured_evidence_loss(
                network, output_values, targets, config
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            train_metrics.update(output_values, targets, losses)
            if (
                batch_index == 1
                or batch_index % config.log_every_batches == 0
                or batch_index == batch_count
            ):
                logger.info(
                    "epoch %d/%d | batch=%d/%d | train_loss=%.6f | elapsed=%.1fs",
                    epoch,
                    config.epochs,
                    batch_index,
                    batch_count,
                    float(losses["loss"].detach().cpu()),
                    time.perf_counter() - epoch_started,
                )
        train_result = train_metrics.finalize()
        tuning_result = _evaluate_network_split(
            network,
            corpus,
            tuning_parent_ids,
            parent_batch_size=config.parent_batch_size,
            random_identity=config.dirty_random_identity,
            perturbation_profile=perturbation_profile,
            learning_config=config,
            device=device,
            conditions=("clean", "dirty", "peak_poor"),
            logger=logger,
            log_every_batches=config.log_every_batches,
            label=f"epoch {epoch}/{config.epochs} tuning",
        )
        selection_loss = float(
            0.5
            * (
                tuning_result["clean"]["loss"]["loss"]
                + tuning_result["dirty"]["loss"]["loss"]
            )
        )
        epoch_report = {
            "epoch": epoch,
            "train": train_result,
            "tuning": tuning_result,
            "selection_loss": selection_loss,
            "elapsed_seconds": float(time.perf_counter() - epoch_started),
        }
        history.append(epoch_report)
        metadata = _checkpoint_metadata(
            mode=input_mode,
            epoch=epoch,
            selection_loss=selection_loss,
            corpus=corpus,
            preflight_path=preflight_path,
            training_parent_ids=training_parent_ids,
            tuning_parent_ids=tuning_parent_ids,
            learning_config=config,
            perturbation_profile=perturbation_profile,
        )
        improved = selection_loss < best_loss
        if improved:
            best_loss = selection_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        training_state = {
            "optimizer": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_selection_loss": best_loss,
        }
        save_checkpoint(
            output / "last.pt",
            model_state=network.state_dict(),
            model_config=asdict(network_config),
            metadata=metadata,
            training_state=training_state,
            overwrite=True,
        )
        if improved:
            save_checkpoint(
                output / "best.pt",
                model_state=network.state_dict(),
                model_config=asdict(network_config),
                metadata=metadata,
                training_state=training_state,
                overwrite=True,
            )
        report.update(
            {
                "status": "running",
                "history": history,
                "best_epoch": best_epoch,
                "best_selection_loss": float(best_loss),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        _atomic_json(report_path, report)
        logger.info(
            "epoch %d/%d complete | train_loss=%.6f | tuning_clean=%.6f | tuning_dirty=%.6f | selection=%.6f | best_epoch=%d",
            epoch,
            config.epochs,
            train_result["loss"]["loss"],
            tuning_result["clean"]["loss"]["loss"],
            tuning_result["dirty"]["loss"]["loss"],
            selection_loss,
            best_epoch,
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            status = "early_stopped"
            logger.info(
                "early stopping | epoch=%d | best_epoch=%d | patience=%d",
                epoch,
                best_epoch,
                config.early_stopping_patience,
            )
            break
    else:
        status = "completed"

    report.update(
        {
            "status": status,
            "history": history,
            "best_epoch": best_epoch,
            "best_selection_loss": float(best_loss),
            "elapsed_seconds": float(time.perf_counter() - started),
            "artifacts": {"best_checkpoint": "best.pt", "last_checkpoint": "last.pt"},
        }
    )
    _atomic_json(report_path, report)
    logger.info(
        "evidence training finished | status=%s | best_epoch=%d | best_selection_loss=%.6f | elapsed=%.1fs",
        status,
        best_epoch,
        best_loss,
        report["elapsed_seconds"],
    )
    return report


def evaluate_generator(
    corpus_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config: EvaluationConfig = EvaluationConfig(),
    perturbation_profile: ObservationPerturbationProfile = ObservationPerturbationProfile(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate one frozen evidence checkpoint on one canonical split."""

    started = time.perf_counter()
    output = Path(output_dir)
    report_path = output / "evaluation_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(output, file_name="evaluation.log")
    device, runtime = resolve_device(device_name)
    corpus = load_corpus(corpus_root)
    payload = load_checkpoint(checkpoint_path)
    metadata = payload["metadata"]
    if metadata.get("corpus_schema") != corpus.manifest.get("schema_version"):
        raise ValueError("checkpoint and evaluation corpus schemas differ.")
    for name, value in (
        ("sample_domain", corpus.benchmark.sample_domain),
        ("sample_unit", corpus.benchmark.sample_unit),
        ("depth_basis", corpus.benchmark.depth_basis),
    ):
        if metadata.get(name) != value:
            raise ValueError(f"checkpoint {name} differs from the evaluation corpus.")
    network_config = EvidenceNetworkConfig.from_mapping(payload["model_config"])
    network = BandlimitedEvidenceNetwork(network_config).to(device)
    network.load_state_dict(payload["model_state"], strict=True)
    learning_value = metadata.get("learning_config")
    if not isinstance(learning_value, Mapping):
        raise ValueError("checkpoint lacks learning_config metadata.")
    learning_config = LearningConfig.from_mapping(learning_value)
    parent_ids = _select_split_parents(corpus, config.split, config.parent_count)
    conditions = ["clean", "dirty", "peak_poor"]
    if metadata.get("training_mode") != "no_seismic":
        conditions.append("matched_seismic_shuffle")
    logger.info(
        "evidence evaluation start | checkpoint=%s | mode=%s | split=%s | parents=%d | device=%s",
        checkpoint_path,
        metadata.get("training_mode"),
        config.split,
        len(parent_ids),
        device,
    )
    metrics = _evaluate_network_split(
        network,
        corpus,
        parent_ids,
        parent_batch_size=config.parent_batch_size,
        random_identity=config.dirty_random_identity,
        perturbation_profile=perturbation_profile,
        learning_config=learning_config,
        device=device,
        conditions=conditions,
        logger=logger,
        log_every_batches=config.log_every_batches,
        label=f"{config.split} evaluation",
    )
    report = {
        "schema": EVALUATION_REPORT_SCHEMA,
        "status": "completed",
        "checkpoint": Path(checkpoint_path).as_posix(),
        "checkpoint_epoch": metadata.get("epoch"),
        "input_mode": metadata.get("training_mode"),
        "runtime": runtime,
        "corpus": {
            "root": Path(corpus_root).as_posix(),
            "schema": corpus.manifest.get("schema_version"),
            "sample_domain": corpus.benchmark.sample_domain,
            "sample_unit": corpus.benchmark.sample_unit,
            "depth_basis": corpus.benchmark.depth_basis,
        },
        "evaluation_config": asdict(config),
        "perturbation_profile": asdict(perturbation_profile),
        "parent_ids": list(parent_ids),
        "metrics": metrics,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _atomic_json(report_path, report)
    logger.info(
        "evidence evaluation finished | split=%s | elapsed=%.1fs | report=%s",
        config.split,
        report["elapsed_seconds"],
        report_path,
    )
    return report


__all__ = [
    "EVALUATION_REPORT_SCHEMA",
    "EvaluationConfig",
    "LearningConfig",
    "TARGET_PREFLIGHT_SCHEMA",
    "TRAINING_REPORT_SCHEMA",
    "TargetPreflightConfig",
    "evaluate_generator",
    "preflight_evidence_targets",
    "train_generator",
    "validate_observation_contract",
]
