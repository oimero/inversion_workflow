"""Training and evaluation behind one compact generator-facing interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace as dataclass_replace
from itertools import product
import logging
from pathlib import Path
import random
import time
from collections.abc import Callable, Iterable
from typing import Any, Mapping

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.nn import functional as F

from cup.physics.numpy_backend import forward_depth, forward_time, velocity_from_ai
from cup.utils.io import write_json
from ginn_v2.artifacts import Corpus, parent_observation_tiles
from ginn_v2.evidence import (
    ObservableTargetProbe,
    build_observable_targets,
    evidence_loss,
)
from ginn_v2.contracts import (
    CoefficientVarianceCalibration,
    ObservableEvidence,
    ObservationTile,
    SegmentExtent,
)
from ginn_v2.generator import (
    ConditionalGenerator,
    SegmentProfileHeadConfig,
)
from ginn_v2.representation import (
    build_lfm_anchor,
    build_segment_profile_features,
    fit_profile_coefficients,
    lfm_residual_from_anchor,
    profile_basis,
    profile_sufficient_statistics,
)
from ginn_v2.semi_markov import (
    SampledPath,
    SemiMarkovConditioning,
    SemiMarkovDecodePolicy,
    SemiMarkovMarginals,
    SemiMarkovPrior,
    exact_semi_markov_posterior,
    prior_with_same_state_renewal,
    renewal_probability_from_reflectivity,
)


CheckpointCallback = Callable[
    [int, ConditionalGenerator, Mapping[str, Any], bool],
    None,
]

_TENSOR_BATCH_KEYS = frozenset(
    {
        "seismic",
        "lfm_residual",
        "observed_valid",
        "lateral_m",
        "lateral_valid",
        "projected_log_ai_increment",
        "signed_reflectivity",
        "state_emission",
        "support",
    }
)


@dataclass(frozen=True)
class LearningConfig:
    epochs: int = 8
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    log_every_batches: int = 20
    gradient_clip_norm: float = 5.0
    increment_weight: float = 1.0
    reflectivity_weight: float = 0.5
    state_weight: float = 0.25
    scale_weight: float = 0.1
    random_seed: int = 20260802

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.log_every_batches <= 0:
            raise ValueError("epochs and log_every_batches must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer controls are invalid.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")
        for name in (
            "increment_weight",
            "reflectivity_weight",
            "state_weight",
            "scale_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.random_seed, int) or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be an integer in [0, 2**32).")


@dataclass(frozen=True)
class SegmentProfileLearningConfig:
    epochs: int = 4
    batch_size: int = 1024
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 5.0
    parameter_nll_weight: float = 0.25
    identifiability_condition_max: float = 100.0
    central_trace_count: int = 5
    training_parents_per_family: int = 128
    tuning_parents_per_family: int = 32
    random_seed: int = 20260803

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "batch_size",
            "central_trace_count",
            "training_parents_per_family",
            "tuning_parents_per_family",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "identifiability_condition_max",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.weight_decay < 0.0 or self.parameter_nll_weight < 0.0:
            raise ValueError("profile optimizer weights cannot be negative.")
        if self.central_trace_count % 2 == 0:
            raise ValueError("central_trace_count must be odd.")
        if not isinstance(self.random_seed, int) or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be an integer in [0, 2**32).")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        smoke: bool = False,
    ) -> "SegmentProfileLearningConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError(f"unknown segment profile learning keys: {unknown}")
        config = cls(**dict(value))
        if not smoke:
            return config
        return dataclass_replace(
            config,
            epochs=1,
            training_parents_per_family=1,
            tuning_parents_per_family=1,
        )


@dataclass(frozen=True)
class CoefficientVarianceCalibrationConfig:
    parents_per_family: int = 32
    batch_size: int = 2048
    central_trace_count: int = 5
    identifiability_condition_max: float = 100.0

    def __post_init__(self) -> None:
        for name in ("parents_per_family", "batch_size", "central_trace_count"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.central_trace_count % 2 == 0:
            raise ValueError("central_trace_count must be odd.")
        if (
            not np.isfinite(self.identifiability_condition_max)
            or self.identifiability_condition_max <= 0.0
        ):
            raise ValueError(
                "identifiability_condition_max must be finite and positive."
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        smoke: bool = False,
    ) -> "CoefficientVarianceCalibrationConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError(
                f"unknown coefficient variance calibration keys: {unknown}"
            )
        config = cls(**dict(value))
        if not smoke:
            return config
        return dataclass_replace(config, parents_per_family=1)


@dataclass(frozen=True)
class TargetAuditConfig:
    """Bounded L0-L2 evidence audit controls."""

    l0_parents_per_family: int = 4
    l1_parents_per_family: int = 2
    l2_training_parents_per_family: int = 16
    l2_tuning_parents_per_family: int = 8
    central_trace_count: int = 5
    figure_parent_count: int = 12
    hidden_channels: int = 24
    probe_layers: int = 3
    batch_size: int = 32
    l1_steps: int = 120
    l2_epochs: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    bootstrap_replicates: int = 500
    random_seed: int = 20260802
    l0_max_seconds: float = 120.0
    l1_max_seconds: float = 300.0
    l2_max_seconds: float = 1200.0
    closure_max_normalized_rmse: float = 0.01
    closure_min_correlation: float = 0.99
    minimum_continuous_std: float = 1.0e-6
    minimum_state_fraction: float = 0.01
    l1_minimum_loss_reduction_fraction: float = 0.50
    smoke: bool = False

    def __post_init__(self) -> None:
        integer_names = (
            "l0_parents_per_family",
            "l1_parents_per_family",
            "l2_training_parents_per_family",
            "l2_tuning_parents_per_family",
            "central_trace_count",
            "figure_parent_count",
            "hidden_channels",
            "probe_layers",
            "batch_size",
            "l1_steps",
            "l2_epochs",
            "bootstrap_replicates",
        )
        for name in integer_names:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        positive_names = (
            "learning_rate",
            "l0_max_seconds",
            "l1_max_seconds",
            "l2_max_seconds",
            "closure_max_normalized_rmse",
            "closure_min_correlation",
            "minimum_continuous_std",
            "minimum_state_fraction",
            "l1_minimum_loss_reduction_fraction",
        )
        for name in positive_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 < self.closure_min_correlation <= 1.0:
            raise ValueError("closure_min_correlation must be in (0, 1].")
        if not 0.0 < self.minimum_state_fraction < 1.0:
            raise ValueError("minimum_state_fraction must be in (0, 1).")
        if not 0.0 < self.l1_minimum_loss_reduction_fraction < 1.0:
            raise ValueError(
                "l1_minimum_loss_reduction_fraction must be in (0, 1)."
            )
        if not isinstance(self.random_seed, int) or not 0 <= self.random_seed < 2**32:
            raise ValueError("random_seed must be an integer in [0, 2**32).")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
        *,
        smoke: bool = False,
    ) -> "TargetAuditConfig":
        payload = dict(value or {})
        payload["smoke"] = bool(smoke)
        if smoke:
            payload.update(
                {
                    "l0_parents_per_family": 1,
                    "l1_parents_per_family": 1,
                    "l2_training_parents_per_family": 2,
                    "l2_tuning_parents_per_family": 1,
                    "central_trace_count": 3,
                    "figure_parent_count": 3,
                    "hidden_channels": 8,
                    "probe_layers": 1,
                    "batch_size": 8,
                    "l1_steps": 4,
                    "l2_epochs": 1,
                    "bootstrap_replicates": 50,
                }
            )
        return cls(**payload)


@dataclass(frozen=True)
class _AuditExample:
    parent_id: str
    geometry_family: str
    tile_id: str
    zone_id: str
    trace_index: int
    seismic: np.ndarray
    lfm_residual: np.ndarray
    observed_valid: np.ndarray
    support: np.ndarray
    projected_log_ai_increment: np.ndarray
    signed_reflectivity: np.ndarray
    state_id: np.ndarray


def seed_training_random_streams(seed: int) -> None:
    """Seed model initialization and training streams for comparable runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_AUDIT_TARGETS = (
    "projected_log_ai_increment",
    "signed_reflectivity",
    "state_emission",
)


def _stratified_parent_ids(
    corpus: Corpus,
    split: str,
    *,
    per_family: int,
    seed: int,
) -> tuple[str, ...]:
    index = corpus.benchmark.index
    rows = index.loc[
        index["split_role"].eq(split) & index["corpus_role"].eq("short_patch")
    ]
    families = tuple(sorted(rows["geometry_family"].astype(str).unique()))
    if families != ("none", "pinchout", "wedge"):
        raise ValueError(
            "target audit requires none, pinchout, and wedge geometry families."
        )
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for family in families:
        values = np.asarray(
            sorted(
                rows.loc[
                    rows["geometry_family"].eq(family), "realization_id"
                ].astype(str)
            ),
            dtype=object,
        )
        if values.size < per_family:
            raise ValueError(
                f"split {split!r} family {family!r} has {values.size} parents; "
                f"the audit requires {per_family}."
            )
        selected.extend(str(value) for value in rng.permutation(values)[:per_family])
    return tuple(selected)


def _central_trace_indices(width: int, count: int) -> np.ndarray:
    if width < count:
        raise ValueError("audit parent has fewer traces than central_trace_count.")
    start = (width - count) // 2
    return np.arange(start, start + count, dtype=np.int64)


def _geometry_family(corpus: Corpus, parent_id: str) -> str:
    selected = corpus.benchmark.index.loc[
        corpus.benchmark.index["realization_id"].eq(parent_id), "geometry_family"
    ]
    if len(selected) != 1:
        raise KeyError(f"cannot resolve geometry family for parent {parent_id!r}.")
    return str(selected.iloc[0])


def _load_audit_examples(
    corpus: Corpus,
    parent_ids: Iterable[str],
    *,
    central_trace_count: int,
) -> tuple[_AuditExample, ...]:
    examples: list[_AuditExample] = []
    for parent_id in parent_ids:
        parent = corpus.benchmark.read_parent(parent_id)
        family = _geometry_family(corpus, parent_id)
        for tile in parent_observation_tiles(corpus, parent_id):
            anchor = build_lfm_anchor(tile)
            target = build_observable_targets(
                tile,
                model_log_ai=parent.model_log_ai,
                state_highres=parent.state_id_highres,
                background_lfm_linear=anchor.model,
                anchor_support=anchor.model_support,
            )
            residual = lfm_residual_from_anchor(tile, anchor)
            for trace in _central_trace_indices(tile.width, central_trace_count):
                if not tile.lateral_valid[trace]:
                    continue
                support = target.support[trace]
                if np.count_nonzero(support) < 3:
                    continue
                examples.append(
                    _AuditExample(
                        parent_id=parent_id,
                        geometry_family=family,
                        tile_id=tile.identity,
                        zone_id=tile.identity.rsplit(":", 1)[-1],
                        trace_index=int(trace),
                        seismic=tile.seismic[trace].astype(np.float32),
                        lfm_residual=residual[trace].astype(np.float32),
                        observed_valid=tile.observed_valid[trace].astype(bool),
                        support=support.astype(bool),
                        projected_log_ai_increment=(
                            target.projected_log_ai_increment[trace].astype(np.float32)
                        ),
                        signed_reflectivity=target.signed_reflectivity[trace].astype(
                            np.float32
                        ),
                        state_id=target.state_id[trace].astype(np.int64),
                    )
                )
    if not examples:
        raise ValueError("target audit selected no supervised trace-zone examples.")
    return tuple(examples)


def _continuous_target_statistics(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError("continuous target statistics require finite samples.")
    quantiles = np.quantile(array, (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0))
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "zero_fraction": float(np.mean(array == 0.0)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "negative_fraction": float(np.mean(array < 0.0)),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("min", "p01", "p05", "p50", "p95", "p99", "max"),
                quantiles,
                strict=True,
            )
        },
        "robust_absolute_scale_p95": float(np.quantile(np.abs(array), 0.95)),
    }


def _state_target_statistics(values: np.ndarray) -> dict[str, Any]:
    state = np.asarray(values, dtype=np.int64).reshape(-1)
    if state.size == 0 or np.any((state < 0) | (state > 2)):
        raise ValueError("state target statistics require labels in [0, 2].")
    counts = np.bincount(state, minlength=3).astype(np.float64)
    probability = counts / np.sum(counts)
    entropy = -float(np.sum(probability * np.log(np.clip(probability, 1.0e-12, 1.0))))
    return {
        "count": int(state.size),
        "class_count": [int(value) for value in counts],
        "class_fraction": [float(value) for value in probability],
        "marginal_entropy": entropy,
        "majority_accuracy": float(np.max(probability)),
    }


def _correlation_arrays(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size < 2 or x.size != y.size or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _forward_parent_log_ai(parent: Any, log_ai: np.ndarray) -> np.ndarray:
    wavelet_time = np.asarray(parent.forward_context["wavelet_time_s"], dtype=np.float64)
    wavelet = np.asarray(parent.forward_context["wavelet_amplitude"], dtype=np.float64)
    if parent.sample_domain == "time":
        return np.asarray(forward_time(log_ai, wavelet_time, wavelet), dtype=np.float64)
    relation = dict(parent.forward_context["ai_velocity_relation"])
    velocity = velocity_from_ai(
        np.exp(log_ai),
        a=float(relation["a"]),
        b=float(relation["b"]),
    )
    return np.asarray(
        forward_depth(
            log_ai,
            velocity,
            parent.model_axis.coordinates,
            wavelet_time,
            wavelet,
            output_chunk_size=int(parent.forward_context["output_chunk_size"]),
        ),
        dtype=np.float64,
    )


def _log_ai_from_reflectivity(
    reflectivity: np.ndarray,
    initial_log_ai: np.ndarray,
) -> np.ndarray:
    values = np.asarray(reflectivity, dtype=np.float64)
    initial = np.asarray(initial_log_ai, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or initial.shape != (values.shape[0],):
        raise ValueError("reflectivity reconstruction shapes are invalid.")
    if np.any(np.abs(values) >= 1.0) or np.any(~np.isfinite(values)):
        raise ValueError("reflectivity reconstruction requires finite values in (-1, 1).")
    output = np.empty((values.shape[0], values.shape[1] + 1), dtype=np.float64)
    output[:, 0] = initial
    output[:, 1:] = initial[:, None] + np.cumsum(2.0 * np.arctanh(values), axis=1)
    return output


def _normalized_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
) -> float:
    predicted = np.asarray(prediction, dtype=np.float64)[support]
    expected = np.asarray(target, dtype=np.float64)[support]
    residual_rms = float(np.sqrt(np.mean((predicted - expected) ** 2)))
    target_rms = float(np.sqrt(np.mean(expected**2)))
    return residual_rms / max(target_rms, 1.0e-12)


def _best_forward_lag_samples(
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    *,
    maximum_lag: int = 8,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    valid = np.asarray(support, dtype=bool)
    if predicted.shape != expected.shape or valid.shape != predicted.shape:
        raise ValueError("forward lag arrays must share one shape.")
    if predicted.ndim != 2 or predicted.shape[1] < 3:
        raise ValueError("forward lag diagnostic requires [trace, sample] arrays.")
    bound = min(int(maximum_lag), predicted.shape[1] - 2)
    candidates: list[tuple[float, int, int]] = []
    for lag in range(-bound, bound + 1):
        if lag < 0:
            left = predicted[:, :lag]
            right = expected[:, -lag:]
            mask = valid[:, :lag] & valid[:, -lag:]
        elif lag > 0:
            left = predicted[:, lag:]
            right = expected[:, :-lag]
            mask = valid[:, lag:] & valid[:, :-lag]
        else:
            left = predicted
            right = expected
            mask = valid
        correlation = _correlation_arrays(left[mask], right[mask])
        if correlation is not None:
            candidates.append((correlation, -abs(lag), lag))
    if not candidates:
        return {"lag_samples": None, "correlation": None}
    correlation, _, lag = max(candidates)
    return {"lag_samples": int(lag), "correlation": float(correlation)}


def _parent_closure(corpus: Corpus, parent_id: str) -> dict[str, Any]:
    parent = corpus.benchmark.read_parent(parent_id)
    log_ai = np.asarray(parent.model_log_ai, dtype=np.float64)
    if np.any(~np.isfinite(log_ai)):
        raise ValueError("target audit forward closure requires finite model_log_ai.")
    reflectivity = np.tanh(0.5 * np.diff(log_ai, axis=-1))
    reconstructed = _log_ai_from_reflectivity(reflectivity, log_ai[:, 0])
    log_ai_max_abs = float(np.max(np.abs(reconstructed - log_ai)))
    forward = _forward_parent_log_ai(parent, reconstructed)
    support = np.asarray(parent.observed_valid, dtype=bool)
    expected = np.asarray(parent.model_consistent_seismic, dtype=np.float64)
    correlation = _correlation_arrays(forward[support], expected[support])
    lag = _best_forward_lag_samples(forward, expected, support)
    return {
        "parent_id": parent_id,
        "geometry_family": _geometry_family(corpus, parent_id),
        "reflectivity_log_ai_max_abs": log_ai_max_abs,
        "forward_normalized_rmse": _normalized_rmse(forward, expected, support),
        "forward_correlation": correlation,
        "forward_best_lag_samples": lag["lag_samples"],
        "forward_best_lag_correlation": lag["correlation"],
    }


def _write_l0_figures(
    corpus: Corpus,
    parent_ids: Iterable[str],
    output_dir: Path,
) -> list[str]:
    import matplotlib.pyplot as plt

    directory = output_dir / "figures" / "l0"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, parent_id in enumerate(parent_ids):
        parent = corpus.benchmark.read_parent(parent_id)
        trace = int(parent.lateral_m.size // 2)
        axis = parent.model_axis.coordinates
        log_ai = np.asarray(parent.model_log_ai[trace], dtype=np.float64)
        lfm = np.asarray(parent.lfm[trace], dtype=np.float64)
        seismic = np.asarray(parent.seismic[trace], dtype=np.float64)
        linear_anchor = np.full_like(log_ai, np.nan)
        for tile in parent_observation_tiles(corpus, parent_id):
            anchor = build_lfm_anchor(tile)
            anchor_support = anchor.model_support[trace]
            linear_anchor[anchor_support] = anchor.model[trace, anchor_support]
        increment = np.where(np.isfinite(linear_anchor), log_ai - linear_anchor, np.nan)
        reflectivity = np.zeros_like(log_ai)
        reflectivity[1:] = np.tanh(0.5 * np.diff(log_ai))
        factor = int(
            round(
                parent.model_axis.sample_interval
                / parent.highres_axis.sample_interval
            )
        )
        state = np.asarray(parent.state_id_highres[trace, ::factor], dtype=np.float64)
        state[state < 0.0] = np.nan
        figure, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(axis, seismic, color="black", linewidth=0.9)
        axes[0].axhline(0.0, color="0.7", linewidth=0.5)
        axes[0].set_ylabel("seismic")
        axes[1].plot(axis, log_ai, label="model log-AI", linewidth=1.0)
        axes[1].plot(axis, lfm, label="full LFM", linewidth=0.9)
        axes[1].set_ylabel("log-AI")
        axes[1].legend(loc="best", fontsize=8)
        axes[2].plot(axis, increment, color="tab:green", linewidth=0.9)
        axes[2].axhline(0.0, color="0.7", linewidth=0.5)
        axes[2].set_ylabel("AI - linear anchor")
        axes[3].plot(axis, reflectivity, color="tab:red", linewidth=0.9)
        state_scale = max(float(np.max(np.abs(reflectivity))), 1.0e-6)
        axes[3].step(
            axis,
            (state - 1.0) * state_scale,
            where="mid",
            color="tab:blue",
            alpha=0.45,
            linewidth=0.8,
            label="state (scaled)",
        )
        axes[3].set_ylabel("reflectivity")
        axes[3].set_xlabel(
            f"{parent.model_axis.sample_domain} [{parent.model_axis.unit}]"
        )
        axes[3].legend(loc="best", fontsize=8)
        family = _geometry_family(corpus, parent_id)
        figure.suptitle(f"L0 target audit | {family} | {parent_id}", fontsize=10)
        figure.tight_layout()
        path = directory / f"target_{index:03d}_{family}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths.append(str(path.relative_to(output_dir)))
    return paths


def _values_for_target(
    examples: Iterable[_AuditExample],
    target_name: str,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for example in examples:
        target = (
            example.state_id
            if target_name == "state_emission"
            else getattr(example, target_name)
        )
        values.append(np.asarray(target)[example.support])
    if not values:
        raise ValueError(f"no values collected for target {target_name!r}.")
    return np.concatenate(values)


def _target_statistics_for_examples(
    examples: tuple[_AuditExample, ...],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("target statistics group is empty.")
    return {
        "example_count": len(examples),
        "projected_log_ai_increment": _continuous_target_statistics(
            _values_for_target(examples, "projected_log_ai_increment")
        ),
        "signed_reflectivity": _continuous_target_statistics(
            _values_for_target(examples, "signed_reflectivity")
        ),
        "state_emission": _state_target_statistics(
            _values_for_target(examples, "state_emission")
        ),
    }


def _run_l0(
    corpus: Corpus,
    config: TargetAuditConfig,
    output_dir: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, float]]:
    started = time.perf_counter()
    parent_ids = _stratified_parent_ids(
        corpus,
        "training",
        per_family=config.l0_parents_per_family,
        seed=config.random_seed,
    )
    examples = _load_audit_examples(
        corpus,
        parent_ids,
        central_trace_count=config.central_trace_count,
    )
    all_statistics = _target_statistics_for_examples(examples)
    target_statistics = {
        target: all_statistics[target] for target in _AUDIT_TARGETS
    }
    grouped: dict[str, Any] = {}
    for family in ("none", "pinchout", "wedge"):
        family_examples = tuple(
            example for example in examples if example.geometry_family == family
        )
        grouped[family] = _target_statistics_for_examples(family_examples)
    zone_statistics = {
        zone_id: _target_statistics_for_examples(
            tuple(example for example in examples if example.zone_id == zone_id)
        )
        for zone_id in sorted({example.zone_id for example in examples})
    }
    parent_statistics = {
        parent_id: _target_statistics_for_examples(
            tuple(example for example in examples if example.parent_id == parent_id)
        )
        for parent_id in parent_ids
    }
    seismic_values = np.concatenate(
        [example.seismic[example.observed_valid] for example in examples]
    )
    residual_values = np.concatenate(
        [example.lfm_residual[example.observed_valid] for example in examples]
    )
    scales = {
        "seismic": float(np.quantile(np.abs(seismic_values), 0.95)),
        "lfm_residual": float(np.quantile(np.abs(residual_values), 0.95)),
        "projected_log_ai_increment": target_statistics[
            "projected_log_ai_increment"
        ]["robust_absolute_scale_p95"],
        "signed_reflectivity": target_statistics["signed_reflectivity"][
            "robust_absolute_scale_p95"
        ],
    }
    closures = [_parent_closure(corpus, parent_id) for parent_id in parent_ids]
    figure_ids = parent_ids[: min(len(parent_ids), config.figure_parent_count)]
    figures = _write_l0_figures(corpus, figure_ids, output_dir)
    failures: list[str] = []
    for target in ("projected_log_ai_increment", "signed_reflectivity"):
        if target_statistics[target]["std"] < config.minimum_continuous_std:
            failures.append(f"{target}:degenerate_std")
    reflectivity_stats = target_statistics["signed_reflectivity"]
    if reflectivity_stats["positive_fraction"] < config.minimum_state_fraction:
        failures.append("signed_reflectivity:insufficient_positive_fraction")
    if reflectivity_stats["negative_fraction"] < config.minimum_state_fraction:
        failures.append("signed_reflectivity:insufficient_negative_fraction")
    for state_id, fraction in enumerate(
        target_statistics["state_emission"]["class_fraction"]
    ):
        if fraction < config.minimum_state_fraction:
            failures.append(f"state_emission:class_{state_id}_underrepresented")
    for name, value in scales.items():
        if not np.isfinite(value) or value <= config.minimum_continuous_std:
            failures.append(f"scale:{name}:degenerate")
    for closure in closures:
        if (
            closure["forward_normalized_rmse"]
            > config.closure_max_normalized_rmse
        ):
            failures.append(f"closure:{closure['parent_id']}:rmse")
        correlation = closure["forward_correlation"]
        if correlation is None or correlation < config.closure_min_correlation:
            failures.append(f"closure:{closure['parent_id']}:correlation")
    elapsed = time.perf_counter() - started
    if elapsed > config.l0_max_seconds:
        failures.append("budget:l0_exceeded")
    report = {
        "status": "passed" if not failures else "failed",
        "elapsed_seconds": elapsed,
        "parent_ids": list(parent_ids),
        "example_count": len(examples),
        "target_statistics": target_statistics,
        "geometry_family_statistics": grouped,
        "zone_statistics": zone_statistics,
        "parent_statistics": parent_statistics,
        "global_scales": scales,
        "forward_closure": closures,
        "figures": figures,
        "failures": failures,
    }
    logger.info(
        "target audit L0 complete | status=%s | parents=%d | examples=%d | "
        "elapsed=%.1fs",
        report["status"],
        len(parent_ids),
        len(examples),
        elapsed,
    )
    return report, scales


def _pack_audit_examples(
    examples: tuple[_AuditExample, ...],
    *,
    include_matched_shuffle: bool = False,
) -> dict[str, Any]:
    maximum = max(example.seismic.size for example in examples)
    count = len(examples)
    seismic = np.zeros((count, maximum), dtype=np.float32)
    residual = np.zeros_like(seismic)
    observed = np.zeros((count, maximum), dtype=bool)
    support = np.zeros_like(observed)
    increment = np.zeros_like(seismic)
    reflectivity = np.zeros_like(seismic)
    state = np.full((count, maximum), -1, dtype=np.int64)
    for index, example in enumerate(examples):
        size = example.seismic.size
        seismic[index, :size] = example.seismic
        residual[index, :size] = example.lfm_residual
        observed[index, :size] = example.observed_valid
        support[index, :size] = example.support
        increment[index, :size] = example.projected_log_ai_increment
        reflectivity[index, :size] = example.signed_reflectivity
        state[index, :size] = example.state_id

    packed: dict[str, Any] = {
        "seismic": seismic,
        "lfm_residual": residual,
        "observed_valid": observed,
        "support": support,
        "projected_log_ai_increment": increment,
        "signed_reflectivity": reflectivity,
        "state_emission": state,
        "parent_id": np.asarray([example.parent_id for example in examples], dtype=object),
        "geometry_family": np.asarray(
            [example.geometry_family for example in examples], dtype=object
        ),
    }
    if not include_matched_shuffle:
        return packed

    shuffled = seismic.copy()
    groups: dict[tuple[str, str], list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault((example.parent_id, example.tile_id), []).append(index)
    paired_group_count = 0
    recipient_valid_samples = 0
    common_valid_samples = 0
    changed_valid_samples = 0
    squared_change = 0.0
    for indices in groups.values():
        ordered = sorted(indices, key=lambda item: examples[item].trace_index)
        if len(ordered) <= 1:
            continue
        paired_group_count += 1
        donors = ordered[1:] + ordered[:1]
        for recipient, donor in zip(ordered, donors, strict=True):
            common = observed[recipient] & observed[donor]
            recipient_valid_samples += int(np.count_nonzero(observed[recipient]))
            common_valid_samples += int(np.count_nonzero(common))
            difference = seismic[donor, common] - seismic[recipient, common]
            changed_valid_samples += int(np.count_nonzero(difference != 0.0))
            squared_change += float(np.sum(difference.astype(np.float64) ** 2))
            shuffled[recipient, common] = seismic[donor, common]
    if paired_group_count == 0 or common_valid_samples == 0:
        raise ValueError(
            "matched seismic shuffle requires overlapping traces per parent-zone."
        )
    packed["shuffled_seismic"] = shuffled
    packed["matched_shuffle_diagnostics"] = {
        "paired_group_count": paired_group_count,
        "recipient_valid_samples": recipient_valid_samples,
        "common_valid_samples": common_valid_samples,
        "common_valid_fraction": common_valid_samples
        / max(recipient_valid_samples, 1),
        "changed_valid_samples": changed_valid_samples,
        "changed_common_fraction": changed_valid_samples
        / max(common_valid_samples, 1),
        "change_rms": float(np.sqrt(squared_change / common_valid_samples)),
    }
    return packed


def _probe_class_weight(data: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    state = np.asarray(data["state_emission"], dtype=np.int64)
    support = np.asarray(data["support"], dtype=bool)
    counts = np.bincount(state[support], minlength=3).astype(np.float64)
    if np.any(counts <= 0.0):
        raise ValueError("probe training requires every state class.")
    weight = np.sum(counts) / (3.0 * counts)
    return torch.as_tensor(weight, dtype=torch.float32, device=device)


def _probe_loss(
    model: ObservableTargetProbe,
    data: Mapping[str, Any],
    indices: np.ndarray,
    *,
    target_name: str,
    input_mode: str,
    scales: Mapping[str, float],
    device: torch.device,
    class_weight: torch.Tensor | None,
) -> torch.Tensor:
    seismic = torch.as_tensor(
        np.asarray(data["seismic"])[indices], dtype=torch.float32, device=device
    )
    residual = torch.as_tensor(
        np.asarray(data["lfm_residual"])[indices],
        dtype=torch.float32,
        device=device,
    )
    observed = torch.as_tensor(
        np.asarray(data["observed_valid"])[indices], dtype=torch.bool, device=device
    )
    support = torch.as_tensor(
        np.asarray(data["support"])[indices], dtype=torch.bool, device=device
    )
    output = model(
        seismic,
        residual,
        observed,
        input_mode=input_mode,
        seismic_scale=float(scales["seismic"]),
        lfm_residual_scale=float(scales["lfm_residual"]),
    )
    if target_name == "state_emission":
        target = torch.as_tensor(
            np.asarray(data[target_name])[indices], dtype=torch.long, device=device
        )
        return F.cross_entropy(
            output[support],
            target[support],
            weight=class_weight,
        )
    target = torch.as_tensor(
        np.asarray(data[target_name])[indices], dtype=torch.float32, device=device
    )
    scale = float(scales[target_name])
    return F.smooth_l1_loss(
        output[support] / scale,
        target[support] / scale,
        beta=1.0,
    )


def _train_probe(
    data: Mapping[str, Any],
    *,
    target_name: str,
    input_mode: str,
    scales: Mapping[str, float],
    config: TargetAuditConfig,
    device: torch.device,
    steps: int,
    seed: int,
    deadline: float,
) -> tuple[ObservableTargetProbe, dict[str, Any]]:
    seed_training_random_streams(seed)
    model = ObservableTargetProbe(
        target_name,
        hidden_channels=config.hidden_channels,
        layers=config.probe_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    indices = np.arange(np.asarray(data["seismic"]).shape[0], dtype=np.int64)
    class_weight = (
        _probe_class_weight(data, device)
        if target_name == "state_emission"
        else None
    )
    model.eval()
    with torch.no_grad():
        initial = float(
            _probe_loss(
                model,
                data,
                indices,
                target_name=target_name,
                input_mode=input_mode,
                scales=scales,
                device=device,
                class_weight=class_weight,
            ).cpu()
        )
    rng = np.random.default_rng(seed + 1)
    completed = 0
    timed_out = False
    model.train()
    for _ in range(steps):
        if time.perf_counter() > deadline:
            timed_out = True
            break
        batch = rng.choice(
            indices,
            size=min(config.batch_size, indices.size),
            replace=False,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = _probe_loss(
            model,
            data,
            batch,
            target_name=target_name,
            input_mode=input_mode,
            scales=scales,
            device=device,
            class_weight=class_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        completed += 1
    model.eval()
    with torch.no_grad():
        final = float(
            _probe_loss(
                model,
                data,
                indices,
                target_name=target_name,
                input_mode=input_mode,
                scales=scales,
                device=device,
                class_weight=class_weight,
            ).cpu()
        )
    reduction = (initial - final) / max(abs(initial), 1.0e-12)
    return model, {
        "initial_loss": initial,
        "final_loss": final,
        "loss_reduction_fraction": reduction,
        "requested_steps": int(steps),
        "completed_steps": int(completed),
        "timed_out": timed_out,
    }


def _predict_probe(
    model: ObservableTargetProbe,
    data: Mapping[str, Any],
    *,
    input_mode: str,
    scales: Mapping[str, float],
    device: torch.device,
    batch_size: int,
    shuffled_seismic: bool = False,
) -> np.ndarray:
    source = "shuffled_seismic" if shuffled_seismic else "seismic"
    seismic = np.asarray(data[source])
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, seismic.shape[0], batch_size):
            stop = min(start + batch_size, seismic.shape[0])
            values = model(
                torch.as_tensor(
                    seismic[start:stop], dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    np.asarray(data["lfm_residual"])[start:stop],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    np.asarray(data["observed_valid"])[start:stop],
                    dtype=torch.bool,
                    device=device,
                ),
                input_mode=input_mode,
                seismic_scale=float(scales["seismic"]),
                lfm_residual_scale=float(scales["lfm_residual"]),
            )
            output.append(values.cpu().numpy())
    return np.concatenate(output, axis=0)


def _balanced_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    recalls = []
    for state in range(3):
        mask = target == state
        if np.any(mask):
            recalls.append(float(np.mean(prediction[mask] == state)))
    if not recalls:
        raise ValueError("balanced accuracy has no represented classes.")
    return float(np.mean(recalls))


def _probe_metrics_by_parent(
    prediction: np.ndarray,
    data: Mapping[str, Any],
    *,
    target_name: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    parent_ids = np.asarray(data["parent_id"], dtype=object)
    target = np.asarray(data[target_name])
    support = np.asarray(data["support"], dtype=bool)
    rows: dict[str, dict[str, float]] = {}
    aggregate_prediction: list[np.ndarray] = []
    aggregate_target: list[np.ndarray] = []
    for parent_id in sorted(set(str(value) for value in parent_ids)):
        examples = parent_ids == parent_id
        mask = support[examples]
        expected = target[examples][mask]
        if target_name == "state_emission":
            predicted = np.argmax(prediction[examples], axis=-1)[mask]
            rows[parent_id] = {
                "balanced_accuracy": _balanced_accuracy(predicted, expected),
                "accuracy": float(np.mean(predicted == expected)),
            }
        else:
            predicted = prediction[examples][mask]
            residual = predicted - expected
            row = {
                "rmse": float(np.sqrt(np.mean(residual * residual))),
                "mae": float(np.mean(np.abs(residual))),
                "correlation": _correlation_arrays(predicted, expected),
            }
            if target_name == "signed_reflectivity":
                nonzero = np.abs(expected) > 1.0e-8
                row["polarity_accuracy"] = (
                    float(np.mean(np.sign(predicted[nonzero]) == np.sign(expected[nonzero])))
                    if np.any(nonzero)
                    else 0.0
                )
            rows[parent_id] = row
        aggregate_prediction.append(predicted)
        aggregate_target.append(expected)
    predicted_all = np.concatenate(aggregate_prediction)
    target_all = np.concatenate(aggregate_target)
    if target_name == "state_emission":
        aggregate = {
            "balanced_accuracy": _balanced_accuracy(predicted_all, target_all),
            "accuracy": float(np.mean(predicted_all == target_all)),
        }
    else:
        residual = predicted_all - target_all
        aggregate = {
            "rmse": float(np.sqrt(np.mean(residual * residual))),
            "mae": float(np.mean(np.abs(residual))),
            "correlation": _correlation_arrays(predicted_all, target_all),
        }
        if target_name == "signed_reflectivity":
            nonzero = np.abs(target_all) > 1.0e-8
            aggregate["polarity_accuracy"] = float(
                np.mean(
                    np.sign(predicted_all[nonzero]) == np.sign(target_all[nonzero])
                )
            )
    return aggregate, rows


def _probe_bootstrap(
    left: Mapping[str, Mapping[str, float]],
    right: Mapping[str, Mapping[str, float]],
    *,
    metric: str,
    lower_is_better: bool,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    parent_ids = sorted(set(left).intersection(right))
    if not parent_ids:
        raise ValueError("probe comparison has no paired parents.")
    delta = np.asarray(
        [left[parent][metric] - right[parent][metric] for parent in parent_ids],
        dtype=np.float64,
    )
    summary = _bootstrap_delta(delta, seed=seed, replicates=replicates)
    summary["metric"] = metric
    summary["lower_is_better"] = bool(lower_is_better)
    summary["passed"] = bool(
        summary["bootstrap_95_high"] < 0.0
        if lower_is_better
        else summary["bootstrap_95_low"] > 0.0
    )
    summary["improvement_parent_fraction"] = float(
        np.mean(delta < 0.0) if lower_is_better else np.mean(delta > 0.0)
    )
    return summary


def _comparison_has_correct_mean(
    comparison: Mapping[str, Any],
) -> bool:
    mean = float(comparison["mean_delta"])
    return mean < 0.0 if bool(comparison["lower_is_better"]) else mean > 0.0


def _run_l1(
    corpus: Corpus,
    config: TargetAuditConfig,
    scales: Mapping[str, float],
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + config.l1_max_seconds
    parent_ids = _stratified_parent_ids(
        corpus,
        "training",
        per_family=config.l1_parents_per_family,
        seed=config.random_seed + 101,
    )
    examples = _load_audit_examples(
        corpus,
        parent_ids,
        central_trace_count=config.central_trace_count,
    )
    data = _pack_audit_examples(examples)
    targets: dict[str, Any] = {}
    for target_index, target_name in enumerate(_AUDIT_TARGETS):
        _, training = _train_probe(
            data,
            target_name=target_name,
            input_mode="full",
            scales=scales,
            config=config,
            device=device,
            steps=config.l1_steps,
            seed=config.random_seed + 1000 + target_index,
            deadline=deadline,
        )
        training["passed"] = bool(
            not training["timed_out"]
            and training["loss_reduction_fraction"]
            >= config.l1_minimum_loss_reduction_fraction
        )
        targets[target_name] = training
    elapsed = time.perf_counter() - started
    failures = [
        f"{target}:tiny_overfit"
        for target, result in targets.items()
        if not result["passed"]
    ]
    if elapsed > config.l1_max_seconds:
        failures.append("budget:l1_exceeded")
    report = {
        "status": "passed" if not failures else "failed",
        "elapsed_seconds": elapsed,
        "parent_ids": list(parent_ids),
        "example_count": len(examples),
        "targets": targets,
        "failures": failures,
    }
    logger.info(
        "target audit L1 complete | status=%s | parents=%d | examples=%d | "
        "elapsed=%.1fs",
        report["status"],
        len(parent_ids),
        len(examples),
        elapsed,
    )
    return report


def _baseline_prediction(
    data: Mapping[str, Any],
    target_name: str,
) -> np.ndarray:
    if target_name == "projected_log_ai_increment":
        return np.asarray(data["lfm_residual"], dtype=np.float32)
    if target_name == "signed_reflectivity":
        return np.zeros_like(np.asarray(data[target_name]), dtype=np.float32)
    state = np.asarray(data[target_name], dtype=np.int64)
    support = np.asarray(data["support"], dtype=bool)
    counts = np.bincount(state[support], minlength=3)
    logits = np.zeros(state.shape + (3,), dtype=np.float32)
    logits[..., int(np.argmax(counts))] = 1.0
    return logits


def _run_l2(
    corpus: Corpus,
    config: TargetAuditConfig,
    scales: Mapping[str, float],
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + config.l2_max_seconds
    training_parent_ids = _stratified_parent_ids(
        corpus,
        "training",
        per_family=config.l2_training_parents_per_family,
        seed=config.random_seed + 202,
    )
    tuning_parent_ids = _stratified_parent_ids(
        corpus,
        "tuning",
        per_family=config.l2_tuning_parents_per_family,
        seed=config.random_seed + 303,
    )
    training_examples = _load_audit_examples(
        corpus,
        training_parent_ids,
        central_trace_count=config.central_trace_count,
    )
    tuning_examples = _load_audit_examples(
        corpus,
        tuning_parent_ids,
        central_trace_count=config.central_trace_count,
    )
    training_data = _pack_audit_examples(training_examples)
    tuning_data = _pack_audit_examples(
        tuning_examples,
        include_matched_shuffle=True,
    )
    steps = max(
        config.l1_steps,
        config.l2_epochs
        * int(np.ceil(len(training_examples) / config.batch_size)),
    )
    target_reports: dict[str, Any] = {}
    for target_index, target_name in enumerate(_AUDIT_TARGETS):
        model_seed = config.random_seed + 2000 + target_index
        full_model, full_training = _train_probe(
            training_data,
            target_name=target_name,
            input_mode="full",
            scales=scales,
            config=config,
            device=device,
            steps=steps,
            seed=model_seed,
            deadline=deadline,
        )
        no_model, no_training = _train_probe(
            training_data,
            target_name=target_name,
            input_mode="no_seismic",
            scales=scales,
            config=config,
            device=device,
            steps=steps,
            seed=model_seed,
            deadline=deadline,
        )
        full_prediction = _predict_probe(
            full_model,
            tuning_data,
            input_mode="full",
            scales=scales,
            device=device,
            batch_size=config.batch_size,
        )
        no_prediction = _predict_probe(
            no_model,
            tuning_data,
            input_mode="no_seismic",
            scales=scales,
            device=device,
            batch_size=config.batch_size,
        )
        shuffled_prediction = _predict_probe(
            full_model,
            tuning_data,
            input_mode="full",
            scales=scales,
            device=device,
            batch_size=config.batch_size,
            shuffled_seismic=True,
        )
        baseline_prediction = _baseline_prediction(tuning_data, target_name)
        metrics: dict[str, Any] = {}
        parent_metrics: dict[str, Any] = {}
        for name, prediction in (
            ("full", full_prediction),
            ("no_seismic", no_prediction),
            ("matched_shuffle", shuffled_prediction),
            ("baseline", baseline_prediction),
        ):
            metrics[name], parent_metrics[name] = _probe_metrics_by_parent(
                prediction,
                tuning_data,
                target_name=target_name,
            )
        metric = "balanced_accuracy" if target_name == "state_emission" else "rmse"
        lower_is_better = target_name != "state_emission"
        full_vs_no = _probe_bootstrap(
            parent_metrics["full"],
            parent_metrics["no_seismic"],
            metric=metric,
            lower_is_better=lower_is_better,
            seed=config.random_seed + 3000 + 2 * target_index,
            replicates=config.bootstrap_replicates,
        )
        full_vs_shuffle = _probe_bootstrap(
            parent_metrics["full"],
            parent_metrics["matched_shuffle"],
            metric=metric,
            lower_is_better=lower_is_better,
            seed=config.random_seed + 3001 + 2 * target_index,
            replicates=config.bootstrap_replicates,
        )
        parent_family: dict[str, str] = {}
        for example in tuning_examples:
            previous = parent_family.setdefault(
                example.parent_id, example.geometry_family
            )
            if previous != example.geometry_family:
                raise ValueError("one audit parent has multiple geometry families.")
        family_comparisons: dict[str, Any] = {}
        family_direction_passed = True
        for family in ("none", "pinchout", "wedge"):
            family_ids = {
                parent_id
                for parent_id, value in parent_family.items()
                if value == family
            }
            family_full = {
                parent_id: parent_metrics["full"][parent_id]
                for parent_id in family_ids
            }
            family_no = {
                parent_id: parent_metrics["no_seismic"][parent_id]
                for parent_id in family_ids
            }
            family_shuffle = {
                parent_id: parent_metrics["matched_shuffle"][parent_id]
                for parent_id in family_ids
            }
            comparison_no = _probe_bootstrap(
                family_full,
                family_no,
                metric=metric,
                lower_is_better=lower_is_better,
                seed=config.random_seed + 4000 + 10 * target_index,
                replicates=config.bootstrap_replicates,
            )
            comparison_shuffle = _probe_bootstrap(
                family_full,
                family_shuffle,
                metric=metric,
                lower_is_better=lower_is_better,
                seed=config.random_seed + 4001 + 10 * target_index,
                replicates=config.bootstrap_replicates,
            )
            family_comparisons[family] = {
                "full_vs_no_seismic": comparison_no,
                "full_vs_matched_shuffle": comparison_shuffle,
            }
            family_direction_passed &= _comparison_has_correct_mean(comparison_no)
            family_direction_passed &= _comparison_has_correct_mean(
                comparison_shuffle
            )
        passed = bool(
            not full_training["timed_out"]
            and not no_training["timed_out"]
            and full_vs_no["passed"]
            and full_vs_shuffle["passed"]
            and family_direction_passed
        )
        parent_rows = {
            parent_id: {
                "geometry_family": parent_family[parent_id],
                "full": parent_metrics["full"][parent_id],
                "no_seismic": parent_metrics["no_seismic"][parent_id],
                "matched_shuffle": parent_metrics["matched_shuffle"][parent_id],
                "baseline": parent_metrics["baseline"][parent_id],
            }
            for parent_id in sorted(parent_family)
        }
        target_reports[target_name] = {
            "status": "passed" if passed else "failed",
            "training": {
                "full": full_training,
                "no_seismic": no_training,
            },
            "metrics": metrics,
            "paired_comparisons": {
                "full_vs_no_seismic": full_vs_no,
                "full_vs_matched_shuffle": full_vs_shuffle,
            },
            "geometry_family_comparisons": family_comparisons,
            "family_mean_direction_passed": family_direction_passed,
            "parents": parent_rows,
        }
    elapsed = time.perf_counter() - started
    failures = [
        f"{target}:information_probe"
        for target, result in target_reports.items()
        if result["status"] != "passed"
    ]
    if elapsed > config.l2_max_seconds:
        failures.append("budget:l2_exceeded")
    report = {
        "status": "passed" if not failures else "failed",
        "elapsed_seconds": elapsed,
        "training_parent_ids": list(training_parent_ids),
        "tuning_parent_ids": list(tuning_parent_ids),
        "training_example_count": len(training_examples),
        "tuning_example_count": len(tuning_examples),
        "training_steps_per_model": steps,
        "matched_shuffle_diagnostics": tuning_data[
            "matched_shuffle_diagnostics"
        ],
        "targets": target_reports,
        "failures": failures,
    }
    logger.info(
        "target audit L2 complete | status=%s | train_parents=%d | "
        "tuning_parents=%d | elapsed=%.1fs",
        report["status"],
        len(training_parent_ids),
        len(tuning_parent_ids),
        elapsed,
    )
    return report


def audit_observable_targets(
    corpus: Corpus,
    output_dir: str | Path,
    *,
    config: TargetAuditConfig = TargetAuditConfig(),
    device: str | torch.device = "cpu",
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run and publish the bounded L0-L2 observable-target audit.

    A formal run stops after a failed level.  Smoke mode executes every seam
    with tiny budgets, records scientific gate failures, and never publishes a
    target contract.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log = logger or logging.getLogger(__name__)
    resolved_device = torch.device(device)
    report: dict[str, Any] = {
        "schema": "structured_ginn_v2_observable_target_audit_v1",
        "status": "running",
        "mode": "smoke" if config.smoke else "formal",
        "corpus": {
            "root": str(corpus.root),
            "manifest_schema": str(corpus.manifest.get("schema_version") or ""),
            "sample_domain": corpus.benchmark.sample_domain,
            "sample_unit": corpus.benchmark.sample_unit,
            "depth_basis": corpus.benchmark.depth_basis,
        },
        "config": asdict(config),
        "levels": {},
    }

    def publish() -> None:
        write_json(output / "target_audit.json", report)

    publish()
    l0, scales = _run_l0(corpus, config, output, log)
    report["levels"]["l0"] = l0
    if l0["status"] != "passed" and not config.smoke:
        report["status"] = "failed_l0"
        publish()
        return report
    publish()

    l1 = _run_l1(corpus, config, scales, resolved_device, log)
    report["levels"]["l1"] = l1
    if l1["status"] != "passed" and not config.smoke:
        report["status"] = "failed_l1"
        publish()
        return report
    publish()

    l2 = _run_l2(corpus, config, scales, resolved_device, log)
    report["levels"]["l2"] = l2
    if config.smoke:
        report["status"] = "smoke_completed"
        publish()
        return report
    if l2["status"] != "passed":
        report["status"] = "failed_l2"
        publish()
        return report

    contract = {
        "schema": "structured_ginn_v2_observable_target_contract_v1",
        "status": "passed",
        "sample_domain": corpus.benchmark.sample_domain,
        "sample_unit": corpus.benchmark.sample_unit,
        "depth_basis": corpus.benchmark.depth_basis,
        "targets": list(_AUDIT_TARGETS),
        "global_scales": scales,
        "audit_report": "target_audit.json",
        "l2_training_parent_ids": l2["training_parent_ids"],
        "l2_tuning_parent_ids": l2["tuning_parent_ids"],
    }
    write_json(output / "observable_target_contract.json", contract)
    report["status"] = "passed"
    report["target_contract"] = "observable_target_contract.json"
    publish()
    return report


def _to_device(
    batch: Mapping[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    boolean = {"observed_valid", "lateral_valid", "support"}
    for key, value in batch.items():
        if key not in _TENSOR_BATCH_KEYS:
            continue
        result[key] = torch.as_tensor(
            value,
            dtype=(
                torch.bool
                if key in boolean
                else torch.long if key == "state_emission" else torch.float32
            ),
            device=device,
        )
    return result


def _forward_loss(
    generator: ConditionalGenerator,
    batch: Mapping[str, torch.Tensor],
    config: LearningConfig,
) -> dict[str, torch.Tensor]:
    output = generator.network(
        batch["seismic"],
        batch["lfm_residual"],
        batch["observed_valid"],
        batch["lateral_m"],
        batch["lateral_valid"],
    )
    return evidence_loss(
        output,
        batch,
        config=generator.network_config,
        increment_weight=config.increment_weight,
        reflectivity_weight=config.reflectivity_weight,
        state_weight=config.state_weight,
        scale_weight=config.scale_weight,
    )


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["torch_cuda"]]
        )


def _evaluate_losses(
    generator: ConditionalGenerator,
    batches: Iterable[Mapping[str, object]],
    config: LearningConfig,
) -> dict[str, float]:
    generator.network.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for source in batches:
            loss = _forward_loss(
                generator,
                _to_device(source, generator.device),
                config,
            )
            count += 1
            for key, value in loss.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    if count == 0:
        raise ValueError("evaluation received no batches.")
    return {key: value / count for key, value in totals.items()}


def _new_statistics() -> dict[str, Any]:
    return {
        "count": 0,
        "increment_squared_error": 0.0,
        "increment_absolute_error": 0.0,
        "increment_scale": 0.0,
        "increment_coverage_50": 0.0,
        "increment_coverage_80": 0.0,
        "increment_coverage_95": 0.0,
        "increment_pair": np.zeros(5, dtype=np.float64),
        "reflectivity_squared_error": 0.0,
        "reflectivity_absolute_error": 0.0,
        "reflectivity_scale": 0.0,
        "reflectivity_coverage_50": 0.0,
        "reflectivity_coverage_80": 0.0,
        "reflectivity_coverage_95": 0.0,
        "reflectivity_pair": np.zeros(5, dtype=np.float64),
        "reflectivity_polarity_count": 0,
        "reflectivity_polarity_correct": 0,
        "state_cross_entropy": 0.0,
        "state_brier": 0.0,
        "state_correct": 0,
        "state_class_count": np.zeros(3, dtype=np.int64),
        "state_class_correct": np.zeros(3, dtype=np.int64),
    }


def _new_increment_statistics() -> dict[str, Any]:
    return {
        "count": 0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
        "pair": np.zeros(5, dtype=np.float64),
    }


def _update_pair(
    moments: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    moments += np.asarray(
        (
            np.sum(prediction),
            np.sum(target),
            np.sum(prediction * prediction),
            np.sum(target * target),
            np.sum(prediction * target),
        ),
        dtype=np.float64,
    )


def _correlation(moments: np.ndarray, count: int) -> float | None:
    if count < 2:
        return None
    sum_x, sum_y, sum_x2, sum_y2, sum_xy = moments
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_x2 - sum_x * sum_x / count
    variance_y = sum_y2 - sum_y * sum_y / count
    denominator = float(np.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0)))
    if denominator <= 0.0:
        return None
    return float(covariance / denominator)


def _update_statistics(
    statistics: dict[str, Any],
    output: Mapping[str, np.ndarray],
    source: Mapping[str, object],
) -> None:
    support = np.asarray(source["support"], dtype=bool) & np.asarray(
        output["support"], dtype=bool
    )
    count = int(np.count_nonzero(support))
    if count == 0:
        raise ValueError("formal evidence evaluation has no supported samples.")
    target_increment = np.asarray(
        source["projected_log_ai_increment"], dtype=np.float64
    )[support]
    prediction_increment = np.asarray(
        output["projected_log_ai_increment_mean"], dtype=np.float64
    )[support]
    increment_scale = np.asarray(
        output["projected_log_ai_increment_scale"], dtype=np.float64
    )[support]
    residual = prediction_increment - target_increment

    statistics["count"] += count
    statistics["increment_squared_error"] += float(np.sum(residual * residual))
    statistics["increment_absolute_error"] += float(np.sum(np.abs(residual)))
    statistics["increment_scale"] += float(np.sum(increment_scale))
    for name, quantile in (
        ("increment_coverage_50", 0.6744897501960817),
        ("increment_coverage_80", 1.2815515655446004),
        ("increment_coverage_95", 1.959963984540054),
    ):
        statistics[name] += float(
            np.count_nonzero(np.abs(residual) <= quantile * increment_scale)
        )
    _update_pair(
        statistics["increment_pair"], prediction_increment, target_increment
    )

    target_reflectivity = np.asarray(
        source["signed_reflectivity"], dtype=np.float64
    )[support]
    prediction_reflectivity = np.asarray(
        output["signed_reflectivity_mean"], dtype=np.float64
    )[support]
    reflectivity_scale = np.asarray(
        output["signed_reflectivity_scale"], dtype=np.float64
    )[support]
    reflectivity_residual = prediction_reflectivity - target_reflectivity
    statistics["reflectivity_squared_error"] += float(
        np.sum(reflectivity_residual**2)
    )
    statistics["reflectivity_absolute_error"] += float(
        np.sum(np.abs(reflectivity_residual))
    )
    statistics["reflectivity_scale"] += float(np.sum(reflectivity_scale))
    for name, quantile in (
        ("reflectivity_coverage_50", 0.6744897501960817),
        ("reflectivity_coverage_80", 1.2815515655446004),
        ("reflectivity_coverage_95", 1.959963984540054),
    ):
        statistics[name] += float(
            np.count_nonzero(
                np.abs(reflectivity_residual) <= quantile * reflectivity_scale
            )
        )
    _update_pair(
        statistics["reflectivity_pair"],
        prediction_reflectivity,
        target_reflectivity,
    )
    nonzero = np.abs(target_reflectivity) > 1.0e-8
    statistics["reflectivity_polarity_count"] += int(np.count_nonzero(nonzero))
    statistics["reflectivity_polarity_correct"] += int(
        np.count_nonzero(
            np.sign(prediction_reflectivity[nonzero])
            == np.sign(target_reflectivity[nonzero])
        )
    )

    target_state = np.asarray(source["state_emission"], dtype=np.int64)[support]
    state_log_potential = np.asarray(
        output["state_log_potential"], dtype=np.float64
    )[support]
    state_probability = np.exp(state_log_potential)
    prediction_state = np.argmax(state_log_potential, axis=-1)
    statistics["state_cross_entropy"] += float(
        np.sum(-state_log_potential[np.arange(target_state.size), target_state])
    )
    one_hot = np.eye(3, dtype=np.float64)[target_state]
    statistics["state_brier"] += float(
        np.sum((state_probability - one_hot) ** 2)
    )
    statistics["state_correct"] += int(
        np.count_nonzero(prediction_state == target_state)
    )
    statistics["state_class_count"] += np.bincount(target_state, minlength=3)
    statistics["state_class_correct"] += np.bincount(
        target_state[prediction_state == target_state], minlength=3
    )


def _update_increment_statistics(
    statistics: dict[str, Any],
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
) -> None:
    predicted = np.asarray(prediction, dtype=np.float64)[support]
    expected = np.asarray(target, dtype=np.float64)[support]
    residual = predicted - expected
    statistics["count"] += int(expected.size)
    statistics["squared_error"] += float(np.sum(residual * residual))
    statistics["absolute_error"] += float(np.sum(np.abs(residual)))
    _update_pair(statistics["pair"], predicted, expected)


def _finalize_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    count = int(statistics["count"])
    if count <= 0:
        raise ValueError("cannot finalize empty evidence statistics.")
    state_count = np.asarray(statistics["state_class_count"], dtype=np.float64)
    state_correct = np.asarray(
        statistics["state_class_correct"], dtype=np.float64
    )
    represented = state_count > 0.0
    balanced_accuracy = float(
        np.mean(state_correct[represented] / state_count[represented])
    )
    polarity_count = int(statistics["reflectivity_polarity_count"])
    return {
        "supported_samples": count,
        "increment_rmse": float(
            np.sqrt(float(statistics["increment_squared_error"]) / count)
        ),
        "increment_mae": float(statistics["increment_absolute_error"]) / count,
        "increment_correlation": _correlation(
            np.asarray(statistics["increment_pair"]), count
        ),
        "increment_mean_scale": float(statistics["increment_scale"]) / count,
        "increment_coverage_50": float(statistics["increment_coverage_50"])
        / count,
        "increment_coverage_80": float(statistics["increment_coverage_80"])
        / count,
        "increment_coverage_95": float(statistics["increment_coverage_95"])
        / count,
        "reflectivity_rmse": float(
            np.sqrt(float(statistics["reflectivity_squared_error"]) / count)
        ),
        "reflectivity_mae": float(statistics["reflectivity_absolute_error"])
        / count,
        "reflectivity_correlation": _correlation(
            np.asarray(statistics["reflectivity_pair"]), count
        ),
        "reflectivity_mean_scale": float(statistics["reflectivity_scale"])
        / count,
        "reflectivity_coverage_50": float(
            statistics["reflectivity_coverage_50"]
        )
        / count,
        "reflectivity_coverage_80": float(
            statistics["reflectivity_coverage_80"]
        )
        / count,
        "reflectivity_coverage_95": float(
            statistics["reflectivity_coverage_95"]
        )
        / count,
        "reflectivity_polarity_accuracy": (
            float(statistics["reflectivity_polarity_correct"]) / polarity_count
            if polarity_count > 0
            else None
        ),
        "state_cross_entropy": float(statistics["state_cross_entropy"]) / count,
        "state_brier": float(statistics["state_brier"]) / count,
        "state_accuracy": float(statistics["state_correct"]) / count,
        "state_balanced_accuracy": balanced_accuracy,
        "state_class_count": [int(value) for value in state_count],
    }


def _finalize_increment_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    count = int(statistics["count"])
    if count <= 0:
        raise ValueError("cannot finalize empty increment baseline statistics.")
    return {
        "supported_samples": count,
        "increment_rmse": float(
            np.sqrt(float(statistics["squared_error"]) / count)
        ),
        "increment_mae": float(statistics["absolute_error"]) / count,
        "increment_correlation": _correlation(
            np.asarray(statistics["pair"]), count
        ),
    }


def _network_output(
    generator: ConditionalGenerator,
    source: Mapping[str, object],
) -> dict[str, np.ndarray]:
    batch = _to_device(source, generator.device)
    output = generator.network(
        batch["seismic"],
        batch["lfm_residual"],
        batch["observed_valid"],
        batch["lateral_m"],
        batch["lateral_valid"],
    )
    return {
        key: value.detach().cpu().numpy()
        for key, value in output.items()
    }


def _matched_shuffle_batch(
    source: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, float]]:
    seismic = np.asarray(source["seismic"], dtype=np.float32)
    observed = np.asarray(source["observed_valid"], dtype=bool)
    lateral_valid = np.asarray(source["lateral_valid"], dtype=bool)
    if seismic.ndim != 3 or observed.shape != seismic.shape:
        raise ValueError("matched shuffle requires [batch, lateral, sample] arrays.")
    if seismic.shape[0] != 1 or lateral_valid.shape != seismic.shape[:2]:
        raise ValueError("matched shuffle expects one parent-zone per batch.")
    valid_indices = np.flatnonzero(lateral_valid[0]).tolist()
    if len(valid_indices) < 2:
        raise ValueError("matched shuffle requires at least two valid lateral traces.")
    donors = valid_indices[1:] + valid_indices[:1]
    shuffled = seismic.copy()
    recipient_count = 0
    common_count = 0
    changed_count = 0
    squared_change = 0.0
    for recipient, donor in zip(valid_indices, donors, strict=True):
        common = observed[0, recipient] & observed[0, donor]
        recipient_count += int(np.count_nonzero(observed[0, recipient]))
        common_count += int(np.count_nonzero(common))
        difference = seismic[0, donor, common] - seismic[0, recipient, common]
        changed_count += int(np.count_nonzero(difference != 0.0))
        squared_change += float(np.sum(difference.astype(np.float64) ** 2))
        shuffled[0, recipient, common] = seismic[0, donor, common]
    if common_count == 0:
        raise ValueError("matched shuffle has no common observation support.")
    result = dict(source)
    result["seismic"] = shuffled
    return result, {
        "batch_count": 1.0,
        "recipient_valid_samples": float(recipient_count),
        "common_valid_samples": float(common_count),
        "changed_valid_samples": float(changed_count),
        "squared_change": squared_change,
    }


def _bootstrap_delta(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    if values.size == 0:
        raise ValueError("paired comparison has no parent values.")
    rng = np.random.default_rng(seed)
    if values.size == 1:
        interval = (float(values[0]), float(values[0]))
    else:
        draws = rng.integers(0, values.size, size=(replicates, values.size))
        means = np.mean(values[draws], axis=1)
        interval = tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))
    return {
        "parent_count": int(values.size),
        "delta_definition": "left_minus_right",
        "lower_is_better": bool(lower_is_better),
        "mean_delta": float(np.mean(values)),
        "bootstrap_95_low": interval[0],
        "bootstrap_95_high": interval[1],
    }


def evaluate_generator(
    generator: ConditionalGenerator,
    batches: Iterable[Mapping[str, object]],
    *,
    controls: Mapping[str, ConditionalGenerator] | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260801,
) -> dict[str, Any]:
    """Evaluate evidence heads, simple LFM baselines, and optional controls.

    Every parent is the statistical unit for paired comparisons.  The same
    interface serves clean and dirty conditions; callers supply the matching
    batch iterator rather than invoking a separate diagnostics pipeline.
    """
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive.")
    if generator.network_config.input_mode != "full":
        raise ValueError("formal evaluation requires a full primary generator.")
    models = {"full": generator, **dict(controls or {})}
    if len(models) != 1 + len(dict(controls or {})):
        raise ValueError("control name collides with the full model name.")
    for name, item in models.items():
        if item.network_config.input_mode != name:
            raise ValueError(
                f"evaluation model {name!r} has input_mode="
                f"{item.network_config.input_mode!r}."
            )
        comparison_config = asdict(item.network_config)
        comparison_config.pop("input_mode")
        full_config = asdict(generator.network_config)
        full_config.pop("input_mode")
        if comparison_config != full_config:
            raise ValueError(
                f"evaluation model {name!r} does not share the full model's "
                "network and input-normalization contract."
            )
    domains = {item.sample_domain for item in models.values()}
    if len(domains) != 1:
        raise ValueError("all evaluated checkpoints must share one sample domain.")
    contracts = {
        str(item.target_contract.to_mapping()) for item in models.values()
    }
    if len(contracts) != 1:
        raise ValueError("all evaluated checkpoints must share one target contract.")
    for item in models.values():
        item.network.eval()

    evaluated_names = tuple(models) + ("matched_shuffle",)
    overall_models = {name: _new_statistics() for name in evaluated_names}
    overall_baselines = {
        "zone_linear_anchor_only": _new_increment_statistics(),
        "full_lfm_only": _new_increment_statistics(),
    }
    parent_models: dict[str, dict[str, dict[str, Any]]] = {}
    parent_baselines: dict[str, dict[str, dict[str, Any]]] = {}
    shuffle_diagnostics = {
        "batch_count": 0.0,
        "recipient_valid_samples": 0.0,
        "common_valid_samples": 0.0,
        "changed_valid_samples": 0.0,
        "squared_change": 0.0,
    }
    batch_count = 0
    with torch.no_grad():
        for source in batches:
            batch_count += 1
            parent_id = source.get("parent_id")
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError("formal evaluation batch requires parent_id.")
            if "background_lfm_linear" not in source:
                raise ValueError(
                    "formal evaluation requires background_lfm_linear for baselines."
                )
            parent_model = parent_models.setdefault(
                parent_id, {name: _new_statistics() for name in evaluated_names}
            )
            parent_baseline = parent_baselines.setdefault(
                parent_id,
                {
                    "zone_linear_anchor_only": _new_increment_statistics(),
                    "full_lfm_only": _new_increment_statistics(),
                },
            )
            for name, model in models.items():
                output = _network_output(model, source)
                _update_statistics(overall_models[name], output, source)
                _update_statistics(parent_model[name], output, source)
            shuffled_source, shuffle = _matched_shuffle_batch(source)
            for key, value in shuffle.items():
                shuffle_diagnostics[key] += value
            shuffled_output = _network_output(generator, shuffled_source)
            _update_statistics(
                overall_models["matched_shuffle"],
                shuffled_output,
                source,
            )
            _update_statistics(
                parent_model["matched_shuffle"],
                shuffled_output,
                source,
            )

            support = np.asarray(source["support"], dtype=bool)
            target = np.asarray(
                source["projected_log_ai_increment"], dtype=np.float64
            )
            anchor_prediction = np.zeros_like(target)
            full_lfm_prediction = (
                np.asarray(source["lfm"], dtype=np.float64)
                - np.asarray(source["background_lfm_linear"], dtype=np.float64)
            )
            for name, prediction in (
                ("zone_linear_anchor_only", anchor_prediction),
                ("full_lfm_only", full_lfm_prediction),
            ):
                _update_increment_statistics(
                    overall_baselines[name], prediction, target, support
                )
                _update_increment_statistics(
                    parent_baseline[name], prediction, target, support
                )
    if batch_count == 0:
        raise ValueError("evaluation received no batches.")

    parent_rows = []
    for parent_id in sorted(parent_models):
        parent_rows.append(
            {
                "parent_id": parent_id,
                "models": {
                    name: _finalize_statistics(statistics)
                    for name, statistics in parent_models[parent_id].items()
                },
                "baselines": {
                    name: _finalize_increment_statistics(statistics)
                    for name, statistics in parent_baselines[parent_id].items()
                },
            }
        )

    comparisons: dict[str, Any] = {}
    comparison_index = 0
    for right in evaluated_names:
        if right == "full":
            continue
        for metric in (
            "increment_rmse",
            "reflectivity_rmse",
            "state_brier",
            "state_balanced_accuracy",
        ):
            values = np.asarray(
                [
                    row["models"]["full"][metric]
                    - row["models"][right][metric]
                    for row in parent_rows
                ],
                dtype=np.float64,
            )
            key = f"full_vs_{right}:{metric}"
            comparisons[key] = _bootstrap_delta(
                values,
                seed=bootstrap_seed + comparison_index,
                replicates=bootstrap_replicates,
                lower_is_better=metric != "state_balanced_accuracy",
            )
            comparison_index += 1
    for right in overall_baselines:
        values = np.asarray(
            [
                row["models"]["full"]["increment_rmse"]
                - row["baselines"][right]["increment_rmse"]
                for row in parent_rows
            ],
            dtype=np.float64,
        )
        key = f"full_vs_{right}:increment_rmse"
        comparisons[key] = _bootstrap_delta(
            values,
            seed=bootstrap_seed + comparison_index,
            replicates=bootstrap_replicates,
        )
        comparison_index += 1

    return {
        "batch_count": batch_count,
        "parent_count": len(parent_rows),
        "models": {
            name: {
                "input_mode": (
                    models[name].network_config.input_mode
                    if name in models
                    else "matched_within_parent_seismic_shuffle"
                ),
                "metrics": _finalize_statistics(overall_models[name]),
            }
            for name in evaluated_names
        },
        "baselines": {
            name: _finalize_increment_statistics(statistics)
            for name, statistics in overall_baselines.items()
        },
        "paired_comparisons": comparisons,
        "matched_shuffle_diagnostics": {
            "batch_count": int(shuffle_diagnostics["batch_count"]),
            "common_valid_fraction": shuffle_diagnostics[
                "common_valid_samples"
            ]
            / max(shuffle_diagnostics["recipient_valid_samples"], 1.0),
            "changed_common_fraction": shuffle_diagnostics[
                "changed_valid_samples"
            ]
            / max(shuffle_diagnostics["common_valid_samples"], 1.0),
            "change_rms": float(
                np.sqrt(
                    shuffle_diagnostics["squared_change"]
                    / max(shuffle_diagnostics["common_valid_samples"], 1.0)
                )
            ),
        },
        "parents": parent_rows,
    }


def _contiguous_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple((int(start), int(stop)) for start, stop in zip(starts, stops))


def _truth_path(state: np.ndarray, object_id: np.ndarray) -> SampledPath:
    state_values = np.asarray(state, dtype=np.int64).reshape(-1)
    object_values = np.asarray(object_id, dtype=np.int64).reshape(-1)
    if state_values.shape != object_values.shape or state_values.size == 0:
        raise ValueError("truth state and object identity must share a non-empty shape.")
    if np.any((state_values < 0) | (state_values > 2)) or np.any(object_values < 0):
        raise ValueError("truth HSMM audit path contains invalid state/object identity.")
    object_starts = np.r_[
        0,
        1 + np.flatnonzero(object_values[1:] != object_values[:-1]),
    ]
    object_stops = np.r_[object_starts[1:], state_values.size]
    for start, stop in zip(object_starts, object_stops, strict=True):
        if np.any(state_values[start:stop] != state_values[start]):
            raise ValueError("one truth object contains more than one state.")

    starts = object_starts
    stops = object_stops
    segments: list[tuple[int, int, int]] = []
    for start, stop in zip(starts, stops, strict=True):
        local = state_values[start:stop]
        segments.append((int(local[0]), int(start), int(stop)))
    return SampledPath(
        segments=tuple(segments),
        state=state_values.astype(np.int8),
        log_score=float("nan"),
    )


def _soft_truth_state(state: np.ndarray, *, confidence: float) -> np.ndarray:
    values = np.asarray(state, dtype=np.int64).reshape(-1)
    if not 1.0 / 3.0 < confidence < 1.0:
        raise ValueError("truth state confidence must lie between 1/3 and 1.")
    probability = np.full(
        (values.size, 3),
        (1.0 - confidence) / 2.0,
        dtype=np.float64,
    )
    probability[np.arange(values.size), values] = confidence
    return probability


def _new_hsmm_path_statistics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "renewal_sample_count": 0,
        "trace_count": 0,
        "state_cross_entropy": 0.0,
        "state_brier": 0.0,
        "state_correct": 0,
        "state_class_count": np.zeros(3, dtype=np.int64),
        "state_class_correct": np.zeros(3, dtype=np.int64),
        "renewal_brier": 0.0,
        "truth_segment_count": 0.0,
        "map_segment_count": 0.0,
        "expected_segment_count": 0.0,
        "truth_duration_fraction": [],
        "map_duration_fraction": [],
    }


def _update_hsmm_path_statistics(
    statistics: dict[str, Any],
    *,
    truth: SampledPath,
    path: SampledPath,
    state_probability: np.ndarray,
    renewal_probability: np.ndarray,
    expected_segment_count: float,
) -> None:
    samples = int(truth.state.size)
    if path.state.shape != truth.state.shape or state_probability.shape != (samples, 3):
        raise ValueError("HSMM path and marginal shapes differ from truth.")
    target = truth.state.astype(np.int64)
    probability = np.asarray(state_probability, dtype=np.float64)
    one_hot = np.eye(3, dtype=np.float64)[target]
    statistics["sample_count"] += samples
    statistics["trace_count"] += 1
    statistics["state_cross_entropy"] += float(
        -np.sum(np.log(np.clip(probability[np.arange(samples), target], 1.0e-12, 1.0)))
    )
    statistics["state_brier"] += float(np.sum((probability - one_hot) ** 2))
    statistics["state_correct"] += int(np.count_nonzero(path.state == target))
    for state in range(3):
        selected = target == state
        statistics["state_class_count"][state] += int(np.count_nonzero(selected))
        statistics["state_class_correct"][state] += int(
            np.count_nonzero(path.state[selected] == state)
        )

    truth_renewal = np.zeros(samples, dtype=np.float64)
    truth_renewal[[segment[1] for segment in truth.segments[1:]]] = 1.0
    if samples > 1:
        difference = np.asarray(renewal_probability, dtype=np.float64)[1:] - truth_renewal[1:]
        statistics["renewal_brier"] += float(np.sum(difference * difference))
        statistics["renewal_sample_count"] += samples - 1
    statistics["truth_segment_count"] += len(truth.segments)
    statistics["map_segment_count"] += len(path.segments)
    statistics["expected_segment_count"] += float(expected_segment_count)
    statistics["truth_duration_fraction"].extend(
        (stop - start) / samples for _, start, stop in truth.segments
    )
    statistics["map_duration_fraction"].extend(
        (stop - start) / samples for _, start, stop in path.segments
    )


def _finalize_hsmm_path_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    samples = int(statistics["sample_count"])
    traces = int(statistics["trace_count"])
    if samples <= 0 or traces <= 0:
        raise ValueError("HSMM path evaluation has no supported traces.")
    class_count = np.asarray(statistics["state_class_count"], dtype=np.int64)
    class_correct = np.asarray(statistics["state_class_correct"], dtype=np.int64)
    valid_class = class_count > 0
    balanced = float(np.mean(class_correct[valid_class] / class_count[valid_class]))
    truth_duration = np.asarray(statistics["truth_duration_fraction"], dtype=np.float64)
    map_duration = np.asarray(statistics["map_duration_fraction"], dtype=np.float64)
    truth_count_mean = float(statistics["truth_segment_count"] / traces)
    map_count_mean = float(statistics["map_segment_count"] / traces)
    expected_count_mean = float(statistics["expected_segment_count"] / traces)
    return {
        "supported_samples": samples,
        "trace_count": traces,
        "state_cross_entropy": float(statistics["state_cross_entropy"] / samples),
        "state_brier": float(statistics["state_brier"] / samples),
        "state_accuracy": float(statistics["state_correct"] / samples),
        "state_balanced_accuracy": balanced,
        "state_class_count": class_count.tolist(),
        "renewal_brier": float(
            statistics["renewal_brier"]
            / max(int(statistics["renewal_sample_count"]), 1)
        ),
        "truth_segment_count_mean": truth_count_mean,
        "map_segment_count_mean": map_count_mean,
        "map_segment_count_bias": map_count_mean - truth_count_mean,
        "posterior_expected_segment_count_mean": expected_count_mean,
        "posterior_expected_segment_count_bias": expected_count_mean - truth_count_mean,
        "truth_duration_fraction_mean": float(np.mean(truth_duration)),
        "truth_duration_fraction_std": float(np.std(truth_duration)),
        "map_duration_fraction_mean": float(np.mean(map_duration)),
        "map_duration_fraction_std": float(np.std(map_duration)),
        "duration_fraction_wasserstein": float(
            wasserstein_distance(truth_duration, map_duration)
        ),
    }


def _profile_fit_on_path(source: np.ndarray, path: SampledPath) -> np.ndarray:
    values = np.asarray(source, dtype=np.float64).reshape(-1)
    result = np.empty_like(values)
    for _, start, stop in path.segments:
        coefficients = fit_profile_coefficients(values[start:stop])
        result[start:stop] = profile_basis(stop - start) @ np.asarray(coefficients)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("HSMM profile diagnostic produced non-finite output.")
    return result


def _new_profile_statistics() -> dict[str, Any]:
    return {
        "count": 0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
        "pair": np.zeros(5, dtype=np.float64),
    }


def _update_profile_statistics(
    statistics: dict[str, Any],
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    if predicted.shape != truth.shape or np.any(~np.isfinite(predicted)) or np.any(~np.isfinite(truth)):
        raise ValueError("profile diagnostic requires finite arrays with one shape.")
    error = predicted - truth
    statistics["count"] += truth.size
    statistics["squared_error"] += float(np.sum(error * error))
    statistics["absolute_error"] += float(np.sum(np.abs(error)))
    _update_pair(statistics["pair"], predicted, truth)


def _finalize_profile_statistics(statistics: Mapping[str, Any]) -> dict[str, Any]:
    count = int(statistics["count"])
    if count <= 0:
        raise ValueError("profile evaluation has no supported samples.")
    return {
        "supported_samples": count,
        "increment_rmse": float(np.sqrt(statistics["squared_error"] / count)),
        "increment_mae": float(statistics["absolute_error"] / count),
        "increment_correlation": _correlation(statistics["pair"], count),
    }


@dataclass(frozen=True)
class _HsmmOracleCase:
    parent_id: str
    trace_index: int
    trace_count: int
    truth_path: SampledPath
    truth_probability: np.ndarray
    predicted_probability: np.ndarray
    renewal_evidence: np.ndarray
    truth_amplitude: np.ndarray
    predicted_amplitude: np.ndarray


def _collect_hsmm_oracle_cases(
    generator: ConditionalGenerator,
    batches: Iterable[Mapping[str, object]],
    *,
    truth_state_confidence: float,
    logger: logging.Logger,
    log_every_batches: int,
) -> tuple[tuple[_HsmmOracleCase, ...], int, int]:
    generator.network.eval()
    cases: list[_HsmmOracleCase] = []
    parent_ids: set[str] = set()
    batch_count = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_count, source in enumerate(batches, start=1):
            parent_id = source.get("parent_id")
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError("HSMM evaluation batch requires parent_id.")
            parent_ids.add(parent_id)
            if "truth_object_id" not in source:
                raise ValueError("HSMM evaluation requires model-grid truth_object_id.")
            output = _network_output(generator, source)
            support = np.asarray(source["support"], dtype=bool)
            truth_state = np.asarray(source["state_emission"], dtype=np.int64)
            truth_object = np.asarray(source["truth_object_id"], dtype=np.int64)
            truth_increment = np.asarray(
                source["projected_log_ai_increment"], dtype=np.float64
            )
            predicted_increment = np.asarray(
                output["projected_log_ai_increment_mean"], dtype=np.float64
            )
            predicted_state = np.exp(
                np.asarray(output["state_log_potential"], dtype=np.float64)
            )
            predicted_reflectivity = np.asarray(
                output["signed_reflectivity_mean"], dtype=np.float64
            )
            predicted_reflectivity_scale = np.asarray(
                output["signed_reflectivity_scale"], dtype=np.float64
            )
            if support.ndim != 3 or support.shape[0] != 1:
                raise ValueError("HSMM evaluation expects one parent-zone per batch.")
            for trace in range(support.shape[1]):
                for start, stop in _contiguous_runs(support[0, trace]):
                    local_truth_state = truth_state[0, trace, start:stop]
                    predicted_probability = np.clip(
                        predicted_state[0, trace, start:stop], 1.0e-8, 1.0
                    )
                    predicted_probability /= np.sum(
                        predicted_probability, axis=1, keepdims=True
                    )
                    cases.append(
                        _HsmmOracleCase(
                            parent_id=parent_id,
                            trace_index=trace,
                            trace_count=support.shape[1],
                            truth_path=_truth_path(
                                local_truth_state,
                                truth_object[0, trace, start:stop],
                            ),
                            truth_probability=_soft_truth_state(
                                local_truth_state,
                                confidence=truth_state_confidence,
                            ),
                            predicted_probability=predicted_probability,
                            renewal_evidence=renewal_probability_from_reflectivity(
                                predicted_reflectivity[0, trace, start:stop],
                                predicted_reflectivity_scale[0, trace, start:stop],
                                generator.decode_policy,
                            ),
                            truth_amplitude=truth_increment[0, trace, start:stop],
                            predicted_amplitude=(
                                predicted_increment[0, trace, start:stop]
                            ),
                        )
                    )
            if batch_count % log_every_batches == 0:
                logger.info(
                    "HSMM evidence cache | batches=%d | parents=%d | traces=%d | elapsed=%.1fs",
                    batch_count,
                    len(parent_ids),
                    len(cases),
                    time.perf_counter() - start_time,
                )
    if batch_count == 0 or not cases:
        raise ValueError("HSMM evaluation received no supported cases.")
    return tuple(cases), len(parent_ids), batch_count


def _direct_state_metrics(cases: tuple[_HsmmOracleCase, ...]) -> dict[str, Any]:
    sample_count = 0
    cross_entropy = 0.0
    brier = 0.0
    class_count = np.zeros(3, dtype=np.int64)
    class_correct = np.zeros(3, dtype=np.int64)
    for case in cases:
        target = case.truth_path.state.astype(np.int64)
        probability = case.predicted_probability
        prediction = np.argmax(probability, axis=1)
        sample_count += target.size
        cross_entropy += float(
            -np.sum(np.log(np.clip(probability[np.arange(target.size), target], 1.0e-12, 1.0)))
        )
        brier += float(np.sum((probability - np.eye(3)[target]) ** 2))
        for state in range(3):
            selected = target == state
            class_count[state] += int(np.count_nonzero(selected))
            class_correct[state] += int(np.count_nonzero(prediction[selected] == state))
    return {
        "supported_samples": sample_count,
        "state_cross_entropy": cross_entropy / sample_count,
        "state_brier": brier / sample_count,
        "state_accuracy": float(np.sum(class_correct) / sample_count),
        "state_balanced_accuracy": float(
            np.mean(class_correct[class_count > 0] / class_count[class_count > 0])
        ),
        "state_class_count": class_count.tolist(),
    }


def _central_hsmm_calibration_cases(
    cases: tuple[_HsmmOracleCase, ...],
) -> tuple[_HsmmOracleCase, ...]:
    """Use the five complete-context center traces for scalar selection."""

    selected = tuple(
        case
        for case in cases
        if (
            (case.trace_count - min(case.trace_count, 5)) // 2
            <= case.trace_index
            < (
                (case.trace_count - min(case.trace_count, 5)) // 2
                + min(case.trace_count, 5)
            )
        )
    )
    if not selected:
        raise ValueError("HSMM calibration has no center-trace cases.")
    return selected


def _evaluate_hsmm_cases(
    cases: tuple[_HsmmOracleCase, ...],
    *,
    parent_count: int,
    batch_count: int,
    prior: SemiMarkovPrior,
    conditioning: SemiMarkovConditioning,
    decode_policy: SemiMarkovDecodePolicy,
    truth_state_confidence: float,
) -> dict[str, Any]:
    state_statistics = {
        name: _new_hsmm_path_statistics()
        for name in ("truth_state", "predicted_state", "prior_only")
    }
    profile_statistics = {
        name: _new_profile_statistics()
        for name in (
            "truth_state_truth_amplitude",
            "truth_state_predicted_amplitude",
            "predicted_state_truth_amplitude",
            "predicted_state_predicted_amplitude",
            "prior_only",
        )
    }
    amplitude_source_statistics = {
        "predicted_amplitude": _new_profile_statistics(),
        "prior_only": _new_profile_statistics(),
    }
    decoding_prior = prior_with_same_state_renewal(
        prior,
        decode_policy.same_state_renewal_probability,
    )
    for case in cases:
        probability_by_condition = {
            "truth_state": case.truth_probability,
            "predicted_state": case.predicted_probability,
            "prior_only": np.full_like(case.predicted_probability, 1.0 / 3.0),
        }
        inference: dict[str, tuple[SampledPath, Any]] = {}
        for name, probability in probability_by_condition.items():
            posterior = exact_semi_markov_posterior(
                probability,
                case.renewal_evidence,
                decoding_prior,
                conditioning,
            )
            path = posterior.map_path()
            marginals = posterior.marginals()
            inference[name] = (path, marginals)
            _update_hsmm_path_statistics(
                state_statistics[name],
                truth=case.truth_path,
                path=path,
                state_probability=marginals.state_probability,
                renewal_probability=marginals.renewal_probability,
                expected_segment_count=marginals.expected_segment_count,
            )

        zero_amplitude = np.zeros_like(case.truth_amplitude)
        _update_profile_statistics(
            amplitude_source_statistics["predicted_amplitude"],
            case.predicted_amplitude,
            case.truth_amplitude,
        )
        _update_profile_statistics(
            amplitude_source_statistics["prior_only"],
            zero_amplitude,
            case.truth_amplitude,
        )
        profile_conditions = {
            "truth_state_truth_amplitude": (
                inference["truth_state"][0], case.truth_amplitude
            ),
            "truth_state_predicted_amplitude": (
                inference["truth_state"][0], case.predicted_amplitude
            ),
            "predicted_state_truth_amplitude": (
                inference["predicted_state"][0], case.truth_amplitude
            ),
            "predicted_state_predicted_amplitude": (
                inference["predicted_state"][0], case.predicted_amplitude
            ),
            "prior_only": (inference["prior_only"][0], zero_amplitude),
        }
        for name, (path, amplitude) in profile_conditions.items():
            _update_profile_statistics(
                profile_statistics[name],
                _profile_fit_on_path(amplitude, path),
                case.truth_amplitude,
            )
    return {
        "schema": "structured_ginn_v2_hsmm_oracle_v1",
        "status": "success",
        "audit_resolution": "model_grid",
        "parent_count": parent_count,
        "batch_count": batch_count,
        "truth_state_confidence": float(truth_state_confidence),
        "decode_policy": decode_policy.to_mapping(),
        "prior": prior.to_mapping(),
        "conditioning": conditioning.to_mapping(),
        "direct_evidence": {"predicted_state": _direct_state_metrics(cases)},
        "evidence_usage": {
            "state_log_potential": "conditions exact state-duration HSMM",
            "projected_log_ai_increment_mean": (
                "segment-wise three-basis diagnostic upper bound"
            ),
            "signed_reflectivity_mean": (
                "creates bounded renewal evidence and also enters the learned "
                "segment parameter head"
            ),
        },
        "state_duration_conditions": {
            name: _finalize_hsmm_path_statistics(statistics)
            for name, statistics in state_statistics.items()
        },
        "amplitude_sources": {
            name: _finalize_profile_statistics(statistics)
            for name, statistics in amplitude_source_statistics.items()
        },
        "profile_substitution": {
            name: _finalize_profile_statistics(statistics)
            for name, statistics in profile_statistics.items()
        },
    }


def _new_hsmm_calibration_statistics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "renewal_sample_count": 0,
        "state_brier": 0.0,
        "renewal_brier": 0.0,
        "state_correct": 0,
        "state_class_count": np.zeros(3, dtype=np.int64),
        "state_class_correct": np.zeros(3, dtype=np.int64),
        "trace_count": 0,
        "truth_segment_count": 0.0,
        "map_segment_count": 0.0,
        "expected_segment_count": 0.0,
        "profile": _new_profile_statistics(),
    }


def _update_hsmm_calibration_statistics(
    statistics: dict[str, Any],
    case: _HsmmOracleCase,
    path: SampledPath,
    marginals: SemiMarkovMarginals,
) -> None:
    target = case.truth_path.state.astype(np.int64)
    probability = np.asarray(marginals.state_probability, dtype=np.float64)
    if probability.shape != (target.size, 3):
        raise ValueError("HSMM calibration marginal shape differs from truth.")
    truth_probability = np.eye(3, dtype=np.float64)[target]
    statistics["sample_count"] += target.size
    statistics["state_brier"] += float(
        np.sum((probability - truth_probability) ** 2)
    )
    statistics["state_correct"] += int(np.count_nonzero(path.state == target))
    statistics["trace_count"] += 1
    statistics["truth_segment_count"] += len(case.truth_path.segments)
    statistics["map_segment_count"] += len(path.segments)
    statistics["expected_segment_count"] += float(
        marginals.expected_segment_count
    )
    truth_renewal = np.zeros(target.size, dtype=np.float64)
    truth_renewal[[segment[1] for segment in case.truth_path.segments[1:]]] = 1.0
    if target.size > 1:
        renewal_error = (
            np.asarray(marginals.renewal_probability, dtype=np.float64)[1:]
            - truth_renewal[1:]
        )
        statistics["renewal_brier"] += float(np.sum(renewal_error**2))
        statistics["renewal_sample_count"] += target.size - 1
    for state in range(3):
        selected = target == state
        statistics["state_class_count"][state] += int(np.count_nonzero(selected))
        statistics["state_class_correct"][state] += int(
            np.count_nonzero(path.state[selected] == state)
        )
    _update_profile_statistics(
        statistics["profile"],
        _profile_fit_on_path(case.truth_amplitude, path),
        case.truth_amplitude,
    )


def _finalize_hsmm_calibration_statistics(
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    samples = int(statistics["sample_count"])
    traces = int(statistics["trace_count"])
    class_count = np.asarray(statistics["state_class_count"], dtype=np.int64)
    class_correct = np.asarray(statistics["state_class_correct"], dtype=np.int64)
    profile = _finalize_profile_statistics(statistics["profile"])
    truth_segments = float(statistics["truth_segment_count"] / traces)
    map_segments = float(statistics["map_segment_count"] / traces)
    expected_segments = float(statistics["expected_segment_count"] / traces)
    return {
        "supported_samples": samples,
        "trace_count": traces,
        "state_accuracy": float(statistics["state_correct"] / samples),
        "state_brier": float(statistics["state_brier"] / samples),
        "renewal_brier": float(
            statistics["renewal_brier"]
            / max(int(statistics["renewal_sample_count"]), 1)
        ),
        "state_balanced_accuracy": float(
            np.mean(class_correct[class_count > 0] / class_count[class_count > 0])
        ),
        "truth_segment_count_mean": truth_segments,
        "map_segment_count_mean": map_segments,
        "map_segment_count_bias": map_segments - truth_segments,
        "posterior_expected_segment_count_mean": expected_segments,
        "posterior_expected_segment_count_bias": expected_segments - truth_segments,
        "truth_amplitude_profile_rmse": profile["increment_rmse"],
        "truth_amplitude_profile_correlation": profile["increment_correlation"],
    }


def _select_hsmm_calibration_candidate(
    candidates: list[dict[str, Any]],
    baseline: dict[str, Any],
    *,
    balanced_accuracy_tolerance: float = 0.02,
    profile_relative_tolerance: float = 0.02,
    profile_absolute_tolerance: float = 1.0e-5,
    map_segment_count_tolerance: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a calibrated posterior using proper scores behind MAP guards."""

    if not candidates:
        raise ValueError("HSMM calibration requires at least one candidate.")
    for value in (
        balanced_accuracy_tolerance,
        profile_relative_tolerance,
        profile_absolute_tolerance,
        map_segment_count_tolerance,
    ):
        if value < 0.0:
            raise ValueError("HSMM selection tolerances must be non-negative.")
    baseline_metrics = baseline["metrics"]
    required = {
        "state_balanced_accuracy",
        "state_brier",
        "renewal_brier",
        "map_segment_count_bias",
        "posterior_expected_segment_count_bias",
        "truth_amplitude_profile_rmse",
    }
    for row in candidates:
        missing = required.difference(row["metrics"])
        if missing:
            raise ValueError(
                f"HSMM calibration candidate lacks metrics: {sorted(missing)}"
            )
    maximum_profile = max(
        baseline_metrics["truth_amplitude_profile_rmse"]
        * (1.0 + profile_relative_tolerance),
        baseline_metrics["truth_amplitude_profile_rmse"]
        + profile_absolute_tolerance,
    )
    admissible = [
        row
        for row in candidates
        if (
            row["metrics"]["state_balanced_accuracy"]
            >= (
                baseline_metrics["state_balanced_accuracy"]
                - balanced_accuracy_tolerance
            )
            and abs(row["metrics"]["map_segment_count_bias"])
            <= (
                abs(baseline_metrics["map_segment_count_bias"])
                + map_segment_count_tolerance
            )
            and row["metrics"]["truth_amplitude_profile_rmse"]
            <= maximum_profile
        )
    ]
    if not admissible:
        return baseline, {
            "admissible_candidate_count": 0,
            "fallback_to_baseline": True,
            "balanced_accuracy_tolerance": balanced_accuracy_tolerance,
            "profile_relative_tolerance": profile_relative_tolerance,
            "profile_absolute_tolerance": profile_absolute_tolerance,
            "map_segment_count_tolerance": map_segment_count_tolerance,
        }

    state_scale = max(float(baseline_metrics["state_brier"]), 1.0e-8)
    renewal_scale = max(float(baseline_metrics["renewal_brier"]), 1.0e-8)
    count_scale = max(
        abs(float(baseline_metrics["posterior_expected_segment_count_bias"])),
        1.0,
    )

    def proper_score(row: Mapping[str, Any]) -> float:
        metrics = row["metrics"]
        return float(
            metrics["state_brier"] / state_scale
            + metrics["renewal_brier"] / renewal_scale
            + abs(metrics["posterior_expected_segment_count_bias"]) / count_scale
        )

    selected = min(
        admissible,
        key=lambda row: (
            proper_score(row),
            row["metrics"]["state_brier"],
            row["metrics"]["renewal_brier"],
            abs(row["metrics"]["posterior_expected_segment_count_bias"]),
            -row["metrics"]["state_balanced_accuracy"],
        ),
    )
    return selected, {
        "admissible_candidate_count": len(admissible),
        "fallback_to_baseline": False,
        "selected_proper_score": proper_score(selected),
        "baseline_proper_score": proper_score(baseline),
        "proper_score_terms": (
            "state_brier/baseline + renewal_brier/baseline + "
            "abs(expected_count_bias)/max(abs(baseline_bias),1)"
        ),
        "balanced_accuracy_tolerance": balanced_accuracy_tolerance,
        "profile_relative_tolerance": profile_relative_tolerance,
        "profile_absolute_tolerance": profile_absolute_tolerance,
        "map_segment_count_tolerance": map_segment_count_tolerance,
    }


def calibrate_semi_markov_fusion(
    generator: ConditionalGenerator,
    batches: Iterable[Mapping[str, object]],
    *,
    prior: SemiMarkovPrior,
    conditioning_candidates: Iterable[SemiMarkovConditioning] | None = None,
    truth_state_confidence: float = 0.999,
    logger: logging.Logger | None = None,
    log_every_batches: int = 5,
) -> dict[str, Any]:
    """Select HSMM term strengths, then run one full selected Oracle."""

    log = logger or logging.getLogger(__name__)
    cases, parent_count, batch_count = _collect_hsmm_oracle_cases(
        generator,
        batches,
        truth_state_confidence=truth_state_confidence,
        logger=log,
        log_every_batches=log_every_batches,
    )
    calibration_cases = _central_hsmm_calibration_cases(cases)
    candidates: list[dict[str, Any]] = []
    state_weights = (0.25, 0.5, 1.0, 2.0, 4.0)
    duration_temperatures = (0.25, 0.5, 1.0, 2.0)
    transition_temperatures = (0.5, 1.0, 2.0, 4.0)
    search = (
        tuple(conditioning_candidates)
        if conditioning_candidates is not None
        else tuple(
            SemiMarkovConditioning(*values)
            for values in product(
                state_weights,
                duration_temperatures,
                transition_temperatures,
            )
        )
    )
    if not search or any(
        not isinstance(value, SemiMarkovConditioning) for value in search
    ):
        raise ValueError(
            "conditioning_candidates must contain SemiMarkovConditioning values."
        )
    baseline_mapping = SemiMarkovConditioning().to_mapping()
    if not any(value.to_mapping() == baseline_mapping for value in search):
        raise ValueError("conditioning_candidates must include the default baseline.")
    start_time = time.perf_counter()
    decoding_prior = prior_with_same_state_renewal(
        prior,
        generator.decode_policy.same_state_renewal_probability,
    )
    for index, conditioning in enumerate(search, start=1):
        statistics = _new_hsmm_calibration_statistics()
        for case in calibration_cases:
            posterior = exact_semi_markov_posterior(
                case.predicted_probability,
                case.renewal_evidence,
                decoding_prior,
                conditioning,
            )
            _update_hsmm_calibration_statistics(
                statistics,
                case,
                posterior.map_path(),
                posterior.marginals(),
            )
        candidates.append(
            {
                "conditioning": conditioning.to_mapping(),
                "metrics": _finalize_hsmm_calibration_statistics(statistics),
            }
        )
        if index % 6 == 0:
            log.info(
                "HSMM calibration | candidates=%d/%d | elapsed=%.1fs",
                index,
                len(search),
                time.perf_counter() - start_time,
            )

    baseline = next(
        row
        for row in candidates
        if row["conditioning"] == baseline_mapping
    )
    selected, selection_diagnostics = _select_hsmm_calibration_candidate(
        candidates,
        baseline,
    )
    selected_conditioning = SemiMarkovConditioning.from_mapping(
        selected["conditioning"]
    )
    searched_state_weights = tuple(
        sorted({value.state_evidence_weight for value in search})
    )
    searched_duration_temperatures = tuple(
        sorted({value.duration_temperature for value in search})
    )
    searched_transition_temperatures = tuple(
        sorted({value.transition_temperature for value in search})
    )

    def on_search_boundary(value: float, values: tuple[float, ...]) -> bool:
        return len(values) > 1 and value in (values[0], values[-1])

    boundary_flags = {
        "state_evidence_weight": on_search_boundary(
            selected_conditioning.state_evidence_weight,
            searched_state_weights,
        ),
        "duration_temperature": on_search_boundary(
            selected_conditioning.duration_temperature,
            searched_duration_temperatures,
        ),
        "transition_temperature": on_search_boundary(
            selected_conditioning.transition_temperature,
            searched_transition_temperatures,
        ),
    }
    oracle = _evaluate_hsmm_cases(
        cases,
        parent_count=parent_count,
        batch_count=batch_count,
        prior=prior,
        conditioning=selected_conditioning,
        decode_policy=generator.decode_policy,
        truth_state_confidence=truth_state_confidence,
    )
    return {
        "schema": "structured_ginn_v2_hsmm_calibration_v4",
        "status": "success",
        "audit_resolution": "model_grid",
        "parent_count": parent_count,
        "batch_count": batch_count,
        "oracle_trace_count": len(cases),
        "calibration_trace_count": len(calibration_cases),
        "calibration_trace_policy": (
            "five center traces per parent-zone batch; all traces enter the "
            "selected full Oracle"
        ),
        "candidate_count": len(candidates),
        "search_grid": {
            "state_evidence_weight": list(searched_state_weights),
            "duration_temperature": list(searched_duration_temperatures),
            "transition_temperature": list(searched_transition_temperatures),
        },
        "selected_on_search_boundary": {
            **boundary_flags,
            "any": any(boundary_flags.values()),
        },
        "selection_rule": (
            "guard MAP balanced accuracy, segment-count bias, and truth-amplitude "
            "profile RMSE near the (1,1,1) baseline; then minimize normalized "
            "state Brier, renewal Brier, and posterior expected count bias"
        ),
        "selection_diagnostics": selection_diagnostics,
        "direct_evidence": {"predicted_state": _direct_state_metrics(cases)},
        "baseline": baseline,
        "selected": selected,
        "candidates": candidates,
        "oracle": oracle,
    }


@dataclass(frozen=True)
class _SegmentProfileDataset:
    features: np.ndarray
    target_coefficient: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    target_square_mean: np.ndarray
    identifiability_rank: np.ndarray
    basis_condition: np.ndarray
    clipping_fraction: np.ndarray
    state_id: np.ndarray
    parent_ids: tuple[str, ...]

    @property
    def segment_count(self) -> int:
        return int(self.features.shape[0])


def _observable_evidence_from_batch(
    generator: ConditionalGenerator,
    source: Mapping[str, object],
    output: Mapping[str, np.ndarray],
) -> ObservableEvidence:
    tile = source.get("observation_tile")
    if not isinstance(tile, ObservationTile):
        raise TypeError("segment profile batch requires an ObservationTile.")
    support = np.asarray(source["support"], dtype=bool)[0]
    support &= np.asarray(output["support"], dtype=bool)[0]
    background = np.asarray(source["background_lfm_linear"], dtype=np.float64)[0]
    tuning = np.full(background.shape, tile.model_axis.sample_interval, dtype=np.float64)
    if tile.sample_domain == "time":
        tuning[support] = 1.0 / (2.0 * generator.dominant_frequency_hz)
    else:
        relation = dict(source.get("ai_velocity_relation") or {})
        if "a" not in relation or "b" not in relation:
            raise ValueError("depth profile evidence requires the AI-Vp relation.")
        impedance = np.exp(background[support])
        velocity = velocity_from_ai(
            impedance,
            a=float(relation["a"]),
            b=float(relation["b"]),
        )
        tuning[support] = velocity / (4.0 * generator.dominant_frequency_hz)
    return ObservableEvidence(
        model_axis=tile.model_axis,
        highres_axis=tile.highres_axis,
        background_lfm_linear=background,
        background_lfm_linear_highres=np.asarray(
            source["lfm_anchor_highres"], dtype=np.float64
        ),
        projected_log_ai_increment_mean=np.asarray(
            output["projected_log_ai_increment_mean"], dtype=np.float64
        )[0],
        projected_log_ai_increment_scale=np.asarray(
            output["projected_log_ai_increment_scale"], dtype=np.float64
        )[0],
        signed_reflectivity_mean=np.asarray(
            output["signed_reflectivity_mean"], dtype=np.float64
        )[0],
        signed_reflectivity_scale=np.asarray(
            output["signed_reflectivity_scale"], dtype=np.float64
        )[0],
        state_log_potential=np.asarray(
            output["state_log_potential"], dtype=np.float64
        )[0],
        local_tuning_scale=tuning,
        support=support,
        highres_support=np.asarray(
            source["highres_truth_support"], dtype=bool
        ),
        lateral_m=tile.lateral_m,
        x_m=tile.x_m,
        y_m=tile.y_m,
        identity=tile.identity,
    )


def _collect_segment_profile_dataset(
    generator: ConditionalGenerator,
    batches: Iterable[Mapping[str, object]],
    *,
    central_trace_count: int,
    logger: logging.Logger,
) -> _SegmentProfileDataset:
    generator.network.eval()
    feature_rows: list[np.ndarray] = []
    coefficient_rows: list[np.ndarray] = []
    gram_rows: list[np.ndarray] = []
    cross_rows: list[np.ndarray] = []
    square_rows: list[float] = []
    rank_rows: list[int] = []
    condition_rows: list[float] = []
    clipping_rows: list[float] = []
    state_rows: list[int] = []
    parents: set[str] = set()
    batch_count = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_count, source in enumerate(batches, start=1):
            parent_id = str(source.get("parent_id") or "")
            if not parent_id:
                raise ValueError("segment profile batch requires parent_id.")
            parents.add(parent_id)
            output = _network_output(generator, source)
            evidence = _observable_evidence_from_batch(generator, source, output)
            width = evidence.support.shape[0]
            count = min(central_trace_count, width)
            start_trace = (width - count) // 2
            selected_traces = set(range(start_trace, start_trace + count))
            trace = np.asarray(source["segment_trace_index"], dtype=np.int64)
            state = np.asarray(source["segment_state_id"], dtype=np.int64)
            start = np.asarray(source["segment_start_index"], dtype=np.int64)
            stop = np.asarray(source["segment_stop_index"], dtype=np.int64)
            duration = np.asarray(
                source["segment_duration_fraction"], dtype=np.float64
            )
            clipping = np.asarray(
                source["segment_clipping_fraction"], dtype=np.float64
            )
            if not (
                trace.shape
                == state.shape
                == start.shape
                == stop.shape
                == duration.shape
                == clipping.shape
            ):
                raise ValueError("segment profile arrays must share one shape.")
            indices = [
                index
                for index, trace_index in enumerate(trace.tolist())
                if trace_index in selected_traces
            ]
            extents = tuple(
                SegmentExtent(
                    trace_index=int(trace[index]),
                    state_id=int(state[index]),
                    start_index=int(start[index]),
                    stop_index=int(stop[index]),
                    duration_fraction=float(duration[index]),
                )
                for index in indices
            )
            if not extents:
                continue
            features = build_segment_profile_features(evidence, extents)
            truth = np.asarray(
                source["highres_log_ai_increment"], dtype=np.float64
            )
            truth_support = np.asarray(
                source["highres_truth_support"], dtype=bool
            )
            for local_index, (source_index, extent) in enumerate(
                zip(indices, extents, strict=True)
            ):
                profile = truth[
                    extent.trace_index,
                    extent.start_index : extent.stop_index,
                ]
                valid = truth_support[
                    extent.trace_index,
                    extent.start_index : extent.stop_index,
                ]
                if not np.all(valid) or np.any(~np.isfinite(profile)):
                    continue
                statistics = profile_sufficient_statistics(profile)
                feature_rows.append(features[local_index])
                coefficient_rows.append(statistics.coefficient)
                gram_rows.append(statistics.gram)
                cross_rows.append(statistics.cross)
                square_rows.append(statistics.target_square_mean)
                rank_rows.append(statistics.identifiability_rank)
                condition_rows.append(statistics.basis_condition)
                clipping_rows.append(float(clipping[source_index]))
                state_rows.append(extent.state_id)
            if batch_count % 20 == 0:
                logger.info(
                    "profile dataset | batches=%d | parents=%d | segments=%d | elapsed=%.1fs",
                    batch_count,
                    len(parents),
                    len(feature_rows),
                    time.perf_counter() - start_time,
                )
    if batch_count == 0 or not feature_rows:
        raise ValueError("segment profile dataset is empty.")
    return _SegmentProfileDataset(
        features=np.asarray(feature_rows, dtype=np.float32),
        target_coefficient=np.asarray(coefficient_rows, dtype=np.float32),
        gram=np.asarray(gram_rows, dtype=np.float32),
        cross=np.asarray(cross_rows, dtype=np.float32),
        target_square_mean=np.asarray(square_rows, dtype=np.float32),
        identifiability_rank=np.asarray(rank_rows, dtype=np.int8),
        basis_condition=np.asarray(condition_rows, dtype=np.float32),
        clipping_fraction=np.asarray(clipping_rows, dtype=np.float32),
        state_id=np.asarray(state_rows, dtype=np.int8),
        parent_ids=tuple(sorted(parents)),
    )


def _profile_prior(
    dataset: _SegmentProfileDataset,
    *,
    condition_max: float,
) -> dict[str, Any]:
    identifiable = (
        (dataset.identifiability_rank == 3)
        & (dataset.basis_condition <= condition_max)
        & (dataset.clipping_fraction == 0.0)
    )
    mean = np.empty((3, 3), dtype=np.float64)
    scale = np.empty((3, 3), dtype=np.float64)
    counts = np.empty(3, dtype=np.int64)
    for state in range(3):
        selected = identifiable & (dataset.state_id == state)
        counts[state] = np.count_nonzero(selected)
        if counts[state] == 0:
            raise ValueError(f"state {state} has no identifiable profile segments.")
        values = dataset.target_coefficient[selected].astype(np.float64)
        mean[state] = np.mean(values, axis=0)
        scale[state] = np.maximum(np.std(values, axis=0), 1.0e-3)
    return {
        "schema": "structured_ginn_v2_segment_profile_prior_v1",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "identifiable_segment_count_by_state": counts.tolist(),
        "identifiability_condition_max": float(condition_max),
    }


def _profile_mse_tensor(
    coefficient: torch.Tensor,
    gram: torch.Tensor,
    cross: torch.Tensor,
    target_square_mean: torch.Tensor,
) -> torch.Tensor:
    value = (
        torch.einsum("bi,bij,bj->b", coefficient, gram, coefficient)
        - 2.0 * torch.einsum("bi,bi->b", coefficient, cross)
        + target_square_mean
    )
    return torch.clamp(value, min=0.0)


def _profile_loss(
    head: torch.nn.Module,
    dataset: _SegmentProfileDataset,
    indices: np.ndarray,
    *,
    device: torch.device,
    config: SegmentProfileLearningConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    selected = np.asarray(indices, dtype=np.int64)
    features = torch.as_tensor(
        dataset.features[selected], dtype=torch.float32, device=device
    )
    target = torch.as_tensor(
        dataset.target_coefficient[selected], dtype=torch.float32, device=device
    )
    gram = torch.as_tensor(dataset.gram[selected], dtype=torch.float32, device=device)
    cross = torch.as_tensor(dataset.cross[selected], dtype=torch.float32, device=device)
    target_square = torch.as_tensor(
        dataset.target_square_mean[selected], dtype=torch.float32, device=device
    )
    output = head(features)
    profile_mse = _profile_mse_tensor(
        output["mean"], gram, cross, target_square
    )
    diagonal = torch.diagonal(gram, dim1=1, dim2=2)
    profile_variance = torch.clamp(
        torch.sum(diagonal * output["scale"] ** 2, dim=1),
        min=1.0e-8,
    )
    profile_nll = 0.5 * (
        profile_mse / profile_variance + torch.log(profile_variance)
    )
    identifiable_numpy = (
        (dataset.identifiability_rank[selected] == 3)
        & (dataset.basis_condition[selected] <= config.identifiability_condition_max)
        & (dataset.clipping_fraction[selected] == 0.0)
    )
    identifiable = torch.as_tensor(
        identifiable_numpy, dtype=torch.bool, device=device
    )
    if torch.any(identifiable):
        residual = (
            output["mean"][identifiable] - target[identifiable]
        ) / output["scale"][identifiable]
        parameter_nll = 0.5 * torch.mean(
            residual**2 + 2.0 * torch.log(output["scale"][identifiable])
        )
    else:
        parameter_nll = torch.zeros((), dtype=features.dtype, device=device)
    loss = torch.mean(profile_nll) + config.parameter_nll_weight * parameter_nll
    return loss, {
        "loss": float(loss.detach().cpu()),
        "profile_rmse": float(torch.sqrt(torch.mean(profile_mse)).detach().cpu()),
        "profile_nll": float(torch.mean(profile_nll).detach().cpu()),
        "parameter_nll": float(parameter_nll.detach().cpu()),
    }


def _coefficient_correlation(
    predicted: np.ndarray,
    truth: np.ndarray,
) -> list[float | None]:
    result: list[float | None] = []
    for index in range(3):
        left = predicted[:, index]
        right = truth[:, index]
        if np.std(left) <= 0.0 or np.std(right) <= 0.0:
            result.append(None)
        else:
            result.append(float(np.corrcoef(left, right)[0, 1]))
    return result


def _profile_method_metrics(
    dataset: _SegmentProfileDataset,
    coefficient: np.ndarray,
    *,
    scale: np.ndarray | None,
    condition_max: float,
) -> dict[str, Any]:
    values = np.asarray(coefficient, dtype=np.float64)
    mse = (
        np.einsum("bi,bij,bj->b", values, dataset.gram, values)
        - 2.0 * np.einsum("bi,bi->b", values, dataset.cross)
        + dataset.target_square_mean
    )
    mse = np.maximum(mse, 0.0)
    identifiable = (
        (dataset.identifiability_rank == 3)
        & (dataset.basis_condition <= condition_max)
        & (dataset.clipping_fraction == 0.0)
    )
    error = values[identifiable] - dataset.target_coefficient[identifiable]
    result: dict[str, Any] = {
        "segment_count": dataset.segment_count,
        "profile_rmse": float(np.sqrt(np.mean(mse))),
        "identifiable_segment_count": int(np.count_nonzero(identifiable)),
        "identifiable_coefficient_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "identifiable_coefficient_mae": np.mean(np.abs(error), axis=0).tolist(),
        "identifiable_coefficient_correlation": _coefficient_correlation(
            values[identifiable], dataset.target_coefficient[identifiable]
        ),
    }
    if scale is not None:
        spread = np.asarray(scale, dtype=np.float64)
        diagonal = np.diagonal(dataset.gram, axis1=1, axis2=2)
        profile_variance = np.maximum(
            np.sum(diagonal * spread**2, axis=1), 1.0e-8
        )
        result["profile_gaussian_nll"] = float(
            np.mean(0.5 * (mse / profile_variance + np.log(profile_variance)))
        )
        absolute = np.abs(error)
        for label, z_value in (("50", 0.67449), ("80", 1.28155), ("95", 1.95996)):
            result[f"identifiable_parameter_coverage_{label}"] = float(
                np.mean(absolute <= z_value * spread[identifiable])
            )
    return result


def _predict_profile_head(
    generator: ConditionalGenerator,
    dataset: _SegmentProfileDataset,
    *,
    batch_size: int,
    apply_variance_calibration: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if generator.profile_head is None:
        raise RuntimeError("generator has no segment profile head.")
    generator.profile_head.eval()
    mean_rows: list[np.ndarray] = []
    scale_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, dataset.segment_count, batch_size):
            stop = min(start + batch_size, dataset.segment_count)
            output = generator.profile_head(
                torch.as_tensor(
                    dataset.features[start:stop],
                    dtype=torch.float32,
                    device=generator.device,
                )
            )
            mean_rows.append(output["mean"].cpu().numpy())
            scale_rows.append(output["scale"].cpu().numpy())
    mean = np.concatenate(mean_rows)
    scale = np.concatenate(scale_rows)
    if (
        apply_variance_calibration
        and generator.coefficient_variance_calibration is not None
    ):
        scale *= np.asarray(
            generator.coefficient_variance_calibration.temperature,
            dtype=np.float64,
        )
    return mean, scale


def _fit_coefficient_scale_temperature(
    residual: np.ndarray,
    base_scale: np.ndarray,
) -> np.ndarray:
    """Closed-form diagonal Gaussian temperature for fixed means."""

    error = np.asarray(residual, dtype=np.float64)
    scale = np.asarray(base_scale, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != 3 or scale.shape != error.shape:
        raise ValueError("coefficient residual and scale must have shape [segment, 3].")
    if error.shape[0] == 0:
        raise ValueError("coefficient variance calibration received no segments.")
    if np.any(~np.isfinite(error)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("coefficient variance calibration inputs must be finite.")
    temperature = np.sqrt(np.mean((error / scale) ** 2, axis=0))
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise FloatingPointError(
            "coefficient variance calibration produced invalid temperatures."
        )
    return temperature


def _coefficient_uncertainty_metrics(
    residual: np.ndarray,
    scale: np.ndarray,
    state_id: np.ndarray,
) -> dict[str, Any]:
    error = np.asarray(residual, dtype=np.float64)
    spread = np.asarray(scale, dtype=np.float64)
    state = np.asarray(state_id, dtype=np.int64)
    if error.shape != spread.shape or error.ndim != 2 or error.shape[1] != 3:
        raise ValueError("coefficient uncertainty arrays must have shape [segment, 3].")
    if state.shape != (error.shape[0],):
        raise ValueError("coefficient uncertainty state array has the wrong shape.")
    standardized = error / spread

    def metrics(selected: np.ndarray) -> dict[str, Any]:
        if not np.any(selected):
            raise ValueError("coefficient uncertainty metrics require selected segments.")
        local_error = error[selected]
        local_scale = spread[selected]
        local_standardized = standardized[selected]
        result: dict[str, Any] = {
            "segment_count": int(np.count_nonzero(selected)),
            "coefficient_bias": np.mean(local_error, axis=0).tolist(),
            "coefficient_rmse": np.sqrt(np.mean(local_error**2, axis=0)).tolist(),
            "mean_scale": np.mean(local_scale, axis=0).tolist(),
            "standardized_rmse": np.sqrt(
                np.mean(local_standardized**2, axis=0)
            ).tolist(),
            "gaussian_nll": np.mean(
                0.5 * (local_standardized**2 + 2.0 * np.log(local_scale)),
                axis=0,
            ).tolist(),
        }
        absolute = np.abs(local_standardized)
        for label, z_value in (("50", 0.67449), ("80", 1.28155), ("95", 1.95996)):
            result[f"coverage_{label}"] = np.mean(
                absolute <= z_value, axis=0
            ).tolist()
            result[f"coverage_{label}_overall"] = float(
                np.mean(absolute <= z_value)
            )
        return result

    all_selected = np.ones(error.shape[0], dtype=bool)
    return {
        "aggregate": metrics(all_selected),
        "by_state": {
            str(state_value): metrics(state == state_value)
            for state_value in range(3)
        },
    }


def calibrate_coefficient_variance(
    generator: ConditionalGenerator,
    batches: Iterable[Mapping[str, object]],
    *,
    config: CoefficientVarianceCalibrationConfig,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Calibrate c0/c1/c2 scale temperatures with fixed means."""

    if generator.profile_head is None:
        raise ValueError("coefficient variance calibration requires a profile head.")
    if generator.coefficient_variance_calibration is not None:
        raise ValueError("checkpoint already contains coefficient variance calibration.")
    log = logger or logging.getLogger(__name__)
    dataset = _collect_segment_profile_dataset(
        generator,
        batches,
        central_trace_count=config.central_trace_count,
        logger=log,
    )
    mean, raw_scale = _predict_profile_head(
        generator,
        dataset,
        batch_size=config.batch_size,
        apply_variance_calibration=False,
    )
    identifiable = (
        (dataset.identifiability_rank == 3)
        & (dataset.basis_condition <= config.identifiability_condition_max)
        & (dataset.clipping_fraction == 0.0)
    )
    identifiable_count = int(np.count_nonzero(identifiable))
    if identifiable_count == 0:
        raise ValueError("calibration split has no identifiable profile segments.")
    residual = (
        dataset.target_coefficient[identifiable].astype(np.float64)
        - mean[identifiable].astype(np.float64)
    )
    base_scale = raw_scale[identifiable].astype(np.float64)
    state = dataset.state_id[identifiable]
    temperature = _fit_coefficient_scale_temperature(residual, base_scale)
    calibrated_scale = base_scale * temperature[None, :]
    before = _coefficient_uncertainty_metrics(residual, base_scale, state)
    after = _coefficient_uncertainty_metrics(residual, calibrated_scale, state)
    calibration = CoefficientVarianceCalibration(
        temperature=tuple(float(value) for value in temperature)
    )
    generator.set_coefficient_variance_calibration(calibration)
    raw_profile = _profile_method_metrics(
        dataset,
        mean,
        scale=raw_scale,
        condition_max=config.identifiability_condition_max,
    )
    calibrated_profile = _profile_method_metrics(
        dataset,
        mean,
        scale=raw_scale * temperature[None, :],
        condition_max=config.identifiability_condition_max,
    )
    log.info(
        "coefficient variance calibrated | parents=%d | identifiable=%d | "
        "temperature=(%.4f, %.4f, %.4f) | standardized_rmse=(%.3f, %.3f, %.3f)",
        len(dataset.parent_ids),
        identifiable_count,
        *temperature.tolist(),
        *after["aggregate"]["standardized_rmse"],
    )
    return {
        "schema": "structured_ginn_v2_coefficient_variance_report_v1",
        "status": "success",
        "method": "closed_form_identifiable_truth_segment_diagonal_temperature",
        "uncertainty_decomposition": {
            "coefficient_scale": "conditional_on_supplied_segment_extent_and_state",
            "segmentation": "reserved_for_hsmm_backward_sampling",
        },
        "parent_count": len(dataset.parent_ids),
        "segment_count": dataset.segment_count,
        "identifiable_segment_count": identifiable_count,
        "identifiability_condition_max": config.identifiability_condition_max,
        "calibration": calibration.to_mapping(),
        "before": before,
        "after": after,
        "profile_distribution_before": raw_profile,
        "profile_distribution_after": calibrated_profile,
        "mean_unchanged": True,
    }


def evaluate_segment_profile_head(
    generator: ConditionalGenerator,
    batches: Iterable[Mapping[str, object]],
    *,
    prior: Mapping[str, Any],
    config: SegmentProfileLearningConfig,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or logging.getLogger(__name__)
    dataset = _collect_segment_profile_dataset(
        generator,
        batches,
        central_trace_count=config.central_trace_count,
        logger=log,
    )
    learned_mean, learned_scale = _predict_profile_head(
        generator, dataset, batch_size=config.batch_size
    )
    prior_mean = np.asarray(prior["mean"], dtype=np.float64)[dataset.state_id]
    prior_scale = np.asarray(prior["scale"], dtype=np.float64)[dataset.state_id]
    deterministic = dataset.features[:, :3].astype(np.float64)
    methods = {
        "deterministic_evidence_fit": _profile_method_metrics(
            dataset,
            deterministic,
            scale=None,
            condition_max=config.identifiability_condition_max,
        ),
        "learned_profile_head": _profile_method_metrics(
            dataset,
            learned_mean,
            scale=learned_scale,
            condition_max=config.identifiability_condition_max,
        ),
        "state_conditioned_prior": _profile_method_metrics(
            dataset,
            prior_mean,
            scale=prior_scale,
            condition_max=config.identifiability_condition_max,
        ),
    }
    learned_rmse = methods["learned_profile_head"]["profile_rmse"]
    deterministic_rmse = methods["deterministic_evidence_fit"]["profile_rmse"]
    prior_rmse = methods["state_conditioned_prior"]["profile_rmse"]
    strongest_baseline_rmse = min(deterministic_rmse, prior_rmse)
    return {
        "schema": "structured_ginn_v2_segment_profile_evaluation_v1",
        "status": "success",
        "parent_count": len(dataset.parent_ids),
        "segment_count": dataset.segment_count,
        "identifiability": {
            "rank_1": int(np.count_nonzero(dataset.identifiability_rank == 1)),
            "rank_2": int(np.count_nonzero(dataset.identifiability_rank == 2)),
            "rank_3": int(np.count_nonzero(dataset.identifiability_rank == 3)),
            "clipped": int(np.count_nonzero(dataset.clipping_fraction > 0.0)),
            "condition_max": config.identifiability_condition_max,
        },
        "methods": methods,
        "learned_minus_deterministic_profile_rmse": float(
            learned_rmse - deterministic_rmse
        ),
        "learned_minus_state_conditioned_prior_profile_rmse": float(
            learned_rmse - prior_rmse
        ),
        "learned_minus_strongest_baseline_profile_rmse": float(
            learned_rmse - strongest_baseline_rmse
        ),
        "learned_beats_deterministic": bool(learned_rmse < deterministic_rmse),
        "learned_beats_state_conditioned_prior": bool(learned_rmse < prior_rmse),
        "learned_beats_strongest_baseline": bool(
            learned_rmse < strongest_baseline_rmse
        ),
    }


def train_segment_profile_head(
    generator: ConditionalGenerator,
    training_batches: Iterable[Mapping[str, object]],
    tuning_batches: Iterable[Mapping[str, object]],
    *,
    head_config: SegmentProfileHeadConfig,
    config: SegmentProfileLearningConfig,
    logger: logging.Logger | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> dict[str, Any]:
    """Freeze observable evidence and train one segment-profile residual head."""

    log = logger or logging.getLogger(__name__)
    seed_training_random_streams(config.random_seed)
    training = _collect_segment_profile_dataset(
        generator,
        training_batches,
        central_trace_count=config.central_trace_count,
        logger=log,
    )
    tuning = _collect_segment_profile_dataset(
        generator,
        tuning_batches,
        central_trace_count=config.central_trace_count,
        logger=log,
    )
    head = generator.attach_profile_head(head_config)
    feature_mean = np.mean(training.features, axis=0)
    feature_scale = np.maximum(np.std(training.features, axis=0), 1.0e-6)
    head.set_feature_normalization(feature_mean, feature_scale)
    for parameter in generator.network.parameters():
        parameter.requires_grad_(False)
    prior = _profile_prior(
        training, condition_max=config.identifiability_condition_max
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.random_seed)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    all_tuning = np.arange(tuning.segment_count)
    for epoch in range(1, config.epochs + 1):
        head.train()
        permutation = rng.permutation(training.segment_count)
        running = 0.0
        batches_seen = 0
        for start in range(0, training.segment_count, config.batch_size):
            indices = permutation[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _profile_loss(
                head,
                training,
                indices,
                device=generator.device,
                config=config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            running += float(loss.detach().cpu())
            batches_seen += 1
        head.eval()
        tuning_totals = {"loss": 0.0, "profile_rmse": 0.0, "profile_nll": 0.0, "parameter_nll": 0.0}
        tuning_batches_seen = 0
        with torch.no_grad():
            for start in range(0, tuning.segment_count, config.batch_size):
                indices = all_tuning[start : start + config.batch_size]
                _, metrics = _profile_loss(
                    head,
                    tuning,
                    indices,
                    device=generator.device,
                    config=config,
                )
                for name, value in metrics.items():
                    tuning_totals[name] += value
                tuning_batches_seen += 1
        row = {
            "epoch": float(epoch),
            "train_loss": running / batches_seen,
            **{
                f"tuning_{name}": value / tuning_batches_seen
                for name, value in tuning_totals.items()
            },
        }
        history.append(row)
        is_best = row["tuning_profile_rmse"] < best_loss
        if is_best:
            best_loss = row["tuning_profile_rmse"]
            best_epoch = epoch
        log.info(
            "profile epoch %d/%d | train_loss=%.6f | tuning_loss=%.6f | tuning_profile_rmse=%.6f%s",
            epoch,
            config.epochs,
            row["train_loss"],
            row["tuning_loss"],
            row["tuning_profile_rmse"],
            " | best" if is_best else "",
        )
        if checkpoint_callback is not None:
            checkpoint_callback(
                epoch,
                generator,
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_tuning_profile_rmse": best_loss,
                    "history": history,
                    "profile_prior": prior,
                    "profile_learning_config": asdict(config),
                    "profile_head_config": asdict(head_config),
                },
                is_best,
            )
    return {
        "best_epoch": best_epoch,
        "best_tuning_profile_rmse": best_loss,
        "history": history,
        "profile_prior": prior,
        "training_parent_count": len(training.parent_ids),
        "training_segment_count": training.segment_count,
        "tuning_parent_count": len(tuning.parent_ids),
        "tuning_segment_count": tuning.segment_count,
    }


def train_generator(
    generator: ConditionalGenerator,
    training_batches: (
        Iterable[Mapping[str, object]]
        | Callable[[], Iterable[Mapping[str, object]]]
    ),
    tuning_batches: (
        Iterable[Mapping[str, object]]
        | Callable[[], Iterable[Mapping[str, object]]]
    ),
    *,
    config: LearningConfig = LearningConfig(),
    logger: logging.Logger | None = None,
    resume_state: Mapping[str, Any] | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> dict[str, object]:
    """Train the evidence bottleneck with epoch-boundary recovery hooks.

    ``resume_state`` is the runtime payload from a previous ``last.pt``
    checkpoint.  The callback is invoked only after a complete epoch and is
    responsible for durable checkpoint publication; the training loop does
    not keep a second copy of the best model in memory.
    """
    log = logger or logging.getLogger(__name__)
    optimizer = torch.optim.AdamW(
        generator.network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    resume = dict(resume_state or {})
    start_epoch = int(resume.get("epoch", 0)) + 1
    if start_epoch < 1:
        raise ValueError("resume epoch must be non-negative.")
    if resume_state is not None and "optimizer_state" not in resume:
        raise ValueError("resume checkpoint lacks optimizer_state.")
    if "optimizer_state" in resume:
        optimizer.load_state_dict(resume["optimizer_state"])
    if "rng_state" in resume:
        _restore_rng_state(resume["rng_state"])

    def batches(source):
        return source() if callable(source) else source

    history: list[dict[str, float]] = [
        {str(key): float(value) for key, value in dict(row).items()}
        for row in resume.get("history", [])
    ]
    best_loss = float(resume.get("best_tuning_loss", float("inf")))
    best_epoch = int(resume.get("best_epoch", 0))
    if start_epoch > config.epochs:
        log.info(
            "checkpoint already reached target epochs | epoch=%d | target=%d",
            start_epoch - 1,
            config.epochs,
        )
        return {
            "best_tuning_loss": best_loss,
            "best_epoch": best_epoch,
            "history": history,
            "epochs_completed": start_epoch - 1,
            "start_epoch": start_epoch,
        }

    for epoch in range(start_epoch, config.epochs + 1):
        generator.network.train()
        running = {
            "loss": 0.0,
            "increment_mean_huber": 0.0,
            "increment_scale_huber": 0.0,
            "reflectivity_mean_huber": 0.0,
            "reflectivity_scale_huber": 0.0,
            "state_cross_entropy": 0.0,
        }
        epoch_peak_loss = float("-inf")
        epoch_peak_batch = 0
        window_peak_loss = float("-inf")
        window_peak_batch = 0
        batch_count = 0
        for batch_index, source in enumerate(batches(training_batches), start=1):
            batch_count = batch_index
            optimizer.zero_grad(set_to_none=True)
            loss = _forward_loss(
                generator,
                _to_device(source, generator.device),
                config,
            )
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                generator.network.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            current = {
                key: float(loss[key].detach().cpu())
                for key in running
            }
            for key, value in current.items():
                running[key] += value
            if current["loss"] > epoch_peak_loss:
                epoch_peak_loss = current["loss"]
                epoch_peak_batch = batch_index
            if current["loss"] > window_peak_loss:
                window_peak_loss = current["loss"]
                window_peak_batch = batch_index
            if batch_index % config.log_every_batches == 0:
                log.info(
                    "epoch %d/%d | batch %d | current_loss=%.6f | "
                    "mean_loss=%.6f | window_peak_loss=%.6f@%d | "
                    "increment=%.6f | reflectivity=%.6f | state_ce=%.6f",
                    epoch,
                    config.epochs,
                    batch_index,
                    current["loss"],
                    running["loss"] / batch_index,
                    window_peak_loss,
                    window_peak_batch,
                    current["increment_mean_huber"],
                    current["reflectivity_mean_huber"],
                    current["state_cross_entropy"],
                )
                window_peak_loss = float("-inf")
                window_peak_batch = 0
        if batch_count == 0:
            raise ValueError("training received no batches.")
        tuning = _evaluate_losses(generator, batches(tuning_batches), config)
        row = {
            "epoch": float(epoch),
            **{
                f"train_{key}": value / batch_count
                for key, value in running.items()
            },
            "train_peak_loss": epoch_peak_loss,
            "train_peak_batch": float(epoch_peak_batch),
            **{f"tuning_{key}": value for key, value in tuning.items()},
        }
        history.append(row)
        log.info(
            "epoch %d/%d complete | train_loss=%.6f | train_huber=%.6f | "
            "train_reflectivity=%.6f | train_state_ce=%.6f | "
            "peak_loss=%.6f@%d | tuning_loss=%.6f",
            epoch,
            config.epochs,
            row["train_loss"],
            row["train_increment_mean_huber"],
            row["train_reflectivity_mean_huber"],
            row["train_state_cross_entropy"],
            row["train_peak_loss"],
            int(row["train_peak_batch"]),
            row["tuning_loss"],
        )

        is_best = tuning["loss"] < best_loss
        if is_best:
            best_loss = tuning["loss"]
            best_epoch = epoch

        runtime_state = {
            "epoch": epoch,
            "best_tuning_loss": float(best_loss),
            "best_epoch": best_epoch,
            "history": [dict(item) for item in history],
            "optimizer_state": optimizer.state_dict(),
            "rng_state": _capture_rng_state(),
        }
        if checkpoint_callback is not None:
            checkpoint_callback(epoch, generator, runtime_state, is_best)

    return {
        "best_tuning_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "epochs_completed": config.epochs,
        "start_epoch": start_epoch,
    }


__all__ = [
    "CheckpointCallback",
    "CoefficientVarianceCalibrationConfig",
    "LearningConfig",
    "SegmentProfileLearningConfig",
    "TargetAuditConfig",
    "audit_observable_targets",
    "calibrate_coefficient_variance",
    "calibrate_semi_markov_fusion",
    "evaluate_generator",
    "evaluate_segment_profile_head",
    "seed_training_random_streams",
    "train_generator",
    "train_segment_profile_head",
]
