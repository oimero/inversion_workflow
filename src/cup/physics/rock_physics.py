"""Domain-independent rock-physics relations and robust multi-well fitting.

This module contains numerical implementation only.  It does not discover
workflow runs, read LAS files, infer units, fill missing values, or write
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cup.physics.calibration import AIVelocityRelation


@dataclass(frozen=True)
class WellAiVpSamples:
    """Finite positive AI--Vp samples belonging to one well."""

    well_id: str
    vp_mps: np.ndarray
    ai_mps_gcc: np.ndarray

    def __post_init__(self) -> None:
        well_id = str(self.well_id).strip()
        if not well_id:
            raise ValueError("well_id must be non-empty.")
        vp = np.asarray(self.vp_mps, dtype=np.float64)
        ai = np.asarray(self.ai_mps_gcc, dtype=np.float64)
        if vp.ndim != 1 or ai.ndim != 1 or vp.shape != ai.shape:
            raise ValueError("vp_mps and ai_mps_gcc must be matching one-dimensional arrays.")
        if vp.size < 2:
            raise ValueError("Each well needs at least two AI--Vp samples.")
        if np.any(~np.isfinite(vp)) or np.any(~np.isfinite(ai)):
            raise ValueError("AI--Vp fitting samples must be finite.")
        if np.any(vp <= 0.0) or np.any(ai <= 0.0):
            raise ValueError("AI--Vp fitting samples must be positive.")
        object.__setattr__(self, "well_id", well_id)
        object.__setattr__(self, "vp_mps", vp)
        object.__setattr__(self, "ai_mps_gcc", ai)


@dataclass(frozen=True)
class EqualWellHuberFit:
    """Result of one deterministic equal-well-weight Huber fit."""

    relation: AIVelocityRelation
    converged: bool
    iterations: int
    robust_scale_mps_gcc: float
    huber_delta_sigma: float
    objective: float
    initial_a: float
    initial_b: float
    well_base_weights: Mapping[str, float]
    well_effective_weights: Mapping[str, float]
    aggregate_qc: Mapping[str, float]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def _weighted_linear_fit(vp: np.ndarray, ai: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((vp, np.ones_like(vp)))
    sqrt_weight = np.sqrt(weights)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * sqrt_weight[:, None],
        ai * sqrt_weight,
        rcond=None,
    )
    if rank != 2 or np.any(~np.isfinite(coefficients)):
        raise ValueError("AI--Vp weighted linear system is rank deficient.")
    return float(coefficients[0]), float(coefficients[1])


def _robust_scale(residual: np.ndarray, weights: np.ndarray) -> float:
    center = _weighted_median(residual, weights)
    mad = _weighted_median(np.abs(residual - center), weights)
    return float(1.4826 * mad)


def _weighted_metrics(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    normalized = weights / np.sum(weights)
    residual = actual - predicted
    mean = float(np.sum(normalized * actual))
    ss_res = float(np.sum(normalized * residual**2))
    ss_tot = float(np.sum(normalized * (actual - mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return {
        "r2": r2,
        "rmse_mps_gcc": float(np.sqrt(ss_res)),
        "mae_mps_gcc": float(np.sum(normalized * np.abs(residual))),
        "bias_mps_gcc": float(np.sum(normalized * residual)),
    }


def fit_equal_well_huber(
    wells: Mapping[str, WellAiVpSamples],
    *,
    huber_delta_sigma: float = 1.345,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> EqualWellHuberFit:
    """Fit ``AI = a*Vp+b`` with equal well mass and Huber IRLS.

    Each well receives base weight ``1 / n_wells`` and divides that mass
    equally among its samples.  Huber weights only downweight residual
    outliers; they never change the base sampling contract.
    """
    if not isinstance(wells, Mapping) or not wells:
        raise ValueError("wells must contain at least one well.")
    if not np.isfinite(huber_delta_sigma) or huber_delta_sigma <= 0.0:
        raise ValueError("huber_delta_sigma must be finite and positive.")
    if isinstance(max_iterations, bool) or int(max_iterations) <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")

    ordered = []
    for key in sorted(wells, key=lambda value: str(value).casefold()):
        sample = wells[key]
        if not isinstance(sample, WellAiVpSamples):
            raise TypeError("wells values must be WellAiVpSamples.")
        if sample.well_id != str(key):
            raise ValueError(f"Well mapping key {key!r} does not match sample well_id {sample.well_id!r}.")
        ordered.append(sample)
    normalized_ids = [sample.well_id.casefold() for sample in ordered]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("Well identifiers must be unique case-insensitively.")

    n_wells = len(ordered)
    vp = np.concatenate([sample.vp_mps for sample in ordered])
    ai = np.concatenate([sample.ai_mps_gcc for sample in ordered])
    slices: dict[str, slice] = {}
    base_parts = []
    start = 0
    for sample in ordered:
        stop = start + sample.vp_mps.size
        slices[sample.well_id] = slice(start, stop)
        base_parts.append(np.full(sample.vp_mps.size, 1.0 / (n_wells * sample.vp_mps.size)))
        start = stop
    base_weights = np.concatenate(base_parts)

    initial_a, initial_b = _weighted_linear_fit(vp, ai, base_weights)
    a, b = initial_a, initial_b
    converged = False
    effective_weights = base_weights.copy()
    scale = float("nan")
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        residual = ai - (a * vp + b)
        scale = _robust_scale(residual, base_weights)
        if not np.isfinite(scale):
            raise ValueError("AI--Vp weighted MAD scale is not finite.")
        if scale == 0.0:
            if np.max(np.abs(residual)) <= 1e-12:
                converged = True
                effective_weights = base_weights.copy()
                break
            raise ValueError("AI--Vp weighted MAD scale is zero for a non-exact fit.")
        cutoff = float(huber_delta_sigma * scale)
        absolute = np.abs(residual)
        robust_weights = np.ones_like(absolute)
        outside = absolute > cutoff
        robust_weights[outside] = cutoff / absolute[outside]
        effective_weights = base_weights * robust_weights
        new_a, new_b = _weighted_linear_fit(vp, ai, effective_weights)
        change = np.linalg.norm([new_a - a, new_b - b])
        reference = 1.0 + np.linalg.norm([a, b])
        a, b = new_a, new_b
        if change <= tolerance * reference:
            converged = True
            break
    if not converged:
        raise ValueError(f"AI--Vp Huber fit did not converge within {max_iterations} iterations.")

    relation = AIVelocityRelation(a=a, b=b)
    inverse_velocity = relation.velocity_from_ai(ai)
    if np.any(~np.isfinite(inverse_velocity)) or np.any(inverse_velocity <= 0.0):
        raise ValueError("Fitted AI--Vp relation produces non-positive inverse velocities.")

    predicted = relation.ai_from_velocity(vp)
    aggregate_qc = _weighted_metrics(ai, predicted, base_weights)
    scale = _robust_scale(ai - predicted, base_weights)
    cutoff = float(huber_delta_sigma * scale) if scale > 0.0 else 0.0
    residual = ai - predicted
    if cutoff > 0.0:
        absolute = np.abs(residual)
        robust = np.ones_like(absolute)
        outside = absolute > cutoff
        robust[outside] = cutoff / absolute[outside]
        effective_weights = base_weights * robust
        scaled = absolute / scale
        loss = np.where(
            scaled <= huber_delta_sigma,
            0.5 * scaled**2,
            huber_delta_sigma * (scaled - 0.5 * huber_delta_sigma),
        )
        objective = float(np.sum(base_weights * loss))
    else:
        effective_weights = base_weights.copy()
        objective = 0.0

    return EqualWellHuberFit(
        relation=relation,
        converged=True,
        iterations=iterations,
        robust_scale_mps_gcc=scale,
        huber_delta_sigma=float(huber_delta_sigma),
        objective=objective,
        initial_a=initial_a,
        initial_b=initial_b,
        well_base_weights={well_id: float(np.sum(base_weights[span])) for well_id, span in slices.items()},
        well_effective_weights={well_id: float(np.sum(effective_weights[span])) for well_id, span in slices.items()},
        aggregate_qc=aggregate_qc,
    )


def well_fit_metrics(sample: WellAiVpSamples, relation: AIVelocityRelation) -> dict[str, float]:
    """Return unweighted per-well QC metrics for one frozen relation."""
    weights = np.full(sample.vp_mps.size, 1.0 / sample.vp_mps.size)
    predicted = relation.ai_from_velocity(sample.vp_mps)
    return _weighted_metrics(sample.ai_mps_gcc, predicted, weights)


__all__ = [
    "EqualWellHuberFit",
    "WellAiVpSamples",
    "fit_equal_well_huber",
    "well_fit_metrics",
]
