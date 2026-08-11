"""LFM anchoring and supervised model-grid target construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cup.physics.numpy_backend import reflectivity_from_log_ai
from evidence.contracts import EvidenceInput, InputContractError


@dataclass(frozen=True)
class LfmAnchor:
    values: np.ndarray
    support: np.ndarray
    intercept: np.ndarray
    slope: np.ndarray


@dataclass(frozen=True)
class EvidenceTargets:
    projected_log_ai_increment: np.ndarray
    signed_reflectivity: np.ndarray
    support: np.ndarray


def common_evidence_support(base_support: np.ndarray) -> np.ndarray:
    """Require both samples that define a lower-interface reflectivity."""

    base = np.asarray(base_support, dtype=bool)
    if base.ndim != 2:
        raise InputContractError("base evidence support must be two-dimensional.")
    support = np.zeros_like(base)
    support[:, 1:] = base[:, 1:] & base[:, :-1]
    return support


def _zone_coordinate(axis: np.ndarray, top: float, bottom: float) -> np.ndarray:
    return 2.0 * (np.asarray(axis, dtype=np.float64) - float(top)) / (
        float(bottom) - float(top)
    ) - 1.0


def build_lfm_anchor(observation: EvidenceInput) -> LfmAnchor:
    """Fit one zone-linear LFM anchor per lateral trace."""

    axis = observation.sample_axis.coordinates
    values = np.full(observation.lfm.shape, np.nan, dtype=np.float64)
    support = np.zeros_like(observation.observed_valid)
    intercept = np.full(observation.width, np.nan, dtype=np.float64)
    slope = np.full(observation.width, np.nan, dtype=np.float64)
    for trace in range(observation.width):
        if not observation.lateral_valid[trace]:
            continue
        top = float(observation.zone_top[trace])
        bottom = float(observation.zone_bottom[trace])
        inside = (
            observation.observed_valid[trace]
            & (axis >= top)
            & (axis <= bottom)
        )
        if np.count_nonzero(inside) < 2:
            raise InputContractError(
                f"trace {trace} has fewer than two valid LFM samples inside the zone."
            )
        coordinate = _zone_coordinate(axis[inside], top, bottom)
        design = np.column_stack((np.ones(coordinate.size), coordinate))
        coefficients, _, rank, _ = np.linalg.lstsq(
            design,
            observation.lfm[trace, inside],
            rcond=None,
        )
        if rank != 2 or np.any(~np.isfinite(coefficients)):
            raise InputContractError(f"trace {trace} LFM anchor is rank deficient.")
        intercept[trace], slope[trace] = coefficients
        axis_coordinate = _zone_coordinate(axis, top, bottom)
        zone = (axis >= top) & (axis <= bottom)
        values[trace, zone] = coefficients[0] + coefficients[1] * axis_coordinate[zone]
        support[trace] = zone & observation.observed_valid[trace]
    return LfmAnchor(
        values=values,
        support=support,
        intercept=intercept,
        slope=slope,
    )


def lfm_residual_from_anchor(
    observation: EvidenceInput,
    anchor: LfmAnchor,
) -> np.ndarray:
    if anchor.values.shape != observation.lfm.shape or anchor.support.shape != observation.lfm.shape:
        raise InputContractError("LFM anchor shape differs from the observation.")
    support = anchor.support & observation.observed_valid
    residual = observation.lfm - anchor.values
    if np.any(~np.isfinite(residual[support])):
        raise InputContractError("LFM residual contains non-finite supported samples.")
    return np.where(support, residual, 0.0)


def build_evidence_targets(
    observation: EvidenceInput,
    *,
    model_log_ai: np.ndarray,
    anchor: LfmAnchor,
) -> EvidenceTargets:
    """Build the two continuous targets on the observation sample axis."""

    log_ai = np.asarray(model_log_ai, dtype=np.float64)
    if log_ai.shape != observation.seismic.shape:
        raise InputContractError("model_log_ai must match the observation.")
    base_support = (
        anchor.support
        & observation.observed_valid
        & observation.lateral_valid[:, None]
        & np.isfinite(log_ai)
        & np.isfinite(anchor.values)
    )
    increment = np.where(base_support, log_ai - anchor.values, 0.0)
    reflectivity = np.zeros_like(log_ai)
    reflectivity[:, 1:] = reflectivity_from_log_ai(log_ai)
    support = common_evidence_support(base_support)
    return EvidenceTargets(
        projected_log_ai_increment=np.where(support, increment, 0.0),
        signed_reflectivity=np.where(support, reflectivity, 0.0),
        support=support,
    )


__all__ = [
    "EvidenceTargets",
    "LfmAnchor",
    "build_evidence_targets",
    "build_lfm_anchor",
    "common_evidence_support",
    "lfm_residual_from_anchor",
]
