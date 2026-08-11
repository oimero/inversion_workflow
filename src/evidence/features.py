"""LFM anchoring and supervised model-grid target construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cup.physics.numpy_backend import reflectivity_from_log_ai
from cup.synthetic.core.records import SampleAxis

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
    state_id: np.ndarray
    support: np.ndarray


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
    state_highres: np.ndarray,
    highres_axis: SampleAxis,
    anchor: LfmAnchor,
) -> EvidenceTargets:
    """Build the three supervised targets on the observation sample axis."""

    log_ai = np.asarray(model_log_ai, dtype=np.float64)
    state = np.asarray(state_highres)
    if log_ai.shape != observation.seismic.shape:
        raise InputContractError("model_log_ai must match the observation.")
    if not isinstance(highres_axis, SampleAxis):
        raise TypeError("highres_axis must be a SampleAxis object.")
    if (
        highres_axis.sample_domain != observation.sample_axis.sample_domain
        or highres_axis.unit != observation.sample_axis.unit
        or highres_axis.depth_basis != observation.sample_axis.depth_basis
    ):
        raise InputContractError("target axes must share domain, unit, and depth basis.")
    expected_highres = (observation.width, highres_axis.coordinates.size)
    if state.shape != expected_highres:
        raise InputContractError("state_highres must match the high-resolution axis.")
    ratio = observation.sample_axis.sample_interval / highres_axis.sample_interval
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1.0e-10):
        raise InputContractError("target axes must be integer nested.")
    nested = highres_axis.coordinates[::factor]
    if nested.shape != observation.sample_axis.coordinates.shape or not np.allclose(
        nested,
        observation.sample_axis.coordinates,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise InputContractError("high-resolution samples do not nest on the model axis.")
    state_model = state[:, ::factor]
    base_support = (
        anchor.support
        & observation.observed_valid
        & observation.lateral_valid[:, None]
        & np.isfinite(log_ai)
        & np.isfinite(anchor.values)
        & (state_model >= 0)
        & (state_model <= 2)
    )
    increment = np.where(base_support, log_ai - anchor.values, 0.0)
    reflectivity = np.zeros_like(log_ai)
    reflectivity[:, 1:] = reflectivity_from_log_ai(log_ai)
    interface_support = base_support.copy()
    interface_support[:, 0] = False
    interface_support[:, 1:] &= base_support[:, :-1]
    support = base_support & interface_support
    return EvidenceTargets(
        projected_log_ai_increment=np.where(support, increment, 0.0),
        signed_reflectivity=np.where(support, reflectivity, 0.0),
        state_id=np.where(support, state_model, -1).astype(np.int64),
        support=support,
    )


__all__ = [
    "EvidenceTargets",
    "LfmAnchor",
    "build_evidence_targets",
    "build_lfm_anchor",
    "lfm_residual_from_anchor",
]
