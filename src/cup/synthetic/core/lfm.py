"""Canonical low-frequency background products for Synthoseis v5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cup.impedance import decompose_log_ai, validate_increment_contract
from cup.synthetic.core.random import RandomNamespace
from cup.synthetic.core.rejections import BenchmarkBuildRejected
from cup.utils.statistics import centered_rms


@dataclass(frozen=True)
class LfmPolicy:
    """Inputs needed to construct the canonical background.

    LFM is deliberately not a variant selector. The canonical increment
    contract is the sole source of the background and target decomposition.
    """

    sample_domain: str
    axis_unit: str
    global_seed: int
    random_namespace: RandomNamespace
    realization_id: str
    horizon_coordinates: np.ndarray
    zone_id_model: np.ndarray | None = None


@dataclass(frozen=True)
class LfmProducts:
    canonical_background_log_ai: np.ndarray
    target_increment_log_ai: np.ndarray
    qc: Mapping[str, Any]


def _canonical_segment_diagnostics(
    valid_mask: np.ndarray, *, minimum_samples: int
) -> dict[str, int] | None:
    short_lengths: list[int] = []
    affected_traces = 0
    for row in np.asarray(valid_mask, dtype=bool).reshape(-1, valid_mask.shape[-1]):
        padded = np.concatenate(([False], row, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        stops = np.flatnonzero(padded[:-1] & ~padded[1:])
        lengths = stops - starts
        short = lengths[lengths < minimum_samples]
        if short.size:
            affected_traces += 1
            short_lengths.extend(int(value) for value in short)
    if not short_lengths:
        return None
    return {
        "minimum_segment_samples": int(minimum_samples),
        "shortest_segment_samples": min(short_lengths),
        "short_segment_count": len(short_lengths),
        "affected_trace_count": affected_traces,
    }


def build_lfm_products(
    target_log_ai: np.ndarray,
    sample_axis: np.ndarray,
    canonical_contract: object,
    *,
    lateral_coordinates: np.ndarray,
    valid_mask: np.ndarray,
    policy: LfmPolicy,
) -> LfmProducts:
    """Build canonical background and increment without a degraded LFM."""

    domain = str(policy.sample_domain).casefold()
    expected_unit = {"time": "s", "depth": "m"}.get(domain)
    if expected_unit != str(policy.axis_unit):
        raise ValueError("canonical LFM domain/unit must be time/s or depth/m")
    target = np.asarray(target_log_ai, dtype=np.float64)
    axis = np.asarray(sample_axis, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool)
    lateral = np.asarray(lateral_coordinates, dtype=np.float64).reshape(-1)
    if target.ndim != 2 or valid.shape != target.shape:
        raise ValueError("canonical LFM target and mask must share a 2-D shape")
    if target.shape != (lateral.size, axis.size):
        raise ValueError("canonical LFM axes do not match target shape")
    if np.any(valid & ~np.isfinite(target)):
        raise ValueError("canonical LFM target has non-finite valid samples")

    resolved_contract = validate_increment_contract(canonical_contract)
    # The canonical background is an operator on the complete model context,
    # not on the public target ROI.  The forward adapters deliberately prepare
    # context above and below the interpreted interval so that both the
    # canonical low-pass and the seismic forward operator have their required
    # support.  Applying ``valid`` here would discard that context and reject
    # structurally thin target intervals even when the complete target trace is
    # perfectly filterable.
    filter_support = np.isfinite(target)
    short_segment_diagnostics = _canonical_segment_diagnostics(
        filter_support, minimum_samples=resolved_contract.minimum_segment_samples
    )
    if short_segment_diagnostics is not None:
        reason = "canonical_lfm_segment_too_short"
        raise BenchmarkBuildRejected(
            [reason],
            diagnostics=short_segment_diagnostics,
            details=[{"reason": reason, **short_segment_diagnostics}],
        )

    background, increment = decompose_log_ai(
        target,
        axis,
        resolved_contract,
    )
    background = np.asarray(background, dtype=np.float64)
    increment = np.asarray(increment, dtype=np.float64)
    if background.shape != target.shape or increment.shape != target.shape:
        raise ValueError("canonical LFM decomposition changed target shape")
    if np.any(valid & (~np.isfinite(background) | ~np.isfinite(increment))):
        raise ValueError("canonical LFM decomposition has non-finite valid samples")
    if np.any(valid & (np.abs(target - background - increment) > 1e-7)):
        raise ValueError("canonical LFM decomposition violates target closure")
    background[~valid] = np.nan
    increment[~valid] = np.nan
    return LfmProducts(
        canonical_background_log_ai=background,
        target_increment_log_ai=increment,
        qc={
            "lfm_status": "canonical",
            "lfm_filter_support_policy": (
                "complete_finite_model_context_then_public_mask"
            ),
            "lfm_filter_support_sample_count": int(
                np.count_nonzero(filter_support)
            ),
            "lfm_valid_sample_count": int(np.count_nonzero(valid)),
            "lfm_canonical_background_rms": centered_rms(background, valid),
            "lfm_target_increment_rms": centered_rms(increment, valid),
        },
    )


__all__ = ["LfmPolicy", "LfmProducts", "build_lfm_products"]
