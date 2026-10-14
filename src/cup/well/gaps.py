"""Shared short well-log gap handling for the well workflow.

The workflow keeps the original observation mask while allowing bounded
joint Vp/Rho gaps to be repaired for signal operations.  This module owns the
repair rule so Step 4, Step 5, and later background consumers use the same
definition of a short gap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cup.utils.masks import true_runs
from cup.well.las import StandardVpRhoLogs
from wtie.processing import grid


def _gap_duration_s(
    md: np.ndarray,
    start: int,
    stop: int,
    table: grid.TimeDepthTable,
) -> float:
    """Return the two-way time span between the samples around one gap."""
    left = max(0, start - 1)
    right = min(md.size - 1, stop)
    if not table.is_md_domain:
        raise ValueError("Gap-aware standard LAS processing requires an MD-domain TimeDepthTable.")
    if md[left] < table.md[0] or md[right] > table.md[-1]:
        return float("inf")
    return float(
        np.interp(md[right], table.md, table.twt)
        - np.interp(md[left], table.md, table.twt)
    )


def _standard_arrays(standard: StandardVpRhoLogs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    md = np.asarray(standard.logs.basis, dtype=np.float64).reshape(-1)
    vp = np.asarray(standard.logs.Vp.values, dtype=np.float64).reshape(-1).copy()
    rho = np.asarray(standard.logs.Rho.values, dtype=np.float64).reshape(-1).copy()
    if vp.shape != md.shape or rho.shape != md.shape:
        raise ValueError("Standard Vp/Rho logs must share one MD basis.")
    if md.size < 2 or np.any(~np.isfinite(md)) or np.any(np.diff(md) <= 0.0):
        raise ValueError("Standard Vp/Rho MD basis must be finite and strictly increasing.")
    return md, vp, rho


def _build_repaired_logs(
    md: np.ndarray,
    vp: np.ndarray,
    rho: np.ndarray,
    *,
    observed_mask: np.ndarray,
    filled_gap_mask: np.ndarray,
    gap_report: dict[str, float | int],
) -> grid.LogSet:
    logs = grid.LogSet(
        {
            "Vp": grid.Log(vp, md, "md", name="Vp", unit="m/s", allow_nan=True),
            "Rho": grid.Log(rho, md, "md", name="Rho", unit="g/cm3", allow_nan=True),
        }
    )
    logs.observed_mask = np.asarray(observed_mask, dtype=bool).copy()
    logs.filled_gap_mask = np.asarray(filled_gap_mask, dtype=bool).copy()
    logs.gap_report = dict(gap_report)
    return logs


def fill_short_joint_gaps(
    standard: StandardVpRhoLogs,
    table: grid.TimeDepthTable,
    *,
    max_short_gap_s: float,
) -> tuple[grid.LogSet, np.ndarray]:
    """Repair bounded joint-invalid Vp/Rho runs measured in two-way seconds.

    The returned ``LogSet`` contains repaired values for short internal gaps
    and NaN values for long or unbounded gaps.  ``filled_gap_mask`` marks only
    samples repaired by this call; ``observed_mask`` remains the mask supplied
    by the LAS source.
    """
    md, vp, rho = _standard_arrays(standard)
    if not table.is_md_domain:
        raise ValueError("Gap-aware standard LAS processing requires an MD-domain TimeDepthTable.")
    if not np.isfinite(max_short_gap_s) or max_short_gap_s < 0.0:
        raise ValueError("max_short_gap_s must be finite and non-negative.")
    joint_valid = np.isfinite(vp) & (vp > 0.0) & np.isfinite(rho) & (rho > 0.0)
    filled = np.asarray(standard.filled_gap_mask, dtype=bool).copy()
    long_gaps: list[float] = []
    for start, stop in true_runs(~joint_valid):
        if start == 0 or stop == md.size:
            continue
        duration = _gap_duration_s(md, start, stop, table)
        if duration > float(max_short_gap_s) + 1e-12:
            long_gaps.append(duration)
            continue
        vp[start:stop] = np.interp(md[start:stop], md[[start - 1, stop]], vp[[start - 1, stop]])
        rho[start:stop] = np.interp(md[start:stop], md[[start - 1, stop]], rho[[start - 1, stop]])
        filled[start:stop] = True
    report: dict[str, float | int] = {
        "short_gap_filled_samples": int(np.count_nonzero(filled)),
        "long_gap_count": int(len(long_gaps)),
        "longest_long_gap_s": float(max(long_gaps, default=0.0)),
    }
    logs = _build_repaired_logs(
        md,
        vp,
        rho,
        observed_mask=np.asarray(standard.observed_mask, dtype=bool),
        filled_gap_mask=filled,
        gap_report=report,
    )
    return logs, filled


def fill_short_depth_gaps(
    standard: StandardVpRhoLogs,
    depth_coordinates: Any,
    *,
    max_short_gap_s: float,
) -> tuple[grid.LogSet, np.ndarray]:
    """Repair bounded joint gaps using a depth-coordinate duration.

    ``depth_coordinates`` must be aligned with the standard LAS MD basis.  A
    NaN coordinate is allowed for geometry-clipped invalid samples and leaves
    a gap unrepaired.  A NaN coordinate at a jointly valid log sample is an
    invalid geometry result and raises immediately.
    """
    md, vp, rho = _standard_arrays(standard)
    depth = np.asarray(depth_coordinates, dtype=np.float64).reshape(-1)
    if depth.shape != md.shape:
        raise ValueError("depth_coordinates must match the standard LAS MD basis.")
    if not np.isfinite(max_short_gap_s) or max_short_gap_s < 0.0:
        raise ValueError("max_short_gap_s must be finite and non-negative.")
    joint_valid = np.isfinite(vp) & (vp > 0.0) & np.isfinite(rho) & (rho > 0.0)
    if np.any(joint_valid & ~np.isfinite(depth)):
        raise ValueError("Finite Vp/Rho samples require finite depth_coordinates.")
    filled = np.asarray(standard.filled_gap_mask, dtype=bool).copy()
    long_gaps: list[float] = []
    for start, stop in true_runs(~joint_valid):
        if start == 0 or stop == md.size:
            continue
        left, right = start - 1, stop
        if (
            not np.isfinite(depth[left])
            or not np.isfinite(depth[right])
            or np.any(~np.isfinite(depth[start:stop]))
        ):
            # The gap is outside the geometry support, so there is no bounded
            # depth duration with which to authorize interpolation.
            continue
        vp_left, vp_right = vp[left], vp[right]
        if not (
            np.isfinite(vp_left)
            and vp_left > 0.0
            and np.isfinite(vp_right)
            and vp_right > 0.0
        ):
            long_gaps.append(float("inf"))
            continue
        average_vp = 0.5 * (float(vp_left) + float(vp_right))
        duration_s = 2.0 * abs(float(depth[right] - depth[left])) / average_vp
        if duration_s <= float(max_short_gap_s) + 1e-12:
            vp[start:stop] = np.interp(md[start:stop], md[[left, right]], vp[[left, right]])
            rho[start:stop] = np.interp(md[start:stop], md[[left, right]], rho[[left, right]])
            filled[start:stop] = True
        else:
            long_gaps.append(duration_s)
    report: dict[str, float | int] = {
        "short_gap_filled_samples": int(np.count_nonzero(filled)),
        "long_gap_count": int(len(long_gaps)),
        "longest_long_gap_s": float(max(long_gaps, default=0.0)),
    }
    logs = _build_repaired_logs(
        md,
        vp,
        rho,
        observed_mask=np.asarray(standard.observed_mask, dtype=bool),
        filled_gap_mask=filled,
        gap_report=report,
    )
    return logs, filled


__all__ = ["fill_short_joint_gaps", "fill_short_depth_gaps"]
