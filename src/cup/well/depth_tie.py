"""Depth-domain Step 4 preparation for candidate time-wavelet extraction.

This module owns the temporary TVDSS-to-relative-TWT adaptation used to feed
the unchanged ``wtie.tie_v1`` extractor.  The resulting local table is not a
geological time-depth table and must not be propagated as a calibration
product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cup.utils.masks import true_runs
from wtie.processing import grid


@dataclass(frozen=True)
class DepthExtractionInputs:
    """Prepared inputs and audit metadata for one vertical depth-domain well."""

    logset_md: grid.LogSet
    seismic_depth: grid.Seismic
    seismic_pseudo_twt: grid.Seismic
    relative_tdt: grid.TimeDepthTable
    relative_tdt_rows: pd.DataFrame
    report: dict[str, Any]


def _joint_valid_runs(logset_md: grid.LogSet) -> list[tuple[int, int]]:
    vp = np.asarray(logset_md.Vp.values, dtype=np.float64)
    rho = np.asarray(logset_md.Rho.values, dtype=np.float64)
    valid = np.isfinite(vp) & (vp > 0.0) & np.isfinite(rho) & (rho > 0.0)
    return [(int(start), int(stop)) for start, stop in true_runs(valid)]


def _crop_logset(logset_md: grid.LogSet, start: int, stop: int) -> grid.LogSet:
    md = np.asarray(logset_md.basis[start:stop], dtype=np.float64)
    vp = np.asarray(logset_md.Vp.values[start:stop], dtype=np.float64)
    rho = np.asarray(logset_md.Rho.values[start:stop], dtype=np.float64)
    if md.size < 2:
        raise ValueError("Depth extraction log interval contains fewer than two samples.")
    return grid.LogSet(
        {
            "Vp": grid.Log(vp, md, "md", name="Vp", unit="m/s", allow_nan=False),
            "Rho": grid.Log(rho, md, "md", name="Rho", unit="g/cm3", allow_nan=False),
        }
    )


def _longest_joint_valid_overlap(
    logset_md: grid.LogSet,
    *,
    kb_m: float,
    seismic_tvdss_m: np.ndarray,
    min_log_samples: int,
) -> tuple[int, int, float, float]:
    md = np.asarray(logset_md.basis, dtype=np.float64)
    candidates: list[tuple[int, float, int, int, float, float]] = []
    for start, stop in _joint_valid_runs(logset_md):
        if stop - start < int(min_log_samples):
            continue
        run_top = float(md[start] - kb_m)
        run_bottom = float(md[stop - 1] - kb_m)
        overlap_top = max(run_top, float(seismic_tvdss_m[0]))
        overlap_bottom = min(run_bottom, float(seismic_tvdss_m[-1]))
        if overlap_bottom <= overlap_top:
            continue
        seismic_count = int(np.count_nonzero((seismic_tvdss_m >= overlap_top) & (seismic_tvdss_m <= overlap_bottom)))
        candidates.append((seismic_count, overlap_bottom - overlap_top, start, stop, overlap_top, overlap_bottom))
    if not candidates:
        raise ValueError("No continuous joint-valid Vp/Rho interval overlaps the depth seismic window.")
    _, _, start, stop, overlap_top, overlap_bottom = max(candidates)
    return start, stop, overlap_top, overlap_bottom


def _interpolate_internal_log_gaps(
    logset_md: grid.LogSet,
    *,
    kb_m: float,
    seismic_tvdss_m: np.ndarray,
) -> tuple[grid.LogSet, float, float, int, int]:
    """Fill internal Vp/Rho gaps without extrapolating beyond measured support."""
    md = np.asarray(logset_md.basis, dtype=np.float64)
    vp = np.asarray(logset_md.Vp.values, dtype=np.float64)
    rho = np.asarray(logset_md.Rho.values, dtype=np.float64)
    vp_valid = np.isfinite(vp) & (vp > 0.0)
    rho_valid = np.isfinite(rho) & (rho > 0.0)
    if int(np.count_nonzero(vp_valid)) < 2 or int(np.count_nonzero(rho_valid)) < 2:
        raise ValueError("Vp and Rho each require at least two positive finite samples for gap interpolation.")

    # Only the common measured support may be interpolated.  np.interp would
    # otherwise silently hold endpoint values outside one curve's coverage.
    support_top_md = max(float(md[vp_valid][0]), float(md[rho_valid][0]))
    support_bottom_md = min(float(md[vp_valid][-1]), float(md[rho_valid][-1]))
    if support_bottom_md <= support_top_md:
        raise ValueError("Vp and Rho have no common measured MD support.")
    support_mask = (md >= support_top_md) & (md <= support_bottom_md)
    support_indices = np.flatnonzero(support_mask)
    if support_indices.size < 2:
        raise ValueError("Vp/Rho common measured support contains fewer than two MD samples.")

    support_md = md[support_mask]
    support_vp = np.interp(support_md, md[vp_valid], vp[vp_valid])
    support_rho = np.interp(support_md, md[rho_valid], rho[rho_valid])
    vp_filled = int(np.count_nonzero(~vp_valid[support_mask]))
    rho_filled = int(np.count_nonzero(~rho_valid[support_mask]))
    interpolated = grid.LogSet(
        {
            "Vp": grid.Log(support_vp, support_md, "md", name="Vp", unit="m/s", allow_nan=False),
            "Rho": grid.Log(support_rho, support_md, "md", name="Rho", unit="g/cm3", allow_nan=False),
        }
    )
    run_top = float(support_md[0] - kb_m)
    run_bottom = float(support_md[-1] - kb_m)
    overlap_top = max(run_top, float(seismic_tvdss_m[0]))
    overlap_bottom = min(run_bottom, float(seismic_tvdss_m[-1]))
    if overlap_bottom <= overlap_top:
        raise ValueError("Interpolated Vp/Rho support does not overlap the depth seismic window.")
    return interpolated, overlap_top, overlap_bottom, vp_filled, rho_filled


def depth_trace_to_relative_twt(
    seismic_depth: grid.Seismic,
    relative_tdt_rows: pd.DataFrame,
    *,
    dt_s: float,
    min_samples: int,
) -> grid.Seismic:
    """Project a finite TVDSS seismic trace onto a regular relative-TWT axis."""
    if not seismic_depth.is_tvdss:
        raise ValueError("Depth extraction requires seismic with TVDSS basis.")
    depth = np.asarray(seismic_depth.basis, dtype=np.float64)
    amplitude = np.asarray(seismic_depth.values, dtype=np.float64)
    if depth.size < 2 or np.any(~np.isfinite(depth)) or np.any(np.diff(depth) <= 0.0):
        raise ValueError("Depth seismic TVDSS axis must be finite and strictly increasing.")
    if np.any(~np.isfinite(amplitude)):
        raise ValueError("Depth seismic contains non-finite amplitudes; Step 4 does not fill them.")
    dt = float(dt_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Pseudo-TWT sampling interval must be finite and positive.")

    tdt_depth = relative_tdt_rows["tvdss_m"].to_numpy(dtype=np.float64)
    tdt_twt = relative_tdt_rows["twt_s"].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(tdt_depth)) or np.any(np.diff(tdt_depth) <= 0.0):
        raise ValueError("Relative TWT mapping TVDSS axis must be finite and strictly increasing.")
    if np.any(~np.isfinite(tdt_twt)) or np.any(np.diff(tdt_twt) <= 0.0):
        raise ValueError("Relative TWT mapping must be finite and strictly increasing.")
    if depth[0] < tdt_depth[0] or depth[-1] > tdt_depth[-1]:
        raise ValueError("Depth seismic window must be contained by the relative TWT mapping.")

    irregular_twt = np.interp(depth, tdt_depth, tdt_twt)
    sample_count = int(np.floor((irregular_twt[-1] - irregular_twt[0]) / dt + 1e-12)) + 1
    if sample_count < int(min_samples):
        raise ValueError(
            f"Pseudo-TWT extraction window has too few samples: {sample_count} < {int(min_samples)}."
        )
    regular_twt = irregular_twt[0] + np.arange(sample_count, dtype=np.float64) * dt
    amplitude_twt = np.interp(regular_twt, irregular_twt, amplitude)
    return grid.Seismic(amplitude_twt, regular_twt, "twt", name="Depth seismic projected to relative TWT")


def prepare_vertical_depth_extraction(
    logset_md: grid.LogSet,
    seismic_depth: grid.Seismic,
    *,
    kb_m: float,
    pseudo_dt_s: float,
    min_tie_samples: int,
    min_log_samples: int = 10,
    log_gap_policy: str = "strict_contiguous",
) -> DepthExtractionInputs:
    """Prepare one vertical well for pseudo-time candidate-wavelet extraction."""
    if not logset_md.is_md:
        raise ValueError("Vertical depth extraction requires MD-domain Vp/Rho logs.")
    kb = float(kb_m)
    if not np.isfinite(kb):
        raise ValueError("kb_m must be finite for TVDSS = MD - KB.")
    if not seismic_depth.is_tvdss:
        raise ValueError("Vertical depth extraction requires TVDSS seismic.")

    seismic_depth_axis = np.asarray(seismic_depth.basis, dtype=np.float64)
    gap_policy = str(log_gap_policy).strip().casefold()
    if gap_policy == "strict_contiguous":
        start, stop, overlap_top, overlap_bottom = _longest_joint_valid_overlap(
            logset_md,
            kb_m=kb,
            seismic_tvdss_m=seismic_depth_axis,
            min_log_samples=min_log_samples,
        )
        extraction_logs = _crop_logset(logset_md, start, stop)
        vp_filled = 0
        rho_filled = 0
    elif gap_policy == "interpolate_internal":
        extraction_logs, overlap_top, overlap_bottom, vp_filled, rho_filled = _interpolate_internal_log_gaps(
            logset_md,
            kb_m=kb,
            seismic_tvdss_m=seismic_depth_axis,
        )
    else:
        raise ValueError(
            "Depth extraction log_gap_policy must be 'strict_contiguous' or 'interpolate_internal'."
        )
    log_tvdss = np.asarray(extraction_logs.basis, dtype=np.float64) - kb

    seismic_mask = (seismic_depth_axis >= overlap_top) & (seismic_depth_axis <= overlap_bottom)
    if int(np.count_nonzero(seismic_mask)) < 2:
        raise ValueError("Depth extraction overlap contains fewer than two seismic samples.")
    cropped_seismic = grid.Seismic(
        np.asarray(seismic_depth.values, dtype=np.float64)[seismic_mask],
        seismic_depth_axis[seismic_mask],
        "tvdss",
        name=seismic_depth.name,
    )

    relative_rows = grid.build_local_tdt_from_vp(
        tvdss_m=log_tvdss,
        vp_mps=np.asarray(extraction_logs.Vp.values, dtype=np.float64),
        md_m=np.asarray(extraction_logs.basis, dtype=np.float64),
        origin_twt_s=0.0,
        method="slowness_trapezoid",
    )
    relative_table = grid.TimeDepthTable(
        twt=relative_rows["twt_s"].to_numpy(dtype=np.float64),
        md=relative_rows["md_m"].to_numpy(dtype=np.float64),
    )
    seismic_twt = depth_trace_to_relative_twt(
        cropped_seismic,
        relative_rows,
        dt_s=pseudo_dt_s,
        min_samples=min_tie_samples,
    )
    report = {
        "sample_domain": "depth",
        "depth_basis": "tvdss",
        "geological_tdt": False,
        "tvdss_start_m": float(cropped_seismic.basis[0]),
        "tvdss_end_m": float(cropped_seismic.basis[-1]),
        "depth_sample_count": int(cropped_seismic.size),
        "relative_twt_start_s": float(seismic_twt.basis[0]),
        "relative_twt_end_s": float(seismic_twt.basis[-1]),
        "pseudo_twt_sample_count": int(seismic_twt.size),
        "pseudo_twt_dt_s": float(pseudo_dt_s),
        "log_gap_policy": gap_policy,
        "interpolated_vp_sample_count": vp_filled,
        "interpolated_rho_sample_count": rho_filled,
        "continuous_log_md_start_m": float(extraction_logs.basis[0]),
        "continuous_log_md_end_m": float(extraction_logs.basis[-1]),
        "continuous_log_sample_count": int(extraction_logs.basis.size),
    }
    return DepthExtractionInputs(
        logset_md=extraction_logs,
        seismic_depth=cropped_seismic,
        seismic_pseudo_twt=seismic_twt,
        relative_tdt=relative_table,
        relative_tdt_rows=relative_rows,
        report=report,
    )
