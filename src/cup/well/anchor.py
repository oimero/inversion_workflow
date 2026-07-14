"""Variant-specific real-well labels built from canonical Step 6 controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cup.impedance import decompose_log_ai, generation_contract, validate_sample_axis
from cup.utils.io import require_contract_fingerprint
from cup.utils.statistics import radius_connected_components
from cup.well.real_field_controls import load_well_control_set


SCHEMA_VERSION = "real_well_supervised_samples_v1"

ANCHOR_SAMPLE_COLUMNS = [
    "well_name",
    "sample_index",
    "sample",
    "sample_domain",
    "sample_unit",
    "inline",
    "xline",
    "x_m",
    "y_m",
    "spatial_cluster_id",
    "spatial_cluster_size",
    "well_log_ai",
    "lfm_log_ai",
    "canonical_background_log_ai",
    "well_target_increment_log_ai",
    "valid_for_fit",
    "valid_reason",
    "sampling_mode",
    "sample_method",
    "wellbore_class",
    "variant_id",
    "lfm_contract_fingerprint_sha256",
    "well_control_contract_fingerprint_sha256",
]


def sample_volume_trilinear(
    volume: np.ndarray,
    *,
    ilines: np.ndarray,
    xlines: np.ndarray,
    twt_s: np.ndarray,
    inline_values: np.ndarray,
    xline_values: np.ndarray,
    sample_twt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an ``[inline, xline, sample]`` volume without extrapolation.

    Historical parameter names are retained because this utility is also used
    outside the unified LFM path; values are domain-neutral sample coordinates.
    """

    data = np.asarray(volume, dtype=np.float64)
    axes = [np.asarray(axis, dtype=np.float64) for axis in (ilines, xlines, twt_s)]
    coords = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in (inline_values, xline_values, sample_twt_s)
    ]
    if data.ndim != 3 or data.shape != tuple(axis.size for axis in axes):
        raise ValueError(f"Volume/axis shape mismatch: volume={data.shape}, axes={[axis.size for axis in axes]}")
    if len({values.size for values in coords}) != 1:
        raise ValueError("Point coordinate arrays must have the same size.")
    for name, axis in zip(("inline", "xline", "sample"), axes):
        if axis.size < 2 or not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"{name} axis must be strictly increasing with at least two samples.")
    fractional = [
        np.interp(values, axis, np.arange(axis.size), left=np.nan, right=np.nan)
        for values, axis in zip(coords, axes)
    ]
    out = np.full(coords[0].shape, np.nan, dtype=np.float64)
    inside = np.ones(coords[0].shape, dtype=bool)
    for values, axis, frac in zip(coords, axes, fractional):
        inside &= np.isfinite(values) & np.isfinite(frac) & (values >= axis[0]) & (values <= axis[-1])
    for point in np.flatnonzero(inside):
        positions = [float(frac[point]) for frac in fractional]
        lower = [min(int(np.floor(value)), data.shape[dim] - 2) for dim, value in enumerate(positions)]
        weights = [value - index for value, index in zip(positions, lower)]
        total = 0.0
        total_weight = 0.0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    weight = (
                        (weights[0] if di else 1.0 - weights[0])
                        * (weights[1] if dj else 1.0 - weights[1])
                        * (weights[2] if dk else 1.0 - weights[2])
                    )
                    if weight <= 0.0:
                        continue
                    value = data[lower[0] + di, lower[1] + dj, lower[2] + dk]
                    if np.isfinite(value):
                        total += weight * float(value)
                        total_weight += weight
        if total_weight > 0.0:
            out[point] = total / total_weight
        else:
            inside[point] = False
    return out, inside & np.isfinite(out)


def build_well_anchor_samples(
    *,
    well_control_run_dir: Path,
    lfm: np.ndarray,
    valid_mask: np.ndarray,
    ilines: np.ndarray,
    xlines: np.ndarray,
    samples: np.ndarray,
    repo_root: Path,
    cluster_radius_m: float,
    variant_id: str,
    lfm_contract_fingerprint_sha256: str,
    expected_well_control_contract_fingerprint_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sample one selected LFM variant and derive canonical well-increment labels."""

    summary_path = well_control_run_dir / "run_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        control_summary = json.load(handle)
    actual_control_fingerprint = require_contract_fingerprint(
        control_summary, label=f"WellControlSet {well_control_run_dir}"
    )
    if actual_control_fingerprint != str(expected_well_control_contract_fingerprint_sha256):
        raise ValueError("Canonical WellControlSet contract identity mismatch.")
    controls = load_well_control_set(well_control_run_dir, repo_root=repo_root)
    if not str(variant_id).strip() or not str(lfm_contract_fingerprint_sha256).strip():
        raise ValueError("variant_id and lfm_contract_fingerprint_sha256 must be explicit.")
    axes = (np.asarray(ilines), np.asarray(xlines), np.asarray(samples))
    if lfm.shape != tuple(axis.size for axis in axes) or valid_mask.shape != lfm.shape:
        raise ValueError("Selected LFM volume/mask does not match explicit axes.")
    selected_samples = np.asarray(samples, dtype=np.float64)
    if selected_samples.size < 2:
        raise ValueError("The real-well target axis must contain at least two samples.")
    increment_contract = generation_contract(
        controls.sample_domain, float(selected_samples[1] - selected_samples[0])
    )
    validate_sample_axis(selected_samples, increment_contract)
    selected_indices = np.searchsorted(controls.sample_axis.values, selected_samples)
    selected_indices = np.clip(selected_indices, 0, controls.sample_axis.values.size - 1)
    if not np.allclose(
        controls.sample_axis.values[selected_indices], selected_samples, rtol=0.0, atol=1e-8
    ):
        raise ValueError("Selected LFM SampleAxis is not an exact subset of the canonical WellControlSet axis.")

    representative = []
    for control in controls.controls:
        valid = control.valid_mask
        representative.append(
            {
                "well_name": control.well_name,
                "x_m": float(np.mean(control.x_m_by_sample[valid])),
                "y_m": float(np.mean(control.y_m_by_sample[valid])),
            }
        )
    cluster_frame = pd.DataFrame.from_records(representative)
    cluster_frame["spatial_cluster_id"] = radius_connected_components(
        cluster_frame[["x_m", "y_m"]].to_numpy(dtype=np.float64), float(cluster_radius_m)
    )
    cluster_frame["spatial_cluster_size"] = cluster_frame.groupby("spatial_cluster_id")["well_name"].transform("count")
    cluster_lookup = cluster_frame.set_index("well_name").to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    well_status = []
    for control in controls.controls:
        control_indices = selected_indices
        control_axis = np.asarray(control.sample_axis.values, dtype=np.float64)[control_indices]
        control_valid = np.asarray(control.valid_mask, dtype=bool)[control_indices]
        control_inline = np.asarray(control.inline_by_sample, dtype=np.float64)[control_indices]
        control_xline = np.asarray(control.xline_by_sample, dtype=np.float64)[control_indices]
        control_x_m = np.asarray(control.x_m_by_sample, dtype=np.float64)[control_indices]
        control_y_m = np.asarray(control.y_m_by_sample, dtype=np.float64)[control_indices]
        control_log_ai = np.asarray(control.log_ai.values, dtype=np.float64)[control_indices]
        sampled_lfm, lfm_inside = sample_volume_trilinear(
            lfm,
            ilines=np.asarray(ilines),
            xlines=np.asarray(xlines),
            twt_s=np.asarray(samples),
            inline_values=control_inline,
            xline_values=control_xline,
            sample_twt_s=control_axis,
        )
        sampled_mask, mask_inside = sample_volume_trilinear(
            np.asarray(valid_mask, dtype=np.float32),
            ilines=np.asarray(ilines),
            xlines=np.asarray(xlines),
            twt_s=np.asarray(samples),
            inline_values=control_inline,
            xline_values=control_xline,
            sample_twt_s=control_axis,
        )
        well_log_ai = control_log_ai
        canonical_background, well_target_increment = decompose_log_ai(
            well_log_ai,
            control_axis,
            increment_contract,
            valid_mask=control_valid,
        )
        valid = (
            control_valid
            & lfm_inside
            & mask_inside
            & (sampled_mask > 0.5)
            & np.isfinite(well_log_ai)
            & np.isfinite(sampled_lfm)
            & np.isfinite(canonical_background)
            & np.isfinite(well_target_increment)
        )
        cluster = cluster_lookup[control.well_name]
        for index in range(selected_samples.size):
            if not control_valid[index]:
                reason = "well_control_invalid"
            elif not lfm_inside[index]:
                reason = "outside_lfm_support"
            elif not mask_inside[index] or not sampled_mask[index] > 0.5:
                reason = "valid_mask_false"
            elif not np.isfinite(sampled_lfm[index]):
                reason = "lfm_log_ai_nonfinite"
            else:
                reason = "ok"
            rows.append(
                {
                    "well_name": control.well_name,
                    "sample_index": int(index),
                    "sample": float(selected_samples[index]),
                    "sample_domain": controls.sample_domain,
                    "sample_unit": controls.sample_unit,
                    "inline": float(control_inline[index]),
                    "xline": float(control_xline[index]),
                    "x_m": float(control_x_m[index]),
                    "y_m": float(control_y_m[index]),
                    "spatial_cluster_id": int(cluster["spatial_cluster_id"]),
                    "spatial_cluster_size": int(cluster["spatial_cluster_size"]),
                    "well_log_ai": float(well_log_ai[index]) if np.isfinite(well_log_ai[index]) else np.nan,
                    "lfm_log_ai": float(sampled_lfm[index]) if np.isfinite(sampled_lfm[index]) else np.nan,
                    "canonical_background_log_ai": (
                        float(canonical_background[index]) if np.isfinite(canonical_background[index]) else np.nan
                    ),
                    "well_target_increment_log_ai": (
                        float(well_target_increment[index]) if valid[index] else np.nan
                    ),
                    "valid_for_fit": bool(valid[index]),
                    "valid_reason": reason,
                    "sampling_mode": control.sampling_mode,
                    "sample_method": "volume_trilinear",
                    "wellbore_class": control.wellbore_class,
                    "variant_id": str(variant_id),
                    "lfm_contract_fingerprint_sha256": str(lfm_contract_fingerprint_sha256),
                    "well_control_contract_fingerprint_sha256": actual_control_fingerprint,
                }
            )
        well_status.append(
            {
                "well_name": control.well_name,
                "n_samples": int(selected_samples.size),
                "n_valid": int(np.count_nonzero(valid)),
                "sampling_mode": control.sampling_mode,
            }
        )
    frame = pd.DataFrame.from_records(rows, columns=ANCHOR_SAMPLE_COLUMNS)
    if frame.empty or not frame["valid_for_fit"].any():
        raise ValueError("Variant-specific real-delta label builder produced no valid samples.")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "variant_id": str(variant_id),
        "lfm_contract_fingerprint_sha256": str(lfm_contract_fingerprint_sha256),
        "well_control_contract_fingerprint_sha256": actual_control_fingerprint,
        "sample_domain": controls.sample_domain,
        "sample_unit": controls.sample_unit,
        "n_wells": int(frame["well_name"].nunique()),
        "n_clusters": int(frame["spatial_cluster_id"].nunique()),
        "n_samples": int(len(frame)),
        "n_valid_samples": int(frame["valid_for_fit"].sum()),
        "cluster_radius_m": float(cluster_radius_m),
        "well_status": well_status,
    }
    return frame, metadata


__all__ = ["ANCHOR_SAMPLE_COLUMNS", "build_well_anchor_samples", "sample_volume_trilinear"]
