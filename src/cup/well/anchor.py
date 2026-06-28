"""Model-independent real-well labels for sparse supervision experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cup.utils.io import resolve_relative_path
from cup.utils.statistics import radius_connected_components


ANCHOR_SAMPLE_COLUMNS = [
    "well_name",
    "sample_index",
    "twt_s",
    "inline",
    "xline",
    "x_m",
    "y_m",
    "spatial_cluster_id",
    "spatial_cluster_size",
    "filtered_log_ai",
    "lfm_log_ai",
    "valid_for_fit",
    "valid_reason",
    "sampling_mode",
    "sample_method",
    "wellbore_class",
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
    """Sample an ``[inline, xline, twt]`` volume without extrapolation."""

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
    for name, axis in zip(("inline", "xline", "twt"), axes):
        if axis.size < 2 or not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"{name} axis must be strictly increasing with at least two samples.")

    fractional = [np.interp(values, axis, np.arange(axis.size), left=np.nan, right=np.nan) for values, axis in zip(coords, axes)]
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
    well_auto_tie_dir: Path,
    well_inventory_file: Path,
    lfm: np.ndarray,
    valid_mask: np.ndarray,
    ilines: np.ndarray,
    xlines: np.ndarray,
    twt_s: np.ndarray,
    repo_root: Path,
    cluster_radius_m: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the frozen per-sample real-well delta-label contract."""

    metrics_path = well_auto_tie_dir / "well_tie_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"well_tie_metrics.csv not found: {metrics_path}")
    if not well_inventory_file.is_file():
        raise FileNotFoundError(f"well_inventory.csv not found: {well_inventory_file}")
    metrics = pd.read_csv(metrics_path)
    inventory = pd.read_csv(well_inventory_file)
    _require_columns(
        metrics,
        {"well_name", "tie_status", "filtered_las_file", "optimized_tdt_file", "optimized_trace_sample_plan_file"},
        metrics_path,
    )
    _require_columns(
        inventory,
        {"well_name", "inline_float", "xline_float", "surface_x", "surface_y", "wellbore_class"},
        well_inventory_file,
    )
    successful = metrics[metrics["tie_status"].astype(str).eq("success")].copy()
    if successful.empty:
        raise ValueError("No successful well ties are available for L0 anchors.")
    inventory = inventory.drop_duplicates("well_name").set_index("well_name", drop=False)

    cluster_rows: list[dict[str, Any]] = []
    for well_name in successful["well_name"].astype(str):
        if well_name not in inventory.index:
            raise ValueError(f"Successful tie well is missing from well inventory: {well_name}")
        row = inventory.loc[well_name]
        x_m = _number(row["surface_x"])
        y_m = _number(row["surface_y"])
        if not (np.isfinite(x_m) and np.isfinite(y_m)):
            raise ValueError(f"Well has invalid surface XY for spatial clustering: {well_name}")
        cluster_rows.append({"well_name": well_name, "x_m": x_m, "y_m": y_m})
    clusters = pd.DataFrame.from_records(cluster_rows)
    clusters["spatial_cluster_id"] = radius_connected_components(
        clusters[["x_m", "y_m"]].to_numpy(dtype=np.float64),
        float(cluster_radius_m),
    )
    clusters["spatial_cluster_size"] = clusters.groupby("spatial_cluster_id")["well_name"].transform("count")
    cluster_lookup = clusters.set_index("well_name").to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    well_status: list[dict[str, Any]] = []
    for _, tie in successful.iterrows():
        well_name = str(tie["well_name"])
        inv = inventory.loc[well_name]
        sample_method, sample_inline, sample_xline, sample_twt, sample_x, sample_y = _well_coordinates(
            tie=tie,
            inventory=inv,
            twt_s=np.asarray(twt_s, dtype=np.float64),
            repo_root=repo_root,
        )
        filtered = _log_ai_at_twt(
            las_path=resolve_relative_path(str(tie["filtered_las_file"]), root=repo_root),
            tdt_path=resolve_relative_path(str(tie["optimized_tdt_file"]), root=repo_root),
            twt_s=sample_twt,
        )
        sampled_lfm, lfm_inside = sample_volume_trilinear(
            lfm,
            ilines=ilines,
            xlines=xlines,
            twt_s=twt_s,
            inline_values=sample_inline,
            xline_values=sample_xline,
            sample_twt_s=sample_twt,
        )
        sampled_mask, mask_inside = sample_volume_trilinear(
            np.asarray(valid_mask, dtype=np.float32),
            ilines=ilines,
            xlines=xlines,
            twt_s=twt_s,
            inline_values=sample_inline,
            xline_values=sample_xline,
            sample_twt_s=sample_twt,
        )
        valid = (
            lfm_inside
            & mask_inside
            & (sampled_mask > 0.5)
            & np.isfinite(filtered)
            & np.isfinite(sampled_lfm)
        )
        cluster = cluster_lookup[well_name]
        for index in range(sample_twt.size):
            reason = "ok"
            if not lfm_inside[index]:
                reason = "outside_lfm_support"
            elif not mask_inside[index] or not sampled_mask[index] > 0.5:
                reason = "valid_mask_false"
            elif not np.isfinite(filtered[index]):
                reason = "filtered_log_ai_nonfinite"
            elif not np.isfinite(sampled_lfm[index]):
                reason = "lfm_log_ai_nonfinite"
            rows.append(
                {
                    "well_name": well_name,
                    "sample_index": int(index),
                    "twt_s": float(sample_twt[index]),
                    "inline": float(sample_inline[index]),
                    "xline": float(sample_xline[index]),
                    "x_m": float(sample_x[index]),
                    "y_m": float(sample_y[index]),
                    "spatial_cluster_id": int(cluster["spatial_cluster_id"]),
                    "spatial_cluster_size": int(cluster["spatial_cluster_size"]),
                    "filtered_log_ai": float(filtered[index]) if np.isfinite(filtered[index]) else np.nan,
                    "lfm_log_ai": float(sampled_lfm[index]) if np.isfinite(sampled_lfm[index]) else np.nan,
                    "valid_for_fit": bool(valid[index]),
                    "valid_reason": reason,
                    "sampling_mode": "volume",
                    "sample_method": sample_method,
                    "wellbore_class": str(inv["wellbore_class"]),
                }
            )
        well_status.append(
            {
                "well_name": well_name,
                "n_samples": int(sample_twt.size),
                "n_valid": int(np.count_nonzero(valid)),
                "sample_method": sample_method,
            }
        )
    frame = pd.DataFrame.from_records(rows, columns=ANCHOR_SAMPLE_COLUMNS)
    if frame.empty or not frame["valid_for_fit"].any():
        raise ValueError("L0 well anchor builder produced no valid samples.")
    metadata = {
        "schema_version": "l0_well_anchor_samples_v1",
        "n_wells": int(frame["well_name"].nunique()),
        "n_clusters": int(frame["spatial_cluster_id"].nunique()),
        "n_samples": int(len(frame)),
        "n_valid_samples": int(frame["valid_for_fit"].sum()),
        "cluster_radius_m": float(cluster_radius_m),
        "well_status": well_status,
    }
    return frame, metadata


def _well_coordinates(
    *,
    tie: Mapping[str, Any],
    inventory: Mapping[str, Any],
    twt_s: np.ndarray,
    repo_root: Path,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    plan_text = str(tie.get("optimized_trace_sample_plan_file") or "").strip()
    if plan_text and plan_text.casefold() != "nan":
        plan_path = resolve_relative_path(plan_text, root=repo_root)
        plan = pd.read_csv(plan_path)
        _require_columns(plan, {"survey_position", "inline_float", "xline_float", "twt_s"}, plan_path)
        plan = plan[plan["survey_position"].astype(str).eq("inside")].copy()
        if plan.empty:
            raise ValueError(f"Trace sample plan has no inside samples: {plan_path}")
        n = len(plan)
        return (
            "volume_trace_plan_trilinear",
            pd.to_numeric(plan["inline_float"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(plan["xline_float"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(plan["twt_s"], errors="coerce").to_numpy(dtype=np.float64),
            _column_or_nan(plan, "x_m", n),
            _column_or_nan(plan, "y_m", n),
        )
    inline = _number(inventory.get("inline_float"))
    xline = _number(inventory.get("xline_float"))
    if not (np.isfinite(inline) and np.isfinite(xline)):
        raise ValueError(f"Well has invalid inline/xline: {tie.get('well_name')}")
    return (
        "volume_vertical_trilinear",
        np.full(twt_s.shape, inline),
        np.full(twt_s.shape, xline),
        twt_s.copy(),
        np.full(twt_s.shape, _number(inventory.get("surface_x"))),
        np.full(twt_s.shape, _number(inventory.get("surface_y"))),
    )


def _log_ai_at_twt(*, las_path: Path, tdt_path: Path, twt_s: np.ndarray) -> np.ndarray:
    import lasio

    las = lasio.read(str(las_path))
    frame = las.df()
    if "AI" not in frame.columns:
        raise ValueError(f"Filtered LAS lacks AI curve: {las_path}")
    md = frame.index.to_numpy(dtype=np.float64)
    ai = frame["AI"].to_numpy(dtype=np.float64)
    tdt = pd.read_csv(tdt_path)
    _require_columns(tdt, {"twt_s", "md_m"}, tdt_path)
    table_twt = pd.to_numeric(tdt["twt_s"], errors="coerce").to_numpy(dtype=np.float64)
    table_md = pd.to_numeric(tdt["md_m"], errors="coerce").to_numpy(dtype=np.float64)
    valid_table = np.isfinite(table_twt) & np.isfinite(table_md)
    valid_ai = np.isfinite(md) & np.isfinite(ai) & (ai > 0.0)
    if np.count_nonzero(valid_table) < 2 or np.count_nonzero(valid_ai) < 2:
        return np.full(twt_s.shape, np.nan)
    table_order = np.argsort(table_twt[valid_table])
    ai_order = np.argsort(md[valid_ai])
    md_at_twt = np.interp(
        twt_s,
        table_twt[valid_table][table_order],
        table_md[valid_table][table_order],
        left=np.nan,
        right=np.nan,
    )
    ai_at_twt = np.interp(
        md_at_twt,
        md[valid_ai][ai_order],
        ai[valid_ai][ai_order],
        left=np.nan,
        right=np.nan,
    )
    return np.where(ai_at_twt > 0.0, np.log(ai_at_twt), np.nan)


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _column_or_nan(frame: pd.DataFrame, name: str, size: int) -> np.ndarray:
    if name not in frame:
        return np.full(size, np.nan, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)


__all__ = ["ANCHOR_SAMPLE_COLUMNS", "build_well_anchor_samples", "sample_volume_trilinear"]
