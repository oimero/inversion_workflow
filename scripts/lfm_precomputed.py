"""Build time-domain point-control AI low-frequency model.

Usage::

    python scripts/lfm_precomputed.py
    python scripts/lfm_precomputed.py --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import (
    build_segy_textual_header,
    load_yaml_config,
    repo_relative_path,
    resolve_relative_path,
    write_json,
)
from cup.utils.io import to_json_compatible
from cup.utils.io import sanitize_filename
from cup.well.assets import normalize_well_name

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 120


DEFAULT_CONFIG: dict[str, Any] = {
    "source_runs": {
        "mode": "latest",
        "well_auto_tie_dir": None,
        "global_wavelet_generation_dir": None,
    },
    "seismic": {"file": "raw/obn-clipped-240-912-872-1544.zgy", "type": "zgy"},
    "target_interval": {"top_horizon": "interpre/H3-1", "bottom_horizon": "interpre/H7-1", "twt_unit": "auto"},
    "control_wells": {
        "min_batch_corr": 0.35,
        "max_batch_nmae": None,
        "include_wells": None,
        "exclude_wells": [],
    },
    "controls": {
        "sample_step_s": None,
        "min_control_samples_per_well": 16,
    },
    "modeling": {
        "boundary_extension_samples": 50,
        "n_slices": 20,
        "variogram": "spherical",
        "exact": True,
        "nugget": 0.0,
        "filter_cutoff_hz": 10.0,
        "filter_order": 6,
        "filter_buffer_seconds": None,
        "filter_buffer_mode": "reflect",
        "post_slice_smoothing": False,
    },
    "export": {"export_volume": True, "zgy_inline_chunk_size": 16},
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = {key: (value.copy() if isinstance(value, dict) else value) for key, value in base.items()}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except RuntimeError:
        pass
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def _latest_run(output_root: Path, prefix: str, required_file: str) -> Path:
    candidates = [p for p in output_root.glob(f"{prefix}_*") if p.is_dir() and (p / required_file).exists()]
    if not candidates:
        raise FileNotFoundError(f"No run found under {output_root} for {prefix}_* containing {required_file}")
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def _resolve_source_dirs(script_cfg: dict[str, Any], output_root: Path) -> dict[str, Path]:
    source_cfg = dict(script_cfg.get("source_runs") or {})
    mode = str(source_cfg.get("mode", "latest")).strip().lower()
    auto_dir_value = source_cfg.get("well_auto_tie_dir")
    wavelet_dir_value = source_cfg.get("global_wavelet_generation_dir")
    if mode != "latest" and (auto_dir_value is None or wavelet_dir_value is None):
        raise ValueError("source_runs.mode other than 'latest' requires explicit source directories.")
    auto_dir = (
        _latest_run(output_root, "well_auto_tie", "well_tie_metrics.csv")
        if auto_dir_value in {None, ""}
        else resolve_relative_path(auto_dir_value, root=REPO_ROOT)
    )
    wavelet_dir = (
        _latest_run(output_root, "global_wavelet_generation", "batch_synthetic_metrics.csv")
        if wavelet_dir_value in {None, ""}
        else resolve_relative_path(wavelet_dir_value, root=REPO_ROOT)
    )
    return {"well_auto_tie_dir": auto_dir, "global_wavelet_generation_dir": wavelet_dir}


def _resolve_artifact_path(value: Any, *, run_dir: Path) -> Path | None:
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, run_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _as_optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _batch_metric_lookup(batch_df: pd.DataFrame) -> dict[str, pd.Series]:
    well_col = "eval_well" if "eval_well" in batch_df.columns else "well_name"
    if well_col not in batch_df.columns:
        raise ValueError("batch_synthetic_metrics.csv must contain eval_well or well_name.")
    lookup: dict[str, pd.Series] = {}
    for _, row in batch_df.iterrows():
        well_name = str(row[well_col])
        if well_name and well_name.casefold() != "nan":
            lookup[normalize_well_name(well_name)] = row
    return lookup


def _plan_lookup(plan_df: pd.DataFrame) -> dict[str, pd.Series]:
    if plan_df.empty or "well_name" not in plan_df.columns:
        return {}
    return {normalize_well_name(str(row["well_name"])): row for _, row in plan_df.iterrows()}


def _batch_corr(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    for key in ("corr", "batch_corr", "selected_corr"):
        if key in row:
            return _as_optional_float(row.get(key))
    return None


def _batch_nmae(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    for key in ("nmae", "batch_nmae", "selected_nmae"):
        if key in row:
            return _as_optional_float(row.get(key))
    return None


def _resolve_segy_options(cfg: dict[str, Any]) -> dict[str, int] | None:
    from cup.seismic.survey import segy_options_from_config

    if "segy" in cfg:
        options = segy_options_from_config(dict(cfg["segy"]))
        return options or None
    return None


def _open_survey(script_cfg: dict[str, Any], cfg: dict[str, Any], data_root: Path) -> tuple[Any, Path, str]:
    from cup.seismic.survey import open_survey

    seismic_cfg = dict(script_cfg.get("seismic") or {})
    seismic_file = resolve_relative_path(seismic_cfg.get("file"), root=data_root)
    seismic_type = str(seismic_cfg.get("type", "segy"))
    survey = open_survey(seismic_file, seismic_type=seismic_type, segy_options=_resolve_segy_options(cfg))
    return survey, seismic_file, seismic_type


def _normalize_horizon_twt_df(df: pd.DataFrame, *, unit: str) -> pd.DataFrame:
    unit_norm = str(unit or "auto").strip().casefold()
    if "interpretation" not in df.columns:
        return df.copy()
    out = df.copy()
    values = out["interpretation"].to_numpy(dtype=float, copy=True)
    finite = np.isfinite(values)
    if unit_norm in {"ms", "msec", "millisecond", "milliseconds"}:
        values[finite] = np.abs(values[finite]) / 1000.0
    elif unit_norm in {"s", "sec", "second", "seconds"}:
        values[finite] = np.abs(values[finite])
    elif unit_norm == "auto":
        if np.any(finite):
            abs_values = np.abs(values[finite])
            values[finite] = abs_values / 1000.0 if float(np.nanmax(abs_values)) > 20.0 else abs_values
    else:
        raise ValueError(f"Unsupported target_interval.twt_unit: {unit}")
    out["interpretation"] = values
    return out


def _build_target_layer(script_cfg: dict[str, Any], geometry: dict[str, Any], qc_dir: Path, data_root: Path) -> tuple[Any, list[dict[str, Any]]]:
    from cup.petrel.load import import_interpretation_petrel
    from cup.seismic.target_zone import TargetZone

    target_cfg = dict(script_cfg["target_interval"])
    top_file = resolve_relative_path(target_cfg["top_horizon"], root=data_root)
    bottom_file = resolve_relative_path(target_cfg["bottom_horizon"], root=data_root)
    twt_unit = str(target_cfg.get("twt_unit", "auto"))
    raw_horizons = {
        top_file.stem: _normalize_horizon_twt_df(import_interpretation_petrel(top_file), unit=twt_unit),
        bottom_file.stem: _normalize_horizon_twt_df(import_interpretation_petrel(bottom_file), unit=twt_unit),
    }
    target_layer = TargetZone(
        raw_horizon_dfs=raw_horizons,
        geometry=geometry,
        horizon_names=list(raw_horizons.keys()),
        qc_output_dir=qc_dir,
        min_thickness=target_cfg.get("min_thickness"),
        nearest_distance_limit=target_cfg.get("nearest_distance_limit"),
        outlier_threshold=target_cfg.get("outlier_threshold"),
        outlier_min_neighbor_count=target_cfg.get("outlier_min_neighbor_count", 2),
    )
    horizons = []
    for path, name in [(top_file, top_file.stem), (bottom_file, bottom_file.stem)]:
        grid = target_layer.get_horizon_grid(name)
        horizons.append(
            {
                "file": repo_relative_path(path, root=REPO_ROOT),
                "mean_twt_s": float(np.nanmean(grid)),
            }
        )
    return target_layer, horizons


def _lowpass_values_on_twt(
    twt: np.ndarray,
    values: np.ndarray,
    *,
    dt_s: float,
    cutoff_hz: float,
    order: int,
    buffer_seconds: float | None,
    buffer_mode: str,
) -> np.ndarray:
    from cup.seismic.lfm_time import lowpass_twt_log
    from wtie.processing import grid

    original_twt = np.asarray(twt, dtype=np.float64).reshape(-1)
    original_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if original_twt.shape != original_values.shape:
        raise ValueError("twt and values must have matching 1D shapes.")
    if dt_s <= 0.0 or not np.isfinite(dt_s):
        raise ValueError(f"lowpass dt_s must be positive and finite, got {dt_s}.")

    out = np.full(original_values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(original_twt) & np.isfinite(original_values)
    finite_indices = np.flatnonzero(finite)
    min_filter_samples = max(4, 3 * int(order) + 2)
    if finite_indices.size < min_filter_samples:
        out[finite_indices] = original_values[finite_indices]
        return out

    twt = original_twt[finite_indices]
    values = original_values[finite_indices]
    order_idx = np.argsort(twt)
    twt = twt[order_idx]
    values = values[order_idx]
    unique_twt, inverse = np.unique(twt, return_inverse=True)
    if unique_twt.size != twt.size:
        value_sum = np.zeros(unique_twt.size, dtype=np.float64)
        count = np.zeros(unique_twt.size, dtype=np.float64)
        np.add.at(value_sum, inverse, values)
        np.add.at(count, inverse, 1.0)
        values = value_sum / np.maximum(count, 1.0)
        twt = unique_twt
    if twt.size < min_filter_samples:
        out[finite_indices] = np.interp(original_twt[finite_indices], twt, values)
        return out
    regular_twt = np.arange(float(twt[0]), float(twt[-1]) + 0.5 * dt_s, float(dt_s))
    regular_values = np.interp(regular_twt, twt, values)
    log = grid.Log(regular_values, regular_twt, "twt", name="AI", unit="m/s*g/cm3", allow_nan=False)
    filtered = lowpass_twt_log(
        log,
        cutoff_hz=float(cutoff_hz),
        order=int(order),
        buffer_seconds=buffer_seconds,
        buffer_mode=buffer_mode,
    )
    out[finite_indices] = np.interp(original_twt[finite_indices], filtered.basis, filtered.values)
    return out


def _sample_zone(target_layer: Any, inline_float: float, xline_float: float, twt_s: float) -> tuple[str | None, float | None]:
    values = target_layer.get_interpretation_values_at_location(float(inline_float), float(xline_float))
    for top_name, bottom_name in target_layer.iter_zones():
        top = float(values[top_name])
        bottom = float(values[bottom_name])
        if not np.isfinite(top) or not np.isfinite(bottom) or bottom <= top:
            continue
        if top <= float(twt_s) <= bottom:
            return f"{top_name}->{bottom_name}", float((float(twt_s) - top) / (bottom - top))
    return None, None


def _nearest_trace_fields(survey: Any, inline_float: float, xline_float: float) -> dict[str, Any]:
    nearest_inline = survey.line_geometry.snap_inline(float(inline_float))
    nearest_xline = survey.line_geometry.snap_xline(float(xline_float))
    inline_index_f, xline_index_f = survey.line_geometry.line_to_index(nearest_inline, nearest_xline)
    inline_index = min(
        int(survey.line_geometry.inline_axis.count) - 1,
        max(0, int(round(inline_index_f))),
    )
    xline_index = min(
        int(survey.line_geometry.xline_axis.count) - 1,
        max(0, int(round(xline_index_f))),
    )
    flat_idx = int(survey.trace_flat_index(inline_index, xline_index))
    return {
        "nearest_inline": nearest_inline,
        "nearest_xline": nearest_xline,
        "inline_index": inline_index,
        "xline_index": xline_index,
        "flat_idx": flat_idx,
    }


def _control_point_row(**kwargs: Any) -> dict[str, Any]:
    columns = [
        "well_name",
        "route",
        "source",
        "twt_s",
        "md_m",
        "x_m",
        "y_m",
        "inline_float",
        "xline_float",
        "zone_name",
        "u_in_zone",
        "ai",
        "weight",
        "flat_idx",
        "sample_index",
    ]
    return {key: kwargs.get(key) for key in columns}


def _build_vertical_points(
    *,
    well_name: str,
    route: str,
    las_file: Path,
    tdt_file: Path,
    surface_x: float,
    surface_y: float,
    target_layer: Any,
    survey: Any,
    samples: np.ndarray,
    modeling_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, list[str]]:
    from cup.well.las import load_vp_rho_logset_from_standard_las
    from cup.well.td import load_workflow_time_depth_table_csv
    from wtie.processing import grid

    logset = load_vp_rho_logset_from_standard_las(las_file)
    table = load_workflow_time_depth_table_csv(tdt_file)
    dt_s = float(np.median(np.diff(samples)))
    ai_twt = grid.convert_log_from_md_to_twt(logset.AI, table, None, dt_s)
    ai_values = _lowpass_values_on_twt(
        ai_twt.basis,
        ai_twt.values,
        dt_s=dt_s,
        cutoff_hz=float(modeling_cfg["filter_cutoff_hz"]),
        order=int(modeling_cfg["filter_order"]),
        buffer_seconds=modeling_cfg.get("filter_buffer_seconds"),
        buffer_mode=str(modeling_cfg.get("filter_buffer_mode", "reflect")),
    )
    ai_at_samples = np.interp(samples, np.asarray(ai_twt.basis, dtype=float), ai_values, left=np.nan, right=np.nan)
    inline_float, xline_float = survey.line_geometry.coord_to_line(float(surface_x), float(surface_y))
    trace_fields = _nearest_trace_fields(survey, inline_float, xline_float)
    md_at_samples = np.interp(samples, table.twt, table.depth, left=np.nan, right=np.nan)

    rows: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    attempted = 0
    for sample_index, (twt_s, ai, md_m) in enumerate(zip(samples, ai_at_samples, md_at_samples)):
        if not np.isfinite(ai) or not np.isfinite(md_m):
            continue
        attempted += 1
        try:
            zone_name, u_in_zone = _sample_zone(target_layer, inline_float, xline_float, float(twt_s))
        except Exception as exc:
            if not diagnostics:
                diagnostics.append(f"zone_sample_error:{type(exc).__name__}:{exc}")
            zone_name, u_in_zone = None, None
        if zone_name is None or u_in_zone is None:
            continue
        rows.append(
            _control_point_row(
                well_name=well_name,
                route=route,
                source="vertical_trace",
                twt_s=float(twt_s),
                md_m=float(md_m),
                x_m=float(surface_x),
                y_m=float(surface_y),
                inline_float=float(inline_float),
                xline_float=float(xline_float),
                zone_name=zone_name,
                u_in_zone=float(u_in_zone),
                ai=float(ai),
                weight=1.0,
                sample_index=int(sample_index),
                **trace_fields,
            )
        )
    return rows, attempted, diagnostics


def _build_deviated_points(
    *,
    well_name: str,
    route: str,
    las_file: Path,
    trace_plan_file: Path,
    target_layer: Any,
    survey: Any,
    sample_step_s: float | None,
    modeling_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, list[str]]:
    from cup.well.las import load_vp_rho_logset_from_standard_las

    logset = load_vp_rho_logset_from_standard_las(las_file)
    ai_log = logset.AI
    ai_basis = np.asarray(ai_log.basis, dtype=float)
    if ai_basis.size < 2 or np.any(np.diff(ai_basis) <= 0.0):
        raise ValueError(f"AI log MD basis must be strictly increasing: {las_file}")
    plan_df = pd.read_csv(trace_plan_file)
    required = {"twt_s", "md_m", "x_m", "y_m", "inline_float", "xline_float", "survey_position"}
    missing = required - set(plan_df.columns)
    if missing:
        raise ValueError(f"trace_sample_plan missing columns: {sorted(missing)}")
    plan_df = plan_df.loc[plan_df["survey_position"].astype(str).eq("inside")].copy()
    plan_df = plan_df.sort_values("twt_s").reset_index(drop=True)
    if sample_step_s is not None and sample_step_s > 0.0 and not plan_df.empty:
        keep = []
        last_twt = -np.inf
        for idx, twt in enumerate(plan_df["twt_s"].to_numpy(dtype=float)):
            if twt >= last_twt + float(sample_step_s) - 1e-9:
                keep.append(idx)
                last_twt = float(twt)
        plan_df = plan_df.iloc[keep].reset_index(drop=True)

    attempted = int(len(plan_df))
    if plan_df.empty:
        return [], attempted, []
    plan_md = plan_df["md_m"].to_numpy(dtype=float)
    finite_md = plan_md[np.isfinite(plan_md)]
    if finite_md.size >= 2 and np.any(np.diff(finite_md) < 0.0):
        raise ValueError(f"trace_sample_plan md_m must be non-decreasing with TWT: {trace_plan_file}")
    ai_raw = np.interp(
        plan_md,
        ai_basis,
        np.asarray(ai_log.values, dtype=float),
        left=np.nan,
        right=np.nan,
    )
    plan_twt = plan_df["twt_s"].to_numpy(dtype=float)
    dt_s = float(np.nanmedian(np.diff(plan_twt))) if len(plan_df) > 1 else float(survey.sample_axis("time").step)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(f"trace_sample_plan twt_s must have a positive sampling interval: {trace_plan_file}")
    if np.isfinite(dt_s) and dt_s > 0.0:
        ai_filtered = _lowpass_values_on_twt(
            plan_twt,
            ai_raw,
            dt_s=dt_s,
            cutoff_hz=float(modeling_cfg["filter_cutoff_hz"]),
            order=int(modeling_cfg["filter_order"]),
            buffer_seconds=modeling_cfg.get("filter_buffer_seconds"),
            buffer_mode=str(modeling_cfg.get("filter_buffer_mode", "reflect")),
        )
        ai_by_row = ai_filtered
    else:
        ai_by_row = ai_raw

    rows: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    for row_index, row in plan_df.iterrows():
        ai = float(ai_by_row[row_index])
        if not np.isfinite(ai):
            continue
        inline_float = float(row["inline_float"])
        xline_float = float(row["xline_float"])
        twt_s = float(row["twt_s"])
        try:
            zone_name, u_in_zone = _sample_zone(target_layer, inline_float, xline_float, twt_s)
        except Exception as exc:
            if not diagnostics:
                diagnostics.append(f"zone_sample_error:{type(exc).__name__}:{exc}")
            zone_name, u_in_zone = None, None
        if zone_name is None or u_in_zone is None:
            continue
        trace_fields = {}
        for key in ["flat_idx", "inline_index", "xline_index", "nearest_inline", "nearest_xline"]:
            if key in row and pd.notna(row[key]):
                trace_fields[key] = int(row[key]) if key.endswith("index") or key == "flat_idx" else float(row[key])
        if "flat_idx" not in trace_fields:
            trace_fields = _nearest_trace_fields(survey, inline_float, xline_float)
        rows.append(
            _control_point_row(
                well_name=well_name,
                route=route,
                source="deviated_trajectory",
                twt_s=twt_s,
                md_m=float(row["md_m"]),
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                inline_float=inline_float,
                xline_float=xline_float,
                zone_name=zone_name,
                u_in_zone=float(u_in_zone),
                ai=ai,
                weight=1.0,
                sample_index=int(row["sample_index"]) if "sample_index" in row and pd.notna(row["sample_index"]) else int(row_index),
                **trace_fields,
            )
        )
    return rows, attempted, diagnostics


def _to_control_points(rows: pd.DataFrame) -> list[Any]:
    from cup.seismic.lfm_time import LfmTimeControlPoint

    points = []
    for _, row in rows.iterrows():
        points.append(
            LfmTimeControlPoint(
                well_name=str(row["well_name"]),
                route=str(row["route"]),
                twt_s=float(row["twt_s"]),
                md_m=float(row["md_m"]),
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                inline_float=float(row["inline_float"]),
                xline_float=float(row["xline_float"]),
                zone_name=str(row["zone_name"]),
                u_in_zone=float(row["u_in_zone"]),
                ai=float(row["ai"]),
                weight=float(row.get("weight", 1.0)),
                source=str(row.get("source", "")),
                flat_idx=None if pd.isna(row.get("flat_idx")) else int(row.get("flat_idx")),
                sample_index=None if pd.isna(row.get("sample_index")) else int(row.get("sample_index")),
            )
        )
    return points


def _plot_controls(control_df: pd.DataFrame, output_path: Path) -> None:
    if control_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    scatter = ax.scatter(
        control_df["inline_float"],
        control_df["xline_float"],
        c=control_df["twt_s"],
        s=8,
        alpha=0.65,
        cmap="viridis",
    )
    ax.set_title("LFM layer control points")
    ax.set_xlabel("Inline")
    ax.set_ylabel("Xline")
    fig.colorbar(scatter, ax=ax, label="TWT (s)")
    _save_fig(output_path)


def _plot_lfm_result(result: Any, output_path: Path) -> None:
    ilines = result.ilines
    xlines = result.xlines
    samples = result.samples
    volume = result.volume
    i_il = len(ilines) // 2
    i_xl = len(xlines) // 2
    i_t = len(samples) // 2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    im0 = axes[0].imshow(
        volume[i_il, :, :].T,
        aspect="auto",
        origin="upper",
        extent=[xlines[0], xlines[-1], samples[-1], samples[0]],
        cmap="viridis",
    )
    axes[0].set_title(f"AI LFM inline @ {ilines[i_il]:.0f}")
    axes[0].set_xlabel("Xline")
    axes[0].set_ylabel("TWT (s)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(
        volume[:, i_xl, :].T,
        aspect="auto",
        origin="upper",
        extent=[ilines[0], ilines[-1], samples[-1], samples[0]],
        cmap="viridis",
    )
    axes[1].set_title(f"AI LFM xline @ {xlines[i_xl]:.0f}")
    axes[1].set_xlabel("Inline")
    axes[1].set_ylabel("TWT (s)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    im2 = axes[2].imshow(
        volume[:, :, i_t].T,
        aspect="auto",
        origin="lower",
        extent=[ilines[0], ilines[-1], xlines[0], xlines[-1]],
        cmap="viridis",
    )
    axes[2].set_title(f"AI LFM time slice @ {samples[i_t]:.3f} s")
    axes[2].set_xlabel("Inline")
    axes[2].set_ylabel("Xline")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    _save_fig(output_path)


def _save_npz(result: Any, npz_file: Path, *, metadata_extra: dict[str, Any]) -> None:
    result.metadata.update(metadata_extra)
    npz_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_file,
        volume=result.volume.astype(np.float32),
        variance_volume=result.variance_volume.astype(np.float32),
        ilines=result.ilines,
        xlines=result.xlines,
        samples=result.samples,
        geometry_json=json.dumps(to_json_compatible(result.geometry), ensure_ascii=False),
        metadata_json=json.dumps(to_json_compatible(result.metadata), ensure_ascii=False),
        coverage_stats_json=json.dumps(to_json_compatible(result.coverage_stats), ensure_ascii=False),
    )


def _try_write_segy(result: Any, segy_file: Path, seismic_file: Path, seismic_type: str, cfg: dict[str, Any]) -> str | None:
    if seismic_type.lower() != "segy":
        return "skipped_non_segy_source"
    try:
        import cigsegy

        segy_cfg = dict(cfg.get("segy") or {})
        keylocs = [
            int(segy_cfg["iline_byte"]),
            int(segy_cfg["xline_byte"]),
            int(segy_cfg["istep"]),
            int(segy_cfg["xstep"]),
        ]
        textual = build_segy_textual_header(
            "Time-domain AI low-frequency model",
            ["artifact=ai_lfm_time.npz", "source=lfm_precomputed.py"],
        )
        cigsegy.create_by_sharing_header(
            str(segy_file),
            str(seismic_file),
            np.ascontiguousarray(result.volume.astype(np.float32)),
            keylocs=keylocs,
            textual=textual,
        )
        return None
    except Exception as exc:
        return str(exc)


def _zgy_corners_from_survey(survey: Any, result: Any) -> tuple[tuple[float, float], ...]:
    il0 = float(result.ilines[0])
    iln = float(result.ilines[-1])
    xl0 = float(result.xlines[0])
    xln = float(result.xlines[-1])
    geometry = survey.line_geometry
    return (
        tuple(float(v) for v in geometry.line_to_coord(il0, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(il0, xln)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xln)),
    )


def _try_write_zgy(
    result: Any,
    zgy_file: Path,
    survey: Any,
    seismic_type: str,
    *,
    inline_chunk_size: int = 16,
) -> str | None:
    if seismic_type.lower() != "zgy":
        return "skipped_non_zgy_source"
    try:
        from pyzgy.write import SeismicWriter

        samples = np.asarray(result.samples, dtype=np.float64)
        if samples.size < 2:
            raise ValueError("ZGY export requires at least two samples.")
        sample_step_s = float(np.median(np.diff(samples)))
        if not np.allclose(np.diff(samples), sample_step_s, rtol=1e-6, atol=1e-9):
            raise ValueError("ZGY export requires a regular sample axis.")

        ilines = np.asarray(result.ilines, dtype=np.float64)
        xlines = np.asarray(result.xlines, dtype=np.float64)
        inline_inc = float(np.median(np.diff(ilines))) if ilines.size > 1 else 0.0
        xline_inc = float(np.median(np.diff(xlines))) if xlines.size > 1 else 0.0
        if ilines.size > 2 and not np.allclose(np.diff(ilines), inline_inc, rtol=0.0, atol=1e-8):
            raise ValueError("ZGY export requires a regular inline axis.")
        if xlines.size > 2 and not np.allclose(np.diff(xlines), xline_inc, rtol=0.0, atol=1e-8):
            raise ValueError("ZGY export requires a regular xline axis.")
        corners = _zgy_corners_from_survey(survey, result)

        zgy_file.parent.mkdir(parents=True, exist_ok=True)
        if zgy_file.exists():
            zgy_file.unlink()
        chunk = max(1, int(inline_chunk_size))
        volume = np.asarray(result.volume, dtype=np.float32)
        with SeismicWriter(
            zgy_file,
            tuple(int(v) for v in volume.shape),
            float(samples[0]) * 1000.0,
            sample_step_s * 1000.0,
            (float(ilines[0]), float(xlines[0])),
            (inline_inc, xline_inc),
            corners=corners,
        ) as writer:
            for il_start in range(0, volume.shape[0], chunk):
                il_end = min(volume.shape[0], il_start + chunk)
                writer.write_subvolume(volume[il_start:il_end], il_start, 0, 0)
        return None
    except Exception as exc:
        return str(exc)


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = _deep_update(DEFAULT_CONFIG, dict(cfg.get("lfm_precomputed") or {}))
    data_root = REPO_ROOT / str(cfg.get("data_root", "data"))
    output_root = REPO_ROOT / str(cfg.get("output_root", "scripts/output"))
    source_dirs = _resolve_source_dirs(script_cfg, output_root)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"lfm_precomputed_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    qc_dir = output_dir / "target_layer_qc"
    figures_dir = output_dir / "figures"
    for directory in [output_dir, qc_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    survey, seismic_file, seismic_type = _open_survey(script_cfg, cfg, data_root)
    geometry = survey.describe_geometry(domain="time")
    if str(geometry.get("sample_domain")).lower() != "time" or str(geometry.get("sample_unit")).lower() != "s":
        raise ValueError(f"Expected time-domain seismic geometry in seconds, got {geometry}")
    target_layer, horizon_metadata = _build_target_layer(script_cfg, geometry, qc_dir, data_root)
    sample_axis = survey.sample_axis("time").values.astype(np.float64)
    sample_step_s = script_cfg["controls"].get("sample_step_s")
    if sample_step_s is not None:
        sample_step_s = float(sample_step_s)
        sample_axis = np.arange(float(sample_axis[0]), float(sample_axis[-1]) + 0.5 * sample_step_s, sample_step_s)

    auto_dir = source_dirs["well_auto_tie_dir"]
    wavelet_dir = source_dirs["global_wavelet_generation_dir"]
    metrics_df = pd.read_csv(auto_dir / "well_tie_metrics.csv")
    plan_df = pd.read_csv(auto_dir / "well_tie_plan.csv") if (auto_dir / "well_tie_plan.csv").exists() else pd.DataFrame()
    batch_df = pd.read_csv(wavelet_dir / "batch_synthetic_metrics.csv")
    batch_lookup = _batch_metric_lookup(batch_df)
    plan_by_key = _plan_lookup(plan_df)

    include_wells = script_cfg["control_wells"].get("include_wells")
    include_keys = {normalize_well_name(w) for w in include_wells} if include_wells else None
    exclude_keys = {normalize_well_name(w) for w in script_cfg["control_wells"].get("exclude_wells", [])}
    min_corr = script_cfg["control_wells"].get("min_batch_corr")
    max_nmae = script_cfg["control_wells"].get("max_batch_nmae")
    min_points = int(script_cfg["controls"].get("min_control_samples_per_well", 16))

    control_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for _, metric in metrics_df.iterrows():
        well_name = str(metric.get("well_name", "")).strip()
        if not well_name:
            continue
        key = normalize_well_name(well_name)
        route = str(metric.get("route", ""))
        tie_status = str(metric.get("tie_status", ""))
        batch_row = batch_lookup.get(key)
        batch_corr = _batch_corr(batch_row)
        batch_nmae = _batch_nmae(batch_row)
        reasons: list[str] = []
        status = "selected"
        if tie_status != "success":
            reasons.append(f"tie_status_{tie_status or 'unknown'}")
        if include_keys is not None and key not in include_keys:
            reasons.append("not_in_include_wells")
        if key in exclude_keys:
            reasons.append("excluded_well")
        if batch_corr is None:
            reasons.append("missing_batch_corr")
        elif min_corr is not None and batch_corr < float(min_corr):
            reasons.append("batch_corr_below_threshold")
        if max_nmae is not None and (batch_nmae is None or batch_nmae > float(max_nmae)):
            reasons.append("batch_nmae_above_threshold")

        las_file = _resolve_artifact_path(metric.get("filtered_las_file"), run_dir=auto_dir)
        tdt_file = _resolve_artifact_path(metric.get("optimized_tdt_file"), run_dir=auto_dir)
        if las_file is None or not las_file.exists():
            reasons.append("missing_filtered_las_file")
        if tdt_file is None or not tdt_file.exists():
            reasons.append("missing_optimized_tdt_file")

        plan = plan_by_key.get(key)
        well_rows: list[dict[str, Any]] = []
        attempted = 0
        diagnostics: list[str] = []
        if reasons:
            status = "rejected"
        else:
            try:
                if route.startswith("deviated"):
                    trace_file = auto_dir / "trace_sample_plan" / f"trace_sample_plan_{sanitize_filename(well_name)}.csv"
                    if not trace_file.exists():
                        raise FileNotFoundError("missing_trace_sample_plan")
                    well_rows, attempted, diagnostics = _build_deviated_points(
                        well_name=well_name,
                        route=route,
                        las_file=las_file,
                        trace_plan_file=trace_file,
                        target_layer=target_layer,
                        survey=survey,
                        sample_step_s=sample_step_s,
                        modeling_cfg=script_cfg["modeling"],
                    )
                else:
                    if plan is None:
                        raise ValueError("missing_well_tie_plan_row")
                    surface_x = _as_optional_float(plan.get("surface_x"))
                    surface_y = _as_optional_float(plan.get("surface_y"))
                    if surface_x is None or surface_y is None:
                        raise ValueError("missing_surface_xy")
                    well_rows, attempted, diagnostics = _build_vertical_points(
                        well_name=well_name,
                        route=route,
                        las_file=las_file,
                        tdt_file=tdt_file,
                        surface_x=surface_x,
                        surface_y=surface_y,
                        target_layer=target_layer,
                        survey=survey,
                        samples=sample_axis,
                        modeling_cfg=script_cfg["modeling"],
                    )
                reasons.extend(diagnostics)
                if len(well_rows) < min_points:
                    status = "rejected"
                    reasons.append("too_few_control_samples")
                else:
                    control_rows.extend(well_rows)
            except Exception as exc:
                status = "rejected" if str(exc) == "missing_trace_sample_plan" else "failed"
                reasons.append(str(exc) or type(exc).__name__)

        unique_trace_count = 0
        if well_rows:
            unique_trace_count = int(pd.DataFrame(well_rows)["flat_idx"].dropna().nunique())
        invalid_count = max(0, int(attempted) - int(len(well_rows)))
        qc_rows.append(
            {
                "well_name": well_name,
                "status": status,
                "route": route,
                "batch_corr": batch_corr,
                "batch_nmae": batch_nmae,
                "control_point_count": int(len(well_rows)) if status == "selected" else 0,
                "invalid_point_count": invalid_count,
                "invalid_point_fraction": float(invalid_count / attempted) if attempted else None,
                "unique_trace_count": unique_trace_count,
                "reasons": ";".join(dict.fromkeys(reasons)),
            }
        )

    control_df = pd.DataFrame.from_records(control_rows)
    qc_df = pd.DataFrame.from_records(qc_rows)
    control_path = output_dir / "lfm_layer_control_points.csv"
    qc_path = output_dir / "lfm_control_qc.csv"
    control_df.to_csv(control_path, index=False, encoding="utf-8-sig")
    qc_df.to_csv(qc_path, index=False, encoding="utf-8-sig")
    if control_df.empty:
        raise ValueError("No LFM control points selected. Check lfm_control_qc.csv for rejection reasons.")

    from cup.seismic.lfm_time import build_lfm_time_model_from_points

    result = build_lfm_time_model_from_points(
        target_layer=target_layer,
        control_points=_to_control_points(control_df),
        boundary_extension_samples=int(script_cfg["modeling"]["boundary_extension_samples"]),
        n_slices=int(script_cfg["modeling"]["n_slices"]),
        variogram=str(script_cfg["modeling"]["variogram"]),
        exact=bool(script_cfg["modeling"]["exact"]),
        nugget=float(script_cfg["modeling"]["nugget"]),
        post_slice_smoothing=bool(script_cfg["modeling"].get("post_slice_smoothing", False)),
    )
    result.metadata.update(
        {
            "filter_cutoff_hz": float(script_cfg["modeling"]["filter_cutoff_hz"]),
            "filter_order": int(script_cfg["modeling"]["filter_order"]),
            "filter_buffer_seconds": script_cfg["modeling"].get("filter_buffer_seconds"),
            "filter_buffer_mode": str(script_cfg["modeling"].get("filter_buffer_mode", "reflect")),
        }
    )
    metadata_extra = {
        "property_name": "AI",
        "target_layer": {
            "min_thickness": script_cfg["target_interval"].get("min_thickness"),
            "nearest_distance_limit": script_cfg["target_interval"].get("nearest_distance_limit"),
            "outlier_threshold": script_cfg["target_interval"].get("outlier_threshold"),
            "outlier_min_neighbor_count": script_cfg["target_interval"].get("outlier_min_neighbor_count", 2),
        },
        "horizons": horizon_metadata,
        "path_style": "repo_relative",
    }
    npz_file = output_dir / "ai_lfm_time.npz"
    _save_npz(result, npz_file, metadata_extra=metadata_extra)

    export_status = None
    export_ext = None
    if bool(script_cfg["export"].get("export_volume", True)):
        if seismic_type.lower() == "segy":
            export_ext = ".segy"
            export_status = _try_write_segy(result, output_dir / "ai_lfm_time.segy", seismic_file, seismic_type, cfg)
        elif seismic_type.lower() == "zgy":
            export_ext = ".zgy"
            export_status = _try_write_zgy(
                result,
                output_dir / "ai_lfm_time.zgy",
                survey,
                seismic_type,
                inline_chunk_size=int(script_cfg["export"].get("zgy_inline_chunk_size", 16)),
            )

    _plot_controls(control_df, figures_dir / "qc_control_points.png")
    _plot_lfm_result(result, figures_dir / "qc_ai_lfm_time.png")
    summary = {
        "source_dirs": {key: repo_relative_path(value, root=REPO_ROOT) for key, value in source_dirs.items()},
        "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
        "seismic_type": seismic_type,
        "config": script_cfg,
        "selected_well_count": int((qc_df["status"] == "selected").sum()),
        "control_point_count": int(len(control_df)),
        "outputs": {
            "ai_lfm_time": repo_relative_path(npz_file, root=REPO_ROOT),
            "control_points": repo_relative_path(control_path, root=REPO_ROOT),
            "control_qc": repo_relative_path(qc_path, root=REPO_ROOT),
        },
        "export_status": "written" if export_status is None else export_status,
        "coverage_stats": result.coverage_stats,
    }
    write_json(output_dir / "run_summary.json", summary)

    print("=== LFM Precomputed ===")
    print(f"Output: {output_dir}")
    print(f"Selected wells: {summary['selected_well_count']}")
    print(f"Control points: {summary['control_point_count']}")
    print(f"NPZ: {npz_file}")
    if export_status:
        print(f"Volume export skipped/failed: {export_status}")


if __name__ == "__main__":
    main()
