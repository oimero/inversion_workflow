"""层位约束低频模型构建与井震标定成果保存。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from cup.seismic.process import TargetLayer
from wtie.processing import grid
from wtie.processing.spectral import apply_butter_lowpass_filter


@dataclass
class WellTieArtifact:
    """单井井震标定后用于低频模型构建的控制数据。"""

    well_name: str
    inline: float
    xline: float
    horizon_times: Dict[str, float]
    lowfreq_ai: pd.DataFrame
    x: Optional[float] = None
    y: Optional[float] = None
    optimized_td: Optional[pd.DataFrame] = None
    wavelet: Optional[pd.DataFrame] = None
    logset_twt: Optional[pd.DataFrame] = None
    summary: Optional[Dict[str, Any]] = None


@dataclass
class LowFrequencyModelResult:
    """低频模型构建结果。"""

    volume: np.ndarray
    variance_volume: np.ndarray
    geometry: Dict[str, Any]
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    metadata: Dict[str, Any]
    wells: list[WellTieArtifact]


def _build_line_axis(line_min: float, line_max: float, line_step: float) -> np.ndarray:
    if line_step <= 0:
        raise ValueError(f"line_step must be positive, got {line_step}.")
    return np.arange(line_min, line_max + line_step, line_step, dtype=float)


def _build_sample_axis(geometry: Dict[str, Any]) -> np.ndarray:
    required_keys = {"sample_min", "sample_max", "sample_step"}
    missing_keys = required_keys - set(geometry)
    if missing_keys:
        raise ValueError(f"geometry is missing required sample keys: {sorted(missing_keys)}")

    sample_min = float(geometry["sample_min"])
    sample_max = float(geometry["sample_max"])
    sample_step = float(geometry["sample_step"])
    if sample_step <= 0:
        raise ValueError(f"sample_step must be positive, got {sample_step}.")
    return np.arange(sample_min, sample_max + sample_step, sample_step, dtype=float)


def _nearest_neighbor_range(inlines: np.ndarray, xlines: np.ndarray) -> float:
    if inlines.size <= 1:
        return 1.0
    coords = np.column_stack([inlines, xlines]).astype(float, copy=False)
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return 1.0
    return float(max(np.median(finite), 1.0))


def _normalize_lowfreq_ai_df(lowfreq_ai: pd.DataFrame) -> pd.DataFrame:
    if lowfreq_ai.shape[1] < 2:
        raise ValueError("lowfreq_ai must have at least two columns for twt and ai.")

    col_names = [str(col) for col in lowfreq_ai.columns]
    lowered_names = [col.lower() for col in col_names]
    if {"twt", "ai"}.issubset(lowered_names):
        twt_col = lowfreq_ai.columns[lowered_names.index("twt")]
        ai_col = lowfreq_ai.columns[lowered_names.index("ai")]
    else:
        twt_col, ai_col = lowfreq_ai.columns[:2]

    out_df = lowfreq_ai[[twt_col, ai_col]].copy()
    out_df.columns = ["twt", "ai"]
    out_df["twt"] = pd.to_numeric(out_df["twt"], errors="coerce")
    out_df["ai"] = pd.to_numeric(out_df["ai"], errors="coerce")
    out_df = out_df[np.isfinite(out_df["twt"]) & np.isfinite(out_df["ai"])].sort_values("twt").reset_index(drop=True)
    if out_df.empty:
        raise ValueError("lowfreq_ai contains no finite samples.")
    if float(np.nanmax(np.abs(out_df["twt"].to_numpy()))) > 10.0:
        out_df["twt"] = out_df["twt"] / 1000.0
    return out_df


def _validate_well_artifact(well: WellTieArtifact, horizon_names: list[str]) -> WellTieArtifact:
    missing_horizons = [name for name in horizon_names if name not in well.horizon_times]
    if missing_horizons:
        raise ValueError(f"well '{well.well_name}' is missing horizon times for {missing_horizons}")
    if not np.isfinite(well.inline) or not np.isfinite(well.xline):
        raise ValueError(f"well '{well.well_name}' must provide finite inline/xline coordinates.")

    out_well = WellTieArtifact(
        well_name=well.well_name,
        inline=float(well.inline),
        xline=float(well.xline),
        horizon_times={name: float(well.horizon_times[name]) for name in horizon_names},
        lowfreq_ai=_normalize_lowfreq_ai_df(well.lowfreq_ai),
        x=None if well.x is None else float(well.x),
        y=None if well.y is None else float(well.y),
        optimized_td=None if well.optimized_td is None else well.optimized_td.copy(),
        wavelet=None if well.wavelet is None else well.wavelet.copy(),
        logset_twt=None if well.logset_twt is None else well.logset_twt.copy(),
        summary=None if well.summary is None else dict(well.summary),
    )
    return out_well


def _interpolate_ai_at_time(lowfreq_ai: pd.DataFrame, twt_s: float) -> float:
    twt = lowfreq_ai["twt"].to_numpy(dtype=float, copy=False)
    ai = lowfreq_ai["ai"].to_numpy(dtype=float, copy=False)
    return float(np.interp(twt_s, twt, ai, left=ai[0], right=ai[-1]))


def _krige_slice_on_line_domain(
    control_inlines: np.ndarray,
    control_xlines: np.ndarray,
    control_values: np.ndarray,
    ilines: np.ndarray,
    xlines: np.ndarray,
    *,
    range_hint: float,
    variogram: str,
    exact: bool,
    nugget: float,
) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(control_inlines) & np.isfinite(control_xlines) & np.isfinite(control_values)
    control_inlines = control_inlines[finite_mask]
    control_xlines = control_xlines[finite_mask]
    control_values = control_values[finite_mask]
    if control_values.size == 0:
        raise ValueError("No finite control points were provided for kriging.")

    if control_values.size == 1 or np.allclose(control_values, control_values[0]):
        constant_grid = np.full((ilines.size, xlines.size), float(control_values[0]), dtype=float)
        zero_var = np.zeros_like(constant_grid)
        return constant_grid, zero_var

    try:
        import gstools as gs
    except ImportError as exc:
        raise ImportError(
            "GSTools is required for kriging-based low-frequency model building. "
            "Please install 'gstools' from environment.yml."
        ) from exc

    model_name = variogram.lower()
    model_cls_map = {
        "spherical": gs.Spherical,
        "exponential": gs.Exponential,
        "gaussian": gs.Gaussian,
    }
    if model_name not in model_cls_map:
        raise ValueError(f"Unsupported variogram model: {variogram}")

    sill = float(np.var(control_values))
    if sill <= 0.0:
        sill = 1.0
    range_value = float(max(range_hint, 1.0))
    cov_model = model_cls_map[model_name](dim=2, var=sill, len_scale=range_value, nugget=float(max(nugget, 0.0)))
    krige = gs.krige.Ordinary(
        cov_model,
        cond_pos=[control_inlines, control_xlines],
        cond_val=control_values,
        exact=exact,
    )
    field, variance = krige((ilines, xlines), mesh_type="structured", return_var=True)
    return np.asarray(field, dtype=float), np.asarray(variance, dtype=float)


def make_lowfreq_ai_log(
    logset_twt: grid.LogSet,
    cutoff_hz: float = 10.0,
    order: int = 5,
) -> grid.Log:
    """对 TWT 域 AI 曲线做低通滤波，生成低频 AI。"""
    ai_log = logset_twt.AI
    fs = 1.0 / ai_log.sampling_rate
    nyquist = 0.5 * fs
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be within (0, {nyquist}), got {cutoff_hz}.")
    filtered = apply_butter_lowpass_filter(ai_log.values.astype(np.float64), highcut=cutoff_hz, fs=fs, order=order)
    return grid.Log(filtered.astype(np.float64), ai_log.basis.copy(), "twt", name="AI_lowfreq")


def save_autotie_artifacts(
    outputs: Any,
    output_dir: Path,
    well_name: str,
    *,
    inline: float,
    xline: float,
    horizon_times: Dict[str, float],
    x: Optional[float] = None,
    y: Optional[float] = None,
    lowfreq_cutoff_hz: float = 10.0,
    lowfreq_filter_order: int = 5,
) -> WellTieArtifact:
    """保存井震标定后的低频模型输入成果，并返回可直接用于建模的对象。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lowfreq_ai_log = make_lowfreq_ai_log(
        outputs.logset_twt,
        cutoff_hz=lowfreq_cutoff_hz,
        order=lowfreq_filter_order,
    )

    optimized_td_df = outputs.table.table.copy()
    wavelet_df = outputs.wavelet.series.reset_index()
    logset_twt_df = outputs.logset_twt.df.reset_index()
    lowfreq_ai_df = lowfreq_ai_log.series.reset_index()
    lowfreq_ai_df.columns = ["twt", "ai"]
    horizon_times_df = pd.DataFrame(
        {"horizon_name": list(horizon_times.keys()), "twt": [float(v) for v in horizon_times.values()]}
    )

    optimized_td_df.to_csv(output_dir / "optimized_td.csv", index=False)
    wavelet_df.to_csv(output_dir / "wavelet.csv", index=False)
    logset_twt_df.to_csv(output_dir / "logset_twt.csv", index=False)
    lowfreq_ai_df.to_csv(output_dir / "lowfreq_ai.csv", index=False)
    horizon_times_df.to_csv(output_dir / "well_horizon_times.csv", index=False)

    summary = {
        "well_name": well_name,
        "inline": float(inline),
        "xline": float(xline),
        "x": None if x is None else float(x),
        "y": None if y is None else float(y),
        "lowfreq_cutoff_hz": float(lowfreq_cutoff_hz),
        "lowfreq_filter_order": int(lowfreq_filter_order),
        "n_log_samples": int(lowfreq_ai_df.shape[0]),
        "twt_min_s": float(lowfreq_ai_df["twt"].min()),
        "twt_max_s": float(lowfreq_ai_df["twt"].max()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return WellTieArtifact(
        well_name=well_name,
        inline=float(inline),
        xline=float(xline),
        x=None if x is None else float(x),
        y=None if y is None else float(y),
        horizon_times={key: float(value) for key, value in horizon_times.items()},
        lowfreq_ai=lowfreq_ai_df,
        optimized_td=optimized_td_df,
        wavelet=wavelet_df,
        logset_twt=logset_twt_df,
        summary=summary,
    )


def build_low_frequency_model(
    target_layer: TargetLayer,
    wells: list[WellTieArtifact],
    *,
    n_slices: int = 32,
    variogram: str = "spherical",
    exact: bool = True,
    nugget: float = 0.0,
) -> LowFrequencyModelResult:
    """基于层位约束比例切片与 ordinary kriging 构建低频 AI 体。"""
    if n_slices < 2:
        raise ValueError(f"n_slices must be >= 2, got {n_slices}.")
    if not wells:
        raise ValueError("wells must contain at least one WellTieArtifact.")

    validated_wells = [_validate_well_artifact(well, target_layer.horizon_names) for well in wells]
    ilines = _build_line_axis(
        float(target_layer.geometry["inline_min"]),
        float(target_layer.geometry["inline_max"]),
        float(target_layer.geometry["inline_step"]),
    )
    xlines = _build_line_axis(
        float(target_layer.geometry["xline_min"]),
        float(target_layer.geometry["xline_max"]),
        float(target_layer.geometry["xline_step"]),
    )
    samples = _build_sample_axis(target_layer.geometry)
    n_il, n_xl, n_sample = ilines.size, xlines.size, samples.size

    volume = np.full((n_il, n_xl, n_sample), np.nan, dtype=np.float32)
    variance_volume = np.full((n_il, n_xl, n_sample), np.nan, dtype=np.float32)
    slice_u = np.linspace(0.0, 1.0, n_slices, dtype=float)

    well_inlines = np.asarray([well.inline for well in validated_wells], dtype=float)
    well_xlines = np.asarray([well.xline for well in validated_wells], dtype=float)
    range_hint = _nearest_neighbor_range(well_inlines, well_xlines)
    kriging_mode = "ordinary_kriging" if len(validated_wells) >= 2 else "single_well_constant"
    is_degenerate = len(validated_wells) < 2

    for top_name, bottom_name in target_layer.iter_zones():
        top_grid, bottom_grid = target_layer.get_zone_sample_index_grids((top_name, bottom_name))
        zone_slice_values = np.full((n_slices, n_il, n_xl), np.nan, dtype=float)
        zone_slice_variance = np.full((n_slices, n_il, n_xl), np.nan, dtype=float)

        for slice_idx, u in enumerate(slice_u):
            control_values = []
            for well in validated_wells:
                t_top = well.horizon_times[top_name]
                t_bottom = well.horizon_times[bottom_name]
                if not np.isfinite(t_top) or not np.isfinite(t_bottom) or t_bottom <= t_top:
                    continue
                t_slice = (1.0 - u) * t_top + u * t_bottom
                control_values.append(_interpolate_ai_at_time(well.lowfreq_ai, t_slice))

            if len(control_values) != len(validated_wells):
                raise ValueError(
                    f"Failed to extract finite control values for zone '{top_name}' -> '{bottom_name}' at u={u:.3f}."
                )

            control_array = np.asarray(control_values, dtype=float)
            field, variance = _krige_slice_on_line_domain(
                well_inlines,
                well_xlines,
                control_array,
                ilines,
                xlines,
                range_hint=range_hint,
                variogram=variogram,
                exact=exact,
                nugget=nugget,
            )
            zone_slice_values[slice_idx] = field
            zone_slice_variance[slice_idx] = variance

        for i_il in range(n_il):
            for i_xl in range(n_xl):
                t_top = top_grid[i_il, i_xl]
                t_bottom = bottom_grid[i_il, i_xl]
                if not np.isfinite(t_top) or not np.isfinite(t_bottom) or t_bottom <= t_top:
                    continue

                idx_top = max(0, int(np.round(t_top)))
                idx_bottom = min(n_sample - 1, int(np.round(t_bottom)))
                if idx_bottom < idx_top:
                    continue

                local_indices = np.arange(idx_top, idx_bottom + 1, dtype=float)
                denom = float(t_bottom - t_top)
                if denom <= 0.0:
                    continue
                u_local = np.clip((local_indices - float(t_top)) / denom, 0.0, 1.0)
                local_values = np.interp(u_local, slice_u, zone_slice_values[:, i_il, i_xl])
                local_variance = np.interp(u_local, slice_u, zone_slice_variance[:, i_il, i_xl])
                volume[i_il, i_xl, idx_top : idx_bottom + 1] = local_values.astype(np.float32)
                variance_volume[i_il, i_xl, idx_top : idx_bottom + 1] = local_variance.astype(np.float32)

    for i_il in range(n_il):
        for i_xl in range(n_xl):
            trace = volume[i_il, i_xl]
            finite = np.isfinite(trace)
            if not np.any(finite):
                continue
            valid_indices = np.flatnonzero(finite)
            first_idx = int(valid_indices[0])
            last_idx = int(valid_indices[-1])
            if first_idx > 0:
                trace[:first_idx] = trace[first_idx]
                variance_volume[i_il, i_xl, :first_idx] = variance_volume[i_il, i_xl, first_idx]
            if last_idx < n_sample - 1:
                trace[last_idx + 1 :] = trace[last_idx]
                variance_volume[i_il, i_xl, last_idx + 1 :] = variance_volume[i_il, i_xl, last_idx]

    metadata = {
        "backend": "gstools",
        "slice_mode": "proportional",
        "kriging_type": kriging_mode,
        "variogram": variogram,
        "exact": bool(exact),
        "nugget": float(nugget),
        "n_slices": int(n_slices),
        "range_hint": float(range_hint),
        "coord_system": "inline_xline",
        "horizon_names": list(target_layer.horizon_names),
        "zone_names": [list(zone) for zone in target_layer.iter_zones()],
        "well_names": [well.well_name for well in validated_wells],
        "is_degenerate": bool(is_degenerate),
        "degradation_reason": "single_well_constant_fill" if len(validated_wells) == 1 else (
            "heuristic_range" if len(validated_wells) < 3 else None
        ),
        "variance_volume_included": True,
    }

    return LowFrequencyModelResult(
        volume=volume,
        variance_volume=variance_volume,
        geometry=dict(target_layer.geometry),
        ilines=ilines,
        xlines=xlines,
        samples=samples,
        metadata=metadata,
        wells=validated_wells,
    )


def save_low_frequency_model_package(
    result: LowFrequencyModelResult,
    output_dir: Path,
) -> Path:
    """保存低频模型体、几何和 metadata。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "lfm_volume.npy", result.volume)
    np.save(output_dir / "kriging_variance.npy", result.variance_volume)
    np.savez(
        output_dir / "geometry.npz",
        ilines=result.ilines,
        xlines=result.xlines,
        samples=result.samples,
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(result.metadata, f, ensure_ascii=False, indent=2)

    wells_dir = output_dir / "wells"
    wells_dir.mkdir(exist_ok=True)
    for well in result.wells:
        well_dir = wells_dir / well.well_name
        well_dir.mkdir(exist_ok=True)
        lowfreq_ai_df = _normalize_lowfreq_ai_df(well.lowfreq_ai)
        lowfreq_ai_df.to_csv(well_dir / "lowfreq_ai.csv", index=False)
        pd.DataFrame(
            {"horizon_name": list(well.horizon_times.keys()), "twt": list(well.horizon_times.values())}
        ).to_csv(well_dir / "well_horizon_times.csv", index=False)
        if well.optimized_td is not None:
            well.optimized_td.to_csv(well_dir / "optimized_td.csv", index=False)
        if well.wavelet is not None:
            well.wavelet.to_csv(well_dir / "wavelet.csv", index=False)
        if well.logset_twt is not None:
            well.logset_twt.to_csv(well_dir / "logset_twt.csv", index=False)

        well_summary = {
            "well_name": well.well_name,
            "inline": well.inline,
            "xline": well.xline,
            "x": well.x,
            "y": well.y,
            "horizon_names": list(well.horizon_times.keys()),
        }
        if well.summary is not None:
            well_summary["autotie_summary"] = well.summary
        with (well_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(well_summary, f, ensure_ascii=False, indent=2)

    return output_dir
