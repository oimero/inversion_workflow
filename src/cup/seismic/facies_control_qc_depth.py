"""Interactive QC helpers for depth-domain AI LFM facies controls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cup.seismic.facies_control_depth import (
    FaciesControlPoint,
    apply_depth_facies_controls,
    build_target_layer_from_lfm_metadata,
    build_trace_xy_grids,
    locate_control_zone,
    raised_cosine_weight,
)
from cup.seismic.survey import open_survey
from cup.utils.io import load_yaml_config, resolve_relative_path
from cup.utils.statistics import normalized_cross_correlation, normalized_mae, rms
from cup.well.wavelet import load_wavelet_csv, make_wavelet
from ginn_depth.data import DepthLfmVolume, load_dynamic_gain_depth_model, load_lfm_depth_npz
from ginn_depth.physics import DepthForwardModel


DEFAULT_INFLUENCE_WEIGHT_THRESHOLD = 0.01


@dataclass
class FaciesControlQCContext:
    repo_root: Path
    ai_lfm: DepthLfmVolume
    vp_lfm: DepthLfmVolume
    dynamic_gain: DepthLfmVolume | None
    survey: Any
    target_layer: Any
    x_grid: np.ndarray
    y_grid: np.ndarray
    wavelet_time_s: np.ndarray
    wavelet_amp: np.ndarray
    forward_model: DepthForwardModel
    gain_mode: str
    seismic_file: Path
    common_config_path: Path
    train_config_path: Path


@dataclass
class LocalControlResult:
    point: FaciesControlPoint
    inline: float
    xline: float
    il_idx: int
    xl_idx: int
    il_slice: slice
    xl_slice: slice
    z_slice: slice
    ilines_window: np.ndarray
    xlines_window: np.ndarray
    source_ai_full_window: np.ndarray
    controlled_ai_full_window: np.ndarray
    vp_full_window: np.ndarray
    dynamic_gain_full_window: np.ndarray | None
    depth_window: np.ndarray
    source_ai_window: np.ndarray
    controlled_ai_window: np.ndarray
    diff_window: np.ndarray
    weight_window: np.ndarray
    vp_window: np.ndarray
    dynamic_gain_window: np.ndarray | None
    x_window: np.ndarray
    y_window: np.ndarray
    zone_top_window: np.ndarray
    zone_bottom_window: np.ndarray
    qc_df: pd.DataFrame
    warning: str | None = None


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward to ``src`` and ``scripts``."""
    path = Path.cwd() if start is None else Path(start)
    path = path.resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find repo root from {path}.")


def _resolve_repo_path(value: str | Path | None, *, repo_root: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return resolve_relative_path(value, root=repo_root)


def _resolve_depth_wavelet(train_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source = str(train_cfg.get("wavelet_source", "precomputed_wavelet"))
    if source == "precomputed_wavelet":
        wavelet_file = train_cfg.get("wavelet_file")
        if not wavelet_file:
            raise ValueError("wavelet_file is required when wavelet_source='precomputed_wavelet'.")
        return load_wavelet_csv(Path(str(wavelet_file)))
    if source == "ricker_wavelet":
        return make_wavelet(
            wavelet_type=str(train_cfg.get("wavelet_type", "ricker")),
            freq=float(train_cfg.get("wavelet_freq", 25.0)),
            dt=float(train_cfg.get("wavelet_dt", 0.001)),
            length=int(train_cfg.get("wavelet_length", 201)),
            gain=1.0,
        )
    raise ValueError(f"Unsupported wavelet_source: {source!r}.")


def load_qc_context(
    *,
    common_config_path: str | Path = "experiments/common_depth.yaml",
    train_config_path: str | Path = "experiments/ginn_depth/train.yaml",
    source_ai_lfm_file: str | Path | None = None,
    vp_lfm_file: str | Path | None = None,
    wavelet_file: str | Path | None = None,
    dynamic_gain_file: str | Path | None = None,
    seismic_file: str | Path | None = None,
    device: str = "cpu",
) -> FaciesControlQCContext:
    """Load all heavy objects used by the interactive QC notebook."""
    repo_root = find_repo_root()
    common_config_path = resolve_relative_path(common_config_path, root=repo_root)
    train_config_path = resolve_relative_path(train_config_path, root=repo_root)
    common_cfg = load_yaml_config(common_config_path, base_dir=repo_root)
    train_cfg = load_yaml_config(train_config_path, base_dir=repo_root)
    data_root = resolve_relative_path(str(common_cfg.get("data_root", "data")), root=repo_root)

    facies_cfg = common_cfg.get("lfm_facies_control_depth", {}) or {}
    source_ai_lfm_file = (
        source_ai_lfm_file
        or facies_cfg.get("source_ai_lfm_file")
        or train_cfg.get("ai_lfm_file")
    )
    vp_lfm_file = vp_lfm_file or train_cfg.get("vp_lfm_file")
    wavelet_file = wavelet_file or train_cfg.get("wavelet_file")
    dynamic_gain_file = dynamic_gain_file or train_cfg.get("dynamic_gain_model")
    if seismic_file is None:
        segy_cfg = common_cfg["segy"]
        seismic_file = resolve_relative_path(str(segy_cfg["file"]), root=data_root)
    else:
        seismic_file = resolve_relative_path(seismic_file, root=repo_root)

    source_ai_lfm_path = _resolve_repo_path(source_ai_lfm_file, repo_root=repo_root)
    vp_lfm_path = _resolve_repo_path(vp_lfm_file, repo_root=repo_root)
    wavelet_path = _resolve_repo_path(wavelet_file, repo_root=repo_root)
    dynamic_gain_path = _resolve_repo_path(dynamic_gain_file, repo_root=repo_root)
    if source_ai_lfm_path is None or vp_lfm_path is None:
        raise ValueError("source AI LFM and Vp LFM paths are required.")

    ai_lfm = load_lfm_depth_npz(source_ai_lfm_path)
    vp_lfm = load_lfm_depth_npz(vp_lfm_path)
    if ai_lfm.shape != vp_lfm.shape:
        raise ValueError(f"AI/Vp shape mismatch: ai={ai_lfm.shape}, vp={vp_lfm.shape}.")

    dynamic_gain = None
    gain_mode = "unit_gain"
    if train_cfg.get("gain_source") == "dynamic_gain_model" and dynamic_gain_path is not None and dynamic_gain_path.exists():
        dynamic_gain = load_dynamic_gain_depth_model(dynamic_gain_path)
        if dynamic_gain.shape != ai_lfm.shape:
            raise ValueError(f"Dynamic gain shape mismatch: gain={dynamic_gain.shape}, ai={ai_lfm.shape}.")
        gain_mode = "dynamic_gain_model"

    if wavelet_path is not None:
        train_cfg = dict(train_cfg)
        train_cfg["wavelet_source"] = "precomputed_wavelet"
        train_cfg["wavelet_file"] = str(wavelet_path)
    wavelet_time_s, wavelet_amp = _resolve_depth_wavelet(train_cfg)
    if gain_mode != "dynamic_gain_model":
        fixed_gain = train_cfg.get("fixed_gain")
        if fixed_gain is not None and str(fixed_gain).strip() != "":
            wavelet_amp = np.asarray(wavelet_amp, dtype=np.float32) * float(fixed_gain)
            gain_mode = "fixed_gain"

    segy_cfg = common_cfg["segy"]
    survey = open_survey(
        seismic_file,
        seismic_type="segy",
        segy_options={
            "iline": int(segy_cfg["iline_byte"]),
            "xline": int(segy_cfg["xline_byte"]),
            "istep": int(segy_cfg["istep"]),
            "xstep": int(segy_cfg["xstep"]),
        },
    )
    target_layer = build_target_layer_from_lfm_metadata(ai_lfm.metadata, ai_lfm.geometry)
    x_grid, y_grid = build_trace_xy_grids(survey, ai_lfm.ilines, ai_lfm.xlines)
    forward_model = DepthForwardModel(
        np.asarray(wavelet_time_s, dtype=np.float32),
        np.asarray(wavelet_amp, dtype=np.float32),
        depth_axis_m=ai_lfm.samples.astype(np.float32),
        amplitude_threshold=float(train_cfg.get("wavelet_amplitude_threshold", 0.0)),
    ).to(torch.device(device)).eval()
    return FaciesControlQCContext(
        repo_root=repo_root,
        ai_lfm=ai_lfm,
        vp_lfm=vp_lfm,
        dynamic_gain=dynamic_gain,
        survey=survey,
        target_layer=target_layer,
        x_grid=x_grid,
        y_grid=y_grid,
        wavelet_time_s=np.asarray(wavelet_time_s, dtype=np.float32),
        wavelet_amp=np.asarray(wavelet_amp, dtype=np.float32),
        forward_model=forward_model,
        gain_mode=gain_mode,
        seismic_file=seismic_file,
        common_config_path=common_config_path,
        train_config_path=train_config_path,
    )


def make_control_point(
    *,
    name: str,
    x: float,
    y: float,
    depth_m: float,
    radius_xy_m: float,
    radius_z_m: float,
    target_ai: float,
    strength: float = 1.0,
) -> FaciesControlPoint:
    return FaciesControlPoint(
        name=name,
        x=float(x),
        y=float(y),
        depth_m=float(depth_m),
        radius_xy_m=float(radius_xy_m),
        radius_z_m=float(radius_z_m),
        target_ai=float(target_ai),
        strength=float(strength),
    )


def extract_local_window_by_xy_radius(
    context: FaciesControlQCContext,
    point: FaciesControlPoint,
    *,
    section_scale: float = 1.2,
) -> tuple[float, float, int, int, slice, slice, slice, str | None]:
    """Extract local indices around a control point using true XY distance."""
    inline, xline = context.survey.coord_to_line(point.x, point.y)
    il_idx, xl_idx = context.ai_lfm.nearest_indices(inline, xline)
    display_radius_xy = max(float(section_scale) * 2.0 * point.radius_xy_m, point.radius_xy_m)
    d_xy_center = np.hypot(context.x_grid - point.x, context.y_grid - point.y)
    trace_mask = d_xy_center <= display_radius_xy
    if not np.any(trace_mask):
        raise ValueError("No traces found within display radius.")
    il_hits, xl_hits = np.where(trace_mask)
    il_slice = slice(max(int(il_hits.min()) - 1, 0), min(int(il_hits.max()) + 2, context.ai_lfm.ilines.size))
    xl_slice = slice(max(int(xl_hits.min()) - 1, 0), min(int(xl_hits.max()) + 2, context.ai_lfm.xlines.size))
    depth_min = point.depth_m - float(section_scale) * point.radius_z_m
    depth_max = point.depth_m + float(section_scale) * point.radius_z_m
    z_hits = np.where((context.ai_lfm.samples >= depth_min) & (context.ai_lfm.samples <= depth_max))[0]
    if z_hits.size == 0:
        z_idx = int(np.argmin(np.abs(context.ai_lfm.samples - point.depth_m)))
        z_slice = slice(max(z_idx - 5, 0), min(z_idx + 6, context.ai_lfm.samples.size))
    else:
        z_slice = slice(max(int(z_hits.min()) - 1, 0), min(int(z_hits.max()) + 2, context.ai_lfm.samples.size))

    warning = None
    if il_slice.start == 0 or il_slice.stop == context.ai_lfm.ilines.size or xl_slice.start == 0 or xl_slice.stop == context.ai_lfm.xlines.size:
        warning = "显示窗口触达工区边界；若影响范围被截断，请减小半径或检查点位。"
    return float(inline), float(xline), il_idx, xl_idx, il_slice, xl_slice, z_slice, warning


def _window_target_layer(target_layer: Any, il_slice: slice, xl_slice: slice, samples: np.ndarray) -> Any:
    from cup.seismic.target_layer import TargetLayer

    out = object.__new__(TargetLayer)
    out.geometry = dict(target_layer.geometry)
    out.geometry["n_il"] = int(il_slice.stop - il_slice.start)
    out.geometry["inline_min"] = float(target_layer.ilines[il_slice][0])
    out.geometry["inline_max"] = float(target_layer.ilines[il_slice][-1])
    out.geometry["n_xl"] = int(xl_slice.stop - xl_slice.start)
    out.geometry["xline_min"] = float(target_layer.xlines[xl_slice][0])
    out.geometry["xline_max"] = float(target_layer.xlines[xl_slice][-1])
    out.geometry["n_sample"] = int(samples.size)
    out.geometry["sample_min"] = float(samples[0])
    out.geometry["sample_max"] = float(samples[-1])
    out.horizon_names = list(target_layer.horizon_names)
    out._il_axis = np.asarray(target_layer.ilines[il_slice], dtype=np.float64)
    out._xl_axis = np.asarray(target_layer.xlines[xl_slice], dtype=np.float64)
    out._sample_axis = np.asarray(samples, dtype=np.float64)
    out._horizon_grids = {
        name: target_layer.get_horizon_grid(name)[il_slice, xl_slice]
        for name in out.horizon_names
    }
    return out


def apply_single_control_to_window(
    context: FaciesControlQCContext,
    point: FaciesControlPoint,
    *,
    section_scale: float = 1.2,
) -> LocalControlResult:
    """Apply one control point to a local AI LFM window for interactive QC."""
    inline, xline, il_idx, xl_idx, il_slice, xl_slice, z_slice, warning = extract_local_window_by_xy_radius(
        context,
        point,
        section_scale=section_scale,
    )
    ai_window = context.ai_lfm.volume[il_slice, xl_slice, :].astype(np.float32, copy=True)
    vp_window = context.vp_lfm.volume[il_slice, xl_slice, :].astype(np.float32, copy=False)
    gain_window = None
    if context.dynamic_gain is not None:
        gain_window = context.dynamic_gain.volume[il_slice, xl_slice, :].astype(np.float32, copy=False)
    x_window = context.x_grid[il_slice, xl_slice]
    y_window = context.y_grid[il_slice, xl_slice]
    window_target_layer = _window_target_layer(context.target_layer, il_slice, xl_slice, context.ai_lfm.samples)
    controlled, qc_df = apply_depth_facies_controls(
        ai_window,
        ilines=context.ai_lfm.ilines[il_slice],
        xlines=context.ai_lfm.xlines[xl_slice],
        samples=context.ai_lfm.samples,
        target_layer=window_target_layer,
        survey=context.survey,
        control_points=[point],
        x_grid=x_window,
        y_grid=y_window,
    )
    weight_window = compute_weight_window(
        point,
        samples=context.ai_lfm.samples,
        target_layer=window_target_layer,
        x_grid=x_window,
        y_grid=y_window,
        zone_top=str(qc_df.iloc[0]["zone_top"]),
        zone_bottom=str(qc_df.iloc[0]["zone_bottom"]),
    )
    zone_top_grid, zone_bottom_grid = get_zone_depth_grids(
        window_target_layer,
        zone_top=str(qc_df.iloc[0]["zone_top"]),
        zone_bottom=str(qc_df.iloc[0]["zone_bottom"]),
    )
    return LocalControlResult(
        point=point,
        inline=inline,
        xline=xline,
        il_idx=il_idx,
        xl_idx=xl_idx,
        il_slice=il_slice,
        xl_slice=xl_slice,
        z_slice=z_slice,
        ilines_window=context.ai_lfm.ilines[il_slice].copy(),
        xlines_window=context.ai_lfm.xlines[xl_slice].copy(),
        source_ai_full_window=ai_window.copy(),
        controlled_ai_full_window=controlled.copy(),
        vp_full_window=vp_window.copy(),
        dynamic_gain_full_window=None if gain_window is None else gain_window.copy(),
        depth_window=context.ai_lfm.samples[z_slice].copy(),
        source_ai_window=ai_window[:, :, z_slice].copy(),
        controlled_ai_window=controlled[:, :, z_slice].copy(),
        diff_window=(controlled - ai_window)[:, :, z_slice].copy(),
        weight_window=weight_window[:, :, z_slice].copy(),
        vp_window=vp_window[:, :, z_slice].copy(),
        dynamic_gain_window=None if gain_window is None else gain_window[:, :, z_slice].copy(),
        x_window=x_window.copy(),
        y_window=y_window.copy(),
        zone_top_window=zone_top_grid.copy(),
        zone_bottom_window=zone_bottom_grid.copy(),
        qc_df=qc_df,
        warning=warning,
    )


def compute_weight_window(
    point: FaciesControlPoint,
    *,
    samples: np.ndarray,
    target_layer: Any,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    zone_top: str,
    zone_bottom: str,
) -> np.ndarray:
    """Compute the facies-control weight volume for a local window."""
    samples = np.asarray(samples, dtype=np.float64)
    zone_top_grid, zone_bottom_grid = get_zone_depth_grids(
        target_layer,
        zone_top=zone_top,
        zone_bottom=zone_bottom,
    )
    d_xy = np.hypot(np.asarray(x_grid) - point.x, np.asarray(y_grid) - point.y)
    w_xy = raised_cosine_weight(d_xy / point.radius_xy_m)
    w_z = raised_cosine_weight(np.abs(samples - point.depth_m) / point.radius_z_m)
    weight = w_xy[:, :, None] * w_z[None, None, :] * float(point.strength)
    in_layer = (samples[None, None, :] >= zone_top_grid[:, :, None]) & (
        samples[None, None, :] <= zone_bottom_grid[:, :, None]
    )
    return np.where(in_layer, weight, 0.0).astype(np.float32)


def get_zone_depth_grids(target_layer: Any, *, zone_top: str, zone_bottom: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered top/bottom horizon grids for one target-layer zone."""
    top = target_layer.get_horizon_grid(zone_top)
    bottom = target_layer.get_horizon_grid(zone_bottom)
    return np.minimum(top, bottom), np.maximum(top, bottom)


def _nearest_local_indices(result: LocalControlResult) -> tuple[int, int]:
    return result.il_idx - int(result.il_slice.start), result.xl_idx - int(result.xl_slice.start)


def forward_qc_at_control_trace(
    context: FaciesControlQCContext,
    result: LocalControlResult,
    *,
    influence_weight_threshold: float = DEFAULT_INFLUENCE_WEIGHT_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Forward model before/after AI at the control trace and compare to real seismic."""
    li, lj = _nearest_local_indices(result)
    ai_before = result.source_ai_full_window[li, lj, :]
    ai_after = result.controlled_ai_full_window[li, lj, :]
    vp = result.vp_full_window[li, lj, :]
    full_depth = context.ai_lfm.samples
    display_depth = context.ai_lfm.samples[result.z_slice]
    device = context.forward_model.matrix_builder.wavelet_amp.device
    with torch.no_grad():
        before = context.forward_model(
            torch.from_numpy(ai_before[None, None, :]).float().to(device),
            torch.from_numpy(vp[None, :]).float().to(device),
            depth_axis_m=full_depth.astype(np.float32),
        ).squeeze().cpu().numpy()
        after = context.forward_model(
            torch.from_numpy(ai_after[None, None, :]).float().to(device),
            torch.from_numpy(vp[None, :]).float().to(device),
            depth_axis_m=full_depth.astype(np.float32),
        ).squeeze().cpu().numpy()
    gain = None
    if result.dynamic_gain_full_window is not None:
        gain = result.dynamic_gain_full_window[li, lj, :]
        before = before * gain
        after = after * gain

    before = before[result.z_slice]
    after = after[result.z_slice]
    gain_display = np.ones_like(display_depth) if gain is None else gain[result.z_slice]
    # Raised-cosine weights approach zero at the radius boundary; 0.01 keeps
    # the metric focused on visibly influenced samples while retaining the lens edge.
    influence_mask = result.weight_window[li, lj, :] > float(influence_weight_threshold)

    real_trace = context.survey.import_seismic_at_well(
        result.point.x,
        result.point.y,
        sample_start=float(display_depth[0]),
        sample_end=float(display_depth[-1]),
        domain="depth",
    )
    real_depth = np.asarray(real_trace.basis, dtype=np.float64)
    real_values = np.asarray(real_trace.values, dtype=np.float64)
    real_interp = np.interp(display_depth, real_depth, real_values, left=np.nan, right=np.nan)
    mask = np.isfinite(real_interp) & np.isfinite(before) & np.isfinite(after)
    influence_valid_mask = mask & influence_mask

    before_rms_scaled, before_scale = rms_match(before, real_interp, mask)
    after_rms_scaled, after_scale = rms_match(after, real_interp, mask)
    before_influence_rms_scaled, before_influence_scale = rms_match(before, real_interp, influence_valid_mask)
    after_influence_rms_scaled, after_influence_scale = rms_match(after, real_interp, influence_valid_mask)
    rows = [
        _waveform_metrics("physical_before", "display_window", real_interp, before, mask),
        _waveform_metrics("physical_after", "display_window", real_interp, after, mask),
        _waveform_metrics("rms_match_before", "display_window", real_interp, before_rms_scaled, mask, scale=before_scale),
        _waveform_metrics("rms_match_after", "display_window", real_interp, after_rms_scaled, mask, scale=after_scale),
        _waveform_metrics("physical_before", "influence_only", real_interp, before, influence_valid_mask),
        _waveform_metrics("physical_after", "influence_only", real_interp, after, influence_valid_mask),
        _waveform_metrics(
            "rms_match_before",
            "influence_only",
            real_interp,
            before_influence_rms_scaled,
            influence_valid_mask,
            scale=before_influence_scale,
        ),
        _waveform_metrics(
            "rms_match_after",
            "influence_only",
            real_interp,
            after_influence_rms_scaled,
            influence_valid_mask,
            scale=after_influence_scale,
        ),
    ]
    data = {
        "depth": display_depth,
        "real": real_interp,
        "synthetic_before": before,
        "synthetic_after": after,
        "synthetic_before_rms_match": before_rms_scaled,
        "synthetic_after_rms_match": after_rms_scaled,
        "gain": gain_display,
        "influence_mask": influence_mask,
        "influence_weight_threshold": float(influence_weight_threshold),
    }
    return pd.DataFrame.from_records(rows), data


def rms_match(values: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    ref_rms = rms(np.asarray(reference)[mask])
    val_rms = rms(np.asarray(values)[mask])
    if not np.isfinite(ref_rms) or not np.isfinite(val_rms) or val_rms <= 0.0:
        return np.asarray(values, dtype=np.float64).copy(), float("nan")
    scale = ref_rms / val_rms
    return np.asarray(values, dtype=np.float64) * scale, float(scale)


def _waveform_metrics(
    name: str,
    window: str,
    reference: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray,
    *,
    scale: float = 1.0,
) -> dict[str, float | str]:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(ref) & np.isfinite(est)
    if int(valid.sum()) == 0:
        return {"metric_set": name, "window": window, "n": 0, "scale": scale}
    ref_v = ref[valid]
    est_v = est[valid]
    return {
        "metric_set": name,
        "window": window,
        "n": int(valid.sum()),
        "corr": normalized_cross_correlation(ref_v, est_v),
        "nmae": normalized_mae(ref_v, est_v),
        "rmse": float(np.sqrt(np.mean((ref_v - est_v) ** 2))),
        "rms_ratio": float(rms(est_v) / rms(ref_v)) if rms(ref_v) > 0.0 else float("nan"),
        "peak_amp_ratio": float(np.nanmax(np.abs(est_v)) / np.nanmax(np.abs(ref_v)))
        if np.nanmax(np.abs(ref_v)) > 0.0
        else float("nan"),
        "scale": float(scale),
    }


def plot_control_sections(result: LocalControlResult) -> None:
    """Plot inline/xline before, after, delta, and weight sections through the control point."""
    li, lj = _nearest_local_indices(result)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    _plot_section_row(
        axes[0],
        result.source_ai_window[:, lj, :],
        result.controlled_ai_window[:, lj, :],
        result.diff_window[:, lj, :],
        result.weight_window[:, lj, :],
        title_prefix="Inline direction",
        x_values=result.ilines_window,
        center_x=float(result.ilines_window[li]),
        x_label="inline",
        depth_values=result.depth_window,
        control_depth=result.point.depth_m,
        zone_top=result.zone_top_window[:, lj],
        zone_bottom=result.zone_bottom_window[:, lj],
    )
    _plot_section_row(
        axes[1],
        result.source_ai_window[li, :, :],
        result.controlled_ai_window[li, :, :],
        result.diff_window[li, :, :],
        result.weight_window[li, :, :],
        title_prefix="Xline direction",
        x_values=result.xlines_window,
        center_x=float(result.xlines_window[lj]),
        x_label="xline",
        depth_values=result.depth_window,
        control_depth=result.point.depth_m,
        zone_top=result.zone_top_window[li, :],
        zone_bottom=result.zone_bottom_window[li, :],
    )
    fig.suptitle(
        f"{result.point.name} | inline={result.inline:.1f}, xline={result.xline:.1f}, "
        f"Rxy={result.point.radius_xy_m:.0f} m, Rz={result.point.radius_z_m:.0f} m"
    )
    plt.show()
    if result.warning:
        print(f"WARNING: {result.warning}")
    print(result.qc_df.drop(columns=["horizon_values"], errors="ignore").to_string(index=False))


def _plot_section_row(
    axes: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    delta: np.ndarray,
    weight: np.ndarray,
    *,
    title_prefix: str,
    x_values: np.ndarray,
    center_x: float,
    x_label: str,
    depth_values: np.ndarray,
    control_depth: float,
    zone_top: np.ndarray,
    zone_bottom: np.ndarray,
) -> None:
    vmin = float(np.nanpercentile(before, 2.0))
    vmax = float(np.nanpercentile(before, 98.0))
    delta_clip = max(float(np.nanpercentile(np.abs(delta), 99.0)), 1.0)
    panels = [
        (before, "before AI", "viridis", vmin, vmax),
        (after, "after AI", "viridis", vmin, vmax),
        (delta, "after - before", "coolwarm", -delta_clip, delta_clip),
        (weight, "control weight", "magma", 0.0, max(float(np.nanmax(weight)), 1e-6)),
    ]
    x_values = np.asarray(x_values, dtype=np.float64)
    extent = [
        *_axis_edges(x_values),
        float(depth_values[-1]),
        float(depth_values[0]),
    ]
    for ax, (values, label, cmap, lo, hi) in zip(axes, panels):
        im = ax.imshow(values.T, aspect="auto", origin="upper", cmap=cmap, vmin=lo, vmax=hi, extent=extent)
        ax.axvline(center_x, color="white", lw=0.8, ls="--")
        ax.axhline(control_depth, color="white", lw=0.8, ls=":")
        ax.plot(x_values, zone_top, color="white", lw=1.1, ls="-", alpha=0.9)
        ax.plot(x_values, zone_bottom, color="white", lw=1.1, ls="-", alpha=0.9)
        ax.set_title(f"{title_prefix}: {label}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("depth (m)")
        plt.colorbar(im, ax=ax, shrink=0.8)


def _axis_edges(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 1.0
    if values.size == 1:
        return float(values[0] - 0.5), float(values[0] + 0.5)
    diffs = np.diff(values)
    return float(values[0] - 0.5 * diffs[0]), float(values[-1] + 0.5 * diffs[-1])


def plot_forward_qc(metrics: pd.DataFrame, waveforms: dict[str, np.ndarray], *, gain_mode: str) -> None:
    depth = waveforms["depth"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    axes[0].plot(waveforms["real"], depth, color="black", lw=1.2, label="real")
    axes[0].plot(waveforms["synthetic_before"], depth, color="tab:blue", lw=1.0, label="before synthetic")
    axes[0].plot(waveforms["synthetic_after"], depth, color="tab:red", lw=1.0, label="after synthetic")
    axes[0].invert_yaxis()
    axes[0].set_title(f"Physical amplitude ({gain_mode})")
    axes[0].set_xlabel("amplitude")
    axes[0].set_ylabel("depth (m)")
    axes[0].legend()

    axes[1].plot(waveforms["real"], depth, color="black", lw=1.2, label="real")
    axes[1].plot(waveforms["synthetic_before_rms_match"], depth, color="tab:blue", lw=1.0, label="before RMS match")
    axes[1].plot(waveforms["synthetic_after_rms_match"], depth, color="tab:red", lw=1.0, label="after RMS match")
    axes[1].invert_yaxis()
    axes[1].set_title("Local RMS matched")
    axes[1].set_xlabel("amplitude")
    axes[1].set_ylabel("depth (m)")
    axes[1].legend()
    for ax in axes:
        _shade_influence_depths(ax, depth, waveforms.get("influence_mask"))
    plt.show()
    print(metrics.to_string(index=False))


def _shade_influence_depths(ax: Any, depth: np.ndarray, influence_mask: np.ndarray | None) -> None:
    if influence_mask is None:
        return
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(influence_mask, dtype=bool)
    if depth.size == 0 or mask.size != depth.size or not np.any(mask):
        return
    for start, stop in _true_runs(mask):
        top = float(depth[start])
        bottom = float(depth[stop - 1])
        ax.axhspan(top, bottom, color="gold", alpha=0.16, lw=0, zorder=0)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def append_trial_record(
    output_csv: str | Path,
    result: LocalControlResult,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Append current parameters and QC metrics to a trial CSV."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metric_row = {
        f"{row['window']}_{row['metric_set']}_{key}": value
        for _, row in metrics.iterrows()
        for key, value in row.items()
        if key not in {"metric_set", "window"}
    }
    qc = result.qc_df.iloc[0].drop(labels=["horizon_values"], errors="ignore").to_dict()
    row = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **qc,
        **metric_row,
    }
    if output_csv.exists():
        df = pd.read_csv(output_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def trial_csv_path(repo_root: Path, control_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(control_name))
    return repo_root / "scripts" / "output" / "lfm_facies_control_depth_qc_trials" / f"{safe}.csv"


def read_notebook_run_summary(context: FaciesControlQCContext) -> dict[str, Any]:
    return {
        "repo_root": str(context.repo_root),
        "ai_shape": tuple(int(v) for v in context.ai_lfm.shape),
        "gain_mode": context.gain_mode,
        "seismic_file": str(context.seismic_file),
        "common_config_path": str(context.common_config_path),
        "train_config_path": str(context.train_config_path),
    }


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
