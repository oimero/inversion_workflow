"""Depth-domain facies-control utilities for AI low-frequency models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from cup.utils.io import resolve_relative_path
from cup.utils.statistics import normalized_cross_correlation, normalized_mae, rms

if TYPE_CHECKING:
    from ginn_depth.data import DepthLfmVolume
    from ginn_depth.physics import DepthForwardModel

SCHEMA_VERSION = "cup_depth_facies_control_v1"
REQUIRED_COLUMNS = {"x", "y", "depth_m", "radius_xy_m", "radius_z_m", "target_ai"}
OPTIONAL_COLUMNS = {"name", "strength"}
DEFAULT_INFLUENCE_WEIGHT_THRESHOLD = 0.01


class _SurveyLike(Protocol):
    def coord_to_line(self, x: float, y: float) -> tuple[float, float]: ...

    def line_to_coord(self, il_no: float, xl_no: float) -> tuple[float, float]: ...


@dataclass(frozen=True)
class FaciesControlPoint:
    name: str
    x: float
    y: float
    depth_m: float
    radius_xy_m: float
    radius_z_m: float
    target_ai: float
    strength: float = 1.0


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
    seis_rms: float
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


def load_depth_facies_control_points_csv(path: str | Path) -> list[FaciesControlPoint]:
    """Load and validate depth-domain facies control points from CSV."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Facies control CSV is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Facies control CSV is empty: {path}")

    points: list[FaciesControlPoint] = []
    for row_idx, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name or name.lower() == "nan":
            name = f"control_{row_idx + 1:03d}"
        strength = 1.0 if "strength" not in df.columns or pd.isna(row.get("strength")) else float(row["strength"])
        point = FaciesControlPoint(
            name=name,
            x=float(row["x"]),
            y=float(row["y"]),
            depth_m=float(row["depth_m"]),
            radius_xy_m=float(row["radius_xy_m"]),
            radius_z_m=float(row["radius_z_m"]),
            target_ai=float(row["target_ai"]),
            strength=strength,
        )
        validate_control_point(point)
        points.append(point)
    return points


def validate_control_point(point: FaciesControlPoint) -> None:
    """Validate numeric constraints for one control point."""
    values = {
        "x": point.x,
        "y": point.y,
        "depth_m": point.depth_m,
        "radius_xy_m": point.radius_xy_m,
        "radius_z_m": point.radius_z_m,
        "target_ai": point.target_ai,
        "strength": point.strength,
    }
    bad = [name for name, value in values.items() if not np.isfinite(float(value))]
    if bad:
        raise ValueError(f"Facies control point {point.name!r} has non-finite values: {bad}")
    if point.radius_xy_m <= 0.0:
        raise ValueError(f"Facies control point {point.name!r} radius_xy_m must be positive.")
    if point.radius_z_m <= 0.0:
        raise ValueError(f"Facies control point {point.name!r} radius_z_m must be positive.")
    if point.target_ai <= 0.0:
        raise ValueError(f"Facies control point {point.name!r} target_ai must be positive.")
    if not (0.0 <= point.strength <= 1.0):
        raise ValueError(f"Facies control point {point.name!r} strength must be within [0, 1].")


def raised_cosine_weight(normalized_distance: np.ndarray | float) -> np.ndarray:
    """Return a compact raised-cosine weight for normalized distance in [0, 1]."""
    d = np.asarray(normalized_distance, dtype=np.float64)
    weight = np.zeros_like(d, dtype=np.float64)
    inside = (d >= 0.0) & (d <= 1.0)
    weight[inside] = 0.5 * (1.0 + np.cos(np.pi * d[inside]))
    return weight


def build_trace_xy_grids(
    survey: _SurveyLike,
    ilines: np.ndarray,
    xlines: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build trace-center XY grids from survey line coordinates."""
    ilines = np.asarray(ilines, dtype=np.float64)
    xlines = np.asarray(xlines, dtype=np.float64)
    if ilines.ndim != 1 or xlines.ndim != 1 or ilines.size == 0 or xlines.size == 0:
        raise ValueError("ilines and xlines must be non-empty 1D arrays.")

    x0, y0 = survey.line_to_coord(float(ilines[0]), float(xlines[0]))
    if ilines.size > 1:
        x1, y1 = survey.line_to_coord(float(ilines[-1]), float(xlines[0]))
        dx_i = (x1 - x0) / float(ilines.size - 1)
        dy_i = (y1 - y0) / float(ilines.size - 1)
    else:
        dx_i = dy_i = 0.0
    if xlines.size > 1:
        x2, y2 = survey.line_to_coord(float(ilines[0]), float(xlines[-1]))
        dx_j = (x2 - x0) / float(xlines.size - 1)
        dy_j = (y2 - y0) / float(xlines.size - 1)
    else:
        dx_j = dy_j = 0.0

    i_idx = np.arange(ilines.size, dtype=np.float64)[:, None]
    j_idx = np.arange(xlines.size, dtype=np.float64)[None, :]
    x_grid = x0 + i_idx * dx_i + j_idx * dx_j
    y_grid = y0 + i_idx * dy_i + j_idx * dy_j
    return x_grid.astype(np.float64), y_grid.astype(np.float64)


def build_target_layer_from_lfm_metadata(
    metadata: dict[str, Any],
    geometry: dict[str, Any],
    *,
    qc_output_dir: str | Path | None = None,
) -> Any:
    """Rebuild a TargetLayer from horizon metadata stored in an AI LFM NPZ."""
    from cup.petrel.load import import_interpretation_petrel
    from cup.seismic.target_layer import TargetLayer

    horizons = metadata.get("horizons", [])
    if not isinstance(horizons, list) or len(horizons) < 2:
        raise ValueError("AI LFM metadata must contain at least two horizons.")

    raw_horizons: dict[str, pd.DataFrame] = {}
    horizon_names: list[str] = []
    for idx, item in enumerate(horizons):
        if not isinstance(item, dict) or not item.get("file"):
            raise ValueError(f"Invalid horizon metadata entry at index {idx}: {item!r}")
        name = str(item.get("name") or f"horizon_{idx}")
        horizon_names.append(name)
        raw_horizons[name] = import_interpretation_petrel(Path(str(item["file"])))

    tl_meta = metadata.get("target_layer", {})
    if not isinstance(tl_meta, dict):
        tl_meta = {}
    return TargetLayer(
        raw_horizon_dfs=raw_horizons,
        geometry=geometry,
        horizon_names=horizon_names,
        qc_output_dir=qc_output_dir,
        min_thickness=tl_meta.get("min_thickness"),
        nearest_distance_limit=tl_meta.get("nearest_distance_limit"),
        outlier_threshold=tl_meta.get("outlier_threshold"),
        outlier_min_neighbor_count=tl_meta.get("outlier_min_neighbor_count", 2),
    )


def locate_control_zone(
    target_layer: Any,
    *,
    inline: float,
    xline: float,
    depth_m: float,
) -> tuple[str, str, dict[str, float]]:
    """Locate the adjacent horizon pair containing a control point depth."""
    horizon_values = target_layer.get_interpretation_values_at_location(inline, xline)
    names = list(target_layer.horizon_names)
    for top_name, bottom_name in zip(names[:-1], names[1:]):
        top = float(horizon_values[top_name])
        bottom = float(horizon_values[bottom_name])
        if not np.isfinite(top) or not np.isfinite(bottom):
            continue
        lo, hi = min(top, bottom), max(top, bottom)
        if lo <= float(depth_m) <= hi:
            return top_name, bottom_name, {name: float(horizon_values[name]) for name in names}
    raise ValueError(
        f"Control depth {float(depth_m):.3f} m is outside all target-layer zones at "
        f"inline={inline:.3f}, xline={xline:.3f}."
    )


def apply_depth_facies_controls(
    volume: np.ndarray,
    *,
    ilines: np.ndarray,
    xlines: np.ndarray,
    samples: np.ndarray,
    target_layer: Any,
    survey: _SurveyLike,
    control_points: list[FaciesControlPoint],
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    error_on_empty: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply facies controls to an AI volume using layer-clipped log-AI blending."""
    ai = np.asarray(volume, dtype=np.float32)
    if ai.ndim != 3:
        raise ValueError(f"Expected volume ndim=3, got {ai.ndim}.")
    ilines = np.asarray(ilines, dtype=np.float64)
    xlines = np.asarray(xlines, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    expected_shape = (ilines.size, xlines.size, samples.size)
    if ai.shape != expected_shape:
        raise ValueError(f"Volume shape {ai.shape} does not match axes {expected_shape}.")
    if np.any(~np.isfinite(ai)) or np.any(ai <= 0.0):
        raise ValueError("AI volume must contain only finite positive values.")

    if x_grid is None or y_grid is None:
        x_grid, y_grid = build_trace_xy_grids(survey, ilines, xlines)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    if x_grid.shape != ai.shape[:2] or y_grid.shape != ai.shape[:2]:
        raise ValueError("x_grid/y_grid shape must match volume trace shape.")

    controlled = ai.copy()
    affected_samples: set[int] = set()
    qc_rows: list[dict[str, Any]] = []
    n_xl = xlines.size
    n_sample = samples.size

    for order, point in enumerate(control_points):
        inline, xline = survey.coord_to_line(point.x, point.y)
        top_name, bottom_name, horizon_values = locate_control_zone(
            target_layer,
            inline=float(inline),
            xline=float(xline),
            depth_m=point.depth_m,
        )
        top_grid = np.asarray(target_layer.get_horizon_grid(top_name), dtype=np.float64)
        bottom_grid = np.asarray(target_layer.get_horizon_grid(bottom_name), dtype=np.float64)
        zone_top = np.minimum(top_grid, bottom_grid)
        zone_bottom = np.maximum(top_grid, bottom_grid)

        d_xy = np.hypot(x_grid - point.x, y_grid - point.y)
        trace_mask = d_xy <= point.radius_xy_m
        trace_indices = np.argwhere(trace_mask)

        affected_trace_count = 0
        affected_sample_count = 0
        overlap_sample_count = 0
        weight_sum = 0.0
        max_weight = 0.0
        before_sum = 0.0
        after_sum = 0.0
        delta_abs_max = 0.0

        for i, j in trace_indices:
            i_int = int(i)
            j_int = int(j)
            z_in_radius = np.abs(samples - point.depth_m) <= point.radius_z_m
            z_in_layer = (samples >= zone_top[i_int, j_int]) & (samples <= zone_bottom[i_int, j_int])
            z_mask = z_in_radius & z_in_layer
            if not np.any(z_mask):
                continue

            sample_idx = np.flatnonzero(z_mask)
            w_xy = float(raised_cosine_weight(float(d_xy[i_int, j_int]) / point.radius_xy_m).item())
            w_z = raised_cosine_weight(np.abs(samples[sample_idx] - point.depth_m) / point.radius_z_m)
            weights = (point.strength * w_xy * w_z).astype(np.float64)
            positive = weights > 0.0
            if not np.any(positive):
                continue
            sample_idx = sample_idx[positive]
            weights = weights[positive]

            before = controlled[i_int, j_int, sample_idx].astype(np.float64)
            after_log = (1.0 - weights) * np.log(before) + weights * np.log(point.target_ai)
            after = np.exp(after_log).astype(np.float64)
            controlled[i_int, j_int, sample_idx] = after.astype(np.float32)

            linear_ids = ((i_int * n_xl + j_int) * n_sample + sample_idx).astype(np.int64)
            overlap_sample_count += sum(int(idx) in affected_samples for idx in linear_ids)
            affected_samples.update(int(idx) for idx in linear_ids)

            affected_trace_count += 1
            affected_sample_count += int(sample_idx.size)
            weight_sum += float(weights.sum())
            max_weight = max(max_weight, float(weights.max()))
            before_sum += float(before.sum())
            after_sum += float(after.sum())
            if sample_idx.size:
                delta_abs_max = max(delta_abs_max, float(np.max(np.abs(after - before))))

        if affected_sample_count == 0 and error_on_empty:
            raise ValueError(f"Facies control point {point.name!r} affected no samples.")

        qc_rows.append(
            {
                "order": int(order),
                "name": point.name,
                "x": float(point.x),
                "y": float(point.y),
                "depth_m": float(point.depth_m),
                "inline": float(inline),
                "xline": float(xline),
                "zone_top": top_name,
                "zone_bottom": bottom_name,
                "radius_xy_m": float(point.radius_xy_m),
                "radius_z_m": float(point.radius_z_m),
                "target_ai": float(point.target_ai),
                "strength": float(point.strength),
                "affected_traces": int(affected_trace_count),
                "affected_samples": int(affected_sample_count),
                "overlap_samples": int(overlap_sample_count),
                "max_weight": float(max_weight),
                "mean_weight": float(weight_sum / affected_sample_count) if affected_sample_count else 0.0,
                "mean_ai_before": float(before_sum / affected_sample_count) if affected_sample_count else np.nan,
                "mean_ai_after": float(after_sum / affected_sample_count) if affected_sample_count else np.nan,
                "mean_ai_delta": float((after_sum - before_sum) / affected_sample_count)
                if affected_sample_count
                else np.nan,
                "max_abs_ai_delta": float(delta_abs_max),
                "horizon_values": horizon_values,
            }
        )

    if np.any(~np.isfinite(controlled)) or np.any(controlled <= 0.0):
        raise ValueError("Controlled AI volume contains non-finite or non-positive values.")
    return controlled.astype(np.float32, copy=False), pd.DataFrame.from_records(qc_rows)


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
    """Load all heavy objects used by the interactive facies-control QC flow."""
    import torch

    from cup.seismic.survey import open_survey
    from cup.utils.io import load_yaml_config
    from ginn_depth.config import DepthGINNConfig
    from ginn_depth.data import (
        build_dataset as build_depth_dataset,
        load_dynamic_gain_depth_model,
        load_lfm_depth_npz,
    )

    repo_root = find_repo_root()
    common_config_path = resolve_relative_path(common_config_path, root=repo_root)
    train_config_path = resolve_relative_path(train_config_path, root=repo_root)
    common_cfg = load_yaml_config(common_config_path, base_dir=repo_root)
    train_cfg = load_yaml_config(train_config_path, base_dir=repo_root)
    data_root = resolve_relative_path(str(common_cfg.get("data_root", "data")), root=repo_root)

    facies_cfg = common_cfg.get("lfm_facies_control_depth", {}) or {}
    source_ai_lfm_file = source_ai_lfm_file or facies_cfg.get("source_ai_lfm_file") or train_cfg.get("ai_lfm_file")
    vp_lfm_file = vp_lfm_file or train_cfg.get("vp_lfm_file")
    wavelet_file = wavelet_file or train_cfg.get("wavelet_file")
    dynamic_gain_file = dynamic_gain_file or train_cfg.get("dynamic_gain_model")
    if seismic_file is None:
        training_seismic_file = train_cfg.get("seismic_file")
        if training_seismic_file:
            seismic_file = resolve_relative_path(str(training_seismic_file), root=repo_root)
        else:
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
    if (
        train_cfg.get("gain_source") == "dynamic_gain_model"
        and dynamic_gain_path is not None
        and dynamic_gain_path.exists()
    ):
        dynamic_gain = load_dynamic_gain_depth_model(dynamic_gain_path)
        if dynamic_gain.shape != ai_lfm.shape:
            raise ValueError(f"Dynamic gain shape mismatch: gain={dynamic_gain.shape}, ai={ai_lfm.shape}.")
        gain_mode = "dynamic_gain_model"

    depth_train_cfg = DepthGINNConfig.from_yaml(train_config_path, base_dir=repo_root)
    depth_train_cfg.seismic_file = seismic_file
    depth_train_cfg.ai_lfm_file = source_ai_lfm_path
    depth_train_cfg.vp_lfm_file = vp_lfm_path
    if wavelet_path is not None:
        depth_train_cfg.wavelet_file = wavelet_path
    if dynamic_gain_path is not None:
        depth_train_cfg.dynamic_gain_model = dynamic_gain_path
    dataset_bundle = build_depth_dataset(depth_train_cfg)
    wavelet_time_s = dataset_bundle.wavelet_time_s
    wavelet_amp = dataset_bundle.wavelet_amp
    seis_rms = float(dataset_bundle.train_dataset.seis_rms)
    if depth_train_cfg.gain_source == "fixed_gain":
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
    forward_model = (
        DepthForwardModel(
            np.asarray(wavelet_time_s, dtype=np.float32),
            np.asarray(wavelet_amp, dtype=np.float32),
            depth_axis_m=ai_lfm.samples.astype(np.float32),
            amplitude_threshold=float(train_cfg.get("wavelet_amplitude_threshold", 0.0)),
        )
        .to(torch.device(device))
        .eval()
    )
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
        seis_rms=seis_rms,
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
    if (
        il_slice.start == 0
        or il_slice.stop == context.ai_lfm.ilines.size
        or xl_slice.start == 0
        or xl_slice.stop == context.ai_lfm.xlines.size
    ):
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
    out._horizon_grids = {name: target_layer.get_horizon_grid(name)[il_slice, xl_slice] for name in out.horizon_names}
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
    import torch

    li, lj = _nearest_local_indices(result)
    ai_before = result.source_ai_full_window[li, lj, :]
    ai_after = result.controlled_ai_full_window[li, lj, :]
    vp = result.vp_full_window[li, lj, :]
    full_depth = context.ai_lfm.samples
    display_depth = context.ai_lfm.samples[result.z_slice]
    device = context.forward_model.matrix_builder.wavelet_amp.device
    with torch.no_grad():
        before = (
            context.forward_model(
                torch.from_numpy(ai_before[None, None, :]).float().to(device),
                torch.from_numpy(vp[None, :]).float().to(device),
                depth_axis_m=full_depth.astype(np.float32),
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        after = (
            context.forward_model(
                torch.from_numpy(ai_after[None, None, :]).float().to(device),
                torch.from_numpy(vp[None, :]).float().to(device),
                depth_axis_m=full_depth.astype(np.float32),
            )
            .squeeze()
            .cpu()
            .numpy()
        )
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
    real_interp = real_interp / float(context.seis_rms)
    mask = np.isfinite(real_interp) & np.isfinite(before) & np.isfinite(after)
    influence_valid_mask = mask & influence_mask

    before_rms_scaled, before_scale = rms_match(before, real_interp, mask)
    after_rms_scaled, after_scale = rms_match(after, real_interp, mask)
    before_influence_rms_scaled, before_influence_scale = rms_match(before, real_interp, influence_valid_mask)
    after_influence_rms_scaled, after_influence_scale = rms_match(after, real_interp, influence_valid_mask)
    rows = [
        _waveform_metrics("train_scale_before", "display_window", real_interp, before, mask),
        _waveform_metrics("train_scale_after", "display_window", real_interp, after, mask),
        _waveform_metrics(
            "rms_match_before",
            "display_window",
            real_interp,
            before_rms_scaled,
            mask,
            scale=before_scale,
        ),
        _waveform_metrics(
            "rms_match_after",
            "display_window",
            real_interp,
            after_rms_scaled,
            mask,
            scale=after_scale,
        ),
        _waveform_metrics("train_scale_before", "influence_only", real_interp, before, influence_valid_mask),
        _waveform_metrics("train_scale_after", "influence_only", real_interp, after, influence_valid_mask),
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
