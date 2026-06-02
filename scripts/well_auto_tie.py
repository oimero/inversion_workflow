"""Run time-domain well auto-tie route 1: vertical wells with TDT.

The script builds a full tie plan for all inventory wells, then executes only
enabled routes.  The first implementation enables ``vertical_with_tdt`` by
default and leaves later routes as explicit skipped/rejected plan rows.

Usage::

    python scripts/well_auto_tie.py
    python scripts/well_auto_tie.py --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.petrel.load import import_interpretation_petrel, import_well_tops_petrel
from cup.seismic.horizon import HorizonSurface
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.trace_sampling import assemble_nearest_trace_from_plan, build_nearest_trace_sample_plan
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.utils.coerce import as_bool
from cup.utils.config import merge_dict_defaults
from cup.well.assets import build_file_lookup
from cup.well.las import export_logset_to_las, load_vp_rho_logset_from_standard_las
from cup.well.td import (
    PreparedTieWindow,
    TargetTieWindow,
    build_tdt_from_anchor,
    crop_logset_md,
    export_vertical_tdt_to_petrel_checkshots,
    find_well_top_md,
    load_petrel_time_depth_table,
    normalize_twt_seconds,
    prepare_anchor_tdt_for_window,
    prepare_tdt_with_sonic_extension,
    tdt_overlaps_window,
    validate_time_depth_table,
    write_time_depth_table_csv,
)
from cup.well.tie import TieRoute, WellTiePlan, WellTieResult, build_auto_tie_search_space, build_tie_plan, plans_dataframe, results_dataframe, scaled_synthetic_metrics
from cup.well.trajectory import WellTrajectory, sample_trajectory_on_twt
from cup.well.wavelet import crop_wavelet_center_energy_normalize
from wtie.processing import grid


# =============================================================================
# CLI and config
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--well", type=str, default=None, help="Optional single well execution filter.")
    return parser.parse_args()


def _script_config(cfg: dict[str, Any]) -> dict[str, Any]:
    script_cfg = dict(cfg.get("well_auto_tie") or {})
    merge_dict_defaults(
        script_cfg,
        "source_runs",
        {
            "mode": "latest",
            "well_inventory_dir": None,
            "well_screen_dir": None,
            "well_preprocess_dir": None,
            "well_trajectory_dir": None,
        },
    )
    script_cfg.setdefault("time_depth_dir", "time_depth_table")
    script_cfg.setdefault("well_trace_dir", "all_well_trace")
    script_cfg.setdefault("well_tops_file", "raw/well_tops")
    merge_dict_defaults(
        script_cfg,
        "target_interval",
        {
            "top_horizon": "interpre/H3-1",
            "bottom_horizon": "interpre/H7-1",
            "margin_top_ms": 100.0,
            "margin_bottom_ms": 100.0,
            "twt_unit": "auto",
        },
    )
    merge_dict_defaults(
        script_cfg,
        "seismic",
        {"file": "raw/obn-clipped-240-912-872-1544.zgy", "type": "zgy"},
    )
    script_cfg.setdefault("enabled_routes", ["vertical_with_tdt", "vertical_anchor_from_tops"])
    script_cfg.setdefault("tutorial_model", "tutorial/trained_net_state_dict.pt")
    script_cfg.setdefault("tutorial_params", "tutorial/network_parameters.yaml")
    script_cfg.setdefault("target_crop_ms", 201.0)
    merge_dict_defaults(
        script_cfg,
        "search_space",
        {
            "logs_median_size_values": [51, 71, 91, 111],
            "logs_median_threshold_bounds": [0.5, 3.0],
            "logs_std_bounds": [20, 50],
            "table_t_shift_bounds": [-0.030, 0.030],
        },
    )
    merge_dict_defaults(script_cfg, "search_params", {"num_iters": 60, "similarity_std": 0.02})
    merge_dict_defaults(
        script_cfg,
        "wavelet_scaling",
        {"min_scale": 50000, "max_scale": 500000, "num_iters": 60},
    )
    merge_dict_defaults(
        script_cfg,
        "reject",
        {
            "allow_near_outside": False,
            "min_tie_samples": 64,
            "max_trajectory_outside_fraction": 0.05,
        },
    )
    merge_dict_defaults(
        script_cfg,
        "coarse_correction",
        {
            "anchor": {
                "enabled": True,
                "apply_to_routes": ["vertical_anchor_from_tops"],
                "config_file": "experiments/well_auto_tie_anchors.yaml",
            },
            "manual_shift": {
                "default_ms": 0.0,
                "by_route_ms": {},
                "config_file": "experiments/well_auto_tie_manual_shifts.yaml",
            },
        },
    )
    return script_cfg


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_relative_path(value, root=REPO_ROOT)


def _resolve_data_path(value: str | Path, *, data_root: Path) -> Path:
    return resolve_relative_path(value, root=data_root)


def _resolve_output_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return _resolve_repo_path(args.output_dir)
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"well_auto_tie_{timestamp}"


def _discover_latest_dir(cfg: dict[str, Any], prefix: str) -> Path:
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    all_candidates = [path for path in output_root.glob(f"{prefix}_*") if path.is_dir()]
    timestamped = [path for path in all_candidates if re.fullmatch(rf"{re.escape(prefix)}_\d{{8}}_\d{{6}}", path.name)]
    candidates = sorted(timestamped or all_candidates, key=lambda path: path.name)
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_* output directory found under output_root.")
    return candidates[-1]


def _resolve_inputs(cfg: dict[str, Any], script_cfg: dict[str, Any]) -> dict[str, Path | None]:
    source_runs = dict(script_cfg.get("source_runs") or {})
    mode = str(source_runs.get("mode", "latest")).strip().casefold()
    if mode != "latest":
        raise ValueError(f"well_auto_tie.source_runs.mode only supports 'latest' for now, got {mode!r}.")

    inventory_dir = (
        _resolve_repo_path(source_runs["well_inventory_dir"])
        if source_runs.get("well_inventory_dir") is not None
        else _discover_latest_dir(cfg, "well_inventory")
    )
    screen_dir = (
        _resolve_repo_path(source_runs["well_screen_dir"])
        if source_runs.get("well_screen_dir") is not None
        else _discover_latest_dir(cfg, "well_screen")
    )
    preprocess_dir = (
        _resolve_repo_path(source_runs["well_preprocess_dir"])
        if source_runs.get("well_preprocess_dir") is not None
        else _discover_latest_dir(cfg, "well_preprocess")
    )

    trajectory_dir = None
    if source_runs.get("well_trajectory_dir") is not None:
        trajectory_dir = _resolve_repo_path(source_runs["well_trajectory_dir"])
    else:
        try:
            trajectory_dir = _discover_latest_dir(cfg, "well_trajectory")
        except FileNotFoundError:
            trajectory_dir = None

    inventory_file = inventory_dir / "well_inventory.csv"
    well_screen_file = screen_dir / "well_screen.csv"
    preprocess_status_file = preprocess_dir / "well_preprocess_status.csv"
    well_trajectory_file = trajectory_dir / "well_trajectory.csv" if trajectory_dir is not None else None

    return {
        "inventory_file": inventory_file,
        "well_screen_file": well_screen_file,
        "preprocess_status_file": preprocess_status_file,
        "preprocessed_las_dir": preprocess_dir / "preprocessed_las",
        "well_trajectory_file": (
            well_trajectory_file if well_trajectory_file is not None and well_trajectory_file.exists() else None
        ),
    }


def _series_value_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def _load_anchor_config(anchor_cfg: dict[str, Any]) -> dict[str, Any]:
    config = {"default": {}, "wells": {}}
    config_file = anchor_cfg.get("config_file")
    if config_file:
        path = _resolve_repo_path(config_file)
        if path.exists():
            with path.open("r", encoding="utf-8") as fp:
                file_config = yaml.safe_load(fp) or {}
            anchors = dict(file_config.get("anchors") or file_config)
            config["default"].update(dict(anchors.get("default") or {}))
            config["wells"].update(dict(anchors.get("wells") or {}))
    if not config["default"].get("well_top") or not config["default"].get("horizon"):
        raise ValueError("coarse_correction.anchor config_file requires default.well_top and default.horizon.")
    return config


def _anchor_spec_for_well(anchor_config: dict[str, Any], well_name: str) -> dict[str, Any]:
    spec = dict(anchor_config.get("default") or {})
    wells = dict(anchor_config.get("wells") or {})
    for key, value in wells.items():
        if str(key).strip().casefold() == well_name.strip().casefold():
            spec.update(dict(value or {}))
            break
    spec.setdefault("twt_unit", "auto")
    return spec


def _anchor_enabled_for_route(config: dict[str, Any], route: str) -> bool:
    anchor_cfg = dict(config["coarse_correction"].get("anchor") or {})
    if not as_bool(anchor_cfg.get("enabled", False)):
        return False
    return str(route) in {str(value) for value in anchor_cfg.get("apply_to_routes") or []}


def _load_manual_shift_config(manual_cfg: dict[str, Any]) -> dict[str, Any]:
    config = {
        "default_ms": float(manual_cfg.get("default_ms", 0.0) or 0.0),
        "by_route_ms": dict(manual_cfg.get("by_route_ms") or {}),
        "wells_ms": {},
    }
    config_file = manual_cfg.get("config_file")
    if config_file:
        path = _resolve_repo_path(config_file)
        if path.exists():
            with path.open("r", encoding="utf-8") as fp:
                file_config = yaml.safe_load(fp) or {}
            shifts = dict(file_config.get("manual_shift") or file_config)
            if "default_ms" in shifts:
                config["default_ms"] = float(shifts.get("default_ms") or 0.0)
            config["by_route_ms"].update(dict(shifts.get("by_route_ms") or {}))
            config["wells_ms"].update(dict(shifts.get("wells_ms") or {}))
    return config


def _manual_shift_seconds(manual_shift_config: dict[str, Any], *, route: str, well_name: str) -> float:
    shift_ms = float(manual_shift_config.get("default_ms", 0.0) or 0.0)
    by_route = dict(manual_shift_config.get("by_route_ms") or {})
    wells = dict(manual_shift_config.get("wells_ms") or {})
    if route in by_route:
        shift_ms = float(by_route[route] or 0.0)
    for key, value in wells.items():
        if str(key).strip().casefold() == well_name.strip().casefold():
            shift_ms = float(value or 0.0)
            break
    return shift_ms / 1000.0


def _shift_time_depth_table(table: grid.TimeDepthTable, shift_s: float) -> grid.TimeDepthTable:
    shift = float(shift_s)
    if abs(shift) <= 1e-12:
        return table
    if table.is_md_domain:
        return grid.TimeDepthTable(twt=np.asarray(table.twt, dtype=np.float64) + shift, md=np.asarray(table.md, dtype=np.float64))
    return grid.TimeDepthTable(twt=np.asarray(table.twt, dtype=np.float64) + shift, tvdss=np.asarray(table.tvdss, dtype=np.float64))


def _anchor_position_for_plan(
    *,
    plan: WellTiePlan,
    anchor_md_m: float,
    trajectory: WellTrajectory | None,
) -> tuple[float, float, str]:
    if trajectory is not None:
        position = trajectory.position_at_md(float(anchor_md_m))
        return float(position["x_m"]), float(position["y_m"]), "trajectory_at_anchor_md"
    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY for anchor sampling.")
    return float(plan.surface_x), float(plan.surface_y), "surface_xy"


def _sample_anchor_for_plan(
    *,
    plan: WellTiePlan,
    survey: Any,
    well_tops_df: pd.DataFrame,
    anchor_config: dict[str, Any],
    horizon_cache: dict[str, HorizonSurface],
    data_root: Path,
    trajectory: WellTrajectory | None = None,
) -> dict[str, Any]:
    spec = _anchor_spec_for_well(anchor_config, plan.well_name)
    anchor_md_m = find_well_top_md(well_tops_df, well_name=plan.well_name, surface=str(spec["well_top"]))
    anchor_x_m, anchor_y_m, anchor_position_source = _anchor_position_for_plan(
        plan=plan,
        anchor_md_m=float(anchor_md_m),
        trajectory=trajectory,
    )
    inline_float, xline_float = survey.line_geometry.coord_to_line(float(anchor_x_m), float(anchor_y_m))
    horizon_path = _resolve_data_path(spec["horizon"], data_root=data_root)
    horizon_surface = _load_horizon_surface(horizon_path, horizon_cache)
    horizon_sample = horizon_surface.sample_at_line(inline_float, xline_float)
    raw_horizon_twt = float(horizon_sample["value"])
    anchor_twt_s = normalize_twt_seconds(raw_horizon_twt, unit=str(spec.get("twt_unit", "auto")))
    return {
        "well_top": str(spec["well_top"]),
        "horizon": repo_relative_path(horizon_path, root=REPO_ROOT),
        "twt_unit": str(spec.get("twt_unit", "auto")),
        "anchor_md_m": float(anchor_md_m),
        "anchor_x_m": float(anchor_x_m),
        "anchor_y_m": float(anchor_y_m),
        "anchor_position_source": anchor_position_source,
        "horizon_twt_raw": float(raw_horizon_twt),
        "anchor_twt_s": float(anchor_twt_s),
        "inline_float": float(inline_float),
        "xline_float": float(xline_float),
        "horizon_sample_method": str(horizon_sample["method"]),
        "horizon_nearest_line_distance": float(horizon_sample["nearest_line_distance"]),
        "horizon_nearest_inline": float(horizon_sample["nearest_inline"]),
        "horizon_nearest_xline": float(horizon_sample["nearest_xline"]),
    }


def _coarse_report(*, anchor_shift_s: float, manual_shift_s: float, anchor_info: dict[str, Any] | None) -> dict[str, Any]:
    total_shift_s = float(anchor_shift_s) + float(manual_shift_s)
    return {
        "coarse_anchor_shift_ms": float(anchor_shift_s) * 1000.0,
        "coarse_manual_shift_ms": float(manual_shift_s) * 1000.0,
        "coarse_total_shift_ms": float(total_shift_s) * 1000.0,
        "coarse_anchor_applied": bool(anchor_info),
        "coarse_manual_applied": bool(abs(float(manual_shift_s)) > 1e-12),
    }


def _with_report(prepared: PreparedTieWindow, values: dict[str, Any]) -> PreparedTieWindow:
    report = dict(prepared.report)
    report.update(values)
    return replace(prepared, report=report)


def _apply_coarse_correction_to_existing_tdt(
    *,
    plan: WellTiePlan,
    table: grid.TimeDepthTable,
    config: dict[str, Any],
    survey: Any,
    well_tops_df: pd.DataFrame,
    anchor_config: dict[str, Any],
    manual_shift_config: dict[str, Any],
    horizon_cache: dict[str, HorizonSurface],
    data_root: Path,
    trajectory: WellTrajectory | None = None,
) -> tuple[grid.TimeDepthTable, dict[str, Any], dict[str, Any] | None]:
    anchor_shift_s = 0.0
    anchor_info: dict[str, Any] | None = None
    if _anchor_enabled_for_route(config, plan.route):
        anchor_info = _sample_anchor_for_plan(
            plan=plan,
            survey=survey,
            well_tops_df=well_tops_df,
            anchor_config=anchor_config,
            horizon_cache=horizon_cache,
            data_root=data_root,
            trajectory=trajectory,
        )
        table_md = np.asarray(table.md, dtype=np.float64)
        table_twt = np.asarray(table.twt, dtype=np.float64)
        anchor_md = float(anchor_info["anchor_md_m"])
        if anchor_md < float(table_md[0]) or anchor_md > float(table_md[-1]):
            raise ValueError(
                f"Anchor MD {anchor_md} is outside TDT MD range [{table_md[0]}, {table_md[-1]}] "
                f"for well {plan.well_name}; cannot apply anchor shift to existing TDT."
            )
        anchor_tdt_twt_s = float(np.interp(anchor_md, table_md, table_twt))
        anchor_shift_s = float(anchor_info["anchor_twt_s"]) - anchor_tdt_twt_s
        anchor_info = {
            **anchor_info,
            "anchor_role": "shift_existing_tdt",
            "tdt_twt_at_anchor_s": anchor_tdt_twt_s,
            "anchor_shift_ms": anchor_shift_s * 1000.0,
        }

    manual_shift_s = _manual_shift_seconds(manual_shift_config, route=plan.route, well_name=plan.well_name)
    total_shift_s = anchor_shift_s + manual_shift_s
    return _shift_time_depth_table(table, total_shift_s), _coarse_report(
        anchor_shift_s=anchor_shift_s,
        manual_shift_s=manual_shift_s,
        anchor_info=anchor_info,
    ), anchor_info


def _load_horizon_surface(path: Path, cache: dict[str, HorizonSurface]) -> HorizonSurface:
    key = str(path.resolve())
    if key not in cache:
        cache[key] = HorizonSurface.from_petrel_dataframe(import_interpretation_petrel(path), name=path.name)
    return cache[key]


class RerouteToAnchor(Exception):
    """Signal that a TDT route should be executed with the anchor route."""

    def __init__(self, reason: str, target_window: TargetTieWindow):
        super().__init__(reason)
        self.reason = reason
        self.target_window = target_window


def _target_horizon_path(value: str, *, data_root: Path) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("target_interval top/bottom horizon cannot be empty.")
    if "/" not in text and "\\" not in text:
        text = f"interpre/{text}"
    return _resolve_data_path(text, data_root=data_root)


def _target_horizon_name(value: str) -> str:
    text = str(value).strip()
    return Path(text).name if ("/" in text or "\\" in text) else text


def _target_horizon_spec(config: dict[str, Any], side: str) -> tuple[str, str]:
    """Return horizon path-like value and display name for the target window."""
    target_cfg = dict(config["target_interval"])
    horizon_key = f"{side}_horizon"
    value = target_cfg.get(horizon_key)
    if value is None:
        raise ValueError(f"target_interval.{horizon_key} is required.")

    return str(value), _target_horizon_name(str(value))


def _target_tie_window_for_plan(
    *,
    plan: WellTiePlan,
    survey: Any,
    config: dict[str, Any],
    horizon_cache: dict[str, HorizonSurface],
    data_root: Path,
) -> TargetTieWindow:
    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY.")
    target_cfg = dict(config["target_interval"])
    top_value, top_name = _target_horizon_spec(config, "top")
    bottom_value, bottom_name = _target_horizon_spec(config, "bottom")
    top_path = _target_horizon_path(top_value, data_root=data_root)
    bottom_path = _target_horizon_path(bottom_value, data_root=data_root)
    inline_float, xline_float = survey.line_geometry.coord_to_line(float(plan.surface_x), float(plan.surface_y))
    top_sample = _load_horizon_surface(top_path, horizon_cache).sample_at_line(inline_float, xline_float)
    bottom_sample = _load_horizon_surface(bottom_path, horizon_cache).sample_at_line(inline_float, xline_float)
    unit = str(target_cfg.get("twt_unit", "auto"))
    top_twt = normalize_twt_seconds(float(top_sample["value"]), unit=unit)
    bottom_twt = normalize_twt_seconds(float(bottom_sample["value"]), unit=unit)
    if bottom_twt < top_twt:
        top_twt, bottom_twt = bottom_twt, top_twt
    margin_top_s = float(target_cfg.get("margin_top_ms", 100.0)) / 1000.0
    margin_bottom_s = float(target_cfg.get("margin_bottom_ms", 100.0)) / 1000.0
    start_s = max(0.0, top_twt - margin_top_s)
    end_s = bottom_twt + margin_bottom_s
    if end_s <= start_s:
        raise ValueError(f"Invalid target tie window [{start_s}, {end_s}] for well {plan.well_name}.")
    return TargetTieWindow(
        top_name=top_name,
        bottom_name=bottom_name,
        top_twt_s=top_twt,
        bottom_twt_s=bottom_twt,
        start_s=start_s,
        end_s=end_s,
        margin_top_s=margin_top_s,
        margin_bottom_s=margin_bottom_s,
        top_sample_method=str(top_sample["method"]),
        bottom_sample_method=str(bottom_sample["method"]),
        top_nearest_line_distance=float(top_sample["nearest_line_distance"]),
        bottom_nearest_line_distance=float(bottom_sample["nearest_line_distance"]),
    )


# =============================================================================
# Output helpers
# =============================================================================


def _build_output_paths(output_dir: Path, well_name: str) -> dict[str, Path]:
    safe = sanitize_filename(well_name)
    figure_dir = output_dir / "figures" / safe
    return {
        "wavelet": output_dir / "wavelets" / f"wavelet_201ms_{safe}.csv",
        "initial_tdt": output_dir / "time_depth" / f"initial_tdt_{safe}.csv",
        "optimized_tdt": output_dir / "time_depth" / f"optimized_tdt_{safe}.csv",
        "petrel_checkshot": output_dir / "petrel_checkshots" / f"optimized_tdt_{safe}.txt",
        "filtered_las": output_dir / "filtered_las" / f"filtered_logs_{safe}.las",
        "synthetic_qc": output_dir / "synthetic_qc" / f"tie_qc_{safe}.csv",
        "seismic_trace": output_dir / "seismic_trace" / f"seismic_trace_{safe}.csv",
        "trace_sample_plan": output_dir / "trace_sample_plan" / f"trace_sample_plan_{safe}.csv",
        "optimized_trace_sample_plan": output_dir / "trace_sample_plan" / f"optimized_trace_sample_plan_{safe}.csv",
        "figure_dir": figure_dir,
        "fig_objective": figure_dir / "optimization_objective.png",
        "fig_tdt": figure_dir / "time_depth_table.png",
        "fig_tie": figure_dir / "synthetic_match.png",
        "fig_wavelet": figure_dir / "wavelet.png",
    }


def _ensure_output_dirs(output_dir: Path) -> None:
    for name in [
        "wavelets",
        "time_depth",
        "petrel_checkshots",
        "filtered_las",
        "synthetic_qc",
        "seismic_trace",
        "trace_sample_plan",
        "figures",
    ]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except RuntimeError:
        pass
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _compute_table_synthetic_metrics(logset_md: Any, seismic: Any, table: Any, wavelet: Any, modeler: Any, dt_s: float) -> tuple[float, float]:
    from wtie.optimize import tie as tie_ops

    logset_twt = tie_ops.convert_logs_from_md_to_twt(logset_md, None, table, dt_s)
    reflectivity = tie_ops.compute_reflectivity(logset_twt, angle_range=seismic.angle_range)
    seismic_match, reflectivity_match = tie_ops.match_seismic_and_reflectivity(seismic, reflectivity)
    _, _, corr, nmae, _ = scaled_synthetic_metrics(modeler, wavelet, reflectivity_match, seismic_match)
    return corr, nmae


def _snap_seismic_sampling_if_close(seismic: Any, expected_dt_s: float, *, rtol: float = 1e-5) -> Any:
    """Rebuild a seismic trace with an exact dt when floating-point drift is negligible."""
    actual_dt = float(seismic.sampling_rate)
    expected = float(expected_dt_s)
    if not np.isfinite(actual_dt) or not np.isfinite(expected) or expected <= 0.0:
        return seismic
    if not np.isclose(actual_dt, expected, rtol=rtol, atol=1e-9):
        return seismic
    snapped_basis = float(seismic.basis[0]) + np.arange(int(seismic.size), dtype=np.float64) * expected
    return type(seismic)(
        np.asarray(seismic.values, dtype=np.float64),
        snapped_basis,
        "twt",
        name=seismic.name,
        theta=getattr(seismic, "theta", 0),
    )


def _finite_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _md_tdt_to_vertical_petrel_tvdss_table(table: Any, *, kb_m: float) -> grid.TimeDepthTable:
    """Convert an MD-domain workflow TDT to the TVDSS-domain table expected by the Petrel exporter."""
    if not getattr(table, "is_md_domain", False):
        raise ValueError("Petrel checkshot export expects an MD-domain optimized TDT.")
    md = np.asarray(table.md, dtype=np.float64)
    tvdss = md - float(kb_m)
    if np.any(~np.isfinite(tvdss)):
        raise ValueError("Petrel checkshot export got non-finite TVDSS values.")
    if np.any(tvdss < 0.0):
        raise ValueError("Petrel vertical checkshot export does not support samples above the KB datum.")
    return grid.TimeDepthTable(twt=np.asarray(table.twt, dtype=np.float64), tvdss=tvdss)


def _export_petrel_checkshot_tdt(paths: dict[str, Path], plan: WellTiePlan, table: Any) -> Path:
    if plan.surface_x is None or plan.surface_y is None or plan.kb_m is None:
        raise ValueError("Petrel checkshot export requires surface_x, surface_y and kb_m.")
    kb_m = _finite_float(plan.kb_m, label="kb_m")
    petrel_table = _md_tdt_to_vertical_petrel_tvdss_table(table, kb_m=kb_m)
    return export_vertical_tdt_to_petrel_checkshots(
        paths["petrel_checkshot"],
        petrel_table,
        well_name=plan.well_name,
        kb=kb_m,
        x=_finite_float(plan.surface_x, label="surface_x"),
        y=_finite_float(plan.surface_y, label="surface_y"),
    )


def _write_optimized_trace_sample_plan(
    *,
    paths: dict[str, Path],
    plan: WellTiePlan,
    survey: Any,
    trajectory: WellTrajectory,
    table: grid.TimeDepthTable,
) -> Path:
    """Write deviated-well spatial samples regenerated from the optimized TDT."""
    if not getattr(table, "is_md_domain", False):
        raise ValueError("Optimized trace sample plan expects an MD-domain TDT.")
    table_twt = np.asarray(table.twt, dtype=np.float64)
    if table_twt.size < 2:
        raise ValueError("Optimized TDT must contain at least two TWT samples.")

    sample_axis = survey.sample_axis("time")
    sample_idx_start, sample_idx_end = sample_axis.window_indices(float(table_twt[0]), float(table_twt[-1]))
    twt_axis = sample_axis.values[sample_idx_start:sample_idx_end]
    if twt_axis.size == 0:
        raise ValueError("Optimized trace sample plan has no seismic samples inside optimized TDT range.")

    samples = sample_trajectory_on_twt(trajectory.with_well_name(plan.well_name), table, twt_axis)
    trace_plan = build_nearest_trace_sample_plan(samples, survey)
    trace_plan.to_dataframe().to_csv(paths["optimized_trace_sample_plan"], index=False)
    return paths["optimized_trace_sample_plan"]


def _filtered_standard_las_logset(logset_md: Any, best_params: Mapping[str, Any]) -> dict[str, grid.Log]:
    """Apply the selected auto-tie MD log filters and convert back to standard LAS mnemonics."""
    from wtie.optimize import tie as tie_ops

    filtered = tie_ops.filter_md_logs(
        logset_md,
        median_size=best_params["logs_median_size"],
        threshold=best_params["logs_median_threshold"],
        std=best_params["logs_std"],
        std2=0.8 * best_params["logs_std"],
    )
    vp = np.asarray(filtered.Logs["Vp"].values, dtype=np.float64)
    rho = np.asarray(filtered.Logs["Rho"].values, dtype=np.float64)
    md = np.asarray(filtered.basis, dtype=np.float64)
    if np.any(~np.isfinite(vp)) or np.any(vp <= 0.0):
        raise ValueError("Filtered Vp contains non-finite or non-positive values.")
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError("Filtered Rho contains non-finite or non-positive values.")
    return {
        "DT_USM": grid.Log(1_000_000.0 / vp, md, "md", name="DT_USM", unit="us/m", allow_nan=False),
        "RHO_GCC": grid.Log(rho, md, "md", name="RHO_GCC", unit="g/cm3", allow_nan=False),
    }


def _export_filtered_las(paths: dict[str, Path], plan: WellTiePlan, logset_md: Any, best_params: Mapping[str, Any]) -> Path:
    filtered_logs = _filtered_standard_las_logset(logset_md, best_params)
    return export_logset_to_las(
        plan.well_name,
        filtered_logs,
        paths["filtered_las"],
        curve_names=["DT_USM", "RHO_GCC"],
    )


def _draw_target_horizons(axes: Any, target_window: TargetTieWindow | None, *, y_scale: float = 1.0) -> None:
    """Draw target horizon markers on TWT-domain QC axes."""
    if target_window is None:
        return
    specs = [
        (target_window.top_twt_s * y_scale, "top", "tab:green"),
        (target_window.bottom_twt_s * y_scale, "bottom", "tab:blue"),
    ]
    flat_axes = np.ravel(axes).tolist()
    for ax in flat_axes:
        if not hasattr(ax, "get_xlim"):
            continue
        xmin, xmax = ax.get_xlim()
        span = xmax - xmin
        label_x = xmin + 0.03 * span if np.isfinite(span) and span != 0.0 else xmin
        for y_ms, label, color in specs:
            ax.axhline(y_ms, color=color, lw=0.9, alpha=0.75, linestyle="--")
            ax.text(
                label_x,
                y_ms,
                f" {label}",
                va="center",
                ha="left",
                fontsize=8,
                color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
            )


def _source_at_twt(rows: pd.DataFrame, twt_s: float) -> str:
    values = rows["twt_s"].to_numpy(dtype=np.float64)
    sources = rows["source"].astype(str).to_numpy(dtype=object)
    index = int(np.searchsorted(values, float(twt_s), side="left"))
    index = max(0, min(index, len(sources) - 1))
    return str(sources[index])


def _clip_prepared_tie_window(
    prepared: PreparedTieWindow,
    *,
    start_s: float,
    end_s: float,
    min_tie_samples: int,
    clip_reason: str,
) -> PreparedTieWindow:
    """Clip a prepared tie window to a smaller TWT interval."""
    if float(end_s) <= float(start_s):
        raise ValueError(f"Invalid clipped tie window [{start_s}, {end_s}].")

    table = prepared.table
    if not table.is_md_domain:
        raise ValueError("well_auto_tie expects prepared tables in MD domain.")
    twt = np.asarray(table.twt, dtype=np.float64)
    md = np.asarray(table.md, dtype=np.float64)
    if float(start_s) < float(twt[0]) - 1e-9 or float(end_s) > float(twt[-1]) + 1e-9:
        raise ValueError(f"Clipped tie window [{start_s}, {end_s}] is outside prepared table range [{twt[0]}, {twt[-1]}].")

    keep = (twt >= float(start_s)) & (twt <= float(end_s))
    rows = pd.DataFrame({"twt_s": twt[keep], "md_m": md[keep]})
    boundary_rows: list[dict[str, Any]] = []
    for boundary in (float(start_s), float(end_s)):
        if rows.empty or not np.any(np.isclose(rows["twt_s"].to_numpy(dtype=np.float64), boundary, rtol=0.0, atol=1e-9)):
            boundary_rows.append(
                {
                    "twt_s": boundary,
                    "md_m": float(np.interp(boundary, twt, md)),
                }
            )
    if boundary_rows:
        rows = pd.concat([rows, pd.DataFrame.from_records(boundary_rows)], ignore_index=True)
    rows = rows.sort_values("twt_s").drop_duplicates(subset=["twt_s"], keep="first").reset_index(drop=True)
    rows["source"] = [_source_at_twt(prepared.table_rows, float(value)) for value in rows["twt_s"]]
    if len(rows) < 2:
        raise ValueError("Clipped prepared TDT has fewer than 2 rows.")

    clipped_table = grid.TimeDepthTable(
        twt=rows["twt_s"].to_numpy(dtype=np.float64),
        md=rows["md_m"].to_numpy(dtype=np.float64),
    )
    md_min = float(np.interp(float(rows["twt_s"].iloc[0]), clipped_table.twt, clipped_table.md))
    md_max = float(np.interp(float(rows["twt_s"].iloc[-1]), clipped_table.twt, clipped_table.md))
    clipped_logset = crop_logset_md(prepared.logset_md, md_min, md_max, min_samples=min_tie_samples)

    report = dict(prepared.report)
    report["tie_window_start_s"] = float(rows["twt_s"].iloc[0])
    report["tie_window_end_s"] = float(rows["twt_s"].iloc[-1])
    report["tie_window_duration_s"] = float(rows["twt_s"].iloc[-1] - rows["twt_s"].iloc[0])
    report["tie_window_md_min_m"] = float(clipped_logset.basis[0])
    report["tie_window_md_max_m"] = float(clipped_logset.basis[-1])
    report["tie_window_log_sample_count"] = int(clipped_logset.basis.size)
    existing_reason = str(report.get("window_clip_reason") or "")
    report["window_clip_reason"] = ";".join([value for value in [existing_reason, clip_reason] if value])
    return PreparedTieWindow(table=clipped_table, logset_md=clipped_logset, table_rows=rows, report=report)


def _write_qc_figures(
    paths: dict[str, Path],
    inputs_table: Any,
    outputs: Any,
    cropped_wavelet: Any,
    seismic_norm: np.ndarray,
    synthetic: np.ndarray,
    *,
    initial_table_rows: pd.DataFrame | None = None,
    target_window: TargetTieWindow | None = None,
) -> None:
    try:
        fig, _ = outputs.plot_optimization_objective(figsize=(6, 3))
        _save_current_figure(paths["fig_objective"])
    except Exception:
        plt.close("all")

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if initial_table_rows is not None and not initial_table_rows.empty:
        rows = initial_table_rows.sort_values("twt_s").reset_index(drop=True)
        run_id = rows["source"].ne(rows["source"].shift()).cumsum()
        for _, part in rows.groupby(run_id, sort=False):
            source = str(part["source"].iloc[0])
            linestyle = "-" if source == "original_tdt" else "--"
            color = "tab:blue"
            ax.plot(part["md_m"], part["twt_s"] * 1000.0, linestyle=linestyle, lw=1.0, color=color, label=source)
    else:
        ax.plot(inputs_table.md, inputs_table.twt * 1000.0, lw=1.0, color="tab:blue", label="initial")
    ax.plot(outputs.table.md, outputs.table.twt * 1000.0, lw=1.0, color="tab:orange", label="optimized")
    ax.invert_yaxis()
    ax.set_xlabel("MD (m)")
    ax.set_ylabel("TWT (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_current_figure(paths["fig_tdt"])

    fig, axes = outputs.plot_tie_window(wiggle_scale=120000, figsize=(12.0, 7.5))
    _draw_target_horizons(axes, target_window, y_scale=1.0)
    _save_current_figure(paths["fig_tie"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(outputs.wavelet.basis * 1000.0, outputs.wavelet.values / np.max(np.abs(outputs.wavelet.values)), label="raw")
    axes[0].plot(
        cropped_wavelet.basis * 1000.0,
        cropped_wavelet.values / np.max(np.abs(cropped_wavelet.values)),
        label="cropped",
    )
    axes[0].axvline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("Time (ms)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)
    freq = np.fft.rfftfreq(cropped_wavelet.size, d=cropped_wavelet.sampling_rate)
    spec = np.abs(np.fft.rfft(cropped_wavelet.values))
    axes[1].plot(freq, spec / max(float(spec.max()), 1e-12))
    axes[1].set_xlim(0, min(125, float(freq[-1])))
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True, alpha=0.25)
    _save_current_figure(paths["fig_wavelet"])


# =============================================================================
# Route execution
# =============================================================================


def _run_vertical_with_tdt(
    *,
    plan: WellTiePlan,
    survey: Any,
    wavelet_extractor: Any,
    modeler: Any,
    config: dict[str, Any],
    output_dir: Path,
    horizon_cache: dict[str, HorizonSurface],
    data_root: Path,
    well_tops_df: pd.DataFrame,
    anchor_config: dict[str, Any],
    manual_shift_config: dict[str, Any],
) -> tuple[WellTieResult, dict[str, Any]]:
    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY.")
    input_las = _resolve_repo_path(plan.input_las)
    time_depth_file = _resolve_repo_path(plan.time_depth_file)

    logset_md = load_vp_rho_logset_from_standard_las(input_las)
    table = load_petrel_time_depth_table(time_depth_file, domain="md")
    target_window = _target_tie_window_for_plan(
        plan=plan,
        survey=survey,
        config=config,
        horizon_cache=horizon_cache,
        data_root=data_root,
    )
    table, coarse_report, anchor_info = _apply_coarse_correction_to_existing_tdt(
        plan=plan,
        table=table,
        config=config,
        survey=survey,
        well_tops_df=well_tops_df,
        anchor_config=anchor_config,
        manual_shift_config=manual_shift_config,
        horizon_cache=horizon_cache,
        data_root=data_root,
    )
    if not tdt_overlaps_window(table, target_window):
        raise RerouteToAnchor("tdt_no_target_window_overlap_reroute_anchor", target_window)
    prepared = prepare_tdt_with_sonic_extension(
        raw_table=table,
        logset_md=logset_md,
        window=target_window,
        min_tie_samples=int(config["reject"]["min_tie_samples"]),
    )
    prepared = _with_report(prepared, coarse_report)
    return _run_tie_with_initial_table(
        plan=plan,
        survey=survey,
        wavelet_extractor=wavelet_extractor,
        modeler=modeler,
        config=config,
        output_dir=output_dir,
        prepared=prepared,
        extra_seed={
            "initial_table_source": "time_depth_file",
            "time_depth_file": repo_relative_path(time_depth_file, root=REPO_ROOT),
            "coarse_correction": coarse_report,
            "anchor": anchor_info,
        },
    )


def _run_vertical_anchor_from_tops(
    *,
    plan: WellTiePlan,
    survey: Any,
    wavelet_extractor: Any,
    modeler: Any,
    config: dict[str, Any],
    output_dir: Path,
    well_tops_df: pd.DataFrame,
    anchor_config: dict[str, Any],
    manual_shift_config: dict[str, Any],
    horizon_cache: dict[str, HorizonSurface],
    data_root: Path,
    rerouted_from: str | None = None,
    reroute_reason: str | None = None,
) -> tuple[WellTieResult, dict[str, Any]]:
    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY.")
    if not _anchor_enabled_for_route(config, plan.route):
        raise ValueError("coarse_correction.anchor must include vertical_anchor_from_tops to build an initial TDT.")

    input_las = _resolve_repo_path(plan.input_las)
    logset_md = load_vp_rho_logset_from_standard_las(input_las)
    anchor_info = _sample_anchor_for_plan(
        plan=plan,
        survey=survey,
        well_tops_df=well_tops_df,
        anchor_config=anchor_config,
        horizon_cache=horizon_cache,
        data_root=data_root,
    )
    table = build_tdt_from_anchor(
        logset_md,
        anchor_md_m=float(anchor_info["anchor_md_m"]),
        anchor_twt_s=float(anchor_info["anchor_twt_s"]),
    )
    manual_shift_s = _manual_shift_seconds(manual_shift_config, route=plan.route, well_name=plan.well_name)
    table = _shift_time_depth_table(table, manual_shift_s)
    anchor_info = {**anchor_info, "anchor_role": "build_initial_tdt", "anchor_shift_ms": 0.0}
    coarse_report = _coarse_report(anchor_shift_s=0.0, manual_shift_s=manual_shift_s, anchor_info=anchor_info)
    target_window = _target_tie_window_for_plan(
        plan=plan,
        survey=survey,
        config=config,
        horizon_cache=horizon_cache,
        data_root=data_root,
    )
    prepared = prepare_anchor_tdt_for_window(
        table=table,
        logset_md=logset_md,
        window=target_window,
        min_tie_samples=int(config["reject"]["min_tie_samples"]),
        support_class="rerouted_to_anchor" if rerouted_from else "anchor_integrated",
    )
    prepared = _with_report(prepared, coarse_report)
    return _run_tie_with_initial_table(
        plan=plan,
        survey=survey,
        wavelet_extractor=wavelet_extractor,
        modeler=modeler,
        config=config,
        output_dir=output_dir,
        prepared=prepared,
        extra_seed={
            "initial_table_source": "well_top_anchor",
            "rerouted_from": rerouted_from or "",
            "reroute_reason": reroute_reason or "",
            "anchor": {
                **anchor_info,
            },
            "coarse_correction": coarse_report,
        },
    )


def _run_deviated_with_tdt(
    *,
    plan: WellTiePlan,
    survey: Any,
    wavelet_extractor: Any,
    modeler: Any,
    config: dict[str, Any],
    output_dir: Path,
    horizon_cache: dict[str, HorizonSurface],
    data_root: Path,
    well_tops_df: pd.DataFrame,
    anchor_config: dict[str, Any],
    manual_shift_config: dict[str, Any],
) -> tuple[WellTieResult, dict[str, Any]]:
    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY.")
    if not plan.well_trace_file:
        raise ValueError("Plan has no well trajectory file.")

    input_las = _resolve_repo_path(plan.input_las)
    time_depth_file = _resolve_repo_path(plan.time_depth_file)
    trace_file = _resolve_repo_path(plan.well_trace_file)

    logset_md = load_vp_rho_logset_from_standard_las(input_las)
    table = load_petrel_time_depth_table(time_depth_file, domain="md")
    trajectory = WellTrajectory.from_petrel_trace(trace_file).with_well_name(plan.well_name)
    target_window = _target_tie_window_for_plan(
        plan=plan,
        survey=survey,
        config=config,
        horizon_cache=horizon_cache,
        data_root=data_root,
    )
    table, coarse_report, anchor_info = _apply_coarse_correction_to_existing_tdt(
        plan=plan,
        table=table,
        config=config,
        survey=survey,
        well_tops_df=well_tops_df,
        anchor_config=anchor_config,
        manual_shift_config=manual_shift_config,
        horizon_cache=horizon_cache,
        data_root=data_root,
        trajectory=trajectory,
    )
    if not tdt_overlaps_window(table, target_window):
        raise ValueError("tdt_no_target_window_overlap")

    prepared = prepare_tdt_with_sonic_extension(
        raw_table=table,
        logset_md=logset_md,
        window=target_window,
        min_tie_samples=int(config["reject"]["min_tie_samples"]),
    )
    prepared = _with_report(prepared, coarse_report)
    sample_start = float(prepared.report["tie_window_start_s"])
    sample_end = float(prepared.report["tie_window_end_s"])
    sample_axis = survey.sample_axis("time")
    sample_idx_start, sample_idx_end = sample_axis.window_indices(sample_start, sample_end)
    twt_axis = sample_axis.values[sample_idx_start:sample_idx_end]

    samples = sample_trajectory_on_twt(trajectory, prepared.table, twt_axis)
    trace_plan = build_nearest_trace_sample_plan(samples, survey)
    paths = _build_output_paths(output_dir, plan.well_name)
    plan_df = trace_plan.to_dataframe()
    plan_df["used_for_tie"] = False
    plan_df.to_csv(paths["trace_sample_plan"], index=False)
    outside_fraction = trace_plan.outside_fraction()
    max_outside = float(config["reject"].get("max_trajectory_outside_fraction", 0.05))
    if outside_fraction is None:
        raise ValueError("trajectory_survey_position_not_checked")
    if outside_fraction > max_outside:
        raise ValueError(
            f"trajectory_outside_fraction_exceeded: {outside_fraction:.6f} > {max_outside:.6f}"
        )

    inside_slice = trace_plan.longest_inside_slice()
    inside_count = int(inside_slice.stop - inside_slice.start)
    min_tie_samples = int(config["reject"]["min_tie_samples"])
    if inside_count < min_tie_samples:
        raise ValueError(f"trajectory_inside_tie_samples_too_few: {inside_count} < {min_tie_samples}")

    cropped_plan = trace_plan.subset(inside_slice)
    cropped_start = float(cropped_plan.rows["twt_s"].iloc[0])
    cropped_end = float(cropped_plan.rows["twt_s"].iloc[-1])
    prepared = _clip_prepared_tie_window(
        prepared,
        start_s=cropped_start,
        end_s=cropped_end,
        min_tie_samples=min_tie_samples,
        clip_reason="trajectory_inside_crop",
    )
    seismic = assemble_nearest_trace_from_plan(
        cropped_plan,
        survey,
        sample_start=float(prepared.report["tie_window_start_s"]),
        sample_end=float(prepared.report["tie_window_end_s"]),
        domain="time",
    )

    plan_df["used_for_tie"] = False
    plan_df.loc[range(inside_slice.start, inside_slice.stop), "used_for_tie"] = True
    plan_df.to_csv(paths["trace_sample_plan"], index=False)

    result, extra = _run_tie_with_initial_table(
        plan=plan,
        survey=survey,
        wavelet_extractor=wavelet_extractor,
        modeler=modeler,
        config=config,
        output_dir=output_dir,
        prepared=prepared,
        extra_seed={
            "initial_table_source": "time_depth_file",
            "time_depth_file": repo_relative_path(time_depth_file, root=REPO_ROOT),
            "well_trace_file": repo_relative_path(trace_file, root=REPO_ROOT),
            "coarse_correction": coarse_report,
            "anchor": anchor_info,
            "trace_sampling": {
                "method": "nearest",
                "trajectory_outside_fraction": float(outside_fraction),
                "trajectory_inside_sample_count": int((trace_plan.rows["survey_position"] == "inside").sum()),
                "trajectory_outside_sample_count": int((trace_plan.rows["survey_position"] == "outside").sum()),
                "used_sample_count": int(len(cropped_plan.rows)),
                "unique_trace_count": int(cropped_plan.rows[["inline_index", "xline_index"]].drop_duplicates().shape[0]),
                "trace_sample_plan_file": repo_relative_path(paths["trace_sample_plan"], root=REPO_ROOT),
            },
        },
        seismic_override=seismic,
        optimized_trace_trajectory=trajectory,
    )
    return result, extra


def _run_tie_with_initial_table(
    *,
    plan: WellTiePlan,
    survey: Any,
    wavelet_extractor: Any,
    modeler: Any,
    config: dict[str, Any],
    output_dir: Path,
    prepared: PreparedTieWindow,
    extra_seed: dict[str, Any] | None = None,
    seismic_override: Any | None = None,
    optimized_trace_trajectory: WellTrajectory | None = None,
) -> tuple[WellTieResult, dict[str, Any]]:
    from wtie.optimize import autotie
    from wtie.utils.datasets.utils import InputSet

    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY.")

    paths = _build_output_paths(output_dir, plan.well_name)
    paths["figure_dir"].mkdir(parents=True, exist_ok=True)
    logset_md = prepared.logset_md
    table = prepared.table
    overlap = validate_time_depth_table(table, logset_md.basis, min_overlap_samples=int(config["reject"]["min_tie_samples"]))
    sample_start = float(prepared.report["tie_window_start_s"])
    sample_end = float(prepared.report["tie_window_end_s"])
    seismic = (
        seismic_override
        if seismic_override is not None
        else survey.read_trace_at_xy(plan.surface_x, plan.surface_y, sample_start=sample_start, sample_end=sample_end, domain="time")
    )
    seismic = _snap_seismic_sampling_if_close(seismic, float(wavelet_extractor.expected_sampling))
    pd.DataFrame({"twt_s": seismic.basis, "seismic": seismic.values}).to_csv(paths["seismic_trace"], index=False)
    write_time_depth_table_csv(table, paths["initial_tdt"], sources=prepared.table_rows["source"].astype(str).tolist())

    wavelet_scaling_params = {
        "wavelet_min_scale": config["wavelet_scaling"]["min_scale"],
        "wavelet_max_scale": config["wavelet_scaling"]["max_scale"],
        "num_iters": config["wavelet_scaling"]["num_iters"],
    }
    search_params = {
        "num_iters": config["search_params"]["num_iters"],
        "similarity_std": config["search_params"]["similarity_std"],
        "suppress_runtime_warnings": True,
        "show_all_warnings": False,
    }
    outputs = autotie.tie_v1(
        InputSet(logset_md=logset_md, seismic=seismic, table=table, wellpath=None),  # type: ignore[arg-type]
        wavelet_extractor,
        modeler,
        wavelet_scaling_params,
        search_params=search_params,
        search_space=build_auto_tie_search_space(config["search_space"]),
        stretch_and_squeeze_params=None,
    )

    cropped_wavelet, crop_info = crop_wavelet_center_energy_normalize(outputs.wavelet, float(config["target_crop_ms"]))
    pd.DataFrame({"time_s": cropped_wavelet.basis, "amplitude": cropped_wavelet.values}).to_csv(paths["wavelet"], index=False)
    write_time_depth_table_csv(outputs.table, paths["optimized_tdt"])

    seismic_norm, synthetic, optimized_corr, optimized_nmae, synthetic_scale = scaled_synthetic_metrics(
        modeler, cropped_wavelet, outputs.r, outputs.seismic
    )
    initial_corr, initial_nmae = _compute_table_synthetic_metrics(
        logset_md, seismic, table, cropped_wavelet, modeler, float(wavelet_extractor.expected_sampling)
    )

    qc_df = pd.DataFrame(
        {
            "twt_s": outputs.seismic.basis,
            "seismic_norm": seismic_norm,
            "reflectivity": outputs.r.values,
            "synthetic_cropped_scaled": synthetic,
            "residual": seismic_norm - synthetic,
        }
    )
    qc_df.to_csv(paths["synthetic_qc"], index=False)
    _write_qc_figures(
        paths,
        table,
        outputs,
        cropped_wavelet,
        seismic_norm,
        synthetic,
        initial_table_rows=prepared.table_rows,
        target_window=TargetTieWindow(
            top_name=str(prepared.report["target_top_name"]),
            bottom_name=str(prepared.report["target_bottom_name"]),
            top_twt_s=float(prepared.report["target_top_twt_s"]),
            bottom_twt_s=float(prepared.report["target_bottom_twt_s"]),
            start_s=float(prepared.report["target_window_start_s"]),
            end_s=float(prepared.report["target_window_end_s"]),
            margin_top_s=0.0,
            margin_bottom_s=0.0,
        ),
    )

    best_params, _ = outputs.ax_client.get_best_parameters()
    best_shift_ms = float(best_params.get("table_t_shift", 0.0)) * 1000.0
    petrel_checkshot_file = _export_petrel_checkshot_tdt(paths, plan, outputs.table)
    filtered_las_file = _export_filtered_las(paths, plan, logset_md, best_params)
    optimized_trace_sample_plan_file = None
    if optimized_trace_trajectory is not None:
        optimized_trace_sample_plan_file = _write_optimized_trace_sample_plan(
            paths=paths,
            plan=plan,
            survey=survey,
            trajectory=optimized_trace_trajectory,
            table=outputs.table,
        )
    result = WellTieResult(
        well_name=plan.well_name,
        route=plan.route,
        tie_status="success",
        initial_corr=initial_corr,
        optimized_corr=optimized_corr,
        optimized_nmae=optimized_nmae,
        best_table_shift_ms=best_shift_ms,
        wavelet_file=repo_relative_path(paths["wavelet"], root=REPO_ROOT),
        optimized_tdt_file=repo_relative_path(paths["optimized_tdt"], root=REPO_ROOT),
        qc_figure_dir=repo_relative_path(paths["figure_dir"], root=REPO_ROOT),
        reasons="",
    )
    extra = dict(extra_seed or {})
    extra.update(
        {
        "initial_nmae": initial_nmae,
        "synthetic_scale": synthetic_scale,
        "crop_info": crop_info,
        "overlap": overlap,
        "tie_window": dict(prepared.report),
        "best_parameters": best_params,
        "synthetic_qc_file": repo_relative_path(paths["synthetic_qc"], root=REPO_ROOT),
        "seismic_trace_file": repo_relative_path(paths["seismic_trace"], root=REPO_ROOT),
        "initial_tdt_file": repo_relative_path(paths["initial_tdt"], root=REPO_ROOT),
        "petrel_checkshot_file": repo_relative_path(petrel_checkshot_file, root=REPO_ROOT),
        "filtered_las_file": repo_relative_path(filtered_las_file, root=REPO_ROOT),
        "optimized_trace_sample_plan_file": (
            None
            if optimized_trace_sample_plan_file is None
            else repo_relative_path(optimized_trace_sample_plan_file, root=REPO_ROOT)
        ),
        }
    )
    return result, extra


# =============================================================================
# Main
# =============================================================================


def _load_wavelet_extractor(model_path: Path, params_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"tutorial_model does not exist: {model_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"tutorial_params does not exist: {params_path}")
    from wtie.utils.datasets import tutorial as tutorial_mod

    with params_path.open("r", encoding="utf-8") as fp:
        training_parameters = yaml.load(fp, Loader=yaml.Loader)
    return tutorial_mod.load_wavelet_extractor(training_parameters, model_path)


def _write_wavelet_inventory(results: Sequence[WellTieResult], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.tie_status != "success" or not result.wavelet_file:
            continue
        wavelet_path = _resolve_repo_path(result.wavelet_file)
        df = pd.read_csv(wavelet_path)
        dt_s = float(np.median(np.diff(df["time_s"].to_numpy(dtype=np.float64)))) if len(df) > 1 else np.nan
        rows.append(
            {
                "source_well": result.well_name,
                "route": result.route,
                "wavelet_file": result.wavelet_file,
                "dt_s": dt_s,
                "n_samples": int(len(df)),
                "tie_corr": result.optimized_corr,
                "tie_nmae": result.optimized_nmae,
                "usable_as_candidate": True,
                "reasons": "",
            }
        )
    columns = [
        "source_well",
        "route",
        "wavelet_file",
        "dt_s",
        "n_samples",
        "tie_corr",
        "tie_nmae",
        "usable_as_candidate",
        "reasons",
    ]
    wavelet_df = pd.DataFrame.from_records(rows, columns=columns)
    wavelet_df.to_csv(output_dir / "wavelet_inventory.csv", index=False)
    return wavelet_df


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = _script_config(cfg)
    data_root = _resolve_repo_path(str(cfg.get("data_root", "data")))
    output_dir = _resolve_output_dir(args, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_output_dirs(output_dir)

    paths = _resolve_inputs(cfg, script_cfg)
    inventory_df = pd.read_csv(paths["inventory_file"])
    well_screen_df = pd.read_csv(paths["well_screen_file"])
    preprocess_df = pd.read_csv(paths["preprocess_status_file"])
    trajectory_df = pd.read_csv(paths["well_trajectory_file"]) if paths["well_trajectory_file"] is not None else None

    time_depth_dir = _resolve_data_path(script_cfg["time_depth_dir"], data_root=data_root)
    well_trace_dir = _resolve_data_path(script_cfg["well_trace_dir"], data_root=data_root)
    well_tops_file = _resolve_data_path(script_cfg["well_tops_file"], data_root=data_root)
    time_depth_lookup = build_file_lookup(time_depth_dir.iterdir() if time_depth_dir.exists() else [], asset_label=str(time_depth_dir))
    trace_lookup = build_file_lookup(well_trace_dir.iterdir() if well_trace_dir.exists() else [], asset_label=str(well_trace_dir))

    plans = build_tie_plan(
        inventory_df=inventory_df,
        preprocess_df=preprocess_df,
        trajectory_df=trajectory_df,
        time_depth_lookup=time_depth_lookup,
        trace_lookup=trace_lookup,
        enabled_routes=script_cfg["enabled_routes"],
        allow_near_outside=as_bool(script_cfg["reject"]["allow_near_outside"]),
    )
    if args.well:
        wanted = args.well.strip().casefold()
        plans = [plan for plan in plans if plan.well_name.casefold() == wanted]
        if not plans:
            raise ValueError(f"Well {args.well!r} was not found in tie plan.")

    plan_df = plans_dataframe(plans)
    plan_path = output_dir / "well_tie_plan.csv"
    plan_df.to_csv(plan_path, index=False)

    seismic_cfg = dict(script_cfg["seismic"])
    seismic_file = _resolve_data_path(seismic_cfg["file"], data_root=data_root)
    survey = open_survey(
        seismic_file,
        seismic_type=str(seismic_cfg.get("type", "segy")),
        segy_options=segy_options_from_config(seismic_cfg) or None,
    )

    model_path = _resolve_data_path(script_cfg["tutorial_model"], data_root=data_root)
    params_path = _resolve_data_path(script_cfg["tutorial_params"], data_root=data_root)

    results: list[WellTieResult] = []
    result_extras: dict[str, Any] = {}
    implemented_routes = {
        TieRoute.VERTICAL_WITH_TDT.value,
        TieRoute.VERTICAL_ANCHOR_FROM_TOPS.value,
        TieRoute.DEVIATED_WITH_TDT.value,
    }
    planned_to_run = [plan for plan in plans if plan.route_status == "planned" and plan.route in implemented_routes]
    if planned_to_run:
        from wtie.modeling.modeling import ConvModeler

        anchor_cfg = dict(script_cfg["coarse_correction"].get("anchor") or {})
        anchor_routes = set(anchor_cfg.get("apply_to_routes") or []) if as_bool(anchor_cfg.get("enabled", False)) else set()
        needs_anchor = any(
            plan.route == TieRoute.VERTICAL_ANCHOR_FROM_TOPS.value or plan.route in anchor_routes
            for plan in planned_to_run
        )
        well_tops_df = import_well_tops_petrel(well_tops_file) if needs_anchor else pd.DataFrame()
        anchor_config = _load_anchor_config(anchor_cfg) if needs_anchor else {}
        manual_shift_config = _load_manual_shift_config(dict(script_cfg["coarse_correction"].get("manual_shift") or {}))
        horizon_cache: dict[str, HorizonSurface] = {}
        wavelet_extractor = _load_wavelet_extractor(model_path, params_path)
        modeler = ConvModeler()
        for plan in planned_to_run:
            try:
                if plan.route == TieRoute.VERTICAL_WITH_TDT.value:
                    result, extra = _run_vertical_with_tdt(
                        plan=plan,
                        survey=survey,
                        wavelet_extractor=wavelet_extractor,
                        modeler=modeler,
                        config=script_cfg,
                        output_dir=output_dir,
                        horizon_cache=horizon_cache,
                        data_root=data_root,
                        well_tops_df=well_tops_df,
                        anchor_config=anchor_config,
                        manual_shift_config=manual_shift_config,
                    )
                elif plan.route == TieRoute.VERTICAL_ANCHOR_FROM_TOPS.value:
                    result, extra = _run_vertical_anchor_from_tops(
                        plan=plan,
                        survey=survey,
                        wavelet_extractor=wavelet_extractor,
                        modeler=modeler,
                        config=script_cfg,
                        output_dir=output_dir,
                        well_tops_df=well_tops_df,
                        anchor_config=anchor_config,
                        manual_shift_config=manual_shift_config,
                        horizon_cache=horizon_cache,
                        data_root=data_root,
                    )
                elif plan.route == TieRoute.DEVIATED_WITH_TDT.value:
                    result, extra = _run_deviated_with_tdt(
                        plan=plan,
                        survey=survey,
                        wavelet_extractor=wavelet_extractor,
                        modeler=modeler,
                        config=script_cfg,
                        output_dir=output_dir,
                        horizon_cache=horizon_cache,
                        data_root=data_root,
                        well_tops_df=well_tops_df,
                        anchor_config=anchor_config,
                        manual_shift_config=manual_shift_config,
                    )
                else:
                    raise NotImplementedError(f"Route is not implemented: {plan.route}")
                results.append(result)
                result_extras[plan.well_name] = extra
            except RerouteToAnchor as reroute:
                try:
                    if not anchor_config:
                        well_tops_df = import_well_tops_petrel(well_tops_file)
                        anchor_config = _load_anchor_config(anchor_cfg)
                    rerouted_plan = replace(plan, route=TieRoute.VERTICAL_ANCHOR_FROM_TOPS.value)
                    result, extra = _run_vertical_anchor_from_tops(
                        plan=rerouted_plan,
                        survey=survey,
                        wavelet_extractor=wavelet_extractor,
                        modeler=modeler,
                        config=script_cfg,
                        output_dir=output_dir,
                        well_tops_df=well_tops_df,
                        anchor_config=anchor_config,
                        horizon_cache=horizon_cache,
                        data_root=data_root,
                        rerouted_from=plan.route,
                        reroute_reason=reroute.reason,
                        manual_shift_config=manual_shift_config,
                    )
                except Exception as exc:
                    reason = str(exc) or type(exc).__name__
                    result = WellTieResult(
                        well_name=plan.well_name,
                        route=TieRoute.VERTICAL_ANCHOR_FROM_TOPS.value,
                        tie_status="failed",
                        reasons=f"{reroute.reason}; anchor reroute failed: {reason}",
                    )
                    extra = {
                        "rerouted_from": plan.route,
                        "reroute_reason": reroute.reason,
                        "tie_window": {
                            "target_top_name": reroute.target_window.top_name,
                            "target_bottom_name": reroute.target_window.bottom_name,
                            "target_window_start_s": reroute.target_window.start_s,
                            "target_window_end_s": reroute.target_window.end_s,
                            "tdt_support_class": "rerouted_to_anchor",
                            "window_clip_reason": "anchor_reroute_failed",
                        },
                    }
                results.append(result)
                result_extras[plan.well_name] = extra
            except Exception as exc:
                reason = str(exc) or type(exc).__name__
                results.append(
                    WellTieResult(
                        well_name=plan.well_name,
                        route=plan.route,
                        tie_status="failed",
                        reasons=reason,
                    )
                )

    metrics_df = results_dataframe(results)
    tie_window_rows = []
    metric_extra_rows = []
    for result in results:
        extra = dict(result_extras.get(result.well_name) or {})
        tie_window = dict(extra.get("tie_window") or {})
        if tie_window:
            tie_window_rows.append({"well_name": result.well_name, "route": result.route, "tie_status": result.tie_status, **tie_window})
            metric_extra_rows.append(
                {
                    "well_name": result.well_name,
                    "tie_window_start_s": tie_window.get("tie_window_start_s"),
                    "tie_window_end_s": tie_window.get("tie_window_end_s"),
                    "tdt_support_class": tie_window.get("tdt_support_class"),
                    "original_tdt_window_fraction": tie_window.get("original_tdt_window_fraction"),
                    "petrel_checkshot_file": extra.get("petrel_checkshot_file"),
                    "filtered_las_file": extra.get("filtered_las_file"),
                    "optimized_trace_sample_plan_file": extra.get("optimized_trace_sample_plan_file"),
                }
            )
    if metric_extra_rows and not metrics_df.empty:
        metrics_df = metrics_df.merge(pd.DataFrame.from_records(metric_extra_rows), on="well_name", how="left")
    metrics_path = output_dir / "well_tie_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    tie_window_path = output_dir / "tie_window_report.csv"
    pd.DataFrame.from_records(tie_window_rows).to_csv(tie_window_path, index=False)
    rejected_df = plan_df.loc[plan_df["route_status"] == "rejected"].copy()
    rejected_path = output_dir / "rejected_wells.csv"
    rejected_df.to_csv(rejected_path, index=False)
    wavelet_df = _write_wavelet_inventory(results, output_dir)
    anchor_rows = []
    for well_name, extra in result_extras.items():
        anchor = extra.get("anchor")
        if anchor:
            anchor_rows.append({"well_name": well_name, **anchor, "initial_tdt_file": extra.get("initial_tdt_file", "")})
    anchor_report_path = output_dir / "anchor_report.csv"
    anchor_report_columns = [
        "well_name",
        "well_top",
        "horizon",
        "twt_unit",
        "anchor_md_m",
        "anchor_x_m",
        "anchor_y_m",
        "anchor_position_source",
        "horizon_twt_raw",
        "anchor_twt_s",
        "inline_float",
        "xline_float",
        "horizon_sample_method",
        "horizon_nearest_line_distance",
        "horizon_nearest_inline",
        "horizon_nearest_xline",
        "anchor_role",
        "tdt_twt_at_anchor_s",
        "anchor_shift_ms",
        "initial_tdt_file",
    ]
    pd.DataFrame.from_records(anchor_rows, columns=anchor_report_columns).to_csv(anchor_report_path, index=False)

    run_summary = {
        "script": "well_auto_tie.py",
        "config_file": repo_relative_path(args.config, root=REPO_ROOT),
        "inputs": {
            "inventory_file": repo_relative_path(paths["inventory_file"], root=REPO_ROOT),
            "well_screen_file": repo_relative_path(paths["well_screen_file"], root=REPO_ROOT),
            "preprocess_status_file": repo_relative_path(paths["preprocess_status_file"], root=REPO_ROOT),
            "well_trajectory_file": repo_relative_path(paths["well_trajectory_file"], root=REPO_ROOT)
            if paths["well_trajectory_file"] is not None
            else None,
            "time_depth_dir": repo_relative_path(time_depth_dir, root=REPO_ROOT),
            "well_tops_file": repo_relative_path(well_tops_file, root=REPO_ROOT),
            "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
            "tutorial_model": repo_relative_path(model_path, root=REPO_ROOT),
            "tutorial_params": repo_relative_path(params_path, root=REPO_ROOT),
        },
        "enabled_routes": list(script_cfg["enabled_routes"]),
        "input_row_counts": {
            "inventory": int(len(inventory_df)),
            "well_screen": int(len(well_screen_df)),
            "preprocess_status": int(len(preprocess_df)),
            "well_trajectory": int(len(trajectory_df)) if trajectory_df is not None else 0,
        },
        "route_counts": _series_value_counts(plan_df["route"]),
        "route_status_counts": _series_value_counts(plan_df["route_status"]),
        "tie_status_counts": _series_value_counts(metrics_df["tie_status"]) if "tie_status" in metrics_df.columns else {},
        "planned_run_count": int(len(planned_to_run)),
        "successful_tie_count": int((metrics_df["tie_status"] == "success").sum()) if not metrics_df.empty else 0,
        "wavelet_count": int(len(wavelet_df)),
        "anchor_report_count": int(len(anchor_rows)),
        "result_extras": result_extras,
        "paths": {
            "well_tie_plan": repo_relative_path(plan_path, root=REPO_ROOT),
            "well_tie_metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
            "rejected_wells": repo_relative_path(rejected_path, root=REPO_ROOT),
            "wavelet_inventory": repo_relative_path(output_dir / "wavelet_inventory.csv", root=REPO_ROOT),
            "anchor_report": repo_relative_path(anchor_report_path, root=REPO_ROOT),
            "tie_window_report": repo_relative_path(tie_window_path, root=REPO_ROOT),
        },
    }
    write_json(output_dir / "run_summary.json", run_summary)
    print(
        f"Wrote auto-tie plan for {len(plan_df)} wells to {repo_relative_path(output_dir, root=REPO_ROOT)}; "
        f"executed {len(planned_to_run)} implemented-route wells."
    )


if __name__ == "__main__":
    main()
