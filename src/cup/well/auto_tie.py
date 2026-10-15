"""Step 4 well auto-tie: domain preparation, optimization and aligned artifacts.

Time and depth routes share the Wtie optimization. The observation coordinate
mapping stays fixed while the well time-depth relation is optimized.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import yaml

from cup.config.workflow import WorkflowConfig
from cup.seismic.horizon import HorizonSurface
from cup.seismic.petrel import import_interpretation_petrel
from cup.seismic.trace_sampling import (
    _bilinear_indices_from_line, assemble_bilinear_trace_from_plan,
    build_bilinear_trace_sample_plan,
)
from cup.seismic.viz import plot_well_waveform_qc
from cup.utils.coerce import as_bool
from cup.utils.coerce import as_bool as _as_bool, optional_float as _optional_float
from cup.utils.io import repo_relative_path, resolve_relative_path, sanitize_filename
from cup.utils.masks import true_runs
from cup.well.gaps import fill_short_depth_gaps
from cup.well.inventory import build_file_lookup, normalize_well_name
from cup.well.las import StandardVpRhoLogs, load_standard_vp_rho_logs
from cup.well.tie import (
    WellTiePlan, WellTieResult, tie_text, well_rows_by_name,
    fill_short_joint_gaps, prepare_continuous_tie_logs, resample_tie_seismic_to_dt,
    scaled_synthetic_metrics,
)
from cup.well.trajectory import (
    PreparedTieWindow, TargetTieWindow, WellTrajectory, build_tdt_from_anchor,
    crop_logset_md, find_well_top_md, load_petrel_time_depth_table,
    normalize_twt_seconds, prepare_anchor_tdt_for_window,
    prepare_tdt_with_sonic_extension, sample_trajectory_on_twt,
    tdt_overlaps_window, validate_time_depth_table, write_time_depth_table_csv,
)
from wtie.processing import grid


# Per-well correction configuration

def resolve_repo_path(value: str | Path, *, repo_root: Path) -> Path:
    return resolve_relative_path(value, root=repo_root)

def resolve_data_path(value: str | Path, *, data_root: Path) -> Path:
    return resolve_relative_path(value, root=data_root)

def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return dict(value)

def load_manual_shift_config(manual_cfg: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    config = {
        "default_ms": float(manual_cfg.get("default_ms", 0.0) or 0.0),
        "by_route_ms": dict(manual_cfg.get("by_route_ms") or {}),
        "wells_ms": {},
    }
    config_file = str(manual_cfg.get("config_file") or "").strip()
    if config_file:
        shifts = load_yaml_mapping(resolve_repo_path(config_file, repo_root=repo_root))
        shifts = dict(shifts.get("manual_shift") or shifts)
        if "default_ms" in shifts:
            config["default_ms"] = float(shifts["default_ms"] or 0.0)
        config["by_route_ms"].update(dict(shifts.get("by_route_ms") or {}))
        config["wells_ms"].update(dict(shifts.get("wells_ms") or {}))
    return config

def manual_shift_seconds(config: dict[str, Any], *, route: str, well_name: str) -> float:
    shift_ms = float(config.get("default_ms", 0.0) or 0.0)
    if route in config.get("by_route_ms", {}):
        shift_ms = float(config["by_route_ms"][route] or 0.0)
    for key, value in dict(config.get("wells_ms") or {}).items():
        if str(key).strip().casefold() == str(well_name).strip().casefold():
            shift_ms = float(value or 0.0)
            break
    return shift_ms / 1000.0

def load_anchor_config(anchor_cfg: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    config = {"default": dict(anchor_cfg.get("default") or {}),
              "wells": dict(anchor_cfg.get("wells") or {}), "enabled": as_bool(anchor_cfg.get("enabled", True)),
              "apply_to_routes": list(anchor_cfg.get("apply_to_routes") or ["vertical_anchor_from_tops"])}
    config_file = str(anchor_cfg.get("config_file") or "").strip()
    if config_file:
        data = load_yaml_mapping(resolve_repo_path(config_file, repo_root=repo_root))
        anchors = dict(data.get("anchors") or data)
        config["default"].update(dict(anchors.get("default") or {}))
        config["wells"].update(dict(anchors.get("wells") or {}))
    if not config["default"].get("well_top") or not config["default"].get("horizon"):
        raise ValueError("coarse_correction.anchor requires default.well_top and default.horizon.")
    return config

def anchor_spec_for_well(config: dict[str, Any], well_name: str) -> dict[str, Any]:
    spec = dict(config.get("default") or {})
    for key, value in dict(config.get("wells") or {}).items():
        if str(key).strip().casefold() == str(well_name).strip().casefold():
            spec.update(dict(value or {}))
            break
    spec.setdefault("twt_unit", "auto")
    return spec

def anchor_enabled_for_route(config: dict[str, Any], route: str) -> bool:
    if not as_bool(config.get("enabled", False)):
        return False
    return str(route) in {str(value) for value in config.get("apply_to_routes") or []}


# Numerical preparation and artifacts

def shift_time_depth_table(table: grid.TimeDepthTable, shift_s: float) -> grid.TimeDepthTable:
    shift = float(shift_s)
    if abs(shift) <= 1e-12:
        return table
    if table.is_md_domain:
        return grid.TimeDepthTable(twt=np.asarray(table.twt) + shift, md=np.asarray(table.md))
    return grid.TimeDepthTable(twt=np.asarray(table.twt) + shift, tvdss=np.asarray(table.tvdss))

def with_report(prepared: PreparedTieWindow, values: Mapping[str, Any]) -> PreparedTieWindow:
    report = dict(prepared.report)
    report.update(values)
    return replace(prepared, report=report)

def snap_seismic_sampling_if_close(seismic: Any, expected_dt_s: float, *, rtol: float = 1e-5) -> Any:
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

def filtered_standard_las_logset(logset_md: Any, best_params: Mapping[str, Any]) -> dict[str, grid.Log]:
    """Filter each temporary continuous MD segment and retain invalid gaps.

    Segments shorter than the filter support are emitted as invalid samples;
    copying their raw values would falsely label an unfiltered fragment as a
    filtered curve.
    """
    from wtie.optimize import tie as tie_ops

    md = np.asarray(logset_md.basis, dtype=np.float64)
    source_vp = np.asarray(logset_md.Vp.values, dtype=np.float64)
    source_rho = np.asarray(logset_md.Rho.values, dtype=np.float64)
    valid = np.isfinite(source_vp) & (source_vp > 0.0) & np.isfinite(source_rho) & (source_rho > 0.0)
    vp = np.full(md.shape, np.nan, dtype=np.float64)
    rho = np.full(md.shape, np.nan, dtype=np.float64)
    min_filter_samples = max(8, int(best_params["logs_median_size"]) + 2)
    for start, stop in true_runs(valid):
        if stop - start < min_filter_samples:
            continue
        segment = grid.LogSet(
            {
                "Vp": grid.Log(source_vp[start:stop], md[start:stop], "md", name="Vp", unit="m/s", allow_nan=False),
                "Rho": grid.Log(source_rho[start:stop], md[start:stop], "md", name="Rho", unit="g/cm3", allow_nan=False),
            }
        )
        segment = tie_ops.filter_md_logs(
            segment,
            median_size=best_params["logs_median_size"],
            threshold=best_params["logs_median_threshold"],
            std=best_params["logs_std"],
            std2=0.8 * best_params["logs_std"],
        )
        vp[start:stop] = segment.Vp.values
        rho[start:stop] = segment.Rho.values
    ai = vp * rho
    return {
        "DT_USM": grid.Log(1_000_000.0 / vp, md, "md", name="DT_USM", unit="us/m", allow_nan=True),
        "RHO_GCC": grid.Log(rho, md, "md", name="RHO_GCC", unit="g/cm3", allow_nan=True),
        "AI": grid.Log(ai, md, "md", name="AI", unit="m/s*g/cm3", allow_nan=True),
    }

def export_filtered_las(
    *,
    input_las: Path,
    output_path: Path,
    best_params: Mapping[str, Any],
    table: grid.TimeDepthTable,
    max_short_gap_s: float,
    well_name: str,
    depth_coordinates: np.ndarray | None = None,
) -> Path:
    from cup.well.las import export_logset_to_las

    standard = load_standard_vp_rho_logs(input_las)
    # The optimizer and the persisted curves share the same repaired input:
    # fill only bounded short gaps, filter, and persist the repair mask.
    # A depth route carries the exact MD-aligned depth coordinates used before
    # local TDT construction.  Reusing them keeps export and mapping on the
    # same physical two-way-time criterion; time routes use their MD-TDT.
    if depth_coordinates is None:
        export_logset, _filled = fill_short_joint_gaps(
            standard,
            table,
            max_short_gap_s=max_short_gap_s,
        )
    else:
        export_logset, _filled = fill_short_depth_gaps(
            standard,
            depth_coordinates,
            max_short_gap_s=max_short_gap_s,
        )
    filtered = filtered_standard_las_logset(export_logset, best_params)
    basis = np.asarray(standard.logs.basis, dtype=np.float64)
    filtered["AI_OBSERVED"] = grid.Log(
        np.asarray(standard.observed_mask, dtype=np.float64),
        basis,
        "md",
        name="AI_OBSERVED",
        unit="1",
        allow_nan=False,
    )
    return export_logset_to_las(
        well_name,
        filtered,
        output_path,
        curve_names=["DT_USM", "RHO_GCC", "AI", "AI_OBSERVED"],
        template_las=input_las,
    )

def output_paths(output_dir: Path, well_name: str, *, sample_domain: str) -> dict[str, Path]:
    safe = sanitize_filename(well_name)
    figure_dir = output_dir / "figures" / safe
    paths = {
        "wavelet": output_dir / "wavelets" / f"wavelet_201ms_{safe}.csv",
        "initial_tdt": output_dir / "time_depth" / f"initial_tdt_{safe}.csv",
        "source_tdt": output_dir / "time_depth" / f"source_tdt_{safe}.csv",
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
    if sample_domain == "depth":
        paths.update(
            {
                "optimized_tdt": output_dir / "time_depth" / f"optimized_tdt_{safe}.csv",
                "depth_seismic_trace": output_dir / "seismic_trace" / f"depth_seismic_trace_{safe}.csv",
                "source_depth_mapping": output_dir / "depth_mapping" / f"source_depth_mapping_{safe}.csv",
                "aligned_depth": output_dir / "depth_mapping" / f"aligned_depth_{safe}.csv",
            }
        )
    return paths

def ensure_output_dirs(output_dir: Path, *, sample_domain: str) -> None:
    names = ["wavelets", "time_depth", "petrel_checkshots", "filtered_las", "synthetic_qc", "seismic_trace", "trace_sample_plan", "figures"]
    if sample_domain == "depth":
        names.append("depth_mapping")
    for name in names:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

def compute_initial_metrics(logset_md: Any, seismic: Any, table: Any, wavelet: Any, modeler: Any, dt_s: float) -> tuple[float, float]:
    from wtie.optimize import tie as tie_ops

    if not np.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("initial synthetic dt_s must be positive.")
    logset_twt = tie_ops.convert_logs_from_md_to_twt(logset_md, None, table, float(dt_s))
    reflectivity = tie_ops.compute_reflectivity(logset_twt, angle_range=seismic.angle_range)
    seismic_on_target = resample_tie_seismic_to_dt(seismic, float(dt_s))
    seismic_match, reflectivity_match = tie_ops.match_seismic_and_reflectivity(seismic_on_target, reflectivity)
    _, _, corr, nmae, _ = scaled_synthetic_metrics(modeler, wavelet, reflectivity_match, seismic_match)
    return corr, nmae


# Well participation and route planning

class TieRoute(str, Enum):
    VERTICAL_WITH_TDT = "vertical_with_tdt"
    VERTICAL_ANCHOR_FROM_TOPS = "vertical_anchor_from_tops"
    DEVIATED_WITH_TDT = "deviated_with_tdt"
    DEVIATED_ANCHOR_FROM_TOPS = "deviated_anchor_from_tops"
    VERTICAL_LOCAL_DEPTH = "vertical_local_depth"
    DEVIATED_LOCAL_DEPTH = "deviated_local_depth"
    REJECTED = "rejected"


def _path_for_lookup(lookup: Mapping[str, Path], well_name: str) -> str:
    path = lookup.get(normalize_well_name(well_name))
    return "" if path is None else Path(path).as_posix()


def build_tie_plan(
    *,
    inventory_df: pd.DataFrame,
    preprocess_df: pd.DataFrame,
    trajectory_df: pd.DataFrame | None,
    time_depth_lookup: Mapping[str, Path],
    trace_lookup: Mapping[str, Path],
    enabled_routes: Sequence[str],
    allow_near_outside: bool = False,
    sample_domain: str = "time",
    well_roles: Mapping[str, str] | Callable[[str], str] | None = None,
) -> list[WellTiePlan]:
    """为清单中的每口井构建一行标定路由计划。

    ``sample_domain`` only changes the asset needed to select a route.  Time
    domain routes use an existing TDT, while depth-domain routes derive their
    local TWT table from the well Vp during execution.  The planner still
    records one row for every inventory well so role-based exclusions remain
    visible in the plan.
    """
    domain = str(sample_domain).strip().casefold()
    if domain not in {"time", "depth"}:
        raise ValueError("sample_domain must be 'time' or 'depth'.")

    def role_for(name: str) -> str:
        if well_roles is None:
            return "trusted"
        value = well_roles(name) if callable(well_roles) else well_roles.get(name)
        role = "trusted" if value is None else str(value).strip().casefold()
        if role not in {"trusted", "background_only", "excluded"}:
            raise ValueError(f"Unsupported well role for {name!r}: {value!r}.")
        return role

    preprocess_by_key = well_rows_by_name(preprocess_df)
    trajectory_by_key = well_rows_by_name(trajectory_df if trajectory_df is not None else pd.DataFrame())
    enabled = {str(route) for route in enabled_routes}
    plans: list[WellTiePlan] = []

    for _, inv in inventory_df.iterrows():
        well_name = str(inv["well_name"])
        key = normalize_well_name(well_name)
        pre = preprocess_by_key.get(key)
        traj = trajectory_by_key.get(key)

        survey_position = tie_text(inv.get("survey_position"))
        in_allowed_survey = survey_position == "inside" or (allow_near_outside and survey_position == "near_outside")
        has_time_depth = _as_bool(inv.get("has_time_depth")) and key in time_depth_lookup
        has_well_trace = _as_bool(inv.get("has_well_trace")) and key in trace_lookup
        has_well_tops = _as_bool(inv.get("has_well_tops"))
        usable_p = (
            pre is not None
            and _as_bool(pre.get("usable_p_sonic"))
            and tie_text(pre.get("preprocess_status")) == "passed"
        )
        usable_rho = (
            pre is not None
            and _as_bool(pre.get("usable_density"))
            and tie_text(pre.get("preprocess_status")) == "passed"
        )
        input_las = tie_text(pre.get("preprocessed_las")) if pre is not None else ""
        wellbore_initial = tie_text(inv.get("wellbore_class")) or "unknown"
        wellbore_qc = tie_text(traj.get("wellbore_class_qc")) if traj is not None else wellbore_initial
        trajectory_status = tie_text(traj.get("trajectory_status")) if traj is not None else ""
        if trajectory_status == "failed":
            wellbore_qc = "unknown"

        reasons: list[str] = []
        if not in_allowed_survey:
            reasons.append(f"survey_position_{survey_position or 'unknown'}")
        if pre is None:
            reasons.append("no_preprocess_record")
        if not usable_p:
            reasons.append("unusable_p_sonic")
        if not usable_rho:
            reasons.append("unusable_density")
        if not input_las:
            reasons.append("no_preprocessed_las")

        well_role = role_for(well_name)
        route = TieRoute.REJECTED.value
        route_status = "rejected"
        route_reasons = list(reasons)

        base_assets_ok = in_allowed_survey and usable_p and usable_rho and bool(input_las)
        if domain == "depth":
            if base_assets_ok and wellbore_qc == "vertical":
                route = TieRoute.VERTICAL_LOCAL_DEPTH.value
            elif base_assets_ok and wellbore_qc == "deviated" and has_well_trace:
                route = TieRoute.DEVIATED_LOCAL_DEPTH.value
        elif base_assets_ok and wellbore_qc == "vertical" and has_time_depth:
            route = TieRoute.VERTICAL_WITH_TDT.value
        elif base_assets_ok and wellbore_qc == "vertical" and (not has_time_depth) and has_well_tops:
            route = TieRoute.VERTICAL_ANCHOR_FROM_TOPS.value
        elif base_assets_ok and wellbore_qc == "deviated" and has_time_depth and has_well_trace:
            route = TieRoute.DEVIATED_WITH_TDT.value
        elif base_assets_ok and wellbore_qc == "deviated" and (not has_time_depth) and has_well_trace and has_well_tops:
            route = TieRoute.DEVIATED_ANCHOR_FROM_TOPS.value
        else:
            if wellbore_qc not in {"vertical", "deviated"}:
                route_reasons.append(f"wellbore_class_{wellbore_qc or 'unknown'}")
            if domain == "time" and not has_time_depth:
                route_reasons.append("no_time_depth")
            if domain == "depth" and wellbore_qc == "deviated" and not has_well_trace:
                route_reasons.append("no_well_trace")
            if domain == "time" and not has_well_tops:
                route_reasons.append("no_well_tops")

        if well_role != "trusted" and route == TieRoute.REJECTED.value:
            route_reasons.append(f"well_role_{well_role}")

        if route != TieRoute.REJECTED.value:
            if well_role != "trusted":
                route_status = "skipped_role"
                route_reasons = [f"well_role_{well_role}"]
            elif route in enabled:
                route_status = "planned"
                route_reasons = []
            else:
                route_status = "skipped_disabled"
                route_reasons = [f"route_disabled_{route}"]

        plans.append(
            WellTiePlan(
                well_name=well_name,
                route=route,
                route_status=route_status,
                wellbore_class_initial=wellbore_initial,
                wellbore_class_qc=wellbore_qc,
                has_time_depth=has_time_depth,
                has_well_trace=has_well_trace,
                has_well_tops=has_well_tops,
                usable_p_sonic=bool(usable_p),
                usable_density=bool(usable_rho),
                input_las=input_las,
                time_depth_file=_path_for_lookup(time_depth_lookup, well_name),
                well_trace_file=_path_for_lookup(trace_lookup, well_name),
                surface_x=_optional_float(inv.get("surface_x")),
                surface_y=_optional_float(inv.get("surface_y")),
                kb_m=_optional_float(inv.get("kb_m")),
                reasons=";".join(dict.fromkeys(route_reasons)),
                well_role=well_role,
            )
        )

    return plans


def plans_dataframe(plans: Sequence[WellTiePlan]) -> pd.DataFrame:
    columns = list(WellTiePlan.__dataclass_fields__.keys())
    return pd.DataFrame.from_records([plan.to_row() for plan in plans], columns=columns)


def results_dataframe(results: Sequence[WellTieResult]) -> pd.DataFrame:
    columns = list(WellTieResult.__dataclass_fields__.keys())
    return pd.DataFrame.from_records([result.to_row() for result in results], columns=columns)


def build_auto_tie_search_space(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """根据配置构建 auto-tie 超参数搜索空间。"""

    def range_parameter(name: str, bounds: Any) -> dict[str, Any]:
        values = list(bounds)
        if len(values) == 2 and values[0] == values[1]:
            return {"name": name, "type": "fixed", "value": values[0]}
        return {"name": name, "type": "range", "bounds": values, "value_type": "float"}

    return [
        {
            "name": "logs_median_size",
            "type": "choice",
            "values": list(config["logs_median_size_values"]),
            "value_type": "int",
            "is_ordered": True,
            "sort_values": True,
        },
        range_parameter("logs_median_threshold", config["logs_median_threshold_bounds"]),
        range_parameter("logs_std", config["logs_std_bounds"]),
        range_parameter("table_t_shift", config["table_t_shift_bounds"]),
    ]

def build_plans(
    *,
    inventory_df: pd.DataFrame,
    preprocess_df: pd.DataFrame,
    trajectory_df: pd.DataFrame | None,
    time_depth_dir: Path,
    well_trace_dir: Path,
    enabled_routes: list[str],
    workflow: WorkflowConfig,
    sample_domain: str,
    allow_near_outside: bool,
) -> list[WellTiePlan]:
    """Build all plan rows and apply the shared trusted/background policy."""
    workflow.well_usage.validate_names(inventory_df["well_name"].astype(str).tolist())
    time_depth_lookup = build_file_lookup(
        time_depth_dir.iterdir() if time_depth_dir.exists() else [],
        asset_label=str(time_depth_dir),
    )
    trace_lookup = build_file_lookup(
        well_trace_dir.iterdir() if well_trace_dir.exists() else [],
        asset_label=str(well_trace_dir),
    )
    roles = {str(name): workflow.well_usage.role_for(str(name)) for name in inventory_df["well_name"]}
    return build_tie_plan(
        inventory_df=inventory_df,
        preprocess_df=preprocess_df,
        trajectory_df=trajectory_df,
        time_depth_lookup=time_depth_lookup,
        trace_lookup=trace_lookup,
        enabled_routes=enabled_routes,
        allow_near_outside=allow_near_outside,
        sample_domain=sample_domain,
        well_roles=roles,
    )

def select_plans(plans: list[WellTiePlan], well_name: str | None) -> list[WellTiePlan]:
    if well_name is None:
        return plans
    wanted = str(well_name).strip().casefold()
    selected = [plan for plan in plans if plan.well_name.strip().casefold() == wanted]
    if not selected:
        raise ValueError(f"Well {well_name!r} was not found in tie plan.")
    return selected


# Time-domain input preparation

class RerouteToAnchor(Exception):
    """Signal that a TDT route should be replaced by the anchor route."""

    def __init__(self, reason: str, target_window: TargetTieWindow):
        super().__init__(reason)
        self.reason = reason
        self.target_window = target_window

def _load_horizon(path: Path, cache: dict[str, HorizonSurface]) -> HorizonSurface:
    key = str(path.resolve())
    if key not in cache:
        cache[key] = HorizonSurface.from_petrel_dataframe(import_interpretation_petrel(path), name=path.name)
    return cache[key]

def _horizon_path(value: str, *, data_root: Path) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("target_interval horizon cannot be empty.")
    if "/" not in text and "\\" not in text:
        text = f"interpre/{text}"
    return resolve_data_path(text, data_root=data_root)

def _horizon_name(value: str) -> str:
    text = str(value).strip()
    return Path(text).name if "/" in text or "\\" in text else text

def _target_spec(config: dict[str, Any], side: str) -> tuple[str, str]:
    value = dict(config.get("target_interval") or {}).get(f"{side}_horizon")
    if value is None or not str(value).strip():
        raise ValueError(f"target_interval.{side}_horizon is required for time-domain auto-tie.")
    return str(value), _horizon_name(str(value))

def target_window_for_plan(
    *,
    plan: WellTiePlan,
    survey: Any,
    config: dict[str, Any],
    horizon_cache: dict[str, HorizonSurface],
    data_root: Path,
) -> TargetTieWindow:
    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY.")
    target_cfg = dict(config.get("target_interval") or {})
    top_value, top_name = _target_spec(config, "top")
    bottom_value, bottom_name = _target_spec(config, "bottom")
    il, xl = survey.line_geometry.coord_to_line(float(plan.surface_x), float(plan.surface_y))
    top_sample = _load_horizon(_horizon_path(top_value, data_root=data_root), horizon_cache).sample_at_line(il, xl)
    bottom_sample = _load_horizon(_horizon_path(bottom_value, data_root=data_root), horizon_cache).sample_at_line(il, xl)
    unit = str(target_cfg.get("twt_unit", "auto"))
    top_twt = normalize_twt_seconds(float(top_sample["value"]), unit=unit)
    bottom_twt = normalize_twt_seconds(float(bottom_sample["value"]), unit=unit)
    if bottom_twt < top_twt:
        top_twt, bottom_twt = bottom_twt, top_twt
    margin_top = float(target_cfg.get("margin_top_ms", 100.0)) / 1000.0
    margin_bottom = float(target_cfg.get("margin_bottom_ms", 100.0)) / 1000.0
    start = max(0.0, top_twt - margin_top)
    end = bottom_twt + margin_bottom
    if end <= start:
        raise ValueError(f"Invalid target tie window [{start}, {end}] for {plan.well_name}.")
    return TargetTieWindow(
        top_name=top_name,
        bottom_name=bottom_name,
        top_twt_s=top_twt,
        bottom_twt_s=bottom_twt,
        start_s=start,
        end_s=end,
        margin_top_s=margin_top,
        margin_bottom_s=margin_bottom,
        top_sample_method=str(top_sample["method"]),
        bottom_sample_method=str(bottom_sample["method"]),
        top_nearest_line_distance=float(top_sample["nearest_line_distance"]),
        bottom_nearest_line_distance=float(bottom_sample["nearest_line_distance"]),
    )

def target_window_for_trajectory(
    *,
    plan: WellTiePlan,
    survey: Any,
    config: dict[str, Any],
    horizon_cache: dict[str, HorizonSurface],
    data_root: Path,
    trajectory: WellTrajectory,
    table: grid.TimeDepthTable,
) -> TargetTieWindow:
    if not table.is_md_domain:
        raise ValueError("Trajectory target window expects an MD-domain TDT.")
    target_cfg = dict(config.get("target_interval") or {})
    top_value, top_name = _target_spec(config, "top")
    bottom_value, bottom_name = _target_spec(config, "bottom")
    top_surface = _load_horizon(_horizon_path(top_value, data_root=data_root), horizon_cache)
    bottom_surface = _load_horizon(_horizon_path(bottom_value, data_root=data_root), horizon_cache)
    table_twt = np.asarray(table.twt, dtype=np.float64)
    table_md = np.asarray(table.md, dtype=np.float64)
    seismic_time = np.asarray(survey.sample_axis("time").values, dtype=np.float64)
    keep = (seismic_time >= table_twt[0]) & (seismic_time <= table_twt[-1])
    if not np.any(keep):
        raise ValueError("No seismic time samples overlap the deviated-well TDT range.")
    samples = sample_trajectory_on_twt(trajectory.with_well_name(plan.well_name), table, seismic_time[keep])
    top_values: list[float] = []
    bottom_values: list[float] = []
    top_methods: list[str] = []
    bottom_methods: list[str] = []
    top_distances: list[float] = []
    bottom_distances: list[float] = []
    for _, row in samples.rows.iterrows():
        il, xl = survey.line_geometry.coord_to_line(float(row["x_m"]), float(row["y_m"]))
        try:
            top = top_surface.sample_at_line(il, xl)
            bottom = bottom_surface.sample_at_line(il, xl)
        except ValueError:
            continue
        top_twt = normalize_twt_seconds(float(top["value"]), unit=str(target_cfg.get("twt_unit", "auto")))
        bottom_twt = normalize_twt_seconds(float(bottom["value"]), unit=str(target_cfg.get("twt_unit", "auto")))
        if bottom_twt < top_twt:
            top_twt, bottom_twt = bottom_twt, top_twt
        if top_twt <= float(row["twt_s"]) <= bottom_twt:
            top_values.append(top_twt)
            bottom_values.append(bottom_twt)
            top_methods.append(str(top["method"]))
            bottom_methods.append(str(bottom["method"]))
            top_distances.append(float(top["nearest_line_distance"]))
            bottom_distances.append(float(bottom["nearest_line_distance"]))
    if not top_values:
        raise ValueError(
            f"No deviated trajectory samples fall inside target horizons {top_name!r} and {bottom_name!r}."
        )
    top_twt = float(np.nanmin(top_values))
    bottom_twt = float(np.nanmax(bottom_values))
    margin_top = float(target_cfg.get("margin_top_ms", 100.0)) / 1000.0
    margin_bottom = float(target_cfg.get("margin_bottom_ms", 100.0)) / 1000.0
    start = max(0.0, min(top_twt, bottom_twt) - margin_top)
    end = max(top_twt, bottom_twt) + margin_bottom
    return TargetTieWindow(
        top_name=top_name,
        bottom_name=bottom_name,
        top_twt_s=min(top_twt, bottom_twt),
        bottom_twt_s=max(top_twt, bottom_twt),
        start_s=start,
        end_s=end,
        margin_top_s=margin_top,
        margin_bottom_s=margin_bottom,
        top_sample_method="trajectory:" + ",".join(sorted(set(top_methods))),
        bottom_sample_method="trajectory:" + ",".join(sorted(set(bottom_methods))),
        top_nearest_line_distance=max(top_distances) if top_distances else None,
        bottom_nearest_line_distance=max(bottom_distances) if bottom_distances else None,
    )

def _coarse_report(anchor_shift_s: float, manual_shift_s: float, anchor_info: dict[str, Any] | None) -> dict[str, Any]:
    total = float(anchor_shift_s) + float(manual_shift_s)
    return {
        "coarse_anchor_shift_ms": float(anchor_shift_s) * 1000.0,
        "coarse_manual_shift_ms": float(manual_shift_s) * 1000.0,
        "coarse_total_shift_ms": total * 1000.0,
        "coarse_anchor_applied": bool(anchor_info),
        "coarse_manual_applied": bool(abs(float(manual_shift_s)) > 1e-12),
    }

def _anchor_position(plan: WellTiePlan, anchor_md_m: float, trajectory: WellTrajectory | None) -> tuple[float, float, str]:
    if trajectory is not None:
        p = trajectory.position_at_md(float(anchor_md_m))
        return float(p["x_m"]), float(p["y_m"]), "trajectory_at_anchor_md"
    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY for anchor sampling.")
    return float(plan.surface_x), float(plan.surface_y), "surface_xy"

def _sample_anchor(
    *,
    plan: WellTiePlan,
    survey: Any,
    tops: pd.DataFrame,
    config: dict[str, Any],
    cache: dict[str, HorizonSurface],
    data_root: Path,
    trajectory: WellTrajectory | None,
) -> dict[str, Any]:
    spec = anchor_spec_for_well(config, plan.well_name)
    md = find_well_top_md(tops, well_name=plan.well_name, surface=str(spec["well_top"]))
    x, y, source = _anchor_position(plan, md, trajectory)
    il, xl = survey.line_geometry.coord_to_line(x, y)
    surface = _load_horizon(_horizon_path(str(spec["horizon"]), data_root=data_root), cache)
    sample = surface.sample_at_line(il, xl)
    twt = normalize_twt_seconds(float(sample["value"]), unit=str(spec.get("twt_unit", "auto")))
    return {
        "well_top": str(spec["well_top"]),
        "horizon": repo_relative_path(_horizon_path(str(spec["horizon"]), data_root=data_root), root=data_root.parent),
        "anchor_md_m": float(md),
        "anchor_x_m": float(x),
        "anchor_y_m": float(y),
        "anchor_position_source": source,
        "horizon_twt_raw": float(sample["value"]),
        "anchor_twt_s": float(twt),
        "inline_float": float(il),
        "xline_float": float(xl),
        "horizon_sample_method": str(sample["method"]),
        "horizon_nearest_line_distance": float(sample["nearest_line_distance"]),
        "horizon_nearest_inline": float(sample["nearest_inline"]),
        "horizon_nearest_xline": float(sample["nearest_xline"]),
    }

def _apply_existing_tdt_correction(
    *,
    plan: WellTiePlan,
    table: grid.TimeDepthTable,
    config: dict[str, Any],
    survey: Any,
    tops: pd.DataFrame,
    anchor_cfg: dict[str, Any],
    manual_cfg: dict[str, Any],
    cache: dict[str, HorizonSurface],
    data_root: Path,
    trajectory: WellTrajectory | None,
) -> tuple[grid.TimeDepthTable, dict[str, Any], dict[str, Any] | None]:
    anchor_shift = 0.0
    anchor_info = None
    if anchor_enabled_for_route(anchor_cfg, plan.route):
        anchor_info = _sample_anchor(
            plan=plan,
            survey=survey,
            tops=tops,
            config=anchor_cfg,
            cache=cache,
            data_root=data_root,
            trajectory=trajectory,
        )
        if not table.is_md_domain:
            raise ValueError("Existing time-domain TDT must be MD-domain.")
        md = np.asarray(table.md, dtype=np.float64)
        anchor_md = float(anchor_info["anchor_md_m"])
        if anchor_md < md[0] or anchor_md > md[-1]:
            raise ValueError(f"Anchor MD {anchor_md} is outside TDT range for {plan.well_name}.")
        current = float(np.interp(anchor_md, md, table.twt))
        anchor_shift = float(anchor_info["anchor_twt_s"]) - current
        anchor_info = {**anchor_info, "anchor_role": "shift_existing_tdt", "tdt_twt_at_anchor_s": current, "anchor_shift_ms": anchor_shift * 1000.0}
    manual = manual_shift_seconds(manual_cfg, route=plan.route, well_name=plan.well_name)
    return shift_time_depth_table(table, anchor_shift + manual), _coarse_report(anchor_shift, manual, anchor_info), anchor_info

def _provisional_anchor_logset(standard: Any) -> tuple[grid.LogSet, int]:
    md = np.asarray(standard.logs.basis, dtype=np.float64)
    vp = np.asarray(standard.logs.Vp.values, dtype=np.float64)
    rho = np.asarray(standard.logs.Rho.values, dtype=np.float64)
    valid = np.isfinite(vp) & (vp > 0.0) & np.isfinite(rho) & (rho > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        raise ValueError("Anchor TDT construction requires two joint-valid Vp/Rho samples.")
    if np.all(valid):
        return standard.logs, 0
    return (
        grid.LogSet(
            {
                "Vp": grid.Log(np.interp(md, md[valid], vp[valid]), md, "md", name="Vp", unit="m/s", allow_nan=False),
                "Rho": grid.Log(np.interp(md, md[valid], rho[valid]), md, "md", name="Rho", unit="g/cm3", allow_nan=False),
            }
        ),
        int(np.count_nonzero(~valid)),
    )

def prepare_time_route(
    *,
    plan: WellTiePlan,
    survey: Any,
    config: dict[str, Any],
    output_dir: Path,
    data_root: Path,
    repo_root: Path,
    well_tops_df: pd.DataFrame,
    anchor_config: dict[str, Any],
    manual_shift_config: dict[str, Any],
    horizon_cache: dict[str, HorizonSurface],
    rerouted_from: str | None = None,
    reroute_reason: str | None = None,
) -> tuple[PreparedTieWindow, dict[str, Any], Any | None, WellTrajectory | None]:
    """Prepare one time-domain route for the shared optimizer."""
    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY.")
    standard = load_standard_vp_rho_logs(resolve_repo_path(plan.input_las, repo_root=repo_root))
    trajectory = None
    if plan.well_trace_file:
        trajectory = WellTrajectory.from_petrel_trace(resolve_repo_path(plan.well_trace_file, repo_root=repo_root)).with_well_name(plan.well_name)
    if plan.route in {"vertical_with_tdt", "deviated_with_tdt"}:
        if not plan.time_depth_file:
            raise ValueError("Time-domain TDT route requires a time_depth_file.")
        table = load_petrel_time_depth_table(resolve_repo_path(plan.time_depth_file, repo_root=repo_root), domain="md")
        table, coarse, anchor = _apply_existing_tdt_correction(
            plan=plan, table=table, config=config, survey=survey, tops=well_tops_df,
            anchor_cfg=anchor_config, manual_cfg=manual_shift_config, cache=horizon_cache, data_root=data_root, trajectory=trajectory,
        )
        target = (
            target_window_for_trajectory(
                plan=plan, survey=survey, config=config, horizon_cache=horizon_cache, data_root=data_root, trajectory=trajectory, table=table
            )
            if plan.route == "deviated_with_tdt" and trajectory is not None
            else target_window_for_plan(plan=plan, survey=survey, config=config, horizon_cache=horizon_cache, data_root=data_root)
        )
        if not tdt_overlaps_window(table, target):
            raise RerouteToAnchor("tdt_no_target_window_overlap_reroute_anchor", target)
    else:
        if not anchor_enabled_for_route(anchor_config, plan.route):
            raise ValueError("coarse_correction.anchor must enable vertical_anchor_from_tops.")
        mapping, filled = _provisional_anchor_logset(standard)
        anchor = _sample_anchor(plan=plan, survey=survey, tops=well_tops_df, config=anchor_config, cache=horizon_cache, data_root=data_root, trajectory=trajectory)
        table = build_tdt_from_anchor(mapping, anchor_md_m=float(anchor["anchor_md_m"]), anchor_twt_s=float(anchor["anchor_twt_s"]))
        manual = manual_shift_seconds(manual_shift_config, route=plan.route, well_name=plan.well_name)
        table = shift_time_depth_table(table, manual)
        coarse = _coarse_report(0.0, manual, anchor)
        coarse.update({"anchor_tdt_mapping_only_interpolation": bool(filled), "anchor_tdt_mapping_only_filled_samples": int(filled)})
        target = target_window_for_plan(plan=plan, survey=survey, config=config, horizon_cache=horizon_cache, data_root=data_root)
    continuous = prepare_continuous_tie_logs(
        standard,
        table,
        window_start_s=target.start_s,
        window_end_s=target.end_s,
        max_short_gap_s=float(config["reject"]["max_short_log_gap_s"]),
        min_tie_samples=int(config["reject"]["min_tie_samples"]),
        seismic_sample_interval_s=float(survey.sample_axis("time").step),
    )
    # Keep the complete, coarse-corrected source mapping as a Step-4
    # artifact.  The optimizer uses a cropped window below, while Step 6
    # needs this full mapping plus the optimized constant shift to recover
    # source LAS samples outside the tie window without extrapolation.
    paths = output_paths(output_dir, plan.well_name, sample_domain="time")
    write_time_depth_table_csv(table, paths["source_tdt"])
    target = replace(target, start_s=continuous.start_twt_s, end_s=continuous.end_twt_s)
    if plan.route in {"vertical_anchor_from_tops", "deviated_anchor_from_tops"}:
        prepared = prepare_anchor_tdt_for_window(
            table=table,
            logset_md=continuous.logs,
            window=target,
            min_tie_samples=int(config["reject"]["min_tie_samples"]),
            support_class="rerouted_to_anchor" if rerouted_from else "anchor_integrated",
        )
    else:
        prepared = prepare_tdt_with_sonic_extension(
            raw_table=table,
            logset_md=continuous.logs,
            window=target,
            min_tie_samples=int(config["reject"]["min_tie_samples"]),
        )
    prepared = with_report(prepared, coarse)
    prepared = with_report(prepared, continuous.qc)
    prepared = with_report(prepared, {"rerouted_from": rerouted_from or "", "reroute_reason": reroute_reason or ""})
    extra = {
        "initial_table_source": "time_depth_file" if plan.route.endswith("with_tdt") else "well_top_anchor",
        "time_depth_file": repo_relative_path(resolve_repo_path(plan.time_depth_file, repo_root=repo_root), root=repo_root) if plan.time_depth_file else "",
        "coarse_correction": coarse,
        "anchor": anchor,
        "source_tdt_file": repo_relative_path(paths["source_tdt"], root=repo_root),
    }
    seismic_override = None
    if trajectory is not None and plan.route == "deviated_with_tdt":
        sample_start = float(prepared.report["tie_window_start_s"])
        sample_end = float(prepared.report["tie_window_end_s"])
        axis = survey.sample_axis("time")
        start, end = axis.window_indices(sample_start, sample_end)
        twt_axis = axis.values[start:end]
        samples = sample_trajectory_on_twt(trajectory, prepared.table, twt_axis)
        trace_plan = build_bilinear_trace_sample_plan(samples, survey)
        if trace_plan.outside_fraction() is None or trace_plan.outside_fraction() > float(config["reject"].get("max_trajectory_outside_fraction", 0.05)):
            raise ValueError("trajectory_outside_fraction_exceeded")
        inside = trace_plan.longest_inside_slice()
        if inside.stop - inside.start < int(config["reject"]["min_tie_samples"]):
            raise ValueError("trajectory_inside_tie_samples_too_few")
        trace_plan = trace_plan.subset(inside)
        seismic_override = assemble_bilinear_trace_from_plan(
            trace_plan,
            survey,
            sample_start=float(prepared.report["tie_window_start_s"]),
            sample_end=float(prepared.report["tie_window_end_s"]),
            domain="time",
        )
        paths = output_paths(output_dir, plan.well_name, sample_domain="time")
        trace_plan.to_dataframe().to_csv(paths["trace_sample_plan"], index=False)
        extra["trace_sampling"] = {
            "method": "bilinear",
            "used_sample_count": int(len(trace_plan.rows)),
            "trace_sample_plan_file": repo_relative_path(paths["trace_sample_plan"], root=repo_root),
        }
    return prepared, extra, seismic_override, trajectory


# Depth-domain input preparation

@dataclass(frozen=True)
class DepthPrepared:
    prepared: PreparedTieWindow
    seismic: grid.Seismic
    source_depth_mapping: pd.DataFrame
    native_seismic: grid.Seismic
    target_top_tvdss_m: float
    target_bottom_tvdss_m: float
    trajectory: WellTrajectory | None
    depth_coordinates: np.ndarray

def _target_horizon_values(config: dict[str, Any]) -> tuple[Any, Any]:
    target = dict(config.get("target_interval") or {})
    top = target.get("top_horizon")
    bottom = target.get("bottom_horizon")
    missing_top = top is None or (isinstance(top, str) and not top.strip())
    missing_bottom = bottom is None or (isinstance(bottom, str) and not bottom.strip())
    if missing_top or missing_bottom:
        top = target.get("top_depth_m")
        bottom = target.get("bottom_depth_m")
    if top is None or bottom is None:
        raise ValueError(
            "Depth-domain auto-tie requires target_interval.top_horizon/bottom_horizon "
            "or target_interval.top_depth_m/bottom_depth_m."
        )
    return top, bottom

def _sample_target_depths(
    *,
    plan: WellTiePlan,
    survey: Any,
    config: dict[str, Any],
    cache: dict[str, HorizonSurface],
    data_root: Path,
    trajectory: WellTrajectory | None,
) -> tuple[float, float, str, str, float, float]:
    top, bottom = _target_horizon_values(config)
    target = dict(config.get("target_interval") or {})
    if np.isscalar(top) and np.isscalar(bottom) and isinstance(top, Real) and isinstance(bottom, Real):
        values = (float(top), float(bottom))
        names = (str(top), str(bottom))
    else:
        top_surface = _load_horizon(_horizon_path(top, data_root=data_root), cache)
        bottom_surface = _load_horizon(_horizon_path(bottom, data_root=data_root), cache)
        if trajectory is None:
            if plan.surface_x is None or plan.surface_y is None:
                raise ValueError("Depth target horizon sampling requires plan surface XY.")
            positions = [(float(plan.surface_x), float(plan.surface_y))]
            top_values = []
            bottom_values = []
            for x, y in positions:
                il, xl = survey.line_geometry.coord_to_line(float(x), float(y))
                top_values.append(float(top_surface.sample_at_line(il, xl)["value"]))
                bottom_values.append(float(bottom_surface.sample_at_line(il, xl)["value"]))
            values = (float(top_values[0]), float(bottom_values[0]))
        else:
            top_values = []
            bottom_values = []
            for x, y in zip(trajectory.x_m, trajectory.y_m):
                il, xl = survey.line_geometry.coord_to_line(float(x), float(y))
                top_values.append(float(top_surface.sample_at_line(il, xl)["value"]))
                bottom_values.append(float(bottom_surface.sample_at_line(il, xl)["value"]))
            top_values_array = np.asarray(top_values, dtype=np.float64)
            bottom_values_array = np.asarray(bottom_values, dtype=np.float64)
            trajectory_depth = np.asarray(trajectory.tvdss_m, dtype=np.float64)
            finite = np.isfinite(trajectory_depth) & np.isfinite(top_values_array) & np.isfinite(bottom_values_array)
            if np.count_nonzero(finite) < 2:
                raise ValueError("Depth target horizons have fewer than two valid trajectory samples.")

            def crossings(horizon_depth: np.ndarray) -> list[float]:
                md = np.asarray(trajectory.md_m, dtype=np.float64)[finite]
                z = trajectory_depth[finite]
                f = z - horizon_depth[finite]
                found: list[float] = []
                for i in range(f.size - 1):
                    if f[i] == 0.0:
                        found.append(float(z[i]))
                    if f[i] * f[i + 1] < 0.0:
                        fraction = float(-f[i] / (f[i + 1] - f[i]))
                        found.append(float(z[i] + fraction * (z[i + 1] - z[i])))
                if f[-1] == 0.0:
                    found.append(float(z[-1]))
                return found

            top_crossings = crossings(top_values_array)
            bottom_crossings = crossings(bottom_values_array)
            if not top_crossings or not bottom_crossings:
                raise ValueError("Depth target horizons do not intersect the valid well trajectory.")
            values = (float(np.median(top_crossings)), float(np.median(bottom_crossings)))
        names = (str(top), str(bottom))
    top_depth, bottom_depth = sorted(values)
    margin_top = float(target.get("margin_top_m", 100.0))
    margin_bottom = float(target.get("margin_bottom_m", 100.0))
    if not np.isfinite(margin_top) or not np.isfinite(margin_bottom) or margin_top < 0 or margin_bottom < 0:
        raise ValueError("Depth target margins must be finite and non-negative metres.")
    return top_depth - margin_top, bottom_depth + margin_bottom, names[0], names[1], top_depth, bottom_depth

def _valid_segments(standard: Any, trajectory: WellTrajectory | None) -> list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    md = np.asarray(standard.logs.basis, dtype=np.float64)
    vp = np.asarray(standard.logs.Vp.values, dtype=np.float64)
    rho = np.asarray(standard.logs.Rho.values, dtype=np.float64)
    valid = np.isfinite(vp) & (vp > 0.0) & np.isfinite(rho) & (rho > 0.0)
    out = []
    for start, stop in true_runs(valid):
        if stop - start < 2:
            continue
        md_seg = md[start:stop]
        if trajectory is None:
            raise ValueError("Vertical depth auto-tie requires a finite KB datum.")
        if md_seg[0] < trajectory.md_m[0] or md_seg[-1] > trajectory.md_m[-1]:
            raise ValueError("Well log MD support lies outside the trajectory MD support.")
        tvdss = np.interp(md_seg, trajectory.md_m, trajectory.tvdss_m)
        if np.any(~np.isfinite(tvdss)) or np.any(np.diff(tvdss) <= 0.0):
            raise ValueError("Depth-domain well trajectory must have strictly increasing TVDSS in each valid log segment.")
        out.append((start, stop, md_seg, tvdss, vp[start:stop]))
    return out

def _vertical_segments(standard: Any, kb_m: float) -> list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    md = np.asarray(standard.logs.basis, dtype=np.float64)
    vp = np.asarray(standard.logs.Vp.values, dtype=np.float64)
    rho = np.asarray(standard.logs.Rho.values, dtype=np.float64)
    valid = np.isfinite(vp) & (vp > 0.0) & np.isfinite(rho) & (rho > 0.0)
    out = []
    for start, stop in true_runs(valid):
        if stop - start < 2:
            continue
        md_seg = md[start:stop]
        out.append((start, stop, md_seg, md_seg - float(kb_m), vp[start:stop]))
    return out

def _select_segment(
    segments: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]],
    *,
    seismic_min: float,
    seismic_max: float,
    target_min: float,
    target_max: float,
    min_samples: int,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    candidates = []
    for item in segments:
        start, stop, md, z, vp = item
        lo = max(float(z[0]), seismic_min, target_min)
        hi = min(float(z[-1]), seismic_max, target_max)
        count = int(np.count_nonzero((z >= lo) & (z <= hi)))
        if count >= max(2, min_samples):
            candidates.append((count, item))
    if not candidates:
        raise ValueError("No continuous finite Vp/Rho segment covers the depth tie window.")
    return max(candidates, key=lambda value: value[0])[1]

def _build_local_table(md: np.ndarray, tvdss: np.ndarray, vp: np.ndarray) -> tuple[grid.TimeDepthTable, pd.DataFrame]:
    tdt = grid.build_local_tdt_from_vp(tvdss_m=tvdss, vp_mps=vp, md_m=md)
    table = grid.TimeDepthTable(twt=tdt["twt_s"].to_numpy(dtype=np.float64), md=tdt["md_m"].to_numpy(dtype=np.float64))
    mapping = pd.DataFrame({"tvdss_m": tdt["tvdss_m"].to_numpy(dtype=np.float64), "twt_s": tdt["twt_s"].to_numpy(dtype=np.float64)})
    return table, mapping

def _windowed_prepared(
    *,
    table: grid.TimeDepthTable,
    logset_md: Any,
    target_start_s: float,
    target_end_s: float,
    target_top_name: str,
    target_bottom_name: str,
    target_top_depth: float,
    target_bottom_depth: float,
    target_top_twt: float,
    target_bottom_twt: float,
    seismic_dt_s: float,
    clip_reason: str,
    min_samples: int,
) -> PreparedTieWindow:
    if target_end_s <= target_start_s:
        raise ValueError("Depth target tie window is empty.")
    if target_start_s < table.twt[0] - 1e-9 or target_end_s > table.twt[-1] + 1e-9:
        raise ValueError("Depth target tie window is outside the continuous local TDT.")
    keep = (table.twt >= target_start_s) & (table.twt <= target_end_s)
    twt = table.twt[keep]
    md = table.md[keep]
    if twt.size < 2:
        twt = np.array([target_start_s, target_end_s], dtype=np.float64)
        md = np.interp(twt, table.twt, table.md)
    else:
        rows = [(float(t), float(m)) for t, m in zip(twt, md)]
        for boundary in (target_start_s, target_end_s):
            if not np.any(np.isclose(twt, boundary, rtol=0.0, atol=1e-9)):
                rows.append((float(boundary), float(np.interp(boundary, table.twt, table.md))))
        rows.sort()
        twt = np.array([row[0] for row in rows], dtype=np.float64)
        md = np.array([row[1] for row in rows], dtype=np.float64)
    clipped = grid.TimeDepthTable(twt=twt, md=md)
    cropped = crop_logset_md(logset_md, float(md[0]), float(md[-1]), min_samples=2)
    if not np.isfinite(seismic_dt_s) or seismic_dt_s <= 0.0:
        raise ValueError("Depth tie seismic sampling interval must be positive and finite.")
    seismic_count = int(np.floor((float(twt[-1]) - float(twt[0])) / float(seismic_dt_s) + 1e-9)) + 1
    if seismic_count < int(min_samples):
        raise ValueError(f"Depth tie window has too few seismic-grid samples: {seismic_count} < {min_samples}.")
    report = {
        "target_top_name": target_top_name,
        "target_bottom_name": target_bottom_name,
        "target_top_twt_s": float(target_top_twt),
        "target_bottom_twt_s": float(target_bottom_twt),
        "target_window_start_s": float(target_start_s),
        "target_window_end_s": float(target_end_s),
        "tie_window_start_s": float(twt[0]),
        "tie_window_end_s": float(twt[-1]),
        "tie_window_duration_s": float(twt[-1] - twt[0]),
        "tie_window_md_min_m": float(cropped.basis[0]),
        "tie_window_md_max_m": float(cropped.basis[-1]),
        "tie_window_log_sample_count": int(cropped.basis.size),
        "target_top_tvdss_m": float(target_top_depth),
        "target_bottom_tvdss_m": float(target_bottom_depth),
        "depth_window_clip_reason": str(clip_reason),
        "continuous_tie_sample_count": seismic_count,
        "continuous_tie_log_sample_count": int(cropped.basis.size),
    }
    return PreparedTieWindow(table=clipped, logset_md=cropped, table_rows=pd.DataFrame({"twt_s": twt, "md_m": md, "source": "depth_vp"}), report=report)

def _native_vertical_trace(survey: Any, plan: WellTiePlan) -> Any:
    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Depth trace sampling requires surface XY.")
    trace = survey.read_trace_at_xy(plan.surface_x, plan.surface_y, domain="depth")
    return trace

def _sample_bilinear_depth(
    survey: Any,
    *,
    positions: list[tuple[float, float, float]],
    trace_cache: dict[tuple[int, int], Any],
) -> np.ndarray:
    rows: list[dict[str, Any]] = []
    for idx, (x, y, z) in enumerate(positions):
        il, xl = survey.line_geometry.coord_to_line(float(x), float(y))
        plan = _bilinear_indices_from_line(survey, il, xl)
        rows.append({"twt_s": float(z), "x_m": x, "y_m": y, "trace_plan_index": idx, **plan, "survey_position": "inside"})
    for row in rows:
        for prefix in ("trace00", "trace01", "trace10", "trace11"):
            if float(row.get(f"{prefix}_weight", 0.0) or 0.0) <= 0:
                continue
            key = (int(row[f"{prefix}_inline_index"]), int(row[f"{prefix}_xline_index"]))
            if key not in trace_cache:
                traces = survey.read_traces_at_indices([key], domain="depth")
                trace_cache[key] = traces[key]
    values = []
    for row in rows:
        value = 0.0
        weight_sum = 0.0
        z = float(row["twt_s"])
        for prefix in ("trace00", "trace01", "trace10", "trace11"):
            weight = float(row.get(f"{prefix}_weight", 0.0) or 0.0)
            if weight <= 0:
                continue
            key = (int(row[f"{prefix}_inline_index"]), int(row[f"{prefix}_xline_index"]))
            trace = trace_cache[key]
            value += weight * float(np.interp(z, trace.basis, trace.values))
            weight_sum += weight
        if weight_sum <= 0:
            raise ValueError("Depth bilinear sample has zero trace weight.")
        values.append(value / weight_sum)
    return np.asarray(values, dtype=np.float64)

def prepare_depth_route(
    *,
    plan: WellTiePlan,
    survey: Any,
    config: dict[str, Any],
    output_dir: Path,
    data_root: Path,
    repo_root: Path,
    horizon_cache: dict[str, HorizonSurface],
) -> DepthPrepared:
    if plan.kb_m is None or not np.isfinite(float(plan.kb_m)):
        raise ValueError(f"Depth-domain route requires finite KB for {plan.well_name}.")
    standard = load_standard_vp_rho_logs(resolve_repo_path(plan.input_las, repo_root=repo_root))
    trajectory = None
    if plan.route == "deviated_local_depth":
        if not plan.well_trace_file:
            raise ValueError("Deviated depth route requires a trajectory file.")
        trajectory = WellTrajectory.from_petrel_trace(resolve_repo_path(plan.well_trace_file, repo_root=repo_root)).with_well_name(plan.well_name)
    target_start_depth, target_end_depth, top_name, bottom_name, target_top_depth, target_bottom_depth = _sample_target_depths(
        plan=plan, survey=survey, config=config, cache=horizon_cache, data_root=data_root, trajectory=trajectory
    )
    if trajectory is None:
        segments = _vertical_segments(standard, float(plan.kb_m))
    else:
        segments = _valid_segments(standard, trajectory)
    dt_s = float(config.get("wavelet_sampling_s", 0.002))
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("wavelet_sampling_s must be positive and finite.")
    md_axis = np.asarray(standard.logs.basis, dtype=np.float64)
    if trajectory is None:
        depth_coordinates = md_axis - float(plan.kb_m)
    else:
        depth_coordinates = np.interp(
            md_axis,
            trajectory.md_m,
            trajectory.tvdss_m,
            left=np.nan,
            right=np.nan,
        )
    mapping_logs, _filled_depth_gaps = fill_short_depth_gaps(
        standard,
        depth_coordinates,
        max_short_gap_s=float(config["reject"]["max_short_log_gap_s"]),
    )
    gap_report = dict(getattr(mapping_logs, "gap_report", {}))
    mapping_standard = StandardVpRhoLogs(
        logs=mapping_logs,
        observed_mask=standard.observed_mask,
        filled_gap_mask=_filled_depth_gaps,
    )
    if trajectory is None:
        segments = _vertical_segments(mapping_standard, float(plan.kb_m))
    else:
        segments = _valid_segments(mapping_standard, trajectory)
    if not segments:
        raise ValueError(f"No continuous finite Vp/Rho segment is available for {plan.well_name}.")
    if trajectory is None:
        raw_trace = _native_vertical_trace(survey, plan)
        seismic_min, seismic_max = float(raw_trace.basis[0]), float(raw_trace.basis[-1])
    else:
        survey_axis = survey.sample_axis("depth")
        seismic_min, seismic_max = float(survey_axis.values[0]), float(survey_axis.values[-1])
        raw_trace = None
    selected = _select_segment(
        segments,
        seismic_min=seismic_min,
        seismic_max=seismic_max,
        target_min=target_start_depth,
        target_max=target_end_depth,
        min_samples=2,
    )
    _start, _stop, md, tvdss, vp = selected
    table, mapping = _build_local_table(md, tvdss, vp)
    mapping_depth = mapping["tvdss_m"].to_numpy(dtype=np.float64)
    mapping_twt = mapping["twt_s"].to_numpy(dtype=np.float64)
    support_start_depth = max(float(target_start_depth), float(seismic_min), float(mapping_depth[0]))
    support_end_depth = min(float(target_end_depth), float(seismic_max), float(mapping_depth[-1]))
    if support_end_depth <= support_start_depth:
        raise ValueError("The target depth window has no common finite log and seismic support.")
    clip_reasons: list[str] = []
    if support_start_depth > float(target_start_depth) + 1e-9:
        clip_reasons.append("top_clipped_to_observed_support")
    if support_end_depth < float(target_end_depth) - 1e-9:
        clip_reasons.append("bottom_clipped_to_observed_support")
    target_start_s = float(np.interp(support_start_depth, mapping_depth, mapping_twt))
    target_end_s = float(np.interp(support_end_depth, mapping_depth, mapping_twt))
    target_top_s = float(np.interp(float(target_top_depth), mapping_depth, mapping_twt, left=np.nan, right=np.nan))
    target_bottom_s = float(np.interp(float(target_bottom_depth), mapping_depth, mapping_twt, left=np.nan, right=np.nan))
    selected_rho = np.asarray(mapping_standard.logs.Rho.values, dtype=np.float64)[_start:_stop]
    selected_logset = grid.LogSet(
        {
            "Vp": grid.Log(vp, md, "md", name="Vp", unit="m/s", allow_nan=False),
            "Rho": grid.Log(selected_rho, md, "md", name="Rho", unit="g/cm3", allow_nan=False),
        }
    )
    selected_logset.observed_mask = np.asarray(standard.observed_mask, dtype=bool)[_start:_stop].copy()
    selected_logset.filled_gap_mask = np.asarray(mapping_standard.filled_gap_mask, dtype=bool)[_start:_stop].copy()
    prepared = _windowed_prepared(
        table=table,
        logset_md=selected_logset,
        target_start_s=target_start_s,
        target_end_s=target_end_s,
        target_top_name=top_name,
        target_bottom_name=bottom_name,
        target_top_depth=target_top_depth,
        target_bottom_depth=target_bottom_depth,
        target_top_twt=target_top_s,
        target_bottom_twt=target_bottom_s,
        seismic_dt_s=dt_s,
        clip_reason=";".join(clip_reasons),
        min_samples=int(config["reject"]["min_tie_samples"]),
    )
    prepared = PreparedTieWindow(
        table=prepared.table,
        logset_md=prepared.logset_md,
        table_rows=prepared.table_rows,
        report={**prepared.report, **gap_report},
    )
    # Use the fixed source mapping to resample the native depth observation on
    # the wavelet grid.  The mapping is never replaced by the optimized TDT.
    if trajectory is None:
        native = raw_trace
        native_support = (np.asarray(native.basis) >= support_start_depth) & (np.asarray(native.basis) <= support_end_depth)
        if np.any(~np.isfinite(np.asarray(native.values)[native_support])):
            raise ValueError(f"Depth seismic trace contains non-finite samples inside tie support for {plan.well_name}.")
        twt_axis = np.arange(prepared.report["tie_window_start_s"], prepared.report["tie_window_end_s"] + 0.5 * dt_s, dt_s)
        z_axis = np.interp(twt_axis, mapping_twt, mapping_depth)
        values = np.interp(z_axis, native.basis, native.values)
        native = type(native)(native.values, native.basis, "tvdss", name=native.name)
    else:
        depth_axis = np.asarray(survey.sample_axis("depth").values, dtype=np.float64)
        keep = (depth_axis >= support_start_depth) & (depth_axis <= support_end_depth)
        z_axis = depth_axis[keep]
        if z_axis.size < 2:
            raise ValueError("Depth survey axis has fewer than two samples in the trajectory tie support.")
        if z_axis[0] < trajectory.tvdss_m[0] or z_axis[-1] > trajectory.tvdss_m[-1]:
            raise ValueError("Trajectory does not cover the native seismic depth support without extrapolation.")
        if np.any(np.diff(trajectory.tvdss_m) <= 0.0):
            raise ValueError("Trajectory TVDSS must be strictly increasing for native depth sampling.")
        md_axis = np.interp(z_axis, trajectory.tvdss_m, trajectory.md_m)
        positions = []
        for md_value, z_value in zip(md_axis, z_axis):
            position = trajectory.position_at_md(float(md_value))
            positions.append((float(position["x_m"]), float(position["y_m"]), float(z_value)))
        native_values = _sample_bilinear_depth(survey, positions=positions, trace_cache={})
        if np.any(~np.isfinite(native_values)):
            raise ValueError("Depth trajectory trace contains non-finite samples in tie support.")
        native = grid.Seismic(native_values, z_axis, "tvdss", name=f"Depth-derived seismic ({plan.well_name})")
        native_twt = np.interp(z_axis, mapping_depth, mapping_twt)
        twt_axis = np.arange(float(native_twt[0]), float(native_twt[-1]) + 0.5 * dt_s, dt_s)
        values = np.interp(twt_axis, native_twt, native_values)
    seismic = grid.Seismic(values, twt_axis, "twt", name=f"Depth-derived seismic ({plan.well_name})")
    prepared = PreparedTieWindow(
        table=prepared.table,
        logset_md=prepared.logset_md,
        table_rows=prepared.table_rows,
        report={**prepared.report, "seismic_sampling_dt_s": dt_s},
    )
    return DepthPrepared(
        prepared=prepared,
        seismic=seismic,
        source_depth_mapping=mapping,
        native_seismic=native,
        target_top_tvdss_m=float(target_top_depth),
        target_bottom_tvdss_m=float(target_bottom_depth),
        trajectory=trajectory,
        depth_coordinates=depth_coordinates,
    )

def write_depth_artifacts(
    *,
    output_dir: Path,
    plan: WellTiePlan,
    prepared: DepthPrepared,
    repo_root: Path,
) -> dict[str, str]:
    paths = output_paths(output_dir, plan.well_name, sample_domain="depth")
    paths["source_depth_mapping"].parent.mkdir(parents=True, exist_ok=True)
    prepared.source_depth_mapping.to_csv(paths["source_depth_mapping"], index=False)
    pd.DataFrame({"tvdss_m": prepared.native_seismic.basis, "seismic": prepared.native_seismic.values}).to_csv(paths["depth_seismic_trace"], index=False)
    return {
        "source_depth_mapping_file": repo_relative_path(paths["source_depth_mapping"], root=repo_root),
        "depth_seismic_trace_file": repo_relative_path(paths["depth_seismic_trace"], root=repo_root),
    }

def write_aligned_depth(
    *,
    optimized_table: grid.TimeDepthTable,
    source_depth_mapping: pd.DataFrame,
    path: Path,
) -> Path:
    if not optimized_table.is_md_domain:
        raise ValueError("Optimized depth TDT must be in MD domain.")
    source_twt = source_depth_mapping["twt_s"].to_numpy(dtype=np.float64)
    source_depth = source_depth_mapping["tvdss_m"].to_numpy(dtype=np.float64)
    optimized_twt = np.asarray(optimized_table.twt, dtype=np.float64)
    aligned = np.interp(optimized_twt, source_twt, source_depth, left=np.nan, right=np.nan)
    finite = np.isfinite(aligned) & np.isfinite(optimized_table.md)
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError("Optimized TDT has fewer than two samples inside the fixed source depth mapping.")
    aligned_md = np.asarray(optimized_table.md, dtype=np.float64)[finite]
    aligned = aligned[finite]
    if np.any(np.diff(aligned_md) <= 0.0) or np.any(np.diff(aligned) <= 0.0):
        raise ValueError("Finite aligned depth support must be strictly increasing in MD and TVDSS.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"md_m": aligned_md, "tvdss_m": aligned}).to_csv(path, index=False)
    return path


# Optimization and per-well export

def run_fixed_tie(
    *,
    logset_md: Any,
    table: Any,
    seismic: Any,
    wavelet_extractor: Any,
    modeler: Any,
    parameters: Mapping[str, Any],
) -> Any:
    """Run the single-parameter Wtie extraction used by benchmark adapters."""
    from wtie.optimize import autotie

    return autotie._intermediate_tie_v1(
        logset_md,
        None,
        table,
        seismic,
        wavelet_extractor,
        modeler,
        dict(parameters),
    )

def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except RuntimeError:
        pass
    fig.savefig(path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)

def _write_qc_figures(
    *,
    paths: dict[str, Path],
    prepared: PreparedTieWindow,
    outputs: Any,
    cropped_wavelet: Any,
    optimized_corr: float,
    optimized_nmae: float,
    synthetic_scale: float,
    target_window: TargetTieWindow | None,
) -> None:
    import matplotlib.pyplot as plt

    fig, _ = outputs.plot_optimization_objective(figsize=(6, 3))
    _save_figure(fig, paths["fig_objective"])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    rows = prepared.table_rows.sort_values("twt_s")
    if not rows.empty:
        ax.plot(rows["md_m"], rows["twt_s"] * 1000.0, lw=1.0, color="tab:blue", label="initial")
    ax.plot(outputs.table.md, outputs.table.twt * 1000.0, lw=1.0, color="tab:orange", label="optimized")
    ax.invert_yaxis()
    ax.set_xlabel("MD (m)")
    ax.set_ylabel("TWT (ms)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, paths["fig_tdt"])
    fig, _ = plot_well_waveform_qc(
        outputs.logset_twt,
        outputs.r,
        outputs.synth_seismic,
        outputs.seismic,
        outputs.xcorr,
        outputs.dxcorr,
        figsize=(12.0, 7.5),
        title=f"Auto tie | corr={optimized_corr:.3f}, nmae={optimized_nmae:.3f}, scale={synthetic_scale:.3g}",
        horizon_markers=([] if target_window is None else [(target_window.top_twt_s, target_window.top_name), (target_window.bottom_twt_s, target_window.bottom_name)]),
    )
    _save_figure(fig, paths["fig_tie"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    wavelet_values = np.asarray(outputs.wavelet.values, dtype=np.float64)
    cropped_values = np.asarray(cropped_wavelet.values, dtype=np.float64)
    axes[0].plot(outputs.wavelet.basis * 1000.0, wavelet_values / max(float(np.max(np.abs(wavelet_values))), 1e-12), label="raw")
    axes[0].plot(cropped_wavelet.basis * 1000.0, cropped_values / max(float(np.max(np.abs(cropped_values))), 1e-12), label="cropped")
    axes[0].axvline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("Time (ms)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)
    freq = np.fft.rfftfreq(cropped_wavelet.size, d=cropped_wavelet.sampling_rate)
    spec = np.abs(np.fft.rfft(cropped_values))
    axes[1].plot(freq, spec / max(float(spec.max()), 1e-12))
    axes[1].set_xlim(0, min(125, float(freq[-1])))
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True, alpha=0.25)
    _save_figure(fig, paths["fig_wavelet"])

def _target_window_from_report(report: Mapping[str, Any]) -> TargetTieWindow | None:
    required = {"target_top_name", "target_bottom_name", "target_top_twt_s", "target_bottom_twt_s", "target_window_start_s", "target_window_end_s"}
    if not required.issubset(report):
        return None
    if not np.isfinite(float(report["target_top_twt_s"])) or not np.isfinite(float(report["target_bottom_twt_s"])):
        return None
    return TargetTieWindow(
        top_name=str(report["target_top_name"]),
        bottom_name=str(report["target_bottom_name"]),
        top_twt_s=float(report["target_top_twt_s"]),
        bottom_twt_s=float(report["target_bottom_twt_s"]),
        start_s=float(report["target_window_start_s"]),
        end_s=float(report["target_window_end_s"]),
        margin_top_s=0.0,
        margin_bottom_s=0.0,
    )

def _export_time_checkshot(*, plan: WellTiePlan, table: Any, path: Path) -> Path:
    from cup.well.trajectory import export_vertical_tdt_to_petrel_checkshots

    if not table.is_md_domain:
        raise ValueError("Time-domain optimized TDT must be MD-domain for checkshot export.")
    if plan.surface_x is None or plan.surface_y is None or plan.kb_m is None:
        raise ValueError("Checkshot export requires surface XY and KB.")
    md = np.asarray(table.md, dtype=np.float64)
    tvdss = md - float(plan.kb_m)
    if np.any(~np.isfinite(tvdss)) or np.any(tvdss < 0.0):
        raise ValueError("Checkshot export requires non-negative finite TVDSS values.")
    petrel_table = type(table)(twt=np.asarray(table.twt, dtype=np.float64), tvdss=tvdss)
    return export_vertical_tdt_to_petrel_checkshots(
        path,
        petrel_table,
        well_name=plan.well_name,
        kb=float(plan.kb_m),
        x=float(plan.surface_x),
        y=float(plan.surface_y),
    )

def _write_optimized_trace_sample_plan(
    *,
    plan: WellTiePlan,
    survey: Any,
    trajectory: Any,
    table: Any,
    path: Path,
) -> Path:
    from cup.seismic.trace_sampling import build_bilinear_trace_sample_plan
    from cup.well.trajectory import sample_trajectory_on_twt

    if not table.is_md_domain:
        raise ValueError("Optimized trajectory trace plan requires an MD-domain TDT.")
    axis = survey.sample_axis("time")
    start, end = axis.window_indices(float(table.twt[0]), float(table.twt[-1]))
    twt = axis.values[start:end]
    if twt.size < 2:
        raise ValueError("Optimized trajectory trace plan has fewer than two seismic samples.")
    samples = sample_trajectory_on_twt(trajectory.with_well_name(plan.well_name), table, twt)
    trace_plan = build_bilinear_trace_sample_plan(samples, survey)
    if not trace_plan.rows["survey_position"].astype(str).eq("inside").all():
        raise ValueError("Optimized trajectory trace plan contains outside samples.")
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_plan.to_dataframe().to_csv(path, index=False)
    return path

def execute_well(
    *,
    plan: WellTiePlan,
    prepared: PreparedTieWindow,
    survey: Any,
    wavelet_extractor: Any,
    modeler: Any,
    config: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
    sample_domain: str,
    seismic_override: Any | None = None,
    extra_seed: dict[str, Any] | None = None,
    depth_prepared: DepthPrepared | None = None,
    input_las: Path | None = None,
    optimized_trace_trajectory: Any | None = None,
) -> tuple[WellTieResult, dict[str, Any]]:
    from wtie.optimize import autotie
    from wtie.utils.datasets.utils import InputSet

    paths = output_paths(output_dir, plan.well_name, sample_domain=sample_domain)
    paths["figure_dir"].mkdir(parents=True, exist_ok=True)
    ensure_output_dirs(output_dir, sample_domain=sample_domain)
    if input_las is None:
        input_las = Path(plan.input_las)
    logset_md = prepared.logset_md
    table = prepared.table
    validate_time_depth_table(table, logset_md.basis, min_overlap_samples=int(config["reject"]["min_tie_samples"]))
    start_s = float(prepared.report["tie_window_start_s"])
    end_s = float(prepared.report["tie_window_end_s"])
    seismic = seismic_override
    if seismic is None:
        if plan.surface_x is None or plan.surface_y is None:
            raise ValueError("Well tie requires surface XY when no seismic override is provided.")
        seismic = survey.read_trace_at_xy(plan.surface_x, plan.surface_y, sample_start=start_s, sample_end=end_s, domain="time")
    seismic = snap_seismic_sampling_if_close(seismic, float(wavelet_extractor.expected_sampling))
    pd.DataFrame({"twt_s": seismic.basis, "seismic": seismic.values}).to_csv(paths["seismic_trace"], index=False)
    write_time_depth_table_csv(table, paths["initial_tdt"], sources=prepared.table_rows["source"].astype(str).tolist())
    if depth_prepared is not None:
        depth_extra = write_depth_artifacts(output_dir=output_dir, plan=plan, prepared=depth_prepared, repo_root=repo_root)
    else:
        depth_extra = {}
    wavelet_scaling = {
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
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Wavelet has even number of samples; appending one zero sample\.")
        outputs = autotie.tie_v1(
            InputSet(logset_md=logset_md, seismic=seismic, table=table, wellpath=None),
            wavelet_extractor,
            modeler,
            wavelet_scaling,
            search_params=search_params,
            search_space=build_auto_tie_search_space(config["search_space"]),
            stretch_and_squeeze_params=None,
        )
    from cup.seismic.wavelet import crop_wavelet_center_energy_normalize

    cropped_wavelet, crop_info = crop_wavelet_center_energy_normalize(outputs.wavelet, float(config["target_crop_ms"]))
    pd.DataFrame({"time_s": cropped_wavelet.basis, "amplitude": cropped_wavelet.values}).to_csv(paths["wavelet"], index=False)
    write_time_depth_table_csv(outputs.table, paths["optimized_tdt"])
    seismic_norm, synthetic, optimized_corr, optimized_nmae, synthetic_scale = scaled_synthetic_metrics(modeler, cropped_wavelet, outputs.r, outputs.seismic)
    initial_corr, initial_nmae = compute_initial_metrics(logset_md, seismic, table, cropped_wavelet, modeler, float(wavelet_extractor.expected_sampling))
    pd.DataFrame({"twt_s": outputs.seismic.basis, "seismic_norm": seismic_norm, "reflectivity": outputs.r.values, "synthetic_cropped_scaled": synthetic, "residual": seismic_norm - synthetic}).to_csv(paths["synthetic_qc"], index=False)
    _write_qc_figures(
        paths=paths,
        prepared=prepared,
        outputs=outputs,
        cropped_wavelet=cropped_wavelet,
        optimized_corr=optimized_corr,
        optimized_nmae=optimized_nmae,
        synthetic_scale=synthetic_scale,
        target_window=_target_window_from_report(prepared.report),
    )
    best_params, _ = outputs.ax_client.get_best_parameters()
    best_shift_ms = float(best_params.get("table_t_shift", 0.0)) * 1000.0
    filtered_path = export_filtered_las(
        input_las=input_las,
        output_path=paths["filtered_las"],
        best_params=best_params,
        table=outputs.table,
        max_short_gap_s=float(config["reject"]["max_short_log_gap_s"]),
        well_name=plan.well_name,
        depth_coordinates=None if depth_prepared is None else depth_prepared.depth_coordinates,
    )
    checkshot_path = None
    if sample_domain == "time":
        checkshot_path = _export_time_checkshot(plan=plan, table=outputs.table, path=paths["petrel_checkshot"])
    optimized_trace_path = None
    if optimized_trace_trajectory is not None:
        optimized_trace_path = _write_optimized_trace_sample_plan(
            plan=plan,
            survey=survey,
            trajectory=optimized_trace_trajectory,
            table=outputs.table,
            path=paths["optimized_trace_sample_plan"],
        )
    aligned_path = None
    if depth_prepared is not None:
        aligned_path = write_aligned_depth(
            optimized_table=outputs.table,
            source_depth_mapping=depth_prepared.source_depth_mapping,
            path=paths["aligned_depth"],
        )
    result = WellTieResult(
        well_name=plan.well_name,
        route=plan.route,
        tie_status="success",
        initial_corr=initial_corr,
        optimized_corr=optimized_corr,
        optimized_nmae=optimized_nmae,
        best_table_shift_ms=best_shift_ms,
        source_tdt_file=str((extra_seed or {}).get("source_tdt_file") or ""),
        wavelet_file=repo_relative_path(paths["wavelet"], root=repo_root),
        optimized_tdt_file=repo_relative_path(paths["optimized_tdt"], root=repo_root),
        qc_figure_dir=repo_relative_path(paths["figure_dir"], root=repo_root),
        reasons="",
    )
    extra = dict(extra_seed or {})
    extra.update(
        {
            "initial_nmae": initial_nmae,
            "synthetic_scale": synthetic_scale,
            "crop_info": crop_info,
            "tie_window": dict(prepared.report),
            "best_parameters": best_params,
            "synthetic_qc_file": repo_relative_path(paths["synthetic_qc"], root=repo_root),
            "seismic_trace_file": repo_relative_path(paths["seismic_trace"], root=repo_root),
            "initial_tdt_file": repo_relative_path(paths["initial_tdt"], root=repo_root),
            "optimized_tdt_file": repo_relative_path(paths["optimized_tdt"], root=repo_root),
            "filtered_las_file": repo_relative_path(filtered_path, root=repo_root),
            "petrel_checkshot_file": None if checkshot_path is None else repo_relative_path(checkshot_path, root=repo_root),
            "optimized_trace_sample_plan_file": None if optimized_trace_path is None else repo_relative_path(optimized_trace_path, root=repo_root),
            "sample_domain": sample_domain,
            **depth_extra,
            "aligned_depth_file": None if aligned_path is None else repo_relative_path(aligned_path, root=repo_root),
        }
    )
    return result, extra


# Wtie resources and candidate inventory

def resolve_wtie_asset(
    value: str | Path | None,
    *,
    filename: str,
    data_root: Path,
    repo_root: Path,
) -> Path:
    """Resolve an explicit asset or the repository's public pretrained asset.

    An explicit value is resolved relative to the configured data root.
    """
    target = repo_root / "opendata" / "pretrained" / "wtie" / filename
    if value is None or not str(value).strip():
        return target.resolve()
    return resolve_relative_path(value, root=data_root)


def load_wavelet_extractor(model_path: Path, params_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"tutorial_model does not exist: {model_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"tutorial_params does not exist: {params_path}")
    from wtie.utils.datasets import tutorial as tutorial_mod

    with params_path.open("r", encoding="utf-8") as fp:
        training_parameters = yaml.load(fp, Loader=yaml.Loader)
    return tutorial_mod.load_wavelet_extractor(training_parameters, model_path)

def write_wavelet_inventory(results: list[WellTieResult], extras: dict[str, dict[str, Any]], *, repo_root: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.tie_status != "success" or not result.wavelet_file:
            continue
        path = resolve_repo_path(result.wavelet_file, repo_root=repo_root)
        df = pd.read_csv(path)
        time = df["time_s"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "source_well": result.well_name,
                "route": result.route,
                "wavelet_file": result.wavelet_file,
                "dt_s": float(np.median(np.diff(time))) if time.size > 1 else np.nan,
                "n_samples": int(time.size),
                "tie_corr": result.optimized_corr,
                "tie_nmae": result.optimized_nmae,
                "usable_as_candidate": True,
                "reasons": "",
            }
        )
    columns = ["source_well", "route", "wavelet_file", "dt_s", "n_samples", "tie_corr", "tie_nmae", "usable_as_candidate", "reasons"]
    out = pd.DataFrame.from_records(rows, columns=columns)
    out.to_csv(output_dir / "wavelet_inventory.csv", index=False)
    return out

__all__ = [
    "TieRoute", "build_tie_plan", "build_plans", "select_plans",
    "plans_dataframe", "results_dataframe", "build_auto_tie_search_space",
    "prepare_time_route", "prepare_depth_route", "execute_well", "run_fixed_tie",
    "resolve_wtie_asset", "load_wavelet_extractor", "write_wavelet_inventory",
]
