"""Route planning and execution helpers for well auto-tie workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cup.well.assets import normalize_well_name


class TieRoute(str, Enum):
    VERTICAL_WITH_TDT = "vertical_with_tdt"
    VERTICAL_ANCHOR_FROM_TOPS = "vertical_anchor_from_tops"
    DEVIATED_WITH_TDT = "deviated_with_tdt"
    DEVIATED_ANCHOR_FROM_TOPS = "deviated_anchor_from_tops"
    REJECTED = "rejected"


@dataclass(frozen=True)
class WellTiePlan:
    well_name: str
    route: str
    route_status: str
    wellbore_class_initial: str
    wellbore_class_qc: str
    has_time_depth: bool
    has_well_trace: bool
    has_well_tops: bool
    usable_p_sonic: bool
    usable_density: bool
    input_las: str
    time_depth_file: str
    well_trace_file: str
    surface_x: float | None
    surface_y: float | None
    kb_m: float | None
    reasons: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WellTieResult:
    well_name: str
    route: str
    tie_status: str
    initial_corr: float | None = None
    optimized_corr: float | None = None
    optimized_nmae: float | None = None
    best_table_shift_ms: float | None = None
    wavelet_file: str = ""
    optimized_tdt_file: str = ""
    qc_figure_dir: str = ""
    reasons: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    return text in {"true", "1", "yes", "y"}


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.strip().casefold() in {"", "nan", "none", "null"} else text


def _build_lookup(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "well_name" not in df.columns:
        return {}
    return {normalize_well_name(row["well_name"]): row for _, row in df.iterrows()}


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
) -> list[WellTiePlan]:
    """Build one route-plan row per inventory well."""
    preprocess_by_key = _build_lookup(preprocess_df)
    trajectory_by_key = _build_lookup(trajectory_df if trajectory_df is not None else pd.DataFrame())
    enabled = {str(route) for route in enabled_routes}
    plans: list[WellTiePlan] = []

    for _, inv in inventory_df.iterrows():
        well_name = str(inv["well_name"])
        key = normalize_well_name(well_name)
        pre = preprocess_by_key.get(key)
        traj = trajectory_by_key.get(key)

        survey_position = _string_value(inv.get("survey_position"))
        in_allowed_survey = survey_position == "inside" or (allow_near_outside and survey_position == "near_outside")
        has_time_depth = _as_bool(inv.get("has_time_depth")) and key in time_depth_lookup
        has_well_trace = _as_bool(inv.get("has_well_trace")) and key in trace_lookup
        has_well_tops = _as_bool(inv.get("has_well_tops"))
        usable_p = (
            pre is not None
            and _as_bool(pre.get("usable_p_sonic"))
            and _string_value(pre.get("preprocess_status")) == "passed"
        )
        usable_rho = (
            pre is not None
            and _as_bool(pre.get("usable_density"))
            and _string_value(pre.get("preprocess_status")) == "passed"
        )
        input_las = _string_value(pre.get("preprocessed_las")) if pre is not None else ""
        wellbore_initial = _string_value(inv.get("wellbore_class")) or "unknown"
        wellbore_qc = _string_value(traj.get("wellbore_class_qc")) if traj is not None else wellbore_initial
        trajectory_status = _string_value(traj.get("trajectory_status")) if traj is not None else ""
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

        route = TieRoute.REJECTED.value
        route_status = "rejected"
        route_reasons = list(reasons)

        base_assets_ok = in_allowed_survey and usable_p and usable_rho and bool(input_las)
        if base_assets_ok and wellbore_qc == "vertical" and has_time_depth:
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
            if not has_time_depth:
                route_reasons.append("no_time_depth")
            if wellbore_qc == "deviated" and not has_well_trace:
                route_reasons.append("no_well_trace")
            if not has_well_tops:
                route_reasons.append("no_well_tops")

        if route != TieRoute.REJECTED.value:
            if route in enabled:
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
    """Build the auto-tie hyperparameter search space from a config mapping."""
    return [
        {
            "name": "logs_median_size",
            "type": "choice",
            "values": list(config["logs_median_size_values"]),
            "value_type": "int",
            "is_ordered": True,
            "sort_values": True,
        },
        {
            "name": "logs_median_threshold",
            "type": "range",
            "bounds": list(config["logs_median_threshold_bounds"]),
            "value_type": "float",
        },
        {"name": "logs_std", "type": "range", "bounds": list(config["logs_std_bounds"]), "value_type": "float"},
        {
            "name": "table_t_shift",
            "type": "range",
            "bounds": list(config["table_t_shift_bounds"]),
            "value_type": "float",
        },
    ]


def scaled_synthetic_metrics(
    modeler: Any,
    wavelet: Any,
    reflectivity: Any,
    seismic: Any,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Compute normalized synthetic-to-seismic match metrics.

    Returns (seismic_norm, synthetic, corr, nmae, scale).
    """
    synthetic_raw = np.asarray(modeler(wavelet.values, reflectivity.values), dtype=np.float64)
    seismic_values = np.asarray(seismic.values, dtype=np.float64)
    seismic_norm = seismic_values - float(np.nanmean(seismic_values))
    std = float(np.nanstd(seismic_norm))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Seismic trace has zero standard deviation.")
    seismic_norm = seismic_norm / std
    denom = max(float(np.dot(synthetic_raw, synthetic_raw)), 1e-12)
    scale = float(np.dot(seismic_norm, synthetic_raw) / denom)
    synthetic = scale * synthetic_raw
    corr = float(np.corrcoef(seismic_norm, synthetic)[0, 1]) if np.std(synthetic) > 0 else np.nan
    nmae = float(np.sum(np.abs(seismic_norm - synthetic)) / max(np.sum(np.abs(seismic_norm)), 1e-12))
    return seismic_norm, synthetic, corr, nmae, scale
