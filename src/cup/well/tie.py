"""Well-tie contracts, gap preparation, routing, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from cup.utils.coerce import as_bool as _as_bool
from cup.utils.coerce import optional_float as _optional_float
from cup.utils.io import sanitize_filename
from cup.utils.masks import true_runs
from cup.utils.statistics import radius_connected_components
from cup.well.inventory import normalize_well_name
from cup.well.las import StandardVpRhoLogs, load_vp_rho_logset_from_standard_las
from cup.well.trajectory import crop_logset_md, load_workflow_time_depth_table_csv
from wtie.processing import grid

WELL_AUTO_TIE_SCHEMA_VERSION = "well_auto_tie_v3"
DEPTH_VERTICAL_AUTO_TIE_SCHEMA_VERSION = "vertical_well_auto_tie_depth_v2"
DEPTH_WAVELET_BATCH_SCHEMA_VERSION = "wavelet_batch_synthetic_depth_v4"

@dataclass(frozen=True)
class ContinuousTieLogResult:
    """Joint-valid Vp/Rho logs cropped to one continuous tie interval."""

    logs: grid.LogSet
    start_twt_s: float
    end_twt_s: float
    qc: dict[str, float | int | bool]


def _gap_duration_s(
    md: np.ndarray,
    start: int,
    stop: int,
    table: grid.TimeDepthTable,
) -> float:
    left = max(0, start - 1)
    right = min(md.size - 1, stop)
    if not table.is_md_domain:
        raise ValueError("Gap-aware standard LAS processing requires an MD-domain TimeDepthTable.")
    if md[left] < table.md[0] or md[right] > table.md[-1]:
        return float("inf")
    return float(
        np.interp(md[right], table.md, table.twt)
        - np.interp(md[left], table.md, table.twt)
    )


def fill_short_joint_gaps(
    standard: StandardVpRhoLogs,
    table: grid.TimeDepthTable,
    *,
    max_short_gap_s: float,
) -> tuple[grid.LogSet, np.ndarray]:
    """Fill bounded joint-invalid runs whose mapped TWT duration is short."""
    if not table.is_md_domain:
        raise ValueError("Gap-aware standard LAS processing requires an MD-domain TimeDepthTable.")
    if not np.isfinite(max_short_gap_s) or max_short_gap_s < 0.0:
        raise ValueError("max_short_gap_s must be finite and non-negative.")
    md = np.asarray(standard.logs.basis, dtype=np.float64)
    vp = np.asarray(standard.logs.Vp.values, dtype=np.float64).copy()
    rho = np.asarray(standard.logs.Rho.values, dtype=np.float64).copy()
    joint_valid = np.isfinite(vp) & (vp > 0.0) & np.isfinite(rho) & (rho > 0.0)
    filled = np.zeros(md.shape, dtype=bool)
    for start, stop in true_runs(~joint_valid):
        if start == 0 or stop == md.size:
            continue
        duration = _gap_duration_s(md, start, stop, table)
        if duration > float(max_short_gap_s) + 1e-12:
            continue
        vp[start:stop] = np.interp(
            md[start:stop],
            md[[start - 1, stop]],
            vp[[start - 1, stop]],
        )
        rho[start:stop] = np.interp(
            md[start:stop],
            md[[start - 1, stop]],
            rho[[start - 1, stop]],
        )
        filled[start:stop] = True
    logs = grid.LogSet(
        {
            "Vp": grid.Log(vp, md, "md", name="Vp", unit="m/s", allow_nan=True),
            "Rho": grid.Log(rho, md, "md", name="Rho", unit="g/cm3", allow_nan=True),
        }
    )
    return logs, filled


def prepare_continuous_tie_logs(
    standard: StandardVpRhoLogs,
    table: grid.TimeDepthTable,
    *,
    window_start_s: float,
    window_end_s: float,
    max_short_gap_s: float,
    min_tie_samples: int,
    seismic_sample_interval_s: float,
) -> ContinuousTieLogResult:
    """Fill short gaps and crop logs to the longest joint-valid window run."""
    if not np.isfinite(window_start_s) or not np.isfinite(window_end_s):
        raise ValueError("Tie-window bounds must be finite.")
    if window_end_s <= window_start_s:
        raise ValueError("window_end_s must be greater than window_start_s.")
    if int(min_tie_samples) < 2:
        raise ValueError("min_tie_samples must be at least 2.")
    logs, filled = fill_short_joint_gaps(
        standard,
        table,
        max_short_gap_s=max_short_gap_s,
    )
    md = np.asarray(logs.basis, dtype=np.float64)
    vp = np.asarray(logs.Vp.values, dtype=np.float64)
    rho = np.asarray(logs.Rho.values, dtype=np.float64)
    twt = np.interp(md, table.md, table.twt, left=np.nan, right=np.nan)
    in_window = (
        np.isfinite(twt)
        & (twt >= float(window_start_s))
        & (twt <= float(window_end_s))
    )
    joint_valid = (
        in_window
        & np.isfinite(vp)
        & (vp > 0.0)
        & np.isfinite(rho)
        & (rho > 0.0)
    )
    runs = true_runs(joint_valid)
    if not runs:
        raise ValueError("No joint-valid Vp/Rho samples remain in the target tie window.")
    start, stop = max(runs, key=lambda item: item[1] - item[0])
    start_twt = float(twt[start])
    end_twt = float(twt[stop - 1])
    seismic_dt_s = float(seismic_sample_interval_s)
    if not np.isfinite(seismic_dt_s) or seismic_dt_s <= 0.0:
        raise ValueError("seismic_sample_interval_s must be positive and finite.")
    seismic_count = int(np.floor((end_twt - start_twt) / seismic_dt_s + 1e-9)) + 1
    if seismic_count < int(min_tie_samples):
        raise ValueError(
            "Longest continuous joint-valid tie segment is too short on the seismic time axis: "
            f"{seismic_count} < {int(min_tie_samples)} samples "
            f"(dt={seismic_dt_s:g}s, duration={end_twt - start_twt:g}s)."
        )

    original_joint = np.asarray(standard.observed_mask, dtype=bool)
    original_window_count = int(np.count_nonzero(in_window))
    observed_window_count = int(np.count_nonzero(in_window & original_joint))
    long_gap_runs = []
    for gap_start, gap_stop in true_runs(in_window & ~joint_valid):
        duration = _gap_duration_s(md, gap_start, gap_stop, table)
        if duration > float(max_short_gap_s) + 1e-12:
            long_gap_runs.append((gap_start, gap_stop, duration))

    cropped = grid.LogSet(
        {
            "Vp": grid.Log(
                vp[start:stop],
                md[start:stop],
                "md",
                name="Vp",
                unit="m/s",
                allow_nan=False,
            ),
            "Rho": grid.Log(
                rho[start:stop],
                md[start:stop],
                "md",
                name="Rho",
                unit="g/cm3",
                allow_nan=False,
            ),
        }
    )
    qc: dict[str, float | int | bool] = {
        "joint_observed_fraction": (
            float(observed_window_count / original_window_count)
            if original_window_count
            else 0.0
        ),
        "short_gap_filled_samples": int(np.count_nonzero(filled & in_window)),
        "long_gap_count": int(len(long_gap_runs)),
        "longest_long_gap_s": float(
            max((item[2] for item in long_gap_runs), default=0.0)
        ),
        "continuous_tie_window_start_s": start_twt,
        "continuous_tie_window_end_s": end_twt,
        "continuous_tie_sample_count": seismic_count,
        "continuous_tie_log_sample_count": int(stop - start),
        "tie_window_clipped_for_log_gap": bool(
            start_twt > float(window_start_s) + 1e-9
            or end_twt < float(window_end_s) - 1e-9
        ),
    }
    return ContinuousTieLogResult(
        logs=cropped,
        start_twt_s=start_twt,
        end_twt_s=end_twt,
        qc=qc,
    )

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


@dataclass(frozen=True)
class WaveletCandidate:
    source_well: str
    route: str
    wavelet_file: Path
    dt_s: float | None
    n_samples: int | None
    tie_corr: float | None
    tie_nmae: float | None
    usable_as_candidate: bool = True
    reasons: str = ""

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["wavelet_file"] = self.wavelet_file.as_posix()
        return row


@dataclass(frozen=True)
class TieEvaluationWell:
    well_name: str
    route: str
    input_las: Path
    optimized_tdt_file: Path
    seismic_trace_file: Path
    surface_x: float | None = None
    surface_y: float | None = None
    tie_window_start_s: float | None = None
    tie_window_end_s: float | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["input_las"] = self.input_las.as_posix()
        row["optimized_tdt_file"] = self.optimized_tdt_file.as_posix()
        row["seismic_trace_file"] = self.seismic_trace_file.as_posix()
        return row


@dataclass(frozen=True)
class WaveletWellMetric:
    """Metric for one candidate wavelet on one evaluation well.

    ``spatial_cluster_id`` is filled by the fifth-step script after joining the
    metrics with ``build_well_spatial_clusters()`` output.
    """

    candidate_wavelet: str
    source_well: str
    eval_well: str
    route: str
    corr: float | None
    nmae: float | None
    best_shift_ms: float | None = None
    scale: float | None = None
    n_eval_samples: int | None = None
    spatial_cluster_id: int | None = None
    is_source_well: bool = False
    status: str = "ok"
    reasons: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TieArtifactIndex:
    auto_tie_dir: Path
    plan_df: pd.DataFrame
    metrics_df: pd.DataFrame
    wavelet_inventory_df: pd.DataFrame
    repo_root: Path | None = None

    def candidate_wavelets(
        self,
        *,
        min_source_tie_corr: float | None = None,
        max_source_tie_nmae: float | None = None,
        include_source_wells: set[str] | None = None,
        exclude_source_wells: set[str] | None = None,
    ) -> list[WaveletCandidate]:
        return load_candidate_wavelets(
            self,
            min_source_tie_corr=min_source_tie_corr,
            max_source_tie_nmae=max_source_tie_nmae,
            include_source_wells=include_source_wells,
            exclude_source_wells=exclude_source_wells,
        )

    def evaluation_wells(self, *, status: str = "success") -> list[TieEvaluationWell]:
        return load_evaluation_wells(self, status=status)


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(round(number))


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.strip().casefold() in {"", "nan", "none", "null"} else text


def _resolve_artifact_path(value: Any, index: TieArtifactIndex) -> Path | None:
    text = _string_value(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidates: list[Path] = []
    if index.repo_root is not None:
        candidates.append(index.repo_root / path)
    candidates.extend([Path.cwd() / path, index.auto_tie_dir / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def _prefer_existing_or_fallback(path: Path | None, fallback: Path) -> Path:
    if path is not None and path.exists():
        return path
    return fallback


def _build_lookup(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "well_name" not in df.columns:
        return {}
    lookup: dict[str, pd.Series] = {}
    display_by_key: dict[str, str] = {}
    for _, row in df.iterrows():
        display = _string_value(row["well_name"])
        if not display:
            continue
        key = normalize_well_name(display)
        previous = display_by_key.get(key)
        if previous is not None:
            raise ValueError(
                f"Duplicate well_name after normalization in workflow CSV: {previous!r} and {display!r}."
            )
        lookup[key] = row
        display_by_key[key] = display
    return lookup


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
    """为清单中的每口井构建一行标定路由计划。"""
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


def load_tie_artifacts(auto_tie_dir: str | Path, *, repo_root: str | Path | None = None) -> TieArtifactIndex:
    """Load the fourth-step auto-tie CSV artifacts for later wavelet evaluation."""
    auto_tie_path = Path(auto_tie_dir)
    root = Path(repo_root) if repo_root is not None else None
    plan_path = auto_tie_path / "well_tie_plan.csv"
    metrics_path = auto_tie_path / "well_tie_metrics.csv"
    wavelet_inventory_path = auto_tie_path / "wavelet_inventory.csv"
    missing = [path for path in [plan_path, metrics_path, wavelet_inventory_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"auto-tie artifact directory is missing files: {[str(path) for path in missing]}")
    return TieArtifactIndex(
        auto_tie_dir=auto_tie_path,
        plan_df=pd.read_csv(plan_path),
        metrics_df=pd.read_csv(metrics_path),
        wavelet_inventory_df=pd.read_csv(wavelet_inventory_path),
        repo_root=root,
    )


def load_candidate_wavelets(
    index: TieArtifactIndex,
    *,
    min_source_tie_corr: float | None = None,
    max_source_tie_nmae: float | None = None,
    include_source_wells: set[str] | None = None,
    exclude_source_wells: set[str] | None = None,
) -> list[WaveletCandidate]:
    """Build candidate-wavelet objects from ``wavelet_inventory.csv``."""
    df = index.wavelet_inventory_df
    required = {"source_well", "route", "wavelet_file", "usable_as_candidate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"wavelet_inventory.csv is missing columns: {sorted(missing)}")
    include_keys = {normalize_well_name(name) for name in include_source_wells} if include_source_wells else None
    exclude_keys = {normalize_well_name(name) for name in exclude_source_wells} if exclude_source_wells else set()
    candidates: list[WaveletCandidate] = []
    for _, row in df.iterrows():
        source_well = _string_value(row.get("source_well"))
        source_key = normalize_well_name(source_well)
        reasons: list[str] = []
        usable = _as_bool(row.get("usable_as_candidate"))
        if not usable:
            reasons.append("inventory_not_usable")
        if include_keys is not None and source_key not in include_keys:
            reasons.append("not_in_include_source_wells")
        if source_key in exclude_keys:
            reasons.append("excluded_source_well")
        tie_corr = _optional_float(row.get("tie_corr"))
        tie_nmae = _optional_float(row.get("tie_nmae"))
        if min_source_tie_corr is not None and (tie_corr is None or tie_corr < float(min_source_tie_corr)):
            reasons.append("source_tie_corr_below_threshold")
        if max_source_tie_nmae is not None and (tie_nmae is None or tie_nmae > float(max_source_tie_nmae)):
            reasons.append("source_tie_nmae_above_threshold")
        wavelet_file = _resolve_artifact_path(row.get("wavelet_file"), index)
        if wavelet_file is None or not wavelet_file.exists():
            reasons.append("missing_wavelet_file")
            wavelet_file = Path(_string_value(row.get("wavelet_file")))
        if reasons:
            continue
        candidates.append(
            WaveletCandidate(
                source_well=source_well,
                route=_string_value(row.get("route")),
                wavelet_file=wavelet_file,
                dt_s=_optional_float(row.get("dt_s")),
                n_samples=_optional_int(row.get("n_samples")),
                tie_corr=tie_corr,
                tie_nmae=tie_nmae,
                usable_as_candidate=True,
                reasons=_string_value(row.get("reasons")),
            )
        )
    return candidates


def load_evaluation_wells(index: TieArtifactIndex, *, status: str = "success") -> list[TieEvaluationWell]:
    """Build evaluation-well objects from fourth-step plan and metrics artifacts.

    The fifth step evaluates against the filtered LAS exported by the fourth
    step, not the third-step preprocessed LAS listed in ``well_tie_plan``.
    """
    plan_by_key = _build_lookup(index.plan_df)
    metrics = index.metrics_df
    if metrics.empty or "well_name" not in metrics.columns:
        return []
    wells: list[TieEvaluationWell] = []
    for _, row in metrics.iterrows():
        if _string_value(row.get("tie_status")) != status:
            continue
        well_name = _string_value(row.get("well_name"))
        if not well_name:
            continue
        key = normalize_well_name(well_name)
        plan = plan_by_key.get(key)
        input_las = _resolve_artifact_path(row.get("filtered_las_file"), index)
        input_las = _prefer_existing_or_fallback(
            input_las,
            index.auto_tie_dir / "filtered_las" / f"filtered_logs_{sanitize_filename(well_name)}.las",
        )
        optimized_tdt = _resolve_artifact_path(row.get("optimized_tdt_file"), index)
        seismic_trace = _resolve_artifact_path(row.get("seismic_trace_file"), index)
        optimized_tdt = _prefer_existing_or_fallback(
            optimized_tdt,
            index.auto_tie_dir / "time_depth" / f"optimized_tdt_{sanitize_filename(well_name)}.csv",
        )
        seismic_trace = _prefer_existing_or_fallback(
            seismic_trace,
            index.auto_tie_dir / "seismic_trace" / f"seismic_trace_{sanitize_filename(well_name)}.csv",
        )
        if input_las is None or not input_las.exists() or not optimized_tdt.exists() or not seismic_trace.exists():
            continue
        wells.append(
            TieEvaluationWell(
                well_name=well_name,
                route=_string_value(row.get("route")),
                input_las=input_las,
                optimized_tdt_file=optimized_tdt,
                seismic_trace_file=seismic_trace,
                surface_x=_optional_float(plan.get("surface_x")) if plan is not None else None,
                surface_y=_optional_float(plan.get("surface_y")) if plan is not None else None,
                tie_window_start_s=_optional_float(row.get("tie_window_start_s")),
                tie_window_end_s=_optional_float(row.get("tie_window_end_s")),
            )
        )
    return wells


def build_well_spatial_clusters(
    evaluation_wells: Sequence[TieEvaluationWell],
    *,
    radius_m: float,
) -> pd.DataFrame:
    """Assign evaluation wells to radius-connected XY clusters."""
    rows = [
        {
            "well_name": well.well_name,
            "x_m": well.surface_x,
            "y_m": well.surface_y,
        }
        for well in evaluation_wells
    ]
    df = pd.DataFrame.from_records(rows, columns=["well_name", "x_m", "y_m"])
    if df.empty:
        df["spatial_cluster_id"] = pd.Series(dtype=np.int64)
        df["spatial_cluster_size"] = pd.Series(dtype=np.int64)
        return df
    labels = radius_connected_components(df[["x_m", "y_m"]].to_numpy(dtype=np.float64), radius_m)
    df["spatial_cluster_id"] = labels
    sizes = df.groupby("spatial_cluster_id")["well_name"].transform("count")
    df["spatial_cluster_size"] = sizes.astype(np.int64)
    return df


def build_reflectivity_for_tie_eval(logset: Any, table: Any, dt_s: float) -> Any:
    """Build TWT reflectivity on a fixed workflow time-depth table."""
    from wtie.optimize import tie as tie_ops

    if not getattr(logset, "is_md", False):
        raise ValueError("tie evaluation expects an MD-domain LogSet.")
    if not getattr(table, "is_md_domain", False):
        raise ValueError("tie evaluation expects an MD-domain TimeDepthTable.")
    logset_twt = tie_ops.convert_logs_from_md_to_twt(logset, None, table, float(dt_s))
    return tie_ops.compute_reflectivity(logset_twt)


def load_saved_seismic_trace_csv(path: str | Path) -> Any:
    """Load a fourth-step saved seismic trace CSV as a ``grid.Seismic``."""
    from wtie.processing import grid

    path = Path(path)
    df = pd.read_csv(path)
    required = {"twt_s", "seismic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"seismic trace CSV is missing columns {sorted(missing)}: {path}")
    twt = df["twt_s"].to_numpy(dtype=np.float64)
    values = df["seismic"].to_numpy(dtype=np.float64)
    finite = np.isfinite(twt) & np.isfinite(values)
    twt = twt[finite]
    values = values[finite]
    if twt.size < 2:
        raise ValueError(f"seismic trace CSV has fewer than 2 finite samples: {path}")
    order = np.argsort(twt)
    twt = twt[order]
    values = values[order]
    if np.any(np.diff(twt) <= 0.0):
        raise ValueError(f"seismic trace TWT must be strictly increasing: {path}")
    return grid.Seismic(values, twt, "twt", name=path.stem)


def _regular_dt_from_basis(basis: np.ndarray, *, label: str) -> float:
    values = np.asarray(basis, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"{label} basis has fewer than 2 samples.")
    diffs = np.diff(values)
    if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0.0):
        raise ValueError(f"{label} basis must be strictly increasing.")
    dt_s = float(np.median(diffs))
    if not np.allclose(diffs, dt_s, rtol=1e-4, atol=1e-9):
        raise ValueError(f"{label} basis is not regularly sampled.")
    return dt_s


def resample_tie_seismic_to_dt(seismic: Any, dt_s: float) -> Any:
    """把一个地震道重采样到目标采样间隔 ``dt_s``。

    第四步保存的工区地震道可能比子波提取器的采样粗（例如 4 ms 与 2 ms），
    而反射系数、子波与合成记录只定义在各自主轴上。匹配前把地震道搬到
    子波网格，后续的卷积与相关性才是在同一套采样上比较。
    """
    from wtie.optimize import tie as tie_ops

    target_dt_s = float(dt_s)
    if not np.isfinite(target_dt_s) or target_dt_s <= 0.0:
        raise ValueError("target dt_s must be finite and positive for seismic resampling.")
    resampled = tie_ops.resample_seismic(seismic, target_dt_s)
    actual_dt_s = _regular_dt_from_basis(resampled.basis, label="resampled seismic trace")
    if not np.isclose(actual_dt_s, target_dt_s, rtol=1e-5, atol=1e-9):
        raise ValueError(
            f"resampled seismic dt {actual_dt_s:g}s does not match requested dt {target_dt_s:g}s."
        )
    return resampled


def prepare_well_for_evaluation(
    well_artifact: TieEvaluationWell,
    *,
    dt_s: float | None = None,
) -> tuple[Any, Any]:
    """Load well artifacts once and return ``(seismic_match, reflectivity_match)``.

    The returned pair can be reused across wavelet evaluations for the same well,
    avoiding repeated LAS / TDT / seismic I/O and MD→TWT conversion.

    Pass *dt_s* to evaluate on a specific sampling interval, typically the wavelet
    sampling rate; the saved seismic trace is resampled to it first.  ``dt_s=None``
    keeps the trace on its own sampling interval.
    """
    from wtie.optimize import tie as tie_ops

    logset, table, seismic = load_continuous_tie_evaluation_inputs(well_artifact)
    if dt_s is None:
        target_dt_s = _regular_dt_from_basis(seismic.basis, label="seismic trace")
    else:
        target_dt_s = float(dt_s)
        seismic = resample_tie_seismic_to_dt(seismic, target_dt_s)
    reflectivity = build_reflectivity_for_tie_eval(logset, table, target_dt_s)
    return tie_ops.match_seismic_and_reflectivity(seismic, reflectivity)


def load_continuous_tie_evaluation_inputs(
    well_artifact: TieEvaluationWell,
) -> tuple[Any, Any, Any]:
    """Load and crop fourth-step artifacts to one gap-free evaluation window."""
    logset = load_vp_rho_logset_from_standard_las(well_artifact.input_las)
    table = load_workflow_time_depth_table_csv(well_artifact.optimized_tdt_file)
    seismic = load_saved_seismic_trace_csv(well_artifact.seismic_trace_file)

    start_s = max(float(seismic.basis[0]), float(table.twt[0]))
    end_s = min(float(seismic.basis[-1]), float(table.twt[-1]))
    if well_artifact.tie_window_start_s is not None:
        start_s = max(start_s, float(well_artifact.tie_window_start_s))
    if well_artifact.tie_window_end_s is not None:
        end_s = min(end_s, float(well_artifact.tie_window_end_s))
    if end_s <= start_s:
        raise ValueError("Fourth-step LAS, TDT, and seismic trace have no common tie window.")

    keep = (seismic.basis >= start_s) & (seismic.basis <= end_s)
    if int(np.count_nonzero(keep)) < 2:
        raise ValueError("Common fourth-step tie window contains fewer than two seismic samples.")
    seismic = type(seismic)(
        seismic.values[keep],
        seismic.basis[keep],
        "twt",
        name=seismic.name,
    )
    md_start = float(np.interp(float(seismic.basis[0]), table.twt, table.md))
    md_end = float(np.interp(float(seismic.basis[-1]), table.twt, table.md))
    logset = crop_logset_md(logset, md_start, md_end, min_samples=2)
    joint_valid = (
        np.isfinite(logset.Vp.values)
        & (logset.Vp.values > 0.0)
        & np.isfinite(logset.Rho.values)
        & (logset.Rho.values > 0.0)
    )
    if not np.all(joint_valid):
        raise ValueError(
            "Fourth-step filtered LAS contains a long DT/RHO gap inside the saved continuous tie window."
        )
    return logset, table, seismic


def evaluate_wavelet_on_well(
    *,
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    well_artifact: TieEvaluationWell,
    candidate_wavelet: str,
    source_well: str,
    modeler: Any,
    seismic_match: Any | None = None,
    reflectivity_match: Any | None = None,
) -> tuple[WaveletWellMetric, pd.DataFrame]:
    """Evaluate one wavelet on one fourth-step optimized tie artifact.

    Pass *seismic_match* and *reflectivity_match* (from
    ``prepare_well_for_evaluation``) to skip repeated I/O during batch
    or consensus evaluation.
    """
    from wtie.processing import grid

    wavelet_time = np.asarray(wavelet_time_s, dtype=np.float64).reshape(-1)
    wavelet_values = np.asarray(wavelet_amplitude, dtype=np.float64).reshape(-1)
    if wavelet_time.shape != wavelet_values.shape:
        raise ValueError("wavelet_time_s and wavelet_amplitude must have matching shapes.")
    if wavelet_time.size < 3:
        raise ValueError("wavelet must contain at least three samples.")
    wavelet_dt_s = _regular_dt_from_basis(wavelet_time, label="wavelet")

    if seismic_match is None or reflectivity_match is None:
        seismic_match, reflectivity_match = prepare_well_for_evaluation(well_artifact, dt_s=wavelet_dt_s)
    seismic_dt_s = _regular_dt_from_basis(seismic_match.basis, label="seismic trace")
    if not np.isclose(wavelet_dt_s, seismic_dt_s, rtol=1e-5, atol=1e-9):
        raise ValueError(f"wavelet dt {wavelet_dt_s:g}s does not match seismic trace dt {seismic_dt_s:g}s.")
    wavelet = grid.Wavelet(wavelet_values, wavelet_time, name=str(candidate_wavelet))
    seismic_norm, synthetic, corr, nmae, scale = scaled_synthetic_metrics(
        modeler,
        wavelet,
        reflectivity_match,
        seismic_match,
    )
    finite = np.isfinite(seismic_norm) & np.isfinite(synthetic)
    metric = WaveletWellMetric(
        candidate_wavelet=str(candidate_wavelet),
        source_well=str(source_well),
        eval_well=well_artifact.well_name,
        route=well_artifact.route,
        corr=corr,
        nmae=nmae,
        best_shift_ms=0.0,
        scale=scale,
        n_eval_samples=int(np.count_nonzero(finite)),
        is_source_well=normalize_well_name(source_well) == normalize_well_name(well_artifact.well_name),
        status="ok",
        reasons="",
    )
    qc = pd.DataFrame(
        {
            "twt_s": seismic_match.basis,
            "seismic_norm": seismic_norm,
            "reflectivity": reflectivity_match.values,
            "synthetic_scaled": synthetic,
            "residual": seismic_norm - synthetic,
        }
    )
    return metric, qc


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


def scaled_synthetic_metrics(
    modeler: Any,
    wavelet: Any,
    reflectivity: Any,
    seismic: Any,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """计算归一化合成记录与地震道的匹配指标。

    返回 ``(seismic_norm, synthetic, corr, nmae, scale)``。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Wavelet has even number of samples; appending one zero sample\.",
        )
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
