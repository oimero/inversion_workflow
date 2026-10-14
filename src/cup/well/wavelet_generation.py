"""Step 5 candidate artifacts, prepared well inputs and wavelet evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cup.utils.coerce import as_bool as _as_bool, optional_float as _optional_float
from cup.utils.statistics import radius_connected_components
from cup.well.inventory import normalize_well_name
from cup.well.las import load_standard_vp_rho_logs
from cup.well.tie import (
    WELL_AUTO_TIE_SCHEMA_VERSION, fill_short_joint_gaps, regular_tie_dt,
    resample_tie_seismic_to_dt, scaled_synthetic_metrics, tie_text, well_rows_by_name,
)
from cup.well.trajectory import crop_logset_md, load_workflow_time_depth_table_csv
from wtie.processing import grid


# Candidate and evaluation contracts

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
    max_short_log_gap_s: float
    surface_x: float | None = None
    surface_y: float | None = None
    tie_window_start_s: float | None = None
    tie_window_end_s: float | None = None
    sample_domain: str = "time"
    aligned_depth_file: Path | None = None
    source_depth_mapping_file: Path | None = None
    depth_seismic_trace_file: Path | None = None
    target_top_tvdss_m: float | None = None
    target_bottom_tvdss_m: float | None = None

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["input_las"] = self.input_las.as_posix()
        row["optimized_tdt_file"] = self.optimized_tdt_file.as_posix()
        row["seismic_trace_file"] = self.seismic_trace_file.as_posix()
        for key in ("aligned_depth_file", "source_depth_mapping_file", "depth_seismic_trace_file"):
            value = getattr(self, key)
            row[key] = None if value is None else value.as_posix()
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


# Fourth-step artifact loading and candidate selection

def _resolve_artifact_path(value: Any, index: TieArtifactIndex) -> Path | None:
    text = tie_text(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    if index.repo_root is not None:
        return index.repo_root / path
    return index.auto_tie_dir / path


def load_tie_artifacts(auto_tie_dir: str | Path, *, repo_root: str | Path | None = None) -> TieArtifactIndex:
    """Load the fourth-step auto-tie CSV artifacts for later wavelet evaluation."""
    auto_tie_path = Path(auto_tie_dir)
    root = Path(repo_root) if repo_root is not None else None
    plan_path = auto_tie_path / "well_tie_plan.csv"
    metrics_path = auto_tie_path / "well_tie_metrics.csv"
    wavelet_inventory_path = auto_tie_path / "wavelet_inventory.csv"
    summary_path = auto_tie_path / "run_summary.json"
    missing = [path for path in [plan_path, metrics_path, wavelet_inventory_path, summary_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"auto-tie artifact directory is missing files: {[str(path) for path in missing]}")
    import json

    with summary_path.open("r", encoding="utf-8") as fp:
        summary = json.load(fp)
    if summary.get("schema_version") != WELL_AUTO_TIE_SCHEMA_VERSION:
        raise ValueError(
            f"auto-tie run_summary schema_version must be {WELL_AUTO_TIE_SCHEMA_VERSION!r}, "
            f"got {summary.get('schema_version')!r}: {summary_path}"
        )
    sample_domain = str(summary.get("sample_domain") or "").strip().casefold()
    if sample_domain not in {"time", "depth"}:
        raise ValueError(f"auto-tie run_summary has invalid sample_domain: {summary.get('sample_domain')!r}")
    expected_unit = "s" if sample_domain == "time" else "m"
    if str(summary.get("sample_unit") or "").strip().casefold() != expected_unit:
        raise ValueError(f"auto-tie run_summary sample_unit must be {expected_unit!r} for {sample_domain}.")
    if sample_domain == "depth" and summary.get("depth_basis") != "tvdss":
        raise ValueError("Depth auto-tie run_summary must declare depth_basis='tvdss'.")
    if sample_domain == "time" and summary.get("depth_basis") is not None:
        raise ValueError("Time auto-tie run_summary must not declare depth_basis.")
    plan_df = pd.read_csv(plan_path)
    metrics_df = pd.read_csv(metrics_path)
    inventory_df = pd.read_csv(wavelet_inventory_path)
    if "sample_domain" in metrics_df.columns and not metrics_df.empty:
        domains = {str(value).strip().casefold() for value in metrics_df["sample_domain"].dropna()}
        if domains != {sample_domain}:
            raise ValueError(f"well_tie_metrics.csv sample_domain values {sorted(domains)} do not match run_summary {sample_domain!r}.")
    return TieArtifactIndex(
        auto_tie_dir=auto_tie_path,
        plan_df=plan_df,
        metrics_df=metrics_df,
        wavelet_inventory_df=inventory_df,
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
    include_keys = (
        {normalize_well_name(name) for name in include_source_wells}
        if include_source_wells is not None
        else None
    )
    exclude_keys = {normalize_well_name(name) for name in exclude_source_wells} if exclude_source_wells else set()
    candidates: list[WaveletCandidate] = []
    for _, row in df.iterrows():
        source_well = tie_text(row.get("source_well"))
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
            wavelet_file = Path(tie_text(row.get("wavelet_file")))
        if reasons:
            continue
        candidates.append(
            WaveletCandidate(
                source_well=source_well,
                route=tie_text(row.get("route")),
                wavelet_file=wavelet_file,
                dt_s=_optional_float(row.get("dt_s")),
                n_samples=_optional_int(row.get("n_samples")),
                tie_corr=tie_corr,
                tie_nmae=tie_nmae,
                usable_as_candidate=True,
                reasons=tie_text(row.get("reasons")),
            )
        )
    return candidates


def load_evaluation_wells(index: TieArtifactIndex, *, status: str = "success") -> list[TieEvaluationWell]:
    """Build evaluation-well objects from fourth-step plan and metrics artifacts.

    The fifth step evaluates against the filtered LAS exported by the fourth
    step, not the third-step preprocessed LAS listed in ``well_tie_plan``.
    """
    plan_by_key = well_rows_by_name(index.plan_df)
    metrics = index.metrics_df
    if metrics.empty or "well_name" not in metrics.columns:
        return []
    required = {
        "tie_status",
        "route",
        "sample_domain",
        "max_short_log_gap_s",
        "filtered_las_file",
        "optimized_tdt_file",
        "seismic_trace_file",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"well_tie_metrics.csv is missing columns: {missing}")
    wells: list[TieEvaluationWell] = []
    for _, row in metrics.iterrows():
        if tie_text(row.get("tie_status")) != status:
            continue
        well_name = tie_text(row.get("well_name"))
        if not well_name:
            continue
        key = normalize_well_name(well_name)
        plan = plan_by_key.get(key)
        input_las = _resolve_artifact_path(row.get("filtered_las_file"), index)
        optimized_tdt = _resolve_artifact_path(row.get("optimized_tdt_file"), index)
        seismic_trace = _resolve_artifact_path(row.get("seismic_trace_file"), index)
        paths = {
            "filtered_las_file": input_las,
            "optimized_tdt_file": optimized_tdt,
            "seismic_trace_file": seismic_trace,
        }
        missing_paths = [name for name, path in paths.items() if path is None or not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                f"Successful auto-tie well {well_name!r} is missing artifacts {missing_paths} in {index.auto_tie_dir}."
            )
        sample_domain = tie_text(row.get("sample_domain"))
        if sample_domain not in {"time", "depth"}:
            raise ValueError(f"Invalid sample_domain for {well_name!r}: {sample_domain!r}")
        max_short_gap_s = _optional_float(row.get("max_short_log_gap_s"))
        if max_short_gap_s is None or not np.isfinite(max_short_gap_s) or max_short_gap_s < 0.0:
            raise ValueError(f"Successful auto-tie well {well_name!r} has invalid max_short_log_gap_s.")
        aligned_depth = _resolve_artifact_path(row.get("aligned_depth_file"), index)
        source_mapping = _resolve_artifact_path(row.get("source_depth_mapping_file"), index)
        depth_seismic = _resolve_artifact_path(row.get("depth_seismic_trace_file"), index)
        if sample_domain == "depth":
            depth_paths = {
                "aligned_depth_file": aligned_depth,
                "source_depth_mapping_file": source_mapping,
                "depth_seismic_trace_file": depth_seismic,
            }
            missing_depth = [name for name, path in depth_paths.items() if path is None or not path.exists()]
            if missing_depth:
                raise FileNotFoundError(
                    f"Successful depth auto-tie well {well_name!r} is missing artifacts {missing_depth}."
                )
        wells.append(
            TieEvaluationWell(
                well_name=well_name,
                route=tie_text(row.get("route")),
                input_las=input_las,  # type: ignore[arg-type]
                optimized_tdt_file=optimized_tdt,  # type: ignore[arg-type]
                seismic_trace_file=seismic_trace,  # type: ignore[arg-type]
                surface_x=_optional_float(plan.get("surface_x")) if plan is not None else None,
                surface_y=_optional_float(plan.get("surface_y")) if plan is not None else None,
                tie_window_start_s=_optional_float(row.get("tie_window_start_s")),
                tie_window_end_s=_optional_float(row.get("tie_window_end_s")),
                sample_domain=sample_domain,
                aligned_depth_file=aligned_depth,
                source_depth_mapping_file=source_mapping,
                depth_seismic_trace_file=depth_seismic,
                target_top_tvdss_m=_optional_float(row.get("target_top_tvdss_m")),
                target_bottom_tvdss_m=_optional_float(row.get("target_bottom_tvdss_m")),
                max_short_log_gap_s=float(max_short_gap_s),
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


# Matched well inputs and candidate evaluation

@dataclass(frozen=True)
class PreparedWellEvaluation:
    """Matched time-grid inputs and optional fixed native-depth sampling."""

    seismic: grid.Seismic
    reflectivity: grid.Reflectivity
    impedance: grid.Log
    depth_sampling: DepthTieSampling | None = None


def prepare_well_for_evaluation(
    well_artifact: TieEvaluationWell,
    *,
    dt_s: float | None = None,
) -> PreparedWellEvaluation:
    """Load and match the well once for candidate and consensus evaluation.

    Pass *dt_s* to evaluate on a specific sampling interval, typically the wavelet
    sampling rate; the saved seismic trace is resampled to it first.  ``dt_s=None``
    keeps the trace on its own sampling interval.
    """
    from wtie.optimize import tie as tie_ops

    logset, table, seismic = load_continuous_tie_evaluation_inputs(well_artifact)
    if dt_s is None:
        target_dt_s = regular_tie_dt(seismic.basis, label="seismic trace")
    else:
        target_dt_s = float(dt_s)
        seismic = resample_tie_seismic_to_dt(seismic, target_dt_s)
    if not logset.is_md or not table.is_md_domain:
        raise ValueError("Wavelet evaluation requires MD logs and an MD time-depth table.")
    logset_twt = tie_ops.convert_logs_from_md_to_twt(logset, None, table, target_dt_s)
    reflectivity = tie_ops.compute_reflectivity(logset_twt)
    if hasattr(logset, "observed_mask"):
        reflectivity.source_observed_mask = np.asarray(logset.observed_mask, dtype=bool).copy()
    if hasattr(logset, "filled_gap_mask"):
        reflectivity.source_filled_gap_mask = np.asarray(logset.filled_gap_mask, dtype=bool).copy()
    seismic_match, reflectivity_match = tie_ops.match_seismic_and_reflectivity(seismic, reflectivity)
    impedance = grid.Log(
        np.interp(seismic_match.basis, logset_twt.AI.basis, logset_twt.AI.values, left=np.nan, right=np.nan),
        seismic_match.basis, "twt", name="AI", unit=logset_twt.AI.unit,
    )
    depth_sampling = (
        prepare_depth_tie_sampling(well_artifact, seismic_match.basis)
        if well_artifact.sample_domain == "depth" else None
    )
    return PreparedWellEvaluation(seismic_match, reflectivity_match, impedance, depth_sampling)


def load_continuous_tie_evaluation_inputs(
    well_artifact: TieEvaluationWell,
) -> tuple[Any, Any, Any]:
    """Load and crop fourth-step artifacts to one gap-free evaluation window."""
    standard = load_standard_vp_rho_logs(well_artifact.input_las)
    table = load_workflow_time_depth_table_csv(well_artifact.optimized_tdt_file)
    seismic = load_saved_seismic_trace_csv(well_artifact.seismic_trace_file)
    evaluation_logs, filled_gap_mask = fill_short_joint_gaps(
        standard,
        table,
        max_short_gap_s=float(well_artifact.max_short_log_gap_s),
    )

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
    logset = crop_logset_md(evaluation_logs, md_start, md_end, min_samples=2)
    source_md = np.asarray(standard.logs.basis, dtype=np.float64)
    cropped_mask = (source_md >= md_start) & (source_md <= md_end)
    logset.observed_mask = np.asarray(standard.observed_mask, dtype=bool)[cropped_mask]
    logset.filled_gap_mask = np.asarray(filled_gap_mask, dtype=bool)[cropped_mask]
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
    prepared: PreparedWellEvaluation | None = None,
) -> tuple[WaveletWellMetric, pd.DataFrame]:
    """Evaluate one wavelet on one fourth-step optimized tie artifact.

    Pass the result of ``prepare_well_for_evaluation`` to reuse matched inputs
    during candidate and consensus scoring. Depth scores use the original
    seismic depth samples with their unchanged source time mapping.
    """
    from wtie.processing import grid

    wavelet_time = np.asarray(wavelet_time_s, dtype=np.float64).reshape(-1)
    wavelet_values = np.asarray(wavelet_amplitude, dtype=np.float64).reshape(-1)
    if wavelet_time.shape != wavelet_values.shape:
        raise ValueError("wavelet_time_s and wavelet_amplitude must have matching shapes.")
    if wavelet_time.size < 3:
        raise ValueError("wavelet must contain at least three samples.")
    wavelet_dt_s = regular_tie_dt(wavelet_time, label="wavelet")

    if prepared is None:
        prepared = prepare_well_for_evaluation(well_artifact, dt_s=wavelet_dt_s)
    seismic_match = prepared.seismic
    reflectivity_match = prepared.reflectivity
    seismic_dt_s = regular_tie_dt(seismic_match.basis, label="seismic trace")
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
            "ai": prepared.impedance.values,
        }
    )
    if prepared.depth_sampling is not None:
        qc, depth_metrics = map_time_qc_to_depth(qc, prepared.depth_sampling)
        metric = replace(
            metric,
            corr=depth_metrics["corr"],
            nmae=depth_metrics["nmae"],
            scale=float(metric.scale) * depth_metrics["scale_multiplier"],
            n_eval_samples=depth_metrics["n_eval_samples"],
        )
    return metric, qc


# Evaluation on the unchanged native depth grid

@dataclass(frozen=True)
class DepthTieSampling:
    """Native observed samples and their unchanged local-time coordinates."""

    seismic: grid.Seismic
    time_at_depth: grid.Log


def prepare_depth_tie_sampling(well: TieEvaluationWell, time_axis: np.ndarray) -> DepthTieSampling:
    """Read Step-4 native seismic and its original depth-to-time mapping once."""
    mapping_path = well.source_depth_mapping_file
    seismic_path = well.depth_seismic_trace_file
    if mapping_path is None or seismic_path is None:
        raise ValueError(f"{well.well_name}: depth evaluation requires the original seismic and its time mapping.")
    mapping = pd.read_csv(mapping_path)
    observed = pd.read_csv(seismic_path)
    for frame, required, path in ((mapping, {"tvdss_m", "twt_s"}, mapping_path),
                                  (observed, {"tvdss_m", "seismic"}, seismic_path)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: missing columns {sorted(required-set(frame.columns))}.")
    table = grid.TimeDepthTable(twt=mapping.twt_s.to_numpy(), tvdss=mapping.tvdss_m.to_numpy())
    native = grid.Seismic(observed.seismic.to_numpy(), observed.tvdss_m.to_numpy(), "tvdss")
    time_axis = np.asarray(time_axis, dtype=np.float64)
    query = np.interp(native.basis, table.tvdss, table.twt, left=np.nan, right=np.nan)
    keep = np.isfinite(query) & (query >= time_axis[0]-1e-10) & (query <= time_axis[-1]+1e-10)
    if np.count_nonzero(keep) < 3:
        raise ValueError(f"{well.well_name}: too few native seismic depth samples cover the matched time interval.")
    depth_axis = native.basis[keep]
    values = native.values[keep]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{well.well_name}: native depth seismic contains non-finite samples inside the evaluation interval.")
    return DepthTieSampling(grid.Seismic(values, depth_axis, "tvdss", name=well.well_name),
                            grid.Log(query[keep], depth_axis, "tvdss", name="Original local TWT", unit="s"))


def map_time_qc_to_depth(qc: pd.DataFrame, sampling: DepthTieSampling) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Resample the synthetic, while keeping the observed depth coordinate fixed."""
    axis = qc.twt_s.to_numpy(dtype=np.float64)
    query = np.asarray(sampling.time_at_depth.values)
    if np.any(query < axis[0]-1e-10) or np.any(query > axis[-1]+1e-10):
        raise ValueError("Time synthetic does not cover the fixed depth evaluation grid.")
    values = np.asarray(sampling.seismic.values, dtype=np.float64)
    std = float(values.std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Native depth seismic has zero standard deviation.")
    observed = (values-values.mean())/std
    synthetic = np.interp(query, axis, qc.synthetic_scaled.to_numpy(dtype=np.float64))
    energy = float(np.dot(synthetic, synthetic))
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Native-depth synthetic has no finite positive energy.")
    scale = float(np.dot(observed, synthetic)/energy)
    synthetic *= scale
    data = {"tvdss_m":sampling.seismic.basis,"twt_s":query,"seismic_norm":observed,
            "synthetic_scaled":synthetic,"residual":observed-synthetic,
            "reflectivity":np.interp(query, axis, qc.reflectivity.to_numpy(dtype=np.float64))}
    if "ai" in qc:
        data["ai"] = np.interp(query, axis, qc.ai.to_numpy(dtype=np.float64))
    result = pd.DataFrame(data)
    metrics = {"corr":float(np.corrcoef(observed, synthetic)[0,1]),
               "nmae":float(np.abs(observed-synthetic).sum()/np.abs(observed).sum()),
               "scale_multiplier":scale,"n_eval_samples":len(result)}
    return result, metrics


__all__ = [
    "WaveletCandidate", "TieEvaluationWell", "WaveletWellMetric", "TieArtifactIndex",
    "PreparedWellEvaluation", "DepthTieSampling", "load_tie_artifacts",
    "load_candidate_wavelets", "load_evaluation_wells", "build_well_spatial_clusters",
    "prepare_well_for_evaluation", "evaluate_wavelet_on_well",
    "prepare_depth_tie_sampling", "map_time_qc_to_depth",
]
