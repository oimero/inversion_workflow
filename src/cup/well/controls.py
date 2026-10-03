"""Canonical real-field well controls and their depth-domain QC."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cup.seismic.forward_inputs import load_forward_inputs
from cup.physics.numpy_backend import forward_depth, reflectivity_from_log_ai
from cup.physics.relations import AIVelocityRelation
from cup.seismic.geometry import SampleAxis, SurveyLineGeometry
from cup.seismic.target_zone import TargetZone
from cup.seismic.viz import plot_well_waveform_qc
from cup.config.artifacts import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    is_consumable_contract_status,
    require_contract_fingerprint,
    resolve_artifact_path,
)
from cup.utils.io import repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.utils.masks import true_runs as _finite_runs
from cup.well.inventory import build_file_lookup, normalize_well_name
from cup.well.scale import gaussian_smooth_finite_runs_numpy
from cup.well.tie import DEPTH_WAVELET_BATCH_SCHEMA_VERSION, WELL_AUTO_TIE_SCHEMA_VERSION
from cup.well.trajectory import WellTrajectory, load_workflow_time_depth_table_csv
from wtie.optimize.similarity import normalized_xcorr
from wtie.processing import grid


SCHEMA_VERSION = "real_field_well_controls_v7"
TIME_SOURCE_SCHEMA = WELL_AUTO_TIE_SCHEMA_VERSION
DEPTH_SOURCE_SCHEMA = DEPTH_WAVELET_BATCH_SCHEMA_VERSION
LINEAR_AI_UNIT = "m/s*g/cm3"

MANIFEST_COLUMNS = [
    "well_name",
    "status",
    "reason",
    "source_run_type",
    "source_run_path",
    "source_summary_path",
    "source_las_path",
    "source_transform_path",
    "wellbore_class",
    "sample_domain",
    "sample_unit",
    "depth_basis",
    "sampling_mode",
    "n_samples",
    "n_valid_samples",
    "n_observed_samples",
    "n_interpolated_samples",
    "n_native_samples",
    "n_valid_native_samples",
    "sample_min",
    "sample_max",
    "well_npz_path",
]


@dataclass(frozen=True)
class WellControl:
    """Aligned filtered well log sampled on the seismic model grid."""

    well_name: str
    sample_axis: SampleAxis
    model_grid_filtered_log_ai: grid.Log
    inline_by_sample: np.ndarray
    xline_by_sample: np.ndarray
    x_m_by_sample: np.ndarray
    y_m_by_sample: np.ndarray
    valid_mask: np.ndarray
    observed_valid_mask: np.ndarray
    wellbore_class: str
    sampling_mode: str
    source_run_type: str
    provenance: Mapping[str, Any]
    native: "NativeWellControl"

    def __post_init__(self) -> None:
        n = self.sample_axis.values.size
        values = np.asarray(self.model_grid_filtered_log_ai.values, dtype=np.float64)
        arrays = {
            "inline_by_sample": np.asarray(self.inline_by_sample, dtype=np.float64),
            "xline_by_sample": np.asarray(self.xline_by_sample, dtype=np.float64),
            "x_m_by_sample": np.asarray(self.x_m_by_sample, dtype=np.float64),
            "y_m_by_sample": np.asarray(self.y_m_by_sample, dtype=np.float64),
            "valid_mask": np.asarray(self.valid_mask, dtype=bool),
            "observed_valid_mask": np.asarray(self.observed_valid_mask, dtype=bool),
        }
        if values.shape != (n,) or not np.array_equal(self.model_grid_filtered_log_ai.basis, self.sample_axis.values):
            raise ValueError(f"{self.well_name}: model_grid_filtered_log_ai must be aligned to the canonical SampleAxis.")
        expected_basis = "twt" if self.sample_axis.domain == "time" else "tvdss"
        if not getattr(self.model_grid_filtered_log_ai, f"is_{expected_basis}"):
            raise ValueError(f"{self.well_name}: model_grid_filtered_log_ai basis is inconsistent with {self.sample_axis.domain}.")
        for name, array in arrays.items():
            if array.shape != (n,):
                raise ValueError(f"{self.well_name}: {name} must have shape ({n},).")
            object.__setattr__(self, name, array)
        valid = arrays["valid_mask"]
        observed = arrays["observed_valid_mask"]
        finite = np.isfinite(values)
        for name in ("inline_by_sample", "xline_by_sample", "x_m_by_sample", "y_m_by_sample"):
            finite &= np.isfinite(arrays[name])
        if not np.array_equal(valid, finite):
            raise ValueError(f"{self.well_name}: valid_mask must exactly describe finite logAI and positions.")
        if np.any(observed & ~valid):
            raise ValueError(f"{self.well_name}: observed_valid_mask must be a subset of valid_mask.")
        if not np.any(valid):
            raise ValueError(f"{self.well_name}: canonical control has no valid samples.")
        if self.native.well_name.casefold() != self.well_name.casefold():
            raise ValueError(f"{self.well_name}: native/model control well names differ.")
        if self.native.sample_domain != self.sample_axis.domain:
            raise ValueError(f"{self.well_name}: native/model control sample domains differ.")


@dataclass(frozen=True)
class NativeWellControl:
    """Aligned filtered well log on its native vertical sampling."""

    well_name: str
    coordinates: np.ndarray
    native_filtered_log_ai: np.ndarray
    valid_mask: np.ndarray
    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        values = np.asarray(self.native_filtered_log_ai, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if coordinates.ndim != 1 or coordinates.size < 2 or values.shape != coordinates.shape or valid.shape != coordinates.shape:
            raise ValueError(f"{self.well_name}: native arrays must be matching 1D arrays with at least two samples.")
        if np.any(~np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0.0):
            raise ValueError(f"{self.well_name}: native coordinates must be finite and strictly increasing.")
        if not np.array_equal(valid, np.isfinite(values)):
            raise ValueError(f"{self.well_name}: native valid_mask must exactly describe finite log-AI values.")
        expected_unit = "s" if self.sample_domain == "time" else "m" if self.sample_domain == "depth" else None
        if expected_unit is None or self.sample_unit != expected_unit:
            raise ValueError(f"{self.well_name}: native sample domain/unit is invalid.")
        if (self.sample_domain == "depth" and self.depth_basis != "tvdss") or (
            self.sample_domain == "time" and self.depth_basis is not None
        ):
            raise ValueError(f"{self.well_name}: native depth_basis is inconsistent with sample_domain.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "native_filtered_log_ai", values)
        object.__setattr__(self, "valid_mask", valid)


@dataclass(frozen=True)
class WellControlSet:
    sample_axis: SampleAxis
    controls: tuple[WellControl, ...]
    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    source_run_type: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_sample_axis(self.sample_axis)
        if not self.controls:
            raise ValueError("WellControlSet requires at least one valid well.")
        names = [control.well_name.casefold() for control in self.controls]
        if len(names) != len(set(names)):
            raise ValueError("WellControlSet well names must be unique (case-insensitive).")
        if (self.sample_axis.domain == "depth" and self.depth_basis != "tvdss") or (
            self.sample_axis.domain == "time" and self.depth_basis is not None
        ):
            raise ValueError("WellControlSet depth_basis is inconsistent with its SampleAxis domain.")
        for control in self.controls:
            if not np.array_equal(control.sample_axis.values, self.sample_axis.values):
                raise ValueError(f"{control.well_name}: SampleAxis differs from WellControlSet.")
            if control.sample_axis.domain != self.sample_domain or control.sample_axis.unit != self.sample_unit:
                raise ValueError(f"{control.well_name}: domain/unit differs from WellControlSet.")
            if control.source_run_type != self.source_run_type:
                raise ValueError(f"{control.well_name}: source_run_type differs from WellControlSet.")


def _validate_sample_axis(axis: SampleAxis) -> None:
    values = np.asarray(axis.values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or (values.size > 1 and np.any(np.diff(values) <= 0.0)):
        raise ValueError("Canonical SampleAxis must be finite and strictly increasing.")
    if values.size > 2 and not np.allclose(np.diff(values), np.diff(values)[0], rtol=1e-9, atol=1e-12):
        raise ValueError("Canonical SampleAxis must be regular.")
    expected_unit = "s" if axis.domain == "time" else "m" if axis.domain == "depth" else None
    if expected_unit is None or axis.unit != expected_unit:
        raise ValueError(f"Unsupported SampleAxis domain/unit: {axis.domain!r}/{axis.unit!r}.")


def _required_columns(frame: pd.DataFrame, columns: set[str], *, path: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _finite_number(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _load_summary(source_run_dir: Path, *, source_run_type: str, domain: str, depth_basis: str | None) -> tuple[Path, dict[str, Any]]:
    path = source_run_dir / "run_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    expected_schema = TIME_SOURCE_SCHEMA if source_run_type == "well_auto_tie" else DEPTH_SOURCE_SCHEMA
    if str(summary.get("schema_version") or "") != expected_schema:
        raise ValueError(f"{path} schema_version must be {expected_schema!r}; rebuild the upstream run.")
    expected_domain = "time" if source_run_type == "well_auto_tie" else "depth"
    if domain != expected_domain or str(summary.get("sample_domain") or "") != expected_domain:
        raise ValueError(f"Source adapter/domain mismatch: {source_run_type!r} cannot produce {domain!r} controls.")
    expected_unit = "s" if expected_domain == "time" else "m"
    if str(summary.get("sample_unit") or "") != expected_unit:
        raise ValueError(
            f"Source adapter unit mismatch: {source_run_type!r} requires {expected_unit!r}."
        )
    summary_basis = summary.get("depth_basis")
    if expected_domain == "depth" and (depth_basis != "tvdss" or summary_basis != "tvdss"):
        raise ValueError("Depth well controls require source and seismic depth_basis='tvdss'.")
    if expected_domain == "time" and summary_basis not in (None, ""):
        raise ValueError("Time source summary must not declare a depth_basis.")
    if not is_consumable_contract_status(summary.get("status")):
        raise ValueError(f"Source run is not consumable: status={summary.get('status')!r}.")
    return path, summary


def _read_ai_las(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import lasio

    las = lasio.read(str(path))
    frame = las.df()
    if "AI" not in frame.columns:
        raise ValueError(f"LAS lacks mandatory AI curve: {path}")
    curve = next((item for item in las.curves if str(item.mnemonic).strip().casefold() == "ai"), None)
    unit = "" if curve is None else str(curve.unit).strip().replace(" ", "")
    if unit.casefold() != LINEAR_AI_UNIT.casefold():
        raise ValueError(f"AI curve unit must be {LINEAR_AI_UNIT!r}, got {unit!r}: {path}")
    md = frame.index.to_numpy(dtype=np.float64)
    ai = frame["AI"].to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(md)) or np.any(np.diff(md) <= 0.0):
        raise ValueError(f"AI LAS MD axis must be finite and strictly increasing: {path}")
    if np.any(np.isfinite(ai) & (ai <= 0.0)):
        raise ValueError(f"AI contains non-positive finite values: {path}")
    valid = np.isfinite(ai) & (ai > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"AI LAS has fewer than two valid positive samples: {path}")
    filtered_log_ai = np.full(ai.shape, np.nan, dtype=np.float64)
    filtered_log_ai[valid] = np.log(ai[valid])
    return md, filtered_log_ai


def _interp_no_extrapolation(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    result = np.full(np.asarray(x).shape, np.nan, dtype=np.float64)
    inside = np.isfinite(x) & (x >= xp[0]) & (x <= xp[-1])
    result[inside] = np.interp(np.asarray(x)[inside], xp, fp)
    return result


def _interp_finite_runs(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Interpolate independently inside each contiguous finite value run."""

    x = np.asarray(x, dtype=np.float64)
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    if xp.ndim != 1 or fp.shape != xp.shape or np.any(~np.isfinite(xp)) or np.any(np.diff(xp) <= 0.0):
        raise ValueError("Finite-run interpolation requires a finite strictly increasing source axis.")
    output = np.full(x.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(fp)
    padded = np.r_[False, finite, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    for start, stop in changes:
        if stop - start == 1:
            exact = np.isclose(x, xp[start], rtol=0.0, atol=1e-10)
            output[exact] = fp[start]
            continue
        inside = np.isfinite(x) & (x >= xp[start]) & (x <= xp[stop - 1])
        output[inside] = np.interp(x[inside], xp[start:stop], fp[start:stop])
    return output


def _native_control(
    *,
    well_name: str,
    coordinates: np.ndarray,
    native_filtered_log_ai: np.ndarray,
    sample_domain: str,
    depth_basis: str | None,
    provenance: Mapping[str, Any],
) -> NativeWellControl:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(native_filtered_log_ai, dtype=np.float64)
    support = np.isfinite(coordinates)
    coordinates = coordinates[support]
    values = values[support]
    if coordinates.size < 2 or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError(f"{well_name}: aligned native coordinates are not strictly increasing.")
    return NativeWellControl(
        well_name=well_name,
        coordinates=coordinates,
        native_filtered_log_ai=values,
        valid_mask=np.isfinite(values),
        sample_domain=sample_domain,
        sample_unit="s" if sample_domain == "time" else "m",
        depth_basis=depth_basis,
        provenance=dict(provenance),
    )


def _control_from_arrays(
    *,
    well_name: str,
    sample_axis: SampleAxis,
    model_grid_filtered_log_ai: np.ndarray,
    inline: np.ndarray,
    xline: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    observed_valid_mask: np.ndarray,
    wellbore_class: str,
    sampling_mode: str,
    source_run_type: str,
    provenance: Mapping[str, Any],
    native: NativeWellControl,
) -> WellControl:
    arrays = [np.asarray(value, dtype=np.float64).copy() for value in (model_grid_filtered_log_ai, inline, xline, x_m, y_m)]
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    observed = np.asarray(observed_valid_mask, dtype=bool)
    if observed.shape != valid.shape:
        raise ValueError(f"{well_name}: observed_valid_mask shape differs from the model axis.")
    observed &= valid
    for value in arrays:
        value[~valid] = np.nan
    basis_type = "twt" if sample_axis.domain == "time" else "tvdss"
    log = grid.Log(arrays[0], sample_axis.values.copy(), basis_type, name="model_grid_filtered_log_ai", unit="ln(m/s*g/cm3)")
    return WellControl(
        well_name=well_name,
        sample_axis=sample_axis,
        model_grid_filtered_log_ai=log,
        inline_by_sample=arrays[1],
        xline_by_sample=arrays[2],
        x_m_by_sample=arrays[3],
        y_m_by_sample=arrays[4],
        valid_mask=valid,
        observed_valid_mask=observed,
        wellbore_class=wellbore_class,
        sampling_mode=sampling_mode,
        source_run_type=source_run_type,
        provenance=dict(provenance),
        native=native,
    )


def _validate_control_geometry(control: WellControl, line_geometry: SurveyLineGeometry) -> None:
    """Ensure physical XY and floating line coordinates describe the same survey positions."""

    for index in np.flatnonzero(control.valid_mask):
        inline, xline = line_geometry.coord_to_line(
            float(control.x_m_by_sample[index]), float(control.y_m_by_sample[index])
        )
        if not (
            np.isclose(inline, control.inline_by_sample[index], rtol=0.0, atol=1e-6)
            and np.isclose(xline, control.xline_by_sample[index], rtol=0.0, atol=1e-6)
        ):
            raise ValueError(
                f"{control.well_name}: sample {index} XY and inline/xline coordinates disagree."
            )


def _time_control(
    *,
    source_row: Mapping[str, Any],
    inventory_row: Mapping[str, Any],
    sample_axis: SampleAxis,
    source_run_dir: Path,
    repo_root: Path,
) -> WellControl:
    well_name = str(source_row["well_name"]).strip()
    tdt_path = resolve_artifact_path(source_row.get("optimized_tdt_file"), root=repo_root, run_dir=source_run_dir)
    if tdt_path is None or not tdt_path.is_file():
        raise FileNotFoundError(f"{well_name}: optimized TDT is missing.")
    table = load_workflow_time_depth_table_csv(tdt_path)
    if not table.is_md_domain:
        raise ValueError(f"{well_name}: optimized TDT must use MD domain.")
    twt = np.asarray(table.twt, dtype=np.float64)
    table_md = np.asarray(table.md, dtype=np.float64)
    plan_path = resolve_artifact_path(
        source_row.get("optimized_trace_sample_plan_file"), root=repo_root, run_dir=source_run_dir
    )
    wellbore_class = str(inventory_row.get("wellbore_class") or "unknown").strip().casefold()
    if wellbore_class == "deviated":
        if plan_path is None or not plan_path.is_file():
            raise FileNotFoundError(f"{well_name}: deviated well lacks optimized trace sample plan.")
        plan = pd.read_csv(plan_path)
        _required_columns(plan, {"twt_s", "inline_float", "xline_float", "x_m", "y_m", "survey_position"}, path=plan_path)
        plan_twt = pd.to_numeric(plan["twt_s"], errors="coerce").to_numpy(dtype=np.float64)
        if plan_twt.size < 2 or np.any(~np.isfinite(plan_twt)) or np.any(np.diff(plan_twt) <= 0.0):
            raise ValueError(f"{well_name}: optimized trace plan TWT must be strictly increasing.")
        inside_plan = plan["survey_position"].astype(str).str.casefold().eq("inside").to_numpy(dtype=bool)
        positions = []
        for name in ("inline_float", "xline_float", "x_m", "y_m"):
            values = pd.to_numeric(plan[name], errors="coerce").to_numpy(dtype=np.float64)
            values[~inside_plan] = np.nan
            positions.append(_interp_finite_runs(sample_axis.values, plan_twt, values))
        sampling_mode = "optimized_trace_plan"
        transform_path = plan_path
    elif wellbore_class == "vertical":
        inline = _finite_number(inventory_row.get("inline_float"), label=f"{well_name}.inline_float")
        xline = _finite_number(inventory_row.get("xline_float"), label=f"{well_name}.xline_float")
        x_m = _finite_number(inventory_row.get("surface_x"), label=f"{well_name}.surface_x")
        y_m = _finite_number(inventory_row.get("surface_y"), label=f"{well_name}.surface_y")
        positions = [np.full(sample_axis.values.shape, value) for value in (inline, xline, x_m, y_m)]
        sampling_mode = "vertical_inventory_position"
        transform_path = tdt_path
    else:
        raise ValueError(f"{well_name}: unsupported/unknown wellbore_class={wellbore_class!r}.")
    native_las_path = resolve_artifact_path(
        source_row.get("filtered_las_file"), root=repo_root, run_dir=source_run_dir
    )
    if native_las_path is None or not native_las_path.is_file():
        raise FileNotFoundError(f"{well_name}: filtered LAS is missing.")
    native_md, native_filtered_log_ai = _read_ai_las(native_las_path)
    native = _native_control(
        well_name=well_name,
        coordinates=_interp_no_extrapolation(native_md, table_md, twt),
        native_filtered_log_ai=native_filtered_log_ai,
        sample_domain="time",
        depth_basis=None,
        provenance={
            "source_las_path": str(native_las_path),
            "alignment_transform_path": str(tdt_path),
            "source_las_role": "filtered",
            "gap_policy": "upstream_filtered_only",
            "source_vertical_coordinate": "md_m",
            "aligned_vertical_coordinate": "twt_s",
        },
    )
    model_grid_filtered_log_ai = _interp_finite_runs(
        sample_axis.values,
        native.coordinates,
        native.native_filtered_log_ai,
    )
    return _control_from_arrays(
        well_name=well_name,
        sample_axis=sample_axis,
        model_grid_filtered_log_ai=model_grid_filtered_log_ai,
        inline=positions[0],
        xline=positions[1],
        x_m=positions[2],
        y_m=positions[3],
        observed_valid_mask=np.isfinite(model_grid_filtered_log_ai),
        wellbore_class=wellbore_class,
        sampling_mode=sampling_mode,
        source_run_type="well_auto_tie",
        provenance={
            "source_las_path": str(native_las_path),
            "native_source_las_path": str(native_las_path),
            "source_las_role": "filtered",
            "gap_policy": "upstream_filtered_only",
            "source_transform_path": str(transform_path),
            "optimized_tdt_path": str(tdt_path),
        },
        native=native,
    )


def _depth_control(
    *,
    source_row: Mapping[str, Any],
    inventory_row: Mapping[str, Any],
    sample_axis: SampleAxis,
    line_geometry: SurveyLineGeometry,
    source_run_dir: Path,
    repo_root: Path,
    trace_lookup: Mapping[str, Path],
) -> WellControl:
    well_name = str(source_row["well_name"]).strip()
    native_las_path = resolve_artifact_path(
        source_row.get("shifted_filtered_las_path"), root=repo_root, run_dir=source_run_dir
    )
    if native_las_path is None or not native_las_path.is_file():
        raise FileNotFoundError(f"{well_name}: shifted filtered LAS is missing.")
    native_md, native_filtered_log_ai = _read_ai_las(native_las_path)
    wellbore_class = str(inventory_row.get("wellbore_class") or "unknown").strip().casefold()
    trace_path = trace_lookup.get(normalize_well_name(well_name))
    if wellbore_class == "deviated":
        if trace_path is None:
            raise FileNotFoundError(f"{well_name}: deviated well lacks a project trajectory file.")
        trajectory = WellTrajectory.from_petrel_trace(trace_path).with_well_name(well_name)
        tvdss_at_md = _interp_no_extrapolation(native_md, trajectory.md_m, trajectory.tvdss_m)
        valid = np.isfinite(tvdss_at_md)
        if np.count_nonzero(valid) < 2 or np.any(np.diff(tvdss_at_md[valid]) <= 0.0):
            raise ValueError(f"{well_name}: trajectory TVDSS support for shifted LAS is not strictly increasing.")
        md_at_sample = _interp_no_extrapolation(
            sample_axis.values, tvdss_at_md[valid], native_md[valid]
        )
        x_m = _interp_no_extrapolation(md_at_sample, trajectory.md_m, trajectory.x_m)
        y_m = _interp_no_extrapolation(md_at_sample, trajectory.md_m, trajectory.y_m)
        inline = np.full(sample_axis.values.shape, np.nan)
        xline = np.full(sample_axis.values.shape, np.nan)
        for index in np.flatnonzero(np.isfinite(x_m) & np.isfinite(y_m)):
            try:
                inline[index], xline[index] = line_geometry.coord_to_line(x_m[index], y_m[index])
            except ValueError:
                continue
        sampling_mode = "trajectory_tvdss"
        transform_path = trace_path
        transform_metadata = {"tvdss_source": "cup.well.trajectory.WellTrajectory"}
        native_tvdss = _interp_no_extrapolation(native_md, trajectory.md_m, trajectory.tvdss_m)
    elif wellbore_class == "vertical":
        kb_m = _finite_number(inventory_row.get("kb_m"), label=f"{well_name}.kb_m")
        inline_value = _finite_number(inventory_row.get("inline_float"), label=f"{well_name}.inline_float")
        xline_value = _finite_number(inventory_row.get("xline_float"), label=f"{well_name}.xline_float")
        x_value = _finite_number(inventory_row.get("surface_x"), label=f"{well_name}.surface_x")
        y_value = _finite_number(inventory_row.get("surface_y"), label=f"{well_name}.surface_y")
        inline, xline, x_m, y_m = [
            np.full(sample_axis.values.shape, value)
            for value in (inline_value, xline_value, x_value, y_value)
        ]
        sampling_mode = "vertical_md_minus_kb"
        transform_path = None
        transform_metadata = {"kb_m": kb_m, "tvdss_formula": "shifted_md_m-kb_m"}
        native_tvdss = native_md - kb_m
    else:
        raise ValueError(f"{well_name}: unsupported/unknown wellbore_class={wellbore_class!r}.")
    native = _native_control(
        well_name=well_name,
        coordinates=native_tvdss,
        native_filtered_log_ai=native_filtered_log_ai,
        sample_domain="depth",
        depth_basis="tvdss",
        provenance={
            "source_las_path": str(native_las_path),
            "alignment_transform_path": "" if transform_path is None else str(transform_path),
            "source_las_role": "filtered",
            "gap_policy": "upstream_filtered_only",
            "source_vertical_coordinate": "shifted_md_m",
            "aligned_vertical_coordinate": "tvdss_m",
            **transform_metadata,
        },
    )
    model_grid_filtered_log_ai = _interp_finite_runs(
        sample_axis.values,
        native.coordinates,
        native.native_filtered_log_ai,
    )
    return _control_from_arrays(
        well_name=well_name,
        sample_axis=sample_axis,
        model_grid_filtered_log_ai=model_grid_filtered_log_ai,
        inline=inline,
        xline=xline,
        x_m=x_m,
        y_m=y_m,
        observed_valid_mask=np.isfinite(model_grid_filtered_log_ai),
        wellbore_class=wellbore_class,
        sampling_mode=sampling_mode,
        source_run_type="wavelet_batch_synthetic_depth",
        provenance={
            "source_las_path": str(native_las_path),
            "native_source_las_path": str(native_las_path),
            "source_las_role": "filtered",
            "gap_policy": "upstream_filtered_only",
            "source_transform_path": "" if transform_path is None else str(transform_path),
            **transform_metadata,
        },
        native=native,
    )


def build_well_control_set(
    *,
    config: Mapping[str, Any],
    sample_axis: SampleAxis,
    line_geometry: SurveyLineGeometry,
    domain: str,
    depth_basis: str | None,
    repo_root: Path,
    data_root: Path,
    seismic_path: Path,
) -> tuple[WellControlSet, pd.DataFrame]:
    """Build canonical controls and a manifest frame without reading any LFM."""

    allowed_config = {
        "source_run_type",
        "source_run_dir",
        "well_inventory_file",
        "well_trace_dir",
    }
    if set(config) != allowed_config:
        raise ValueError(f"real_field_well_controls must contain exactly {sorted(allowed_config)}.")
    _validate_sample_axis(sample_axis)
    seismic_path = Path(seismic_path)
    if not seismic_path.is_file():
        raise FileNotFoundError(seismic_path)
    source_run_type = str(config.get("source_run_type") or "").strip()
    if source_run_type not in {"well_auto_tie", "wavelet_batch_synthetic_depth"}:
        raise ValueError("real_field_well_controls.source_run_type must be explicit and supported.")
    source_run_dir = resolve_relative_path(str(config.get("source_run_dir") or ""), root=repo_root)
    if not source_run_dir.is_dir():
        raise FileNotFoundError(source_run_dir)
    summary_path, source_summary = _load_summary(
        source_run_dir, source_run_type=source_run_type, domain=domain, depth_basis=depth_basis
    )
    inventory_path = resolve_relative_path(str(config.get("well_inventory_file") or ""), root=repo_root)
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    inventory = pd.read_csv(inventory_path)
    _required_columns(
        inventory,
        {"well_name", "wellbore_class", "surface_x", "surface_y", "inline_float", "xline_float", "kb_m"},
        path=inventory_path,
    )
    inventory_names = [normalize_well_name(value) for value in inventory["well_name"]]
    invalid_names = {"", "nan", "none", "null", "<na>"}
    if any(name.casefold() in invalid_names for name in inventory_names) or len(inventory_names) != len(set(inventory_names)):
        raise ValueError(f"Well inventory names must be non-empty and unique after normalization: {inventory_path}")
    inventory_index = {
        normalize_well_name(row["well_name"]): row for _, row in inventory.iterrows()
    }

    metrics_name = "well_tie_metrics.csv" if source_run_type == "well_auto_tie" else "wavelet_batch_metrics.csv"
    metrics_path = source_run_dir / metrics_name
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    recorded_metrics = (
        dict(source_summary.get("paths") or {}).get("well_tie_metrics")
        if source_run_type == "well_auto_tie"
        else dict(source_summary.get("outputs") or {}).get("metrics_csv")
    )
    recorded_metrics_path = resolve_artifact_path(recorded_metrics, root=repo_root, run_dir=source_run_dir)
    if recorded_metrics_path is None or recorded_metrics_path.resolve() != metrics_path.resolve():
        raise ValueError("Source run summary metrics path does not match the selected source run.")
    metrics = pd.read_csv(metrics_path)
    if source_run_type == "well_auto_tie":
        _required_columns(
            metrics,
            {
                "well_name",
                "tie_status",
                "filtered_las_file",
                "optimized_tdt_file",
                "optimized_trace_sample_plan_file",
            },
            path=metrics_path,
        )
        success = metrics["tie_status"].astype(str).str.casefold().eq("success")
    else:
        _required_columns(
            metrics,
            {
                "well_name",
                "status",
                "shifted_filtered_las_path",
            },
            path=metrics_path,
        )
        success = metrics["status"].astype(str).str.casefold().eq("ok")
    metric_names = [normalize_well_name(value) for value in metrics["well_name"]]
    if any(name.casefold() in invalid_names for name in metric_names) or len(metric_names) != len(set(metric_names)):
        raise ValueError(f"Source metrics well names must be non-empty and unique after normalization: {metrics_path}")

    trace_dir = resolve_relative_path(str(config.get("well_trace_dir") or ""), root=data_root)
    trace_lookup = build_file_lookup(trace_dir.iterdir(), asset_label=str(trace_dir)) if trace_dir.is_dir() else {}
    controls: list[WellControl] = []
    rows: list[dict[str, Any]] = []
    source_contract_fingerprint = require_contract_fingerprint(
        source_summary, label=f"source run {summary_path}"
    )
    inventory_summary_path = inventory_path.parent / "run_summary.json"
    if not inventory_summary_path.is_file():
        raise FileNotFoundError(inventory_summary_path)
    with inventory_summary_path.open("r", encoding="utf-8") as handle:
        inventory_summary = json.load(handle)
    inventory_contract_fingerprint = require_contract_fingerprint(
        inventory_summary, label=f"well inventory run {inventory_summary_path}"
    )
    for (_, source_row), source_ok in zip(metrics.iterrows(), success.to_numpy(dtype=bool)):
        well_name = str(source_row["well_name"]).strip()
        base = {
            "well_name": well_name,
            "status": "failed",
            "reason": "source_status_not_success" if not source_ok else "",
            "source_run_type": source_run_type,
            "source_run_path": repo_relative_path(source_run_dir, root=repo_root),
            "source_summary_path": repo_relative_path(summary_path, root=repo_root),
            "source_las_path": "",
            "source_transform_path": "",
            "wellbore_class": "",
            "sample_domain": sample_axis.domain,
            "sample_unit": sample_axis.unit,
            "depth_basis": depth_basis or "",
            "sampling_mode": "",
            "n_samples": int(sample_axis.values.size),
            "n_valid_samples": 0,
            "n_observed_samples": 0,
            "n_interpolated_samples": 0,
            "n_native_samples": 0,
            "n_valid_native_samples": 0,
            "sample_min": float(sample_axis.values[0]),
            "sample_max": float(sample_axis.values[-1]),
            "well_npz_path": "",
        }
        if not source_ok:
            rows.append(base)
            continue
        inventory_row = inventory_index.get(normalize_well_name(well_name))
        if inventory_row is None:
            base["reason"] = "missing_well_inventory_row"
            rows.append(base)
            continue
        try:
            if source_run_type == "well_auto_tie":
                control = _time_control(
                    source_row=source_row,
                    inventory_row=inventory_row,
                    sample_axis=sample_axis,
                    source_run_dir=source_run_dir,
                    repo_root=repo_root,
                )
            else:
                control = _depth_control(
                    source_row=source_row,
                    inventory_row=inventory_row,
                    sample_axis=sample_axis,
                    line_geometry=line_geometry,
                    source_run_dir=source_run_dir,
                    repo_root=repo_root,
                    trace_lookup=trace_lookup,
                )
        except (FileNotFoundError, ValueError) as exc:
            base["reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(base)
            continue
        try:
            _validate_control_geometry(control, line_geometry)
        except ValueError as exc:
            base["reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(base)
            continue
        controls.append(control)
        provenance = dict(control.provenance)
        transform_text = str(provenance.get("source_transform_path") or "").strip()
        base.update(
            {
                "status": "ok",
                "reason": "",
                "source_las_path": repo_relative_path(provenance["source_las_path"], root=repo_root),
                "source_transform_path": (
                    repo_relative_path(transform_text, root=repo_root) if transform_text else ""
                ),
                "wellbore_class": control.wellbore_class,
                "sampling_mode": control.sampling_mode,
                "n_valid_samples": int(np.count_nonzero(control.valid_mask)),
                "n_observed_samples": int(np.count_nonzero(control.observed_valid_mask)),
                "n_interpolated_samples": int(
                    np.count_nonzero(control.valid_mask & ~control.observed_valid_mask)
                ),
                "n_native_samples": int(control.native.coordinates.size),
                "n_valid_native_samples": int(np.count_nonzero(control.native.valid_mask)),
            }
        )
        rows.append(base)
    if not controls:
        reasons = "; ".join(f"{row['well_name']}: {row['reason']}" for row in rows)
        raise ValueError(f"No valid canonical well controls were built. {reasons}")
    control_set = WellControlSet(
        sample_axis=sample_axis,
        controls=tuple(controls),
        sample_domain=sample_axis.domain,
        sample_unit=sample_axis.unit,
        depth_basis=depth_basis,
        source_run_type=source_run_type,
        provenance={
            "source_run_path": str(source_run_dir),
            "source_summary_path": str(summary_path),
            "metrics_path": str(metrics_path),
            "well_inventory_path": str(inventory_path),
            "target_seismic_path": str(seismic_path),
            "native_source_role": "filtered",
            "gap_policy": "upstream_filtered_only",
            "input_contracts": {
                "source_run": {
                    "path": repo_relative_path(summary_path, root=repo_root),
                    "contract_fingerprint_sha256": source_contract_fingerprint,
                },
                "well_inventory": {
                    "path": repo_relative_path(inventory_summary_path, root=repo_root),
                    "contract_fingerprint_sha256": inventory_contract_fingerprint,
                },
            },
        },
    )
    return control_set, pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS)


def write_well_control_set(
    control_set: WellControlSet,
    manifest: pd.DataFrame,
    *,
    output_dir: Path,
    repo_root: Path,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    def portable_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(value)
        for key, item in list(out.items()):
            if key.endswith("_path") and str(item).strip():
                out[key] = repo_relative_path(str(item), root=repo_root)
        return out

    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError("Well-control manifest columns do not match the frozen v6 contract.")
    manifest_names = [normalize_well_name(value) for value in manifest["well_name"]]
    if len(manifest_names) != len(set(manifest_names)):
        raise ValueError("Well-control manifest well names must be unique after normalization.")
    status_values = set(manifest["status"].astype(str))
    if not status_values.issubset({"ok", "failed"}):
        raise ValueError(f"Well-control manifest contains unsupported status values: {sorted(status_values)}")
    failed_rows = manifest[manifest["status"].astype(str).eq("failed")]
    if failed_rows["reason"].astype(str).str.strip().eq("").any():
        raise ValueError("Every failed well-control manifest row must include a reason.")
    successful_names = {
        normalize_well_name(value)
        for value in manifest.loc[manifest["status"].astype(str).eq("ok"), "well_name"]
    }
    control_names = {normalize_well_name(control.well_name) for control in control_set.controls}
    if successful_names != control_names:
        raise ValueError("Well-control manifest successful rows do not exactly match the control set.")
    filenames = [f"{sanitize_filename(control.well_name)}.npz" for control in control_set.controls]
    folded_filenames = [name.casefold() for name in filenames]
    if len(folded_filenames) != len(set(folded_filenames)):
        raise ValueError("Canonical well names collide after output filename sanitization.")
    output_dir.mkdir(parents=True, exist_ok=False)
    wells_dir = output_dir / "wells"
    wells_dir.mkdir()
    manifest_out = manifest.copy()
    for control in control_set.controls:
        path = wells_dir / f"{sanitize_filename(control.well_name)}.npz"
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "well_name": control.well_name,
            "source_run_type": control.source_run_type,
            "sample_domain": control.sample_axis.domain,
            "sample_unit": control.sample_axis.unit,
            "depth_basis": control_set.depth_basis,
            "wellbore_class": control.wellbore_class,
            "sampling_mode": control.sampling_mode,
            "linear_ai_unit": LINEAR_AI_UNIT,
            "value_domain": "log(AI)",
            "model_axis_value_key": "model_grid_filtered_log_ai",
            "model_axis_valid_mask_key": "valid_mask",
            "model_axis_observed_valid_mask_key": "observed_valid_mask",
            "native_value_key": "native_filtered_log_ai",
            "native_source_role": "filtered",
            "gap_policy": "upstream_filtered_only",
            "provenance": portable_provenance(control.provenance),
            "native": {
                "sample_domain": control.native.sample_domain,
                "sample_unit": control.native.sample_unit,
                "depth_basis": control.native.depth_basis,
                "coordinate_key": "native_coordinates",
                "valid_mask_key": "native_valid_mask",
                "provenance": portable_provenance(control.native.provenance),
            },
        }
        np.savez_compressed(
            path,
            samples=control.sample_axis.values.astype(np.float64),
            model_grid_filtered_log_ai=np.asarray(control.model_grid_filtered_log_ai.values, dtype=np.float32),
            inline=control.inline_by_sample.astype(np.float64),
            xline=control.xline_by_sample.astype(np.float64),
            x_m=control.x_m_by_sample.astype(np.float64),
            y_m=control.y_m_by_sample.astype(np.float64),
            valid_mask=control.valid_mask.astype(bool),
            observed_valid_mask=control.observed_valid_mask.astype(bool),
            native_coordinates=control.native.coordinates.astype(np.float64),
            native_filtered_log_ai=control.native.native_filtered_log_ai.astype(np.float32),
            native_valid_mask=control.native.valid_mask.astype(bool),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
        match = manifest_out["well_name"].astype(str).str.casefold().eq(control.well_name.casefold())
        manifest_out.loc[match, "well_npz_path"] = repo_relative_path(path, root=repo_root)
    manifest_path = output_dir / "well_control_manifest.csv"
    manifest_out.to_csv(manifest_path, index=False)
    input_contracts = dict(control_set.provenance.get("input_contracts") or {})
    primary_artifacts = {"well_control_manifest": manifest_path}
    primary_artifacts.update(
        {
            f"well:{sanitize_filename(control.well_name)}": wells_dir / f"{sanitize_filename(control.well_name)}.npz"
            for control in control_set.controls
        }
    )
    contract_fingerprint = contract_fingerprint_sha256(
        contract_schema_version=SCHEMA_VERSION,
        semantics={
            "sample_domain": control_set.sample_domain,
            "sample_unit": control_set.sample_unit,
            "depth_basis": control_set.depth_basis,
            "value_domain": "log(AI)",
            "linear_ai_unit": LINEAR_AI_UNIT,
            "well_control_layers": ["model_axis", "native"],
            "model_axis_masks": ["valid_mask", "observed_valid_mask"],
            "native_source_role": "filtered",
            "gap_policy": "upstream_filtered_only",
        },
        business_config=resolved_config,
        input_contracts=input_contracts,
        primary_artifacts=primary_artifacts,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": contract_fingerprint,
        "input_contracts": input_contracts,
        "resolved_config": dict(resolved_config),
        "source_adapter": control_set.source_run_type,
        "sample_axis": control_set.sample_axis.describe(),
        "depth_basis": control_set.depth_basis,
        "native_source_role": "filtered",
        "gap_policy": "upstream_filtered_only",
        "counts": {
            "candidate_wells": int(len(manifest_out)),
            "successful_wells": int((manifest_out["status"] == "ok").sum()),
            "failed_wells": int((manifest_out["status"] != "ok").sum()),
            "valid_samples": int(sum(np.count_nonzero(item.valid_mask) for item in control_set.controls)),
            "observed_samples": int(
                sum(np.count_nonzero(item.observed_valid_mask) for item in control_set.controls)
            ),
            "interpolated_samples": int(
                sum(
                    np.count_nonzero(item.valid_mask & ~item.observed_valid_mask)
                    for item in control_set.controls
                )
            ),
            "valid_native_samples": int(
                sum(np.count_nonzero(item.native.valid_mask) for item in control_set.controls)
            ),
        },
        "provenance": portable_provenance(control_set.provenance),
        "outputs": {
            "well_control_manifest": repo_relative_path(manifest_path, root=repo_root),
            "wells_dir": repo_relative_path(wells_dir, root=repo_root),
        },
    }
    from cup.utils.io import write_json

    write_json(output_dir / "run_summary.json", summary)
    return summary


def load_well_control_set(run_dir: Path, *, repo_root: Path) -> WellControlSet:
    """Load and semantically validate a canonical immutable Step 6 run."""

    summary_path = run_dir / "run_summary.json"
    manifest_path = run_dir / "well_control_manifest.csv"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete well-control run: {run_dir}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("schema_version") != SCHEMA_VERSION or not is_consumable_contract_status(summary.get("status")):
        raise ValueError(
            f"Unsupported or unsuccessful well-control run: {run_dir}; "
            f"regenerate Step 6 with schema {SCHEMA_VERSION}."
        )
    if summary.get("native_source_role") != "filtered" or summary.get("gap_policy") != "upstream_filtered_only":
        raise ValueError(f"Well-control run does not use the filtered-LAS/upstream-gap contract: {run_dir}")
    require_contract_fingerprint(summary, label=f"WellControlSet {run_dir}")
    outputs = dict(summary.get("outputs") or {})
    recorded_manifest_path = resolve_relative_path(
        str(outputs.get("well_control_manifest") or ""), root=repo_root
    )
    if (
        recorded_manifest_path.resolve() != manifest_path.resolve()
    ):
        raise ValueError("well_control_manifest.csv path does not match run_summary.json.")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    _required_columns(manifest, set(MANIFEST_COLUMNS), path=manifest_path)
    manifest_names = [normalize_well_name(value) for value in manifest["well_name"]]
    if len(manifest_names) != len(set(manifest_names)):
        raise ValueError("well_control_manifest.csv contains duplicate normalized well names.")
    status_values = set(manifest["status"].astype(str))
    if not status_values.issubset({"ok", "failed"}):
        raise ValueError(f"well_control_manifest.csv contains unsupported statuses: {sorted(status_values)}")
    failed = manifest[manifest["status"].astype(str).eq("failed")]
    if failed["well_npz_path"].astype(str).str.strip().ne("").any():
        raise ValueError("Failed well-control manifest rows must not reference consumable NPZ files.")
    successful = manifest[manifest["status"].astype(str).eq("ok")]
    if successful.empty:
        raise ValueError("Successful well-control run has no successful manifest rows.")
    first_row = successful.iloc[0]
    first_path = resolve_relative_path(str(first_row["well_npz_path"]), root=repo_root)
    with np.load(first_path, allow_pickle=False) as first_data:
        first_samples = np.asarray(first_data["samples"], dtype=np.float64)
    axis_info = dict(summary["sample_axis"])
    axis = SampleAxis(
        values=first_samples,
        domain=str(axis_info["sample_domain"]),
        unit=str(axis_info["sample_unit"]),
        depth_basis=summary.get("depth_basis"),
    )
    described_axis = axis.describe()
    for key in ("n_sample", "sample_min", "sample_max", "sample_step", "sample_domain", "sample_unit"):
        recorded = axis_info.get(key)
        actual = described_axis[key]
        if isinstance(actual, float):
            try:
                matches = np.isclose(float(recorded), actual, rtol=0.0, atol=1e-10)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = recorded == actual
        if not matches:
            raise ValueError(
                f"WellControlSet run_summary SampleAxis {key} mismatch: "
                f"recorded={recorded!r}, actual={actual!r}."
            )
    controls: list[WellControl] = []
    for _, row in successful.iterrows():
        path = resolve_relative_path(str(row["well_npz_path"]), root=repo_root)
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {
                "samples",
                "model_grid_filtered_log_ai",
                "inline",
                "xline",
                "x_m",
                "y_m",
                "valid_mask",
                "observed_valid_mask",
                "native_coordinates",
                "native_filtered_log_ai",
                "native_valid_mask",
                "metadata_json",
            }:
                raise ValueError(f"Unexpected well-control NPZ keys: {path}")
            if data["samples"].dtype != np.dtype("float64") or any(
                data[key].dtype != np.dtype("float64") for key in ("inline", "xline", "x_m", "y_m")
            ):
                raise ValueError(f"Well-control sample/position arrays must be float64: {path}")
            if (
                data["model_grid_filtered_log_ai"].dtype != np.dtype("float32")
                or data["valid_mask"].dtype != np.dtype("bool")
                or data["observed_valid_mask"].dtype != np.dtype("bool")
            ):
                raise ValueError(
                    f"Well-control model_grid_filtered_log_ai/valid/observed mask dtypes must be float32/bool: {path}"
                )
            if (
                data["native_coordinates"].dtype != np.dtype("float64")
                or data["native_filtered_log_ai"].dtype != np.dtype("float32")
                or data["native_valid_mask"].dtype != np.dtype("bool")
            ):
                raise ValueError(f"Native well-control coordinate/log/mask dtypes are invalid: {path}")
            metadata_array = np.asarray(data["metadata_json"])
            if metadata_array.ndim != 0 or metadata_array.dtype.kind not in {"U", "S"}:
                raise ValueError(f"Well-control metadata_json must be a scalar string: {path}")
            metadata = json.loads(str(metadata_array.item()))
            if metadata.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported well-control NPZ schema: {path}")
            expected_metadata = {
                "source_run_type": str(summary["source_adapter"]),
                "sample_domain": axis.domain,
                "sample_unit": axis.unit,
                "depth_basis": summary.get("depth_basis"),
                "linear_ai_unit": LINEAR_AI_UNIT,
                "value_domain": "log(AI)",
                "model_axis_value_key": "model_grid_filtered_log_ai",
                "model_axis_valid_mask_key": "valid_mask",
                "model_axis_observed_valid_mask_key": "observed_valid_mask",
                "native_value_key": "native_filtered_log_ai",
                "native_source_role": "filtered",
                "gap_policy": "upstream_filtered_only",
            }
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise ValueError(
                        f"Well-control metadata {key} mismatch in {path}: "
                        f"expected {expected!r}, got {metadata.get(key)!r}."
                    )
            if normalize_well_name(metadata.get("well_name")) != normalize_well_name(row["well_name"]):
                raise ValueError(f"Well-control manifest/NPZ well_name mismatch: {path}")
            native_metadata = dict(metadata.get("native") or {})
            if (
                native_metadata.get("sample_domain") != axis.domain
                or native_metadata.get("sample_unit") != axis.unit
                or native_metadata.get("depth_basis") != summary.get("depth_basis")
                or native_metadata.get("coordinate_key") != "native_coordinates"
                or native_metadata.get("valid_mask_key") != "native_valid_mask"
            ):
                raise ValueError(f"Native well-control metadata is inconsistent: {path}")
            native = NativeWellControl(
                well_name=str(metadata["well_name"]),
                coordinates=np.asarray(data["native_coordinates"], dtype=np.float64),
                native_filtered_log_ai=np.asarray(data["native_filtered_log_ai"], dtype=np.float64),
                valid_mask=np.asarray(data["native_valid_mask"], dtype=bool),
                sample_domain=axis.domain,
                sample_unit=axis.unit,
                depth_basis=summary.get("depth_basis"),
                provenance=dict(native_metadata.get("provenance") or {}),
            )
            file_axis = np.asarray(data["samples"], dtype=np.float64)
            if not np.array_equal(file_axis, axis.values):
                raise ValueError(f"Well-control SampleAxis mismatch: {path}")
            control = _control_from_arrays(
                well_name=str(metadata["well_name"]),
                sample_axis=axis,
                model_grid_filtered_log_ai=np.asarray(data["model_grid_filtered_log_ai"], dtype=np.float64),
                inline=np.asarray(data["inline"], dtype=np.float64),
                xline=np.asarray(data["xline"], dtype=np.float64),
                x_m=np.asarray(data["x_m"], dtype=np.float64),
                y_m=np.asarray(data["y_m"], dtype=np.float64),
                observed_valid_mask=np.asarray(data["observed_valid_mask"], dtype=bool),
                wellbore_class=str(metadata["wellbore_class"]),
                sampling_mode=str(metadata["sampling_mode"]),
                source_run_type=str(metadata["source_run_type"]),
                provenance=dict(metadata["provenance"]),
                native=native,
            )
            if not np.array_equal(control.valid_mask, np.asarray(data["valid_mask"], dtype=bool)):
                raise ValueError(f"Well-control NPZ valid_mask disagrees with finite values: {path}")
            if not np.array_equal(
                control.observed_valid_mask,
                np.asarray(data["observed_valid_mask"], dtype=bool),
            ):
                raise ValueError(f"Well-control NPZ observed_valid_mask changed during loading: {path}")
            controls.append(control)
    return WellControlSet(
        sample_axis=axis,
        controls=tuple(controls),
        sample_domain=axis.domain,
        sample_unit=axis.unit,
        depth_basis=summary.get("depth_basis"),
        source_run_type=str(summary["source_adapter"]),
        provenance=dict(summary["provenance"]),
    )

QC_SCHEMA_VERSION = "real_field_well_control_qc_v1"


def _safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 3 or np.std(a[valid]) <= 0.0 or np.std(b[valid]) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def load_depth_forward_inputs(
    run_dir: Path,
    *,
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, float, float, Path]:
    path = run_dir / "forward_model_inputs.json"
    time_s, amplitude, relation, _payload = load_forward_inputs(
        run_dir,
        repo_root=repo_root,
        domain="depth",
        depth_basis="tvdss",
    )
    if relation is None:
        raise ValueError("Depth forward inputs must contain ai_velocity_relation.")
    return time_s, amplitude, relation.a, relation.b, path


def forward_depth_finite_runs(
    log_ai: np.ndarray,
    depth_m: np.ndarray,
    *,
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    relation_a: float,
    relation_b: float,
) -> np.ndarray:
    values = np.asarray(log_ai, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    relation = AIVelocityRelation(a=relation_a, b=relation_b)
    for start, stop in _finite_runs(np.isfinite(values)):
        if stop - start < 2:
            continue
        local_log_ai = values[start:stop]
        velocity = relation.velocity_from_ai(np.exp(local_log_ai))
        output[start:stop] = forward_depth(
            local_log_ai,
            velocity,
            depth[start:stop],
            wavelet_time_s,
            wavelet_amplitude,
        )
    return output


def sample_seismic_along_control(control: WellControl, survey: Any) -> np.ndarray:
    """Bilinearly sample one seismic value at every well-path/sample intersection."""

    needed: set[tuple[int, int]] = set()
    plans: dict[int, list[tuple[tuple[int, int], float]]] = {}
    for sample_index in np.flatnonzero(
        np.isfinite(control.inline_by_sample) & np.isfinite(control.xline_by_sample)
    ):
        i_float, j_float = survey.line_geometry.line_to_index(
            float(control.inline_by_sample[sample_index]),
            float(control.xline_by_sample[sample_index]),
        )
        i0, i1 = int(np.floor(i_float)), int(np.ceil(i_float))
        j0, j1 = int(np.floor(j_float)), int(np.ceil(j_float))
        wi, wj = float(i_float - i0), float(j_float - j0)
        local: dict[tuple[int, int], float] = {}
        for key, weight in (
            ((i0, j0), (1.0 - wi) * (1.0 - wj)),
            ((i0, j1), (1.0 - wi) * wj),
            ((i1, j0), wi * (1.0 - wj)),
            ((i1, j1), wi * wj),
        ):
            if weight <= 0.0:
                continue
            if survey.trace_flat_index(*key) < 0:
                raise ValueError(f"{control.well_name}: trajectory intersects a missing seismic trace.")
            local[key] = local.get(key, 0.0) + weight
            needed.add(key)
        plans[int(sample_index)] = sorted(local.items())
    if not needed:
        raise ValueError(f"{control.well_name}: no valid seismic sampling positions.")

    traces = survey.read_traces_at_indices(sorted(needed), domain=control.sample_axis.domain)
    output = np.full(control.sample_axis.values.shape, np.nan, dtype=np.float64)
    for sample_index, weighted_indices in plans.items():
        value = 0.0
        for key, weight in weighted_indices:
            trace = traces[key]
            if not np.array_equal(
                np.asarray(trace.basis, dtype=np.float64),
                control.sample_axis.values,
            ):
                raise ValueError("Survey trace axis differs from the canonical well-control axis.")
            value += weight * float(trace.values[sample_index])
        output[sample_index] = value
    return output


def horizon_markers_along_control(
    control: WellControl,
    target_zone: TargetZone,
) -> list[tuple[float, str]]:
    axis = control.sample_axis.values
    position_valid = np.isfinite(control.inline_by_sample) & np.isfinite(control.xline_by_sample)
    markers: list[tuple[float, str]] = []
    for name in target_zone.horizon_names:
        local_horizon = np.full(axis.shape, np.nan, dtype=np.float64)
        for index in np.flatnonzero(position_valid):
            local_horizon[index] = target_zone.get_horizon_interpretation_at_location(
                name,
                float(control.inline_by_sample[index]),
                float(control.xline_by_sample[index]),
            )
        valid = np.isfinite(local_horizon)
        if not np.any(valid):
            raise ValueError(f"{control.well_name}: horizon {name!r} has no well-path support.")
        candidates = np.flatnonzero(valid)
        selected = int(candidates[np.argmin(np.abs(axis[candidates] - local_horizon[candidates]))])
        markers.append((float(axis[selected]), name))
    if any(markers[index + 1][0] <= markers[index][0] for index in range(len(markers) - 1)):
        raise ValueError(f"{control.well_name}: target horizons are not ordered along the well path.")
    return markers


def _target_support_slice(
    axis: np.ndarray,
    valid: np.ndarray,
    markers: list[tuple[float, str]],
) -> tuple[slice, float]:
    target = (axis >= markers[0][0]) & (axis <= markers[-1][0])
    target_count = int(np.count_nonzero(target))
    runs = _finite_runs(valid & target)
    if not runs:
        raise ValueError("No common full/body/seismic support inside the target interval.")
    start, stop = max(runs, key=lambda item: item[1] - item[0])
    if stop - start < 8:
        raise ValueError("Fewer than eight common samples inside the target interval.")
    return slice(start, stop), float((stop - start) / target_count)


def _dynamic_xcorr(
    real: grid.Seismic,
    synthetic: grid.Seismic,
    *,
    window_axis_units: float,
) -> grid.DynamicXCorr:
    step = float(real.sampling_rate)
    half = max(2, int(round(float(window_axis_units) / step)) // 2)
    first = np.pad(np.asarray(real.values, dtype=np.float64), half, mode="reflect")
    second = np.pad(np.asarray(synthetic.values, dtype=np.float64), half, mode="reflect")
    rows = []
    for index in range(real.size):
        rows.append(
            normalized_xcorr(
                first[index : index + 2 * half],
                second[index : index + 2 * half],
            )
        )
    return grid.DynamicXCorr(
        np.asarray(rows, dtype=np.float64),
        np.asarray(real.basis, dtype=np.float64),
        "twt" if real.is_twt else "tvdss",
        name="Local lag [s]" if real.is_twt else "Local lag [m]",
    )


def _waveform_objects(
    *,
    axis: np.ndarray,
    log_ai: np.ndarray,
    synthetic: np.ndarray,
    real: np.ndarray,
    dynamic_window_axis_units: float,
    name: str,
    basis_type: str = "tvdss",
) -> tuple[grid.Log, grid.Log, grid.Seismic, grid.Seismic, grid.XCorr, grid.DynamicXCorr]:
    if basis_type not in {"twt", "tvdss"}:
        raise ValueError("Waveform objects require an explicit TWT or TVDSS basis.")
    window_axis_units = float(dynamic_window_axis_units)
    if not np.isfinite(window_axis_units) or window_axis_units <= 0.0:
        raise ValueError("dynamic_window_axis_units must be finite and positive.")
    linear_ai = grid.Log(
        np.exp(log_ai),
        axis,
        basis_type,
        name=name,
        unit="m/s*g/cm3",
    )
    reflectivity_values = np.r_[0.0, reflectivity_from_log_ai(log_ai)]
    reflectivity = grid.Log(
        reflectivity_values,
        axis,
        basis_type,
        name="Reflectivity",
    )
    synthetic_trace = grid.Seismic(synthetic, axis, basis_type, name="Synthetic")
    real_trace = grid.Seismic(real, axis, basis_type, name="Seismic")
    xcorr_values = normalized_xcorr(real, synthetic)
    lags = float(axis[1] - axis[0]) * np.arange(-(axis.size - 1), axis.size)
    xcorr = grid.XCorr(xcorr_values, lags, "tlag" if basis_type == "twt" else "zlag", name="XCorr")
    dynamic = _dynamic_xcorr(
        real_trace,
        synthetic_trace,
        window_axis_units=window_axis_units,
    )
    return linear_ai, reflectivity, synthetic_trace, real_trace, xcorr, dynamic


def _event_windows(
    seismic: np.ndarray,
    *,
    threshold_fraction: float,
    maximum: int,
) -> list[tuple[int, int]]:
    values = np.asarray(seismic, dtype=np.float64)
    centered = values - np.median(values)
    sign = np.sign(centered)
    for index in range(1, sign.size):
        if sign[index] == 0.0:
            sign[index] = sign[index - 1]
    if sign[0] == 0.0:
        nonzero = np.flatnonzero(sign)
        sign[: nonzero[0] if nonzero.size else sign.size] = sign[nonzero[0]] if nonzero.size else 1.0
    changes = np.r_[True, sign[1:] != sign[:-1], True]
    runs = np.flatnonzero(changes)
    threshold = float(threshold_fraction) * float(np.percentile(np.abs(centered), 95.0))
    scored: list[tuple[float, int, int]] = []
    for start, stop in zip(runs[:-1], runs[1:]):
        if stop - start < 2:
            continue
        score = float(np.max(np.abs(centered[start:stop])))
        if score >= threshold:
            scored.append((score, int(start), int(stop)))
    selected = sorted(scored, reverse=True)[: int(maximum)]
    return sorted((start, stop) for _score, start, stop in selected)


def _plot_event_comparison(
    *,
    output_path: Path,
    well_name: str,
    axis: np.ndarray,
    model_grid_filtered_log_ai: np.ndarray,
    body_log_ai: np.ndarray,
    real: np.ndarray,
    full_synthetic: np.ndarray,
    body_synthetic: np.ndarray,
    body_fwhm_m: float,
    threshold_fraction: float,
    maximum_events: int,
) -> int:
    events = _event_windows(
        real,
        threshold_fraction=threshold_fraction,
        maximum=maximum_events,
    )
    if not events:
        raise ValueError(f"{well_name}: no target-interval waveform events passed the threshold.")
    fig, axes = plt.subplots(
        len(events),
        4,
        figsize=(14.5, max(3.2, 2.6 * len(events))),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (event_start, event_stop) in enumerate(events):
        width = event_stop - event_start
        pad = max(3, width)
        start = max(0, event_start - pad)
        stop = min(axis.size, event_stop + pad)
        local = slice(start, stop)
        local_axis = axis[local]

        axes[row, 0].plot(real[local], local_axis, color="black", lw=1.5)
        axes[row, 0].axhspan(axis[event_start], axis[event_stop - 1], color="tab:blue", alpha=0.12)
        axes[row, 0].set_title("Real seismic" if row == 0 else "")

        axes[row, 1].plot(model_grid_filtered_log_ai[local], local_axis, color="black", lw=1.1, label="full")
        axes[row, 1].plot(
            body_log_ai[local],
            local_axis,
            color="tab:red",
            lw=1.5,
            label=f"{body_fwhm_m:g} m body",
        )
        axes[row, 1].set_title("log-AI" if row == 0 else "")
        if row == 0:
            axes[row, 1].legend(fontsize=8)

        residual = model_grid_filtered_log_ai[local] - body_log_ai[local]
        axes[row, 2].plot(
            residual,
            local_axis,
            color="tab:purple",
            lw=1.2,
            label=f"full − {body_fwhm_m:g} m body",
        )
        axes[row, 2].axvline(0.0, color="black", lw=0.8, alpha=0.6)
        axes[row, 2].set_title("log-AI residual" if row == 0 else "")
        if row == 0:
            axes[row, 2].legend(fontsize=8)

        axes[row, 3].plot(full_synthetic[local], local_axis, color="tab:blue", lw=1.2, label="full forward")
        axes[row, 3].plot(body_synthetic[local], local_axis, color="tab:orange", lw=1.2, label="body forward")
        axes[row, 3].set_title("Shared-gain synthetic" if row == 0 else "")
        if row == 0:
            axes[row, 3].legend(fontsize=8)

        for column in range(4):
            axes[row, column].set_ylim(float(local_axis[-1]), float(local_axis[0]))
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_ylabel("TVDSS [m]" if column == 0 else "")
    fig.suptitle(f"{well_name} | target-interval full/body waveform slices")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return len(events)


def write_depth_well_control_qc(
    control_set: WellControlSet,
    *,
    survey: Any,
    target_zone: TargetZone,
    forward_inputs_run_dir: Path,
    output_dir: Path,
    repo_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the target-interval QC figures for every successful Step 6 well."""

    expected = {
        "body_smoothing_fwhm_m",
        "dynamic_correlation_window_m",
        "event_threshold_fraction",
        "max_event_windows_per_well",
    }
    if set(config) != expected:
        raise ValueError(f"real_field_well_controls_qc must contain exactly {sorted(expected)}.")
    if control_set.sample_domain != "depth" or control_set.depth_basis != "tvdss":
        raise ValueError("Current Step 6 forward QC requires depth/TVDSS well controls.")
    body_fwhm = float(config["body_smoothing_fwhm_m"])
    dynamic_window = float(config["dynamic_correlation_window_m"])
    threshold_fraction = float(config["event_threshold_fraction"])
    maximum_events = int(config["max_event_windows_per_well"])
    if body_fwhm <= 0.0 or dynamic_window <= 0.0 or not 0.0 < threshold_fraction <= 1.0 or maximum_events < 1:
        raise ValueError("Step 6 QC numeric settings are invalid.")
    wavelet_time, wavelet_amp, relation_a, relation_b, forward_inputs_path = (
        load_depth_forward_inputs(forward_inputs_run_dir, repo_root=repo_root)
    )

    figures_root = output_dir / "figures"
    figures_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for control in control_set.controls:
        well_dir = figures_root / sanitize_filename(control.well_name)
        well_dir.mkdir()
        axis = control.sample_axis.values
        model_grid_filtered_log_ai = np.asarray(control.model_grid_filtered_log_ai.values, dtype=np.float64)
        body_log_ai = gaussian_smooth_finite_runs_numpy(
            model_grid_filtered_log_ai,
            axis,
            fwhm_m=body_fwhm,
        )
        real = sample_seismic_along_control(control, survey)
        full_forward = forward_depth_finite_runs(
            model_grid_filtered_log_ai,
            axis,
            wavelet_time_s=wavelet_time,
            wavelet_amplitude=wavelet_amp,
            relation_a=relation_a,
            relation_b=relation_b,
        )
        body_forward = forward_depth_finite_runs(
            body_log_ai,
            axis,
            wavelet_time_s=wavelet_time,
            wavelet_amplitude=wavelet_amp,
            relation_a=relation_a,
            relation_b=relation_b,
        )
        markers = horizon_markers_along_control(control, target_zone)
        common = (
            control.valid_mask
            & np.isfinite(real)
            & np.isfinite(full_forward)
            & np.isfinite(body_forward)
        )
        selected, support_fraction = _target_support_slice(axis, common, markers)
        local_axis = axis[selected]
        local_real = real[selected]
        real_std = float(np.std(local_real))
        if not np.isfinite(real_std) or real_std <= 0.0:
            raise ValueError(f"{control.well_name}: target seismic has zero variance.")
        local_real = (local_real - float(np.mean(local_real))) / real_std
        local_full_forward = full_forward[selected]
        local_body_forward = body_forward[selected]
        denominator = float(np.dot(local_full_forward, local_full_forward))
        signed_gain = (
            float(np.dot(local_real, local_full_forward) / denominator)
            if denominator > 0.0
            else 1.0
        )
        gain = abs(signed_gain)
        local_full_forward = gain * local_full_forward
        local_body_forward = gain * local_body_forward
        local_markers = [item for item in markers if local_axis[0] <= item[0] <= local_axis[-1]]

        full_objects = _waveform_objects(
            axis=local_axis,
            log_ai=model_grid_filtered_log_ai[selected],
            synthetic=local_full_forward,
            real=local_real,
            dynamic_window_axis_units=dynamic_window,
            name="Full AI",
        )
        full_corr = _safe_corr(local_real, local_full_forward)
        fig, _ = plot_well_waveform_qc(
            [full_objects[0]],
            full_objects[1],
            full_objects[2],
            full_objects[3],
            full_objects[4],
            full_objects[5],
            figsize=(13.0, 7.5),
            synthetic_ai=full_objects[0],
            title=f"Step 6 full forward QC | {control.well_name} | corr={full_corr:.3f}",
            horizon_markers=local_markers,
        )
        full_path = well_dir / "full_waveform_qc.png"
        fig.savefig(full_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        body_objects = _waveform_objects(
            axis=local_axis,
            log_ai=body_log_ai[selected],
            synthetic=local_body_forward,
            real=local_real,
            dynamic_window_axis_units=dynamic_window,
            name=f"{body_fwhm:g} m body AI",
        )
        body_corr = _safe_corr(local_real, local_body_forward)
        fig, _ = plot_well_waveform_qc(
            [body_objects[0]],
            body_objects[1],
            body_objects[2],
            body_objects[3],
            body_objects[4],
            body_objects[5],
            figsize=(13.0, 7.5),
            synthetic_ai=body_objects[0],
            title=f"Step 6 {body_fwhm:g} m body forward QC | {control.well_name} | corr={body_corr:.3f}",
            horizon_markers=local_markers,
        )
        body_path = well_dir / "body_waveform_qc.png"
        fig.savefig(body_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        comparison_path = well_dir / "event_waveform_comparison.png"
        event_count = _plot_event_comparison(
            output_path=comparison_path,
            well_name=control.well_name,
            axis=local_axis,
            model_grid_filtered_log_ai=model_grid_filtered_log_ai[selected],
            body_log_ai=body_log_ai[selected],
            real=local_real,
            full_synthetic=local_full_forward,
            body_synthetic=local_body_forward,
            body_fwhm_m=body_fwhm,
            threshold_fraction=threshold_fraction,
            maximum_events=maximum_events,
        )
        rows.append(
            {
                "well_name": control.well_name,
                "target_support_fraction": support_fraction,
                "shared_forward_gain": gain,
                "signed_forward_gain": signed_gain,
                "full_forward_correlation": full_corr,
                "body_forward_correlation": body_corr,
                "event_window_count": event_count,
                "full_waveform_qc": repo_relative_path(full_path, root=repo_root),
                "body_waveform_qc": repo_relative_path(body_path, root=repo_root),
                "event_waveform_comparison": repo_relative_path(comparison_path, root=repo_root),
            }
        )

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame.from_records(rows).to_csv(metrics_path, index=False)
    manifest = {
        "schema_version": QC_SCHEMA_VERSION,
        "status": "ok",
        "sample_domain": "depth",
        "depth_basis": "tvdss",
        "config": dict(config),
        "forward_model_inputs": repo_relative_path(forward_inputs_path, root=repo_root),
        "well_count": len(rows),
        "outputs": {
            "metrics_csv": repo_relative_path(metrics_path, root=repo_root),
            "figures_dir": repo_relative_path(figures_root, root=repo_root),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = [
    "DEPTH_SOURCE_SCHEMA",
    "LINEAR_AI_UNIT",
    "MANIFEST_COLUMNS",
    "QC_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TIME_SOURCE_SCHEMA",
    "NativeWellControl",
    "WellControl",
    "WellControlSet",
    "build_well_control_set",
    "forward_depth_finite_runs",
    "horizon_markers_along_control",
    "load_depth_forward_inputs",
    "load_well_control_set",
    "sample_seismic_along_control",
    "write_depth_well_control_qc",
    "write_well_control_set",
]
