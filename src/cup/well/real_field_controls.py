"""Canonical real-field well controls shared by every LFM builder.

This module is deliberately independent of any LFM implementation.  Source
adapters convert upstream time/depth products into one sampled ``log(AI)``
contract; consumers never need to know which upstream workflow produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cup.seismic.geometry import SampleAxis, SurveyLineGeometry
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_artifact_path,
    resolve_relative_path,
    sanitize_filename,
)
from cup.well.assets import build_file_lookup, normalize_well_name
from cup.well.td import load_workflow_time_depth_table_csv
from cup.well.trajectory import WellTrajectory
from wtie.processing import grid


SCHEMA_VERSION = "real_field_well_controls_v3"
TIME_SOURCE_SCHEMA = "well_auto_tie_v3"
DEPTH_SOURCE_SCHEMA = "wavelet_batch_synthetic_depth_v3"
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
    "sample_min",
    "sample_max",
    "well_npz_path",
]


@dataclass(frozen=True)
class WellControl:
    well_name: str
    sample_axis: SampleAxis
    log_ai: grid.Log
    inline_by_sample: np.ndarray
    xline_by_sample: np.ndarray
    x_m_by_sample: np.ndarray
    y_m_by_sample: np.ndarray
    valid_mask: np.ndarray
    wellbore_class: str
    sampling_mode: str
    source_run_type: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        n = self.sample_axis.values.size
        values = np.asarray(self.log_ai.values, dtype=np.float64)
        arrays = {
            "inline_by_sample": np.asarray(self.inline_by_sample, dtype=np.float64),
            "xline_by_sample": np.asarray(self.xline_by_sample, dtype=np.float64),
            "x_m_by_sample": np.asarray(self.x_m_by_sample, dtype=np.float64),
            "y_m_by_sample": np.asarray(self.y_m_by_sample, dtype=np.float64),
            "valid_mask": np.asarray(self.valid_mask, dtype=bool),
        }
        if values.shape != (n,) or not np.array_equal(self.log_ai.basis, self.sample_axis.values):
            raise ValueError(f"{self.well_name}: log_ai must be aligned to the canonical SampleAxis.")
        expected_basis = "twt" if self.sample_axis.domain == "time" else "tvdss"
        if not getattr(self.log_ai, f"is_{expected_basis}"):
            raise ValueError(f"{self.well_name}: log_ai basis is inconsistent with {self.sample_axis.domain}.")
        for name, array in arrays.items():
            if array.shape != (n,):
                raise ValueError(f"{self.well_name}: {name} must have shape ({n},).")
            object.__setattr__(self, name, array)
        valid = arrays["valid_mask"]
        finite = np.isfinite(values)
        for name in ("inline_by_sample", "xline_by_sample", "x_m_by_sample", "y_m_by_sample"):
            finite &= np.isfinite(arrays[name])
        if not np.array_equal(valid, finite):
            raise ValueError(f"{self.well_name}: valid_mask must exactly describe finite logAI and positions.")
        if not np.any(valid):
            raise ValueError(f"{self.well_name}: canonical control has no valid samples.")


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
    if str(summary.get("status") or "") not in {"ok", "success"}:
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
    log_ai = np.full(ai.shape, np.nan, dtype=np.float64)
    log_ai[valid] = np.log(ai[valid])
    return md, log_ai


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


def _control_from_arrays(
    *,
    well_name: str,
    sample_axis: SampleAxis,
    log_ai: np.ndarray,
    inline: np.ndarray,
    xline: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    wellbore_class: str,
    sampling_mode: str,
    source_run_type: str,
    provenance: Mapping[str, Any],
) -> WellControl:
    arrays = [np.asarray(value, dtype=np.float64).copy() for value in (log_ai, inline, xline, x_m, y_m)]
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    for value in arrays:
        value[~valid] = np.nan
    basis_type = "twt" if sample_axis.domain == "time" else "tvdss"
    log = grid.Log(arrays[0], sample_axis.values.copy(), basis_type, name="log_ai", unit="ln(m/s*g/cm3)")
    return WellControl(
        well_name=well_name,
        sample_axis=sample_axis,
        log_ai=log,
        inline_by_sample=arrays[1],
        xline_by_sample=arrays[2],
        x_m_by_sample=arrays[3],
        y_m_by_sample=arrays[4],
        valid_mask=valid,
        wellbore_class=wellbore_class,
        sampling_mode=sampling_mode,
        source_run_type=source_run_type,
        provenance=dict(provenance),
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
    las_path = resolve_artifact_path(source_row.get("filtered_las_file"), root=repo_root, run_dir=source_run_dir)
    tdt_path = resolve_artifact_path(source_row.get("optimized_tdt_file"), root=repo_root, run_dir=source_run_dir)
    if las_path is None or not las_path.is_file() or tdt_path is None or not tdt_path.is_file():
        raise FileNotFoundError(f"{well_name}: filtered LAS or optimized TDT is missing.")
    md, log_ai_md = _read_ai_las(las_path)
    table = load_workflow_time_depth_table_csv(tdt_path)
    if not table.is_md_domain:
        raise ValueError(f"{well_name}: optimized TDT must use MD domain.")
    twt = np.asarray(table.twt, dtype=np.float64)
    table_md = np.asarray(table.md, dtype=np.float64)
    md_at_twt = _interp_no_extrapolation(sample_axis.values, twt, table_md)
    log_ai = _interp_finite_runs(md_at_twt, md, log_ai_md)

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
    return _control_from_arrays(
        well_name=well_name,
        sample_axis=sample_axis,
        log_ai=log_ai,
        inline=positions[0],
        xline=positions[1],
        x_m=positions[2],
        y_m=positions[3],
        wellbore_class=wellbore_class,
        sampling_mode=sampling_mode,
        source_run_type="well_auto_tie",
        provenance={
            "source_las_path": str(las_path),
            "source_transform_path": str(transform_path),
            "optimized_tdt_path": str(tdt_path),
        },
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
    las_path = resolve_artifact_path(source_row.get("shifted_filtered_las_path"), root=repo_root, run_dir=source_run_dir)
    if las_path is None or not las_path.is_file():
        raise FileNotFoundError(f"{well_name}: shifted filtered LAS is missing.")
    md, log_ai_md = _read_ai_las(las_path)
    wellbore_class = str(inventory_row.get("wellbore_class") or "unknown").strip().casefold()
    trace_path = trace_lookup.get(normalize_well_name(well_name))
    if wellbore_class == "deviated":
        if trace_path is None:
            raise FileNotFoundError(f"{well_name}: deviated well lacks a project trajectory file.")
        trajectory = WellTrajectory.from_petrel_trace(trace_path).with_well_name(well_name)
        tvdss_at_md = _interp_no_extrapolation(md, trajectory.md_m, trajectory.tvdss_m)
        valid = np.isfinite(tvdss_at_md)
        if np.count_nonzero(valid) < 2 or np.any(np.diff(tvdss_at_md[valid]) <= 0.0):
            raise ValueError(f"{well_name}: trajectory TVDSS support for shifted LAS is not strictly increasing.")
        md_at_sample = _interp_no_extrapolation(sample_axis.values, tvdss_at_md[valid], md[valid])
        log_ai = _interp_finite_runs(md_at_sample, md, log_ai_md)
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
    elif wellbore_class == "vertical":
        kb_m = _finite_number(inventory_row.get("kb_m"), label=f"{well_name}.kb_m")
        tvdss_las = md - kb_m
        log_ai = _interp_finite_runs(sample_axis.values, tvdss_las, log_ai_md)
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
    else:
        raise ValueError(f"{well_name}: unsupported/unknown wellbore_class={wellbore_class!r}.")
    return _control_from_arrays(
        well_name=well_name,
        sample_axis=sample_axis,
        log_ai=log_ai,
        inline=inline,
        xline=xline,
        x_m=x_m,
        y_m=y_m,
        wellbore_class=wellbore_class,
        sampling_mode=sampling_mode,
        source_run_type="wavelet_batch_synthetic_depth",
        provenance={
            "source_las_path": str(las_path),
            "source_transform_path": "" if transform_path is None else str(transform_path),
            **transform_metadata,
        },
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

    allowed_config = {"source_run_type", "source_run_dir", "well_inventory_file", "well_trace_dir"}
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
            {"well_name", "tie_status", "filtered_las_file", "optimized_tdt_file", "optimized_trace_sample_plan_file"},
            path=metrics_path,
        )
        success = metrics["tie_status"].astype(str).str.casefold().eq("success")
    else:
        _required_columns(metrics, {"well_name", "status", "shifted_filtered_las_path"}, path=metrics_path)
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
        raise ValueError("Well-control manifest columns do not match the frozen v3 contract.")
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
            "provenance": portable_provenance(control.provenance),
        }
        np.savez_compressed(
            path,
            samples=control.sample_axis.values.astype(np.float64),
            log_ai=np.asarray(control.log_ai.values, dtype=np.float32),
            inline=control.inline_by_sample.astype(np.float64),
            xline=control.xline_by_sample.astype(np.float64),
            x_m=control.x_m_by_sample.astype(np.float64),
            y_m=control.y_m_by_sample.astype(np.float64),
            valid_mask=control.valid_mask.astype(bool),
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
        "counts": {
            "candidate_wells": int(len(manifest_out)),
            "successful_wells": int((manifest_out["status"] == "ok").sum()),
            "failed_wells": int((manifest_out["status"] != "ok").sum()),
            "valid_samples": int(sum(np.count_nonzero(item.valid_mask) for item in control_set.controls)),
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
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("status") != "ok":
        raise ValueError(f"Unsupported or unsuccessful well-control run: {run_dir}")
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
            if set(data.files) != {"samples", "log_ai", "inline", "xline", "x_m", "y_m", "valid_mask", "metadata_json"}:
                raise ValueError(f"Unexpected well-control NPZ keys: {path}")
            if data["samples"].dtype != np.dtype("float64") or any(
                data[key].dtype != np.dtype("float64") for key in ("inline", "xline", "x_m", "y_m")
            ):
                raise ValueError(f"Well-control sample/position arrays must be float64: {path}")
            if data["log_ai"].dtype != np.dtype("float32") or data["valid_mask"].dtype != np.dtype("bool"):
                raise ValueError(f"Well-control log_ai/valid_mask dtypes must be float32/bool: {path}")
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
            }
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise ValueError(
                        f"Well-control metadata {key} mismatch in {path}: "
                        f"expected {expected!r}, got {metadata.get(key)!r}."
                    )
            if normalize_well_name(metadata.get("well_name")) != normalize_well_name(row["well_name"]):
                raise ValueError(f"Well-control manifest/NPZ well_name mismatch: {path}")
            file_axis = np.asarray(data["samples"], dtype=np.float64)
            if not np.array_equal(file_axis, axis.values):
                raise ValueError(f"Well-control SampleAxis mismatch: {path}")
            control = _control_from_arrays(
                well_name=str(metadata["well_name"]),
                sample_axis=axis,
                log_ai=np.asarray(data["log_ai"], dtype=np.float64),
                inline=np.asarray(data["inline"], dtype=np.float64),
                xline=np.asarray(data["xline"], dtype=np.float64),
                x_m=np.asarray(data["x_m"], dtype=np.float64),
                y_m=np.asarray(data["y_m"], dtype=np.float64),
                wellbore_class=str(metadata["wellbore_class"]),
                sampling_mode=str(metadata["sampling_mode"]),
                source_run_type=str(metadata["source_run_type"]),
                provenance=dict(metadata["provenance"]),
            )
            if not np.array_equal(control.valid_mask, np.asarray(data["valid_mask"], dtype=bool)):
                raise ValueError(f"Well-control NPZ valid_mask disagrees with finite values: {path}")
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

__all__ = [
    "DEPTH_SOURCE_SCHEMA",
    "LINEAR_AI_UNIT",
    "MANIFEST_COLUMNS",
    "SCHEMA_VERSION",
    "TIME_SOURCE_SCHEMA",
    "WellControl",
    "WellControlSet",
    "build_well_control_set",
    "load_well_control_set",
    "write_well_control_set",
]
