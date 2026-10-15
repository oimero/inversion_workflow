"""Step 6 well-control data contracts, source assembly, persistence, and QC."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cup.config.artifacts import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    is_consumable_contract_status,
    require_contract_fingerprint,
    resolve_artifact_path,
)
from cup.physics.numpy_backend import forward_depth, forward_time, reflectivity_from_log_ai
from cup.physics.relations import AIVelocityRelation
from cup.physics.support import complete_response_mask, relative_twt_with_gaps
from cup.seismic.forward_inputs import load_forward_inputs
from cup.seismic.geometry import SampleAxis, SurveyLineGeometry
from cup.seismic.target_zone import TargetZone
from cup.seismic.viz import plot_well_waveform_qc
from cup.seismic.wavelet import load_wavelet_csv
from cup.utils.io import repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.utils.masks import true_runs as _finite_runs
from cup.well.evaluation_support import (
    EvaluationSupport,
    MIN_EVALUATION_SUPPORT_SAMPLES,
    derive_evaluation_support,
    load_evaluation_support_manifest,
    write_evaluation_support_manifest,
)
from cup.well.inventory import build_file_lookup, normalize_well_name
from cup.well.gaps import fill_short_depth_gaps, fill_short_joint_gaps
from cup.well.las import StandardVpRhoLogs, load_standard_vp_rho_logs
from cup.well.tie import WELL_AUTO_TIE_SCHEMA_VERSION
from cup.well.trajectory import WellTrajectory, load_workflow_time_depth_table_csv
from wtie.optimize.similarity import normalized_xcorr
from wtie.processing import grid

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
_GAUSSIAN_TRUNCATE_SIGMA = 8.0

# Data contracts

SCHEMA_VERSION = "real_field_well_controls_v9"
WELL_TIE_SOURCE_SCHEMA = WELL_AUTO_TIE_SCHEMA_VERSION
SOURCE_RUN_TYPE = "well_auto_tie"
LINEAR_AI_UNIT = "m/s*g/cm3"
MANIFEST_COLUMNS = [
    "well_name", "well_role", "status", "reason", "source_run_type",
    "source_run_path", "source_summary_path", "source_las_path",
    "source_transform_path", "wellbore_class", "sample_domain", "sample_unit",
    "depth_basis", "sampling_mode", "n_samples", "n_valid_samples",
    "n_observed_samples", "n_interpolated_samples", "n_native_samples",
    "n_valid_native_samples", "sample_min", "sample_max", "well_npz_path",
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
    well_role: str = "trusted"
    model_grid_body_log_ai: grid.Log | None = None

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
        role = str(self.well_role).strip().casefold()
        if role not in {"trusted", "background_only"}:
            raise ValueError(f"{self.well_name}: unsupported well_role={self.well_role!r}.")
        object.__setattr__(self, "well_role", role)
        if (self.model_grid_body_log_ai is None) != (self.native.native_body_log_ai is None):
            raise ValueError(f"{self.well_name}: body curves must exist on both native and model axes.")
        if self.model_grid_body_log_ai is not None:
            body = self.model_grid_body_log_ai
            if not np.array_equal(body.basis, self.sample_axis.values) or not getattr(body, f"is_{expected_basis}"):
                raise ValueError(f"{self.well_name}: model body curve must share the canonical sample axis.")
            if not np.array_equal(np.isfinite(body.values), valid):
                raise ValueError(f"{self.well_name}: body smoothing must preserve the model valid mask.")


@dataclass(frozen=True)
class NativeWellControl:
    """Aligned filtered well log on its native vertical sampling."""

    well_name: str
    coordinates: np.ndarray
    native_filtered_log_ai: np.ndarray
    valid_mask: np.ndarray
    observed_mask: np.ndarray
    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    provenance: Mapping[str, Any]
    native_body_log_ai: np.ndarray | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        values = np.asarray(self.native_filtered_log_ai, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        observed = np.asarray(self.observed_mask, dtype=bool)
        if coordinates.ndim != 1 or coordinates.size < 2 or any(
            array.shape != coordinates.shape for array in (values, valid, observed)
        ):
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
        object.__setattr__(self, "observed_mask", observed)
        if self.native_body_log_ai is not None:
            body = np.asarray(self.native_body_log_ai, dtype=np.float64)
            if body.shape != coordinates.shape or not np.array_equal(np.isfinite(body), valid):
                raise ValueError(f"{self.well_name}: native body curve must preserve shape and valid support.")
            object.__setattr__(self, "native_body_log_ai", body)


@dataclass(frozen=True)
class WellControlSet:
    sample_axis: SampleAxis
    controls: tuple[WellControl, ...]
    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    source_run_type: str
    provenance: Mapping[str, Any]
    supervision_curve: str = "filtered"
    body_smoothing_fwhm_axis_units: float | None = None

    def __post_init__(self) -> None:
        validate_sample_axis(self.sample_axis)
        if self.supervision_curve not in {"filtered", "body"}:
            raise ValueError("supervision_curve must be 'filtered' or 'body'.")
        width = self.body_smoothing_fwhm_axis_units
        if width is not None and (isinstance(width, (bool, np.bool_)) or not np.isfinite(width) or width <= 0.0):
            raise ValueError("body_smoothing_fwhm_axis_units must be finite and positive.")
        if self.supervision_curve == "body" and width is None:
            raise ValueError("Body supervision requires a body smoothing FWHM.")
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
            if self.supervision_curve == "body" and control.well_role == "trusted" and control.native.native_body_log_ai is None:
                raise ValueError(f"{control.well_name}: body supervision requires saved body curves.")
            if not np.array_equal(control.sample_axis.values, self.sample_axis.values):
                raise ValueError(f"{control.well_name}: SampleAxis differs from WellControlSet.")
            if control.sample_axis.domain != self.sample_domain or control.sample_axis.unit != self.sample_unit:
                raise ValueError(f"{control.well_name}: domain/unit differs from WellControlSet.")
            if control.source_run_type != self.source_run_type:
                raise ValueError(f"{control.well_name}: source_run_type differs from WellControlSet.")

    @property
    def trusted_well_names(self) -> tuple[str, ...]:
        return tuple(control.well_name for control in self.controls if control.well_role == "trusted")

    @property
    def background_well_names(self) -> tuple[str, ...]:
        return tuple(control.well_name for control in self.controls if control.well_role == "background_only")

    def role_for(self, well_name: str) -> str:
        key = normalize_well_name(well_name).casefold()
        for control in self.controls:
            if normalize_well_name(control.well_name).casefold() == key:
                return control.well_role
        return "excluded"


def validate_sample_axis(axis: SampleAxis) -> None:
    values = np.asarray(axis.values, dtype=np.float64)
    if np.any(~np.isfinite(values)) or (values.size > 1 and np.any(np.diff(values) <= 0.0)):
        raise ValueError("Canonical SampleAxis must be finite and strictly increasing.")
    if values.size > 2 and not np.allclose(np.diff(values), np.diff(values)[0], rtol=1e-9, atol=1e-12):
        raise ValueError("Canonical SampleAxis must be regular.")
    expected_unit = "s" if axis.domain == "time" else "m" if axis.domain == "depth" else None
    if expected_unit is None or axis.unit != expected_unit:
        raise ValueError(f"Unsupported SampleAxis domain/unit: {axis.domain!r}/{axis.unit!r}.")


def control_from_arrays(
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
    well_role: str = "trusted",
    model_grid_body_log_ai: np.ndarray | None = None,
) -> WellControl:
    arrays = [
        np.asarray(value, dtype=np.float64).copy()
        for value in (model_grid_filtered_log_ai, inline, xline, x_m, y_m)
    ]
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    observed = np.asarray(observed_valid_mask, dtype=bool)
    if observed.shape != valid.shape:
        raise ValueError(f"{well_name}: observed_valid_mask shape differs from the model axis.")
    observed &= valid
    for value in arrays:
        value[~valid] = np.nan
    basis_type = "twt" if sample_axis.domain == "time" else "tvdss"
    log = grid.Log(
        arrays[0], sample_axis.values.copy(), basis_type,
        name="model_grid_filtered_log_ai", unit="ln(m/s*g/cm3)",
    )
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
        well_role=well_role,
        model_grid_body_log_ai=(None if model_grid_body_log_ai is None else grid.Log(
            np.asarray(model_grid_body_log_ai, dtype=np.float64), sample_axis.values.copy(), basis_type,
            name="model_grid_body_log_ai", unit="ln(m/s*g/cm3)",
        )),
    )

# Source assembly

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


def _load_summary(source_run_dir: Path, *, domain: str, depth_basis: str | None) -> tuple[Path, dict[str, Any]]:
    path = source_run_dir / "run_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if str(summary.get("schema_version") or "") != WELL_TIE_SOURCE_SCHEMA:
        raise ValueError(f"{path} schema_version must be {WELL_TIE_SOURCE_SCHEMA!r}; rebuild the upstream run.")
    if str(summary.get("sample_domain") or "") != domain:
        raise ValueError(f"Step 4 well_auto_tie domain does not match the workflow: {domain!r}.")
    expected_unit = "s" if domain == "time" else "m"
    if str(summary.get("sample_unit") or "") != expected_unit:
        raise ValueError(f"Step 4 well_auto_tie requires sample_unit={expected_unit!r} for {domain!r} data.")
    summary_basis = summary.get("depth_basis")
    if domain == "depth" and (depth_basis != "tvdss" or summary_basis != "tvdss"):
        raise ValueError("Depth well controls require source and seismic depth_basis='tvdss'.")
    if domain == "time" and summary_basis not in (None, ""):
        raise ValueError("Time source summary must not declare a depth_basis.")
    if not is_consumable_contract_status(summary.get("status")):
        raise ValueError(f"Source run is not consumable: status={summary.get('status')!r}.")
    return path, summary


def _read_ai_las(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    standard = load_standard_vp_rho_logs(path, require_observation_mask=True)
    if not np.array_equal(md, standard.logs.basis):
        raise ValueError(f"AI and observation mask MD axes differ: {path}")
    return md, filtered_log_ai, standard.observed_mask


def _log_ai_from_repaired_logs(logs: grid.LogSet) -> np.ndarray:
    ai = np.asarray(logs.Vp.values, dtype=np.float64) * np.asarray(logs.Rho.values, dtype=np.float64)
    result = np.full(ai.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ai) & (ai > 0.0)
    result[valid] = np.log(ai[valid])
    return result


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
    observed_mask: np.ndarray,
    sample_domain: str,
    depth_basis: str | None,
    provenance: Mapping[str, Any],
) -> NativeWellControl:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(native_filtered_log_ai, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    if observed.shape != coordinates.shape:
        raise ValueError(f"{well_name}: native observation mask must match coordinates.")
    support = np.isfinite(coordinates)
    coordinates = coordinates[support]
    values = values[support]
    observed = observed[support]
    if coordinates.size < 2 or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError(f"{well_name}: aligned native coordinates are not strictly increasing.")
    return NativeWellControl(
        well_name=well_name,
        coordinates=coordinates,
        native_filtered_log_ai=values,
        valid_mask=np.isfinite(values),
        observed_mask=observed,
        sample_domain=sample_domain,
        sample_unit="s" if sample_domain == "time" else "m",
        depth_basis=depth_basis,
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


def _required_artifact_path(value: Any, *, root: Path, run_dir: Path, label: str) -> Path:
    path = resolve_artifact_path(value, root=root, run_dir=run_dir)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {value!r}")
    return path


def _read_axis_mapping(path: Path, *, columns: set[str], label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _required_columns(frame, columns, path=path)
    frame = frame.loc[:, sorted(columns)].apply(pd.to_numeric, errors="coerce")
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"{label} contains missing or non-numeric mapping samples: {path}")
    values = frame.to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite mapping samples: {path}")
    return frame


def _trajectory_for_well(trace_lookup: Mapping[str, Path], well_name: str) -> WellTrajectory:
    path = trace_lookup.get(normalize_well_name(well_name))
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{well_name}: a valid project trajectory is required for a deviated well.")
    return WellTrajectory.from_petrel_trace(path).with_well_name(well_name)


def _positions_from_axis(
    *,
    well_name: str,
    wellbore_class: str,
    inventory_row: Mapping[str, Any],
    sample_axis: SampleAxis,
    md_at_sample: np.ndarray,
    line_geometry: SurveyLineGeometry,
    trace_lookup: Mapping[str, Path],
) -> tuple[list[np.ndarray], str, Path | None, dict[str, Any]]:
    axis = sample_axis.values
    md_at_sample = np.asarray(md_at_sample, dtype=np.float64)
    if md_at_sample.shape != axis.shape:
        raise ValueError(f"{well_name}: MD mapping does not match the canonical sample axis.")
    if wellbore_class == "vertical":
        values = [
            _finite_number(inventory_row.get("inline_float"), label=f"{well_name}.inline_float"),
            _finite_number(inventory_row.get("xline_float"), label=f"{well_name}.xline_float"),
            _finite_number(inventory_row.get("surface_x"), label=f"{well_name}.surface_x"),
            _finite_number(inventory_row.get("surface_y"), label=f"{well_name}.surface_y"),
        ]
        return [np.full(axis.shape, value, dtype=np.float64) for value in values], "vertical_inventory_position", None, {}
    if wellbore_class != "deviated":
        raise ValueError(f"{well_name}: unsupported/unknown wellbore_class={wellbore_class!r}.")
    trajectory_path = trace_lookup.get(normalize_well_name(well_name))
    trajectory = _trajectory_for_well(trace_lookup, well_name)
    x_m = _interp_no_extrapolation(md_at_sample, trajectory.md_m, trajectory.x_m)
    y_m = _interp_no_extrapolation(md_at_sample, trajectory.md_m, trajectory.y_m)
    inline = np.full(axis.shape, np.nan, dtype=np.float64)
    xline = np.full(axis.shape, np.nan, dtype=np.float64)
    for index in np.flatnonzero(np.isfinite(x_m) & np.isfinite(y_m)):
        try:
            inline[index], xline[index] = line_geometry.coord_to_line(float(x_m[index]), float(y_m[index]))
        except ValueError:
            continue
    return [inline, xline, x_m, y_m], "trajectory_md_mapping", trajectory_path, {
        "tvdss_source": "cup.well.trajectory.WellTrajectory",
    }


def _positions_from_trace_plan(
    *,
    sample_axis: SampleAxis,
    plan_path: Path,
) -> list[np.ndarray]:
    plan = pd.read_csv(plan_path)
    _required_columns(plan, {"twt_s", "inline_float", "xline_float", "x_m", "y_m", "survey_position"}, path=plan_path)
    plan_twt = pd.to_numeric(plan["twt_s"], errors="coerce").to_numpy(dtype=np.float64)
    if plan_twt.size < 2 or np.any(~np.isfinite(plan_twt)) or np.any(np.diff(plan_twt) <= 0.0):
        raise ValueError(f"Optimized trace plan TWT must be strictly increasing: {plan_path}")
    inside = plan["survey_position"].astype(str).str.casefold().eq("inside").to_numpy(dtype=bool)
    positions: list[np.ndarray] = []
    for name in ("inline_float", "xline_float", "x_m", "y_m"):
        values = pd.to_numeric(plan[name], errors="coerce").to_numpy(dtype=np.float64)
        values[~inside] = np.nan
        positions.append(_interp_finite_runs(sample_axis.values, plan_twt, values))
    return positions


def _make_control_from_native(
    *,
    well_name: str,
    sample_axis: SampleAxis,
    native_coordinates: np.ndarray,
    native_filtered_log_ai: np.ndarray,
    native_observed_mask: np.ndarray,
    positions: list[np.ndarray],
    wellbore_class: str,
    sampling_mode: str,
    provenance: Mapping[str, Any],
    depth_basis: str | None,
    line_geometry: SurveyLineGeometry,
    well_role: str,
) -> WellControl:
    native = _native_control(
        well_name=well_name,
        coordinates=native_coordinates,
        native_filtered_log_ai=native_filtered_log_ai,
        observed_mask=native_observed_mask,
        sample_domain=sample_axis.domain,
        depth_basis=depth_basis,
        provenance=provenance,
    )
    model_values = _interp_finite_runs(sample_axis.values, native.coordinates, native.native_filtered_log_ai)
    observed_values = _interp_finite_runs(
        sample_axis.values,
        native.coordinates,
        np.where(native.observed_mask, native.native_filtered_log_ai, np.nan),
    )
    control = control_from_arrays(
        well_name=well_name,
        sample_axis=sample_axis,
        model_grid_filtered_log_ai=model_values,
        inline=positions[0],
        xline=positions[1],
        x_m=positions[2],
        y_m=positions[3],
        observed_valid_mask=np.isfinite(observed_values),
        wellbore_class=wellbore_class,
        sampling_mode=sampling_mode,
        source_run_type=SOURCE_RUN_TYPE,
        provenance=dict(provenance),
        native=native,
        well_role=well_role,
    )
    _validate_control_geometry(control, line_geometry)
    return control


def _merge_native_context(
    control: WellControl,
    *,
    candidate_coordinates: np.ndarray,
    candidate_values: np.ndarray,
    candidate_observed: np.ndarray,
) -> NativeWellControl:
    """Merge a full source curve while preserving the existing native layer.

    The source curve is the authority for samples outside the old tie-window
    support.  The old native interval is copied as one indivisible block,
    including its NaN gaps, so a slightly different source mapping cannot
    insert samples into or alter the existing Step-6 contract.
    """

    candidate_coordinates = np.asarray(candidate_coordinates, dtype=np.float64)
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    candidate_observed = np.asarray(candidate_observed, dtype=bool)
    if candidate_coordinates.ndim != 1 or candidate_coordinates.size < 2:
        raise ValueError(f"{control.well_name}: source context has fewer than two samples.")
    if any(array.shape != candidate_coordinates.shape for array in (candidate_values, candidate_observed)):
        raise ValueError(f"{control.well_name}: source context arrays have inconsistent shapes.")
    if np.any(~np.isfinite(candidate_coordinates)) or np.any(np.diff(candidate_coordinates) <= 0.0):
        raise ValueError(f"{control.well_name}: source context coordinates must be strictly increasing.")

    old = control.native
    old_start = float(old.coordinates[0])
    old_stop = float(old.coordinates[-1])
    before = candidate_coordinates < old_start
    after = candidate_coordinates > old_stop
    coordinates = np.concatenate((candidate_coordinates[before], old.coordinates, candidate_coordinates[after]))
    values = np.concatenate((candidate_values[before], old.native_filtered_log_ai, candidate_values[after]))
    observed = np.concatenate((candidate_observed[before], old.observed_mask, candidate_observed[after]))
    return NativeWellControl(
        well_name=old.well_name,
        coordinates=coordinates,
        native_filtered_log_ai=values,
        valid_mask=np.isfinite(values),
        observed_mask=observed & np.isfinite(values),
        sample_domain=old.sample_domain,
        sample_unit=old.sample_unit,
        depth_basis=old.depth_basis,
        provenance=dict(old.provenance),
    )


def _fixed_vertical_positions(control: WellControl) -> tuple[float, float, float, float]:
    positions = (
        np.asarray(control.inline_by_sample, dtype=np.float64),
        np.asarray(control.xline_by_sample, dtype=np.float64),
        np.asarray(control.x_m_by_sample, dtype=np.float64),
        np.asarray(control.y_m_by_sample, dtype=np.float64),
    )
    result: list[float] = []
    for values in positions:
        finite = values[np.isfinite(values)]
        if finite.size == 0 or not np.allclose(finite, finite[0], rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{control.well_name}: vertical context extension requires fixed well positions.")
        result.append(float(finite[0]))
    return tuple(result)  # type: ignore[return-value]


def _extend_vertical_control(
    control: WellControl,
    *,
    candidate_coordinates: np.ndarray,
    candidate_values: np.ndarray,
    candidate_observed: np.ndarray,
    context_provenance: Mapping[str, Any],
) -> WellControl:
    """Add finite source runs outside the old support to a trusted vertical well."""

    if str(control.wellbore_class).casefold() != "vertical":
        return control
    native = _merge_native_context(
        control,
        candidate_coordinates=candidate_coordinates,
        candidate_values=candidate_values,
        candidate_observed=candidate_observed,
    )
    axis = np.asarray(control.sample_axis.values, dtype=np.float64)
    model_from_native = _interp_finite_runs(axis, native.coordinates, native.native_filtered_log_ai)
    observed_from_native = _interp_finite_runs(
        axis,
        native.coordinates,
        np.where(native.observed_mask, native.native_filtered_log_ai, np.nan),
    )
    old_values = np.asarray(control.model_grid_filtered_log_ai.values, dtype=np.float64)
    old_valid = np.asarray(control.valid_mask, dtype=bool)
    old_observed = np.asarray(control.observed_valid_mask, dtype=bool)
    model_values = model_from_native.copy()
    model_values[old_valid] = old_values[old_valid]
    added = np.isfinite(model_values) & ~old_valid
    inline = np.asarray(control.inline_by_sample, dtype=np.float64).copy()
    xline = np.asarray(control.xline_by_sample, dtype=np.float64).copy()
    x_m = np.asarray(control.x_m_by_sample, dtype=np.float64).copy()
    y_m = np.asarray(control.y_m_by_sample, dtype=np.float64).copy()
    if np.any(added):
        fixed = _fixed_vertical_positions(control)
        for values, value in zip((inline, xline, x_m, y_m), fixed):
            values[added] = value
    observed = old_observed.copy()
    observed[~old_valid] = np.isfinite(observed_from_native[~old_valid])
    provenance = dict(control.provenance)
    provenance["context_extension"] = dict(context_provenance)
    return control_from_arrays(
        well_name=control.well_name,
        sample_axis=control.sample_axis,
        model_grid_filtered_log_ai=model_values,
        inline=inline,
        xline=xline,
        x_m=x_m,
        y_m=y_m,
        observed_valid_mask=observed,
        wellbore_class=control.wellbore_class,
        sampling_mode=control.sampling_mode,
        source_run_type=control.source_run_type,
        provenance=provenance,
        native=native,
        well_role=control.well_role,
    )


def _shifted_source_time_coordinates(
    source_md: np.ndarray,
    *,
    source_tdt: grid.TimeDepthTable,
    shift_ms: float,
) -> np.ndarray:
    if not source_tdt.is_md_domain:
        raise ValueError("Complete time source TDT must use MD coordinates.")
    source_md = np.asarray(source_md, dtype=np.float64)
    source_twt = _interp_no_extrapolation(source_md, np.asarray(source_tdt.md), np.asarray(source_tdt.twt))
    return source_twt + float(shift_ms) / 1000.0


def _shifted_source_depth_coordinates(
    source_md: np.ndarray,
    *,
    source_mapping: pd.DataFrame,
    kb_m: float,
    shift_ms: float,
) -> np.ndarray:
    source_tvdss = source_mapping["tvdss_m"].to_numpy(dtype=np.float64)
    source_twt = source_mapping["twt_s"].to_numpy(dtype=np.float64)
    if np.any(np.diff(source_tvdss) <= 0.0) or np.any(np.diff(source_twt) <= 0.0):
        raise ValueError("Complete depth source mapping must increase in TVDSS and TWT.")
    source_depth = np.asarray(source_md, dtype=np.float64) - float(kb_m)
    inside_depth = (source_depth >= source_tvdss[0]) & (source_depth <= source_tvdss[-1])
    source_t0 = np.full(source_depth.shape, np.nan, dtype=np.float64)
    source_t0[inside_depth] = np.interp(source_depth[inside_depth], source_tvdss, source_twt)
    shifted_twt = source_t0 + float(shift_ms) / 1000.0
    inside_time = inside_depth & (shifted_twt >= source_twt[0]) & (shifted_twt <= source_twt[-1])
    result = np.full(source_depth.shape, np.nan, dtype=np.float64)
    result[inside_time] = np.interp(shifted_twt[inside_time], source_twt, source_tvdss)
    return result


def _time_control(
    *,
    source_row: Mapping[str, Any],
    inventory_row: Mapping[str, Any],
    sample_axis: SampleAxis,
    source_run_dir: Path,
    repo_root: Path,
    line_geometry: SurveyLineGeometry,
    trace_lookup: Mapping[str, Path],
    well_role: str,
    native_las_path: Path | None = None,
    tdt_path: Path | None = None,
    trace_plan_path: Path | None = None,
    source_las_role: str = "filtered",
    max_short_gap_s: float | None = None,
) -> WellControl:
    well_name = str(source_row["well_name"]).strip()
    tdt_path = tdt_path or _required_artifact_path(
        source_row.get("optimized_tdt_file"), root=repo_root, run_dir=source_run_dir, label=f"{well_name} optimized TDT"
    )
    table = load_workflow_time_depth_table_csv(tdt_path)
    if not table.is_md_domain:
        raise ValueError(f"{well_name}: time-depth mapping must use MD domain.")
    table_twt = np.asarray(table.twt, dtype=np.float64)
    table_md = np.asarray(table.md, dtype=np.float64)
    if native_las_path is None:
        native_las_path = _required_artifact_path(
            source_row.get("filtered_las_file"), root=repo_root, run_dir=source_run_dir, label=f"{well_name} filtered LAS"
        )
    if source_las_role == "preprocessed_background":
        if max_short_gap_s is None:
            raise ValueError(f"{well_name}: background repair requires the Step 4 short-gap threshold.")
        standard = load_standard_vp_rho_logs(native_las_path)
        repaired, _ = fill_short_joint_gaps(standard, table, max_short_gap_s=max_short_gap_s)
        native_md = np.asarray(repaired.basis, dtype=np.float64)
        native_log_ai = _log_ai_from_repaired_logs(repaired)
        native_observed = standard.observed_mask
    else:
        native_md, native_log_ai, native_observed = _read_ai_las(native_las_path)
    wellbore_class = str(inventory_row.get("wellbore_class") or "unknown").strip().casefold()
    md_at_sample = _interp_no_extrapolation(sample_axis.values, table_twt, table_md)
    positions, sampling_mode, transform_path, transform_metadata = _positions_from_axis(
        well_name=well_name,
        wellbore_class=wellbore_class,
        inventory_row=inventory_row,
        sample_axis=sample_axis,
        md_at_sample=md_at_sample,
        line_geometry=line_geometry,
        trace_lookup=trace_lookup,
    )
    if trace_plan_path is not None and trace_plan_path.is_file() and wellbore_class == "deviated":
        positions = _positions_from_trace_plan(sample_axis=sample_axis, plan_path=trace_plan_path)
        sampling_mode = "optimized_trace_plan"
        transform_path = trace_plan_path
    native_twt = _interp_no_extrapolation(native_md, table_md, table_twt)
    control = _make_control_from_native(
        well_name=well_name,
        sample_axis=sample_axis,
        native_coordinates=native_twt,
        native_filtered_log_ai=native_log_ai,
        native_observed_mask=native_observed,
        positions=positions,
        wellbore_class=wellbore_class,
        sampling_mode=sampling_mode,
        provenance={
            "source_las_path": str(native_las_path),
            "native_source_las_path": str(native_las_path),
            "source_las_role": source_las_role,
            "gap_policy": "persist_repaired_short_gaps",
            "source_transform_path": str(transform_path or tdt_path),
            "optimized_tdt_path": str(tdt_path),
            "source_vertical_coordinate": "md_m",
            "aligned_vertical_coordinate": "twt_s",
            **transform_metadata,
        },
        depth_basis=None,
        line_geometry=line_geometry,
        well_role=well_role,
    )
    if source_las_role != "filtered" or well_role != "trusted":
        return control

    source_tdt_path = _required_artifact_path(
        source_row.get("source_tdt_file"),
        root=repo_root,
        run_dir=source_run_dir,
        label=(
            f"{well_name} complete source TDT; rerun Step 4 to produce "
            "source_tdt_file"
        ),
    )
    source_tdt = load_workflow_time_depth_table_csv(source_tdt_path)
    shift_ms = _finite_number(
        source_row.get("best_table_shift_ms"),
        label=f"{well_name}.best_table_shift_ms",
    )
    candidate_coordinates = _shifted_source_time_coordinates(
        native_md,
        source_tdt=source_tdt,
        shift_ms=shift_ms,
    )
    support = np.isfinite(candidate_coordinates)
    if int(np.count_nonzero(support)) < 2:
        raise ValueError(f"{well_name}: complete source TDT has fewer than two source LAS samples in support.")
    source_provenance = {
        "method": "complete_source_tdt_plus_best_constant_table_shift",
        "source_tdt_path": str(source_tdt_path),
        "optimized_tdt_path": str(tdt_path),
        "best_table_shift_ms": float(shift_ms),
        "no_extrapolation": True,
        "original_model_grid_support_preserved": True,
    }
    return _extend_vertical_control(
        control,
        candidate_coordinates=candidate_coordinates[support],
        candidate_values=native_log_ai[support],
        candidate_observed=native_observed[support],
        context_provenance=source_provenance,
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
    well_role: str,
    native_las_path: Path | None = None,
    aligned_depth_path: Path | None = None,
    source_depth_mapping_path: Path | None = None,
    source_las_role: str = "filtered",
) -> WellControl:
    well_name = str(source_row["well_name"]).strip()
    native_las_path = native_las_path or _required_artifact_path(
        source_row.get("filtered_las_file"), root=repo_root, run_dir=source_run_dir, label=f"{well_name} filtered LAS"
    )
    aligned_depth_path = aligned_depth_path or _required_artifact_path(
        source_row.get("aligned_depth_file"), root=repo_root, run_dir=source_run_dir, label=f"{well_name} aligned depth mapping"
    )
    source_depth_mapping_path = source_depth_mapping_path or _required_artifact_path(
        source_row.get("source_depth_mapping_file"), root=repo_root, run_dir=source_run_dir, label=f"{well_name} source depth mapping"
    )
    aligned = _read_axis_mapping(aligned_depth_path, columns={"md_m", "tvdss_m"}, label=f"{well_name} aligned depth mapping")
    source_mapping = _read_axis_mapping(
        source_depth_mapping_path,
        columns={"tvdss_m", "twt_s"},
        label=f"{well_name} source depth mapping",
    )
    aligned_md = aligned["md_m"].to_numpy(dtype=np.float64)
    aligned_tvdss = aligned["tvdss_m"].to_numpy(dtype=np.float64)
    if np.any(np.diff(aligned_md) <= 0.0) or np.any(np.diff(aligned_tvdss) <= 0.0):
        raise ValueError(f"{well_name}: aligned depth mapping must be increasing in MD and TVDSS.")
    native_md, native_log_ai, native_observed = _read_ai_las(native_las_path)
    native_tvdss = _interp_no_extrapolation(native_md, aligned_md, aligned_tvdss)
    md_at_sample = _interp_no_extrapolation(sample_axis.values, aligned_tvdss, aligned_md)
    wellbore_class = str(inventory_row.get("wellbore_class") or "unknown").strip().casefold()
    positions, sampling_mode, transform_path, transform_metadata = _positions_from_axis(
        well_name=well_name,
        wellbore_class=wellbore_class,
        inventory_row=inventory_row,
        sample_axis=sample_axis,
        md_at_sample=md_at_sample,
        line_geometry=line_geometry,
        trace_lookup=trace_lookup,
    )
    control = _make_control_from_native(
        well_name=well_name,
        sample_axis=sample_axis,
        native_coordinates=native_tvdss,
        native_filtered_log_ai=native_log_ai,
        native_observed_mask=native_observed,
        positions=positions,
        wellbore_class=wellbore_class,
        sampling_mode="aligned_depth_" + sampling_mode,
        provenance={
            "source_las_path": str(native_las_path),
            "native_source_las_path": str(native_las_path),
            "source_las_role": source_las_role,
            "gap_policy": "persist_repaired_short_gaps",
            "source_transform_path": str(aligned_depth_path),
            "aligned_depth_path": str(aligned_depth_path),
            "source_depth_mapping_path": str(source_depth_mapping_path),
            "source_vertical_coordinate": "md_m",
            "aligned_vertical_coordinate": "tvdss_m",
            "position_transform_path": "" if transform_path is None else str(transform_path),
            **transform_metadata,
        },
        depth_basis="tvdss",
        line_geometry=line_geometry,
        well_role=well_role,
    )
    if source_las_role != "filtered" or well_role != "trusted" or wellbore_class != "vertical":
        return control

    kb_m = _finite_number(inventory_row.get("kb_m"), label=f"{well_name}.kb_m")
    shift_ms = _finite_number(
        source_row.get("best_table_shift_ms"),
        label=f"{well_name}.best_table_shift_ms",
    )
    coarse_shift_ms = _finite_number(
        source_row.get("coarse_total_shift_ms", 0.0),
        label=f"{well_name}.coarse_total_shift_ms",
    )
    candidate_coordinates = _shifted_source_depth_coordinates(
        native_md,
        source_mapping=source_mapping,
        kb_m=kb_m,
        shift_ms=coarse_shift_ms + shift_ms,
    )
    support = np.isfinite(candidate_coordinates)
    if int(np.count_nonzero(support)) < 2:
        raise ValueError(f"{well_name}: complete source depth mapping has fewer than two source LAS samples in support.")
    source_provenance = {
        "method": "source_t0_plus_coarse_and_fine_table_shifts_then_source_mapping_inverse",
        "source_depth_mapping_path": str(source_depth_mapping_path),
        "optimized_tdt_path": str(source_row.get("optimized_tdt_file") or ""),
        "best_table_shift_ms": float(shift_ms),
        "coarse_total_shift_ms": float(coarse_shift_ms),
        "kb_m": float(kb_m),
        "no_extrapolation": True,
        "original_model_grid_support_preserved": True,
    }
    return _extend_vertical_control(
        control,
        candidate_coordinates=candidate_coordinates[support],
        candidate_values=native_log_ai[support],
        candidate_observed=native_observed[support],
        context_provenance=source_provenance,
    )


def _background_depth_control(
    *,
    source_row: Mapping[str, Any],
    inventory_row: Mapping[str, Any],
    sample_axis: SampleAxis,
    line_geometry: SurveyLineGeometry,
    preprocess_dir: Path,
    repo_root: Path,
    trace_lookup: Mapping[str, Path],
    well_role: str,
    max_short_gap_s: float,
) -> WellControl:
    well_name = str(source_row["well_name"]).strip()
    las_value = source_row.get("preprocessed_las")
    native_las_path = resolve_artifact_path(las_value, root=repo_root, run_dir=preprocess_dir)
    if native_las_path is None:
        native_las_path = preprocess_dir / "preprocessed_las" / f"{sanitize_filename(well_name)}.las"
    if not native_las_path.is_file():
        raise FileNotFoundError(f"{well_name}: Step 3 preprocessed LAS is missing: {native_las_path}")
    standard = load_standard_vp_rho_logs(native_las_path)
    native_md = np.asarray(standard.logs.basis, dtype=np.float64)
    wellbore_class = str(inventory_row.get("wellbore_class") or "unknown").strip().casefold()
    if wellbore_class == "vertical":
        kb_m = _finite_number(inventory_row.get("kb_m"), label=f"{well_name}.kb_m")
        native_tvdss = native_md - kb_m
        md_at_sample = sample_axis.values + kb_m
        mode = "background_vertical_md_minus_kb"
        transform_metadata = {"kb_m": kb_m, "tvdss_formula": "md_m-kb_m"}
    else:
        trajectory = _trajectory_for_well(trace_lookup, well_name)
        native_tvdss = _interp_no_extrapolation(native_md, trajectory.md_m, trajectory.tvdss_m)
        md_at_sample = _interp_no_extrapolation(sample_axis.values, trajectory.tvdss_m, trajectory.md_m)
        mode = "background_trajectory_tvdss"
        transform_metadata = {"tvdss_source": "cup.well.trajectory.WellTrajectory"}
    geometry_support = np.isfinite(native_tvdss)
    geometry_logs = grid.LogSet({
        key: grid.Log(
            np.where(geometry_support, values, np.nan), native_md, "md", unit=unit, allow_nan=True,
        )
        for key, values, unit in (
            ("Vp", standard.logs.Vp.values, "m/s"),
            ("Rho", standard.logs.Rho.values, "g/cm3"),
        )
    })
    repaired, _ = fill_short_depth_gaps(
        StandardVpRhoLogs(geometry_logs, standard.observed_mask),
        native_tvdss, max_short_gap_s=max_short_gap_s,
    )
    native_log_ai = _log_ai_from_repaired_logs(repaired)
    positions, _position_mode, transform_path, position_metadata = _positions_from_axis(
        well_name=well_name,
        wellbore_class=wellbore_class,
        inventory_row=inventory_row,
        sample_axis=sample_axis,
        md_at_sample=md_at_sample,
        line_geometry=line_geometry,
        trace_lookup=trace_lookup,
    )
    return _make_control_from_native(
        well_name=well_name,
        sample_axis=sample_axis,
        native_coordinates=native_tvdss,
        native_filtered_log_ai=native_log_ai,
        native_observed_mask=standard.observed_mask,
        positions=positions,
        wellbore_class=wellbore_class,
        sampling_mode=mode,
        provenance={
            "source_las_path": str(native_las_path),
            "native_source_las_path": str(native_las_path),
            "source_las_role": "preprocessed_background",
            "gap_policy": "persist_repaired_short_gaps",
            "max_short_log_gap_s": max_short_gap_s,
            "source_transform_path": "" if transform_path is None else str(transform_path),
            "source_vertical_coordinate": "md_m",
            "aligned_vertical_coordinate": "tvdss_m",
            **transform_metadata,
            **position_metadata,
        },
        depth_basis="tvdss",
        line_geometry=line_geometry,
        well_role=well_role,
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
    well_roles: Mapping[str, str] | None = None,
) -> tuple[WellControlSet, pd.DataFrame]:
    """Build canonical time/depth controls from one Step-4 run.

    Trusted wells consume the Step-4 filtered LAS and its domain-specific
    alignment artifacts.  Background-only wells consume the Step-3 standard
    preprocessed LAS and explicit project TDT/trajectory assets.  Both paths
    produce the same ``well_auto_tie`` control contract.
    """

    allowed_config = {
        "source_runs", "well_trace_dir", "time_depth_dir", "well_roles",
        "well_role_reasons", "central_trusted_well_names",
        "supervision_curve", "body_smoothing",
    }
    unknown = set(config) - allowed_config
    if unknown:
        raise ValueError(f"real_field_well_controls has unknown settings: {sorted(unknown)}.")
    supervision_curve = config.get("supervision_curve", "filtered")
    if supervision_curve not in {"filtered", "body"}:
        raise ValueError("real_field_well_controls.supervision_curve must be 'filtered' or 'body'.")
    body_config = config.get("body_smoothing")
    body_fwhm = None
    if body_config is not None:
        if not isinstance(body_config, Mapping) or set(body_config) != {"fwhm_axis_units"}:
            raise ValueError("body_smoothing must contain exactly fwhm_axis_units.")
        value = body_config["fwhm_axis_units"]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("body_smoothing.fwhm_axis_units must be a finite positive number.")
        body_fwhm = _finite_number(value, label="body_smoothing.fwhm_axis_units")
        if body_fwhm <= 0.0:
            raise ValueError("body_smoothing.fwhm_axis_units must be positive.")
    if supervision_curve == "body" and body_fwhm is None:
        raise ValueError("Body supervision requires real_field_well_controls.body_smoothing.")
    source_runs = config.get("source_runs")
    if not isinstance(source_runs, Mapping):
        raise ValueError("real_field_well_controls.source_runs must be a mapping.")
    required_source_keys = {"well_auto_tie_dir", "well_preprocess_dir", "well_inventory_dir"}
    missing_source_keys = sorted(required_source_keys - set(source_runs))
    if missing_source_keys:
        raise ValueError(f"real_field_well_controls.source_runs is missing {missing_source_keys}.")
    validate_sample_axis(sample_axis)
    seismic_path = Path(seismic_path)
    if not seismic_path.is_file():
        raise FileNotFoundError(seismic_path)
    source_run_dir = resolve_relative_path(str(source_runs.get("well_auto_tie_dir") or ""), root=repo_root)
    if not source_run_dir.is_dir():
        raise FileNotFoundError(source_run_dir)
    summary_path, source_summary = _load_summary(source_run_dir, domain=domain, depth_basis=depth_basis)
    if source_summary.get("gap_policy") != "persist_repaired_short_gaps":
        raise ValueError("Step 4 must persist repaired short gaps in its filtered curves.")
    max_short_gap_s = _finite_number(
        source_summary.get("max_short_log_gap_s"), label="Step 4 max_short_log_gap_s",
    )
    if max_short_gap_s < 0.0:
        raise ValueError("Step 4 max_short_log_gap_s must be non-negative seconds.")
    preprocess_dir = resolve_relative_path(str(source_runs.get("well_preprocess_dir") or ""), root=repo_root)
    if not preprocess_dir.is_dir():
        raise FileNotFoundError(preprocess_dir)
    inventory_dir = resolve_relative_path(str(source_runs.get("well_inventory_dir") or ""), root=repo_root)
    inventory_path = inventory_dir if inventory_dir.is_file() else inventory_dir / "well_inventory.csv"
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
    metrics_path = source_run_dir / "well_tie_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    recorded_metrics = dict(source_summary.get("paths") or {}).get("well_tie_metrics")
    if not recorded_metrics:
        raise ValueError("Step 4 run_summary.paths.well_tie_metrics is required.")
    recorded_metrics_path = resolve_artifact_path(recorded_metrics, root=repo_root, run_dir=source_run_dir)
    if recorded_metrics_path is None or recorded_metrics_path.resolve() != metrics_path.resolve():
        raise ValueError("Source run summary metrics path does not match the selected source run.")
    metrics = pd.read_csv(metrics_path)
    _required_columns(metrics, {"well_name", "tie_status", "filtered_las_file", "optimized_tdt_file"}, path=metrics_path)
    if domain == "time":
        if "source_tdt_file" not in metrics:
            raise ValueError("Time well controls require source_tdt_file; rerun Step 4 to save the complete source TDT.")
        _required_columns(metrics, {"source_tdt_file", "best_table_shift_ms"}, path=metrics_path)
    elif domain == "depth":
        _required_columns(metrics, {"best_table_shift_ms"}, path=metrics_path)
    metric_names = [normalize_well_name(value) for value in metrics["well_name"]]
    if any(name.casefold() in invalid_names for name in metric_names) or len(metric_names) != len(set(metric_names)):
        raise ValueError(f"Source metrics well names must be non-empty and unique after normalization: {metrics_path}")

    trace_dir = resolve_relative_path(str(config.get("well_trace_dir") or ""), root=data_root)
    trace_lookup = build_file_lookup(trace_dir.iterdir(), asset_label=str(trace_dir)) if trace_dir.is_dir() else {}
    time_depth_dir = resolve_relative_path(str(config.get("time_depth_dir") or ""), root=data_root)
    time_depth_lookup = build_file_lookup(time_depth_dir.iterdir(), asset_label=str(time_depth_dir)) if time_depth_dir.is_dir() else {}
    preprocess_status_path = preprocess_dir / "well_preprocess_status.csv"
    if not preprocess_status_path.is_file():
        raise FileNotFoundError(preprocess_status_path)
    preprocess = pd.read_csv(preprocess_status_path)
    _required_columns(preprocess, {"well_name", "preprocess_status", "preprocessed_las"}, path=preprocess_status_path)
    preprocess_index = {normalize_well_name(row["well_name"]): row for _, row in preprocess.iterrows()}
    metric_index = {normalize_well_name(row["well_name"]): row for _, row in metrics.iterrows()}
    if len(metric_index) != len(metrics):
        raise ValueError(f"Source metrics well names must be unique after normalization: {metrics_path}")
    role_source = well_roles if well_roles is not None else config.get("well_roles")
    if role_source is not None and not isinstance(role_source, Mapping):
        raise ValueError("real_field_well_controls.well_roles must be a mapping.")
    role_index = {normalize_well_name(name): str(role).strip().casefold() for name, role in (role_source or {}).items()}
    role_reason_source = config.get("well_role_reasons") or {}
    if not isinstance(role_reason_source, Mapping):
        raise ValueError("real_field_well_controls.well_role_reasons must be a mapping.")
    role_reason_index = {
        normalize_well_name(name): str(reason).strip()
        for name, reason in role_reason_source.items()
    }
    controls: list[WellControl] = []
    rows: list[dict[str, Any]] = []
    for _, inventory_row in inventory.iterrows():
        well_name = str(inventory_row["well_name"]).strip()
        key = normalize_well_name(well_name)
        role = role_index.get(key, "trusted")
        if role not in {"trusted", "background_only", "excluded"}:
            raise ValueError(f"{well_name}: unsupported central well role {role!r}.")
        base = {
            "well_name": well_name,
            "well_role": role,
            "status": "excluded" if role == "excluded" else "failed",
            "reason": role_reason_index.get(key, "well_role_excluded") if role == "excluded" else "",
            "source_run_type": SOURCE_RUN_TYPE,
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
        if role == "excluded":
            rows.append(base)
            continue
        source_row = metric_index.get(key)
        try:
            if role == "trusted":
                if source_row is None or str(source_row.get("tie_status") or "").casefold() != "success":
                    raise ValueError("Step 4 well_auto_tie has no successful result for this trusted well.")
                if domain == "time":
                    plan_path = resolve_artifact_path(
                        source_row.get("optimized_trace_sample_plan_file"), root=repo_root, run_dir=source_run_dir
                    )
                    control = _time_control(
                        source_row=source_row,
                        inventory_row=inventory_row,
                        sample_axis=sample_axis,
                        source_run_dir=source_run_dir,
                        repo_root=repo_root,
                        line_geometry=line_geometry,
                        trace_lookup=trace_lookup,
                        well_role=role,
                        trace_plan_path=plan_path,
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
                        well_role=role,
                    )
            else:
                preprocess_row = preprocess_index.get(key)
                if preprocess_row is None or str(preprocess_row.get("preprocess_status") or "").casefold() != "passed":
                    raise ValueError("Step 3 preprocessing has no passed record for this background-only well.")
                if domain == "time":
                    tdt_path = time_depth_lookup.get(key)
                    if tdt_path is None:
                        raise FileNotFoundError(f"{well_name}: background-only time-domain well lacks an MD TDT.")
                    preprocessed_path = resolve_artifact_path(
                        preprocess_row.get("preprocessed_las"), root=repo_root, run_dir=preprocess_dir
                    )
                    if preprocessed_path is None:
                        preprocessed_path = preprocess_dir / "preprocessed_las" / f"{sanitize_filename(well_name)}.las"
                    control = _time_control(
                        source_row={"well_name": well_name, "optimized_tdt_file": str(tdt_path), "filtered_las_file": str(preprocessed_path or "")},
                        inventory_row=inventory_row,
                        sample_axis=sample_axis,
                        source_run_dir=preprocess_dir,
                        repo_root=repo_root,
                        line_geometry=line_geometry,
                        trace_lookup=trace_lookup,
                        well_role=role,
                        native_las_path=preprocessed_path,
                        tdt_path=tdt_path,
                        source_las_role="preprocessed_background",
                        max_short_gap_s=max_short_gap_s,
                    )
                else:
                    control = _background_depth_control(
                        source_row={"well_name": well_name, **dict(preprocess_row)},
                        inventory_row=inventory_row,
                        sample_axis=sample_axis,
                        line_geometry=line_geometry,
                        preprocess_dir=preprocess_dir,
                        repo_root=repo_root,
                        trace_lookup=trace_lookup,
                        well_role=role,
                        max_short_gap_s=max_short_gap_s,
                    )
        except (FileNotFoundError, ValueError) as exc:
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
    failed_required = [row for row in rows if row["well_role"] != "excluded" and row["status"] != "ok"]
    if failed_required:
        reasons = "; ".join(f"{row['well_name']} ({row['well_role']}): {row['reason']}" for row in failed_required)
        raise ValueError(f"Canonical well controls are missing required wells: {reasons}")
    if not controls:
        raise ValueError("No non-excluded canonical well controls were built.")
    if body_fwhm is not None:
        controls = [_with_body_curve(control, fwhm_axis_units=body_fwhm) for control in controls]
    control_set = WellControlSet(
        sample_axis=sample_axis,
        controls=tuple(controls),
        sample_domain=sample_axis.domain,
        sample_unit=sample_axis.unit,
        depth_basis=depth_basis,
        source_run_type=SOURCE_RUN_TYPE,
        supervision_curve=supervision_curve,
        body_smoothing_fwhm_axis_units=body_fwhm,
        provenance={
            "source_run_path": str(source_run_dir),
            "source_summary_path": str(summary_path),
            "metrics_path": str(metrics_path),
            "well_inventory_path": str(inventory_path),
            "well_preprocess_path": str(preprocess_status_path),
            "well_roles": {control.well_name: control.well_role for control in controls},
            "central_trusted_well_names": config.get("central_trusted_well_names"),
            "target_seismic_path": str(seismic_path),
            "native_source_role": "role_specific_source",
            "gap_policy": "persist_repaired_short_gaps",
            "max_short_log_gap_s": max_short_gap_s,
            "input_contracts": {
                "source_run": {"path": repo_relative_path(summary_path, root=repo_root)},
                "well_inventory": {"path": repo_relative_path(inventory_path, root=repo_root)},
            },
        },
    )
    return control_set, pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS)

# Quality control

QC_SCHEMA_VERSION = "real_field_well_control_qc_v4"

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


def _complete_forward_mask(
    log_ai: np.ndarray,
    axis: np.ndarray,
    wavelet_time_s: np.ndarray,
    *,
    sample_domain: str,
    relation: AIVelocityRelation | None = None,
) -> np.ndarray:
    values = np.asarray(log_ai, dtype=np.float64)
    valid = np.isfinite(values)
    if sample_domain == "time":
        times = axis
    elif sample_domain == "depth":
        if relation is None:
            raise ValueError("Depth forward support requires an AI--velocity relation.")
        velocity = np.full(values.shape, np.nan, dtype=np.float64)
        velocity[valid] = relation.velocity_from_ai(np.exp(values[valid]))
        times = relative_twt_with_gaps(velocity, axis, valid)
    else:
        raise ValueError(f"Unsupported forward sample domain: {sample_domain!r}")
    return complete_response_mask(times, valid, wavelet_time_s)


def _waveform_evaluation_support(
    control: WellControl,
    curve_support: EvaluationSupport,
    *,
    real: np.ndarray,
    full_forward: np.ndarray,
    body_forward: np.ndarray,
    full_response_mask: np.ndarray,
    body_response_mask: np.ndarray,
) -> EvaluationSupport:
    """Choose complete waveform response within the fixed curve interval."""
    return build_evaluation_support(
        control,
        target_interval=(curve_support.support_start, curve_support.support_stop),
        observed_support=np.isfinite(real),
        well_curve_support=control.valid_mask,
        additional_support=(
            np.isfinite(full_forward), np.isfinite(body_forward),
            full_response_mask, body_response_mask,
        ),
    )


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


def build_evaluation_support(
    control: WellControl,
    *,
    target_interval: tuple[float, float],
    observed_support: np.ndarray,
    well_curve_support: np.ndarray,
    additional_support: tuple[np.ndarray, ...] = (),
    minimum_samples: int = MIN_EVALUATION_SUPPORT_SAMPLES,
) -> EvaluationSupport:
    """Build the canonical fixed support for one well.

    This small adapter is also the entry point for benchmark/pseudo-well
    producers: they provide an explicit target interval and masks, while Step
    6 supplies the interpreted interval and finite forward masks below.
    """

    return derive_evaluation_support(
        well_name=control.well_name,
        sample_axis=control.sample_axis,
        target_interval=target_interval,
        observed_support=observed_support,
        well_curve_support=well_curve_support,
        additional_support=additional_support,
        minimum_samples=minimum_samples,
    )


def build_target_zone_evaluation_support(
    control: WellControl,
    target_zone: TargetZone,
    *,
    observed_support: np.ndarray,
    well_curve_support: np.ndarray,
    additional_support: tuple[np.ndarray, ...] = (),
    minimum_samples: int = MIN_EVALUATION_SUPPORT_SAMPLES,
) -> EvaluationSupport:
    """Build support from the ordered geological target-zone markers."""

    markers = horizon_markers_along_control(control, target_zone)
    return build_evaluation_support(
        control,
        target_interval=(float(markers[0][0]), float(markers[-1][0])),
        observed_support=observed_support,
        well_curve_support=well_curve_support,
        additional_support=additional_support,
        minimum_samples=minimum_samples,
    )


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


def _validate_gaussian_inputs(
    values: Any,
    coordinates: Any,
    fwhm: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Validate values and physical coordinates for Gaussian body smoothing."""

    array = np.asarray(values)
    coordinate_array = np.asarray(coordinates)
    if not np.issubdtype(array.dtype, np.floating) or array.ndim < 1:
        raise TypeError("values must be a floating array with shape [..., samples].")
    if not np.issubdtype(coordinate_array.dtype, np.floating):
        raise TypeError("coordinates must have a floating dtype.")
    shared_coordinates = coordinate_array.ndim == 1
    if shared_coordinates:
        coordinate_array = np.broadcast_to(coordinate_array, array.shape)
    if coordinate_array.shape != array.shape:
        raise ValueError("coordinates must have shape (samples,) or match values.")
    if np.any(~np.isfinite(array)) or np.any(~np.isfinite(coordinate_array)):
        raise ValueError("values and coordinates must contain only finite values.")
    flat_coordinates = coordinate_array.reshape((-1, coordinate_array.shape[-1]))
    if np.any(np.diff(flat_coordinates, axis=-1) <= 0.0):
        raise ValueError("coordinates must be strictly increasing along every trace.")
    width = float(fwhm)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("fwhm must be finite and positive.")
    return array, coordinate_array, width, shared_coordinates


def _gaussian_smooth(
    values: Any,
    coordinates: Any,
    *,
    fwhm: float,
) -> np.ndarray:
    """Apply a normalized Gaussian defined by physical coordinate distance."""

    array, coordinate_array, width, shared_coordinates = _validate_gaussian_inputs(
        values,
        coordinates,
        fwhm,
    )
    sigma = width * _FWHM_TO_SIGMA
    flat_values = array.reshape((-1, array.shape[-1]))
    flat_coordinates = coordinate_array.reshape((-1, coordinate_array.shape[-1]))
    output = np.empty(flat_values.shape, dtype=np.result_type(array.dtype, np.float64))
    radius = _GAUSSIAN_TRUNCATE_SIGMA * sigma
    if shared_coordinates:
        axis = flat_coordinates[0]
        starts = np.searchsorted(axis, axis - radius, side="left")
        stops = np.searchsorted(axis, axis + radius, side="right")
        for index, (start, stop) in enumerate(zip(starts, stops)):
            weights = np.exp(-0.5 * np.square((axis[start:stop] - axis[index]) / sigma))
            weights /= np.sum(weights)
            output[:, index] = flat_values[:, start:stop] @ weights
    else:
        for trace_index, (trace_values, axis) in enumerate(zip(flat_values, flat_coordinates)):
            starts = np.searchsorted(axis, axis - radius, side="left")
            stops = np.searchsorted(axis, axis + radius, side="right")
            for sample_index, (start, stop) in enumerate(zip(starts, stops)):
                weights = np.exp(
                    -0.5 * np.square((axis[start:stop] - axis[sample_index]) / sigma)
                )
                weights /= np.sum(weights)
                output[trace_index, sample_index] = trace_values[start:stop] @ weights
    return output.reshape(array.shape).astype(array.dtype, copy=False)


def _gaussian_smooth_finite_runs(
    values: Any,
    coordinates: Any,
    *,
    fwhm: float,
) -> np.ndarray:
    """Smooth contiguous finite runs without crossing missing-log gaps."""

    array = np.asarray(values)
    coordinate_array = np.asarray(coordinates)
    if array.ndim != 1 or coordinate_array.shape != array.shape:
        raise ValueError("finite-run smoothing requires matching one-dimensional arrays.")
    if not np.issubdtype(array.dtype, np.floating) or not np.issubdtype(
        coordinate_array.dtype,
        np.floating,
    ):
        raise TypeError("values and coordinates must have floating dtypes.")
    if np.any(~np.isfinite(coordinate_array)) or np.any(np.diff(coordinate_array) <= 0.0):
        raise ValueError("coordinates must be finite and strictly increasing.")
    output = np.full(array.shape, np.nan, dtype=array.dtype)
    for start, stop in _finite_runs(np.isfinite(array)):
        output[start:stop] = _gaussian_smooth(
            array[start:stop],
            coordinate_array[start:stop],
            fwhm=fwhm,
        )
    return output


def _with_body_curve(control: WellControl, *, fwhm_axis_units: float) -> WellControl:
    """Smooth trusted native log-AI segments, then project once to the model axis."""
    if control.well_role != "trusted":
        return control
    native_body = _gaussian_smooth_finite_runs(
        control.native.native_filtered_log_ai,
        control.native.coordinates,
        fwhm=fwhm_axis_units,
    )
    model_body = _interp_finite_runs(
        control.sample_axis.values, control.native.coordinates, native_body,
    )
    model_body[~control.valid_mask] = np.nan
    native = replace(control.native, native_body_log_ai=native_body)
    basis_type = "twt" if control.sample_axis.domain == "time" else "tvdss"
    return replace(
        control,
        native=native,
        model_grid_body_log_ai=grid.Log(
            model_body, control.sample_axis.values.copy(), basis_type,
            name="model_grid_body_log_ai", unit="ln(m/s*g/cm3)",
        ),
    )


def _body_log_ai_for_qc(control: WellControl, *, fwhm: float) -> np.ndarray:
    if control.model_grid_body_log_ai is not None:
        return np.asarray(control.model_grid_body_log_ai.values, dtype=np.float64)
    return _gaussian_smooth_finite_runs(
        np.asarray(control.model_grid_filtered_log_ai.values, dtype=np.float64),
        control.sample_axis.values, fwhm=fwhm,
    )


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
    selected = sorted(scored, reverse=True)[:maximum]
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
    events = _event_windows(real, threshold_fraction=threshold_fraction, maximum=maximum_events)
    if not events:
        raise ValueError(f"{well_name}: no target-interval waveform events passed the threshold.")
    fig, axes = plt.subplots(
        len(events), 4, figsize=(14.5, max(3.2, 2.6 * len(events))),
        squeeze=False, constrained_layout=True,
    )
    for row, (event_start, event_stop) in enumerate(events):
        pad = max(3, event_stop - event_start)
        local = slice(max(0, event_start - pad), min(axis.size, event_stop + pad))
        local_axis = axis[local]
        axes[row, 0].plot(real[local], local_axis, color="black", lw=1.5)
        axes[row, 0].axhspan(axis[event_start], axis[event_stop - 1], color="tab:blue", alpha=0.12)
        axes[row, 0].set_title("Real seismic" if row == 0 else "")
        axes[row, 1].plot(
            model_grid_filtered_log_ai[local], local_axis, color="black", lw=1.1, label="filtered",
        )
        axes[row, 1].plot(
            body_log_ai[local], local_axis, color="tab:red", lw=1.5,
            label=f"body, FWHM {body_fwhm_m:g} m",
        )
        axes[row, 1].set_title("log-AI" if row == 0 else "")
        axes[row, 2].plot(
            model_grid_filtered_log_ai[local] - body_log_ai[local], local_axis,
            color="tab:purple", lw=1.2, label="filtered - body",
        )
        axes[row, 2].axvline(0.0, color="black", lw=0.8, alpha=0.6)
        axes[row, 2].set_title("log-AI residual" if row == 0 else "")
        axes[row, 3].plot(
            full_synthetic[local], local_axis, color="tab:blue", lw=1.2, label="filtered forward",
        )
        axes[row, 3].plot(
            body_synthetic[local], local_axis, color="tab:orange", lw=1.2, label="body forward",
        )
        axes[row, 3].set_title("Shared-gain synthetic" if row == 0 else "")
        if row == 0:
            for column in (1, 2, 3):
                axes[row, column].legend(fontsize=8)
        for column in range(4):
            axes[row, column].set_ylim(float(local_axis[-1]), float(local_axis[0]))
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_ylabel("TVDSS [m]" if column == 0 else "")
    fig.suptitle(f"{well_name} | filtered/body waveform slices | FWHM {body_fwhm_m:g} m")
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
        "body_smoothing_fwhm_m", "dynamic_correlation_window_m",
        "event_threshold_fraction", "max_event_windows_per_well",
    }
    if set(config) != expected:
        raise ValueError(f"real_field_well_controls.qc must contain exactly {sorted(expected)}.")
    if control_set.sample_domain != "depth" or control_set.depth_basis != "tvdss":
        raise ValueError("Current Step 6 forward QC requires depth/TVDSS well controls.")
    if any(isinstance(config[key], (bool, np.bool_)) for key in expected):
        raise ValueError("Step 6 QC numeric settings cannot be boolean.")
    body_fwhm = float(config["body_smoothing_fwhm_m"])
    saved_width = control_set.body_smoothing_fwhm_axis_units
    if saved_width is not None and not np.isclose(body_fwhm, saved_width, rtol=0.0, atol=1e-12):
        raise ValueError("QC body smoothing width must match the saved body curves.")
    dynamic_window = float(config["dynamic_correlation_window_m"])
    threshold_fraction = float(config["event_threshold_fraction"])
    maximum_events = config["max_event_windows_per_well"]
    if (
        not np.all(np.isfinite([body_fwhm, dynamic_window, threshold_fraction]))
        or body_fwhm <= 0.0 or dynamic_window <= 0.0 or not 0.0 < threshold_fraction <= 1.0
        or not isinstance(maximum_events, (int, np.integer)) or maximum_events < 1
    ):
        raise ValueError("Step 6 QC numeric settings are invalid.")
    wavelet_time, wavelet_amp, relation_a, relation_b, forward_inputs_path = (
        load_depth_forward_inputs(forward_inputs_run_dir, repo_root=repo_root)
    )

    figures_root = output_dir / "figures"
    figures_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    evaluation_supports: dict[str, EvaluationSupport] = {}
    for control in control_set.controls:
        well_dir = figures_root / sanitize_filename(control.well_name)
        well_dir.mkdir()
        axis = control.sample_axis.values
        model_grid_filtered_log_ai = np.asarray(control.model_grid_filtered_log_ai.values, dtype=np.float64)
        body_log_ai = _body_log_ai_for_qc(control, fwhm=body_fwhm)
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
            body_log_ai, axis,
            wavelet_time_s=wavelet_time, wavelet_amplitude=wavelet_amp,
            relation_a=relation_a, relation_b=relation_b,
        )
        markers = horizon_markers_along_control(control, target_zone)
        support = build_evaluation_support(
            control,
            target_interval=(float(markers[0][0]), float(markers[-1][0])),
            observed_support=np.isfinite(real),
            well_curve_support=control.valid_mask,
            additional_support=(
                np.isfinite(real),
                np.isfinite(full_forward),
            ),
        )
        evaluation_supports[control.well_name] = support
        relation = AIVelocityRelation(a=relation_a, b=relation_b)
        waveform_support = _waveform_evaluation_support(
            control, support, real=real, full_forward=full_forward, body_forward=body_forward,
            full_response_mask=_complete_forward_mask(
                model_grid_filtered_log_ai, axis, wavelet_time,
                sample_domain="depth", relation=relation,
            ),
            body_response_mask=_complete_forward_mask(
                body_log_ai, axis, wavelet_time, sample_domain="depth", relation=relation,
            ),
        )
        selected = slice(waveform_support.start_index, waveform_support.stop_index)
        if np.any(~np.isfinite(body_forward[selected])):
            raise ValueError(f"{control.well_name}: body forward does not cover the baseline evaluation support.")
        support_fraction = support.support_fraction
        local_axis = axis[selected]
        local_real = real[selected]
        real_std = float(np.std(local_real))
        if not np.isfinite(real_std) or real_std <= 0.0:
            raise ValueError(f"{control.well_name}: target seismic has zero variance.")
        local_real = (local_real - float(np.mean(local_real))) / real_std
        local_full_forward = full_forward[selected]
        local_body_forward = body_forward[selected]
        denominator = float(np.dot(local_full_forward, local_full_forward))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(f"{control.well_name}: baseline synthetic has no positive finite energy.")
        signed_gain = float(np.dot(local_real, local_full_forward) / denominator)
        gain = abs(signed_gain)
        body_to_full_rms = float(np.sqrt(np.dot(local_body_forward, local_body_forward) / denominator))
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
            axis=local_axis, log_ai=body_log_ai[selected],
            synthetic=local_body_forward, real=local_real,
            dynamic_window_axis_units=dynamic_window, name=f"Body AI, FWHM {body_fwhm:g} m",
        )
        body_corr = _safe_corr(local_real, local_body_forward)
        fig, _ = plot_well_waveform_qc(
            [body_objects[0]], *body_objects[1:], figsize=(13.0, 7.5),
            synthetic_ai=body_objects[0],
            title=f"Step 6 body forward QC | {control.well_name} | FWHM {body_fwhm:g} m | corr={body_corr:.3f}",
            horizon_markers=local_markers,
        )
        body_path = well_dir / "body_waveform_qc.png"
        fig.savefig(body_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        comparison_path = well_dir / "event_waveform_comparison.png"
        event_count = _plot_event_comparison(
            output_path=comparison_path, well_name=control.well_name, axis=local_axis,
            model_grid_filtered_log_ai=model_grid_filtered_log_ai[selected],
            body_log_ai=body_log_ai[selected], real=local_real,
            full_synthetic=local_full_forward, body_synthetic=local_body_forward,
            body_fwhm_m=body_fwhm, threshold_fraction=threshold_fraction,
            maximum_events=maximum_events,
        )
        rows.append(
            {
                "well_name": control.well_name,
                "target_interval_start": support.target_interval_start,
                "target_interval_stop": support.target_interval_stop,
                "evaluation_support_start": support.support_start,
                "evaluation_support_stop": support.support_stop,
                "evaluation_support_start_index": support.start_index,
                "evaluation_support_stop_index": support.stop_index,
                "evaluation_support_samples": support.support_samples,
                "evaluation_support_candidate_samples": support.candidate_samples,
                "waveform_support_start": waveform_support.support_start,
                "waveform_support_stop": waveform_support.support_stop,
                "waveform_support_start_index": waveform_support.start_index,
                "waveform_support_stop_index": waveform_support.stop_index,
                "waveform_support_samples": waveform_support.support_samples,
                "target_support_fraction": support_fraction,
                "shared_forward_gain": gain,
                "signed_forward_gain": signed_gain,
                "full_forward_correlation": full_corr,
                "body_smoothing_fwhm_m": body_fwhm,
                "body_forward_correlation": body_corr,
                "body_forward_correlation_change": body_corr - full_corr,
                "full_forward_mse": float(np.mean(np.square(local_full_forward - local_real))),
                "body_forward_mse": float(np.mean(np.square(local_body_forward - local_real))),
                "body_to_full_forward_rms_ratio": body_to_full_rms,
                "event_window_count": event_count,
                "full_waveform_qc": repo_relative_path(full_path, root=repo_root),
                "body_waveform_qc": repo_relative_path(body_path, root=repo_root),
                "event_waveform_comparison": repo_relative_path(comparison_path, root=repo_root),
            }
        )

    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame.from_records(rows).to_csv(metrics_path, index=False)
    support_path = output_dir / "evaluation_support.json"
    write_evaluation_support_manifest(
        support_path,
        sample_axis=control_set.sample_axis,
        supports=evaluation_supports,
        source="real_field_well_controls.qc",
    )
    manifest = {
        "schema_version": QC_SCHEMA_VERSION,
        "status": "ok",
        "sample_domain": "depth",
        "depth_basis": "tvdss",
        "config": dict(config),
        "forward_model_inputs": repo_relative_path(forward_inputs_path, root=repo_root),
        "evaluation_support": repo_relative_path(support_path, root=repo_root),
        "well_count": len(rows),
        "outputs": {
            "metrics_csv": repo_relative_path(metrics_path, root=repo_root),
            "figures_dir": repo_relative_path(figures_root, root=repo_root),
            "evaluation_support": repo_relative_path(support_path, root=repo_root),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _trusted_control_set(control_set: WellControlSet) -> WellControlSet:
    controls = tuple(control for control in control_set.controls if control.well_role == "trusted")
    if not controls:
        raise ValueError("Step 6 QC requires at least one trusted well.")
    return WellControlSet(
        sample_axis=control_set.sample_axis,
        controls=controls,
        sample_domain=control_set.sample_domain,
        sample_unit=control_set.sample_unit,
        depth_basis=control_set.depth_basis,
        source_run_type=control_set.source_run_type,
        provenance=control_set.provenance,
        supervision_curve=control_set.supervision_curve,
        body_smoothing_fwhm_axis_units=control_set.body_smoothing_fwhm_axis_units,
    )


def forward_time_finite_runs(
    log_ai: np.ndarray,
    *,
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    sample_step_s: float,
) -> np.ndarray:
    values = np.asarray(log_ai, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for start, stop in _finite_runs(np.isfinite(values)):
        if stop - start < 2:
            continue
        output[start:stop] = forward_time(
            values[start:stop],
            np.asarray(wavelet_time_s, dtype=np.float64),
            np.asarray(wavelet_amplitude, dtype=np.float64),
            sample_step_s=float(sample_step_s),
        )
    return output


def _load_qc_wavelet(
    *,
    config: Mapping[str, Any],
    domain: str,
    depth_basis: str | None,
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, AIVelocityRelation | None, Path]:
    forward_run = str(config.get("forward_model_inputs_run_dir") or "").strip()
    if forward_run:
        run_dir = resolve_relative_path(forward_run, root=repo_root)
        time_s, amplitude, relation, _payload = load_forward_inputs(
            run_dir, repo_root=repo_root, domain=domain, depth_basis=depth_basis
        )
        return time_s, amplitude, relation, run_dir / "forward_model_inputs.json"
    wavelet_file = str(config.get("wavelet_file") or "").strip()
    wavelet_run = str(config.get("wavelet_generation_run_dir") or "").strip()
    if not wavelet_file and wavelet_run:
        wavelet_file = str(resolve_relative_path(wavelet_run, root=repo_root) / "selected_wavelet.csv")
    if not wavelet_file:
        raise ValueError(
            "Step 6 QC requires forward_model_inputs_run_dir or wavelet_generation_run_dir/wavelet_file."
        )
    path = resolve_relative_path(wavelet_file, root=repo_root)
    time_s, amplitude = load_wavelet_csv(path)
    relation_info = config.get("ai_velocity_relation")
    relation = AIVelocityRelation.from_mapping(relation_info) if relation_info is not None else None
    if domain == "depth" and relation is None:
        raise ValueError("Depth Step 6 QC requires ai_velocity_relation when loading a wavelet directly.")
    return time_s, amplitude, relation, path


def _time_qc_settings(config: Mapping[str, Any]) -> tuple[float, float]:
    required = {"body_smoothing_fwhm_s", "dynamic_correlation_window_s"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Time Step 6 QC is missing settings: {missing}")
    values = {}
    for key in sorted(required):
        value = config[key]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} must be a finite positive number.")
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{key} must be a finite positive number.")
        values[key] = number
    return values["body_smoothing_fwhm_s"], values["dynamic_correlation_window_s"]


def write_well_control_qc(
    control_set: WellControlSet,
    *,
    survey: Any,
    target_zone: TargetZone,
    output_dir: Path,
    repo_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Write trusted-well waveform QC and fixed evaluation support in either domain."""

    trusted = _trusted_control_set(control_set)
    settings = dict(config)
    if control_set.sample_domain == "depth":
        forward_run = str(settings.get("forward_model_inputs_run_dir") or "").strip()
        if not forward_run:
            raise ValueError("Depth Step 6 QC requires forward_model_inputs_run_dir.")
        depth_config = dict(settings)
        depth_config.pop("forward_model_inputs_run_dir", None)
        return write_depth_well_control_qc(
            trusted,
            survey=survey,
            target_zone=target_zone,
            forward_inputs_run_dir=resolve_relative_path(forward_run, root=repo_root),
            output_dir=output_dir,
            repo_root=repo_root,
            config=depth_config,
        )
    if control_set.sample_domain != "time" or control_set.depth_basis is not None:
        raise ValueError("Step 6 QC supports only time or TVDSS depth controls.")
    body_fwhm_s, dynamic_window_s = _time_qc_settings(settings)
    saved_width = trusted.body_smoothing_fwhm_axis_units
    if saved_width is not None and not np.isclose(body_fwhm_s, saved_width, rtol=0.0, atol=1e-12):
        raise ValueError("QC body smoothing width must match the saved body curves.")
    wavelet_time, wavelet_amplitude, _relation, wavelet_path = _load_qc_wavelet(
        config=settings, domain="time", depth_basis=None, repo_root=repo_root
    )
    figures_root = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=False)
    figures_root.mkdir()
    rows: list[dict[str, Any]] = []
    supports: dict[str, EvaluationSupport] = {}
    step_s = float(trusted.sample_axis.values[1] - trusted.sample_axis.values[0])
    for control in trusted.controls:
        axis = control.sample_axis.values
        real = sample_seismic_along_control(control, survey)
        filtered_log_ai = np.asarray(control.model_grid_filtered_log_ai.values, dtype=np.float64)
        body_log_ai = _body_log_ai_for_qc(control, fwhm=body_fwhm_s)
        full_forward = forward_time_finite_runs(
            filtered_log_ai,
            wavelet_time_s=wavelet_time,
            wavelet_amplitude=wavelet_amplitude,
            sample_step_s=step_s,
        )
        body_forward = forward_time_finite_runs(
            body_log_ai,
            wavelet_time_s=wavelet_time,
            wavelet_amplitude=wavelet_amplitude,
            sample_step_s=step_s,
        )
        markers = horizon_markers_along_control(control, target_zone)
        support = build_evaluation_support(
            control,
            target_interval=(float(markers[0][0]), float(markers[-1][0])),
            observed_support=np.isfinite(real),
            well_curve_support=control.valid_mask,
            additional_support=(np.isfinite(real), np.isfinite(full_forward), np.isfinite(body_forward)),
        )
        supports[control.well_name] = support
        waveform_support = _waveform_evaluation_support(
            control, support, real=real, full_forward=full_forward, body_forward=body_forward,
            full_response_mask=_complete_forward_mask(
                filtered_log_ai, axis, wavelet_time, sample_domain="time",
            ),
            body_response_mask=_complete_forward_mask(
                body_log_ai, axis, wavelet_time, sample_domain="time",
            ),
        )
        selected = slice(waveform_support.start_index, waveform_support.stop_index)
        local_real = np.asarray(real[selected], dtype=np.float64)
        local_forward = np.asarray(full_forward[selected], dtype=np.float64)
        local_body_forward = np.asarray(body_forward[selected], dtype=np.float64)
        real_std = float(np.std(local_real))
        denominator = float(np.dot(local_forward, local_forward))
        if not np.isfinite(real_std) or real_std <= 0.0 or not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(f"{control.well_name}: time-domain QC support has insufficient waveform energy.")
        local_real = (local_real - float(np.mean(local_real))) / real_std
        gain = float(np.dot(local_real, local_forward) / denominator)
        local_forward = abs(gain) * local_forward
        local_body_forward = abs(gain) * local_body_forward
        local_axis = axis[selected]
        full_objects = _waveform_objects(
            axis=local_axis,
            log_ai=filtered_log_ai[selected],
            synthetic=local_forward,
            real=local_real,
            dynamic_window_axis_units=dynamic_window_s,
            name="Filtered AI",
            basis_type="twt",
        )
        body_objects = _waveform_objects(
            axis=local_axis,
            log_ai=body_log_ai[selected],
            synthetic=local_body_forward,
            real=local_real,
            dynamic_window_axis_units=dynamic_window_s,
            name=f"Body AI, FWHM {body_fwhm_s:g} s",
            basis_type="twt",
        )
        correlation = _safe_corr(local_real, local_forward)
        body_correlation = _safe_corr(local_real, local_body_forward)
        well_dir = figures_root / sanitize_filename(control.well_name)
        well_dir.mkdir()
        figure, _ = plot_well_waveform_qc(
            [full_objects[0]], *full_objects[1:], figsize=(13.0, 7.5), synthetic_ai=full_objects[0],
            title=f"Step 6 time forward QC | {control.well_name} | corr={correlation:.3f}",
            horizon_markers=[item for item in markers if local_axis[0] <= item[0] <= local_axis[-1]],
        )
        full_path = well_dir / "full_waveform_qc.png"
        figure.savefig(full_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        figure, _ = plot_well_waveform_qc(
            [body_objects[0]], *body_objects[1:], figsize=(13.0, 7.5), synthetic_ai=body_objects[0],
            title=f"Step 6 time body QC | {control.well_name} | FWHM={body_fwhm_s:g} s | corr={body_correlation:.3f}",
            horizon_markers=[item for item in markers if local_axis[0] <= item[0] <= local_axis[-1]],
        )
        body_path = well_dir / "body_waveform_qc.png"
        figure.savefig(body_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        rows.append({
            "well_name": control.well_name,
            "well_role": control.well_role,
            "target_interval_start": support.target_interval_start,
            "target_interval_stop": support.target_interval_stop,
            "evaluation_support_start": support.support_start,
            "evaluation_support_stop": support.support_stop,
            "evaluation_support_samples": support.support_samples,
            "waveform_support_start": waveform_support.support_start,
            "waveform_support_stop": waveform_support.support_stop,
            "waveform_support_start_index": waveform_support.start_index,
            "waveform_support_stop_index": waveform_support.stop_index,
            "waveform_support_samples": waveform_support.support_samples,
            "full_forward_correlation": correlation,
            "body_smoothing_fwhm_s": body_fwhm_s,
            "body_forward_correlation": body_correlation,
            "body_forward_correlation_change": body_correlation - correlation,
            "full_forward_mse": float(np.mean(np.square(local_forward - local_real))),
            "body_forward_mse": float(np.mean(np.square(local_body_forward - local_real))),
            "body_to_full_forward_rms_ratio": float(
                np.sqrt(np.dot(local_body_forward, local_body_forward) / max(np.dot(local_forward, local_forward), 1e-30))
            ),
            "shared_forward_gain": abs(gain),
            "full_waveform_qc": repo_relative_path(full_path, root=repo_root),
            "body_waveform_qc": repo_relative_path(body_path, root=repo_root),
        })
    metrics_path = output_dir / "metrics.csv"
    pd.DataFrame.from_records(rows).to_csv(metrics_path, index=False)
    support_path = output_dir / "evaluation_support.json"
    write_evaluation_support_manifest(
        support_path, sample_axis=trusted.sample_axis, supports=supports, source="real_field_well_controls.qc"
    )
    support_relative = repo_relative_path(support_path, root=repo_root)
    manifest = {
        "schema_version": QC_SCHEMA_VERSION,
        "status": "ok",
        "sample_domain": "time",
        "depth_basis": None,
        "wavelet": repo_relative_path(wavelet_path, root=repo_root),
        "evaluation_support": support_relative,
        "well_count": len(rows),
        "outputs": {
            "metrics_csv": repo_relative_path(metrics_path, root=repo_root),
            "figures_dir": repo_relative_path(figures_root, root=repo_root),
            "evaluation_support": support_relative,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest

# Serialization and loading


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
        raise ValueError(f"Well-control manifest columns do not match the {SCHEMA_VERSION} contract.")
    manifest_names = [normalize_well_name(value) for value in manifest["well_name"]]
    if len(manifest_names) != len(set(manifest_names)):
        raise ValueError("Well-control manifest well names must be unique after normalization.")
    role_values = set(manifest["well_role"].astype(str).str.casefold())
    if not role_values.issubset({"trusted", "background_only", "excluded"}):
        raise ValueError(f"Well-control manifest contains unsupported well roles: {sorted(role_values)}")
    status_values = set(manifest["status"].astype(str))
    if not status_values.issubset({"ok", "failed", "excluded"}):
        raise ValueError(f"Well-control manifest contains unsupported status values: {sorted(status_values)}")
    failed_rows = manifest[manifest["status"].astype(str).eq("failed")]
    if failed_rows["reason"].astype(str).str.strip().eq("").any():
        raise ValueError("Every failed well-control manifest row must include a reason.")
    excluded_rows = manifest[manifest["status"].astype(str).eq("excluded")]
    if excluded_rows["well_role"].astype(str).str.casefold().ne("excluded").any():
        raise ValueError("Only excluded wells may use the excluded manifest status.")
    successful_names = {
        normalize_well_name(value)
        for value in manifest.loc[manifest["status"].astype(str).eq("ok"), "well_name"]
    }
    control_names = {normalize_well_name(control.well_name) for control in control_set.controls}
    if successful_names != control_names:
        raise ValueError("Well-control manifest successful rows do not exactly match the control set.")
    manifest_roles = {
        normalize_well_name(row["well_name"]): str(row["well_role"]).strip().casefold()
        for _, row in manifest.loc[manifest["status"].astype(str).eq("ok")].iterrows()
    }
    for control in control_set.controls:
        if manifest_roles.get(normalize_well_name(control.well_name)) != control.well_role:
            raise ValueError(f"Well-control manifest role differs from the control set: {control.well_name}")
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
            "well_role": control.well_role,
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
            "supervision_curve": control_set.supervision_curve,
            "body_smoothing_fwhm_axis_units": control_set.body_smoothing_fwhm_axis_units,
            "native_source_role": str(control.provenance.get("source_las_role") or "source"),
            "gap_policy": "persist_repaired_short_gaps",
            "provenance": portable_provenance(control.provenance),
            "native": {
                "sample_domain": control.native.sample_domain,
                "sample_unit": control.native.sample_unit,
                "depth_basis": control.native.depth_basis,
                "coordinate_key": "native_coordinates",
                "valid_mask_key": "native_valid_mask",
                "observed_mask_key": "native_observed_mask",
                "provenance": portable_provenance(control.native.provenance),
            },
        }
        body_arrays = {}
        if control.model_grid_body_log_ai is not None:
            body_arrays = {
                "model_grid_body_log_ai": np.asarray(control.model_grid_body_log_ai.values, dtype=np.float32),
                "native_body_log_ai": np.asarray(control.native.native_body_log_ai, dtype=np.float32),
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
            native_observed_mask=control.native.observed_mask.astype(bool),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
            **body_arrays,
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
            "native_source_role": "role_specific_source",
            "gap_policy": "persist_repaired_short_gaps",
        },
        business_config=resolved_config,
        input_contracts={},
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
        "native_source_role": "role_specific_source",
        "gap_policy": "persist_repaired_short_gaps",
        "supervision_curve": control_set.supervision_curve,
        "body_smoothing_fwhm_axis_units": control_set.body_smoothing_fwhm_axis_units,
        "counts": {
            "candidate_wells": int(len(manifest_out)),
            "successful_wells": int((manifest_out["status"] == "ok").sum()),
            "failed_wells": int((manifest_out["status"] == "failed").sum()),
            "excluded_wells": int((manifest_out["status"] == "excluded").sum()),
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


def _evaluation_support_path_from_run(run_dir: Path, *, repo_root: Path, summary: Mapping[str, Any]) -> Path | None:
    """Resolve an explicitly recorded Step-6 support manifest, if present."""

    qc = summary.get("qc")
    recorded = qc.get("evaluation_support") if isinstance(qc, Mapping) else None
    if not recorded and isinstance(qc, Mapping):
        outputs = qc.get("outputs")
        recorded = outputs.get("evaluation_support") if isinstance(outputs, Mapping) else None
    if not recorded:
        outputs = summary.get("outputs")
        recorded = outputs.get("evaluation_support") if isinstance(outputs, Mapping) else None
    if recorded:
        path = resolve_relative_path(str(recorded), root=repo_root)
        if not path.is_file():
            raise FileNotFoundError(f"Recorded evaluation-support manifest is missing: {path}")
        return path
    conventional = Path(run_dir) / "qc" / "evaluation_support.json"
    return conventional if conventional.is_file() else None


def load_evaluation_support_for_run(
    run_dir: Path,
    *,
    sample_axis: SampleAxis,
    repo_root: Path,
) -> dict[str, EvaluationSupport]:
    """Load the required canonical support from a new Step-6 run."""

    summary_path = Path(run_dir) / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    path = _evaluation_support_path_from_run(Path(run_dir), repo_root=repo_root, summary=summary)
    if path is None:
        raise FileNotFoundError(
            f"Step-6 run {run_dir} has no evaluation-support manifest; "
            "rerun Step 6 to create qc/evaluation_support.json."
        )
    return load_evaluation_support_manifest(path, sample_axis=sample_axis)


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
    if summary.get("native_source_role") != "role_specific_source" or summary.get("gap_policy") != "persist_repaired_short_gaps":
        raise ValueError(f"Well-control run does not use the role-specific source contract: {run_dir}")
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
    role_values = set(manifest["well_role"].astype(str).str.casefold())
    if not role_values.issubset({"trusted", "background_only", "excluded"}):
        raise ValueError(f"well_control_manifest.csv contains unsupported well roles: {sorted(role_values)}")
    status_values = set(manifest["status"].astype(str))
    if not status_values.issubset({"ok", "failed", "excluded"}):
        raise ValueError(f"well_control_manifest.csv contains unsupported statuses: {sorted(status_values)}")
    unavailable = manifest[manifest["status"].astype(str).isin(["failed", "excluded"])]
    if unavailable["well_npz_path"].astype(str).str.strip().ne("").any():
        raise ValueError("Failed or excluded well-control rows must not reference consumable NPZ files.")
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
            base_keys = {
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
                "native_observed_mask",
                "metadata_json",
            }
            body_keys = {"model_grid_body_log_ai", "native_body_log_ai"}
            has_body = body_keys.issubset(data.files)
            if set(data.files) != base_keys | (body_keys if has_body else set()):
                raise ValueError(f"Unexpected well-control NPZ keys: {path}")
            if has_body and any(data[key].dtype != np.dtype("float32") for key in body_keys):
                raise ValueError(f"Body well-control arrays must be float32: {path}")
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
                or data["native_observed_mask"].dtype != np.dtype("bool")
            ):
                raise ValueError(f"Native well-control coordinate/log/mask dtypes are invalid: {path}")
            metadata_array = np.asarray(data["metadata_json"])
            if metadata_array.ndim != 0 or metadata_array.dtype.kind not in {"U", "S"}:
                raise ValueError(f"Well-control metadata_json must be a scalar string: {path}")
            metadata = json.loads(str(metadata_array.item()))
            if metadata.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"Unsupported well-control NPZ schema: {path}")
            if metadata.get("supervision_curve", "filtered") != summary.get("supervision_curve", "filtered"):
                raise ValueError(f"Well-control supervision curve differs from its run summary: {path}")
            if metadata.get("body_smoothing_fwhm_axis_units") != summary.get("body_smoothing_fwhm_axis_units"):
                raise ValueError(f"Well-control body smoothing width differs from its run summary: {path}")
            expected_metadata = {
                "source_run_type": str(summary["source_adapter"]),
                "well_role": str(row["well_role"]).strip().casefold(),
                "sample_domain": axis.domain,
                "sample_unit": axis.unit,
                "depth_basis": summary.get("depth_basis"),
                "linear_ai_unit": LINEAR_AI_UNIT,
                "value_domain": "log(AI)",
                "model_axis_value_key": "model_grid_filtered_log_ai",
                "model_axis_valid_mask_key": "valid_mask",
                "model_axis_observed_valid_mask_key": "observed_valid_mask",
                "native_value_key": "native_filtered_log_ai",
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
                or native_metadata.get("observed_mask_key") != "native_observed_mask"
            ):
                raise ValueError(f"Native well-control metadata is inconsistent: {path}")
            native = NativeWellControl(
                well_name=str(metadata["well_name"]),
                coordinates=np.asarray(data["native_coordinates"], dtype=np.float64),
                native_filtered_log_ai=np.asarray(data["native_filtered_log_ai"], dtype=np.float64),
                valid_mask=np.asarray(data["native_valid_mask"], dtype=bool),
                observed_mask=np.asarray(data["native_observed_mask"], dtype=bool),
                sample_domain=axis.domain,
                sample_unit=axis.unit,
                depth_basis=summary.get("depth_basis"),
                provenance=dict(native_metadata.get("provenance") or {}),
                native_body_log_ai=(np.asarray(data["native_body_log_ai"], dtype=np.float64) if has_body else None),
            )
            file_axis = np.asarray(data["samples"], dtype=np.float64)
            if not np.array_equal(file_axis, axis.values):
                raise ValueError(f"Well-control SampleAxis mismatch: {path}")
            control = control_from_arrays(
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
                well_role=str(metadata["well_role"]),
                model_grid_body_log_ai=(np.asarray(data["model_grid_body_log_ai"], dtype=np.float64) if has_body else None),
            )
            if not np.array_equal(control.valid_mask, np.asarray(data["valid_mask"], dtype=bool)):
                raise ValueError(f"Well-control NPZ valid_mask disagrees with finite values: {path}")
            if not np.array_equal(
                control.observed_valid_mask,
                np.asarray(data["observed_valid_mask"], dtype=bool),
            ):
                raise ValueError(f"Well-control NPZ observed_valid_mask changed during loading: {path}")
            controls.append(control)
    provenance = dict(summary["provenance"])
    support_path = _evaluation_support_path_from_run(run_dir, repo_root=repo_root, summary=summary)
    if support_path is not None:
        provenance["evaluation_support_path"] = str(support_path)
    return WellControlSet(
        sample_axis=axis,
        controls=tuple(controls),
        sample_domain=axis.domain,
        sample_unit=axis.unit,
        depth_basis=summary.get("depth_basis"),
        source_run_type=str(summary["source_adapter"]),
        provenance=provenance,
        supervision_curve=summary.get("supervision_curve", "filtered"),
        body_smoothing_fwhm_axis_units=summary.get("body_smoothing_fwhm_axis_units"),
    )


__all__ = [
    "LINEAR_AI_UNIT", "MANIFEST_COLUMNS", "QC_SCHEMA_VERSION", "SCHEMA_VERSION",
    "SOURCE_RUN_TYPE", "WELL_TIE_SOURCE_SCHEMA",
    "NativeWellControl", "WellControl", "WellControlSet",
    "control_from_arrays", "validate_sample_axis",
    "build_evaluation_support", "build_target_zone_evaluation_support",
    "build_well_control_set", "forward_depth_finite_runs", "forward_time_finite_runs",
    "horizon_markers_along_control", "load_depth_forward_inputs",
    "load_evaluation_support_for_run", "load_well_control_set",
    "sample_seismic_along_control", "write_depth_well_control_qc",
    "write_well_control_qc", "write_well_control_set",
  ]
