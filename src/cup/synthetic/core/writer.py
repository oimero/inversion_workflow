"""Single-artifact HDF5 writer for structured synthetic samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping
import uuid

import h5py
import numpy as np

from cup.synthetic.core.artifacts import write_dataset
from cup.synthetic.core.records import (
    DepthForwardExtras,
    StructuredSampleRecord,
    TimeForwardExtras,
)


@dataclass(frozen=True)
class ArtifactReference:
    hdf5_group: str
    seismic_input_dataset: str
    seismic_model_consistent_dataset: str
    valid_mask_dataset: str
    lfm_dataset: str
    model_log_ai_dataset: str


def _dataset(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
    *,
    unit: str,
    sample_domain: str,
    axis_path: str,
    axis_order: str | list[str],
    dtype: object | None = None,
) -> h5py.Dataset:
    array = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    return write_dataset(
        group,
        name,
        array,
        unit=unit,
        sample_domain=sample_domain,
        axis_path=axis_path,
        axis_order=axis_order,
    )


def _json_attribute(group: h5py.Group, name: str, value: Any) -> None:
    group.attrs[name] = json.dumps(value, allow_nan=False, sort_keys=True)


def serialize_qc_attributes(qc_values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in qc_values.items()
        if np.isscalar(value) and not isinstance(value, (dict, list, tuple))
    }


def _write_qc_attributes(group: h5py.Group, qc_values: Mapping[str, Any]) -> None:
    for key, value in serialize_qc_attributes(qc_values).items():
        group.attrs[key] = value


def _string_column(group: h5py.Group, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def _columnar_table(
    parent: h5py.Group,
    name: str,
    rows: tuple[Mapping[str, Any], ...],
    *,
    columns: tuple[str, ...],
) -> h5py.Group:
    if not rows:
        raise ValueError(f"structured truth requires non-empty {name} table")
    missing = [
        (index, column)
        for index, row in enumerate(rows)
        for column in columns
        if column not in row
    ]
    if missing:
        raise ValueError(f"structured {name} table is missing fields: {missing[:5]}")
    group = parent.create_group(name)
    group.attrs["row_count"] = len(rows)
    for column in columns:
        values = [row[column] for row in rows]
        if all(isinstance(value, (str, bytes, np.str_)) for value in values):
            _string_column(group, column, [str(value) for value in values])
        elif all(isinstance(value, (bool, np.bool_)) for value in values):
            group.create_dataset(column, data=np.asarray(values, dtype=np.bool_))
        elif all(isinstance(value, (int, np.integer)) for value in values):
            group.create_dataset(column, data=np.asarray(values, dtype=np.int64))
        else:
            array = np.asarray(values, dtype=np.float64)
            if np.any(~np.isfinite(array)):
                raise ValueError(f"structured {name}.{column} contains non-finite values")
            group.create_dataset(column, data=array)
    return group


def validate_structured_truth_tables(record: StructuredSampleRecord) -> None:
    """Validate parent-local zone/segment topology before HDF5 publication."""
    truth = record.truth
    zone_rows = truth.structured_zone_truth
    segment_rows = truth.structured_segment_truth
    if not zone_rows or not segment_rows:
        raise ValueError("structured sample requires explicit zone and segment truth")
    n_lateral = truth.lateral_m.size
    zone_keys = {
        (int(row["lateral_index"]), str(row["zone_id"])) for row in zone_rows
    }
    if len(zone_keys) != len(zone_rows):
        raise ValueError("structured zone table contains duplicate lateral/zone keys")
    if {key[0] for key in zone_keys} != set(range(n_lateral)):
        raise ValueError("structured zone table does not cover every lateral trace")
    tolerance = max(float(truth.highres_sample_interval) * 1e-6, 1e-9)
    for lateral_index, zone_id in sorted(zone_keys):
        zone = next(
            row
            for row in zone_rows
            if int(row["lateral_index"]) == lateral_index
            and str(row["zone_id"]) == zone_id
        )
        selected = sorted(
            (
                row
                for row in segment_rows
                if int(row["lateral_index"]) == lateral_index
                and str(row["zone_id"]) == zone_id
            ),
            key=lambda row: (float(row["top"]), int(row["object_id"])),
        )
        if not selected:
            raise ValueError(f"structured zone {zone_id!r} has no segments")
        if not np.isclose(
            float(selected[0]["top"]), float(zone["top"]), rtol=0.0, atol=tolerance
        ) or not np.isclose(
            float(selected[-1]["bottom"]),
            float(zone["bottom"]),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("structured segment endpoints do not cover their zone")
        for previous, current in zip(selected, selected[1:], strict=False):
            if not np.isclose(
                float(previous["bottom"]),
                float(current["top"]),
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("structured segment endpoints are not contiguous")
    for zone_id in sorted({str(row["zone_id"]) for row in zone_rows}):
        rows = [row for row in zone_rows if str(row["zone_id"]) == zone_id]
        for name in ("background_a", "background_b"):
            values = np.asarray([row[name] for row in rows], dtype=np.float64)
            if not np.all(np.isfinite(values)) or not np.allclose(
                values, values[0], rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"structured {zone_id!r} {name} must be realization-zone constant"
                )


def _write_axes(
    root: h5py.Group,
    record: StructuredSampleRecord,
    *,
    published_path: str,
) -> tuple[str, str]:
    truth = record.truth
    domain = truth.sample_domain
    path = published_path
    axes = root.create_group("axes")
    high_name = "tvdss_highres_m" if domain == "depth" else "twt_highres_s"
    model_name = "tvdss_model_m" if domain == "depth" else "twt_model_s"
    high_path = f"{path}/axes/{high_name}"
    model_path = f"{path}/axes/{model_name}"
    for name, values, unit, axis_path, axis_order in (
        ("lateral_m", truth.lateral_m, "m", f"{path}/axes/lateral_m", "lateral"),
        ("inline_float", truth.inline_float, "line", f"{path}/axes/lateral_m", "lateral"),
        ("xline_float", truth.xline_float, "line", f"{path}/axes/lateral_m", "lateral"),
        ("x_m", truth.x_m, "m", f"{path}/axes/lateral_m", "lateral"),
        ("y_m", truth.y_m, "m", f"{path}/axes/lateral_m", "lateral"),
        (high_name, truth.highres_axis, truth.axis_unit, high_path, "sample"),
        (
            model_name,
            record.projected.model_axis.coordinates,
            record.projected.model_axis.unit,
            model_path,
            "sample",
        ),
    ):
        _dataset(
            axes,
            name,
            values,
            unit=unit,
            sample_domain=domain,
            axis_path=axis_path,
            axis_order=axis_order,
            dtype=np.float64,
        )
    if domain == "time":
        for name, values in (
            ("twt_forward_highres_s", truth.highres_axis[1:]),
            (
                "twt_forward_model_s",
                record.projected.model_axis.coordinates[1:],
            ),
        ):
            axis_path = f"{path}/axes/{name}"
            _dataset(
                axes,
                name,
                values,
                unit="s",
                sample_domain=domain,
                axis_path=axis_path,
                axis_order="sample",
                dtype=np.float64,
            )
    return high_path, model_path


def _write_identity(root: h5py.Group, record: StructuredSampleRecord) -> None:
    truth = record.truth
    group = root.create_group("identity")
    group.attrs["realization_id"] = truth.realization_id
    group.attrs["scenario_id"] = truth.scenario.scenario_id
    group.attrs["geometry_family"] = truth.scenario.geometry_family
    group.attrs["duration_mode"] = truth.scenario.duration_mode
    identity = record.domain_metadata.get("structured_identity")
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("structured sample requires structured_identity metadata")
    _json_attribute(group, "structured_identity_json", dict(identity))
    _json_attribute(group, "lfm_source_identity_json", dict(record.lfm.source_identity))
    xline_step = float(record.domain_metadata.get("xline_step"))
    if not np.isfinite(xline_step) or xline_step == 0.0:
        raise ValueError("structured sample requires finite non-zero xline_step")
    group.attrs["xline_step"] = xline_step


def _write_observed(
    root: h5py.Group,
    record: StructuredSampleRecord,
    *,
    model_axis_path: str,
) -> None:
    domain = record.truth.sample_domain
    amplitude_unit = "arbitrary_amplitude"
    group = root.create_group("observed")
    for name, values, unit, dtype in (
        ("seismic", record.forward.seismic_observed, amplitude_unit, np.float32),
        ("lfm", record.lfm.values, "ln(m/s*g/cm3)", np.float32),
        ("valid", record.valid_mask, "bool", None),
    ):
        _dataset(
            group,
            name,
            values,
            unit=unit,
            sample_domain=domain,
            axis_path=model_axis_path,
            axis_order="lateral,sample",
            dtype=dtype,
        )


def _write_truth(
    root: h5py.Group,
    record: StructuredSampleRecord,
    *,
    high_axis_path: str,
    model_axis_path: str,
) -> None:
    truth = record.truth
    projected = record.projected
    domain = truth.sample_domain
    group = root.create_group("truth")
    highres = (
        ("log_ai_highres", truth.log_ai_highres, "ln(m/s*g/cm3)", np.float32),
        ("state_id_highres", truth.state_id_highres, "category", np.int8),
        ("object_id_highres", truth.object_id_highres, "category", np.int32),
        (
            "object_xi_highres",
            truth.object_xi_highres,
            "normalized_object",
            np.float32,
        ),
        ("zone_id_highres", truth.zone_id_highres, "category", np.int16),
        ("boundary_mask_highres", truth.boundary_mask_highres, "bool", None),
        (
            "truth_valid_highres",
            np.isfinite(truth.log_ai_highres),
            "bool",
            None,
        ),
        ("clipping_mask_highres", truth.clipping_mask_highres, "bool", None),
    )
    model = (
        (
            "model_log_ai",
            projected.model_target_log_ai,
            "ln(m/s*g/cm3)",
            np.float32,
        ),
        (
            "state_fraction_model",
            projected.state_fraction_model,
            "fraction",
            np.float32,
        ),
        (
            "dominant_object_id_model",
            projected.dominant_object_id_model,
            "category",
            np.int32,
        ),
        (
            "zone_id_model",
            projected.zone_id_model,
            "category",
            np.int16,
        ),
        (
            "boundary_fraction_model",
            projected.boundary_fraction_model,
            "fraction",
            np.float32,
        ),
        (
            "boundary_mask_model",
            projected.boundary_mask_model,
            "bool",
            None,
        ),
        (
            "categorical_valid_model",
            projected.categorical_valid_mask_model,
            "bool",
            None,
        ),
        (
            "hidden_transition_count_model",
            projected.hidden_transition_count_model,
            "count",
            np.int16,
        ),
        (
            "projection_collapse_mask_model",
            projected.projection_collapse_mask_model,
            "bool",
            None,
        ),
    )
    for name, values, unit, dtype in highres:
        _dataset(
            group,
            name,
            values,
            unit=unit,
            sample_domain=domain,
            axis_path=high_axis_path,
            axis_order="lateral,sample",
            dtype=dtype,
        )
    for name, values, unit, dtype in model:
        order = "lateral,sample,state" if np.asarray(values).ndim == 3 else "lateral,sample"
        _dataset(
            group,
            name,
            values,
            unit=unit,
            sample_domain=domain,
            axis_path=model_axis_path,
            axis_order=order,
            dtype=dtype,
        )
    zone_columns = (
        "zone_id",
        "zone_grid_value",
        "lateral_index",
        "top",
        "bottom",
        "background_a",
        "background_b",
        "zone_valid",
    )
    segment_columns = (
        "zone_id",
        "zone_grid_value",
        "object_id",
        "state",
        "state_id",
        "lateral_index",
        "top",
        "bottom",
        "duration_fraction",
        "duration_samples",
        "c0_raw",
        "c1_raw",
        "c2_raw",
        "c0_projected",
        "c1_projected",
        "c2_projected",
        "c0_effective",
        "c1_effective",
        "c2_effective",
        "segment_supervision_valid",
    )
    _columnar_table(
        group,
        "zones",
        truth.structured_zone_truth,
        columns=zone_columns,
    )
    _columnar_table(
        group,
        "segments",
        truth.structured_segment_truth,
        columns=segment_columns,
    )


def _write_forward(
    root: h5py.Group,
    record: StructuredSampleRecord,
    *,
    high_axis_path: str,
    model_axis_path: str,
) -> None:
    domain = record.truth.sample_domain
    amplitude_unit = "arbitrary_amplitude"
    forward = record.forward
    group = root.create_group("forward")
    _dataset(
        group,
        "model_consistent_seismic",
        forward.seismic_model_consistent,
        unit=amplitude_unit,
        sample_domain=domain,
        axis_path=model_axis_path,
        axis_order="lateral,sample",
        dtype=np.float32,
    )
    context = forward.metadata.get("structured_forward_context")
    if not isinstance(context, Mapping):
        raise ValueError("structured sample requires structured_forward_context metadata")
    context_group = group.create_group("context")
    wavelet_time = np.asarray(context.get("wavelet_time_s"), dtype=np.float64)
    wavelet_amplitude = np.asarray(context.get("wavelet_amplitude"), dtype=np.float64)
    if (
        wavelet_time.ndim != 1
        or wavelet_time.size < 3
        or wavelet_time.size != wavelet_amplitude.size
        or np.any(~np.isfinite(wavelet_time))
        or np.any(~np.isfinite(wavelet_amplitude))
    ):
        raise ValueError("structured forward context wavelet is invalid")
    context_group.create_dataset("wavelet_time_s", data=wavelet_time)
    context_group.create_dataset("wavelet_amplitude", data=wavelet_amplitude)
    context_group.attrs["output_chunk_size"] = int(context["output_chunk_size"])
    _json_attribute(
        context_group,
        "ai_velocity_relation_json",
        context.get("ai_velocity_relation"),
    )
    if not isinstance(forward.extras, (DepthForwardExtras, TimeForwardExtras)):
        raise TypeError("structured sample has unsupported forward extras")


def write_structured_sample(
    h5: h5py.File,
    sample: StructuredSampleRecord,
) -> ArtifactReference:
    """Write, validate and commit one complete parent realization."""
    if not isinstance(sample, StructuredSampleRecord):
        raise TypeError("write_structured_sample requires StructuredSampleRecord")
    validate_structured_truth_tables(sample)
    realization_id = sample.truth.realization_id
    final_path = f"/realizations/{realization_id}"
    if final_path in h5:
        raise FileExistsError(f"structured realization already exists: {realization_id}")
    staging_root = h5.require_group("/__staging__")
    staging_name = uuid.uuid4().hex
    root = staging_root.create_group(staging_name)
    try:
        root.attrs["complete"] = False
        root.attrs["sample_domain"] = sample.truth.sample_domain
        root.attrs["sample_unit"] = sample.truth.axis_unit
        if sample.truth.sample_domain == "depth":
            root.attrs["depth_basis"] = "tvdss"
        _write_identity(root, sample)
        high_axis_path, model_axis_path = _write_axes(
            root,
            sample,
            published_path=final_path,
        )
        _write_observed(root, sample, model_axis_path=model_axis_path)
        _write_truth(
            root,
            sample,
            high_axis_path=high_axis_path,
            model_axis_path=model_axis_path,
        )
        _write_forward(
            root,
            sample,
            high_axis_path=high_axis_path,
            model_axis_path=model_axis_path,
        )
        qc = root.create_group("qc")
        _write_qc_attributes(qc, sample.qc)
        root.attrs["complete"] = True
        h5.require_group("/realizations")
        h5.move(root.name, final_path)
    except Exception:
        if root.name in h5:
            del h5[root.name]
        raise
    path = final_path
    return ArtifactReference(
        hdf5_group=path,
        seismic_input_dataset=f"{path}/observed/seismic",
        seismic_model_consistent_dataset=f"{path}/forward/model_consistent_seismic",
        valid_mask_dataset=f"{path}/observed/valid",
        lfm_dataset=f"{path}/observed/lfm",
        model_log_ai_dataset=f"{path}/truth/model_log_ai",
    )


__all__ = [
    "ArtifactReference",
    "serialize_qc_attributes",
    "validate_structured_truth_tables",
    "write_structured_sample",
]
