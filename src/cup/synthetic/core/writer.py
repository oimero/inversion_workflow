"""Pure serialization of materialized Benchmark samples and variants to v4 HDF5."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import h5py
import numpy as np

from cup.synthetic.core.artifacts import write_dataset
from cup.synthetic.core.records import (
    BenchmarkSample,
    BenchmarkVariant,
    DepthForwardExtras,
    TimeForwardExtras,
)


@dataclass(frozen=True)
class ArtifactReference:
    hdf5_group: str
    seismic_input_dataset: str
    seismic_model_consistent_dataset: str
    valid_mask_dataset: str


@dataclass(frozen=True)
class DatasetPlanItem:
    group_path: str
    name: str
    values: np.ndarray
    unit: str
    dtype: object | None = np.float32


def _common_dataset_plan(
    sample: BenchmarkSample, *, amplitude_unit: str, model_consistent_dtype: object | None
) -> tuple[DatasetPlanItem, ...]:
    forward = sample.forward
    return (
        DatasetPlanItem("priors", "canonical_background_log_ai", sample.canonical_background_log_ai, "ln(m/s*g/cm3)"),
        DatasetPlanItem("targets", "target_increment_log_ai", sample.target_increment_log_ai, "ln(m/s*g/cm3)"),
        DatasetPlanItem("priors", "lfm_ideal", sample.input_lfm_canonical_log_ai, "ln(m/s*g/cm3)"),
        DatasetPlanItem("priors", "lfm_controlled_degraded", sample.input_lfm_controlled_degraded_log_ai, "ln(m/s*g/cm3)"),
        DatasetPlanItem("priors/input_lfm_variants/canonical", "log_ai", sample.input_lfm_canonical_log_ai, "ln(m/s*g/cm3)"),
        DatasetPlanItem("priors/input_lfm_variants/controlled_default", "log_ai", sample.input_lfm_controlled_degraded_log_ai, "ln(m/s*g/cm3)"),
        DatasetPlanItem("seismic", "seismic_observed", forward.seismic_observed, amplitude_unit),
        DatasetPlanItem("seismic", "seismic_model_consistent", forward.seismic_model_consistent, amplitude_unit, model_consistent_dtype),
        DatasetPlanItem("seismic", "subgrid_forward_residual", forward.subgrid_forward_residual, amplitude_unit),
        DatasetPlanItem("residuals", "residual_vs_lfm_ideal", sample.residuals.residual_vs_lfm_ideal, "ln(m/s*g/cm3)"),
        DatasetPlanItem("residuals", "residual_vs_lfm_controlled_degraded", sample.residuals.residual_vs_lfm_controlled_degraded, "ln(m/s*g/cm3)"),
        DatasetPlanItem("masks", "valid_mask", sample.valid_mask, "bool", None),
    )


def _write_common_dataset_plan(
    root: h5py.Group, sample: BenchmarkSample, *, sample_domain: str,
    model_axis_path: str, axis_order: str | list[str], amplitude_unit: str,
    model_consistent_dtype: object | None,
) -> None:
    for item in _common_dataset_plan(
        sample, amplitude_unit=amplitude_unit,
        model_consistent_dtype=model_consistent_dtype,
    ):
        group = root.require_group(item.group_path)
        values = np.asarray(item.values, dtype=item.dtype) if item.dtype is not None else np.asarray(item.values)
        _dataset(
            group, item.name, values, unit=item.unit, sample_domain=sample_domain,
            axis_path=model_axis_path, axis_order=axis_order,
        )


def _dataset(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
    *,
    unit: str,
    sample_domain: str,
    axis_path: str,
    axis_order: str | list[str],
) -> h5py.Dataset:
    return write_dataset(
        group,
        name,
        values,
        unit=unit,
        sample_domain=sample_domain,
        axis_path=axis_path,
        axis_order=axis_order,
    )


def _write_depth_sample(h5: h5py.File, sample: BenchmarkSample) -> ArtifactReference:
    truth_record = sample.truth
    projected = sample.projected
    forward = sample.forward
    if not isinstance(forward.extras, DepthForwardExtras):
        raise TypeError("depth Benchmark sample requires DepthForwardExtras.")
    path = f"/realizations/{truth_record.realization_id}"
    root = h5.create_group(path)
    root.attrs["sample_domain"] = "depth"
    root.attrs["depth_basis"] = "tvdss"
    increment_contract = sample.domain_metadata.get("increment_contract")
    if increment_contract is not None:
        root.attrs["increment_contract_json"] = json.dumps(
            increment_contract, sort_keys=True
        )

    axes = root.create_group("axes")
    axis_values = {
        "lateral_m": truth_record.lateral_m,
        "tvdss_highres_m": truth_record.highres_axis,
        "tvdss_model_m": projected.model_axis.coordinates,
        "inline_float": truth_record.inline_float,
        "xline_float": truth_record.xline_float,
        "x_m": truth_record.x_m,
        "y_m": truth_record.y_m,
    }
    for name, values in axis_values.items():
        if name == "tvdss_highres_m":
            axis_path, axis_order, unit = f"{path}/axes/{name}", "tvdss_highres", "m"
        elif name == "tvdss_model_m":
            axis_path, axis_order, unit = f"{path}/axes/{name}", "tvdss_model", "m"
        else:
            axis_path, axis_order = f"{path}/axes/lateral_m", "lateral"
            unit = "line" if "line" in name else "m"
        _dataset(
            axes,
            name,
            np.asarray(values, dtype=np.float64),
            unit=unit,
            sample_domain="depth",
            axis_path=axis_path,
            axis_order=axis_order,
        )

    truth = root.create_group("truth")
    high_axis_path = f"{path}/axes/tvdss_highres_m"
    model_axis_path = f"{path}/axes/tvdss_model_m"
    for name, values, unit, axis_path in (
        ("log_ai_highres", truth_record.log_ai_highres, "ln(m/s*g/cm3)", high_axis_path),
        ("vp_highres_mps", forward.extras.vp_highres_mps, "m/s", high_axis_path),
        (
            "model_target_log_ai",
            projected.model_target_log_ai,
            "ln(m/s*g/cm3)",
            model_axis_path,
        ),
        ("vp_model_mps", forward.extras.vp_model_mps, "m/s", model_axis_path),
    ):
        _dataset(
            truth,
            name,
            np.asarray(values, dtype=np.float32),
            unit=unit,
            sample_domain="depth",
            axis_path=axis_path,
            axis_order="lateral,tvdss",
        )

    categorical_values = {
        "state_id_highres": truth_record.state_id_highres,
        "object_id_highres": truth_record.object_id_highres,
        "object_xi_highres": truth_record.object_xi_highres,
        "zone_id_highres": truth_record.zone_id_highres,
        "geometry_event_mask_highres": truth_record.geometry_event_mask_highres,
        "boundary_mask_highres": truth_record.boundary_mask_highres,
        "boundary_fraction_model": projected.boundary_fraction_model,
        "boundary_mask_model": projected.boundary_mask_model,
        "state_fraction_model": projected.state_fraction_model,
        "dominant_object_id_model": projected.dominant_object_id_model,
        "zone_id_model": projected.zone_id_model,
    }
    _dataset(
        truth,
        "geometry_event_mask_highres",
        truth_record.geometry_event_mask_highres,
        unit="bool",
        sample_domain="depth",
        axis_path=high_axis_path,
        axis_order="lateral,tvdss",
    )
    _dataset(
        truth,
        "boundary_mask_model",
        projected.boundary_mask_model,
        unit="bool",
        sample_domain="depth",
        axis_path=model_axis_path,
        axis_order="lateral,tvdss",
    )
    categorical = truth.create_group("categorical")
    for name, values in categorical_values.items():
        highres = np.asarray(values).shape[1] == truth_record.highres_axis.size
        _dataset(
            categorical,
            name,
            np.asarray(values),
            unit="category",
            sample_domain="depth",
            axis_path=high_axis_path if highres else model_axis_path,
            axis_order=("lateral,tvdss,state" if np.asarray(values).ndim == 3 else "lateral,tvdss"),
        )

    _write_common_dataset_plan(
        root,
        sample,
        sample_domain="depth",
        model_axis_path=model_axis_path,
        axis_order="lateral,tvdss",
        amplitude_unit="amplitude",
        model_consistent_dtype=np.float32,
    )
    qc = root.create_group("qc")
    for key, value in sample.qc.items():
        if np.isscalar(value) and not isinstance(value, (dict, list, tuple)):
            qc.attrs[key] = value
    return ArtifactReference(
        hdf5_group=path,
        seismic_input_dataset=f"{path}/seismic/seismic_observed",
        seismic_model_consistent_dataset=(
            f"{path}/seismic/seismic_model_consistent"
        ),
        valid_mask_dataset=f"{path}/masks/valid_mask",
    )


def _write_time_sample(h5: h5py.File, sample: BenchmarkSample) -> ArtifactReference:
    truth_record = sample.truth
    projected = sample.projected
    forward = sample.forward
    if not isinstance(forward.extras, TimeForwardExtras):
        raise TypeError("time Benchmark sample requires TimeForwardExtras.")
    path = f"/realizations/{truth_record.realization_id}"
    root = h5.create_group(path)
    root.attrs["scenario_id"] = truth_record.scenario.scenario_id
    root.attrs["status"] = str(sample.qc.get("status", "ok"))
    root.attrs["suite"] = str(sample.qc.get("suite", "field_conditioned"))
    increment_contract = sample.domain_metadata.get("increment_contract")
    if increment_contract is not None:
        root.attrs["increment_contract_json"] = json.dumps(
            increment_contract, sort_keys=True
        )

    axes = root.create_group("axes")
    axis_values = (
        ("lateral_m", truth_record.lateral_m, "m", ["lateral"], ""),
        ("inline_float", truth_record.inline_float, "line", ["lateral"], ""),
        ("xline_float", truth_record.xline_float, "line", ["lateral"], ""),
        ("x_m", truth_record.x_m, "m", ["lateral"], ""),
        ("y_m", truth_record.y_m, "m", ["lateral"], ""),
        ("twt_highres_s", truth_record.highres_axis, "s", ["twt"], ""),
        ("twt_model_s", projected.model_axis.coordinates, "s", ["twt"], ""),
        ("twt_forward_highres_s", truth_record.highres_axis[1:], "s", ["twt_forward"], ""),
        (
            "twt_forward_model_s",
            projected.model_axis.coordinates[1:],
            "s",
            ["twt_forward"],
            "",
        ),
    )
    for name, values, unit, axis_order, axis_path in axis_values:
        _dataset(
            axes,
            name,
            np.asarray(values, dtype=np.float64),
            unit=unit,
            sample_domain="time",
            axis_path=axis_path,
            axis_order=axis_order,
        )

    high_axis = f"{path}/axes/twt_highres_s"
    model_axis = f"{path}/axes/twt_model_s"
    high_forward_axis = f"{path}/axes/twt_forward_highres_s"
    model_forward_axis = f"{path}/axes/twt_forward_model_s"
    truth = root.create_group("truth")
    truth_values = (
        ("truth_log_ai_highres", truth_record.log_ai_highres, "ln(m/s*g/cm3)", high_axis),
        ("model_target_log_ai", projected.model_target_log_ai, "ln(m/s*g/cm3)", model_axis),
        ("rgt_highres", truth_record.rgt_highres, "normalized_zone", high_axis),
        ("rgt_model", projected.rgt_model, "normalized_zone", model_axis),
        ("state_id_highres", truth_record.state_id_highres, "category", high_axis),
        ("object_id_highres", truth_record.object_id_highres, "category", high_axis),
        ("object_xi_highres", truth_record.object_xi_highres, "normalized_object", high_axis),
        ("zone_id_highres", truth_record.zone_id_highres, "category", high_axis),
        ("geometry_event_mask_highres", truth_record.geometry_event_mask_highres, "bool", high_axis),
        ("boundary_mask_highres", truth_record.boundary_mask_highres, "bool", high_axis),
        ("boundary_fraction_model", projected.boundary_fraction_model, "fraction", model_axis),
        ("boundary_mask_model", projected.boundary_mask_model, "bool", model_axis),
        ("dominant_object_id_model", projected.dominant_object_id_model, "category", model_axis),
        ("zone_id_model", projected.zone_id_model, "category", model_axis),
    )
    for name, values, unit, axis_path in truth_values:
        _dataset(
            truth,
            name,
            np.asarray(values),
            unit=unit,
            sample_domain="time",
            axis_path=axis_path,
            axis_order=["lateral", "twt"],
        )
    _dataset(
        truth,
        "state_fraction_model",
        projected.state_fraction_model,
        unit="fraction",
        sample_domain="time",
        axis_path=model_axis,
        axis_order=["lateral", "twt", "state"],
    )
    for name, values, axis_path in (
        ("reflectivity_highres", forward.extras.reflectivity_highres, high_forward_axis),
        ("reflectivity_model", forward.extras.reflectivity_model, model_forward_axis),
        ("forward_valid_mask_highres", forward.extras.forward_valid_mask_highres, high_forward_axis),
        ("forward_valid_mask_model", forward.extras.forward_valid_mask_model, model_forward_axis),
    ):
        _dataset(
            truth,
            name,
            np.asarray(values),
            unit="bool" if "mask" in name else "ratio",
            sample_domain="time",
            axis_path=axis_path,
            axis_order=["lateral", "twt_forward"],
        )

    _write_common_dataset_plan(
        root,
        sample,
        sample_domain="time",
        model_axis_path=model_axis,
        axis_order=["lateral", "twt"],
        amplitude_unit="normalized_amplitude",
        model_consistent_dtype=None,
    )
    qc = root.create_group("qc")
    for key, value in sample.qc.items():
        serialized = key.startswith(
            ("model_grid_", "highres_", "lfm_", "residual_vs_")
        ) and not key.startswith("model_grid_closure_") and key != "highres_forward_reasons"
        if serialized and np.isscalar(value) and not isinstance(value, (dict, list, tuple)):
            qc.attrs[key] = value
    return ArtifactReference(
        hdf5_group=path,
        seismic_input_dataset=f"{path}/seismic/seismic_observed",
        seismic_model_consistent_dataset=f"{path}/seismic/seismic_model_consistent",
        valid_mask_dataset=f"{path}/masks/valid_mask",
    )


def write_benchmark_sample(h5: h5py.File, sample: BenchmarkSample) -> ArtifactReference:
    if sample.truth.sample_domain == "depth":
        return _write_depth_sample(h5, sample)
    if sample.truth.sample_domain == "time":
        return _write_time_sample(h5, sample)
    raise ValueError(f"Unsupported Benchmark sample domain: {sample.truth.sample_domain}")


def write_benchmark_variant(
    h5: h5py.File,
    variant: BenchmarkVariant,
) -> ArtifactReference:
    owner_path = f"/realizations/{variant.owner_realization_id}"
    owner = h5[owner_path]
    variants = owner.require_group("seismic_variants")
    group = variants.create_group(variant.variant_id)
    metadata = dict(variant.metadata)
    group.attrs["variant_id"] = variant.variant_id
    group.attrs["mismatch_family"] = str(metadata["mismatch_family"])
    group.attrs["operator_source"] = str(metadata["operator_source"])
    group.attrs["parameters_json"] = json.dumps(
        dict(metadata.get("parameters") or {}), sort_keys=True
    )
    if variant.sample_domain == "time":
        axis_path = f"{owner_path}/axes/twt_model_s"
        axis_order: str | list[str] = ["lateral", "twt"]
        amplitude_unit = "normalized_amplitude"
    elif variant.sample_domain == "depth":
        axis_path = f"{owner_path}/axes/tvdss_model_m"
        axis_order = "lateral,tvdss"
        amplitude_unit = "amplitude"
    else:
        raise ValueError(f"Unsupported variant domain: {variant.sample_domain!r}")
    for name, values, unit in (
        ("seismic_observed", variant.seismic_observed, amplitude_unit),
        ("positive_gain", variant.positive_gain, "ratio"),
        ("additive_noise", variant.additive_noise, amplitude_unit),
    ):
        _dataset(
            group,
            name,
            np.asarray(values, dtype=np.float32),
            unit=unit,
            sample_domain=variant.sample_domain,
            axis_path=axis_path,
            axis_order=axis_order,
        )
    if variant.sample_domain == "time":
        qc = group.create_group("qc")
        for key, value in variant.qc.items():
            qc.attrs[key] = value
    path = f"{owner_path}/seismic_variants/{variant.variant_id}"
    return ArtifactReference(
        hdf5_group=path,
        seismic_input_dataset=f"{path}/seismic_observed",
        seismic_model_consistent_dataset=f"{owner_path}/seismic/seismic_model_consistent",
        valid_mask_dataset=f"{owner_path}/masks/valid_mask",
    )


__all__ = [
    "ArtifactReference",
    "write_benchmark_sample",
    "write_benchmark_variant",
]
