"""HDF5 and manifest I/O for synthoseis-lite development artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from cup.synthetic.generation import GeneratedSection
from cup.synthetic.forward import HighresForwardResult
from cup.synthetic.probes import ProbeFrequency, ProbeResult


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
    *,
    unit: str,
    domain: str,
    axis_order: list[str],
    axis_dataset: str = "",
) -> h5py.Dataset:
    array = np.asarray(values)
    dataset = group.create_dataset(name, data=array, compression="gzip", shuffle=True)
    dataset.attrs["sha256"] = hashlib.sha256(
        np.ascontiguousarray(array).view(np.uint8).tobytes()
    ).hexdigest()
    dataset.attrs["unit"] = unit
    dataset.attrs["domain"] = domain
    dataset.attrs["axis_order"] = np.asarray(axis_order, dtype=h5py.string_dtype("utf-8"))
    if axis_dataset:
        dataset.attrs["axis_dataset"] = axis_dataset
    return dataset


def write_generated_section(h5: h5py.File, section: GeneratedSection) -> str:
    path = f"realizations/{section.realization_id}"
    root = h5.create_group(path)
    root.attrs["scenario_id"] = section.scenario.scenario_id
    root.attrs["status"] = str(section.qc.get("status", "ok"))
    root.attrs["suite"] = str(section.qc.get("suite", "field_conditioned"))
    axes = root.create_group("axes")
    _dataset(axes, "lateral_m", section.lateral_m, unit="m", domain="distance", axis_order=["lateral"])
    _dataset(axes, "inline_float", section.inline_float, unit="line", domain="survey", axis_order=["lateral"])
    _dataset(axes, "xline_float", section.xline_float, unit="line", domain="survey", axis_order=["lateral"])
    _dataset(axes, "x_m", section.x_m, unit="m", domain="projected_xy", axis_order=["lateral"])
    _dataset(axes, "y_m", section.y_m, unit="m", domain="projected_xy", axis_order=["lateral"])
    _dataset(axes, "twt_highres_s", section.twt_highres_s, unit="s", domain="twt", axis_order=["twt"])
    _dataset(axes, "twt_model_s", section.twt_model_s, unit="s", domain="twt", axis_order=["twt"])
    _dataset(
        axes,
        "twt_forward_highres_s",
        section.twt_highres_s[1:],
        unit="s",
        domain="twt",
        axis_order=["twt_forward"],
    )
    _dataset(
        axes,
        "twt_forward_model_s",
        section.twt_model_s[1:],
        unit="s",
        domain="twt",
        axis_order=["twt_forward"],
    )
    truth = root.create_group("truth")
    high_axis = f"/{path}/axes/twt_highres_s"
    model_axis = f"/{path}/axes/twt_model_s"
    high_forward_axis = f"/{path}/axes/twt_forward_highres_s"
    model_forward_axis = f"/{path}/axes/twt_forward_model_s"
    for name, values, unit, axis in [
        ("truth_log_ai_highres", section.truth_log_ai_highres, "ln(m/s*g/cm3)", high_axis),
        ("model_target_log_ai", section.model_target_log_ai, "ln(m/s*g/cm3)", model_axis),
        ("rgt_highres", section.rgt_highres, "normalized_zone", high_axis),
        ("rgt_model", section.rgt_model, "normalized_zone", model_axis),
        ("state_id_highres", section.state_id_highres, "category", high_axis),
        ("object_id_highres", section.object_id_highres, "category", high_axis),
        ("object_xi_highres", section.object_xi_highres, "normalized_object", high_axis),
        ("zone_id_highres", section.zone_id_highres, "category", high_axis),
        (
            "geometry_event_mask_highres",
            section.geometry_event_mask_highres,
            "bool",
            high_axis,
        ),
        ("boundary_mask_highres", section.boundary_mask_highres, "bool", high_axis),
        ("boundary_fraction_model", section.boundary_fraction_model, "fraction", model_axis),
        ("boundary_mask_model", section.boundary_mask_model, "bool", model_axis),
        ("dominant_object_id_model", section.dominant_object_id_model, "category", model_axis),
        ("zone_id_model", section.zone_id_model, "category", model_axis),
        ("valid_mask_model", section.valid_mask_model, "bool", model_axis),
    ]:
        _dataset(
            truth,
            name,
            values,
            unit=unit,
            domain="twt",
            axis_order=["lateral", "twt"],
            axis_dataset=axis,
        )
    _dataset(
        truth,
        "state_fraction_model",
        section.state_fraction_model,
        unit="fraction",
        domain="twt",
        axis_order=["lateral", "twt", "state"],
        axis_dataset=model_axis,
    )
    _dataset(
        truth,
        "reflectivity_highres",
        section.reflectivity_highres,
        unit="ratio",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=high_forward_axis,
    )
    _dataset(
        truth,
        "reflectivity_model",
        section.reflectivity_model,
        unit="ratio",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=model_forward_axis,
    )
    _dataset(
        truth,
        "forward_valid_mask_highres",
        section.forward_valid_mask_highres,
        unit="bool",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=high_forward_axis,
    )
    _dataset(
        truth,
        "forward_valid_mask_model",
        section.forward_valid_mask_model,
        unit="bool",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=model_forward_axis,
    )
    seismic = root.create_group("seismic")
    _dataset(
        seismic,
        "seismic_model_consistent",
        section.seismic_model_consistent,
        unit="normalized_amplitude",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=model_forward_axis,
    )
    return f"/{path}"


def write_highres_forward_result(
    h5: h5py.File,
    *,
    realization_path: str,
    result: HighresForwardResult,
) -> str:
    root = h5[realization_path]
    seismic = root["seismic"]
    axis = f"{realization_path}/axes/twt_forward_model_s"
    _dataset(
        seismic,
        "seismic_from_highres_truth_model_grid",
        result.seismic_model_grid.astype(np.float32),
        unit="normalized_amplitude",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=axis,
    )
    qc = root.require_group("qc")
    for key, value in result.qc.items():
        qc.attrs[key] = value
    return f"{realization_path}/seismic/seismic_from_highres_truth_model_grid"


def write_probe_result(
    h5: h5py.File,
    *,
    realization_path: str,
    frequency: ProbeFrequency,
    result: ProbeResult,
    highres_forward: HighresForwardResult | None,
) -> str:
    root = h5[realization_path]
    probes = root.require_group("probes")
    group = probes.create_group(result.variant.variant_id)
    group.attrs["frequency_hz"] = frequency.frequency_hz
    group.attrs["phase"] = result.variant.phase
    group.attrs["lateral_shape"] = result.variant.lateral_shape
    group.attrs["amplitude_multiplier"] = result.variant.amplitude_multiplier
    group.attrs["paired_zero_variant_id"] = (
        result.variant.paired_zero_variant_id
    )
    group.attrs["evidence_status"] = frequency.evidence_status
    group.attrs["operator_support"] = frequency.operator_support
    group.attrs["experiment_class"] = frequency.experiment_class
    group.attrs["calibration_status"] = frequency.calibration_status
    group.attrs["target_semantics"] = "base_model_target_plus_probe_increment"
    group.attrs["base_truth_dataset"] = (
        f"{realization_path}/truth/truth_log_ai_highres"
    )
    group.attrs["base_model_target_dataset"] = (
        f"{realization_path}/truth/model_target_log_ai"
    )
    truth = group.create_group("truth")
    high_axis = f"{realization_path}/axes/twt_highres_s"
    model_axis = f"{realization_path}/axes/twt_model_s"
    forward_axis = f"{realization_path}/axes/twt_forward_model_s"
    _dataset(
        truth,
        "probe_log_ai_highres",
        result.probe_log_ai_highres.astype(np.float32),
        unit="ln(m/s*g/cm3)",
        domain="twt",
        axis_order=["lateral", "twt"],
        axis_dataset=high_axis,
    )
    _dataset(
        truth,
        "probe_log_ai_model_grid",
        result.probe_log_ai_model_grid.astype(np.float32),
        unit="ln(m/s*g/cm3)",
        domain="twt",
        axis_order=["lateral", "twt"],
        axis_dataset=model_axis,
    )
    _dataset(
        truth,
        "reflectivity_model",
        result.reflectivity_model.astype(np.float32),
        unit="ratio",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=forward_axis,
    )
    seismic = group.create_group("seismic")
    _dataset(
        seismic,
        "seismic_model_consistent",
        result.seismic_model_consistent.astype(np.float32),
        unit="normalized_amplitude",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=forward_axis,
    )
    if highres_forward is not None:
        _dataset(
            seismic,
            "seismic_from_highres_truth_model_grid",
            highres_forward.seismic_model_grid.astype(np.float32),
            unit="normalized_amplitude",
            domain="twt",
            axis_order=["lateral", "twt_forward"],
            axis_dataset=forward_axis,
        )
    qc = group.create_group("qc")
    for key, value in {
        **result.qc,
        **({} if highres_forward is None else highres_forward.qc),
    }.items():
        qc.attrs[key] = value
    return f"{realization_path}/probes/{result.variant.variant_id}"
