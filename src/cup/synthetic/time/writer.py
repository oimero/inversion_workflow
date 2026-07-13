"""HDF5 and manifest I/O for synthoseis-lite development artifacts."""

from __future__ import annotations

import json
import h5py
import numpy as np

from cup.impedance import decompose_log_ai, generation_contract
from cup.synthetic.core import write_dataset
from cup.synthetic.core.generation import GeneratedSection
from cup.synthetic.time.forward import HighresForwardResult
from cup.synthetic.time.lfm import LfmResult
from cup.synthetic.time.seismic_variants import SeismicVariantResult


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
    if name in {
        "lateral_m",
        "inline_float",
        "xline_float",
        "x_m",
        "y_m",
        "twt_highres_s",
        "twt_model_s",
    }:
        values = np.asarray(values, dtype=np.float64)
    return write_dataset(
        group,
        name,
        values,
        unit=unit,
        sample_domain="time",
        axis_path=axis_dataset,
        axis_order=axis_order,
    )


def write_generated_section(h5: h5py.File, section: GeneratedSection) -> str:
    path = f"realizations/{section.realization_id}"
    root = h5.create_group(path)
    root.attrs["scenario_id"] = section.scenario.scenario_id
    root.attrs["status"] = str(section.qc.get("status", "ok"))
    root.attrs["suite"] = str(section.qc.get("suite", "field_conditioned"))
    axes = root.create_group("axes")
    _dataset(
        axes,
        "lateral_m",
        section.lateral_m,
        unit="m",
        domain="distance",
        axis_order=["lateral"],
    )
    _dataset(
        axes,
        "inline_float",
        section.inline_float,
        unit="line",
        domain="survey",
        axis_order=["lateral"],
    )
    _dataset(
        axes,
        "xline_float",
        section.xline_float,
        unit="line",
        domain="survey",
        axis_order=["lateral"],
    )
    _dataset(
        axes,
        "x_m",
        section.x_m,
        unit="m",
        domain="projected_xy",
        axis_order=["lateral"],
    )
    _dataset(
        axes,
        "y_m",
        section.y_m,
        unit="m",
        domain="projected_xy",
        axis_order=["lateral"],
    )
    _dataset(
        axes,
        "twt_highres_s",
        section.twt_highres_s,
        unit="s",
        domain="twt",
        axis_order=["twt"],
    )
    _dataset(
        axes,
        "twt_model_s",
        section.twt_model_s,
        unit="s",
        domain="twt",
        axis_order=["twt"],
    )
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
        (
            "truth_log_ai_highres",
            section.truth_log_ai_highres,
            "ln(m/s*g/cm3)",
            high_axis,
        ),
        (
            "model_target_log_ai",
            section.model_target_log_ai,
            "ln(m/s*g/cm3)",
            model_axis,
        ),
        ("rgt_highres", section.rgt_highres, "normalized_zone", high_axis),
        ("rgt_model", section.rgt_model, "normalized_zone", model_axis),
        ("state_id_highres", section.state_id_highres, "category", high_axis),
        ("object_id_highres", section.object_id_highres, "category", high_axis),
        (
            "object_xi_highres",
            section.object_xi_highres,
            "normalized_object",
            high_axis,
        ),
        ("zone_id_highres", section.zone_id_highres, "category", high_axis),
        (
            "geometry_event_mask_highres",
            section.geometry_event_mask_highres,
            "bool",
            high_axis,
        ),
        ("boundary_mask_highres", section.boundary_mask_highres, "bool", high_axis),
        (
            "boundary_fraction_model",
            section.boundary_fraction_model,
            "fraction",
            model_axis,
        ),
        ("boundary_mask_model", section.boundary_mask_model, "bool", model_axis),
        (
            "dominant_object_id_model",
            section.dominant_object_id_model,
            "category",
            model_axis,
        ),
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
    canonical_contract = generation_contract(
        "time", float(np.diff(np.asarray(section.twt_model_s, dtype=np.float64)[:2])[0])
    )
    canonical_background, target_increment = decompose_log_ai(
        np.asarray(section.model_target_log_ai, dtype=np.float64),
        np.asarray(section.twt_model_s, dtype=np.float64),
        canonical_contract,
        valid_mask=np.asarray(section.valid_mask_model, dtype=bool),
    )
    root.attrs["increment_contract_json"] = json.dumps(
        canonical_contract.as_dict(), sort_keys=True
    )
    priors = root.create_group("priors")
    targets = root.create_group("targets")
    _dataset(
        priors,
        "canonical_background_log_ai",
        canonical_background.astype(np.float32),
        unit="ln(m/s*g/cm3)",
        domain="twt",
        axis_order=["lateral", "twt"],
        axis_dataset=model_axis,
    )
    _dataset(
        targets,
        "target_increment_log_ai",
        target_increment.astype(np.float32),
        unit="ln(m/s*g/cm3)",
        domain="twt",
        axis_order=["lateral", "twt"],
        axis_dataset=model_axis,
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
    masks = root.create_group("masks")
    _dataset(
        masks,
        "valid_mask_model",
        section.valid_mask_model,
        unit="bool",
        domain="twt",
        axis_order=["lateral", "twt"],
        axis_dataset=model_axis,
    )
    physics_valid = np.zeros_like(section.valid_mask_model, dtype=bool)
    physics_valid[:, 0] = np.asarray(section.forward_valid_mask_model, dtype=bool)[:, 0]
    physics_valid[:, 1:] = np.asarray(section.forward_valid_mask_model, dtype=bool)
    physics_valid &= np.asarray(section.valid_mask_model, dtype=bool)
    _dataset(
        masks,
        "physics_valid_mask",
        physics_valid,
        unit="bool",
        domain="twt",
        axis_order=["lateral", "twt"],
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
    model = np.asarray(root["seismic/seismic_model_consistent"][()], dtype=np.float64)
    residual_model = np.asarray(result.seismic_model_grid, dtype=np.float64) - model
    model_axis = f"{realization_path}/axes/twt_model_s"
    target_shape = root["truth/model_target_log_ai"].shape
    padded = np.zeros(target_shape, dtype=np.float32)
    padded[:, 1:] = residual_model.astype(np.float32)
    _dataset(
        seismic,
        "subgrid_forward_residual",
        padded,
        unit="normalized_amplitude",
        domain="twt",
        axis_order=["lateral", "twt"],
        axis_dataset=model_axis,
    )
    qc = root.require_group("qc")
    for key, value in result.qc.items():
        qc.attrs[key] = value
    return f"{realization_path}/seismic/seismic_from_highres_truth_model_grid"


def write_lfm_result(
    h5: h5py.File,
    *,
    realization_path: str,
    result: LfmResult,
) -> dict[str, str]:
    root = h5[realization_path]
    axis = f"{realization_path}/axes/twt_model_s"
    priors = root.require_group("priors")
    residuals = root.require_group("residuals")
    paths = {}
    variants = priors.require_group("input_lfm_variants")
    variant_values = dict(result.input_lfm_variants or {})
    if not variant_values:
        variant_values = {"controlled_default": result.lfm_controlled_degraded}
    for variant_id, values in variant_values.items():
        variant_group = variants.require_group(str(variant_id))
        _dataset(
            variant_group,
            "log_ai",
            np.asarray(values, dtype=np.float32),
            unit="ln(m/s*g/cm3)",
            domain="twt",
            axis_order=["lateral", "twt"],
            axis_dataset=axis,
        )
    default_variant = "controlled_default" if "controlled_default" in variant_values else sorted(variant_values)[0]
    paths["input_lfm_log_ai_dataset"] = (
        f"{realization_path}/priors/input_lfm_variants/{default_variant}/log_ai"
    )
    paths["lfm_variant_id"] = default_variant
    for name, values in (
        ("lfm_ideal", result.lfm_ideal),
        ("lfm_controlled_degraded", result.lfm_controlled_degraded),
    ):
        _dataset(
            priors,
            name,
            values.astype(np.float32),
            unit="ln(m/s*g/cm3)",
            domain="twt",
            axis_order=["lateral", "twt"],
            axis_dataset=axis,
        )
        paths[name] = f"{realization_path}/priors/{name}"
    for name, values in (
        ("residual_vs_lfm_ideal", result.residual_vs_lfm_ideal),
        (
            "residual_vs_lfm_controlled_degraded",
            result.residual_vs_lfm_controlled_degraded,
        ),
    ):
        _dataset(
            residuals,
            name,
            values.astype(np.float32),
            unit="ln(m/s*g/cm3)",
            domain="twt",
            axis_order=["lateral", "twt"],
            axis_dataset=axis,
        )
        paths[name] = f"{realization_path}/residuals/{name}"
    qc = root.require_group("qc")
    for key, value in result.qc.items():
        qc.attrs[key] = value
    return paths


def write_seismic_variant_result(
    h5: h5py.File,
    *,
    owner_path: str,
    result: SeismicVariantResult,
) -> str:
    owner = h5[owner_path]
    variants = owner.require_group("seismic_variants")
    group = variants.create_group(result.variant_id)
    group.attrs["variant_id"] = result.variant_id
    group.attrs["mismatch_family"] = result.mismatch_family
    axis_root = owner_path
    forward_axis = f"{axis_root}/axes/twt_forward_model_s"
    _dataset(
        group,
        "seismic_observed",
        result.seismic_observed.astype(np.float32),
        unit="normalized_amplitude",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=forward_axis,
    )
    _dataset(
        group,
        "positive_gain",
        result.positive_gain.astype(np.float32),
        unit="ratio",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=forward_axis,
    )
    _dataset(
        group,
        "additive_noise",
        result.additive_noise.astype(np.float32),
        unit="normalized_amplitude",
        domain="twt",
        axis_order=["lateral", "twt_forward"],
        axis_dataset=forward_axis,
    )
    qc = group.create_group("qc")
    for key, value in result.qc.items():
        qc.attrs[key] = value
    return f"{owner_path}/seismic_variants/{result.variant_id}"
