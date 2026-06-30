"""Generation orchestration for synthoseis-lite benchmark suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import scipy

from cup.seismic.wavelet import (
    infer_wavelet_dt,
    load_wavelet_csv,
    validate_wavelet_normalization,
)
from cup.synthetic.calibration import (
    GENERATOR_FAMILY,
    ImpedanceCalibration,
    load_calibration,
)
from cup.synthetic.canonical import (
    CANONICAL_FAMILIES,
    canonical_reference_impedance,
    canonical_scenarios,
    generate_canonical_section,
)
from cup.synthetic.config import DATA_SCHEMA, IMPLEMENTATION_SCOPE
from cup.synthetic.core import (
    build_attempt_plan,
    file_chain_sha256,
    geometry_feasibility_rows,
    rejection_reason_summary,
)
from cup.synthetic.forward import (
    HighresForwardResult,
    HighresWavelet,
    antialias_taps,
    highres_forward_to_model_grid,
    model_grid_closure_qc,
    resample_wavelet_to_highres,
)
from cup.synthetic.figures import write_generation_figures
from cup.synthetic.generation import (
    GenerationRejected,
    GenerationScenario,
    generate_field_section,
)
from cup.synthetic.geometry import build_section_geometries
from cup.synthetic.io import (
    write_generated_section,
    write_highres_forward_result,
    write_lfm_result,
    write_probe_result,
    write_seismic_variant_result,
)
from cup.synthetic.lfm import LfmResult, derive_lfm_priors
from cup.synthetic.probes import (
    ProbeFrequency,
    build_probe_frequency_catalog,
    frequency_catalog_rows,
    generate_probe,
    probe_variants,
)
from cup.synthetic.seismic_variants import generate_seismic_variants
from cup.config.workflow import WorkflowConfig
from cup.utils.io import (
    array_sha256,
    repo_relative_path,
    resolve_relative_path,
    sha256_file,
    write_json,
)


def _validate_calibration_horizon_contract(
    calibration: ImpedanceCalibration,
    script_cfg: Mapping[str, Any],
) -> None:
    configured = tuple(str(item["name"]) for item in script_cfg["horizons"])
    if configured != calibration.ordered_horizons:
        raise ValueError(
            "impedance_calibration_horizon_mismatch:"
            f"configured={list(configured)}:calibrated={list(calibration.ordered_horizons)}"
        )


def _time_forward_model_inputs_sha256(sources: Mapping[str, Path]) -> str:
    return file_chain_sha256(
        {
            "selected_wavelet.csv": sources["wavelet_generation_dir"]
            / "selected_wavelet.csv",
            "selected_wavelet_summary.json": sources["wavelet_generation_dir"]
            / "selected_wavelet_summary.json",
            "frequency_evidence_bands.csv": sources["forward_observability_dir"]
            / "frequency_evidence_bands.csv",
            "well_frequency_sensitivity.csv": sources["forward_observability_dir"]
            / "well_frequency_sensitivity.csv",
        }
    )


def generation_scenarios(script_cfg: Mapping[str, Any]) -> list[GenerationScenario]:
    generation = script_cfg["generation"]
    impedance = script_cfg["impedance"]
    scenarios: list[GenerationScenario] = []
    for duration_mode in generation["duration_modes"]:
        for correlation in impedance["correlation_length_section_fractions"]:
            pairs = zip(
                impedance["coefficient_sigma_multipliers"],
                impedance["thickness_log_sigma_values"],
            )
            for coefficient_sigma, thickness_sigma in pairs:
                for family in generation["geometry_families"]:
                    directions = (
                        generation["geometry_directions"]
                        if family != "none"
                        else ["none"]
                    )
                    variants = ["035", "065"] if family == "pinchout" else [""]
                    for direction in directions:
                        for variant in variants:
                            scenario_id = (
                                f"{duration_mode}__lx{correlation:g}__a{coefficient_sigma:g}"
                                f"__t{thickness_sigma:g}__{family}__{direction}"
                                + (f"__{variant}" if variant else "")
                            )
                            scenarios.append(
                                GenerationScenario(
                                    scenario_id=scenario_id,
                                    duration_mode=str(duration_mode),
                                    geometry_family=str(family),
                                    geometry_direction=str(direction),
                                    correlation_length_fraction=float(correlation),
                                    coefficient_sigma_multiplier=float(
                                        coefficient_sigma
                                    ),
                                    thickness_log_sigma=float(thickness_sigma),
                                    variant_id=str(variant),
                                )
                            )
    return scenarios


def _load_probe_frequencies(
    *,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> list[ProbeFrequency]:
    config = script_cfg["probe_selection"]
    if not bool(config["enabled"]):
        return []
    evidence = pd.read_csv(
        sources["forward_observability_dir"] / "frequency_evidence_bands.csv"
    )
    sensitivity = pd.read_csv(
        sources["forward_observability_dir"] / "well_frequency_sensitivity.csv"
    )
    return build_probe_frequency_catalog(
        evidence,
        sensitivity,
        weak_representatives_per_band=int(config["weak_representatives_per_band"]),
        unsupported_representatives_per_band=int(
            config["unsupported_representatives_per_band"]
        ),
        minimum_clusters=int(config["minimum_noise_equivalent_clusters"]),
    )


def _section_forward_qc(
    section: Any,
    *,
    wavelet: np.ndarray,
    highres_wavelet: HighresWavelet | None,
    required: bool,
) -> tuple[HighresForwardResult | None, dict[str, Any]]:
    qc = model_grid_closure_qc(
        section.model_target_log_ai,
        section.seismic_model_consistent,
        wavelet,
    )
    if highres_wavelet is None:
        return None, {
            **qc,
            "highres_forward_status": "disabled",
            "highres_forward_reasons": "",
        }
    try:
        result = highres_forward_to_model_grid(
            section.truth_log_ai_highres,
            section.seismic_model_consistent,
            highres_wavelet=highres_wavelet,
            forward_valid_mask_model=section.forward_valid_mask_model,
        )
        return result, {
            **qc,
            **result.qc,
            "highres_forward_reasons": "",
        }
    except Exception as exc:
        if required:
            raise ValueError(f"highres_forward_qc_failed:{exc}") from exc
        return None, {
            **qc,
            "highres_forward_status": "failed",
            "highres_forward_reasons": f"{type(exc).__name__}:{exc}",
        }


def _lfm_records(result: LfmResult, *, base_path: str) -> dict[str, Any]:
    return {
        **result.qc,
        "lfm_versions": "ideal;controlled_degraded",
        "lfm_ideal_dataset": ("" if not base_path else f"{base_path}/priors/lfm_ideal"),
        "lfm_controlled_degraded_dataset": (
            "" if not base_path else f"{base_path}/priors/lfm_controlled_degraded"
        ),
        "residual_vs_lfm_ideal_dataset": (
            "" if not base_path else f"{base_path}/residuals/residual_vs_lfm_ideal"
        ),
        "residual_vs_lfm_controlled_degraded_dataset": (
            ""
            if not base_path
            else f"{base_path}/residuals/residual_vs_lfm_controlled_degraded"
        ),
    }


def _seismic_variant_records_for_sample(
    *,
    h5: h5py.File,
    owner_path: str,
    source_index_record: Mapping[str, Any],
    seismic_model_consistent: np.ndarray,
    forward_valid_mask: np.ndarray,
    lateral_m: np.ndarray,
    script_cfg: Mapping[str, Any],
    qc_only: bool,
    source_variant_id: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = generate_seismic_variants(
        seismic_model_consistent=seismic_model_consistent,
        forward_valid_mask=forward_valid_mask,
        lateral_m=lateral_m,
        config=script_cfg["seismic_mismatch"],
        global_seed=int(script_cfg["global_seed"]),
        generator_family=GENERATOR_FAMILY,
        realization_id=str(source_index_record["parent_realization_id"]),
        source_variant_id=source_variant_id or str(source_index_record["sample_id"]),
    )
    index_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    source_sample_id = str(source_index_record["sample_id"])
    source_kind = str(source_index_record.get("sample_kind", "base"))
    for result in results:
        variant_group = (
            ""
            if qc_only
            else write_seismic_variant_result(
                h5,
                owner_path=owner_path,
                result=result,
            )
        )
        sample_id = f"{source_sample_id}__seismic__{result.variant_id}"
        sample_kind = (
            "frequency_probe_seismic_variant"
            if source_kind == "frequency_probe"
            else "seismic_variant"
        )
        record = {
            **dict(source_index_record),
            "sample_id": sample_id,
            "realization_id": sample_id,
            "source_sample_id": source_sample_id,
            "source_sample_kind": source_kind,
            "sample_kind": sample_kind,
            "hdf5_group": variant_group,
            "seismic_variant_id": result.variant_id,
            "seismic_mismatch_family": result.mismatch_family,
            "seismic_observed_dataset": (
                "" if not variant_group else f"{variant_group}/seismic_observed"
            ),
            "positive_gain_dataset": (
                "" if not variant_group else f"{variant_group}/positive_gain"
            ),
            "additive_noise_dataset": (
                "" if not variant_group else f"{variant_group}/additive_noise"
            ),
        }
        index_records.append(record)
        result_records.append({**record, **result.qc})
    return index_records, result_records


def _probe_records_for_parent(
    *,
    h5: h5py.File,
    parent_path: str,
    section: Any,
    suite: str,
    section_id: str,
    split: str,
    evaluation_role: str,
    frequencies: Sequence[ProbeFrequency],
    script_cfg: Mapping[str, Any],
    wavelet: np.ndarray,
    highres_wavelet: HighresWavelet | None,
    base_highres_forward: HighresForwardResult | None,
    qc_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = script_cfg["probe_selection"]
    taps = antialias_taps(int(script_cfg["sampling"]["vertical_oversampling_factor"]))
    index_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    seismic_variant_records: list[dict[str, Any]] = []
    for frequency in frequencies:
        variants = probe_variants(
            frequency,
            amplitude_multipliers=config["amplitude_multipliers"],
            phases=config["phases"],
            lateral_shapes=config["lateral_shapes"],
        )
        for variant in variants:
            result = generate_probe(
                section,
                frequency,
                variant,
                wavelet=wavelet,
                antialias_filter_taps=taps,
                vertical_tukey_alpha=float(config["vertical_tukey_alpha"]),
                lateral_shapes=config["lateral_shapes"],
                low_probe_energy_warning_fraction=float(
                    config["low_probe_energy_warning_fraction"]
                ),
            )
            closure = model_grid_closure_qc(
                result.model_target_log_ai,
                result.seismic_model_consistent,
                wavelet,
            )
            lfm_result = derive_lfm_priors(
                section,
                config=script_cfg["lfm"],
                global_seed=int(script_cfg["global_seed"]),
                generator_family=GENERATOR_FAMILY,
                model_target_log_ai=result.model_target_log_ai,
                degradation_variant_id=section.realization_id,
            )
            if variant.amplitude_multiplier == 0.0:
                highres_result = base_highres_forward
                highres_qc = (
                    {} if base_highres_forward is None else base_highres_forward.qc
                )
            elif highres_wavelet is not None:
                try:
                    highres_result = highres_forward_to_model_grid(
                        section.truth_log_ai_highres + result.probe_log_ai_highres,
                        result.seismic_model_consistent,
                        highres_wavelet=highres_wavelet,
                        forward_valid_mask_model=(section.forward_valid_mask_model),
                    )
                    highres_qc = highres_result.qc
                except Exception as exc:
                    if bool(script_cfg["forward_qc"]["highres_mismatch_required"]):
                        raise ValueError(
                            f"highres_forward_qc_failed:"
                            f"{section.realization_id}:{variant.variant_id}:{exc}"
                        ) from exc
                    highres_result = None
                    highres_qc = {
                        "highres_forward_status": "failed",
                        "highres_forward_reasons": (f"{type(exc).__name__}:{exc}"),
                    }
            else:
                highres_result = None
                highres_qc = {
                    "highres_forward_status": "disabled",
                    "highres_forward_reasons": "",
                }
            hdf5_group = (
                ""
                if qc_only
                else write_probe_result(
                    h5,
                    realization_path=parent_path,
                    frequency=frequency,
                    result=result,
                    highres_forward=highres_result,
                    lfm_result=lfm_result,
                )
            )
            sample_id = f"{section.realization_id}__probe__{variant.variant_id}"
            pair_id = (
                f"{section.realization_id}__probe__{variant.paired_zero_variant_id}"
            )
            index_record = {
                "sample_id": sample_id,
                "realization_id": sample_id,
                "parent_realization_id": section.realization_id,
                "suite": suite,
                "section_id": section_id,
                "scenario_id": "frequency_probe",
                "geometry_family": section.scenario.geometry_family,
                "duration_mode": section.scenario.duration_mode,
                "split": split,
                "evaluation_role": evaluation_role,
                "hdf5_group": hdf5_group,
                "attempt_id": "",
                "status": "ok",
                "reasons": "",
                "sample_kind": "frequency_probe",
                "probe_variant_id": variant.variant_id,
                "paired_zero_sample_id": pair_id,
                "probe_frequency_hz": frequency.frequency_hz,
                "probe_phase": variant.phase,
                "probe_lateral_shape": variant.lateral_shape,
                "probe_amplitude_multiplier": (variant.amplitude_multiplier),
                **_lfm_records(lfm_result, base_path=hdf5_group),
            }
            index_records.append(index_record)
            seismic_index, seismic_results = _seismic_variant_records_for_sample(
                h5=h5,
                owner_path=(
                    hdf5_group
                    if hdf5_group
                    else f"{parent_path}/probes/{variant.variant_id}"
                ),
                source_index_record=index_record,
                seismic_model_consistent=result.seismic_model_consistent,
                forward_valid_mask=section.forward_valid_mask_model,
                lateral_m=section.lateral_m,
                script_cfg=script_cfg,
                qc_only=qc_only,
                source_variant_id=variant.variant_id,
            )
            index_records.extend(seismic_index)
            seismic_variant_records.extend(seismic_results)
            result_records.append(
                {
                    **index_record,
                    "evidence_status": frequency.evidence_status,
                    "operator_support": frequency.operator_support,
                    "experiment_class": frequency.experiment_class,
                    "selection_reason": frequency.selection_reason,
                    "noise_equivalent_calibration_status": (
                        frequency.calibration_status
                    ),
                    "wavelet_uncertainty_warning": bool(
                        np.isfinite(frequency.conservative_to_nominal_ratio)
                        and frequency.conservative_to_nominal_ratio
                        > float(config["conservative_to_nominal_warning_ratio"])
                    ),
                    "valid_nominal_cluster_count": (
                        frequency.valid_nominal_cluster_count
                    ),
                    "valid_conservative_cluster_count": (
                        frequency.valid_conservative_cluster_count
                    ),
                    **result.qc,
                    **closure,
                    **highres_qc,
                    **lfm_result.qc,
                }
            )
    return index_records, result_records, seismic_variant_records


def _run_canonical_generation(
    *,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    calibration: ImpedanceCalibration,
    calibration_path: Path,
    repo_root: Path,
    output_dir: Path,
    wavelet_time: np.ndarray,
    wavelet: np.ndarray,
    output_dt: float,
    qc_only: bool,
) -> dict[str, Any]:
    config = script_cfg["canonical"]
    if not bool(config["enabled"]):
        raise ValueError("synthoseis_lite.geometry.canonical.enabled is false.")
    scenarios = canonical_scenarios(config)
    index_records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    qc_records: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    seismic_variant_records: list[dict[str, Any]] = []
    probe_frequencies = _load_probe_frequencies(
        script_cfg=script_cfg,
        sources=sources,
    )
    highres_wavelet = (
        resample_wavelet_to_highres(
            wavelet_time,
            wavelet,
            factor=int(script_cfg["sampling"]["vertical_oversampling_factor"]),
        )
        if bool(script_cfg["forward_qc"]["highres_mismatch_enabled"])
        else None
    )
    h5_path = output_dir / "synthetic_benchmark.h5"
    forward_model_inputs_sha256 = _time_forward_model_inputs_sha256(sources)
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema"] = DATA_SCHEMA
        h5.attrs["schema_version"] = DATA_SCHEMA
        h5.attrs["sample_domain"] = "time"
        h5.attrs["generator_family"] = calibration.generator_family
        h5.attrs["implementation_scope"] = IMPLEMENTATION_SCOPE
        h5.attrs["suite"] = "canonical"
        h5.attrs["forward_model_inputs_sha256"] = forward_model_inputs_sha256
        h5.attrs["impedance_calibration_sha256"] = sha256_file(calibration_path)
        h5.attrs["qc_only"] = bool(qc_only)
        for scenario in scenarios:
            generated = generate_canonical_section(
                calibration,
                scenario=scenario,
                config=config,
                output_dt_s=output_dt,
                wavelet_time_s=wavelet_time,
                wavelet=wavelet,
                vertical_oversampling_factor=int(
                    script_cfg["sampling"]["vertical_oversampling_factor"]
                ),
            )
            thickness_error = float(generated.qc["maximum_thickness_absolute_error_s"])
            if thickness_error > calibration.truth_dt_s + 1e-12:
                raise ValueError(
                    f"canonical_geometry_qc_failed:{scenario.scenario_id}:thickness_error"
                )
            if scenario.family == "pinchout":
                termination_error = float(generated.qc["termination_absolute_error_m"])
                termination_tolerance = float(
                    generated.qc["termination_grid_resolution_tolerance_m"]
                )
                if termination_error > termination_tolerance + 1e-12:
                    raise ValueError(
                        f"canonical_geometry_qc_failed:{scenario.scenario_id}:termination_error"
                    )
            highres_result, forward_qc = _section_forward_qc(
                generated,
                wavelet=wavelet,
                highres_wavelet=highres_wavelet,
                required=bool(script_cfg["forward_qc"]["highres_mismatch_required"]),
            )
            generated.qc.update(forward_qc)
            lfm_result = derive_lfm_priors(
                generated,
                config=script_cfg["lfm"],
                global_seed=int(script_cfg["global_seed"]),
                generator_family=GENERATOR_FAMILY,
                degradation_variant_id=generated.realization_id,
            )
            generated.qc.update(lfm_result.qc)
            hdf5_group = "" if qc_only else write_generated_section(h5, generated)
            if not qc_only and highres_result is not None:
                write_highres_forward_result(
                    h5,
                    realization_path=hdf5_group,
                    result=highres_result,
                )
            if not qc_only:
                write_lfm_result(
                    h5,
                    realization_path=hdf5_group,
                    result=lfm_result,
                )
            object_records.extend(generated.object_catalog)
            record = {
                "sample_id": scenario.scenario_id,
                "realization_id": scenario.scenario_id,
                "parent_realization_id": scenario.scenario_id,
                "suite": "canonical",
                "section_id": "canonical",
                "scenario_id": scenario.scenario_id,
                "geometry_family": scenario.family,
                "duration_mode": "canonical",
                "split": "benchmark",
                "evaluation_role": "development_pool",
                "hdf5_group": hdf5_group,
                "attempt_id": 0,
                "status": "ok",
                "reasons": "",
                "sample_kind": "base",
                "canonical_parameter_name": scenario.parameter_name,
                "canonical_parameter_value": scenario.parameter_value,
                "canonical_parameter_unit": scenario.parameter_unit,
                **_lfm_records(lfm_result, base_path=hdf5_group),
            }
            index_records.append(record)
            seismic_index, seismic_results = _seismic_variant_records_for_sample(
                h5=h5,
                owner_path=(
                    hdf5_group
                    if hdf5_group
                    else f"/realizations/{generated.realization_id}"
                ),
                source_index_record=record,
                seismic_model_consistent=generated.seismic_model_consistent,
                forward_valid_mask=generated.forward_valid_mask_model,
                lateral_m=generated.lateral_m,
                script_cfg=script_cfg,
                qc_only=qc_only,
                source_variant_id="base",
            )
            index_records.extend(seismic_index)
            seismic_variant_records.extend(seismic_results)
            qc_records.append({**record, **generated.qc})
            if scenario.family == "frequency_probe" and probe_frequencies:
                (
                    probe_index,
                    parent_probe_records,
                    probe_seismic_records,
                ) = _probe_records_for_parent(
                    h5=h5,
                    parent_path=(
                        hdf5_group
                        if hdf5_group
                        else f"/realizations/{generated.realization_id}"
                    ),
                    section=generated,
                    suite="canonical",
                    section_id="canonical",
                    split="benchmark",
                    evaluation_role="development_pool",
                    frequencies=probe_frequencies,
                    script_cfg=script_cfg,
                    wavelet=wavelet,
                    highres_wavelet=highres_wavelet,
                    base_highres_forward=highres_result,
                    qc_only=qc_only,
                )
                index_records.extend(probe_index)
                probe_records.extend(parent_probe_records)
                seismic_variant_records.extend(probe_seismic_records)

    index = pd.DataFrame.from_records(index_records)
    index.to_csv(output_dir / "sample_index.csv", index=False)
    object_columns = [
        "realization_id",
        "scenario_id",
        "zone_id",
        "object_id",
        "state",
        "state_id",
        "event_target",
        "minimum_duration_s",
        "maximum_duration_s",
        "minimum_truth_samples",
        "maximum_truth_samples",
        "canonical_family",
        "canonical_parameter_name",
        "canonical_parameter_value",
        "canonical_parameter_unit",
        "expected_termination_lateral_m",
        "expected_section_drop_s",
        "contrast_multiplier_start",
        "contrast_multiplier_end",
    ]
    pd.DataFrame.from_records(object_records, columns=object_columns).to_csv(
        output_dir / "object_catalog.csv",
        index=False,
    )
    qc_frame = pd.DataFrame.from_records(qc_records)
    qc_frame.to_csv(output_dir / "generation_qc.csv", index=False)
    qc_frame.to_csv(output_dir / "canonical_geometry_qc.csv", index=False)
    pd.DataFrame.from_records(probe_records).to_csv(
        output_dir / "frequency_probe_results.csv",
        index=False,
    )
    pd.DataFrame.from_records(seismic_variant_records).to_csv(
        output_dir / "seismic_variant_results.csv",
        index=False,
    )
    probe_frequency_frame = pd.DataFrame.from_records(
        frequency_catalog_rows(probe_frequencies)
    )
    if not probe_frequency_frame.empty:
        probe_frequency_frame["wavelet_uncertainty_warning"] = probe_frequency_frame[
            "conservative_to_nominal_ratio"
        ] > float(
            script_cfg["probe_selection"]["conservative_to_nominal_warning_ratio"]
        )
    probe_frequency_frame.to_csv(
        output_dir / "probe_frequency_catalog.csv",
        index=False,
    )
    rejection_columns = [
        "realization_id",
        "section_id",
        "scenario_id",
        "geometry_family",
        "attempt_id",
        "reason",
        "zone_id",
        "object_id",
        "state",
        "event_target",
        "count",
        "denominator",
        "fraction",
        "threshold",
        "metric",
        "value",
        "lower",
        "upper",
        "excess_ratio",
        "lateral_index",
    ]
    pd.DataFrame(columns=rejection_columns).to_csv(
        output_dir / "generation_rejection_details.csv",
        index=False,
    )
    catalog = index[index["sample_kind"].eq("base")][
        [
            "section_id",
            "scenario_id",
            "geometry_family",
            "canonical_parameter_name",
            "canonical_parameter_value",
            "canonical_parameter_unit",
            "status",
        ]
    ].copy()
    catalog["attempt_count"] = 1
    catalog["acceptance_fraction"] = 1.0
    catalog["acceptance_status"] = "fixed_public_benchmark"
    catalog.to_csv(output_dir / "scenario_catalog.csv", index=False)
    figure_summary = write_generation_figures(
        output_dir,
        script_cfg.get("figures", {}),
        suite="canonical",
        qc_only=qc_only,
    )
    reference = canonical_reference_impedance(calibration)
    manifest = {
        "schema": DATA_SCHEMA,
        "schema_version": DATA_SCHEMA,
        "sample_domain": "time",
        "generator_family": calibration.generator_family,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "suite": "canonical",
        "development_limited": False,
        "qc_only": bool(qc_only),
        "source_runs": {
            key: repo_relative_path(path, root=repo_root)
            for key, path in sources.items()
        },
        "forward_model_inputs_sha256": forward_model_inputs_sha256,
        "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
        "impedance_calibration_sha256": sha256_file(calibration_path),
        "global_seed": int(script_cfg["global_seed"]),
        "output_dt_s": output_dt,
        "truth_dt_s": calibration.truth_dt_s,
        "n_sections": 1,
        "n_scenarios": len(scenarios),
        "attempts_per_scenario": 1,
        "canonical_families": list(CANONICAL_FAMILIES),
        "canonical_config": dict(config),
        "canonical_reference_impedance": reference,
        "probe_selection": dict(script_cfg["probe_selection"]),
        "probe_frequency_count": len(probe_frequencies),
        "probe_variant_count": len(probe_records),
        "seismic_variant_count": len(seismic_variant_records),
        "forward_qc": dict(script_cfg["forward_qc"]),
        "lfm": dict(script_cfg["lfm"]),
        "seismic_mismatch": dict(script_cfg["seismic_mismatch"]),
        "highres_wavelet": (
            {}
            if highres_wavelet is None
            else {
                "dt_s": float(highres_wavelet.time_s[1] - highres_wavelet.time_s[0]),
                "n_samples": int(highres_wavelet.amplitude.size),
                "l2_energy": float(np.linalg.norm(highres_wavelet.amplitude)),
                "sha256": array_sha256(highres_wavelet.amplitude),
            }
        ),
        "antialias_filter": {
            "implementation": "scipy.signal.firwin/resample_poly",
            "scipy_version": scipy.__version__,
            "factor": int(script_cfg["sampling"]["vertical_oversampling_factor"]),
            "numtaps": int(
                antialias_taps(
                    int(script_cfg["sampling"]["vertical_oversampling_factor"])
                ).size
            ),
            "cutoff_output_nyquist_fraction": 0.9,
            "kaiser_beta": 8.6,
            "taps_sha256": array_sha256(
                antialias_taps(
                    int(script_cfg["sampling"]["vertical_oversampling_factor"])
                )
            ),
        },
        "probe_source_hashes": {
            "frequency_evidence_bands.csv": sha256_file(
                sources["forward_observability_dir"] / "frequency_evidence_bands.csv"
            ),
            "well_frequency_sensitivity.csv": sha256_file(
                sources["forward_observability_dir"] / "well_frequency_sensitivity.csv"
            ),
        },
        "not_yet_implemented": [],
        "figures": {
            "generated_count": int(figure_summary.get("generated_count", 0)),
            "skipped_count": int(figure_summary.get("skipped_count", 0)),
            "figure_manifest": repo_relative_path(
                Path(
                    str(
                        figure_summary.get(
                            "figure_manifest",
                            output_dir / "figures" / "figure_manifest.json",
                        )
                    )
                ),
                root=repo_root,
            ),
        },
        "files": {
            "synthetic_benchmark.h5": sha256_file(h5_path),
            "sample_index.csv": sha256_file(output_dir / "sample_index.csv"),
            "object_catalog.csv": sha256_file(output_dir / "object_catalog.csv"),
            "generation_qc.csv": sha256_file(output_dir / "generation_qc.csv"),
            "canonical_geometry_qc.csv": sha256_file(
                output_dir / "canonical_geometry_qc.csv"
            ),
            "frequency_probe_results.csv": sha256_file(
                output_dir / "frequency_probe_results.csv"
            ),
            "probe_frequency_catalog.csv": sha256_file(
                output_dir / "probe_frequency_catalog.csv"
            ),
            "seismic_variant_results.csv": sha256_file(
                output_dir / "seismic_variant_results.csv"
            ),
            "generation_rejection_details.csv": sha256_file(
                output_dir / "generation_rejection_details.csv"
            ),
            "scenario_catalog.csv": sha256_file(output_dir / "scenario_catalog.csv"),
            "figures/figure_manifest.json": sha256_file(
                output_dir / "figures" / "figure_manifest.json"
            ),
        },
    }
    write_json(output_dir / "benchmark_manifest.json", manifest)
    summary = {
        **manifest,
        "status": "ok",
        "accepted_realizations": len(scenarios),
        "rejected_realizations": 0,
        "failed_scenario_count": 0,
        "probe_variant_count": len(probe_records),
        "seismic_variant_count": len(seismic_variant_records),
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def run_generation(
    *,
    workflow: WorkflowConfig,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    calibration_path: Path,
    repo_root: Path,
    output_dir: Path,
    debug_attempt_limit: int | None = None,
    geometry_families: Sequence[str] | None = None,
    qc_only: bool = False,
    suite: str = "field_conditioned",
) -> dict[str, Any]:
    calibration = load_calibration(calibration_path)
    _validate_calibration_horizon_contract(calibration, script_cfg)
    for key, recorded in calibration.source_runs.items():
        if (
            key in sources
            and resolve_relative_path(recorded, root=repo_root).resolve()
            != sources[key].resolve()
        ):
            raise ValueError(f"impedance_calibration_source_mismatch:{key}")
    for relative_name, expected_hash in calibration.source_hashes.items():
        directory_key, filename = relative_name.split("/", maxsplit=1)
        if directory_key not in sources:
            raise ValueError(f"impedance_calibration_source_mismatch:{directory_key}")
        actual_hash = sha256_file(sources[directory_key] / filename)
        if actual_hash != expected_hash:
            raise ValueError(
                f"impedance_calibration_source_mismatch:sha256:{relative_name}"
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    wavelet_time, wavelet = load_wavelet_csv(
        sources["wavelet_generation_dir"] / "selected_wavelet.csv"
    )
    wavelet, qc = validate_wavelet_normalization(
        wavelet_time,
        wavelet,
        expected_l2_energy=1.0,
        l2_energy_tolerance=1e-5,
        max_center_abs_time_s=1e-9,
        allow_small_renormalization=True,
    )
    if qc.status != "ok":
        raise ValueError(f"invalid_wavelet:{qc.reasons}")
    output_dt = infer_wavelet_dt(wavelet_time)
    if suite == "canonical":
        if geometry_families:
            raise ValueError(
                "--geometry-family is only valid for the field_conditioned suite."
            )
        return _run_canonical_generation(
            script_cfg=script_cfg,
            sources=sources,
            calibration=calibration,
            calibration_path=calibration_path,
            repo_root=repo_root,
            output_dir=output_dir,
            wavelet_time=wavelet_time,
            wavelet=wavelet,
            output_dt=output_dt,
            qc_only=qc_only,
        )
    if suite != "field_conditioned":
        raise ValueError(f"Unsupported synthoseis-lite suite: {suite}")
    sections = build_section_geometries(
        workflow=workflow,
        script_cfg=script_cfg,
        repo_root=repo_root,
    )
    section_geometry_qc_path = output_dir / "section_geometry_qc.csv"
    section_geometry_qc = pd.DataFrame.from_records(
        [row for section in sections for row in section.qc_rows]
    )
    section_geometry_qc.to_csv(section_geometry_qc_path, index=False)
    scenarios = generation_scenarios(script_cfg)
    if geometry_families:
        selected = {str(value) for value in geometry_families}
        unknown = selected.difference({"none", "wedge", "pinchout"})
        if unknown:
            raise ValueError(f"Unsupported geometry filters: {sorted(unknown)}")
        scenarios = [
            scenario for scenario in scenarios if scenario.geometry_family in selected
        ]
        if not scenarios:
            raise ValueError("No generation scenarios remain after geometry filtering.")
    attempts = int(script_cfg["generation"]["attempts_per_scenario"])
    development_limited = debug_attempt_limit is not None
    if debug_attempt_limit is not None:
        attempts = min(attempts, int(debug_attempt_limit))
    held_out_geometry_family = str(script_cfg["splits"]["held_out_geometry_family"])
    attempt_plan = build_attempt_plan(
        section_ids=[str(section.section_id) for section in sections],
        scenarios=scenarios,
        attempts_per_scenario=attempts,
        held_out_geometry_family=held_out_geometry_family,
    )
    attempt_plan.to_csv(output_dir / "attempt_plan.csv", index=False)
    sections_by_id = {str(section.section_id): section for section in sections}
    scenarios_by_id = {str(scenario.scenario_id): scenario for scenario in scenarios}
    feasibility_path = output_dir / "section_geometry_feasibility_qc.csv"
    pd.DataFrame.from_records(
        geometry_feasibility_rows(
            sections=sections,
            ordered_horizons=[str(item["name"]) for item in script_cfg["horizons"]],
            vertical_axis_name="twt_s",
            minimum_highres_cells=int(script_cfg["impedance"]["minimum_highres_cells"]),
            highres_step=calibration.truth_dt_s,
            duration_reference="minimum",
        )
    ).to_csv(feasibility_path, index=False)
    index_records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    object_lateral_records: list[dict[str, Any]] = []
    qc_records: list[dict[str, Any]] = []
    rejection_records: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    seismic_variant_records: list[dict[str, Any]] = []
    probe_frequencies = _load_probe_frequencies(
        script_cfg=script_cfg,
        sources=sources,
    )
    highres_wavelet = (
        resample_wavelet_to_highres(
            wavelet_time,
            wavelet,
            factor=int(script_cfg["sampling"]["vertical_oversampling_factor"]),
        )
        if bool(script_cfg["forward_qc"]["highres_mismatch_enabled"])
        else None
    )
    probe_parent_counts = {section.section_id: 0 for section in sections}
    h5_path = output_dir / "synthetic_benchmark.h5"
    forward_model_inputs_sha256 = _time_forward_model_inputs_sha256(sources)
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema"] = DATA_SCHEMA
        h5.attrs["schema_version"] = DATA_SCHEMA
        h5.attrs["sample_domain"] = "time"
        h5.attrs["generator_family"] = calibration.generator_family
        h5.attrs["implementation_scope"] = IMPLEMENTATION_SCOPE
        h5.attrs["suite"] = "field_conditioned"
        h5.attrs["forward_model_inputs_sha256"] = forward_model_inputs_sha256
        h5.attrs["impedance_calibration_sha256"] = sha256_file(calibration_path)
        h5.attrs["qc_only"] = bool(qc_only)
        for plan_row in attempt_plan.to_dict(orient="records"):
            section = sections_by_id[str(plan_row["section_id"])]
            scenario = scenarios_by_id[str(plan_row["scenario_id"])]
            attempt_id = int(plan_row["attempt_id"])
            realization_id = str(plan_row["parent_realization_id"])
            evaluation_role = str(plan_row["evaluation_role"])
            hdf5_group = ""
            probe_index_local: list[dict[str, Any]] = []
            probe_records_local: list[dict[str, Any]] = []
            seismic_variant_records_local: list[dict[str, Any]] = []
            base_lfm_index: dict[str, Any] = {}
            try:
                minimum_truth_samples = (
                    2 if scenario.duration_mode == "ultra_thin_stress" else 4
                )
                generated = generate_field_section(
                    calibration,
                    realization_id=realization_id,
                    scenario=scenario,
                    global_seed=int(script_cfg["global_seed"]),
                    lateral_m=section.lateral_m,
                    inline_float=section.inline_float,
                    xline_float=section.xline_float,
                    x_m=section.x_m,
                    y_m=section.y_m,
                    horizon_twt_s=section.horizon_twt_s,
                    output_dt_s=output_dt,
                    wavelet=wavelet,
                    vertical_oversampling_factor=int(
                        script_cfg["sampling"]["vertical_oversampling_factor"]
                    ),
                    minimum_truth_samples=minimum_truth_samples,
                    max_global_reversal_fraction=float(
                        script_cfg["impedance"]["max_global_reversal_fraction"]
                    ),
                    max_object_reversal_fraction=float(
                        script_cfg["impedance"]["max_object_reversal_fraction"]
                    ),
                    max_global_clipping_fraction=float(
                        script_cfg["impedance"]["max_global_clipping_fraction"]
                    ),
                    max_object_clipping_fraction=float(
                        script_cfg["impedance"]["max_object_clipping_fraction"]
                    ),
                    sequence_minimum_duration_reference="minimum",
                )
                highres_result, forward_qc = _section_forward_qc(
                    generated,
                    wavelet=wavelet,
                    highres_wavelet=highres_wavelet,
                    required=bool(
                        script_cfg["forward_qc"]["highres_mismatch_required"]
                    ),
                )
                generated.qc.update(forward_qc)
                lfm_result = derive_lfm_priors(
                    generated,
                    config=script_cfg["lfm"],
                    global_seed=int(script_cfg["global_seed"]),
                    generator_family=GENERATOR_FAMILY,
                    degradation_variant_id=generated.realization_id,
                )
                generated.qc.update(lfm_result.qc)
                hdf5_group = "" if qc_only else write_generated_section(h5, generated)
                if not qc_only and highres_result is not None:
                    write_highres_forward_result(
                        h5,
                        realization_path=hdf5_group,
                        result=highres_result,
                    )
                if not qc_only:
                    write_lfm_result(
                        h5,
                        realization_path=hdf5_group,
                        result=lfm_result,
                    )
                base_lfm_index = _lfm_records(
                    lfm_result,
                    base_path=hdf5_group,
                )
                base_record_for_variants = {
                    "sample_id": realization_id,
                    "realization_id": realization_id,
                    "parent_realization_id": realization_id,
                    "suite": "field_conditioned",
                    "section_id": section.section_id,
                    "scenario_id": scenario.scenario_id,
                    "geometry_family": scenario.geometry_family,
                    "duration_mode": scenario.duration_mode,
                    "split": "",
                    "evaluation_role": evaluation_role,
                    "hdf5_group": hdf5_group,
                    "attempt_id": attempt_id,
                    "status": "ok",
                    "reasons": "",
                    "sample_kind": "base",
                    **base_lfm_index,
                }
                (
                    base_seismic_index,
                    base_seismic_results,
                ) = _seismic_variant_records_for_sample(
                    h5=h5,
                    owner_path=(
                        hdf5_group
                        if hdf5_group
                        else f"/realizations/{generated.realization_id}"
                    ),
                    source_index_record=base_record_for_variants,
                    seismic_model_consistent=generated.seismic_model_consistent,
                    forward_valid_mask=generated.forward_valid_mask_model,
                    lateral_m=generated.lateral_m,
                    script_cfg=script_cfg,
                    qc_only=qc_only,
                    source_variant_id="base",
                )
                probe_index_local.extend(base_seismic_index)
                seismic_variant_records_local.extend(base_seismic_results)
                probe_config = script_cfg["probe_selection"]
                is_probe_parent = (
                    probe_frequencies
                    and scenario.geometry_family
                    == str(probe_config["field_parent_geometry_family"])
                    and probe_parent_counts[section.section_id]
                    < int(probe_config["field_parents_per_section"])
                )
                if is_probe_parent:
                    try:
                        (
                            probe_index_extra,
                            probe_records_extra,
                            probe_seismic_records_extra,
                        ) = _probe_records_for_parent(
                            h5=h5,
                            parent_path=(
                                hdf5_group
                                if hdf5_group
                                else f"/realizations/{generated.realization_id}"
                            ),
                            section=generated,
                            suite="field_conditioned",
                            section_id=section.section_id,
                            split="benchmark",
                            evaluation_role=evaluation_role,
                            frequencies=probe_frequencies,
                            script_cfg=script_cfg,
                            wavelet=wavelet,
                            highres_wavelet=highres_wavelet,
                            base_highres_forward=highres_result,
                            qc_only=qc_only,
                        )
                        probe_index_local.extend(probe_index_extra)
                        probe_records_local.extend(probe_records_extra)
                        seismic_variant_records_local.extend(
                            probe_seismic_records_extra
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "frequency_probe_generation_failed:"
                            f"{generated.realization_id}:{exc}"
                        ) from exc
                    probe_parent_counts[section.section_id] += 1
                object_records.extend(generated.object_catalog)
                object_lateral_records.extend(generated.object_lateral_coefficients)
                status = "ok"
                reasons = ""
                qc_payload = generated.qc
            except GenerationRejected as exc:
                hdf5_group = ""
                status = "rejected"
                reasons = ";".join(exc.reasons)
                qc_payload = exc.diagnostics
                rejection_records.extend(
                    {
                        "realization_id": realization_id,
                        "section_id": section.section_id,
                        "scenario_id": scenario.scenario_id,
                        "geometry_family": scenario.geometry_family,
                        "attempt_id": attempt_id,
                        **detail,
                    }
                    for detail in exc.details
                )
            except Exception as exc:
                if str(exc).startswith(
                    (
                        "highres_forward_qc_failed",
                        "invalid_model_grid_forward",
                        "frequency_probe_generation_failed",
                    )
                ):
                    raise
                if hdf5_group and hdf5_group in h5:
                    del h5[hdf5_group]
                hdf5_group = ""
                status = "rejected"
                reasons = f"{type(exc).__name__}:{exc}"
                qc_payload = {}
            record = {
                "sample_id": realization_id,
                "realization_id": realization_id,
                "parent_realization_id": realization_id,
                "suite": "field_conditioned",
                "section_id": section.section_id,
                "scenario_id": scenario.scenario_id,
                "geometry_family": scenario.geometry_family,
                "duration_mode": scenario.duration_mode,
                "split": "",
                "evaluation_role": evaluation_role,
                "hdf5_group": hdf5_group,
                "attempt_id": attempt_id,
                "status": status,
                "reasons": reasons,
                "sample_kind": "base",
                **base_lfm_index,
            }
            index_records.append(record)
            if status == "ok":
                index_records.extend(probe_index_local)
                probe_records.extend(probe_records_local)
                seismic_variant_records.extend(seismic_variant_records_local)
            qc_records.append(
                {
                    **record,
                    **{
                        key: value
                        for key, value in qc_payload.items()
                        if key != "field_qc"
                    },
                }
            )
    index = pd.DataFrame.from_records(index_records)
    index.to_csv(output_dir / "sample_index.csv", index=False)
    object_columns = [
        "realization_id",
        "scenario_id",
        "zone_id",
        "object_id",
        "state",
        "state_id",
        "base_duration_fraction",
        "event_target",
        "duration_fraction_start",
        "duration_fraction_end",
        "minimum_duration_fraction",
        "maximum_duration_fraction",
        "minimum_duration_s",
        "maximum_duration_s",
        "minimum_truth_samples",
        "maximum_truth_samples",
        "event_multiplier_start",
        "event_multiplier_end",
        "minimum_event_multiplier",
        "maximum_event_multiplier",
        "reversal_fraction",
        "clipping_fraction",
        "profile_violation_fraction",
        "profile_projection_fraction",
        "mean_profile_projection_scale",
        "minimum_profile_projection_scale",
        "c0_conditioning_fraction",
        "mean_c0_conditioning_adjustment",
        "maximum_c0_conditioning_adjustment",
    ]
    pd.DataFrame.from_records(object_records, columns=object_columns).to_csv(
        output_dir / "object_catalog.csv",
        index=False,
    )
    object_lateral_columns = [
        "realization_id",
        "scenario_id",
        "zone_id",
        "local_object_index",
        "calibration_object_id",
        "object_id",
        "state",
        "state_id",
        "event_target",
        "lateral_index",
        "lateral_m",
        "c0",
        "c1",
        "c2",
        "thickness_fraction",
        "object_top_s",
        "object_bottom_s",
        "profile_projection_scale",
        "c0_conditioning_adjustment",
    ]
    pd.DataFrame.from_records(
        object_lateral_records,
        columns=object_lateral_columns,
    ).to_csv(output_dir / "object_lateral_coefficients.csv", index=False)
    pd.DataFrame.from_records(qc_records).to_csv(
        output_dir / "generation_qc.csv", index=False
    )
    pd.DataFrame.from_records(probe_records).to_csv(
        output_dir / "frequency_probe_results.csv",
        index=False,
    )
    pd.DataFrame.from_records(seismic_variant_records).to_csv(
        output_dir / "seismic_variant_results.csv",
        index=False,
    )
    probe_frequency_frame = pd.DataFrame.from_records(
        frequency_catalog_rows(probe_frequencies)
    )
    if not probe_frequency_frame.empty:
        probe_frequency_frame["wavelet_uncertainty_warning"] = probe_frequency_frame[
            "conservative_to_nominal_ratio"
        ] > float(
            script_cfg["probe_selection"]["conservative_to_nominal_warning_ratio"]
        )
    probe_frequency_frame.to_csv(
        output_dir / "probe_frequency_catalog.csv",
        index=False,
    )
    rejection_columns = [
        "realization_id",
        "section_id",
        "scenario_id",
        "geometry_family",
        "attempt_id",
        "reason",
        "zone_id",
        "object_id",
        "state",
        "event_target",
        "count",
        "denominator",
        "fraction",
        "threshold",
        "metric",
        "value",
        "lower",
        "upper",
        "excess_ratio",
        "lateral_index",
    ]
    pd.DataFrame.from_records(rejection_records, columns=rejection_columns).to_csv(
        output_dir / "generation_rejection_details.csv",
        index=False,
    )
    rejection_summary = rejection_reason_summary(
        pd.DataFrame.from_records(rejection_records, columns=rejection_columns),
        index,
    )
    rejection_summary_path = output_dir / "rejection_reason_summary.csv"
    rejection_summary.to_csv(rejection_summary_path, index=False)
    catalog = (
        index[index["sample_kind"].eq("base")]
        .groupby(["section_id", "scenario_id"], dropna=False)["status"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    catalog["attempt_count"] = catalog.get("ok", 0) + catalog.get("rejected", 0)
    catalog["acceptance_fraction"] = catalog.get("ok", 0) / catalog["attempt_count"]
    if development_limited:
        catalog["acceptance_status"] = "development_limit_no_verdict"
    else:
        minimum = int(script_cfg["impedance"]["minimum_attempts_per_scenario"])
        warning = float(script_cfg["impedance"]["scenario_acceptance_warning_fraction"])
        failure = float(script_cfg["impedance"]["scenario_acceptance_failure_fraction"])
        catalog["acceptance_status"] = np.where(
            catalog["attempt_count"] < minimum,
            "insufficient_attempts_for_acceptance_qc",
            np.where(
                catalog["acceptance_fraction"] < failure,
                "failed",
                np.where(catalog["acceptance_fraction"] < warning, "warning", "ok"),
            ),
        )
    catalog.to_csv(output_dir / "scenario_catalog.csv", index=False)
    figure_summary = write_generation_figures(
        output_dir,
        script_cfg.get("figures", {}),
        suite="field_conditioned",
        qc_only=qc_only,
    )
    manifest = {
        "schema": DATA_SCHEMA,
        "schema_version": DATA_SCHEMA,
        "generator_family": calibration.generator_family,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "development_limited": development_limited,
        "qc_only": bool(qc_only),
        "suite": "field_conditioned",
        "source_runs": {
            key: repo_relative_path(path, root=repo_root)
            for key, path in sources.items()
        },
        "sample_domain": "time",
        "forward_model_inputs_sha256": forward_model_inputs_sha256,
        "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
        "impedance_calibration_sha256": sha256_file(calibration_path),
        "global_seed": int(script_cfg["global_seed"]),
        "output_dt_s": output_dt,
        "truth_dt_s": calibration.truth_dt_s,
        "n_sections": len(sections),
        "n_scenarios": len(scenarios),
        "attempts_per_scenario": attempts,
        "geometry_filters": sorted(
            {scenario.geometry_family for scenario in scenarios}
        ),
        "field_geometry": {
            "mode": str(
                script_cfg.get("target_zone", {}).get("mode", "filled_target_zone")
            ),
            "target_zone": dict(script_cfg.get("target_zone") or {}),
            "section_geometry_qc": repo_relative_path(
                section_geometry_qc_path, root=repo_root
            ),
            "section_geometry_feasibility_qc": repo_relative_path(
                feasibility_path, root=repo_root
            ),
        },
        "probe_selection": dict(script_cfg["probe_selection"]),
        "probe_frequency_count": len(probe_frequencies),
        "probe_variant_count": len(probe_records),
        "probe_parent_counts": probe_parent_counts,
        "seismic_variant_count": len(seismic_variant_records),
        "forward_qc": dict(script_cfg["forward_qc"]),
        "lfm": dict(script_cfg["lfm"]),
        "seismic_mismatch": dict(script_cfg["seismic_mismatch"]),
        "highres_wavelet": (
            {}
            if highres_wavelet is None
            else {
                "dt_s": float(highres_wavelet.time_s[1] - highres_wavelet.time_s[0]),
                "n_samples": int(highres_wavelet.amplitude.size),
                "l2_energy": float(np.linalg.norm(highres_wavelet.amplitude)),
                "sha256": array_sha256(highres_wavelet.amplitude),
            }
        ),
        "antialias_filter": {
            "implementation": "scipy.signal.firwin/resample_poly",
            "scipy_version": scipy.__version__,
            "factor": int(script_cfg["sampling"]["vertical_oversampling_factor"]),
            "numtaps": int(
                antialias_taps(
                    int(script_cfg["sampling"]["vertical_oversampling_factor"])
                ).size
            ),
            "cutoff_output_nyquist_fraction": 0.9,
            "kaiser_beta": 8.6,
            "taps_sha256": array_sha256(
                antialias_taps(
                    int(script_cfg["sampling"]["vertical_oversampling_factor"])
                )
            ),
        },
        "probe_source_hashes": {
            "frequency_evidence_bands.csv": sha256_file(
                sources["forward_observability_dir"] / "frequency_evidence_bands.csv"
            ),
            "well_frequency_sensitivity.csv": sha256_file(
                sources["forward_observability_dir"] / "well_frequency_sensitivity.csv"
            ),
        },
        "not_yet_implemented": [],
        "split_policy": {
            "assignment_unit": "parent_realization",
            "held_out_geometry_family": held_out_geometry_family,
            "split_assignment_owner": "training",
        },
        "rejection_reason_summary": (
            []
            if rejection_summary.empty
            else rejection_summary.to_dict(orient="records")
        ),
        "sample_counts": {
            "by_evaluation_role": {
                str(key): int(value)
                for key, value in index[
                    index["sample_kind"].astype(str).eq("base")
                    & index["status"].astype(str).eq("ok")
                ]
                .groupby("evaluation_role")
                .size()
                .items()
            }
            if not index.empty and "evaluation_role" in index
            else {},
        },
        "figures": {
            "generated_count": int(figure_summary.get("generated_count", 0)),
            "skipped_count": int(figure_summary.get("skipped_count", 0)),
            "figure_manifest": repo_relative_path(
                Path(
                    str(
                        figure_summary.get(
                            "figure_manifest",
                            output_dir / "figures" / "figure_manifest.json",
                        )
                    )
                ),
                root=repo_root,
            ),
        },
        "files": {
            "synthetic_benchmark.h5": sha256_file(h5_path),
            "sample_index.csv": sha256_file(output_dir / "sample_index.csv"),
            "object_catalog.csv": sha256_file(output_dir / "object_catalog.csv"),
            "object_lateral_coefficients.csv": sha256_file(
                output_dir / "object_lateral_coefficients.csv"
            ),
            "generation_qc.csv": sha256_file(output_dir / "generation_qc.csv"),
            "frequency_probe_results.csv": sha256_file(
                output_dir / "frequency_probe_results.csv"
            ),
            "probe_frequency_catalog.csv": sha256_file(
                output_dir / "probe_frequency_catalog.csv"
            ),
            "seismic_variant_results.csv": sha256_file(
                output_dir / "seismic_variant_results.csv"
            ),
            "generation_rejection_details.csv": sha256_file(
                output_dir / "generation_rejection_details.csv"
            ),
            "rejection_reason_summary.csv": sha256_file(rejection_summary_path),
            "scenario_catalog.csv": sha256_file(output_dir / "scenario_catalog.csv"),
            "attempt_plan.csv": sha256_file(output_dir / "attempt_plan.csv"),
            "section_geometry_qc.csv": sha256_file(section_geometry_qc_path),
            "section_geometry_feasibility_qc.csv": sha256_file(feasibility_path),
            "figures/figure_manifest.json": sha256_file(
                output_dir / "figures" / "figure_manifest.json"
            ),
        },
    }
    write_json(output_dir / "benchmark_manifest.json", manifest)
    failure_statuses = {"failed", "insufficient_attempts_for_acceptance_qc"}
    failed_scenarios = catalog["acceptance_status"].isin(failure_statuses)
    summary = {
        **manifest,
        "status": (
            "development_limited"
            if development_limited
            else ("failed_acceptance_qc" if bool(failed_scenarios.any()) else "ok")
        ),
        "accepted_realizations": int(
            (index["sample_kind"].eq("base") & index["status"].eq("ok")).sum()
        ),
        "rejected_realizations": int(
            (index["sample_kind"].eq("base") & index["status"].eq("rejected")).sum()
        ),
        "failed_scenario_count": int(failed_scenarios.sum()),
        "probe_variant_count": len(probe_records),
        "seismic_variant_count": len(seismic_variant_records),
    }
    write_json(output_dir / "run_summary.json", summary)
    if (not development_limited) and bool(failed_scenarios.any()):
        failed = catalog.loc[
            failed_scenarios,
            ["section_id", "scenario_id", "acceptance_status"],
        ].to_dict(orient="records")
        raise RuntimeError(f"field_conditioned_acceptance_qc_failed:{failed}")
    return summary
