"""Depth field-conditioned pipeline and domain-specific variant orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import scipy
from scipy.signal import hilbert

from cup.impedance import (
    build_lfm_producer_contract,
    generation_contract,
)
from cup.petrel.load import import_interpretation_petrel
from cup.seismic.survey import open_survey
from cup.seismic.target_zone import TargetZone
from cup.seismic.wavelet import load_wavelet_csv
from cup.synthetic.core.calibration import ImpedanceCalibration
from cup.synthetic.core import build_attempt_plan as build_core_attempt_plan
from cup.synthetic.core import (
    build_lfm_degradation_metadata,
    build_seismic_variant_metadata,
    build_seismic_input_contract,
    build_mask_contract,
    geometry_feasibility_rows,
    limit_attempt_plan,
    rejection_reason_summary,
    validate_debug_attempt_limit,
)
from cup.synthetic.core.field_runner import (
    AttemptProgressLog,
    acceptance_enforcement,
    build_acceptance_catalog,
    configure_generation_logger,
    run_attempt_preflight,
    stable_records_frame,
)
from cup.synthetic.reporting.figures import write_generation_figures
from cup.synthetic.core.lfm import LfmPolicy
from cup.synthetic.core.geometry import resample_section_path, validate_line_geometry
from cup.synthetic.core.signal import finite_support_fir, valid_filter_decimate
from cup.synthetic.core.variant_runner import generate_seismic_variants
from cup.synthetic.core.random import RandomNamespace
from cup.synthetic.core.records import BenchmarkVariant, DepthForwardExtras
from cup.synthetic.core.rejections import StagedRejection, frozen_external_reason
from cup.synthetic.core.sample_builder import (
    BenchmarkBuildPolicy,
    BenchmarkBuilder,
    CanonicalIncrementPolicy,
)
from cup.synthetic.core.scenarios import GenerationScenario, generation_scenarios
from cup.synthetic.core.truth import TruthGenerationRequest, generate_field_conditioned_truth
from cup.synthetic.core.writer import write_benchmark_sample, write_benchmark_variant
from cup.synthetic.depth.config import CALIBRATION_SCHEMA, GENERATOR_FAMILY, SCHEMA_VERSION
from cup.synthetic.depth.model import DepthGeneratedSection, DepthSectionGeometry
from cup.synthetic.depth.calibration_adapter import (
    depth_catalog_from_synthetic_truth,
    load_depth_calibration_for_object_core,
)
from cup.synthetic.depth.forward_adapter import DepthForwardAdapter
from cup.synthetic.schemas import (
    RANDOM_STREAM_CONTRACT_VERSION,
    SCIENCE_CONTRACT,
    SCIENCE_REVISION,
    require_science_contract,
)
from cup.physics.execution import DepthForwardExecutor
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
    write_json,
)


def _survey(workflow: Any, *, repo_root: Path) -> Any:
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    path = resolve_relative_path(workflow.seismic.file, root=data_root)
    options = {
        key: value
        for key, value in workflow.seismic.as_dict().items()
        if key in {"iline", "xline", "istep", "xstep"}
    }
    survey = open_survey(path, workflow.seismic.type, segy_options=options or None)
    validate_line_geometry(survey.line_geometry)
    return survey


def _resample_path(
    points: Sequence[Mapping[str, float]],
    *,
    geometry: Any,
    interval_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return resample_section_path(points, geometry=geometry, sample_interval_m=interval_m)


def build_depth_sections(
    *, workflow: Any, script_cfg: Mapping[str, Any], repo_root: Path
) -> tuple[list[DepthSectionGeometry], Any]:
    survey = _survey(workflow, repo_root=repo_root)
    geometry = survey.describe_geometry(domain="depth")
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    raw = {
        str(item["name"]): import_interpretation_petrel(
            resolve_relative_path(item["file"], root=data_root)
        )
        for item in script_cfg["horizons"]
    }
    zone = TargetZone(
        raw,
        geometry,
        [str(item["name"]) for item in script_cfg["horizons"]],
        min_thickness=float(script_cfg["sampling"]["expected_model_dz_m"]),
    )
    surfaces = [
        zone.get_horizon_surface(str(item["name"])) for item in script_cfg["horizons"]
    ]
    sections: list[DepthSectionGeometry] = []
    for section in script_cfg["sections"]:
        lateral, il, xl, x, y = _resample_path(
            section["path"],
            geometry=survey.line_geometry,
            interval_m=float(script_cfg["lateral_sample_interval_m"]),
        )
        values = np.column_stack(
            [
                np.asarray(
                    [surface.value_at_line(i, j) for i, j in zip(il, xl)],
                    dtype=np.float64,
                )
                for surface in surfaces
            ]
        )
        if np.any(~np.isfinite(values)) or np.any(np.diff(values, axis=1) <= 0.0):
            raise ValueError(
                f"unsupported_or_crossing_horizons:{section['section_id']}"
            )
        qc = []
        for sample_index, (distance, i, j) in enumerate(zip(lateral, il, xl)):
            index_i = survey.line_geometry.inline_axis.index_of_line(float(i))
            index_j = survey.line_geometry.xline_axis.index_of_line(float(j))
            nearest_i = int(
                np.clip(round(index_i), 0, zone.no_support_mask.shape[0] - 1)
            )
            nearest_j = int(
                np.clip(round(index_j), 0, zone.no_support_mask.shape[1] - 1)
            )
            if bool(zone.no_support_mask[nearest_i, nearest_j]):
                raise ValueError(
                    f"section_has_no_interpretation_support:{section['section_id']}:{sample_index}"
                )
            for horizon_index, surface in enumerate(surfaces):
                sample = surface.sample_at_line(float(i), float(j))
                qc.append(
                    {
                        "section_id": section["section_id"],
                        "sample_index": sample_index,
                        "lateral_m": distance,
                        "inline_float": i,
                        "xline_float": j,
                        "inline_index_float": index_i,
                        "xline_index_float": index_j,
                        "inline_step": survey.line_geometry.inline_axis.step,
                        "xline_step": survey.line_geometry.xline_axis.step,
                        "horizon_name": script_cfg["horizons"][horizon_index]["name"],
                        "horizon_tvdss_m": sample.value,
                        "sample_method": sample.method,
                        "support_status": sample.support_status,
                    }
                )
        sections.append(
            DepthSectionGeometry(
                section_id=str(section["section_id"]),
                lateral_m=lateral,
                inline_float=il,
                xline_float=xl,
                x_m=x,
                y_m=y,
                horizon_tvdss_m=values,
                qc_rows=tuple(qc),
            )
        )
    return sections, survey


def _phase_rotate(wavelet: np.ndarray, degrees: float) -> np.ndarray:
    analytic = hilbert(np.asarray(wavelet, dtype=np.float64))
    return np.real(analytic * np.exp(1j * np.deg2rad(float(degrees))))


def _shift_wavelet(
    time_s: np.ndarray, amplitude: np.ndarray, shift_s: float
) -> np.ndarray:
    return np.interp(time_s - float(shift_s), time_s, amplitude, left=0.0, right=0.0)


def generate_depth_realization(
    calibration: ImpedanceCalibration,
    calibration_payload: Mapping[str, Any],
    *,
    section: DepthSectionGeometry,
    scenario: GenerationScenario,
    attempt_id: int,
    script_cfg: Mapping[str, Any],
    forward_inputs: Mapping[str, Any],
    survey: Any,
    repo_root: Path,
    forward_executor: DepthForwardExecutor | None = None,
    preflight_only: bool = False,
) -> DepthGeneratedSection | None:
    """Build one depth realization through the shared truth and base-sample Seam."""
    realization_id = f"{section.section_id}__{scenario.scenario_id}__a{attempt_id:03d}"
    wavelet_path = resolve_relative_path(
        forward_inputs["wavelet"]["path"], root=repo_root
    )
    wavelet_time, wavelet = load_wavelet_csv(wavelet_path)
    executor = forward_executor or DepthForwardExecutor(script_cfg["seismic_forward"])
    factor = int(script_cfg["sampling"]["vertical_oversampling_factor"])
    model_dz = float(script_cfg["sampling"]["expected_model_dz_m"])
    survey_axis = np.asarray(
        survey.sample_axis(domain="depth").values, dtype=np.float64
    )
    adapter = DepthForwardAdapter()
    preparation = adapter.prepare(
        horizon_tvdss_m=section.horizon_tvdss_m,
        survey_axis_m=survey_axis,
        wavelet_time_s=wavelet_time,
        wavelet=wavelet,
        model_dz_m=model_dz,
        vertical_oversampling_factor=factor,
        antialias_config=script_cfg["sampling"]["antialias"],
        maximum_allowed_vp_mps=float(calibration_payload["maximum_allowed_vp_mps"]),
        ai_velocity_relation=forward_inputs["ai_velocity_relation"],
        executor=executor,
    )
    namespace = RandomNamespace(
        benchmark_version=SCHEMA_VERSION,
        science_revision=SCIENCE_REVISION,
        random_stream_contract_version=RANDOM_STREAM_CONTRACT_VERSION,
        generator_family=calibration.generator_family,
    )
    truth = generate_field_conditioned_truth(
        calibration,
        TruthGenerationRequest(
            realization_id=realization_id,
            scenario=scenario,
            global_seed=int(script_cfg["global_seed"]),
            random_namespace=namespace,
            sample_domain="depth",
            axis_unit="m",
            lateral_m=section.lateral_m,
            inline_float=section.inline_float,
            xline_float=section.xline_float,
            x_m=section.x_m,
            y_m=section.y_m,
            horizon_coordinates=section.horizon_tvdss_m,
            model_sample_interval=model_dz,
            vertical_oversampling_factor=factor,
            minimum_highres_cells=int(script_cfg["impedance"]["minimum_highres_cells"]),
            vertical_axis_origin=float(survey_axis[0]),
            context_extent=preparation.required_context_extent,
            sequence_minimum_duration_reference="minimum",
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
        ),
    )
    if not np.array_equal(
        truth.highres_axis[::factor], preparation.model_axis.coordinates
    ):
        raise ValueError("Depth highres/model axes are not strictly nested.")
    if preflight_only:
        return None

    canonical_contract = generation_contract("depth", model_dz)
    sample = BenchmarkBuilder().build(
        truth=truth,
        preparation=preparation,
        forward_adapter=adapter,
        canonical_policy=CanonicalIncrementPolicy(contract=canonical_contract),
        lfm_policy=LfmPolicy(
            sample_domain="depth",
            axis_unit="m",
            global_seed=int(script_cfg["global_seed"]),
            random_namespace=namespace,
            realization_id=realization_id,
            horizon_coordinates=section.horizon_tvdss_m,
            controlled_degraded=script_cfg["lfm"]["controlled_degraded"],
        ),
        build_policy=BenchmarkBuildPolicy(
            domain_metadata={
                "sample_domain": "depth",
                "depth_basis": "tvdss",
                "increment_contract": canonical_contract.as_dict(),
            }
        ),
    )
    projected = sample.projected
    forward = sample.forward
    if not isinstance(forward.extras, DepthForwardExtras):
        raise TypeError("depth builder returned non-depth forward extras.")
    categorical = {
        "state_id_highres": truth.state_id_highres,
        "object_id_highres": truth.object_id_highres,
        "object_xi_highres": truth.object_xi_highres,
        "zone_id_highres": truth.zone_id_highres,
        "geometry_event_mask_highres": truth.geometry_event_mask_highres,
        "boundary_mask_highres": truth.boundary_mask_highres,
        "boundary_fraction_model": projected.boundary_fraction_model,
        "boundary_mask_model": projected.boundary_mask_model,
        "state_fraction_model": projected.state_fraction_model,
        "dominant_object_id_model": projected.dominant_object_id_model,
        "zone_id_model": projected.zone_id_model,
    }
    return DepthGeneratedSection(
        realization_id=realization_id,
        scenario=scenario,
        geometry=section,
        tvdss_highres_m=truth.highres_axis,
        tvdss_model_m=projected.model_axis.coordinates,
        log_ai_highres=truth.log_ai_highres,
        vp_highres_mps=forward.extras.vp_highres_mps,
        model_target_log_ai=projected.model_target_log_ai,
        vp_model_mps=forward.extras.vp_model_mps,
        seismic_observed=forward.seismic_observed,
        seismic_model_consistent=forward.seismic_model_consistent,
        subgrid_forward_residual=forward.subgrid_forward_residual,
        lfm_ideal=sample.input_lfm_canonical_log_ai,
        lfm_controlled_degraded=sample.input_lfm_controlled_degraded_log_ai,
        residual_vs_lfm_ideal=sample.residuals.residual_vs_lfm_ideal,
        residual_vs_lfm_controlled_degraded=(
            sample.residuals.residual_vs_lfm_controlled_degraded
        ),
        valid_mask_model=sample.valid_mask,
        categorical=categorical,
        object_catalog=depth_catalog_from_synthetic_truth(truth.object_catalog),
        object_lateral_coefficients=depth_catalog_from_synthetic_truth(
            truth.object_lateral_coefficients
        ),
        qc={key: value for key, value in sample.qc.items() if key != "field_qc"},
        benchmark_sample=sample,
    )


def build_attempt_plan(
    script_cfg: Mapping[str, Any],
    sections: Sequence[DepthSectionGeometry],
    *,
    geometry_families: Sequence[str] | None = None,
) -> pd.DataFrame:
    scenarios = generation_scenarios(script_cfg)
    if geometry_families:
        selected = {str(value) for value in geometry_families}
        configured = {
            str(value) for value in script_cfg["generation"]["geometry_families"]
        }
        unknown = sorted(selected - configured)
        if unknown:
            raise ValueError(f"Unsupported depth geometry filters: {unknown}")
        scenarios = [
            scenario for scenario in scenarios if scenario.geometry_family in selected
        ]
        if not scenarios:
            raise ValueError(
                "No depth generation scenarios remain after geometry filtering."
            )
    return build_core_attempt_plan(
        section_ids=[str(section.section_id) for section in sections],
        scenarios=scenarios,
        attempts_per_scenario=int(script_cfg["generation"]["attempts_per_scenario"]),
        held_out_geometry_family=str(script_cfg["splits"]["held_out_geometry_family"]),
        geometry_families=geometry_families,
    )


def _write_variants(
    h5: h5py.File | None,
    section: DepthGeneratedSection,
    *,
    script_cfg: Mapping[str, Any],
    forward_inputs: Mapping[str, Any],
    repo_root: Path,
    forward_executor: DepthForwardExecutor,
) -> list[dict[str, Any]]:
    """Depth configuration and Forward Adapters for the shared variant runner."""
    config = script_cfg["seismic_mismatch"]
    if not bool(config["enabled"]):
        return []
    wavelet_time, wavelet = load_wavelet_csv(
        resolve_relative_path(forward_inputs["wavelet"]["path"], root=repo_root)
    )
    factor = int(script_cfg["sampling"]["vertical_oversampling_factor"])
    taps = finite_support_fir(factor)
    depth_noise = dict(config["noise"])
    depth_gain = dict(config["gain"])
    common_config = {
        "enabled": True,
        "wavelet": dict(config["wavelet"]),
        "noise": {
            "white_noise_rms_fraction": float(depth_noise["white_noise_rms_fraction"]),
            "colored_noise_rms_fraction": float(depth_noise["colored_noise_rms_fraction"]),
            "colored_axis_correlation_length": float(
                depth_noise["colored_vertical_correlation_m"]
            ),
        },
        "gain": {
            "global_log_sigma": float(depth_gain["global_log_sigma"]),
            "tracewise_log_sigma": float(depth_gain["tracewise_log_sigma"]),
            "axis_lateral_log_sigma": float(depth_gain["vertical_lateral_log_sigma"]),
            "lateral_correlation_fraction": float(depth_gain["lateral_correlation_fraction"]),
            "axis_correlation_fraction": float(depth_gain["vertical_correlation_fraction"]),
        },
        "combined": {
            key: value
            for key, value in dict(config["combined"]).items()
            if key != "depth_static_m"
        },
    }

    def perturbed_forward(
        phase_degrees: float, shift_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rotated = _phase_rotate(wavelet, phase_degrees)
        perturbed = _shift_wavelet(wavelet_time, rotated, shift_s)
        highres = forward_executor(
            section.log_ai_highres,
            section.vp_highres_mps,
            section.tvdss_highres_m,
            wavelet_time,
            perturbed,
        )
        observed, support_1d = valid_filter_decimate(
            highres, factor=factor, taps=taps
        )
        observed = observed[..., : section.seismic_observed.shape[-1]]
        support = np.broadcast_to(
            support_1d[: observed.shape[-1]], observed.shape
        )
        base_mask = np.asarray(section.valid_mask_model, dtype=bool)
        if np.any(base_mask & ~support):
            raise ValueError(
                "invalid_seismic_variant:perturbed_wavelet_support_incomplete"
            )
        return observed, base_mask

    results = generate_seismic_variants(
        seismic_input=section.seismic_observed,
        valid_mask=section.valid_mask_model,
        lateral_m=section.geometry.lateral_m,
        sample_axis=section.tvdss_model_m,
        config=common_config,
        global_seed=int(script_cfg["global_seed"]),
        generator_family=GENERATOR_FAMILY,
        realization_id=section.realization_id,
        perturbed_wavelet_forward=perturbed_forward,
        axis_static_shifts=tuple(
            float(value) for value in config["depth_static"]["shift_m"]
        ),
        combined_axis_static_shift=float(
            dict(config["combined"]).get("depth_static_m", 0.0)
        ),
        base_operator_support=np.isfinite(section.seismic_observed),
    )
    rows: list[dict[str, Any]] = []
    for result in results:
        if h5 is not None:
            write_benchmark_variant(
                h5,
                BenchmarkVariant(
                    owner_realization_id=section.realization_id,
                    variant_id=result.variant_id,
                    sample_kind="seismic_variant",
                    seismic_observed=result.seismic_observed,
                    positive_gain=result.positive_gain,
                    additive_noise=result.additive_noise,
                    metadata={
                        "mismatch_family": result.mismatch_family,
                        "operator_source": result.operator_source,
                        "parameters": result.parameters,
                    },
                    qc=result.qc,
                    sample_domain="depth",
                ),
            )
        rows.append({
            **build_seismic_variant_metadata(
                variant_id=result.variant_id,
                mismatch_family=result.mismatch_family,
                operator_source=result.operator_source,
                parameters=result.parameters,
            ),
            **result.qc,
        })
    return rows

def run_depth_generation(
    *,
    workflow: Any,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    forward_inputs: Mapping[str, Any],
    config_provenance: Mapping[str, str],
    calibration_path: Path,
    repo_root: Path,
    output_dir: Path,
    debug_attempt_limit: int | None = None,
    geometry_families: Sequence[str] | None = None,
    qc_only: bool = False,
) -> dict[str, Any]:
    debug_attempt_limit = validate_debug_attempt_limit(debug_attempt_limit)
    calibration, calibration_payload = load_depth_calibration_for_object_core(
        calibration_path
    )
    calibration_summary_path = calibration_path.parent / "run_summary.json"
    with calibration_summary_path.open("r", encoding="utf-8") as handle:
        calibration_summary = json.load(handle)
    require_science_contract(calibration_summary, label="depth calibration run summary")
    if (
        calibration_summary.get("schema_version") != CALIBRATION_SCHEMA
        or calibration_summary.get("status") != "success"
    ):
        raise ValueError(
            f"Depth calibration run is not a successful {CALIBRATION_SCHEMA} contract."
        )
    recorded_calibration_path = resolve_relative_path(
        str(
            dict(calibration_summary.get("outputs") or {}).get(
                "impedance_calibration"
            )
            or ""
        ),
        root=repo_root,
    )
    if recorded_calibration_path.resolve() != calibration_path.resolve():
        raise ValueError(
            "Depth calibration run summary points to a different impedance_calibration.json."
        )
    calibration_contract_fingerprint = require_contract_fingerprint(
        calibration_summary, label=f"depth calibration {calibration_summary_path.parent}"
    )
    recorded_rock_contract = dict(
        calibration.input_contracts.get("rock_physics_analysis") or {}
    )
    if str(recorded_rock_contract.get("contract_fingerprint_sha256") or "") != str(
        forward_inputs["_rock_physics_contract_fingerprint_sha256"]
    ):
        raise ValueError("impedance calibration and forward inputs use different rock-physics contracts.")
    recorded_forward_contract = dict(
        calibration.input_contracts.get("depth_forward_model_inputs") or {}
    )
    if str(recorded_forward_contract.get("contract_fingerprint_sha256") or "") != str(
        forward_inputs["_contract_fingerprint_sha256"]
    ):
        raise ValueError(
            "impedance calibration and generation use different depth forward-model inputs."
        )
    input_contracts = {
        "calibration": {
            "path": repo_relative_path(calibration_summary_path, root=repo_root),
            "contract_fingerprint_sha256": calibration_contract_fingerprint,
        },
        "rock_physics_analysis": {
            "path": repo_relative_path(
                sources["rock_physics_analysis_dir"] / "run_summary.json",
                root=repo_root,
            ),
            "contract_fingerprint_sha256": str(
                forward_inputs["_rock_physics_contract_fingerprint_sha256"]
            ),
        },
        "depth_forward_model_inputs": {
            "path": repo_relative_path(
                sources["depth_forward_model_inputs_dir"] / "run_summary.json",
                root=repo_root,
            ),
            "contract_fingerprint_sha256": str(
                forward_inputs["_contract_fingerprint_sha256"]
            ),
        },
    }
    if list(calibration_payload.get("horizon_contract") or []) != list(
        script_cfg["horizons"]
    ):
        raise ValueError(
            "impedance calibration horizon contract differs from current common config."
        )
    expected_truth_dz = float(script_cfg["sampling"]["expected_model_dz_m"]) / int(
        script_cfg["sampling"]["vertical_oversampling_factor"]
    )
    if not np.isclose(
        float(calibration_payload["truth_dz_m"]),
        expected_truth_dz,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "impedance calibration truth_dz_m differs from current sampling config."
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_generation_logger(output_dir, sample_domain="depth")
    logger.info("Depth Synthoseis generation started")
    forward_executor = DepthForwardExecutor(script_cfg["seismic_forward"])
    logger.info(
        "Depth forward backend: requested=%s resolved=%s dtype=%s",
        forward_executor.requested,
        forward_executor.resolved,
        forward_executor.dtype,
    )
    sections, survey = build_depth_sections(
        workflow=workflow, script_cfg=script_cfg, repo_root=repo_root
    )
    pd.DataFrame([row for section in sections for row in section.qc_rows]).to_csv(
        output_dir / "section_geometry_qc.csv", index=False
    )
    feasibility_path = output_dir / "section_geometry_feasibility_qc.csv"
    pd.DataFrame.from_records(
        geometry_feasibility_rows(
            sections=sections,
            ordered_horizons=[str(item["name"]) for item in script_cfg["horizons"]],
            vertical_axis_name="tvdss_m",
            minimum_highres_cells=int(script_cfg["impedance"]["minimum_highres_cells"]),
            highres_step=expected_truth_dz,
            duration_reference="minimum",
        )
    ).to_csv(feasibility_path, index=False)
    plan = build_attempt_plan(script_cfg, sections, geometry_families=geometry_families)
    plan = limit_attempt_plan(plan, debug_attempt_limit)
    plan.to_csv(output_dir / "attempt_plan.csv", index=False)
    scenarios = {item.scenario_id: item for item in generation_scenarios(script_cfg)}
    sections_by_id = {item.section_id: item for item in sections}
    development = debug_attempt_limit is not None
    acceptance_qc = dict(script_cfg["generation"]["acceptance_qc"])

    def validate_attempt(row: Mapping[str, Any]) -> None:
        generate_depth_realization(
            calibration,
            calibration_payload,
            section=sections_by_id[str(row["section_id"])],
            scenario=scenarios[str(row["scenario_id"])],
            attempt_id=int(row["attempt_id"]),
            script_cfg=script_cfg,
            forward_inputs=forward_inputs,
            survey=survey,
            repo_root=repo_root,
            forward_executor=forward_executor,
            preflight_only=True,
        )

    preflight = run_attempt_preflight(
        plan,
        validator=validate_attempt,
        rejection_exceptions=(StagedRejection, ValueError, FloatingPointError),
        qc_config=acceptance_qc,
        output_dir=output_dir,
        logger=logger,
        development_limited=development,
        rejection_formatter=lambda exc: frozen_external_reason(
            exc, sample_domain="depth"
        ),
    )
    enforcement = acceptance_enforcement(acceptance_qc)
    preflight_summary = {
        "sample_domain": "depth",
        "status": "failed" if not preflight.failed.empty else "ok",
        "enforcement": enforcement,
        "planned_attempts": int(len(plan)),
        "accepted_attempts": int(len(preflight.accepted_plan)),
        "rejected_attempts": int(len(plan) - len(preflight.accepted_plan)),
        "failed_scenario_count": int(len(preflight.failed)),
    }
    write_json(output_dir / "preflight_summary.json", preflight_summary)
    if preflight.accepted_plan.empty:
        raise RuntimeError("depth_generation_preflight_no_accepted_realizations")
    if enforcement == "fail_fast" and not preflight.failed.empty:
        failed = preflight.failed[
            ["section_id", "scenario_id", "acceptance_status"]
        ].to_dict(orient="records")
        raise RuntimeError(f"depth_generation_preflight_acceptance_qc_failed:{failed}")
    if not preflight.failed.empty:
        logger.warning(
            "preflight acceptance QC has %d failed scenarios; enforcement=warn, "
            "generation will preserve accepted realizations",
            len(preflight.failed),
        )
    index_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = list(preflight.rejection_details)
    object_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    highres_rows: list[dict[str, Any]] = []
    subgrid_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    generation_qc_rows: list[dict[str, Any]] = []
    h5_path = output_dir / "synthetic_benchmark.h5"
    with AttemptProgressLog(
        output_dir / "attempt_progress.csv",
        phase="generation",
        plan=preflight.accepted_plan,
        qc_config=acceptance_qc,
        logger=logger,
        append=True,
    ) as production_progress, h5py.File(h5_path, "w") as h5:
        h5.attrs["schema"] = SCHEMA_VERSION
        h5.attrs["schema_version"] = SCHEMA_VERSION
        for key, value in SCIENCE_CONTRACT.items():
            h5.attrs[key] = value
        h5.attrs["sample_domain"] = "depth"
        h5.attrs["depth_basis"] = "tvdss"
        h5.attrs["axis_positive_direction"] = "down"
        h5.attrs["generator_family"] = GENERATOR_FAMILY
        h5.attrs["suite"] = "field_conditioned"
        h5.attrs["global_seed"] = int(script_cfg["global_seed"])
        h5.attrs["qc_only"] = bool(qc_only)
        for sequence_index, row in enumerate(
            preflight.accepted_plan.to_dict(orient="records"), start=1
        ):
            attempt_started = time.perf_counter()
            section = sections_by_id[str(row["section_id"])]
            scenario = scenarios[str(row["scenario_id"])]
            base_id = str(row["parent_realization_id"])
            common = {
                "sample_domain": "depth",
                "depth_basis": "tvdss",
                "suite": "field_conditioned",
                "parent_realization_id": base_id,
                "section_id": section.section_id,
                "scenario_id": scenario.scenario_id,
                "geometry_family": scenario.geometry_family,
                "geometry_direction": scenario.geometry_direction,
                "duration_mode": scenario.duration_mode,
                "attempt_id": int(row["attempt_id"]),
                "evaluation_role": row["evaluation_role"],
                "held_out_geometry_family": script_cfg["splits"][
                    "held_out_geometry_family"
                ],
            }
            progress_status = "rejected"
            progress_reason = ""
            try:
                generated = generate_depth_realization(
                    calibration,
                    calibration_payload,
                    section=section,
                    scenario=scenario,
                    attempt_id=int(row["attempt_id"]),
                    script_cfg=script_cfg,
                    forward_inputs=forward_inputs,
                    survey=survey,
                    repo_root=repo_root,
                    forward_executor=forward_executor,
                )
                if generated is None:
                    raise RuntimeError("depth_generation_returned_no_realization")
                if qc_only:
                    group_path = ""
                else:
                    if generated.benchmark_sample is None:
                        raise RuntimeError("depth_generation_missing_benchmark_sample")
                    group_path = write_benchmark_sample(
                        h5, generated.benchmark_sample
                    ).hdf5_group
                if generated.realization_id != base_id:
                    raise RuntimeError("depth_generation_parent_identity_changed")
                common.update({
                    "status": "ok",
                    "model_sample_count": generated.tvdss_model_m.size,
                    "model_dz_m": float(np.diff(generated.tvdss_model_m[:2])[0]),
                    "physics_halo_m": generated.qc["physics_halo_m"],
                    "physics_halo_samples": generated.qc["physics_halo_samples"],
                    "lfm_variant_id": "controlled_default",
                    "input_lfm_log_ai_dataset": (
                        "" if qc_only else f"{group_path}/priors/input_lfm_variants/controlled_default/log_ai"
                    ),
                })
                local_index_rows = [
                    {
                        **common,
                        "sample_id": base_id,
                        "sample_kind": "base",
                        "source_sample_id": "",
                        "hdf5_group": group_path,
                        "seismic_input_dataset": ""
                        if qc_only
                        else f"{group_path}/seismic/seismic_observed",
                        "seismic_model_consistent_dataset": ""
                        if qc_only
                        else f"{group_path}/seismic/seismic_model_consistent",
                        "valid_mask_dataset": ""
                        if qc_only
                        else f"{group_path}/masks/valid_mask",
                        "valid_sample_count": int(np.count_nonzero(generated.valid_mask_model)),
                    }
                ]
                local_variant_rows = []
                generated_variants = _write_variants(
                    None if qc_only else h5,
                    generated,
                    script_cfg=script_cfg,
                    forward_inputs=forward_inputs,
                    repo_root=repo_root,
                    forward_executor=forward_executor,
                )
                for variant in generated_variants:
                    variant_id = f"{base_id}__{variant['variant_id']}"
                    variant_path = (
                        ""
                        if qc_only
                        else f"{group_path}/seismic_variants/{variant['variant_id']}"
                    )
                    local_index_rows.append(
                        {
                            **common,
                            "sample_id": variant_id,
                            "sample_kind": "seismic_variant",
                            "seismic_variant_id": variant["variant_id"],
                            "source_sample_id": base_id,
                            "hdf5_group": variant_path,
                            "seismic_input_dataset": ""
                            if qc_only
                            else f"{variant_path}/seismic_observed",
                            "seismic_model_consistent_dataset": ""
                            if qc_only
                            else f"{group_path}/seismic/seismic_model_consistent",
                            "valid_mask_dataset": ""
                            if qc_only
                            else f"{group_path}/masks/valid_mask",
                            "valid_sample_count": int(np.count_nonzero(generated.valid_mask_model)),
                            "seismic_mismatch_family": variant["mismatch_family"],
                            "seismic_variant_operator_source": variant["operator_source"],
                            "seismic_variant_parameters_json": json.dumps(
                                {
                                    str(key): value
                                    for key, value in variant.items()
                                    if key not in {
                                        "variant_id",
                                        "mismatch_family",
                                        "operator_source",
                                    }
                                },
                                sort_keys=True,
                            ),
                            "mismatch_parameters_json": json.dumps(
                                {
                                    key: value
                                    for key, value in variant.items()
                                    if key not in {"variant_id", "mismatch_family"}
                                },
                                sort_keys=True,
                            ),
                        }
                    )
                    local_variant_rows.append(
                        {"parent_realization_id": base_id, **variant}
                    )
                local_highres_row = {
                    "parent_realization_id": base_id,
                    "physics_halo_m": generated.qc["physics_halo_m"],
                    "antialias_filter_half_width_m": generated.qc[
                        "antialias_filter_half_width_m"
                    ],
                    "context_m": generated.qc["context_m"],
                    "vertical_oversampling_factor": int(
                        script_cfg["sampling"]["vertical_oversampling_factor"]
                    ),
                    "highres_dz_m": float(np.diff(generated.tvdss_highres_m[:2])[0]),
                    "model_dz_m": float(np.diff(generated.tvdss_model_m[:2])[0]),
                    "antialias_numtaps": generated.qc["antialias_numtaps"],
                }
                local_subgrid_row = {
                    "parent_realization_id": base_id,
                    "seismic_observed_rms": generated.qc["seismic_observed_rms"],
                    "seismic_model_consistent_rms": generated.qc[
                        "seismic_model_consistent_rms"
                    ],
                    "subgrid_residual_rms": generated.qc["subgrid_residual_rms"],
                    "subgrid_residual_nrmse": generated.qc["subgrid_residual_nrmse"],
                    "subgrid_observed_model_correlation": generated.qc[
                        "subgrid_observed_model_correlation"
                    ],
                    "subgrid_amplitude_scale_ratio": generated.qc[
                        "subgrid_amplitude_scale_ratio"
                    ],
                }
                local_generation_qc_row = {
                    **common,
                    "sample_id": base_id,
                    "sample_kind": "base",
                    "reasons": "",
                    **generated.qc,
                }
                # Commit tabular records only after the complete HDF5 parent,
                # including every configured variant and QC row, is ready.
                index_rows.extend(local_index_rows)
                variant_rows.extend(local_variant_rows)
                object_rows.extend(generated.object_catalog)
                coefficient_rows.extend(generated.object_lateral_coefficients)
                highres_rows.append(local_highres_row)
                subgrid_rows.append(local_subgrid_row)
                generation_qc_rows.append(local_generation_qc_row)
                progress_status = "accepted"
            except (StagedRejection, ValueError, FloatingPointError) as exc:
                failed_group = f"/realizations/{base_id}"
                if (not qc_only) and failed_group in h5:
                    del h5[failed_group]
                reason = frozen_external_reason(exc, sample_domain="depth")
                index_rows.append(
                    {
                        **common,
                        "sample_id": base_id,
                        "sample_kind": "base",
                        "source_sample_id": "",
                        "hdf5_group": "",
                        "seismic_input_dataset": "",
                        "seismic_model_consistent_dataset": "",
                        "valid_mask_dataset": "",
                        "valid_sample_count": "",
                        "status": "rejected",
                        "reasons": reason,
                    }
                )
                rejection_rows.append(
                    {
                        **row,
                        "status": "rejected",
                        "reason": reason,
                    }
                )
                generation_qc_rows.append(
                    {
                        **common,
                        "sample_id": base_id,
                        "sample_kind": "base",
                        "status": "rejected",
                        "reasons": reason,
                    }
                )
                progress_reason = reason
            production_progress.record(
                row,
                sequence_index=sequence_index,
                status=progress_status,
                reason=progress_reason,
                elapsed_s=time.perf_counter() - attempt_started,
            )

    index = stable_records_frame(
        index_rows,
        sort_by=("section_id", "scenario_id", "attempt_id", "sample_kind", "sample_id"),
    )
    index.to_csv(output_dir / "sample_index.csv", index=False)
    stable_records_frame(
        object_rows,
        sort_by=("realization_id", "zone_id", "object_id"),
    ).to_csv(
        output_dir / "object_catalog.csv", index=False
    )
    stable_records_frame(
        coefficient_rows,
        sort_by=("realization_id", "zone_id", "object_id", "lateral_index"),
    ).to_csv(
        output_dir / "object_lateral_coefficients.csv", index=False
    )
    stable_records_frame(
        rejection_rows,
        sort_by=("section_id", "scenario_id", "attempt_id", "reason"),
    ).to_csv(
        output_dir / "generation_rejection_details.csv", index=False
    )
    stable_records_frame(
        highres_rows,
        sort_by=("parent_realization_id",),
    ).to_csv(
        output_dir / "highres_forward_qc.csv", index=False
    )
    stable_records_frame(
        subgrid_rows,
        sort_by=("parent_realization_id",),
    ).to_csv(
        output_dir / "subgrid_forward_qc.csv", index=False
    )
    stable_records_frame(
        variant_rows,
        sort_by=("parent_realization_id", "variant_id"),
    ).to_csv(
        output_dir / "seismic_variant_results.csv", index=False
    )
    stable_records_frame(
        generation_qc_rows,
        sort_by=("section_id", "scenario_id", "attempt_id", "sample_kind", "sample_id"),
    ).to_csv(
        output_dir / "generation_qc.csv", index=False
    )
    rejection_summary = rejection_reason_summary(
        stable_records_frame(
            rejection_rows,
            sort_by=("section_id", "scenario_id", "attempt_id", "reason"),
        ),
        index,
    )
    rejection_summary_path = output_dir / "rejection_reason_summary.csv"
    rejection_summary.to_csv(rejection_summary_path, index=False)
    base = (
        index[
            index.get("sample_kind", pd.Series(dtype=str)).eq("base")
            & index.get("status", pd.Series(dtype=str)).eq("ok")
        ].copy()
        if not index.empty
        else index
    )
    successful_parent_ids = (
        base["parent_realization_id"].astype(str)
        if not base.empty
        else pd.Series(dtype=str)
    )
    catalog = build_acceptance_catalog(
        plan,
        accepted_parent_ids=successful_parent_ids,
        qc_config=acceptance_qc,
        development_limited=development,
    )
    catalog.to_csv(output_dir / "scenario_catalog.csv", index=False)
    failed_scenarios = catalog["acceptance_status"].isin(
        {"failed", "insufficient_attempts"}
    )
    failure_reason = "depth_generation_no_accepted_realizations" if base.empty else ""
    if (
        not failure_reason
        and not development
        and enforcement == "fail_fast"
        and bool(failed_scenarios.any())
    ):
        failure_reason = "depth_generation_acceptance_qc_failed"
    completed_with_warnings = (
        (not development)
        and not failure_reason
        and bool(failed_scenarios.any())
    )

    figure_summary = write_generation_figures(
        output_dir,
        script_cfg.get("figures", {}),
        suite="field_conditioned",
        qc_only=qc_only,
    )

    contract_fields: dict[str, str] = {}
    if not failure_reason:
        contract_fields = {
            "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
            "contract_fingerprint_sha256": contract_fingerprint_sha256(
                contract_schema_version=SCHEMA_VERSION,
                semantics={
                    **SCIENCE_CONTRACT,
                    "sample_domain": "depth",
                    "sample_unit": "m",
                    "depth_basis": "tvdss",
                    "suite": "field_conditioned",
                    "generator_family": GENERATOR_FAMILY,
                    "sampling": dict(script_cfg["sampling"]),
                },
                business_config={
                    "global_seed": int(script_cfg["global_seed"]),
                    "generation": dict(script_cfg["generation"]),
                    "seismic_input": dict(script_cfg["seismic_input"]),
                    "seismic_forward": dict(script_cfg["seismic_forward"]),
                    "lfm": dict(script_cfg["lfm"]),
                    "seismic_mismatch": dict(script_cfg["seismic_mismatch"]),
                    "mask_contract": build_mask_contract(),
                },
                input_contracts=input_contracts,
                primary_artifacts={
                    "synthetic_benchmark": h5_path,
                    "sample_index": output_dir / "sample_index.csv",
                },
            ),
        }
    manifest = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        **SCIENCE_CONTRACT,
        "status": "failed"
        if failure_reason
        else (
            "development_limited"
            if development
            else ("completed_with_warnings" if completed_with_warnings else "success")
        ),
        **contract_fields,
        "failure_reason": failure_reason,
        "input_contracts": input_contracts,
        "sample_domain": "depth",
        "mask_contract": build_mask_contract(),
        "increment_contract": generation_contract("depth", float(script_cfg["sampling"]["expected_model_dz_m"])).as_dict(),
        "seismic_input_contract": build_seismic_input_contract(
            "depth", operator="depth_ai_vp_highres_forward_antialias"
        ),
        "seismic_forward": forward_executor.manifest_fields,
        "lfm_degradation": build_lfm_degradation_metadata(
            "depth",
            axis_unit="m",
            component_values=script_cfg["lfm"]["controlled_degraded"],
        ),
        "input_lfm_variants": ["canonical", "controlled_default"],
        "lfm_contract": build_lfm_producer_contract(
            generation_contract(
                "depth", float(script_cfg["sampling"]["expected_model_dz_m"])
            ),
            producer_schema=SCHEMA_VERSION,
            variant_selection={
                "selected": "controlled_default",
                "available": ["canonical", "controlled_default"],
            },
        ),
        "depth_basis": "tvdss",
        "generator_family": GENERATOR_FAMILY,
        "suite": "field_conditioned",
        "development_limited": development,
        "qc_only": bool(qc_only),
        "training_consumable": not bool(qc_only),
        "forward_model_inputs_path": str(forward_inputs["_path"]),
        "global_seed": int(script_cfg["global_seed"]),
        "n_sections": len(sections),
        "n_scenarios": int(plan["scenario_id"].nunique()),
        "attempts_per_scenario": min(
            int(script_cfg["generation"]["attempts_per_scenario"]),
            int(debug_attempt_limit or script_cfg["generation"]["attempts_per_scenario"]),
        ),
        "accepted_parent_realizations": int(len(base)),
        "rejected_parent_realizations": int(len(plan) - len(base)),
        "forward_inputs": {
            "wavelet_path": forward_inputs["wavelet"]["path"],
            "ai_velocity_relation_path": forward_inputs["ai_velocity_relation"]["path"],
            "shifted_las_sources": list(
                calibration_payload.get("shifted_las_sources") or []
            ),
        },
        "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
        "canonical_enabled": False,
        "geometry_filters": sorted({str(value) for value in geometry_families})
        if geometry_families
        else sorted(
            {str(value) for value in script_cfg["generation"]["geometry_families"]}
        ),
        "acceptance_qc": acceptance_qc,
        "preflight": preflight_summary,
        "sampling": dict(script_cfg["sampling"]),
        "lfm": dict(script_cfg["lfm"]),
        "seismic_mismatch": dict(script_cfg["seismic_mismatch"]),
        "source_runs": {
            key: repo_relative_path(path, root=repo_root)
            for key, path in sources.items()
        },
        "config_provenance": dict(config_provenance),
        "rejection_reason_summary": (
            []
            if rejection_summary.empty
            else rejection_summary.to_dict(orient="records")
        ),
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
        "split_policy": {
            "assignment_unit": "parent_realization",
            "held_out_geometry_family": script_cfg["splits"][
                "held_out_geometry_family"
            ],
            "split_assignment_owner": "training",
        },
        "sample_counts": {
            "by_evaluation_role": {
                str(key): int(value)
                for key, value in base.groupby("evaluation_role").size().items()
            }
            if not base.empty
            else {},
            "seen_parent_realizations": int(
                (
                    ~base["geometry_family"].eq(
                        script_cfg["splits"]["held_out_geometry_family"]
                    )
                ).sum()
            )
            if not base.empty
            else 0,
            "held_out_parent_realizations": int(
                base["geometry_family"]
                .eq(script_cfg["splits"]["held_out_geometry_family"])
                .sum()
            )
            if not base.empty
            else 0,
        },
        "random_stream": {
            "algorithm": "SHA-256/PCG64DXSM",
            "benchmark_version": SCHEMA_VERSION,
            "science_revision": SCIENCE_REVISION,
            "random_stream_contract_version": RANDOM_STREAM_CONTRACT_VERSION,
            "stream_purpose_registry": [
                "state_sequence",
                "duration",
                "zone_background",
                "coefficient_<name>",
                "coefficient_lateral",
                "thickness_lateral",
                "lfm_degradation/<component coefficient name>",
                "seismic_mismatch/<variant id>/<component coefficient name>",
            ],
        },
        "quality_warnings": (
            []
            if not completed_with_warnings
            else ["scenario_acceptance_qc_failed"]
        ),
    }
    write_json(output_dir / "benchmark_manifest.json", manifest)
    summary = {
        **manifest,
        "accepted_parent_realizations": int(len(base)),
        "rejected_parent_realizations": int(len(plan) - len(base)),
        "failed_scenario_count": int(failed_scenarios.sum()),
    }
    write_json(output_dir / "run_summary.json", summary)
    if failure_reason:
        raise RuntimeError(failure_reason)
    logger.info(
        "Depth Synthoseis generation finished: status=%s accepted=%d rejected=%d",
        summary["status"],
        summary["accepted_parent_realizations"],
        summary["rejected_parent_realizations"],
    )
    return summary


__all__ = ["build_attempt_plan", "build_depth_sections", "run_depth_generation"]
