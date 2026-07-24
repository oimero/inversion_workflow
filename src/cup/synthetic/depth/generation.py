"""Depth field-conditioned pipeline and domain-specific view orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from cup.impedance import generation_contract
from cup.petrel.load import import_interpretation_petrel
from cup.seismic.survey import open_survey
from cup.seismic.target_zone import TargetZone
from cup.seismic.wavelet import load_wavelet_csv
from cup.synthetic.core.calibration import ImpedanceCalibration
from cup.synthetic.core import (
    build_seismic_input_contract,
    build_mask_contract,
    geometry_feasibility_rows,
)
from cup.synthetic.core.field_runner import (
    stable_records_frame,
)
from cup.synthetic.core.lfm import LfmPolicy
from cup.synthetic.core.geometry import resample_section_path, validate_line_geometry
from cup.synthetic.core.signal import finite_support_fir, valid_filter_decimate
from cup.synthetic.core.pipeline import (
    GenerationAttempt,
    GenerationSession,
    SeismicViewContext,
)
from cup.synthetic.adapters import DepthSyntheticDomainAdapter
from cup.synthetic.core.pipeline import SyntheticBenchmarkPipeline
from cup.synthetic.core.random import RandomNamespace
from cup.synthetic.core.records import DepthForwardExtras
from cup.synthetic.core.rejections import ForwardRejected
from cup.synthetic.core.sample_builder import (
    BenchmarkBuildPolicy,
    BenchmarkBuilder,
    CanonicalIncrementPolicy,
)
from cup.synthetic.core.scenarios import GenerationScenario, generation_scenarios
from cup.synthetic.core.truth import TruthGenerationRequest, generate_field_conditioned_truth
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
    is_consumable_contract_status,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
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
                horizon_coordinates=values,
                sample_domain="depth",
                axis_unit="m",
                depth_basis="tvdss",
                xline_step=float(survey.line_geometry.xline_axis.step),
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
        ),
        build_policy=BenchmarkBuildPolicy(
            domain_metadata={
                "sample_domain": "depth",
                "depth_basis": "tvdss",
                "increment_contract": canonical_contract.as_dict(),
                "xline_step": float(section.xline_step),
                "lfm_source_identity": {
                    "kind": "synthetic_target_derived_lfm",
                    "construction": "canonical_background_component_of_target_decomposition",
                    "target_dependency": True,
                    "contract": canonical_contract.as_dict(),
                },
                "structured_identity": {
                    "producer": {
                        "name": "synthoseis_lite",
                        "artifact_type": "structured_truth_v1",
                    },
                    "calibration": {
                        "generator_family": calibration.generator_family,
                    },
                    "projection": {
                        "name": "cup.synthetic.core.projection",
                        "operator": "finite_support_fir_decimate",
                    },
                    "forward": {
                        "sample_domain": "depth",
                        "depth_basis": "tvdss",
                        "operator": "cup.physics.execution.DepthForwardExecutor",
                    },
                },
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
        canonical_background_log_ai=sample.input_lfm_canonical_log_ai,
        target_increment_log_ai=sample.target_increment_log_ai,
        valid_mask_model=sample.valid_mask,
        categorical=categorical,
        object_catalog=depth_catalog_from_synthetic_truth(truth.object_catalog),
        object_lateral_coefficients=depth_catalog_from_synthetic_truth(
            truth.object_lateral_coefficients
        ),
        qc={key: value for key, value in sample.qc.items() if key != "field_qc"},
        benchmark_sample=sample,
    )


class DepthGenerationSession:
    """Prepare depth-domain science for the shared parent lifecycle."""

    @classmethod
    def prepare(
        cls,
        script_cfg: Mapping[str, Any],
        calibration: ImpedanceCalibration,
        *,
        output_dir: Path,
        workflow: Any,
        sources: Mapping[str, Path],
        forward_inputs: Mapping[str, Any],
        config_provenance: Mapping[str, str],
        calibration_path: Path,
        amplitude_prior_path: Path | None = None,
        repo_root: Path,
        debug_attempt_limit: int | None = None,
        geometry_families: Sequence[str] | None = None,
        qc_only: bool = False,
        **_: Any,
    ) -> GenerationSession:
        calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_summary_path = calibration_path.parent / "run_summary.json"
        calibration_summary = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
        require_science_contract(calibration_summary, label="depth calibration run summary")
        if calibration_summary.get("schema_version") != CALIBRATION_SCHEMA or not is_consumable_contract_status(calibration_summary.get("status")):
            raise ValueError(f"Depth calibration run is not a consumable {CALIBRATION_SCHEMA} contract.")
        recorded_calibration_path = resolve_relative_path(
            str(dict(calibration_summary.get("outputs") or {}).get("impedance_calibration") or ""),
            root=repo_root,
        )
        if recorded_calibration_path.resolve() != calibration_path.resolve():
            raise ValueError("Depth calibration run summary points to a different impedance_calibration.json.")
        calibration_contract_fingerprint = require_contract_fingerprint(
            calibration_summary, label=f"depth calibration {calibration_summary_path.parent}"
        )
        if str(dict(calibration.input_contracts.get("rock_physics_analysis") or {}).get("contract_fingerprint_sha256") or "") != str(forward_inputs["_rock_physics_contract_fingerprint_sha256"]):
            raise ValueError("impedance calibration and forward inputs use different rock-physics contracts.")
        if str(dict(calibration.input_contracts.get("depth_forward_model_inputs") or {}).get("contract_fingerprint_sha256") or "") != str(forward_inputs["_contract_fingerprint_sha256"]):
            raise ValueError("impedance calibration and generation use different depth forward-model inputs.")
        input_contracts = {
            "calibration": {
                "path": repo_relative_path(calibration_summary_path, root=repo_root),
                "contract_fingerprint_sha256": calibration_contract_fingerprint,
            },
            "rock_physics_analysis": {
                "path": repo_relative_path(sources["rock_physics_analysis_dir"] / "run_summary.json", root=repo_root),
                "contract_fingerprint_sha256": str(forward_inputs["_rock_physics_contract_fingerprint_sha256"]),
            },
            "depth_forward_model_inputs": {
                "path": repo_relative_path(sources["depth_forward_model_inputs_dir"] / "run_summary.json", root=repo_root),
                "contract_fingerprint_sha256": str(forward_inputs["_contract_fingerprint_sha256"]),
            },
        }
        from cup.synthetic.core.amplitude_calibration import resolve_calibrated_seismic_views
        from cup.synthetic.depth.amplitude_calibration import depth_pilot_compatibility

        amplitude_pilot_compatibility = depth_pilot_compatibility(
            workflow=workflow,
            script_cfg=script_cfg,
            calibration_path=calibration_path,
            forward_inputs=forward_inputs,
            repo_root=repo_root,
        )
        resolved_views, amplitude_provenance = resolve_calibrated_seismic_views(
            script_cfg["seismic_views"],
            prior_path=amplitude_prior_path,
            repo_root=repo_root,
            sample_domain="depth",
            ordered_horizons=[str(item["name"]) for item in script_cfg["horizons"]],
            expected_pilot_compatibility=amplitude_pilot_compatibility,
        )
        if amplitude_provenance is not None:
            input_contracts["seismic_amplitude_prior"] = {
                "path": str(amplitude_provenance["path"]),
                "contract_fingerprint_sha256": str(
                    amplitude_provenance["contract_fingerprint_sha256"]
                ),
                "artifact_sha256": str(amplitude_provenance["artifact_sha256"]),
                "prior_sha256": str(amplitude_provenance["prior_sha256"]),
                "pilot_compatibility_sha256": str(
                    amplitude_provenance["pilot_compatibility_sha256"]
                ),
            }
        if list(calibration_payload.get("horizon_contract") or []) != list(script_cfg["horizons"]):
            raise ValueError("impedance calibration horizon contract differs from current common config.")
        expected_truth_dz = float(script_cfg["sampling"]["expected_model_dz_m"]) / int(script_cfg["sampling"]["vertical_oversampling_factor"])
        if not np.isclose(float(calibration_payload["truth_dz_m"]), expected_truth_dz, rtol=0.0, atol=1e-12):
            raise ValueError("impedance calibration truth_dz_m differs from current sampling config.")
        forward_executor = DepthForwardExecutor(script_cfg["seismic_forward"])
        view_wavelet_time, view_wavelet = load_wavelet_csv(
            resolve_relative_path(forward_inputs["wavelet"]["path"], root=repo_root)
        )
        view_factor = int(script_cfg["sampling"]["vertical_oversampling_factor"])
        view_taps = finite_support_fir(view_factor)
        sections, survey = build_depth_sections(workflow=workflow, script_cfg=script_cfg, repo_root=repo_root)
        pd.DataFrame([row for section in sections for row in section.qc_rows]).to_csv(output_dir / "section_geometry_qc.csv", index=False)
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
        scenarios_list = generation_scenarios(script_cfg)
        if geometry_families:
            selected = {str(value) for value in geometry_families}
            configured = {
                str(value) for value in script_cfg["generation"]["geometry_families"]
            }
            unknown = sorted(selected - configured)
            if unknown:
                raise ValueError(f"Unsupported depth geometry filters: {unknown}")
            scenarios_list = [
                item for item in scenarios_list if item.geometry_family in selected
            ]
            if not scenarios_list:
                raise ValueError(
                    "No depth generation scenarios remain after geometry filtering."
                )
        scenarios = {item.scenario_id: item for item in scenarios_list}
        sections_by_id = {item.section_id: item for item in sections}
        acceptance_qc = dict(script_cfg["generation"]["acceptance_qc"])

        def build_parent(
            row: Mapping[str, Any],
            h5: h5py.File | None = None,
            qc_only_value: bool = False,
            *,
            preflight_only: bool = False,
        ) -> GenerationAttempt:
            section = sections_by_id[str(row["section_id"])]
            scenario = scenarios[str(row["scenario_id"])]
            base_id = str(row["parent_realization_id"])
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
                preflight_only=preflight_only,
            )
            if preflight_only:
                return GenerationAttempt(base_id, None)
            if generated is None or generated.benchmark_sample is None:
                raise RuntimeError("depth_generation_missing_benchmark_sample")
            return GenerationAttempt(
                base_id,
                generated.benchmark_sample,
                qc_row={**dict(generated.qc), "sample_id": base_id, "sample_kind": "base", "status": "ok"},
                domain_rows={
                    "object_catalog": [dict(item) for item in generated.object_catalog],
                    "object_lateral_coefficients": [dict(item) for item in generated.object_lateral_coefficients],
                    "highres_forward_qc": [{"parent_realization_id": base_id, **{key: generated.qc.get(key) for key in ("physics_halo_m", "antialias_filter_half_width_m", "context_m", "antialias_numtaps")}}],
                    "subgrid_forward_qc": [{"parent_realization_id": base_id, **{key: generated.qc.get(key) for key in ("seismic_observed_rms", "seismic_model_consistent_rms", "subgrid_residual_rms", "subgrid_residual_nrmse", "subgrid_observed_model_correlation", "subgrid_amplitude_scale_ratio")}}],
                },
            )

        def view_context(sample: Any, parent_id: str):
            extras = sample.forward.extras
            if not isinstance(extras, DepthForwardExtras):
                raise TypeError("depth view context requires DepthForwardExtras")

            def perturbed_forward(phase_degrees: float, shift_s: float):
                rotated = _phase_rotate(view_wavelet, phase_degrees)
                perturbed = _shift_wavelet(view_wavelet_time, rotated, shift_s)
                highres = forward_executor(
                    sample.truth.log_ai_highres,
                    extras.vp_highres_mps,
                    sample.truth.highres_axis,
                    view_wavelet_time,
                    perturbed,
                )
                observed, support_1d = valid_filter_decimate(
                    highres, factor=view_factor, taps=view_taps
                )
                observed = observed[..., : sample.forward.seismic_observed.shape[-1]]
                support = np.broadcast_to(support_1d[: observed.shape[-1]], observed.shape)
                if np.any(np.asarray(sample.valid_mask, dtype=bool) & ~support):
                    reason = "invalid_seismic_view:perturbed_wavelet_support_incomplete"
                    raise ForwardRejected(
                        [reason], diagnostics={}, details=[{"reason": reason}]
                    )
                return observed, support

            return (
                SeismicViewContext(
                    realization_id=parent_id,
                    base_seismic=sample.forward.seismic_observed,
                    public_valid_mask=sample.valid_mask,
                    operator_source_support=np.asarray(sample.forward.support.observed, dtype=bool),
                    lateral_m=sample.truth.lateral_m,
                    sample_axis=sample.projected.model_axis.coordinates,
                    rgt_model=sample.projected.rgt_model,
                ),
                perturbed_forward,
            )

        def validate(row: Mapping[str, Any]) -> None:
            build_parent(row, preflight_only=True)

        def write_domain_outputs(directory: Path, rows: Mapping[str, list[dict[str, Any]]]) -> None:
            for name, sort_by in (
                ("object_catalog", ("realization_id", "zone_id", "object_id")),
                ("object_lateral_coefficients", ("realization_id", "zone_id", "object_id", "lateral_index")),
                ("highres_forward_qc", ("parent_realization_id",)),
                ("subgrid_forward_qc", ("parent_realization_id",)),
            ):
                stable_records_frame(rows.get(name, []), sort_by=sort_by).to_csv(directory / f"{name}.csv", index=False)

        manifest_fields = {
            **SCIENCE_CONTRACT,
            "generator_family": GENERATOR_FAMILY,
            "source_runs": {key: repo_relative_path(path, root=repo_root) for key, path in sources.items()},
            "config_provenance": dict(config_provenance),
            "mask_contract": build_mask_contract(),
            "increment_contract": generation_contract("depth", float(script_cfg["sampling"]["expected_model_dz_m"])).as_dict(),
            "seismic_input_contract": build_seismic_input_contract("depth", operator="depth_ai_vp_highres_forward_antialias"),
            "seismic_forward": forward_executor.manifest_fields,
            "forward_model_inputs_path": repo_relative_path(
                Path(str(forward_inputs["_path"])), root=repo_root
            ),
            "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
            "benchmark_purpose": str(
                script_cfg.get("benchmark_purpose") or "field_conditioned_benchmark"
            ),
            "amplitude_pilot_compatibility": amplitude_pilot_compatibility,
            "depth_basis": "tvdss",
            "n_sections": len(sections),
            "geometry_filters": sorted({str(value) for value in geometry_families}) if geometry_families else sorted({str(value) for value in script_cfg["generation"]["geometry_families"]}),
            "field_geometry": {
                "section_geometry_qc": repo_relative_path(output_dir / "section_geometry_qc.csv", root=repo_root),
                "section_geometry_feasibility_qc": repo_relative_path(feasibility_path, root=repo_root),
            },
            "random_stream": {
                "algorithm": "SHA-256/PCG64DXSM",
                "benchmark_version": SCHEMA_VERSION,
                "science_revision": SCIENCE_REVISION,
                "random_stream_contract_version": RANDOM_STREAM_CONTRACT_VERSION,
            },
        }
        return GenerationSession(
            plan=None,
            acceptance_qc=acceptance_qc,
            development_limited=debug_attempt_limit is not None,
            sample_domain="depth",
            sample_unit="m",
            depth_basis="tvdss",
            schema_version=SCHEMA_VERSION,
            generator_family=GENERATOR_FAMILY,
            hdf5_attributes={"depth_basis": "tvdss", "axis_positive_direction": "down"},
            section_ids=tuple(str(section.section_id) for section in sections),
            scenarios=tuple(scenarios_list),
            attempts_per_scenario=int(script_cfg["generation"]["attempts_per_scenario"]),
            held_out_geometry_family=str(script_cfg["splits"]["held_out_geometry_family"]),
            geometry_families=None,
            debug_attempt_limit=debug_attempt_limit,
            input_contracts=input_contracts,
            manifest_fields=manifest_fields,
            validate_attempt=validate,
            build_attempt=build_parent,
            view_context=view_context,
            write_domain_outputs=write_domain_outputs,
            resolved_seismic_views=resolved_views,
        )


# Public entrypoint for the v5 branch.  The shared Pipeline owns the parent,
# view, acceptance and publication lifecycle; this function only loads the
# domain calibration and constructs its Adapter.
def run_depth_generation(
    *,
    workflow: Any,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    forward_inputs: Mapping[str, Any],
    config_provenance: Mapping[str, str],
    calibration_path: Path,
    amplitude_prior_path: Path | None = None,
    repo_root: Path,
    output_dir: Path,
    debug_attempt_limit: int | None = None,
    geometry_families: Sequence[str] | None = None,
    qc_only: bool = False,
    structured_artifact_oracle: Callable[[Path, Any, Sequence[str]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    calibration, _ = load_depth_calibration_for_object_core(calibration_path)
    adapter = DepthSyntheticDomainAdapter(
        generator_family=GENERATOR_FAMILY,
        runtime={
            "workflow": workflow,
            "sources": sources,
            "forward_inputs": forward_inputs,
            "config_provenance": config_provenance,
            "calibration_path": calibration_path,
            "amplitude_prior_path": amplitude_prior_path,
            "repo_root": repo_root,
        }
    )
    return SyntheticBenchmarkPipeline(adapter).generate(
        script_cfg,
        calibration,
        output_dir=output_dir,
        debug_attempt_limit=debug_attempt_limit,
        geometry_families=geometry_families,
        qc_only=qc_only,
        workflow=workflow,
        sources=sources,
        forward_inputs=forward_inputs,
        config_provenance=config_provenance,
        calibration_path=calibration_path,
        amplitude_prior_path=amplitude_prior_path,
        repo_root=repo_root,
        structured_artifact_oracle=structured_artifact_oracle,
    )


__all__ = ["build_depth_sections", "run_depth_generation"]
