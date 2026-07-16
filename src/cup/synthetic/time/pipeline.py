"""Generation orchestration for synthoseis-lite benchmark suites."""

from __future__ import annotations

import json
from pathlib import Path
import time
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
from cup.synthetic.core.calibration import (
    GENERATOR_FAMILY,
    SCHEMA_VERSION as CALIBRATION_SCHEMA,
    ImpedanceCalibration,
    load_calibration,
)
from cup.synthetic.time.config import DATA_SCHEMA, IMPLEMENTATION_SCOPE
from cup.impedance import build_lfm_producer_contract, generation_contract
from cup.synthetic.core import (
    build_attempt_plan,
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
from cup.synthetic.time.forward import (
    resample_wavelet_to_highres,
)
from cup.synthetic.reporting.figures import write_generation_figures
from cup.synthetic.core.rejections import StagedRejection
from cup.synthetic.core.records import BenchmarkVariant
from cup.synthetic.core.scenarios import GenerationScenario, generation_scenarios
from cup.synthetic.core.writer import write_benchmark_sample, write_benchmark_variant
from cup.synthetic.time.geometry import build_section_geometries
from cup.synthetic.time.sample_builder import (
    build_time_field_sample,
)
from cup.synthetic.core.variant_runner import generate_seismic_variants
from cup.synthetic.schemas import SCIENCE_CONTRACT, require_science_contract
from cup.physics.numpy_backend import forward_time
from cup.synthetic.core.signal import finite_support_fir, valid_filter_decimate
from cup.config.workflow import WorkflowConfig
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
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


def _generation_input_contracts(
    *,
    calibration_path: Path,
    sources: Mapping[str, Path],
    repo_root: Path,
) -> dict[str, dict[str, str]]:
    calibration_summary_path = calibration_path.parent / "run_summary.json"
    with calibration_summary_path.open("r", encoding="utf-8") as handle:
        calibration_summary = json.load(handle)
    require_science_contract(calibration_summary, label="time calibration run summary")
    if (
        calibration_summary.get("schema_version") != CALIBRATION_SCHEMA
        or calibration_summary.get("status") != "success"
    ):
        raise ValueError(
            f"Calibration run is not a successful {CALIBRATION_SCHEMA} contract."
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
            "Calibration run summary points to a different impedance_calibration.json."
        )
    contracts = {
        "calibration": {
            "path": repo_relative_path(calibration_summary_path, root=repo_root),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                calibration_summary, label=f"calibration {calibration_summary_path.parent}"
            ),
        }
    }
    for source_key, role in (("wavelet_generation_dir", "wavelet_generation"),):
        if source_key not in sources:
            continue
        source_dir = sources[source_key]
        current_summary_path = source_dir / "run_summary.json"
        with current_summary_path.open("r", encoding="utf-8") as handle:
            current_summary = json.load(handle)
        contracts[role] = {
            "path": repo_relative_path(current_summary_path, root=repo_root),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                current_summary, label=f"{role} {source_dir}"
            ),
        }
    return contracts


def _benchmark_lfm_records(generated: Any, *, base_path: str) -> dict[str, Any]:
    return {
        **{key: value for key, value in generated.qc.items() if key.startswith("lfm_")},
        "lfm_versions": "canonical_background;input_lfm_variants",
        "lfm_variant_id": "controlled_default",
        "input_lfm_log_ai_dataset": (
            "" if not base_path else f"{base_path}/priors/input_lfm_variants/controlled_default/log_ai"
        ),
        "lfm_ideal_dataset": "" if not base_path else f"{base_path}/priors/lfm_ideal",
        "lfm_controlled_degraded_dataset": (
            "" if not base_path else f"{base_path}/priors/lfm_controlled_degraded"
        ),
        "residual_vs_lfm_ideal_dataset": (
            "" if not base_path else f"{base_path}/residuals/residual_vs_lfm_ideal"
        ),
        "residual_vs_lfm_controlled_degraded_dataset": (
            "" if not base_path else f"{base_path}/residuals/residual_vs_lfm_controlled_degraded"
        ),
    }


def _time_perturbed_wavelet_forward(
    generated: Any, *, wavelet_time: np.ndarray, wavelet: np.ndarray,
    phase_degrees: float, shift_s: float, factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Time Forward Adapter for a physically perturbed wavelet."""
    analytic = scipy.signal.hilbert(np.asarray(wavelet, dtype=np.float64))
    rotated = np.real(analytic * np.exp(1j * np.deg2rad(float(phase_degrees))))
    perturbed = np.interp(
        wavelet_time - float(shift_s), wavelet_time, rotated, left=0.0, right=0.0
    )
    highres_wavelet = resample_wavelet_to_highres(wavelet_time, perturbed, factor=factor)
    seismic_highres = forward_time(
        generated.sample.truth.log_ai_highres,
        highres_wavelet.time_s,
        highres_wavelet.amplitude,
    )
    observed, support_1d = valid_filter_decimate(
        seismic_highres, factor=factor, taps=highres_wavelet.filter_taps
    )
    observed = observed[..., : generated.seismic_observed.shape[-1]]
    support = np.broadcast_to(support_1d[: observed.shape[-1]], observed.shape)
    mask = np.asarray(generated.valid_mask_model, dtype=bool)
    if np.any(mask & ~support):
        raise ValueError("invalid_seismic_variant:perturbed_wavelet_support_incomplete")
    return observed, mask


def _seismic_variant_records_for_sample(
    *,
    h5: h5py.File,
    owner_path: str,
    source_index_record: Mapping[str, Any],
    seismic_input: np.ndarray,
    valid_mask: np.ndarray,
    seismic_model_consistent_dataset: str,
    lateral_m: np.ndarray,
    sample_axis: np.ndarray,
    script_cfg: Mapping[str, Any],
    qc_only: bool,
    perturbed_wavelet_forward=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = generate_seismic_variants(
        seismic_input=seismic_input,
        valid_mask=valid_mask,
        lateral_m=lateral_m,
        sample_axis=sample_axis,
        config=script_cfg["seismic_mismatch"],
        global_seed=int(script_cfg["global_seed"]),
        generator_family=GENERATOR_FAMILY,
        realization_id=str(source_index_record["parent_realization_id"]),
        perturbed_wavelet_forward=perturbed_wavelet_forward,
    )
    index_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    source_sample_id = str(source_index_record["sample_id"])
    for result in results:
        variant_metadata = build_seismic_variant_metadata(
            variant_id=result.variant_id,
            mismatch_family=result.mismatch_family,
            operator_source=result.operator_source,
            parameters=result.parameters,
        )
        variant_group = (
            ""
            if qc_only
            else write_benchmark_variant(
                h5,
                BenchmarkVariant(
                    owner_realization_id=str(
                        source_index_record["parent_realization_id"]
                    ),
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
                    sample_domain="time",
                ),
            ).hdf5_group
        )
        sample_id = f"{source_sample_id}__seismic__{result.variant_id}"
        record = {
            **dict(source_index_record),
            "sample_id": sample_id,
            "realization_id": sample_id,
            "source_sample_id": source_sample_id,
            "source_sample_kind": str(source_index_record.get("sample_kind", "base")),
            "sample_kind": "seismic_variant",
            "hdf5_group": variant_group,
            "seismic_variant_id": variant_metadata["variant_id"],
            "seismic_mismatch_family": variant_metadata["mismatch_family"],
            "seismic_variant_operator_source": variant_metadata["operator_source"],
            "seismic_variant_parameters_json": json.dumps(
                {
                    str(key): value
                    for key, value in variant_metadata.items()
                    if key not in {
                        "variant_id",
                        "mismatch_family",
                        "operator_source",
                    }
                },
                sort_keys=True,
            ),
            "seismic_input_dataset": (
                "" if not variant_group else f"{variant_group}/seismic_observed"
            ),
            "seismic_model_consistent_dataset": seismic_model_consistent_dataset,
            "valid_mask_dataset": source_index_record.get("valid_mask_dataset", ""),
            "positive_gain_dataset": (
                "" if not variant_group else f"{variant_group}/positive_gain"
            ),
            "additive_noise_dataset": (
                "" if not variant_group else f"{variant_group}/additive_noise"
            ),
        }
        index_records.append(record)
        result_records.append({**record, **variant_metadata, **result.qc})
    return index_records, result_records


def run_generation(
    *,
    workflow: WorkflowConfig,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    config_provenance: Mapping[str, str],
    calibration_path: Path,
    repo_root: Path,
    output_dir: Path,
    debug_attempt_limit: int | None = None,
    geometry_families: Sequence[str] | None = None,
    qc_only: bool = False,
) -> dict[str, Any]:
    debug_attempt_limit = validate_debug_attempt_limit(debug_attempt_limit)
    calibration = load_calibration(calibration_path)
    _validate_calibration_horizon_contract(calibration, script_cfg)
    input_contracts = _generation_input_contracts(
        calibration_path=calibration_path,
        sources=sources,
        repo_root=repo_root,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_generation_logger(output_dir, sample_domain="time")
    logger.info("Synthoseis field-conditioned generation started")
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
    development_limited = debug_attempt_limit is not None
    configured_attempts = int(script_cfg["generation"]["attempts_per_scenario"])
    held_out_geometry_family = str(script_cfg["splits"]["held_out_geometry_family"])
    attempt_plan = build_attempt_plan(
        section_ids=[str(section.section_id) for section in sections],
        scenarios=scenarios,
        attempts_per_scenario=configured_attempts,
        held_out_geometry_family=held_out_geometry_family,
    )
    attempt_plan = limit_attempt_plan(attempt_plan, debug_attempt_limit)
    attempts = min(configured_attempts, int(debug_attempt_limit or configured_attempts))
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
            highres_step=calibration.truth_sample_interval,
            duration_reference="minimum",
        )
    ).to_csv(feasibility_path, index=False)
    acceptance_qc = dict(script_cfg["generation"]["acceptance_qc"])

    def validate_attempt(plan_row: Mapping[str, Any]) -> None:
        section = sections_by_id[str(plan_row["section_id"])]
        scenario = scenarios_by_id[str(plan_row["scenario_id"])]
        build_time_field_sample(
            calibration,
            realization_id=str(plan_row["parent_realization_id"]),
            scenario=scenario,
            section=section,
            script_cfg=script_cfg,
            wavelet_time_s=wavelet_time,
            wavelet=wavelet,
            preflight_only=True,
        )

    preflight = run_attempt_preflight(
        attempt_plan,
        validator=validate_attempt,
        rejection_exceptions=(
            StagedRejection,
            ValueError,
            FloatingPointError,
        ),
        qc_config=acceptance_qc,
        output_dir=output_dir,
        logger=logger,
        development_limited=development_limited,
    )
    enforcement = acceptance_enforcement(acceptance_qc)
    preflight_summary = {
        "sample_domain": "time",
        "status": "failed" if not preflight.failed.empty else "ok",
        "enforcement": enforcement,
        "planned_attempts": int(len(attempt_plan)),
        "accepted_attempts": int(len(preflight.accepted_plan)),
        "rejected_attempts": int(len(attempt_plan) - len(preflight.accepted_plan)),
        "failed_scenario_count": int(len(preflight.failed)),
    }
    write_json(output_dir / "preflight_summary.json", preflight_summary)
    if preflight.accepted_plan.empty:
        raise RuntimeError("field_conditioned_preflight_no_accepted_realizations")
    if enforcement == "fail_fast" and not preflight.failed.empty:
        failed = preflight.failed[
            ["section_id", "scenario_id", "acceptance_status"]
        ].to_dict(orient="records")
        raise RuntimeError(f"field_conditioned_preflight_acceptance_qc_failed:{failed}")
    if not preflight.failed.empty:
        logger.warning(
            "preflight acceptance QC has %d failed scenarios; enforcement=warn, "
            "generation will preserve accepted realizations",
            len(preflight.failed),
        )
    index_records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    object_lateral_records: list[dict[str, Any]] = []
    qc_records: list[dict[str, Any]] = []
    rejection_records: list[dict[str, Any]] = list(preflight.rejection_details)
    seismic_variant_records: list[dict[str, Any]] = []
    highres_wavelet = (
        resample_wavelet_to_highres(
            wavelet_time,
            wavelet,
            factor=int(script_cfg["sampling"]["vertical_oversampling_factor"]),
        )
        if bool(script_cfg["forward_qc"]["highres_forward_enabled"])
        else None
    )
    h5_path = output_dir / "synthetic_benchmark.h5"
    with AttemptProgressLog(
        output_dir / "attempt_progress.csv",
        phase="generation",
        plan=preflight.accepted_plan,
        qc_config=acceptance_qc,
        logger=logger,
        append=True,
    ) as production_progress, h5py.File(h5_path, "w") as h5:
        h5.attrs["schema"] = DATA_SCHEMA
        h5.attrs["schema_version"] = DATA_SCHEMA
        for key, value in SCIENCE_CONTRACT.items():
            h5.attrs[key] = value
        h5.attrs["sample_domain"] = "time"
        h5.attrs["generator_family"] = calibration.generator_family
        h5.attrs["implementation_scope"] = IMPLEMENTATION_SCOPE
        h5.attrs["suite"] = "field_conditioned"
        h5.attrs["global_seed"] = int(script_cfg["global_seed"])
        h5.attrs["qc_only"] = bool(qc_only)
        for sequence_index, plan_row in enumerate(
            preflight.accepted_plan.to_dict(orient="records"), start=1
        ):
            attempt_started = time.perf_counter()
            section = sections_by_id[str(plan_row["section_id"])]
            scenario = scenarios_by_id[str(plan_row["scenario_id"])]
            attempt_id = int(plan_row["attempt_id"])
            realization_id = str(plan_row["parent_realization_id"])
            evaluation_role = str(plan_row["evaluation_role"])
            hdf5_group = ""
            seismic_variant_records_local: list[dict[str, Any]] = []
            base_lfm_index: dict[str, Any] = {}
            try:
                generated = build_time_field_sample(
                    calibration,
                    realization_id=realization_id,
                    scenario=scenario,
                    section=section,
                    script_cfg=script_cfg,
                    wavelet_time_s=wavelet_time,
                    wavelet=wavelet,
                )
                if generated is None:
                    raise RuntimeError("time production builder returned no sample")
                reference = (
                    None
                    if qc_only
                    else write_benchmark_sample(h5, generated.sample)
                )
                hdf5_group = "" if reference is None else reference.hdf5_group
                base_lfm_index = _benchmark_lfm_records(
                    generated, base_path=hdf5_group
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
                    "seismic_input_dataset": (
                        ""
                        if not hdf5_group
                        else f"{hdf5_group}/seismic/seismic_observed"
                    ),
                    "seismic_model_consistent_dataset": (
                        ""
                        if not hdf5_group
                        else f"{hdf5_group}/seismic/seismic_model_consistent"
                    ),
                    "valid_mask_dataset": (
                        ""
                        if not hdf5_group
                        else f"{hdf5_group}/masks/valid_mask"
                    ),
                    "valid_sample_count": int(np.count_nonzero(generated.valid_mask_model)),
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
                    seismic_input=generated.seismic_observed,
                    valid_mask=np.asarray(generated.valid_mask_model, dtype=bool),
                    seismic_model_consistent_dataset=(
                        ""
                        if not hdf5_group
                        else f"{hdf5_group}/seismic/seismic_model_consistent"
                    ),
                    lateral_m=generated.lateral_m,
                    sample_axis=generated.sample.projected.model_axis.coordinates,
                    script_cfg=script_cfg,
                    qc_only=qc_only,
                    perturbed_wavelet_forward=(
                        lambda phase_degrees, shift_s, generated=generated: _time_perturbed_wavelet_forward(
                            generated,
                            wavelet_time=wavelet_time,
                            wavelet=wavelet,
                            phase_degrees=phase_degrees,
                            shift_s=shift_s,
                            factor=int(script_cfg["sampling"]["vertical_oversampling_factor"]),
                        )
                    ),
                )
                index_records.extend(base_seismic_index)
                seismic_variant_records_local.extend(base_seismic_results)
                object_records.extend(generated.object_catalog)
                object_lateral_records.extend(generated.object_lateral_coefficients)
                status = "ok"
                reasons = ""
                qc_payload = generated.qc
            except StagedRejection as exc:
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
            except (ValueError, FloatingPointError) as exc:
                if str(exc).startswith(
                    (
                        "highres_forward_qc_failed",
                        "invalid_model_grid_forward",
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
                "seismic_input_dataset": (
                    ""
                    if not hdf5_group
                    else f"{hdf5_group}/seismic/seismic_observed"
                ),
                "seismic_model_consistent_dataset": (
                    ""
                    if not hdf5_group
                    else f"{hdf5_group}/seismic/seismic_model_consistent"
                ),
                "valid_mask_dataset": (
                    ""
                    if not hdf5_group
                    else f"{hdf5_group}/masks/valid_mask"
                ),
                "valid_sample_count": int(np.count_nonzero(generated.valid_mask_model)) if status == "ok" else "",
                **base_lfm_index,
            }
            index_records.append(record)
            if status == "ok":
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
            production_progress.record(
                plan_row,
                sequence_index=sequence_index,
                status="accepted" if status == "ok" else "rejected",
                reason=reasons,
                elapsed_s=time.perf_counter() - attempt_started,
            )
    index = stable_records_frame(
        index_records,
        sort_by=("section_id", "scenario_id", "attempt_id", "sample_kind", "sample_id"),
    )
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
    stable_records_frame(
        object_records,
        columns=object_columns,
        sort_by=("realization_id", "zone_id", "object_id"),
    ).to_csv(
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
    stable_records_frame(
        object_lateral_records,
        columns=object_lateral_columns,
        sort_by=("realization_id", "zone_id", "object_id", "lateral_index"),
    ).to_csv(output_dir / "object_lateral_coefficients.csv", index=False)
    stable_records_frame(
        qc_records,
        sort_by=("section_id", "scenario_id", "attempt_id", "sample_kind", "sample_id"),
    ).to_csv(
        output_dir / "generation_qc.csv", index=False
    )
    stable_records_frame(
        seismic_variant_records,
        sort_by=("parent_realization_id", "variant_id"),
    ).to_csv(
        output_dir / "seismic_variant_results.csv",
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
    stable_records_frame(
        rejection_records,
        columns=rejection_columns,
        sort_by=("section_id", "scenario_id", "attempt_id", "reason", "zone_id", "object_id"),
    ).to_csv(
        output_dir / "generation_rejection_details.csv",
        index=False,
    )
    rejection_summary = rejection_reason_summary(
        stable_records_frame(
            rejection_records,
            columns=rejection_columns,
            sort_by=("section_id", "scenario_id", "attempt_id", "reason", "zone_id", "object_id"),
        ),
        index,
    )
    rejection_summary_path = output_dir / "rejection_reason_summary.csv"
    rejection_summary.to_csv(rejection_summary_path, index=False)
    successful_parent_ids = index.loc[
        index["sample_kind"].eq("base") & index["status"].eq("ok"),
        "parent_realization_id",
    ].astype(str)
    catalog = build_acceptance_catalog(
        attempt_plan,
        accepted_parent_ids=successful_parent_ids,
        qc_config=acceptance_qc,
        development_limited=development_limited,
    )
    catalog.to_csv(output_dir / "scenario_catalog.csv", index=False)
    failure_statuses = {"failed", "insufficient_attempts"}
    failed_scenarios = catalog["acceptance_status"].isin(failure_statuses)
    failure_reason = (
        "field_conditioned_no_accepted_realizations"
        if successful_parent_ids.empty
        else ""
    )
    if (
        not failure_reason
        and not development_limited
        and enforcement == "fail_fast"
        and bool(failed_scenarios.any())
    ):
        failure_reason = "field_conditioned_acceptance_qc_failed"
    completed_with_warnings = (
        not development_limited
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
                contract_schema_version=DATA_SCHEMA,
                semantics={
                    **SCIENCE_CONTRACT,
                    "sample_domain": "time",
                    "suite": "field_conditioned",
                    "output_dt_s": output_dt,
                    "truth_dt_s": calibration.truth_sample_interval,
                    "generator_family": calibration.generator_family,
                },
                business_config={
                    "global_seed": int(script_cfg["global_seed"]),
                    "sampling": dict(script_cfg["sampling"]),
                    "generation": dict(script_cfg["generation"]),
                    "seismic_input": dict(script_cfg["seismic_input"]),
                    "mask_contract": build_mask_contract(),
                    "forward_qc": dict(script_cfg["forward_qc"]),
                    "lfm": dict(script_cfg["lfm"]),
                    "seismic_mismatch": dict(script_cfg["seismic_mismatch"]),
                },
                input_contracts=input_contracts,
                primary_artifacts={
                    "synthetic_benchmark": h5_path,
                    "sample_index": output_dir / "sample_index.csv",
                },
            ),
        }
    manifest = {
        "schema": DATA_SCHEMA,
        "schema_version": DATA_SCHEMA,
        **SCIENCE_CONTRACT,
        "status": (
            "failed"
            if failure_reason
            else (
                "development_limited"
                if development_limited
                else ("completed_with_warnings" if completed_with_warnings else "success")
            )
        ),
        **contract_fields,
        "failure_reason": failure_reason,
        "input_contracts": input_contracts,
        "generator_family": calibration.generator_family,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "development_limited": development_limited,
        "qc_only": bool(qc_only),
        "training_consumable": not bool(qc_only),
        "suite": "field_conditioned",
        "canonical_enabled": False,
        "source_runs": {
            key: repo_relative_path(path, root=repo_root)
            for key, path in sources.items()
        },
        "config_provenance": dict(config_provenance),
        "sample_domain": "time",
        "mask_contract": build_mask_contract(),
        "increment_contract": generation_contract("time", output_dt).as_dict(),
        "seismic_input_contract": build_seismic_input_contract(
            "time", operator="time_forward_highres_wavelet_antialias"
        ),
        "lfm_degradation": build_lfm_degradation_metadata(
            "time",
            axis_unit="s",
            component_values=script_cfg["lfm"]["controlled_degraded"],
        ),
        "input_lfm_variants": ["canonical", "controlled_default"],
        "lfm_contract": build_lfm_producer_contract(
            generation_contract("time", output_dt),
            producer_schema=DATA_SCHEMA,
            variant_selection={
                "selected": "controlled_default",
                "available": ["canonical", "controlled_default"],
            },
        ),
        "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
        "global_seed": int(script_cfg["global_seed"]),
        "random_stream": {
            "algorithm": "SHA-256/PCG64DXSM",
            "benchmark_version": DATA_SCHEMA,
            "science_revision": SCIENCE_CONTRACT["science_revision"],
            "random_stream_contract_version": SCIENCE_CONTRACT[
                "random_stream_contract_version"
            ],
            "stream_purpose_registry": [
                "state_sequence", "duration", "zone_background",
                "coefficient_<name>", "coefficient_lateral", "thickness_lateral",
                "lfm_degradation/<component coefficient name>",
                "seismic_mismatch/<variant id>/<component coefficient name>",
            ],
        },
        "output_dt_s": output_dt,
        "truth_dt_s": calibration.truth_sample_interval,
        "n_sections": len(sections),
        "n_scenarios": len(scenarios),
        "attempts_per_scenario": attempts,
        "accepted_parent_realizations": int(len(successful_parent_ids)),
        "rejected_parent_realizations": int(
            len(attempt_plan) - len(successful_parent_ids)
        ),
        "acceptance_qc": acceptance_qc,
        "preflight": preflight_summary,
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
            }
        ),
        "antialias_filter": {
            "implementation": "finite_support_projection_v1",
            "scipy_version": scipy.__version__,
            "factor": int(script_cfg["sampling"]["vertical_oversampling_factor"]),
            "numtaps": int(
                finite_support_fir(
                    int(script_cfg["sampling"]["vertical_oversampling_factor"])
                ).size
            ),
            "cutoff_output_nyquist_fraction": 0.9,
            "kaiser_beta": 8.6,
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
        "quality_warnings": (
            []
            if not completed_with_warnings
            else ["scenario_acceptance_qc_failed"]
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
    }
    write_json(output_dir / "benchmark_manifest.json", manifest)
    summary = {
        **manifest,
        "status": manifest["status"],
        "accepted_realizations": int(
            (index["sample_kind"].eq("base") & index["status"].eq("ok")).sum()
        ),
        "rejected_realizations": int(len(attempt_plan) - len(successful_parent_ids)),
        "failed_scenario_count": int(failed_scenarios.sum()),
        "quality_warnings": (
            []
            if not completed_with_warnings
            else ["scenario_acceptance_qc_failed"]
        ),
        "seismic_variant_count": len(seismic_variant_records),
    }
    write_json(output_dir / "run_summary.json", summary)
    if failure_reason == "field_conditioned_no_accepted_realizations":
        raise RuntimeError(failure_reason)
    if failure_reason == "field_conditioned_acceptance_qc_failed":
        failed = catalog.loc[
            failed_scenarios,
            ["section_id", "scenario_id", "acceptance_status"],
        ].to_dict(orient="records")
        raise RuntimeError(f"field_conditioned_acceptance_qc_failed:{failed}")
    logger.info(
        "Synthoseis generation finished: status=%s accepted=%d rejected=%d",
        summary["status"],
        summary["accepted_realizations"],
        summary["rejected_realizations"],
    )
    return summary
