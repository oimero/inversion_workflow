"""Generation orchestration for synthoseis-lite benchmark suites."""

from __future__ import annotations

import json
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
from cup.synthetic.core.calibration import (
    GENERATOR_FAMILY,
    SCHEMA_VERSION as CALIBRATION_SCHEMA,
    ImpedanceCalibration,
    load_calibration,
)
from cup.synthetic.time.config import DATA_SCHEMA, IMPLEMENTATION_SCOPE
from cup.impedance import generation_contract
from cup.synthetic.core import (
    build_seismic_input_contract,
    build_mask_contract,
    geometry_feasibility_rows,
)
from cup.synthetic.core.field_runner import (
    stable_records_frame,
)
from cup.synthetic.time.forward import (
    resample_wavelet_to_highres,
)
from cup.synthetic.core.scenarios import GenerationScenario, generation_scenarios
from cup.synthetic.time.geometry import build_section_geometries
from cup.synthetic.time.sample_builder import (
    build_time_field_sample,
)
from cup.synthetic.time.model import TimeBenchmarkSample
from cup.synthetic.core.pipeline import (
    GenerationAttempt,
    GenerationSession,
    SeismicViewContext,
)
from cup.synthetic.adapters import TimeSyntheticDomainAdapter
from cup.synthetic.core.pipeline import SyntheticBenchmarkPipeline
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


def _benchmark_canonical_records(generated: Any, *, base_path: str) -> dict[str, Any]:
    return {
        **{key: value for key, value in generated.qc.items() if key.startswith("lfm_")},
        "canonical_background_dataset": (
            "" if not base_path else f"{base_path}/priors/canonical_background_log_ai"
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
        raise ValueError("invalid_seismic_view:perturbed_wavelet_support_incomplete")
    return observed, support


class TimeGenerationSession:
    """Prepare time-domain science for the shared generation lifecycle.

    This class deliberately stops at an in-memory :class:`BenchmarkSample`.
    HDF5, view-index and acceptance publication are performed by
    ``SyntheticBenchmarkPipeline``.
    """

    @classmethod
    def prepare(
        cls,
        script_cfg: Mapping[str, Any],
        calibration: ImpedanceCalibration,
        *,
        output_dir: Path,
        workflow: WorkflowConfig,
        sources: Mapping[str, Path],
        config_provenance: Mapping[str, str],
        calibration_path: Path,
        repo_root: Path,
        debug_attempt_limit: int | None = None,
        geometry_families: Sequence[str] | None = None,
        qc_only: bool = False,
        **_: Any,
    ) -> GenerationSession:
        _validate_calibration_horizon_contract(calibration, script_cfg)
        input_contracts = _generation_input_contracts(
            calibration_path=calibration_path,
            sources=sources,
            repo_root=repo_root,
        )
        wavelet_time, wavelet = load_wavelet_csv(
            sources["wavelet_generation_dir"] / "selected_wavelet.csv"
        )
        wavelet, wavelet_qc = validate_wavelet_normalization(
            wavelet_time,
            wavelet,
            expected_l2_energy=1.0,
            l2_energy_tolerance=1e-5,
            max_center_abs_time_s=1e-9,
            allow_small_renormalization=True,
        )
        if wavelet_qc.status != "ok":
            raise ValueError(f"invalid_wavelet:{wavelet_qc.reasons}")
        output_dt = infer_wavelet_dt(wavelet_time)
        sections = build_section_geometries(
            workflow=workflow, script_cfg=script_cfg, repo_root=repo_root
        )
        pd.DataFrame.from_records(
            [row for section in sections for row in section.qc_rows]
        ).to_csv(output_dir / "section_geometry_qc.csv", index=False)
        scenarios = generation_scenarios(script_cfg)
        if geometry_families:
            selected = {str(value) for value in geometry_families}
            unknown = selected.difference({"none", "wedge", "pinchout"})
            if unknown:
                raise ValueError(f"Unsupported geometry filters: {sorted(unknown)}")
            scenarios = [item for item in scenarios if item.geometry_family in selected]
            if not scenarios:
                raise ValueError("No generation scenarios remain after geometry filtering.")
        acceptance_qc = dict(script_cfg["generation"]["acceptance_qc"])
        sections_by_id = {str(section.section_id): section for section in sections}
        scenarios_by_id = {str(item.scenario_id): item for item in scenarios}
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

        def build_parent(
            row: Mapping[str, Any],
            h5: h5py.File | None = None,
            qc_only_value: bool = False,
            *,
            preflight_only: bool = False,
        ) -> GenerationAttempt:
            section = sections_by_id[str(row["section_id"])]
            scenario = scenarios_by_id[str(row["scenario_id"])]
            parent_id = str(row["parent_realization_id"])
            generated = build_time_field_sample(
                calibration,
                realization_id=parent_id,
                scenario=scenario,
                section=section,
                script_cfg=script_cfg,
                wavelet_time_s=wavelet_time,
                wavelet=wavelet,
                preflight_only=preflight_only,
            )
            if preflight_only:
                return GenerationAttempt(parent_id, None)
            if generated is None:
                raise RuntimeError("time production builder returned no sample")
            sample = generated.sample
            return GenerationAttempt(
                parent_id,
                sample,
                qc_row={**dict(sample.qc), "sample_id": parent_id, "sample_kind": "base", "status": "ok"},
                domain_rows={
                    "object_catalog": [dict(item) for item in generated.object_catalog],
                    "object_lateral_coefficients": [dict(item) for item in generated.object_lateral_coefficients],
                },
            )

        def view_context(sample: Any, parent_id: str):
            wrapped = TimeBenchmarkSample(sample)
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
                lambda phase_degrees, shift_s: _time_perturbed_wavelet_forward(
                    wrapped,
                    wavelet_time=wavelet_time,
                    wavelet=wavelet,
                    phase_degrees=phase_degrees,
                    shift_s=shift_s,
                    factor=int(script_cfg["sampling"]["vertical_oversampling_factor"]),
                ),
            )

        def validate(row: Mapping[str, Any]) -> None:
            build_parent(row, preflight_only=True)

        def write_domain_outputs(directory: Path, rows: Mapping[str, list[dict[str, Any]]]) -> None:
            object_columns = [
                "realization_id", "scenario_id", "zone_id", "object_id", "state", "state_id",
                "base_duration_fraction", "event_target", "duration_fraction_start", "duration_fraction_end",
                "minimum_duration_fraction", "maximum_duration_fraction", "minimum_duration_s", "maximum_duration_s",
                "minimum_truth_samples", "maximum_truth_samples", "event_multiplier_start", "event_multiplier_end",
                "minimum_event_multiplier", "maximum_event_multiplier", "reversal_fraction", "clipping_fraction",
                "profile_violation_fraction", "profile_projection_fraction", "mean_profile_projection_scale",
                "minimum_profile_projection_scale", "c0_conditioning_fraction", "mean_c0_conditioning_adjustment",
                "maximum_c0_conditioning_adjustment",
            ]
            stable_records_frame(rows.get("object_catalog", []), columns=object_columns, sort_by=("realization_id", "zone_id", "object_id")).to_csv(directory / "object_catalog.csv", index=False)
            lateral_columns = [
                "realization_id", "scenario_id", "zone_id", "local_object_index", "calibration_object_id", "object_id",
                "state", "state_id", "event_target", "lateral_index", "lateral_m", "c0", "c1", "c2",
                "thickness_fraction", "object_top_s", "object_bottom_s", "profile_projection_scale", "c0_conditioning_adjustment",
            ]
            stable_records_frame(rows.get("object_lateral_coefficients", []), columns=lateral_columns, sort_by=("realization_id", "zone_id", "object_id", "lateral_index")).to_csv(directory / "object_lateral_coefficients.csv", index=False)

        manifest_fields = {
            **SCIENCE_CONTRACT,
            "implementation_scope": IMPLEMENTATION_SCOPE,
            "generator_family": calibration.generator_family,
            "source_runs": {key: repo_relative_path(path, root=repo_root) for key, path in sources.items()},
            "config_provenance": dict(config_provenance),
            "mask_contract": build_mask_contract(),
            "increment_contract": generation_contract("time", output_dt).as_dict(),
            "seismic_input_contract": build_seismic_input_contract("time", operator="time_forward_highres_wavelet_antialias"),
            "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
            "output_dt_s": output_dt,
            "truth_dt_s": calibration.truth_sample_interval,
            "n_sections": len(sections),
            "geometry_filters": sorted({item.geometry_family for item in scenarios}),
            "field_geometry": {
                "section_geometry_qc": repo_relative_path(output_dir / "section_geometry_qc.csv", root=repo_root),
                "section_geometry_feasibility_qc": repo_relative_path(feasibility_path, root=repo_root),
            },
            "random_stream": {
                "algorithm": "SHA-256/PCG64DXSM",
                "benchmark_version": DATA_SCHEMA,
                "science_revision": SCIENCE_CONTRACT["science_revision"],
                "random_stream_contract_version": SCIENCE_CONTRACT["random_stream_contract_version"],
            },
        }
        return GenerationSession(
            plan=None,
            acceptance_qc=acceptance_qc,
            development_limited=debug_attempt_limit is not None,
            sample_domain="time",
            sample_unit="s",
            depth_basis=None,
            schema_version=DATA_SCHEMA,
            generator_family=calibration.generator_family,
            hdf5_attributes={"implementation_scope": IMPLEMENTATION_SCOPE},
            section_ids=tuple(str(section.section_id) for section in sections),
            scenarios=tuple(scenarios),
            attempts_per_scenario=int(script_cfg["generation"]["attempts_per_scenario"]),
            held_out_geometry_family=str(script_cfg["splits"]["held_out_geometry_family"]),
            debug_attempt_limit=debug_attempt_limit,
            input_contracts=input_contracts,
            manifest_fields=manifest_fields,
            validate_attempt=validate,
            build_attempt=build_parent,
            view_context=view_context,
            write_domain_outputs=write_domain_outputs,
        )
# Public entrypoint: source loading and Adapter construction only.
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
    calibration = load_calibration(calibration_path)
    adapter = TimeSyntheticDomainAdapter(
        generator_family=calibration.generator_family,
        runtime={
            "workflow": workflow,
            "sources": sources,
            "config_provenance": config_provenance,
            "calibration_path": calibration_path,
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
        config_provenance=config_provenance,
        calibration_path=calibration_path,
        repo_root=repo_root,
    )
