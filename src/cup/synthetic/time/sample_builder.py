"""Typed time inputs into the shared truth and Benchmark Builder Seams."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from cup.impedance import generation_contract
from cup.synthetic.core.calibration import ImpedanceCalibration
from cup.synthetic.core.lfm import LfmPolicy
from cup.synthetic.core.projection import time_projection_policy
from cup.synthetic.core.random import RandomNamespace
from cup.synthetic.core.records import DomainPreparation, SampleAxis
from cup.synthetic.core.sample_builder import (
    BenchmarkBuildPolicy,
    BenchmarkBuilder,
    CanonicalIncrementPolicy,
)
from cup.synthetic.core.scenarios import GenerationScenario
from cup.synthetic.core.truth import TruthGenerationRequest, generate_field_conditioned_truth
from cup.synthetic.schemas import BENCHMARK_SCHEMA_VERSION
from cup.synthetic.time.forward_adapter import TimeForwardAdapter
from cup.synthetic.time.forward_adapter import TimeForwardConfiguration
from cup.synthetic.time.model import TimeBenchmarkSample
from cup.synthetic.time.forward import resample_wavelet_to_highres


def build_time_field_sample(
    calibration: ImpedanceCalibration,
    *,
    realization_id: str,
    scenario: GenerationScenario,
    section: Any,
    script_cfg: Mapping[str, Any],
    wavelet_time_s,
    wavelet,
    preflight_only: bool = False,
) -> TimeBenchmarkSample | None:
    output_dt = float(script_cfg["sampling"]["expected_output_dt_s"])
    factor = int(script_cfg["sampling"]["vertical_oversampling_factor"])
    adapter = TimeForwardAdapter()
    preparation = adapter.prepare(
        horizon_twt_s=section.horizon_twt_s,
        wavelet_time_s=wavelet_time_s,
        wavelet=wavelet,
        output_dt_s=output_dt,
        vertical_oversampling_factor=factor,
    )
    namespace = RandomNamespace(
        benchmark_version=BENCHMARK_SCHEMA_VERSION,
        generator_family=calibration.generator_family,
    )
    minimum_cells = 2 if scenario.duration_mode == "ultra_thin_stress" else 4
    truth = generate_field_conditioned_truth(
        calibration,
        TruthGenerationRequest(
            realization_id=realization_id,
            scenario=scenario,
            global_seed=int(script_cfg["global_seed"]),
            random_namespace=namespace,
            sample_domain="time",
            axis_unit="s",
            lateral_m=section.lateral_m,
            inline_float=section.inline_float,
            xline_float=section.xline_float,
            x_m=section.x_m,
            y_m=section.y_m,
            horizon_coordinates=section.horizon_twt_s,
            model_sample_interval=output_dt,
            vertical_oversampling_factor=factor,
            minimum_highres_cells=minimum_cells,
            vertical_axis_origin=None,
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
    if preflight_only:
        return None
    contract = generation_contract(
        "time", float(preparation.model_axis.coordinates[1] - preparation.model_axis.coordinates[0])
    )
    sample = BenchmarkBuilder().build(
        truth=truth,
        preparation=preparation,
        forward_adapter=adapter,
        canonical_policy=CanonicalIncrementPolicy(contract=contract),
        lfm_policy=LfmPolicy(
            sample_domain="time",
            axis_unit="s",
            global_seed=int(script_cfg["global_seed"]),
            random_namespace=namespace,
            realization_id=realization_id,
            horizon_coordinates=section.horizon_twt_s,
            controlled_degraded=script_cfg["lfm"]["controlled_degraded"],
            algorithm="time",
            zone_id_model=None,
            degradation_variant_id=realization_id,
        ),
        build_policy=BenchmarkBuildPolicy(
            require_forward_support=True,
            domain_metadata={
                "sample_domain": "time",
                "increment_contract": contract.as_dict(),
            },
        ),
    )
    return TimeBenchmarkSample(sample)


def build_time_canonical_sample(
    calibration: ImpedanceCalibration,
    *,
    scenario,
    canonical_config: Mapping[str, Any],
    script_cfg: Mapping[str, Any],
    wavelet_time_s,
    wavelet,
) -> TimeBenchmarkSample:
    from cup.synthetic.time.canonical import generate_canonical_truth

    output_dt = float(script_cfg["sampling"]["expected_output_dt_s"])
    factor = int(script_cfg["sampling"]["vertical_oversampling_factor"])
    truth = generate_canonical_truth(
        calibration,
        scenario=scenario,
        config=canonical_config,
        output_dt_s=output_dt,
        wavelet_time_s=wavelet_time_s,
        wavelet=wavelet,
        vertical_oversampling_factor=factor,
    )
    model_axis_values = truth.highres_axis[::factor]
    model_axis = SampleAxis(
        sample_domain="time",
        unit="s",
        coordinates=model_axis_values,
        sample_interval=output_dt,
        positive_direction="increasing_time",
    )
    preparation = DomainPreparation(
        model_axis=model_axis,
        required_context_extent=(len(wavelet) // 2) * output_dt,
        projection_policy=time_projection_policy(),
        forward_configuration=TimeForwardConfiguration(
            wavelet_time_s=wavelet_time_s,
            wavelet=wavelet,
            highres_wavelet=resample_wavelet_to_highres(
                wavelet_time_s, wavelet, factor=factor
            ),
            context_s=(len(wavelet) // 2) * output_dt,
        ),
    )
    namespace = RandomNamespace(
        benchmark_version=BENCHMARK_SCHEMA_VERSION,
        generator_family=calibration.generator_family,
    )
    contract = generation_contract("time", model_axis.sample_interval)
    sample = BenchmarkBuilder().build(
        truth=truth,
        preparation=preparation,
        forward_adapter=TimeForwardAdapter(),
        canonical_policy=CanonicalIncrementPolicy(contract=contract),
        lfm_policy=LfmPolicy(
            sample_domain="time",
            axis_unit="s",
            global_seed=int(script_cfg["global_seed"]),
            random_namespace=namespace,
            realization_id=truth.realization_id,
            horizon_coordinates=np.empty((truth.lateral_m.size, 0)),
            controlled_degraded=script_cfg["lfm"]["controlled_degraded"],
            algorithm="time",
            degradation_variant_id=truth.realization_id,
        ),
        build_policy=BenchmarkBuildPolicy(
            require_forward_support=True,
            domain_metadata={
                "sample_domain": "time",
                "increment_contract": contract.as_dict(),
            },
        ),
    )
    return TimeBenchmarkSample(sample)


__all__ = ["build_time_canonical_sample", "build_time_field_sample"]
