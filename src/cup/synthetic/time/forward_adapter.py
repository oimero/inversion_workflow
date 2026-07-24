"""Time-domain axis preparation and acoustic forward Adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cup.physics.numpy_backend import forward_time, reflectivity_from_log_ai
from cup.synthetic.core.signal import fir_half_width, required_context
from cup.synthetic.core.records import (
    DomainPreparation,
    ForwardResult,
    ForwardSupport,
    ProjectedTruth,
    SampleAxis,
    TimeForwardExtras,
)
from cup.synthetic.core.truth import SyntheticTruth
from cup.synthetic.time.forward import (
    HighresWavelet,
    forward_sample_valid_mask,
    highres_forward_to_model_grid,
    model_grid_closure_qc,
    resample_wavelet_to_highres,
)


@dataclass(frozen=True)
class TimeForwardConfiguration:
    wavelet_time_s: np.ndarray
    wavelet: np.ndarray
    highres_wavelet: HighresWavelet
    context_s: float


class TimeForwardAdapter:
    def prepare(
        self,
        *,
        horizon_twt_s: np.ndarray,
        wavelet_time_s: np.ndarray,
        wavelet: np.ndarray,
        output_dt_s: float,
        vertical_oversampling_factor: int,
    ) -> DomainPreparation:
        horizons = np.asarray(horizon_twt_s, dtype=np.float64)
        wavelet_values = np.asarray(wavelet, dtype=np.float64).reshape(-1)
        factor = int(vertical_oversampling_factor)
        output_dt = float(output_dt_s)
        truth_dt = output_dt / factor
        wavelet_halo = (wavelet_values.size // 2) * output_dt
        projection_half = fir_half_width(factor, truth_dt)
        context = required_context(
            projection_fir_half_width=projection_half,
            forward_input_halo=wavelet_halo,
            observed_decimation_fir_half_width=projection_half,
        )
        start = np.floor((float(np.min(horizons[:, 0])) - context) / truth_dt) * truth_dt
        end = np.ceil((float(np.max(horizons[:, -1])) + context) / truth_dt) * truth_dt
        n_model_intervals = int(np.ceil((end - start) / output_dt))
        model_axis = start + np.arange(n_model_intervals + 1, dtype=np.float64) * output_dt
        return DomainPreparation(
            model_axis=SampleAxis(
                sample_domain="time",
                unit="s",
                coordinates=model_axis,
                sample_interval=output_dt,
                positive_direction="increasing_time",
            ),
            required_context_extent=context,
            forward_configuration=TimeForwardConfiguration(
                wavelet_time_s=np.asarray(wavelet_time_s, dtype=np.float64),
                wavelet=wavelet_values,
                highres_wavelet=resample_wavelet_to_highres(
                    wavelet_time_s,
                    wavelet_values,
                    factor=factor,
                ),
                context_s=context,
            ),
        )

    def forward(
        self,
        truth: SyntheticTruth,
        projected: ProjectedTruth,
        preparation: DomainPreparation,
    ) -> ForwardResult:
        config = preparation.forward_configuration
        if not isinstance(config, TimeForwardConfiguration):
            raise TypeError("TimeForwardAdapter requires TimeForwardConfiguration.")
        model_target = np.asarray(projected.model_target_log_ai, dtype=np.float64)
        model_seismic = forward_time(
            model_target,
            config.wavelet_time_s,
            config.wavelet,
        )
        forward_valid_highres = np.isfinite(truth.log_ai_highres[:, :-1]) & np.isfinite(
            truth.log_ai_highres[:, 1:]
        )
        forward_valid_model = np.isfinite(model_target[:, :-1]) & np.isfinite(
            model_target[:, 1:]
        )
        highres = highres_forward_to_model_grid(
            truth.log_ai_highres,
            model_seismic,
            highres_wavelet=config.highres_wavelet,
            forward_valid_mask_model=forward_valid_model,
        )
        observed = np.asarray(highres.seismic_model_grid, dtype=np.float64)
        observed_support = (
            forward_sample_valid_mask(forward_valid_model)
            & highres.decimation_support_1d[None, :]
        )
        physics_support = (
            np.isfinite(model_target)
            & np.isfinite(observed)
            & np.isfinite(model_seismic)
            & observed_support
        )
        qc = {
            **model_grid_closure_qc(
                model_target,
                model_seismic,
                config.wavelet_time_s,
                config.wavelet,
            ),
            **highres.qc,
            "highres_forward_reasons": "",
        }
        return ForwardResult(
            seismic_observed=observed,
            seismic_model_consistent=model_seismic,
            subgrid_forward_residual=observed - model_seismic,
            support=ForwardSupport(
                highres=forward_sample_valid_mask(forward_valid_highres),
                model=forward_sample_valid_mask(forward_valid_model),
                observed=observed_support,
                physics=physics_support,
            ),
            qc=qc,
            metadata={
                "sample_domain": "time",
                "structured_forward_context": {
                    "wavelet_time_s": config.wavelet_time_s.tolist(),
                    "wavelet_amplitude": config.wavelet.tolist(),
                    "ai_velocity_relation": None,
                    "output_chunk_size": 64,
                },
            },
            extras=TimeForwardExtras(
                reflectivity_highres=reflectivity_from_log_ai(truth.log_ai_highres),
                reflectivity_model=reflectivity_from_log_ai(model_target),
                forward_valid_mask_highres=forward_valid_highres,
                forward_valid_mask_model=forward_valid_model,
            ),
        )


__all__ = ["TimeForwardAdapter", "TimeForwardConfiguration"]
