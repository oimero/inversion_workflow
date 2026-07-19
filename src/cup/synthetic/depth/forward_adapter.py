"""Depth-domain preparation and acoustic forward Adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import scipy

from cup.physics.execution import DepthForwardExecutor
from cup.physics.numpy_backend import velocity_from_ai
from cup.synthetic.core.signal import finite_support_fir, required_context, valid_filter_decimate
from cup.synthetic.core.records import (
    DepthForwardExtras,
    DomainPreparation,
    ForwardResult,
    ForwardSupport,
    ProjectedTruth,
    SampleAxis,
)
from cup.synthetic.core.rejections import ForwardRejected
from cup.synthetic.core.truth import SyntheticTruth


@dataclass(frozen=True)
class DepthForwardConfiguration:
    wavelet_time_s: np.ndarray
    wavelet: np.ndarray
    antialias_taps: np.ndarray
    ai_velocity_relation: Mapping[str, float]
    executor: DepthForwardExecutor
    physics_halo_m: float
    antialias_half_width_m: float
    context_m: float
    maximum_allowed_vp_mps: float


class DepthForwardAdapter:
    def prepare(
        self,
        *,
        horizon_tvdss_m: np.ndarray,
        survey_axis_m: np.ndarray,
        wavelet_time_s: np.ndarray,
        wavelet: np.ndarray,
        model_dz_m: float,
        vertical_oversampling_factor: int,
        antialias_config: Mapping[str, Any],
        maximum_allowed_vp_mps: float,
        ai_velocity_relation: Mapping[str, float],
        executor: DepthForwardExecutor,
    ) -> DomainPreparation:
        factor = int(vertical_oversampling_factor)
        model_dz = float(model_dz_m)
        high_dz = model_dz / factor
        expected = {
            "taps_per_factor": 32,
            "cutoff_output_nyquist_fraction": 0.9,
            "kaiser_beta": 8.6,
        }
        for key, value in expected.items():
            if not np.isclose(float(antialias_config[key]), value, rtol=0.0, atol=1e-12):
                raise ValueError(f"sampling.antialias.{key} must equal science-v2 value {value}")
        taps = finite_support_fir(factor)
        antialias_half_m = (taps.size // 2) * high_dz
        wavelet_half_s = float(np.max(np.abs(wavelet_time_s)))
        maximum_vp = float(maximum_allowed_vp_mps)
        halo = np.ceil((0.5 * maximum_vp * wavelet_half_s) / model_dz) * model_dz
        context = required_context(
            projection_fir_half_width=antialias_half_m,
            forward_input_halo=float(halo),
            observed_decimation_fir_half_width=antialias_half_m,
        )
        survey_axis = np.asarray(survey_axis_m, dtype=np.float64)
        origin = float(survey_axis[0])
        horizons = np.asarray(horizon_tvdss_m, dtype=np.float64)
        start = origin + np.floor(
            (float(np.min(horizons[:, 0])) - context - origin) / model_dz
        ) * model_dz
        end = origin + np.ceil(
            (float(np.max(horizons[:, -1])) + context - origin) / model_dz
        ) * model_dz
        count = int(np.ceil((end - start) / model_dz)) + 1
        model_axis = start + np.arange(count, dtype=np.float64) * model_dz
        survey_indices = np.searchsorted(survey_axis, model_axis)
        if np.any(survey_indices >= survey_axis.size) or not np.allclose(
            survey_axis[survey_indices], model_axis, rtol=0.0, atol=1e-9
        ):
            reason = "section_context_outside_survey_axis"
            diagnostics = {
                "survey_axis_min_m": float(survey_axis[0]),
                "survey_axis_max_m": float(survey_axis[-1]),
                "requested_axis_min_m": float(model_axis[0]),
                "requested_axis_max_m": float(model_axis[-1]),
                "required_context_m": float(context),
            }
            raise ForwardRejected(
                [reason],
                diagnostics=diagnostics,
                details=[{"reason": reason, **diagnostics}],
            )
        return DomainPreparation(
            model_axis=SampleAxis(
                sample_domain="depth",
                unit="m",
                coordinates=model_axis,
                sample_interval=model_dz,
                positive_direction="down",
                depth_basis="tvdss",
            ),
            required_context_extent=context,
            forward_configuration=DepthForwardConfiguration(
                wavelet_time_s=np.asarray(wavelet_time_s, dtype=np.float64),
                wavelet=np.asarray(wavelet, dtype=np.float64),
                antialias_taps=taps,
                ai_velocity_relation=dict(ai_velocity_relation),
                executor=executor,
                physics_halo_m=float(halo),
                antialias_half_width_m=float(antialias_half_m),
                context_m=context,
                maximum_allowed_vp_mps=maximum_vp,
            ),
        )

    def forward(
        self,
        truth: SyntheticTruth,
        projected: ProjectedTruth,
        preparation: DomainPreparation,
    ) -> ForwardResult:
        config = preparation.forward_configuration
        if not isinstance(config, DepthForwardConfiguration):
            raise TypeError("DepthForwardAdapter requires DepthForwardConfiguration.")
        factor = int(
            round(preparation.model_axis.sample_interval / truth.highres_sample_interval)
        )
        relation = config.ai_velocity_relation
        vp_high = velocity_from_ai(
            np.exp(truth.log_ai_highres),
            a=float(relation["a"]),
            b=float(relation["b"]),
        )
        vp_model = velocity_from_ai(
            np.exp(projected.model_target_log_ai),
            a=float(relation["a"]),
            b=float(relation["b"]),
        )
        seismic_high = config.executor(
            truth.log_ai_highres,
            vp_high,
            truth.highres_axis,
            config.wavelet_time_s,
            config.wavelet,
        )
        seismic_observed, observed_support_1d = valid_filter_decimate(
            seismic_high,
            factor=factor,
            taps=config.antialias_taps,
        )
        seismic_model = config.executor(
            projected.model_target_log_ai,
            vp_model,
            preparation.model_axis.coordinates,
            config.wavelet_time_s,
            config.wavelet,
        )
        residual = seismic_observed - seismic_model
        observed_support = np.broadcast_to(observed_support_1d, seismic_observed.shape)
        physics_support = (
            np.isfinite(projected.model_target_log_ai)
            & np.isfinite(vp_model)
            & np.isfinite(seismic_observed)
            & np.isfinite(seismic_model)
        )
        valid = projected.geometric_valid_mask_model
        observed_values = seismic_observed[valid]
        model_values = seismic_model[valid]
        residual_values = residual[valid]
        observed_rms = float(np.sqrt(np.mean(observed_values**2)))
        model_rms = float(np.sqrt(np.mean(model_values**2)))
        residual_rms = float(np.sqrt(np.mean(residual_values**2)))
        correlation = (
            float(np.corrcoef(observed_values, model_values)[0, 1])
            if observed_values.size >= 2
            and np.std(observed_values) > 0.0
            and np.std(model_values) > 0.0
            else float("nan")
        )
        return ForwardResult(
            seismic_observed=seismic_observed,
            seismic_model_consistent=seismic_model,
            subgrid_forward_residual=residual,
            support=ForwardSupport(
                highres=np.isfinite(seismic_high),
                model=np.isfinite(seismic_model),
                observed=observed_support,
                physics=physics_support,
            ),
            qc={
                "physics_halo_m": config.physics_halo_m,
                "antialias_filter_half_width_m": config.antialias_half_width_m,
                "physics_halo_samples": int(
                    round(config.physics_halo_m / preparation.model_axis.sample_interval)
                ),
                "context_m": config.context_m,
                "maximum_allowed_vp_mps": config.maximum_allowed_vp_mps,
                "antialias_numtaps": int(config.antialias_taps.size),
                "antialias_scipy_version": scipy.__version__,
                "seismic_observed_rms": observed_rms,
                "seismic_model_consistent_rms": model_rms,
                "subgrid_residual_rms": residual_rms,
                "subgrid_residual_nrmse": residual_rms
                / max(observed_rms, np.finfo(np.float64).eps),
                "subgrid_observed_model_correlation": correlation,
                "subgrid_amplitude_scale_ratio": observed_rms
                / max(model_rms, np.finfo(np.float64).eps),
            },
            metadata={"sample_domain": "depth", "depth_basis": "tvdss"},
            extras=DepthForwardExtras(
                vp_highres_mps=vp_high,
                vp_model_mps=vp_model,
            ),
        )


__all__ = ["DepthForwardAdapter", "DepthForwardConfiguration"]
