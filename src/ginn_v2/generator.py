"""The deep ConditionalGenerator interface."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
from scipy.special import ndtri
import torch

from ginn_v2.contracts import (
    BandlimitedEvidence,
    EnsembleSummary,
    GenerationPolicy,
    InputContractError,
    ObservationTile,
    Segment,
    StructuredPrediction,
    StructuredRealization,
)
from ginn_v2.evidence import (
    BandlimitedEvidenceNetwork,
    EvidenceNetworkConfig,
    tuning_scale_on_model_axis,
)
from ginn_v2.representation import (
    build_lfm_anchor,
    decode_segments_numpy,
    fit_profile_coefficients,
    project_highres_to_model,
)
from ginn_v2.semi_markov import (
    SemiMarkovPrior,
    coordinate_stable_uniforms,
    exact_semi_markov_posterior,
)


class ConditionalGenerator:
    """Observe band-limited evidence, then realize structured microgeology."""

    def __init__(
        self,
        network: BandlimitedEvidenceNetwork,
        *,
        prior: SemiMarkovPrior,
        dominant_frequency_hz: float,
        sample_domain: str,
        device: str | torch.device = "cpu",
    ) -> None:
        if not np.isfinite(dominant_frequency_hz) or dominant_frequency_hz <= 0.0:
            raise ValueError("dominant_frequency_hz must be finite and positive.")
        self.network = network
        self.prior = prior
        self.dominant_frequency_hz = float(dominant_frequency_hz)
        domain = str(sample_domain).strip().casefold()
        if domain not in {"time", "depth"}:
            raise ValueError("sample_domain must be time or depth.")
        self.sample_domain = domain
        self.device = torch.device(device)
        self.network.to(self.device)

    @property
    def network_config(self) -> EvidenceNetworkConfig:
        return self.network.config

    def observe(
        self,
        tile: ObservationTile,
        *,
        vp_model_mps: np.ndarray | None = None,
    ) -> BandlimitedEvidence:
        if tile.sample_domain != self.sample_domain:
            raise InputContractError(
                "observation domain differs from the generator checkpoint."
            )
        anchor = build_lfm_anchor(tile)
        tuning = tuning_scale_on_model_axis(
            tile,
            dominant_frequency=self.dominant_frequency_hz,
            vp_model_mps=vp_model_mps,
        )
        self.network.eval()
        with torch.no_grad():
            output = self.network(
                torch.as_tensor(tile.seismic[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(tile.lfm[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(
                    tile.observed_valid[None],
                    dtype=torch.bool,
                    device=self.device,
                ),
                torch.as_tensor(tile.lateral_m[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(
                    tile.lateral_valid[None],
                    dtype=torch.bool,
                    device=self.device,
                ),
            )
        support = (
            output["support"][0].cpu().numpy().astype(bool)
            & anchor.model_support
        )
        mean = output["increment_mean"][0].cpu().numpy().astype(np.float64)
        scale = output["increment_scale"][0].cpu().numpy().astype(np.float64)
        occupancy = output["state_occupancy"][0].cpu().numpy().astype(np.float64)
        activity = output["interface_activity"][0].cpu().numpy().astype(np.float64)
        mean[~support] = 0.0
        scale[~support] = 1.0
        occupancy[~support] = 1.0 / 3.0
        activity[~support] = 0.0
        tuning[~support] = tile.model_axis.sample_interval
        background = anchor.model.copy()
        background[~support] = 0.0
        return BandlimitedEvidence(
            model_axis=tile.model_axis,
            highres_axis=tile.highres_axis,
            background_lfm_linear=background,
            bandlimited_increment_mean=mean,
            bandlimited_increment_scale=scale,
            state_occupancy=occupancy,
            interface_activity=activity,
            local_tuning_scale=tuning,
            support=support,
            lateral_m=tile.lateral_m,
            x_m=tile.x_m,
            y_m=tile.y_m,
        )

    @staticmethod
    def _highres_fields(
        evidence: BandlimitedEvidence,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model_axis = evidence.model_axis.coordinates
        high_axis = evidence.highres_axis.coordinates
        width = evidence.bandlimited_increment_mean.shape[0]
        increment = np.empty((width, high_axis.size), dtype=np.float64)
        scale = np.empty_like(increment)
        occupancy = np.empty((width, high_axis.size, 3), dtype=np.float64)
        activity = np.empty_like(increment)
        for trace in range(width):
            increment[trace] = np.interp(
                high_axis,
                model_axis,
                evidence.bandlimited_increment_mean[trace],
            )
            scale[trace] = np.interp(
                high_axis,
                model_axis,
                evidence.bandlimited_increment_scale[trace],
            )
            activity[trace] = np.interp(
                high_axis,
                model_axis,
                evidence.interface_activity[trace],
            )
            for state in range(3):
                occupancy[trace, :, state] = np.interp(
                    high_axis,
                    model_axis,
                    evidence.state_occupancy[trace, :, state],
                )
        occupancy = np.clip(occupancy, 1.0e-8, None)
        occupancy /= occupancy.sum(axis=-1, keepdims=True)
        return increment, scale, occupancy, np.clip(activity, 1.0e-5, 1.0 - 1.0e-5)

    def realize(
        self,
        evidence: BandlimitedEvidence,
        policy: GenerationPolicy = GenerationPolicy(),
    ) -> StructuredPrediction:
        if evidence.model_axis.sample_domain != self.sample_domain:
            raise InputContractError(
                "evidence domain differs from the generator checkpoint."
            )
        background_highres = np.empty(
            (
                evidence.background_lfm_linear.shape[0],
                evidence.highres_axis.coordinates.size,
            ),
            dtype=np.float64,
        )
        highres_support = np.zeros_like(background_highres, dtype=bool)
        for trace in range(background_highres.shape[0]):
            background_highres[trace] = np.interp(
                evidence.highres_axis.coordinates,
                evidence.model_axis.coordinates,
                evidence.background_lfm_linear[trace],
            )
            supported = np.flatnonzero(evidence.support[trace])
            if supported.size < 2 or np.any(np.diff(supported) != 1):
                raise InputContractError(
                    "realize requires one contiguous zone support per trace."
                )
            lower = evidence.model_axis.coordinates[supported[0]]
            upper = evidence.model_axis.coordinates[supported[-1]]
            highres_support[trace] = (
                (evidence.highres_axis.coordinates >= lower)
                & (evidence.highres_axis.coordinates <= upper)
            )
            background_highres[trace, ~highres_support[trace]] = np.nan
        increment, increment_scale, occupancy, activity = self._highres_fields(evidence)
        x = evidence.lateral_m if evidence.x_m is None else evidence.x_m
        y = np.zeros_like(evidence.lateral_m) if evidence.y_m is None else evidence.y_m
        draw_count = evidence.highres_axis.coordinates.size + 4
        uniforms = coordinate_stable_uniforms(
            x,
            y,
            realization_count=policy.realization_count,
            draw_count=draw_count,
            random_identity=policy.random_identity,
            correlation_length_m=policy.lateral_correlation_m,
        )
        identities = tuple(
            int(np.random.SeedSequence([policy.random_identity, item]).generate_state(1)[0])
            for item in range(policy.realization_count)
        )
        realizations: list[StructuredRealization] = []
        width, high_samples = increment.shape
        extents = tuple(
            (
                int(np.flatnonzero(highres_support[trace])[0]),
                int(np.flatnonzero(highres_support[trace])[-1]) + 1,
            )
            for trace in range(width)
        )
        posteriors = tuple(
            exact_semi_markov_posterior(
                occupancy[trace, start:stop],
                activity[trace, start:stop],
                self.prior,
            )
            for trace, (start, stop) in enumerate(extents)
        )
        for member in range(policy.realization_count):
            segments: list[Segment] = []
            conditional_score = 0.0
            for trace in range(width):
                zone_start, zone_stop = extents[trace]
                path = posteriors[trace].sample(uniforms[member, trace])
                conditional_score += path.log_score
                for segment_index, (state_id, start, stop) in enumerate(path.segments):
                    start += zone_start
                    stop += zone_start
                    c0, c1, c2 = fit_profile_coefficients(
                        increment[trace, start:stop]
                    )
                    local_scale = float(np.mean(increment_scale[trace, start:stop]))
                    noise = np.asarray(
                        [
                            ndtri(
                                uniforms[
                                    member,
                                    trace,
                                    (segment_index * 3 + offset) % draw_count,
                                ]
                            )
                            for offset in range(3)
                        ]
                    )
                    coefficient_scale = local_scale * np.asarray((0.25, 0.15, 0.15))
                    coefficient = np.asarray((c0, c1, c2)) + noise * coefficient_scale
                    segments.append(
                        Segment(
                            trace_index=trace,
                            state_id=int(state_id),
                            start_index=int(start),
                            stop_index=int(stop),
                            c0=float(coefficient[0]),
                            c1=float(coefficient[1]),
                            c2=float(coefficient[2]),
                            log_score=float(path.log_score),
                        )
                    )
            decoded, state = decode_segments_numpy(background_highres, segments)
            projected, projection_support = project_highres_to_model(
                decoded,
                highres_interval=evidence.highres_axis.sample_interval,
                model_interval=evidence.model_axis.sample_interval,
            )
            if projected.shape != evidence.background_lfm_linear.shape:
                raise InputContractError("projected realization does not match model axis.")
            if projection_support.shape != (evidence.model_axis.coordinates.size,):
                raise InputContractError("projection support must be one model-axis mask.")
            realizations.append(
                StructuredRealization(
                    identity=identities[member],
                    log_ai_highres=decoded,
                    state_highres=state,
                    projected_log_ai=projected,
                    segments=tuple(segments),
                    conditional_log_score=float(conditional_score),
                )
            )

        high_stack = np.stack([item.log_ai_highres for item in realizations])
        projected_stack = np.stack([item.projected_log_ai for item in realizations])
        state_stack = np.stack([item.state_highres for item in realizations])
        high_mean = np.nanmean(high_stack, axis=0)
        projected_mean = np.nanmean(projected_stack, axis=0)
        occupancy_high = np.stack(
            [np.mean(state_stack == state, axis=0) for state in range(3)],
            axis=-1,
        )
        interface = np.zeros_like(high_stack, dtype=np.float64)
        interface[..., 1:] = state_stack[..., 1:] != state_stack[..., :-1]
        counts = np.asarray(
            [
                [
                    sum(segment.trace_index == trace for segment in item.segments)
                    for trace in range(width)
                ]
                for item in realizations
            ],
            dtype=np.float64,
        )
        duration_mean = np.zeros(width, dtype=np.float64)
        duration_std = np.zeros(width, dtype=np.float64)
        for trace in range(width):
            zone_length = extents[trace][1] - extents[trace][0]
            values = np.asarray(
                [
                    (segment.stop_index - segment.start_index) / zone_length
                    for item in realizations
                    for segment in item.segments
                    if segment.trace_index == trace
                ],
                dtype=np.float64,
            )
            duration_mean[trace] = float(np.mean(values))
            duration_std[trace] = float(np.std(values))
        summary = EnsembleSummary(
            log_ai_highres_mean=high_mean,
            log_ai_highres_std=np.nanstd(high_stack, axis=0),
            projected_log_ai_mean=projected_mean,
            projected_log_ai_std=np.nanstd(projected_stack, axis=0),
            state_occupancy_highres=occupancy_high,
            interface_density_highres=np.mean(interface, axis=0),
            segment_count_mean=np.mean(counts, axis=0),
            segment_count_std=np.std(counts, axis=0),
            segment_duration_fraction_mean=duration_mean,
            segment_duration_fraction_std=duration_std,
        )
        distance = np.nanmean(
            (projected_stack - projected_mean[None]) ** 2,
            axis=(1, 2),
        )
        scores = np.asarray([item.conditional_log_score for item in realizations])
        order = np.lexsort((-scores, distance))
        representative_index = int(order[0])
        return StructuredPrediction(
            evidence=evidence,
            representative=realizations[representative_index],
            summary=summary,
            realization_identities=identities,
            realizations=tuple(realizations) if policy.retain_realizations else None,
            diagnostics={
                "representative_index": float(representative_index),
                "posterior_forward_recursions_per_trace": 1.0,
                "realization_count": float(policy.realization_count),
            },
        )

    def state_dict_payload(self) -> dict[str, Any]:
        return {
            "schema": "structured_ginn_v2_generator_v1",
            "network_config": asdict(self.network.config),
            "network_state": self.network.state_dict(),
            "prior": {
                "initial_probability": self.prior.initial_probability,
                "transition_probability": self.prior.transition_probability,
                "duration_fraction_mean": self.prior.duration_fraction_mean,
                "duration_fraction_std": self.prior.duration_fraction_std,
                "maximum_duration_fraction": self.prior.maximum_duration_fraction,
            },
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "sample_domain": self.sample_domain,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
    ) -> "ConditionalGenerator":
        if payload.get("schema") != "structured_ginn_v2_generator_v1":
            raise ValueError("unsupported Structured GINN V2 checkpoint schema.")
        config = EvidenceNetworkConfig.from_mapping(payload["network_config"])
        network = BandlimitedEvidenceNetwork(config)
        network.load_state_dict(payload["network_state"], strict=True)
        prior_payload = payload["prior"]
        prior = SemiMarkovPrior(
            initial_probability=prior_payload["initial_probability"],
            transition_probability=prior_payload["transition_probability"],
            duration_fraction_mean=prior_payload["duration_fraction_mean"],
            duration_fraction_std=prior_payload["duration_fraction_std"],
            maximum_duration_fraction=float(
                prior_payload["maximum_duration_fraction"]
            ),
        )
        return cls(
            network,
            prior=prior,
            dominant_frequency_hz=float(payload["dominant_frequency_hz"]),
            sample_domain=str(payload["sample_domain"]),
            device=device,
        )


__all__ = ["ConditionalGenerator"]
