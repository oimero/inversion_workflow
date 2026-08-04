"""The deep ConditionalGenerator interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from scipy.special import ndtri
from torch import nn
from torch.nn import functional as F

from ginn_v2.contracts import (
    CoefficientVarianceCalibration,
    EnsembleSummary,
    GenerationPolicy,
    InputContractError,
    ObservableEvidence,
    ObservableTargetContract,
    ObservationTile,
    Segment,
    SegmentExtent,
    SegmentParameterDistribution,
    StructuredPrediction,
    StructuredRealization,
)
from ginn_v2.augmentation import stable_random_identity
from ginn_v2.evidence import (
    EvidenceNetworkConfig,
    ObservableEvidenceNetwork,
    tuning_scale_on_model_axis,
)
from ginn_v2.representation import (
    SEGMENT_PROFILE_FEATURES,
    build_lfm_anchor,
    build_segment_profile_features,
    decode_segments_numpy,
    lfm_residual_from_anchor,
    path_to_highres_extents,
    profile_basis,
    project_supported_highres_to_model,
)
from ginn_v2.semi_markov import (
    SemiMarkovConditioning,
    SemiMarkovPrior,
    coordinate_stable_uniforms,
    exact_semi_markov_posterior,
)


@dataclass(frozen=True)
class SegmentProfileHeadConfig:
    hidden_channels: int = 48
    hidden_layers: int = 2
    minimum_scale: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.hidden_channels <= 0 or self.hidden_layers <= 0:
            raise ValueError("profile head dimensions must be positive.")
        if not np.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be finite and positive.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SegmentProfileHeadConfig":
        allowed = {"hidden_channels", "hidden_layers", "minimum_scale"}
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError(f"unknown segment profile head keys: {unknown}")
        return cls(**dict(value))


class SegmentProfileHead(nn.Module):
    """Residual coefficient distribution over deterministic evidence fitting."""

    def __init__(self, config: SegmentProfileHeadConfig) -> None:
        super().__init__()
        self.config = config
        width = len(SEGMENT_PROFILE_FEATURES)
        layers: list[nn.Module] = []
        input_width = width
        for _ in range(config.hidden_layers):
            layers.extend(
                (
                    nn.Linear(input_width, config.hidden_channels),
                    nn.GELU(),
                )
            )
            input_width = config.hidden_channels
        self.trunk = nn.Sequential(*layers)
        self.output = nn.Linear(input_width, 6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias[:3])
        nn.init.constant_(self.output.bias[3:], -3.0)
        self.register_buffer("feature_mean", torch.zeros(width))
        self.register_buffer("feature_scale", torch.ones(width))

    def set_feature_normalization(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        center = torch.as_tensor(mean, dtype=self.feature_mean.dtype)
        spread = torch.as_tensor(scale, dtype=self.feature_scale.dtype)
        if center.shape != self.feature_mean.shape or spread.shape != self.feature_scale.shape:
            raise ValueError("profile feature normalization has the wrong shape.")
        if torch.any(~torch.isfinite(center)) or torch.any(~torch.isfinite(spread)) or torch.any(spread <= 0.0):
            raise ValueError("profile feature normalization must be finite and positive.")
        self.feature_mean.copy_(center)
        self.feature_scale.copy_(spread)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != len(SEGMENT_PROFILE_FEATURES):
            raise ValueError("segment profile features have the wrong shape.")
        hidden = self.trunk((features - self.feature_mean) / self.feature_scale)
        output = self.output(hidden)
        return {
            "mean": features[:, :3] + output[:, :3],
            "scale": F.softplus(output[:, 3:]) + self.config.minimum_scale,
        }


class ConditionalGenerator:
    """Observe audited evidence; structured realization is the next stage seam."""

    def __init__(
        self,
        network: ObservableEvidenceNetwork,
        *,
        target_contract: ObservableTargetContract,
        dominant_frequency_hz: float,
        sample_domain: str,
        profile_head: SegmentProfileHead | None = None,
        coefficient_variance_calibration: CoefficientVarianceCalibration | None = None,
        semi_markov_prior: SemiMarkovPrior | None = None,
        semi_markov_conditioning: SemiMarkovConditioning | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if not np.isfinite(dominant_frequency_hz) or dominant_frequency_hz <= 0.0:
            raise ValueError("dominant_frequency_hz must be finite and positive.")
        domain = str(sample_domain).strip().casefold()
        if domain not in {"time", "depth"}:
            raise ValueError("sample_domain must be time or depth.")
        if target_contract.sample_domain != domain:
            raise ValueError("target contract and generator domains differ.")
        self.network = network
        self.target_contract = target_contract
        self.dominant_frequency_hz = float(dominant_frequency_hz)
        self.sample_domain = domain
        self.device = torch.device(device)
        self.network.to(self.device)
        self.profile_head = profile_head
        if self.profile_head is not None:
            self.profile_head.to(self.device)
        if coefficient_variance_calibration is not None and self.profile_head is None:
            raise ValueError("coefficient variance calibration requires a profile head.")
        self.coefficient_variance_calibration = coefficient_variance_calibration
        if (semi_markov_prior is None) != (semi_markov_conditioning is None):
            raise ValueError("semi-Markov prior and conditioning must be supplied together.")
        self.semi_markov_prior = semi_markov_prior
        self.semi_markov_conditioning = semi_markov_conditioning

    @property
    def network_config(self) -> EvidenceNetworkConfig:
        return self.network.config

    def observe(
        self,
        tile: ObservationTile,
        *,
        vp_model_mps: np.ndarray | None = None,
    ) -> ObservableEvidence:
        axis = tile.model_axis
        contract = self.target_contract
        if (
            tile.sample_domain != self.sample_domain
            or axis.unit != contract.sample_unit
            or axis.depth_basis != contract.depth_basis
        ):
            raise InputContractError(
                "observation domain, unit, or depth basis differs from the "
                "audited target contract."
            )
        anchor = build_lfm_anchor(tile)
        lfm_residual = lfm_residual_from_anchor(tile, anchor)
        tuning = tuning_scale_on_model_axis(
            tile,
            dominant_frequency=self.dominant_frequency_hz,
            vp_model_mps=vp_model_mps,
        )
        self.network.eval()
        with torch.no_grad():
            output = self.network(
                torch.as_tensor(
                    tile.seismic[None], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    lfm_residual[None], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    tile.observed_valid[None], dtype=torch.bool, device=self.device
                ),
                torch.as_tensor(
                    tile.lateral_m[None], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    tile.lateral_valid[None], dtype=torch.bool, device=self.device
                ),
            )
        support = output["support"][0].cpu().numpy().astype(bool)
        support &= anchor.model_support
        increment_mean = output[
            "projected_log_ai_increment_mean"
        ][0].cpu().numpy().astype(np.float64)
        increment_scale = output[
            "projected_log_ai_increment_scale"
        ][0].cpu().numpy().astype(np.float64)
        reflectivity_mean = output[
            "signed_reflectivity_mean"
        ][0].cpu().numpy().astype(np.float64)
        reflectivity_scale = output[
            "signed_reflectivity_scale"
        ][0].cpu().numpy().astype(np.float64)
        state_log_potential = output[
            "state_log_potential"
        ][0].cpu().numpy().astype(np.float64)

        increment_mean[~support] = 0.0
        increment_scale[~support] = (
            self.network_config.projected_log_ai_increment_scale
        )
        reflectivity_mean[~support] = 0.0
        reflectivity_scale[~support] = self.network_config.signed_reflectivity_scale
        state_log_potential[~support] = -np.log(3.0)
        tuning[~support] = axis.sample_interval
        background = anchor.model.copy()
        background[~support] = 0.0
        return ObservableEvidence(
            model_axis=axis,
            highres_axis=tile.highres_axis,
            background_lfm_linear=background,
            background_lfm_linear_highres=anchor.highres,
            projected_log_ai_increment_mean=increment_mean,
            projected_log_ai_increment_scale=increment_scale,
            signed_reflectivity_mean=reflectivity_mean,
            signed_reflectivity_scale=reflectivity_scale,
            state_log_potential=state_log_potential,
            local_tuning_scale=tuning,
            support=support,
            highres_support=anchor.highres_support,
            lateral_m=tile.lateral_m,
            x_m=tile.x_m,
            y_m=tile.y_m,
            identity=tile.identity,
        )

    def realize(
        self,
        evidence: ObservableEvidence,
        policy: GenerationPolicy = GenerationPolicy(),
    ) -> StructuredPrediction:
        """Sample one exact per-trace HSMM posterior and decoded K-member ensemble."""

        if self.profile_head is None:
            raise RuntimeError("structured realization requires a profile head.")
        if self.coefficient_variance_calibration is None:
            raise RuntimeError(
                "structured realization requires coefficient variance calibration."
            )
        if self.semi_markov_prior is None or self.semi_markov_conditioning is None:
            raise RuntimeError("structured realization requires a semi-Markov contract.")
        if evidence.model_axis.sample_domain != self.sample_domain:
            raise InputContractError("evidence and generator sample domains differ.")

        posteriors: list[tuple[int, np.ndarray, np.ndarray, object]] = []
        maximum_model_samples = 0
        for trace in range(evidence.support.shape[0]):
            model_indices = np.flatnonzero(evidence.support[trace])
            highres_indices = np.flatnonzero(evidence.highres_support[trace])
            if model_indices.size == 0 and highres_indices.size == 0:
                continue
            if (
                model_indices.size == 0
                or highres_indices.size == 0
                or np.any(np.diff(model_indices) != 1)
                or np.any(np.diff(highres_indices) != 1)
            ):
                raise InputContractError(
                    "each realized trace requires one contiguous model/highres zone."
                )
            posterior = exact_semi_markov_posterior(
                np.exp(evidence.state_log_potential[trace, model_indices]),
                np.full(model_indices.size, 0.5, dtype=np.float64),
                self.semi_markov_prior,
                self.semi_markov_conditioning,
            )
            posteriors.append((trace, model_indices, highres_indices, posterior))
            maximum_model_samples = max(maximum_model_samples, model_indices.size)
        if not posteriors:
            raise InputContractError("structured realization has no supported traces.")

        x_m = evidence.lateral_m if evidence.x_m is None else evidence.x_m
        y_m = np.zeros_like(evidence.lateral_m) if evidence.y_m is None else evidence.y_m
        path_draw_count = maximum_model_samples + 1
        coefficient_draw_offset = path_draw_count
        draw_count = path_draw_count + 3 * maximum_model_samples
        random_field_identity = stable_random_identity(
            "structured_ginn_v2_realize",
            policy.random_identity,
            evidence.identity,
        )
        uniforms = coordinate_stable_uniforms(
            x_m,
            y_m,
            realization_count=policy.realization_count,
            draw_count=draw_count,
            random_identity=random_field_identity,
            correlation_length_m=policy.lateral_correlation_m,
        )
        member_identities = tuple(
            stable_random_identity(
                "structured_ginn_v2_member",
                policy.random_identity,
                member_index,
            )
            for member_index in range(policy.realization_count)
        )
        members: list[StructuredRealization] = []
        interface_rows: list[np.ndarray] = []
        segment_count_rows: list[np.ndarray] = []
        duration_rows: list[list[list[float]]] = []
        projection_support: np.ndarray | None = None

        for member_index, member_identity in enumerate(member_identities):
            extents: list[SegmentExtent] = []
            local_segment_order: list[int] = []
            path_score = 0.0
            member_durations: list[list[float]] = [
                [] for _ in range(evidence.support.shape[0])
            ]
            member_segment_count = np.zeros(
                evidence.support.shape[0], dtype=np.int32
            )
            member_interface = np.zeros(
                evidence.highres_support.shape, dtype=np.float64
            )
            for trace, model_indices, highres_indices, posterior_object in posteriors:
                posterior = posterior_object
                path = posterior.sample(uniforms[member_index, trace, :path_draw_count])
                mapped = path_to_highres_extents(
                    path,
                    trace_index=trace,
                    model_indices=model_indices,
                    model_coordinates=evidence.model_axis.coordinates,
                    highres_indices=highres_indices,
                    highres_coordinates=evidence.highres_axis.coordinates,
                )
                extents.extend(mapped)
                local_segment_order.extend(range(len(mapped)))
                path_score += float(path.log_score)
                member_segment_count[trace] = len(mapped)
                member_durations[trace].extend(
                    float(item.duration_fraction) for item in mapped
                )
                zone_start = int(highres_indices[0])
                for item in mapped:
                    if item.start_index != zone_start:
                        member_interface[trace, item.start_index] = 1.0
            extent_tuple = tuple(extents)
            distributions = self.parameterize_segments(evidence, extent_tuple)
            if len(distributions) != len(local_segment_order):
                raise RuntimeError("segment parameterization changed the path length.")
            trace_segment_cursor = np.zeros(
                evidence.support.shape[0], dtype=np.int64
            )
            sampled_segments: list[Segment] = []
            for distribution in distributions:
                extent = distribution.extent
                local_order = int(trace_segment_cursor[extent.trace_index])
                trace_segment_cursor[extent.trace_index] += 1
                draw_start = coefficient_draw_offset + 3 * local_order
                gaussian = ndtri(
                    uniforms[
                        member_index,
                        extent.trace_index,
                        draw_start : draw_start + 3,
                    ]
                )
                coefficient = (
                    np.asarray(distribution.mean, dtype=np.float64)
                    + np.asarray(distribution.scale, dtype=np.float64) * gaussian
                )
                sampled_segments.append(
                    Segment(
                        trace_index=extent.trace_index,
                        state_id=extent.state_id,
                        start_index=extent.start_index,
                        stop_index=extent.stop_index,
                        c0=float(coefficient[0]),
                        c1=float(coefficient[1]),
                        c2=float(coefficient[2]),
                    )
                )
            decoded, state_highres = decode_segments_numpy(
                evidence.background_lfm_linear_highres,
                sampled_segments,
            )
            projected, current_projection_support = project_supported_highres_to_model(
                decoded,
                evidence.highres_support,
                highres_interval=evidence.highres_axis.sample_interval,
                model_interval=evidence.model_axis.sample_interval,
            )
            current_projection_support &= evidence.support
            projected[~current_projection_support] = np.nan
            if projection_support is None:
                projection_support = current_projection_support
            elif not np.array_equal(projection_support, current_projection_support):
                raise RuntimeError("ensemble members produced different projection support.")
            members.append(
                StructuredRealization(
                    identity=member_identity,
                    log_ai_highres=decoded,
                    state_highres=state_highres,
                    projected_log_ai=projected,
                    segments=tuple(sampled_segments),
                    conditional_log_score=float(path_score),
                )
            )
            interface_rows.append(member_interface)
            segment_count_rows.append(member_segment_count)
            duration_rows.append(member_durations)

        if projection_support is None or not np.any(projection_support):
            raise InputContractError(
                "structured realization has no complete projection support."
            )
        highres_stack = np.stack([item.log_ai_highres for item in members])
        projected_stack = np.stack([item.projected_log_ai for item in members])
        state_stack = np.stack([item.state_highres for item in members])
        highres_mean = np.full(evidence.highres_support.shape, np.nan, dtype=np.float64)
        highres_std = np.full_like(highres_mean, np.nan)
        highres_mean[evidence.highres_support] = np.mean(
            highres_stack[:, evidence.highres_support], axis=0
        )
        highres_std[evidence.highres_support] = np.std(
            highres_stack[:, evidence.highres_support], axis=0
        )
        projected_mean = np.full(evidence.support.shape, np.nan, dtype=np.float64)
        projected_std = np.full_like(projected_mean, np.nan)
        projected_mean[projection_support] = np.mean(
            projected_stack[:, projection_support], axis=0
        )
        projected_std[projection_support] = np.std(
            projected_stack[:, projection_support], axis=0
        )
        state_occupancy = np.zeros(
            evidence.highres_support.shape + (3,), dtype=np.float64
        )
        for state in range(3):
            state_occupancy[..., state] = np.mean(state_stack == state, axis=0)
        state_occupancy[~evidence.highres_support] = 0.0
        interface_density = np.mean(np.stack(interface_rows), axis=0)
        segment_count = np.stack(segment_count_rows).astype(np.float64)
        duration_mean = np.full(evidence.support.shape[0], np.nan, dtype=np.float64)
        duration_std = np.full_like(duration_mean, np.nan)
        for trace in range(evidence.support.shape[0]):
            values = np.asarray(
                [
                    duration
                    for member_duration in duration_rows
                    for duration in member_duration[trace]
                ],
                dtype=np.float64,
            )
            if values.size:
                duration_mean[trace] = float(np.mean(values))
                duration_std[trace] = float(np.std(values))

        evidence_target = (
            evidence.background_lfm_linear
            + evidence.projected_log_ai_increment_mean
        )
        squared_error = np.asarray(
            [
                np.mean(
                    (member.projected_log_ai[projection_support]
                    - evidence_target[projection_support])
                    ** 2
                )
                for member in members
            ],
            dtype=np.float64,
        )
        conditional_score = np.asarray(
            [member.conditional_log_score for member in members],
            dtype=np.float64,
        )
        representative_index = int(
            np.lexsort((-conditional_score, squared_error))[0]
        )
        summary = EnsembleSummary(
            log_ai_highres_mean=highres_mean,
            log_ai_highres_std=highres_std,
            projected_log_ai_mean=projected_mean,
            projected_log_ai_std=projected_std,
            projected_support=projection_support,
            state_occupancy_highres=state_occupancy,
            interface_density_highres=interface_density,
            segment_count_mean=np.mean(segment_count, axis=0),
            segment_count_std=np.std(segment_count, axis=0),
            segment_duration_fraction_mean=duration_mean,
            segment_duration_fraction_std=duration_std,
        )
        return StructuredPrediction(
            evidence=evidence,
            representative=members[representative_index],
            summary=summary,
            realization_identities=member_identities,
            realizations=tuple(members) if policy.retain_realizations else None,
            diagnostics={
                "semi_markov_forward_recursions": float(len(posteriors)),
                "semi_markov_backward_samples": float(
                    len(posteriors) * policy.realization_count
                ),
                "realization_count": float(policy.realization_count),
                "representative_member_index": float(representative_index),
                "representative_evidence_rmse": float(
                    np.sqrt(squared_error[representative_index])
                ),
                "projection_supported_samples": float(
                    np.count_nonzero(projection_support)
                ),
            },
        )

    def attach_profile_head(
        self,
        config: SegmentProfileHeadConfig = SegmentProfileHeadConfig(),
    ) -> SegmentProfileHead:
        if self.profile_head is not None:
            raise ValueError("generator already has a segment profile head.")
        self.profile_head = SegmentProfileHead(config).to(self.device)
        return self.profile_head

    def set_coefficient_variance_calibration(
        self,
        calibration: CoefficientVarianceCalibration,
    ) -> None:
        if self.profile_head is None:
            raise ValueError("coefficient variance calibration requires a profile head.")
        if self.coefficient_variance_calibration is not None:
            raise ValueError("generator already has coefficient variance calibration.")
        self.coefficient_variance_calibration = calibration

    def set_semi_markov_contract(
        self,
        prior: SemiMarkovPrior,
        conditioning: SemiMarkovConditioning,
    ) -> None:
        if self.coefficient_variance_calibration is None:
            raise ValueError(
                "semi-Markov realization contract requires calibrated coefficients."
            )
        if self.semi_markov_prior is not None:
            raise ValueError("generator already has a semi-Markov contract.")
        self.semi_markov_prior = prior
        self.semi_markov_conditioning = conditioning

    def parameterize_segments(
        self,
        evidence: ObservableEvidence,
        segments: tuple[SegmentExtent, ...],
    ) -> tuple[SegmentParameterDistribution, ...]:
        if self.profile_head is None:
            raise RuntimeError("generator has no trained segment profile head.")
        features = build_segment_profile_features(evidence, segments)
        self.profile_head.eval()
        with torch.no_grad():
            output = self.profile_head(
                torch.as_tensor(features, dtype=torch.float32, device=self.device)
            )
        mean = output["mean"].cpu().numpy().astype(np.float64)
        scale = output["scale"].cpu().numpy().astype(np.float64)
        if self.coefficient_variance_calibration is not None:
            scale *= np.asarray(
                self.coefficient_variance_calibration.temperature,
                dtype=np.float64,
            )
        result: list[SegmentParameterDistribution] = []
        for index, extent in enumerate(segments):
            basis = profile_basis(extent.stop_index - extent.start_index)
            rank = int(np.linalg.matrix_rank(basis, tol=1.0e-10))
            condition = float(np.linalg.cond(basis)) if rank == 3 else float("inf")
            result.append(
                SegmentParameterDistribution(
                    extent=extent,
                    mean=tuple(float(value) for value in mean[index]),
                    scale=tuple(float(value) for value in scale[index]),
                    parameter_identifiability_rank=rank,
                    parameter_basis_condition=condition,
                )
            )
        return tuple(result)

    def state_dict_payload(self) -> dict[str, Any]:
        payload = {
            "schema": (
                "structured_ginn_v2_generator_v6"
                if self.semi_markov_prior is not None
                else (
                    "structured_ginn_v2_generator_v5"
                    if self.coefficient_variance_calibration is not None
                    else (
                        "structured_ginn_v2_generator_v4"
                        if self.profile_head is not None
                        else "structured_ginn_v2_generator_v3"
                    )
                )
            ),
            "network_config": asdict(self.network.config),
            "network_state": self.network.state_dict(),
            "target_contract": self.target_contract.to_mapping(),
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "sample_domain": self.sample_domain,
        }
        if self.profile_head is not None:
            payload["profile_head_config"] = asdict(self.profile_head.config)
            payload["profile_head_state"] = self.profile_head.state_dict()
        if self.coefficient_variance_calibration is not None:
            payload["coefficient_variance_calibration"] = (
                self.coefficient_variance_calibration.to_mapping()
            )
        if self.semi_markov_prior is not None:
            if self.semi_markov_conditioning is None:
                raise RuntimeError("generator semi-Markov conditioning is missing.")
            payload["semi_markov_prior"] = self.semi_markov_prior.to_mapping()
            payload["semi_markov_conditioning"] = (
                self.semi_markov_conditioning.to_mapping()
            )
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
    ) -> "ConditionalGenerator":
        schema = payload.get("schema")
        if schema not in {
            "structured_ginn_v2_generator_v3",
            "structured_ginn_v2_generator_v4",
            "structured_ginn_v2_generator_v5",
            "structured_ginn_v2_generator_v6",
        }:
            raise ValueError("unsupported Structured GINN V2 checkpoint schema.")
        target_contract = ObservableTargetContract.from_mapping(
            payload["target_contract"]
        )
        config = EvidenceNetworkConfig.from_mapping(payload["network_config"])
        network = ObservableEvidenceNetwork(config)
        network.load_state_dict(payload["network_state"], strict=True)
        profile_head = None
        if schema in {
            "structured_ginn_v2_generator_v4",
            "structured_ginn_v2_generator_v5",
            "structured_ginn_v2_generator_v6",
        }:
            profile_head = SegmentProfileHead(
                SegmentProfileHeadConfig.from_mapping(payload["profile_head_config"])
            )
            profile_head.load_state_dict(payload["profile_head_state"], strict=True)
        variance_calibration = None
        if schema in {
            "structured_ginn_v2_generator_v5",
            "structured_ginn_v2_generator_v6",
        }:
            variance_calibration = CoefficientVarianceCalibration.from_mapping(
                payload["coefficient_variance_calibration"]
            )
        semi_markov_prior = None
        semi_markov_conditioning = None
        if schema == "structured_ginn_v2_generator_v6":
            semi_markov_prior = SemiMarkovPrior.from_mapping(
                payload["semi_markov_prior"]
            )
            semi_markov_conditioning = SemiMarkovConditioning.from_mapping(
                payload["semi_markov_conditioning"]
            )
        return cls(
            network,
            target_contract=target_contract,
            dominant_frequency_hz=float(payload["dominant_frequency_hz"]),
            sample_domain=str(payload["sample_domain"]),
            profile_head=profile_head,
            coefficient_variance_calibration=variance_calibration,
            semi_markov_prior=semi_markov_prior,
            semi_markov_conditioning=semi_markov_conditioning,
            device=device,
        )


__all__ = [
    "ConditionalGenerator",
    "SegmentProfileHead",
    "SegmentProfileHeadConfig",
]
