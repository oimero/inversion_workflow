"""The deep ConditionalGenerator interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ginn_v2.contracts import (
    CoefficientVarianceCalibration,
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
from ginn_v2.evidence import (
    EvidenceNetworkConfig,
    ObservableEvidenceNetwork,
    tuning_scale_on_model_axis,
)
from ginn_v2.representation import (
    SEGMENT_PROFILE_FEATURES,
    build_lfm_anchor,
    build_segment_profile_features,
    canonicalize_same_state_segments,
    decode_segments_numpy,
    lfm_residual_from_anchor,
    path_to_highres_extents,
    profile_basis,
    project_supported_highres_to_model,
)
from ginn_v2.semi_markov import (
    SemiMarkovConditioning,
    SemiMarkovDecodePolicy,
    SemiMarkovPrior,
    exact_semi_markov_posterior,
    prior_with_same_state_renewal,
    renewal_probability_from_reflectivity,
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
        decode_policy: SemiMarkovDecodePolicy = SemiMarkovDecodePolicy(),
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
        if not isinstance(decode_policy, SemiMarkovDecodePolicy):
            raise TypeError("decode_policy must be SemiMarkovDecodePolicy.")
        self.decode_policy = decode_policy

    @property
    def network_config(self) -> EvidenceNetworkConfig:
        return self.network.config

    def observe(
        self,
        tile: ObservationTile,
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

    def predict(self, tile: ObservationTile) -> StructuredPrediction:
        """Observe and decode one deterministic structured MAP prediction."""

        return self.decode(self.observe(tile))

    def decode(
        self,
        evidence: ObservableEvidence,
        *,
        policy: SemiMarkovDecodePolicy | None = None,
        conditioning: SemiMarkovConditioning | None = None,
    ) -> StructuredPrediction:
        """Decode cached evidence under one explicit deterministic policy."""

        resolved_policy = self.decode_policy if policy is None else policy
        resolved_conditioning = (
            self.semi_markov_conditioning
            if conditioning is None
            else conditioning
        )
        if not isinstance(resolved_policy, SemiMarkovDecodePolicy):
            raise TypeError("policy must be SemiMarkovDecodePolicy.")
        if not isinstance(resolved_conditioning, SemiMarkovConditioning):
            raise TypeError("conditioning must be SemiMarkovConditioning.")
        return self._decode_evidence(
            evidence,
            policy=resolved_policy,
            conditioning=resolved_conditioning,
        )

    def _decode_evidence(
        self,
        evidence: ObservableEvidence,
        *,
        policy: SemiMarkovDecodePolicy,
        conditioning: SemiMarkovConditioning,
    ) -> StructuredPrediction:
        """Implementation behind the cached-evidence decode interface."""

        if self.profile_head is None:
            raise RuntimeError("structured prediction requires a profile head.")
        if self.semi_markov_prior is None or self.semi_markov_conditioning is None:
            raise RuntimeError("structured prediction requires a semi-Markov contract.")
        if evidence.model_axis.sample_domain != self.sample_domain:
            raise InputContractError("evidence and generator sample domains differ.")

        model_shape = evidence.support.shape
        state_probability = np.full(
            model_shape + (3,), 1.0 / 3.0, dtype=np.float64
        )
        renewal_probability = np.zeros(model_shape, dtype=np.float64)
        segments: list[Segment] = []
        forward_recursions = 0
        conditional_log_score = 0.0
        renewal_evidence_sum = 0.0
        renewal_evidence_count = 0
        same_state_renewal_count = 0
        state_change_count = 0
        decoding_prior = prior_with_same_state_renewal(
            self.semi_markov_prior,
            policy.same_state_renewal_probability,
        )

        for trace in range(model_shape[0]):
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
                    "each deterministic trace requires one contiguous model/highres zone."
                )
            renewal_evidence = renewal_probability_from_reflectivity(
                evidence.signed_reflectivity_mean[trace, model_indices],
                evidence.signed_reflectivity_scale[trace, model_indices],
                policy,
            )
            posterior = exact_semi_markov_posterior(
                np.exp(evidence.state_log_potential[trace, model_indices]),
                renewal_evidence,
                decoding_prior,
                conditioning,
            )
            marginals = posterior.marginals()
            state_probability[trace, model_indices] = marginals.state_probability
            renewal_probability[trace, model_indices] = marginals.renewal_probability
            path = posterior.map_path()
            for previous, current in zip(path.segments[:-1], path.segments[1:]):
                if previous[0] == current[0]:
                    same_state_renewal_count += 1
                else:
                    state_change_count += 1
            renewal_evidence_sum += float(np.sum(renewal_evidence))
            renewal_evidence_count += int(renewal_evidence.size)
            conditional_log_score += float(path.log_score)
            mapped = path_to_highres_extents(
                path,
                trace_index=trace,
                model_indices=model_indices,
                model_coordinates=evidence.model_axis.coordinates,
                highres_indices=highres_indices,
                highres_coordinates=evidence.highres_axis.coordinates,
            )
            distributions = self.parameterize_segments(evidence, mapped)
            if len(distributions) != len(mapped):
                raise RuntimeError("deterministic parameterization changed the MAP path.")
            for extent, distribution in zip(mapped, distributions, strict=True):
                coefficients = np.asarray(distribution.mean, dtype=np.float64)
                segments.append(
                    Segment(
                        trace_index=trace,
                        state_id=extent.state_id,
                        start_index=extent.start_index,
                        stop_index=extent.stop_index,
                        c0=float(coefficients[0]),
                        c1=float(coefficients[1]),
                        c2=float(coefficients[2]),
                    )
                )
            forward_recursions += int(posterior.forward_recursions)

        if not segments:
            raise InputContractError("deterministic prediction has no supported segments.")
        latent_segment_count = len(segments)
        canonical_segments, canonical_merge_count = canonicalize_same_state_segments(
            evidence,
            segments,
            merge_scale_fraction=policy.same_state_merge_scale_fraction,
        )
        segments = list(canonical_segments)
        decoded, state_highres = decode_segments_numpy(
            evidence.background_lfm_linear_highres,
            segments,
        )
        projected, projection_support = project_supported_highres_to_model(
            decoded,
            evidence.highres_support,
            highres_interval=evidence.highres_axis.sample_interval,
            model_interval=evidence.model_axis.sample_interval,
        )
        projection_support &= evidence.support
        projected[~projection_support] = np.nan
        if not np.any(projection_support):
            raise InputContractError(
                "deterministic prediction has no complete projection support."
            )

        bandlimited = (
            evidence.background_lfm_linear
            + evidence.projected_log_ai_increment_mean
        )
        consistency = projection_support & np.isfinite(bandlimited)
        projection_consistency_rmse = float(
            np.sqrt(np.mean((projected[consistency] - bandlimited[consistency]) ** 2))
        )
        realization = StructuredRealization(
            log_ai_highres=decoded,
            state_highres=state_highres,
            projected_log_ai=projected,
            segments=tuple(segments),
            conditional_log_score=conditional_log_score,
        )
        return StructuredPrediction(
            evidence=evidence,
            realization=realization,
            state_probability=state_probability,
            renewal_probability=renewal_probability,
            diagnostics={
                "semi_markov_forward_recursions": float(forward_recursions),
                "deterministic_map": 1.0,
                "map_segment_count": float(latent_segment_count),
                "published_segment_count": float(len(segments)),
                "canonical_same_state_merge_count": float(
                    canonical_merge_count
                ),
                "map_same_state_renewal_count": float(same_state_renewal_count),
                "map_state_change_count": float(state_change_count),
                "renewal_evidence_mean": float(
                    renewal_evidence_sum / max(renewal_evidence_count, 1)
                ),
                "same_state_renewal_probability": float(
                    policy.same_state_renewal_probability
                ),
                "renewal_snr_threshold": float(policy.renewal_snr_threshold),
                "same_state_merge_scale_fraction": float(
                    policy.same_state_merge_scale_fraction
                ),
                "duration_temperature": float(
                    conditioning.duration_temperature
                ),
                "transition_temperature": float(
                    conditioning.transition_temperature
                ),
                "projection_consistency_rmse": projection_consistency_rmse,
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
        if self.profile_head is None:
            raise ValueError("semi-Markov prediction requires a profile head.")
        if self.semi_markov_prior is not None:
            raise ValueError("generator already has a semi-Markov contract.")
        self.semi_markov_prior = prior
        self.semi_markov_conditioning = conditioning

    def set_decode_policy(self, policy: SemiMarkovDecodePolicy) -> None:
        """Set one versioned deterministic decode policy before inference."""

        if not isinstance(policy, SemiMarkovDecodePolicy):
            raise TypeError("decode policy must be SemiMarkovDecodePolicy.")
        self.decode_policy = policy

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
                "structured_ginn_v2_generator_v8"
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
            "decode_policy": self.decode_policy.to_mapping(),
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
            "structured_ginn_v2_generator_v7",
            "structured_ginn_v2_generator_v8",
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
            "structured_ginn_v2_generator_v7",
            "structured_ginn_v2_generator_v8",
        }:
            profile_head = SegmentProfileHead(
                SegmentProfileHeadConfig.from_mapping(payload["profile_head_config"])
            )
            profile_head.load_state_dict(payload["profile_head_state"], strict=True)
        variance_calibration = None
        if "coefficient_variance_calibration" in payload:
            variance_calibration = CoefficientVarianceCalibration.from_mapping(
                payload["coefficient_variance_calibration"]
            )
        semi_markov_prior = None
        semi_markov_conditioning = None
        if schema in {
            "structured_ginn_v2_generator_v7",
            "structured_ginn_v2_generator_v8",
        }:
            semi_markov_prior = SemiMarkovPrior.from_mapping(
                payload["semi_markov_prior"]
            )
            semi_markov_conditioning = SemiMarkovConditioning.from_mapping(
                payload["semi_markov_conditioning"]
            )
        decode_policy = (
            SemiMarkovDecodePolicy.from_mapping(payload["decode_policy"])
            if "decode_policy" in payload
            else SemiMarkovDecodePolicy()
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
            decode_policy=decode_policy,
            device=device,
        )


__all__ = [
    "ConditionalGenerator",
    "SegmentProfileHead",
    "SegmentProfileHeadConfig",
]
