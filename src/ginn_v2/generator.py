"""The deep ConditionalGenerator interface."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import torch

from ginn_v2.contracts import (
    GenerationPolicy,
    InputContractError,
    ObservableEvidence,
    ObservableTargetContract,
    ObservationTile,
    StructuredPrediction,
)
from ginn_v2.evidence import (
    EvidenceNetworkConfig,
    ObservableEvidenceNetwork,
    tuning_scale_on_model_axis,
)
from ginn_v2.representation import build_lfm_anchor, lfm_residual_from_anchor


class ConditionalGenerator:
    """Observe audited evidence; structured realization is the next stage seam."""

    def __init__(
        self,
        network: ObservableEvidenceNetwork,
        *,
        target_contract: ObservableTargetContract,
        dominant_frequency_hz: float,
        sample_domain: str,
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
            projected_log_ai_increment_mean=increment_mean,
            projected_log_ai_increment_scale=increment_scale,
            signed_reflectivity_mean=reflectivity_mean,
            signed_reflectivity_scale=reflectivity_scale,
            state_log_potential=state_log_potential,
            local_tuning_scale=tuning,
            support=support,
            lateral_m=tile.lateral_m,
            x_m=tile.x_m,
            y_m=tile.y_m,
        )

    def realize(
        self,
        evidence: ObservableEvidence,
        policy: GenerationPolicy = GenerationPolicy(),
    ) -> StructuredPrediction:
        del evidence, policy
        raise NotImplementedError(
            "Conditional HSMM realization starts in stage 1 step 2; this "
            "checkpoint currently exposes audited observable evidence only."
        )

    def state_dict_payload(self) -> dict[str, Any]:
        return {
            "schema": "structured_ginn_v2_generator_v3",
            "network_config": asdict(self.network.config),
            "network_state": self.network.state_dict(),
            "target_contract": self.target_contract.to_mapping(),
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
        if payload.get("schema") != "structured_ginn_v2_generator_v3":
            raise ValueError("unsupported Structured GINN V2 checkpoint schema.")
        target_contract = ObservableTargetContract.from_mapping(
            payload["target_contract"]
        )
        config = EvidenceNetworkConfig.from_mapping(payload["network_config"])
        network = ObservableEvidenceNetwork(config)
        network.load_state_dict(payload["network_state"], strict=True)
        return cls(
            network,
            target_contract=target_contract,
            dominant_frequency_hz=float(payload["dominant_frequency_hz"]),
            sample_domain=str(payload["sample_domain"]),
            device=device,
        )


__all__ = ["ConditionalGenerator"]
