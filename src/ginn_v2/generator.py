"""The public seam for the Structured GINN V2 conditional generator."""

from __future__ import annotations

import numpy as np
import torch

from ginn_v2.contracts import (
    DomainMismatchError,
    GenerationPolicy,
    InputContractError,
    ObservableEvidence,
    ObservationTile,
    StructuredEnsemble,
)
from ginn_v2.evidence import (
    EvidenceNetworkConfig,
    ObservableEvidenceNetwork,
    tuning_scale_on_model_axis,
)
from ginn_v2.representation import build_lfm_anchor, lfm_residual_from_anchor


class ConditionalGenerator:
    """Observe a tile and own the complete section-generation implementation.

    Stage 0 exposes the seam and the evidence adapter.  Event-track decoding,
    sampling, and rasterization are implemented behind this seam in Stage 1;
    callers do not assemble per-trace paths themselves.
    """

    def __init__(
        self,
        network: ObservableEvidenceNetwork,
        *,
        dominant_frequency_hz: float,
        sample_domain: str,
        device: str | torch.device = "cpu",
    ) -> None:
        domain = str(sample_domain).strip().casefold()
        if domain not in {"time", "depth"}:
            raise ValueError("sample_domain must be time or depth.")
        if not np.isfinite(dominant_frequency_hz) or dominant_frequency_hz <= 0.0:
            raise ValueError("dominant_frequency_hz must be finite and positive.")
        if network.config.input_mode not in {"full", "no_seismic"}:
            raise ValueError("evidence network input_mode is invalid.")
        self.network = network
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
        """Convert one tile into the public band-limited evidence contract."""

        if tile.sample_domain != self.sample_domain:
            raise DomainMismatchError("observation and generator domains differ.")
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
                torch.as_tensor(tile.seismic[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(lfm_residual[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(tile.observed_valid[None], dtype=torch.bool, device=self.device),
                torch.as_tensor(tile.lateral_m[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(tile.lateral_valid[None], dtype=torch.bool, device=self.device),
            )

        support = output["support"][0].cpu().numpy().astype(bool)
        support &= anchor.model_support
        increment_mean = output["projected_log_ai_increment_mean"][0].cpu().numpy().astype(np.float64)
        increment_scale = output["projected_log_ai_increment_scale"][0].cpu().numpy().astype(np.float64)
        reflectivity_mean = output["signed_reflectivity_mean"][0].cpu().numpy().astype(np.float64)
        reflectivity_scale = output["signed_reflectivity_scale"][0].cpu().numpy().astype(np.float64)
        state_log_potential = output["state_log_potential"][0].cpu().numpy().astype(np.float64)

        increment_mean[~support] = 0.0
        increment_scale[~support] = self.network_config.projected_log_ai_increment_scale
        reflectivity_mean[~support] = 0.0
        reflectivity_scale[~support] = self.network_config.signed_reflectivity_scale
        state_log_potential[~support] = -np.log(3.0)
        tuning[~support] = tile.model_axis.sample_interval
        background = anchor.model.copy()
        background[~support] = 0.0
        return ObservableEvidence(
            model_axis=tile.model_axis,
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

    def generate(
        self,
        tile: ObservationTile,
        policy: GenerationPolicy = GenerationPolicy(),
        *,
        vp_model_mps: np.ndarray | None = None,
    ) -> StructuredEnsemble:
        """Generate one complete section ensemble behind the public seam."""

        self.observe(tile, vp_model_mps=vp_model_mps)
        raise NotImplementedError(
            "section-level ordered event generation is implemented in Structured GINN V2 Stage 1."
        )


__all__ = ["ConditionalGenerator"]
