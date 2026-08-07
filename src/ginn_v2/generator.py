"""The public seam for the Structured GINN V2 conditional generator."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from ginn_v2.contracts import (
    DomainMismatchError,
    BandlimitedEvidence,
    GenerationPolicy,
    InputContractError,
    ObservationTile,
    StructuredEnsemble,
)
from ginn_v2.evidence import (
    EvidenceNetworkConfig,
    BandlimitedEvidenceNetwork,
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
        network: BandlimitedEvidenceNetwork,
        *,
        dominant_frequency_hz: float,
        sample_domain: str,
        device: str | torch.device = "cpu",
        event_network: torch.nn.Module | None = None,
        producer_prior: Any | None = None,
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
        self.event_network = event_network
        self.producer_prior = producer_prior
        if self.event_network is not None:
            self.event_network.to(self.device)
        if (self.event_network is None) != (self.producer_prior is None):
            raise ValueError("event_network and producer_prior must be supplied together.")

    @property
    def network_config(self) -> EvidenceNetworkConfig:
        return self.network.config

    def observe(
        self,
        tile: ObservationTile,
        *,
        vp_model_mps: np.ndarray | None = None,
    ) -> BandlimitedEvidence:
        """Convert one tile into the public band-limited evidence contract."""

        velocity = None if vp_model_mps is None else (vp_model_mps,)
        return self.observe_many((tile,), vp_model_mps=velocity)[0]

    def observe_many(
        self,
        tiles: Sequence[ObservationTile],
        *,
        vp_model_mps: Sequence[np.ndarray | None] | None = None,
    ) -> tuple[BandlimitedEvidence, ...]:
        """Batch observations without widening the evidence-to-event seam."""

        parsed = tuple(tiles)
        if not parsed:
            raise InputContractError("observe_many requires at least one tile.")
        if any(tile.sample_domain != self.sample_domain for tile in parsed):
            raise DomainMismatchError("observation and generator domains differ.")
        shape = parsed[0].seismic.shape
        if any(tile.seismic.shape != shape for tile in parsed):
            raise InputContractError("batched observation tiles must share one shape.")
        if vp_model_mps is None:
            velocities: tuple[np.ndarray | None, ...] = tuple(None for _ in parsed)
        else:
            velocities = tuple(vp_model_mps)
            if len(velocities) != len(parsed):
                raise InputContractError("batched Vp inputs must match the tile count.")
        anchors = tuple(build_lfm_anchor(tile) for tile in parsed)
        residuals = tuple(
            lfm_residual_from_anchor(tile, anchor)
            for tile, anchor in zip(parsed, anchors, strict=True)
        )
        tunings = tuple(
            tuning_scale_on_model_axis(
                tile,
                dominant_frequency=self.dominant_frequency_hz,
                vp_model_mps=velocity,
            )
            for tile, velocity in zip(parsed, velocities, strict=True)
        )
        self.network.eval()
        with torch.no_grad():
            output = self.network(
                torch.as_tensor(np.stack([tile.seismic for tile in parsed]), dtype=torch.float32, device=self.device),
                torch.as_tensor(np.stack(residuals), dtype=torch.float32, device=self.device),
                torch.as_tensor(np.stack([tile.observed_valid for tile in parsed]), dtype=torch.bool, device=self.device),
                torch.as_tensor(np.stack([tile.lateral_m for tile in parsed]), dtype=torch.float32, device=self.device),
                torch.as_tensor(np.stack([tile.lateral_valid for tile in parsed]), dtype=torch.bool, device=self.device),
            )
        state_output = torch.softmax(output["state_logits"], dim=-1).cpu().numpy()
        results: list[BandlimitedEvidence] = []
        for index, (tile, anchor, tuning) in enumerate(
            zip(parsed, anchors, tunings, strict=True)
        ):
            support = output["support"][index].cpu().numpy().astype(bool)
            support &= anchor.model_support
            increment_mean = output["projected_log_ai_increment_mean"][index].cpu().numpy().astype(np.float64)
            increment_scale = output["projected_log_ai_increment_scale"][index].cpu().numpy().astype(np.float64)
            reflectivity_mean = output["signed_reflectivity_mean"][index].cpu().numpy().astype(np.float64)
            reflectivity_scale = output["signed_reflectivity_scale"][index].cpu().numpy().astype(np.float64)
            state_fraction = state_output[index].astype(np.float64)
            increment_mean[~support] = 0.0
            increment_scale[~support] = self.network_config.projected_log_ai_increment_scale
            reflectivity_mean[~support] = 0.0
            reflectivity_scale[~support] = self.network_config.signed_reflectivity_scale
            state_fraction[~support] = 1.0 / 3.0
            tuning[~support] = tile.model_axis.sample_interval
            background = anchor.model.copy()
            background[~support] = 0.0
            results.append(
                BandlimitedEvidence(
                    model_axis=tile.model_axis,
                    highres_axis=tile.highres_axis,
                    background_lfm_linear=background,
                    background_lfm_linear_highres=anchor.highres,
                    projected_log_ai_increment_mean=increment_mean,
                    projected_log_ai_increment_scale=increment_scale,
                    signed_reflectivity_mean=reflectivity_mean,
                    signed_reflectivity_scale=reflectivity_scale,
                    state_fraction=state_fraction,
                    local_tuning_scale=tuning,
                    support=support,
                    highres_support=anchor.highres_support,
                    lateral_m=tile.lateral_m,
                    x_m=tile.x_m,
                    y_m=tile.y_m,
                    identity=tile.identity,
                )
            )
        return tuple(results)

    def generate(
        self,
        tile: ObservationTile,
        policy: GenerationPolicy = GenerationPolicy(),
        *,
        vp_model_mps: np.ndarray | None = None,
    ) -> StructuredEnsemble:
        """Generate one complete section ensemble behind the public seam."""

        if self.event_network is None or self.producer_prior is None:
            raise InputContractError("generator has no fitted EventTrack decoder.")
        evidence = self.observe(tile, vp_model_mps=vp_model_mps)
        from ginn_v2.events import sample_event_track_realizations

        realizations = sample_event_track_realizations(
            self.event_network,
            self.producer_prior,
            evidence,
            tile,
            policy,
        )
        target = (
            evidence.background_lfm_linear
            + evidence.projected_log_ai_increment_mean
        )
        scale = np.maximum(evidence.projected_log_ai_increment_scale, 0.01)
        scores: list[float] = []
        raw_rmse: list[float] = []
        for realization in realizations:
            valid = evidence.support & realization.model_support
            if not np.any(valid):
                raise InputContractError(
                    "sampled realization has no overlap with band-limited evidence."
                )
            residual = realization.model_log_ai[valid] - target[valid]
            scores.append(float(np.sqrt(np.mean((residual / scale[valid]) ** 2))))
            raw_rmse.append(float(np.sqrt(np.mean(residual**2))))
        event_counts = np.asarray(
            [len(item.tracks) for item in realizations], dtype=np.int64
        )
        density_center = float(np.median(event_counts))
        density_deviation = np.abs(event_counts.astype(np.float64) - density_center)
        candidate_count = max(1, int(math.ceil(policy.realization_count / 2.0)))
        density_candidates = np.argsort(density_deviation, kind="stable")[
            :candidate_count
        ]
        representative_index = int(
            density_candidates[
                np.argmin(np.asarray(scores, dtype=np.float64)[density_candidates])
            ]
        )
        representative = realizations[representative_index]

        highres_stack = np.stack(
            [item.log_ai_highres for item in realizations], axis=0
        )
        model_stack = np.stack([item.model_log_ai for item in realizations], axis=0)
        state_stack = np.stack([item.state_highres for item in realizations], axis=0)
        highres_support = np.asarray(evidence.highres_support, dtype=bool).copy()
        model_support = np.asarray(evidence.support, dtype=bool).copy()
        state_occupancy = np.stack(
            [np.mean(state_stack == state, axis=0) for state in range(3)], axis=-1
        )
        for realization in realizations:
            highres_support &= np.isfinite(realization.log_ai_highres)
            model_support &= realization.model_support & np.isfinite(
                realization.model_log_ai
            )
        if not np.any(highres_support) or not np.any(model_support):
            raise InputContractError(
                "ensemble members have no common supported samples."
            )
        highres_mean = np.where(highres_support, np.mean(highres_stack, axis=0), 0.0)
        highres_std = np.where(highres_support, np.std(highres_stack, axis=0), 0.0)
        model_mean = np.where(model_support, np.mean(model_stack, axis=0), 0.0)
        model_std = np.where(model_support, np.std(model_stack, axis=0), 0.0)
        state_occupancy[~highres_support] = 0.0
        ensemble_residual = model_mean[model_support] - target[model_support]
        retained = realizations if policy.retain_realizations else tuple()
        return StructuredEnsemble(
            evidence=evidence,
            representative=representative,
            realizations=retained,
            summary={
                "realization_count": policy.realization_count,
                "member_identities": tuple(item.identity for item in realizations),
                "representative_member_index": representative_index,
                "representative_identity": representative.identity,
                "representative_density_candidate_count": candidate_count,
                "event_count_median": density_center,
                "event_count": event_counts,
                "highres_log_ai_mean": highres_mean,
                "highres_log_ai_std": highres_std,
                "model_log_ai_mean": model_mean,
                "model_log_ai_std": model_std,
                "state_occupancy_highres": state_occupancy,
                "highres_support": highres_support,
                "model_support": model_support,
                "stage": "conditional_event_track_ensemble",
            },
            diagnostics={
                "supported_model_fraction": float(np.mean(model_support)),
                "representative_evidence_standardized_rmse": scores[
                    representative_index
                ],
                "representative_evidence_rmse": raw_rmse[representative_index],
                "ensemble_mean_evidence_rmse": float(
                    np.sqrt(np.mean(ensemble_residual**2))
                ),
                "event_count_mean": float(np.mean(event_counts)),
                "event_count_std": float(np.std(event_counts)),
                "representative_event_count_deviation_from_median": float(
                    density_deviation[representative_index]
                ),
                "highres_ensemble_std_mean": float(
                    np.mean(highres_std[highres_support])
                ),
                "model_ensemble_std_mean": float(np.mean(model_std[model_support])),
            },
        )


__all__ = ["ConditionalGenerator"]
