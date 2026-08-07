"""Joint ordered EventTrack generation behind one section-level seam.

The model in this module consumes only the published ``BandlimitedEvidence``
contract and a frozen ``ProducerPrior``.  Each autoregressive renewal step is
shared by the whole section: duration and profile are lateral fields, not
independently generated trace candidates that need matching afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cup.synthetic.core.prior import ProducerPrior
from ginn_v2.artifacts import (
    Corpus,
    StructuredTrainingTile,
    load_checkpoint,
    load_corpus,
    parent_training_tiles,
    save_checkpoint,
)
from ginn_v2.contracts import (
    BandlimitedEvidence,
    EventTrack,
    EventTrackRealization,
    GenerationPolicy,
    InputContractError,
    ObservationTile,
    Segment,
    StructuredEnsemble,
)
from ginn_v2.evidence import (
    BandlimitedEvidenceNetwork,
    EvidenceNetworkConfig,
    ObservationPerturbationProfile,
    build_paired_observation_views,
)
from ginn_v2.representation import (
    build_lfm_anchor,
    decode_segments_numpy,
    fit_profile_coefficients,
    project_supported_highres_to_model,
)
from ginn_v2.runtime import configure_training_logger, resolve_device


EVENT_TRAINING_REPORT_SCHEMA = "structured_ginn_v2_event_training_report_v3"
EVENT_EVALUATION_REPORT_SCHEMA = "structured_ginn_v2_event_evaluation_report_v4"
EVENT_POLICY_CALIBRATION_REPORT_SCHEMA = (
    "structured_ginn_v2_event_policy_calibration_report_v3"
)
EVENT_GENERATION_POLICY_SCHEMA = "structured_ginn_v2_event_generation_policy_v3"
_REUSABLE_EVENT_POLICY_CALIBRATION_SCHEMAS = frozenset(
    {
        "structured_ginn_v2_event_policy_calibration_report_v2",
        EVENT_POLICY_CALIBRATION_REPORT_SCHEMA,
    }
)


@dataclass(frozen=True)
class EventGeneratorConfig:
    """Architecture and bounded-conditioning contract for ordered tracks."""

    architecture: str = "autoregressive_coordinate_v1"
    hidden_channels: int = 48
    vertical_layers: int = 2
    maximum_events: int = 80
    lateral_distance_scale_m: float = 900.0
    evidence_potential_bound: float = 2.0
    duration_sigma_bound: float = 2.0
    coefficient_sigma_bound: float = 2.0
    presence_threshold: float = 0.5
    soft_raster_temperature_fraction: float = 0.01
    minimum_remaining_fraction: float = 1.0e-4

    def __post_init__(self) -> None:
        integer_values = {
            "hidden_channels": self.hidden_channels,
            "vertical_layers": self.vertical_layers,
            "maximum_events": self.maximum_events,
        }
        if any(isinstance(value, bool) or int(value) <= 0 for value in integer_values.values()):
            raise ValueError("event generator integer configuration must be positive.")
        finite_positive = {
            "lateral_distance_scale_m": self.lateral_distance_scale_m,
            "evidence_potential_bound": self.evidence_potential_bound,
            "duration_sigma_bound": self.duration_sigma_bound,
            "coefficient_sigma_bound": self.coefficient_sigma_bound,
            "soft_raster_temperature_fraction": self.soft_raster_temperature_fraction,
            "minimum_remaining_fraction": self.minimum_remaining_fraction,
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in finite_positive.values()):
            raise ValueError("event generator scales must be finite and positive.")
        if not 0.0 < self.presence_threshold < 1.0:
            raise ValueError("presence_threshold must lie strictly between zero and one.")
        if self.architecture != "autoregressive_coordinate_v1":
            raise ValueError("unsupported EventTrack architecture.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EventGeneratorConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class EventLearningConfig:
    """Parent-atomic deterministic EventTrack training budget."""

    epochs: int = 4
    parent_batch_size: int = 2
    training_parent_count: int = 120
    tuning_parent_count: int = 60
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    random_seed: int = 20260806
    dirty_random_identity: int = 2026080602
    log_every_batches: int = 20
    early_stopping_patience: int = 2
    include_peak_poor: bool = True
    self_conditioning_initial_fraction: float = 0.25
    self_conditioning_final_fraction: float = 0.75
    state_loss_weight: float = 0.50
    presence_loss_weight: float = 0.10
    duration_loss_weight: float = 1.00
    profile_loss_weight: float = 1.00
    coefficient_loss_weight: float = 0.25
    reconstruction_loss_weight: float = 1.00
    evidence_consistency_weight: float = 0.25
    state_occupancy_weight: float = 0.25
    lateral_continuity_weight: float = 0.05
    cumulative_boundary_loss_weight: float = 1.00
    renewal_density_loss_weight: float = 1.00

    def __post_init__(self) -> None:
        for name in (
            "epochs",
            "parent_batch_size",
            "training_parent_count",
            "tuning_parent_count",
            "log_every_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative.")
        if not (
            0.0
            <= self.self_conditioning_initial_fraction
            <= self.self_conditioning_final_fraction
            <= 1.0
        ):
            raise ValueError(
                "self-conditioning fractions must satisfy 0 <= initial <= final <= 1."
            )
        for name in (
            "state_loss_weight",
            "presence_loss_weight",
            "duration_loss_weight",
            "profile_loss_weight",
            "coefficient_loss_weight",
            "reconstruction_loss_weight",
            "evidence_consistency_weight",
            "state_occupancy_weight",
            "lateral_continuity_weight",
            "cumulative_boundary_loss_weight",
            "renewal_density_loss_weight",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EventLearningConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class EventEvaluationConfig:
    split: str = "tuning"
    parent_count: int = 60
    parent_batch_size: int = 2
    dirty_random_identity: int = 2026080702
    log_every_batches: int = 20
    maximum_evidence_rmse_degradation: float = 0.02

    def __post_init__(self) -> None:
        if self.split not in {"training", "tuning", "calibration", "section_gate"}:
            raise ValueError("event evaluation split is invalid.")
        for name in ("parent_count", "parent_batch_size", "log_every_batches"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            not np.isfinite(self.maximum_evidence_rmse_degradation)
            or self.maximum_evidence_rmse_degradation < 0.0
        ):
            raise ValueError(
                "maximum_evidence_rmse_degradation must be finite and non-negative."
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EventEvaluationConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class EventPolicyCalibrationConfig:
    """Two-pass calibration of event density and profile dispersion."""

    parent_count: int = 12
    parent_batch_size: int = 2
    density_realization_count: int = 4
    profile_realization_count: int = 8
    final_realization_count: int = 16
    random_identity: int = 2026080704
    lateral_correlation_m: float = 900.0
    structure_sampling_temperature: float = 0.65
    baseline_profile_sampling_temperature: float = 0.5
    density_candidates: tuple[float, ...] = (0.85, 0.90, 0.95, 1.0)
    profile_temperature_candidates: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    target_coverage: float = 0.90
    maximum_evidence_rmse_degradation: float = 0.02
    log_every_batches: int = 4

    def __post_init__(self) -> None:
        for name in (
            "parent_count",
            "parent_batch_size",
            "density_realization_count",
            "profile_realization_count",
            "final_realization_count",
            "log_every_batches",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        positive = (
            self.lateral_correlation_m,
            self.structure_sampling_temperature,
            self.baseline_profile_sampling_temperature,
            *self.density_candidates,
            *self.profile_temperature_candidates,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("event policy calibration values must be finite and positive.")
        if not self.density_candidates or not self.profile_temperature_candidates:
            raise ValueError("event policy calibration candidate sets cannot be empty.")
        if not 0.0 < self.target_coverage < 1.0:
            raise ValueError("target_coverage must lie strictly between zero and one.")
        if (
            not np.isfinite(self.maximum_evidence_rmse_degradation)
            or self.maximum_evidence_rmse_degradation < 0.0
        ):
            raise ValueError(
                "maximum_evidence_rmse_degradation must be finite and non-negative."
            )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "EventPolicyCalibrationConfig":
        parsed = dict(value)
        for name in ("density_candidates", "profile_temperature_candidates"):
            if name in parsed:
                parsed[name] = tuple(float(item) for item in parsed[name])
        return cls(**parsed)


def validate_event_track_order(
    tracks: Sequence[EventTrack],
    *,
    width: int,
    tolerance: float = 1.0e-6,
) -> None:
    """Validate the topology-free part of one ordered section realization."""

    if width <= 0 or not tracks:
        raise InputContractError("an event section requires a positive width and tracks.")
    identities = [track.event_id for track in tracks]
    if len(set(identities)) != len(identities):
        raise InputContractError("event identities must be unique within a zone.")
    for track in tracks:
        if track.presence.size != width:
            raise InputContractError("event track width differs from section width.")
    duration = np.stack([track.duration_fraction for track in tracks], axis=0)
    presence = np.stack([track.presence for track in tracks], axis=0)
    if np.any(duration[~presence] > tolerance):
        raise InputContractError("inactive events must have zero duration.")
    totals = np.sum(duration, axis=0)
    active_trace = np.any(presence, axis=0)
    if np.any(~np.isclose(totals[active_trace], 1.0, rtol=0.0, atol=tolerance)):
        raise InputContractError("active event durations must fill each valid zone trace.")
    if np.any(np.abs(totals[~active_trace]) > tolerance):
        raise InputContractError("inactive zone traces cannot carry event duration.")


def _stable_random_seed(*parts: object) -> int:
    """Derive a stable random stream identity without creating a file-integrity gate."""

    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _coordinate_gaussian_field(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    correlation_m: float,
    seed_parts: Sequence[object],
    mode_count: int = 64,
) -> np.ndarray:
    """Sample one coordinate-stable isotropic Gaussian field with random Fourier modes."""

    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise InputContractError("sampling coordinates must be finite one-dimensional arrays.")
    rng = np.random.default_rng(_stable_random_seed(*seed_parts))
    frequency = rng.normal(
        loc=0.0,
        scale=1.0 / float(correlation_m),
        size=(mode_count, 2),
    )
    cosine_weight = rng.normal(size=mode_count)
    sine_weight = rng.normal(size=mode_count)
    phase = frequency[:, 0, None] * x[None] + frequency[:, 1, None] * y[None]
    field = (
        cosine_weight[:, None] * np.cos(phase)
        + sine_weight[:, None] * np.sin(phase)
    ).sum(axis=0) / math.sqrt(float(mode_count))
    if np.any(~np.isfinite(field)):
        raise InputContractError("spatial random field produced non-finite values.")
    return field


class EventTrackNetwork(nn.Module):
    """Autoregressively unfold prior events at cumulative zone coordinates."""

    def __init__(self, config: EventGeneratorConfig, producer_prior: ProducerPrior) -> None:
        super().__init__()
        self.config = config
        self.zone_ids = tuple(zone.zone_id for zone in producer_prior.zones)
        channels = config.hidden_channels
        layers: list[nn.Module] = [nn.Conv1d(10, channels, kernel_size=5, padding=2), nn.GELU()]
        for _ in range(config.vertical_layers):
            layers.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size=5, padding=2),
                    nn.GroupNorm(1, channels),
                    nn.GELU(),
                ]
            )
        self.vertical = nn.Sequential(*layers)
        self.trace_mixer = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.GELU(),
        )
        self.zone_embedding = nn.Embedding(len(self.zone_ids), channels)
        self.previous_state_embedding = nn.Embedding(4, channels)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.recurrent = nn.GRUCell(channels, channels)
        self.state_head = nn.Linear(channels, 3)
        self.field_core = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.GELU(),
        )
        self.presence_head = nn.Linear(channels, 1)
        self.duration_head = nn.Linear(channels, 1)
        self.coefficient_head = nn.Linear(channels, 3)

        initial = []
        transition = []
        duration_median = []
        duration_sigma = []
        coefficient_median = []
        coefficient_sigma = []
        coefficient_lower = []
        coefficient_upper = []
        for zone in producer_prior.zones:
            initial.append(np.log(zone.initial_probability))
            transition.append(np.log(np.clip(zone.transition_probability, 1.0e-30, None)))
            duration_median.append([state.log_duration.median for state in zone.states])
            duration_sigma.append([state.log_duration.robust_sigma for state in zone.states])
            coefficient_median.append(
                [
                    [state.profile_distributions[name].median for name in ("c0", "c1", "c2")]
                    for state in zone.states
                ]
            )
            coefficient_sigma.append(
                [
                    [state.profile_distributions[name].robust_sigma for name in ("c0", "c1", "c2")]
                    for state in zone.states
                ]
            )
            coefficient_lower.append(
                [
                    [state.profile_distributions[name].lower for name in ("c0", "c1", "c2")]
                    for state in zone.states
                ]
            )
            coefficient_upper.append(
                [
                    [state.profile_distributions[name].upper for name in ("c0", "c1", "c2")]
                    for state in zone.states
                ]
            )
        self.register_buffer("initial_log_probability", torch.tensor(np.asarray(initial), dtype=torch.float32))
        self.register_buffer("transition_log_probability", torch.tensor(np.asarray(transition), dtype=torch.float32))
        self.register_buffer("duration_log_median", torch.tensor(np.asarray(duration_median), dtype=torch.float32))
        self.register_buffer("duration_log_sigma", torch.tensor(np.asarray(duration_sigma), dtype=torch.float32))
        self.register_buffer("coefficient_median", torch.tensor(np.asarray(coefficient_median), dtype=torch.float32))
        self.register_buffer("coefficient_sigma", torch.tensor(np.asarray(coefficient_sigma), dtype=torch.float32))
        self.register_buffer("coefficient_lower", torch.tensor(np.asarray(coefficient_lower), dtype=torch.float32))
        self.register_buffer("coefficient_upper", torch.tensor(np.asarray(coefficient_upper), dtype=torch.float32))

    def zone_index(self, zone_id: str) -> int:
        try:
            return self.zone_ids.index(str(zone_id))
        except ValueError as error:
            raise InputContractError(f"event model has no zone {zone_id!r}.") from error

    def _encode(
        self,
        evidence: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        increment = evidence["increment_mean"]
        support = evidence["support"].bool()
        if increment.ndim != 3 or support.shape != increment.shape:
            raise ValueError("event evidence must have shape [batch, lateral, sample].")
        batch, width, samples = increment.shape
        scale_inc = evidence["increment_scale"].clamp_min(1.0e-4)
        scale_ref = evidence["reflectivity_scale"].clamp_min(1.0e-4)
        coordinate = torch.linspace(-1.0, 1.0, samples, device=increment.device, dtype=increment.dtype)
        coordinate = coordinate.view(1, 1, samples).expand(batch, width, samples)
        fields = torch.stack(
            (
                torch.where(support, increment / scale_inc, torch.zeros_like(increment)),
                torch.where(support, torch.log(scale_inc), torch.zeros_like(increment)),
                torch.where(support, evidence["reflectivity_mean"] / scale_ref, torch.zeros_like(increment)),
                torch.where(support, torch.log(scale_ref), torch.zeros_like(increment)),
                *[torch.where(support, evidence["state_fraction"][..., item], torch.zeros_like(increment)) for item in range(3)],
                support.to(dtype=increment.dtype),
                coordinate,
                torch.where(support, evidence["background"], torch.zeros_like(increment)),
            ),
            dim=2,
        ).reshape(batch * width, 10, samples)
        hidden = self.vertical(fields).reshape(batch, width, -1, samples)
        support_float = support.to(dtype=hidden.dtype).unsqueeze(2)
        trace_feature = (hidden * support_float).sum(dim=-1) / support_float.sum(dim=-1).clamp_min(1.0)
        lateral_m = evidence["lateral_m"]
        lateral_valid = evidence["lateral_valid"].bool()
        distance = torch.abs(lateral_m[:, :, None] - lateral_m[:, None, :])
        weight = torch.exp(-distance / self.config.lateral_distance_scale_m)
        weight = weight * (lateral_valid[:, :, None] & lateral_valid[:, None, :]).to(weight.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        mixed = torch.einsum("bij,bjc->bic", weight, trace_feature)
        trace_feature = self.trace_mixer(torch.cat((trace_feature, mixed), dim=-1))
        global_feature = (trace_feature * lateral_valid[..., None]).sum(dim=1) / lateral_valid.sum(dim=1, keepdim=True).clamp_min(1)
        return hidden, trace_feature, global_feature

    @staticmethod
    def _query_vertical(
        hidden: torch.Tensor,
        coordinate: torch.Tensor,
    ) -> torch.Tensor:
        """Linearly query [batch, lateral, channel, sample] on the full model axis."""

        batch, width, channels, samples = hidden.shape
        if coordinate.shape != (batch, width):
            raise ValueError("event query coordinate must match [batch, lateral].")
        position = coordinate.clamp(0.0, 1.0) * float(samples - 1)
        lower = torch.floor(position).to(torch.long)
        upper = torch.clamp(lower + 1, max=samples - 1)
        fraction = (position - lower.to(position.dtype))[..., None]
        values = hidden.permute(0, 1, 3, 2)
        lower_value = torch.gather(
            values, 2, lower[..., None, None].expand(-1, -1, 1, channels)
        ).squeeze(2)
        upper_value = torch.gather(
            values, 2, upper[..., None, None].expand(-1, -1, 1, channels)
        ).squeeze(2)
        return lower_value + fraction * (upper_value - lower_value)

    def _step(
        self,
        hidden: torch.Tensor,
        trace_feature: torch.Tensor,
        recurrent: torch.Tensor,
        zone_index: torch.Tensor,
        previous_state_feature: torch.Tensor,
        query_coordinate: torch.Tensor,
        zone_coordinate: torch.Tensor,
        remaining: torch.Tensor,
        active: torch.Tensor,
        lateral_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        local = self._query_vertical(hidden, query_coordinate)
        pool_mask = active & lateral_valid
        empty = ~torch.any(pool_mask, dim=1)
        if torch.any(empty):
            pool_mask = pool_mask.clone()
            pool_mask[empty] = lateral_valid[empty]
        pooled = (local * pool_mask[..., None]).sum(dim=1) / pool_mask.sum(dim=1, keepdim=True).clamp_min(1)
        mean_coordinate = (zone_coordinate * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp_min(1)
        mean_remaining = (remaining * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp_min(1)
        recurrent_input = (
            pooled
            + self.zone_embedding(zone_index)
            + previous_state_feature
            + self.coordinate_projection(
                torch.stack((mean_coordinate, mean_remaining), dim=-1)
            )
        )
        recurrent = self.recurrent(recurrent_input, recurrent)
        field_feature = self.field_core(
            torch.cat(
                (
                    recurrent[:, None, :].expand(-1, trace_feature.shape[1], -1),
                    local,
                    trace_feature,
                ),
                dim=-1,
            )
        )
        bound = self.config.evidence_potential_bound
        return recurrent, {
            "state_residual": bound * torch.tanh(self.state_head(recurrent)),
            "presence_logits": self.presence_head(field_feature).squeeze(-1),
            "duration_residual": torch.tanh(self.duration_head(field_feature).squeeze(-1)),
            "coefficient_residual": torch.tanh(self.coefficient_head(field_feature)),
        }

    def forward(
        self,
        evidence: Mapping[str, torch.Tensor],
        zone_index: torch.Tensor,
        *,
        teacher_duration: torch.Tensor,
        teacher_presence: torch.Tensor,
        teacher_state: torch.Tensor,
        self_conditioning_fraction: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Unfold truth-ordered events with deterministic soft self-conditioning."""

        hidden, trace_feature, global_feature = self._encode(evidence)
        batch, events, width = teacher_duration.shape
        if (
            zone_index.shape != (batch,)
            or teacher_presence.shape != teacher_duration.shape
            or teacher_state.shape != (batch, events)
            or width != trace_feature.shape[1]
        ):
            raise ValueError("teacher-forced EventTrack tensors have inconsistent shapes.")
        if not 0.0 <= self_conditioning_fraction <= 1.0:
            raise ValueError("self_conditioning_fraction must lie in [0, 1].")
        top = torch.cumsum(teacher_duration, dim=1) - teacher_duration
        center = top + 0.5 * teacher_duration
        remaining = (1.0 - top).clamp_min(0.0)
        predicted_top = torch.zeros_like(top[:, 0])
        previous_prediction: torch.Tensor | None = None
        recurrent = global_feature
        outputs: dict[str, list[torch.Tensor]] = {
            "state_residual": [],
            "state_logits": [],
            "presence_logits": [],
            "duration_residual": [],
            "coefficient_residual": [],
            "rollout_duration": [],
            "conditioned_coordinate": [],
        }
        lateral_valid = evidence["lateral_valid"].bool()
        zone_start = evidence["zone_start_fraction"]
        zone_stop = evidence["zone_stop_fraction"]
        for event in range(events):
            if event == 0:
                prior_log_probability = self.initial_log_probability[zone_index]
                previous_state_feature = self.previous_state_embedding(
                    torch.full(
                        (batch,), 3, dtype=torch.long, device=zone_index.device
                    )
                )
            else:
                truth_previous = F.one_hot(
                    teacher_state[:, event - 1].clamp_min(0), num_classes=3
                ).to(dtype=hidden.dtype)
                if previous_prediction is None:
                    raise RuntimeError("missing previous event state distribution.")
                previous_probability = (
                    (1.0 - self_conditioning_fraction) * truth_previous
                    + self_conditioning_fraction * previous_prediction
                )
                previous_probability = previous_probability / previous_probability.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1.0e-8)
                previous_state_feature = torch.matmul(
                    previous_probability,
                    self.previous_state_embedding.weight[:3],
                )
                transition_probability = torch.exp(
                    self.transition_log_probability[zone_index]
                )
                prior_log_probability = torch.log(
                    torch.einsum(
                        "bi,bij->bj", previous_probability, transition_probability
                    ).clamp_min(1.0e-30)
                )
            prior_state_probability = torch.softmax(prior_log_probability, dim=-1)
            expected_prior_duration = torch.sum(
                prior_state_probability
                * torch.exp(self.duration_log_median[zone_index]),
                dim=-1,
            )
            predicted_center = torch.minimum(
                predicted_top + 0.5 * expected_prior_duration[:, None],
                torch.ones_like(predicted_top),
            )
            conditioned_center = (
                (1.0 - self_conditioning_fraction) * center[:, event]
                + self_conditioning_fraction * predicted_center
            )
            predicted_remaining = (1.0 - predicted_top).clamp_min(0.0)
            conditioned_remaining = (
                (1.0 - self_conditioning_fraction) * remaining[:, event]
                + self_conditioning_fraction * predicted_remaining
            )
            query_coordinate = zone_start + conditioned_center * (
                zone_stop - zone_start
            )
            recurrent, step = self._step(
                hidden,
                trace_feature,
                recurrent,
                zone_index,
                previous_state_feature,
                query_coordinate,
                conditioned_center,
                conditioned_remaining,
                teacher_presence[:, event],
                lateral_valid,
            )
            state_logits = prior_log_probability + step["state_residual"]
            state_probability = torch.softmax(state_logits, dim=-1)
            hard_state_probability = F.one_hot(
                torch.argmax(state_probability, dim=-1), num_classes=3
            ).to(dtype=state_probability.dtype)
            rollout_state_probability = (
                hard_state_probability
                + state_probability
                - state_probability.detach()
            )
            state_duration = torch.exp(
                self.duration_log_median[zone_index][:, None, :]
                + self.config.duration_sigma_bound
                * self.duration_log_sigma[zone_index][:, None, :]
                * step["duration_residual"][..., None]
            )
            rollout_duration = torch.sum(
                rollout_state_probability[:, None, :] * state_duration,
                dim=-1,
            )
            rollout_duration = torch.where(
                teacher_presence[:, event],
                torch.minimum(rollout_duration, predicted_remaining),
                torch.zeros_like(rollout_duration),
            )
            predicted_top = (predicted_top + rollout_duration).detach()
            previous_prediction = hard_state_probability.detach()
            step["state_logits"] = state_logits
            step["rollout_duration"] = rollout_duration
            step["conditioned_coordinate"] = conditioned_center
            for name in outputs:
                outputs[name].append(step[name])
        return {
            "state_residual": torch.stack(outputs["state_residual"], dim=1),
            "state_logits": torch.stack(outputs["state_logits"], dim=1),
            "presence_logits": torch.stack(outputs["presence_logits"], dim=1),
            "duration_residual": torch.stack(outputs["duration_residual"], dim=1),
            "coefficient_residual": torch.stack(outputs["coefficient_residual"], dim=1),
            "rollout_duration": torch.stack(outputs["rollout_duration"], dim=1),
            "conditioned_coordinate": torch.stack(
                outputs["conditioned_coordinate"], dim=1
            ),
        }

    def duration_fraction(
        self,
        output: Mapping[str, torch.Tensor],
        zone_index: torch.Tensor,
        states: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        events = states.shape[1]
        zone = zone_index[:, None].expand(-1, events)
        valid_state = states.clamp_min(0)
        raw = self.raw_duration_fraction(output, zone_index, states)
        log_weight = torch.where(
            presence,
            torch.log(raw.clamp_min(1.0e-8)),
            torch.full_like(raw, -1.0e9),
        )
        fraction = torch.softmax(log_weight, dim=1) * presence.to(log_weight.dtype)
        return fraction / fraction.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

    def raw_duration_fraction(
        self,
        output: Mapping[str, torch.Tensor],
        zone_index: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        events = states.shape[1]
        zone = zone_index[:, None].expand(-1, events)
        valid_state = states.clamp_min(0)
        median = self.duration_log_median[zone, valid_state][..., None]
        sigma = self.duration_log_sigma[zone, valid_state][..., None]
        log_duration = (
            median
            + self.config.duration_sigma_bound
            * sigma
            * output["duration_residual"][:, :events]
        )
        return torch.exp(log_duration)

    def coefficients(
        self,
        output: Mapping[str, torch.Tensor],
        zone_index: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        events = states.shape[1]
        zone = zone_index[:, None].expand(-1, events)
        valid_state = states.clamp_min(0)
        median = self.coefficient_median[zone, valid_state][..., None, :]
        sigma = self.coefficient_sigma[zone, valid_state][..., None, :]
        lower = self.coefficient_lower[zone, valid_state][..., None, :]
        upper = self.coefficient_upper[zone, valid_state][..., None, :]
        value = median + self.config.coefficient_sigma_bound * sigma * output["coefficient_residual"][:, :events]
        return torch.maximum(lower, torch.minimum(upper, value))

    def rollout(
        self,
        evidence: Mapping[str, torch.Tensor],
        zone_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Deterministically unfold renewal events until every trace is filled."""

        hidden, trace_feature, global_feature = self._encode(evidence)
        batch, width = evidence["lateral_valid"].shape
        if batch != 1:
            raise ValueError("deterministic EventTrack rollout currently accepts one zone.")
        lateral_valid = evidence["lateral_valid"].bool()
        zone_start = evidence["zone_start_fraction"]
        zone_stop = evidence["zone_stop_fraction"]
        cumulative = torch.zeros((batch, width), dtype=hidden.dtype, device=hidden.device)
        remaining = lateral_valid.to(hidden.dtype)
        recurrent = global_feature
        previous_state = torch.full((batch,), 3, dtype=torch.long, device=hidden.device)
        states: list[torch.Tensor] = []
        presences: list[torch.Tensor] = []
        durations: list[torch.Tensor] = []
        coefficients: list[torch.Tensor] = []
        for event in range(self.config.maximum_events):
            active = lateral_valid & (remaining > self.config.minimum_remaining_fraction)
            if not torch.any(active):
                break
            if event == 0:
                next_probability = torch.exp(self.initial_log_probability[zone_index])
            else:
                next_probability = torch.exp(
                    self.transition_log_probability[zone_index, previous_state]
                )
            expected_duration = torch.sum(
                next_probability * torch.exp(self.duration_log_median[zone_index]),
                dim=-1,
            )
            zone_coordinate = torch.minimum(
                cumulative + 0.5 * expected_duration[:, None],
                torch.ones_like(cumulative),
            )
            query_coordinate = zone_start + zone_coordinate * (zone_stop - zone_start)
            recurrent, step = self._step(
                hidden,
                trace_feature,
                recurrent,
                zone_index,
                self.previous_state_embedding(previous_state),
                query_coordinate,
                zone_coordinate,
                remaining,
                active,
                lateral_valid,
            )
            base = (
                self.initial_log_probability[zone_index]
                if event == 0
                else self.transition_log_probability[zone_index, previous_state]
            )
            state = torch.argmax(base + step["state_residual"], dim=-1)
            single_output = {
                "duration_residual": step["duration_residual"][:, None],
                "coefficient_residual": step["coefficient_residual"][:, None],
            }
            state_event = state[:, None]
            raw_duration = self.raw_duration_fraction(
                single_output, zone_index, state_event
            )[:, 0]
            probability = torch.sigmoid(step["presence_logits"])
            presence = active & (probability >= self.config.presence_threshold)
            for batch_index in range(batch):
                presence[batch_index] = _longest_true_run_tensor(
                    presence[batch_index], probability[batch_index], active[batch_index]
                )
            if not torch.any(presence):
                valid_probability = torch.where(active, probability, torch.full_like(probability, -1.0))
                presence[0, torch.argmax(valid_probability[0])] = True
            if event == self.config.maximum_events - 1:
                presence = active
            duration = torch.where(
                presence,
                torch.minimum(raw_duration, remaining),
                torch.zeros_like(raw_duration),
            )
            coefficient = self.coefficients(
                single_output, zone_index, state_event
            )[:, 0]
            states.append(state)
            presences.append(presence)
            durations.append(duration)
            coefficients.append(coefficient)
            cumulative = cumulative + duration
            remaining = torch.where(lateral_valid, (1.0 - cumulative).clamp_min(0.0), torch.zeros_like(cumulative))
            previous_state = state
        if not durations:
            raise InputContractError("EventTrack rollout produced no events.")
        duration_stack = torch.stack(durations, dim=1)
        presence_stack = torch.stack(presences, dim=1)
        unfinished = lateral_valid & (
            torch.sum(duration_stack, dim=1) < 1.0 - self.config.minimum_remaining_fraction
        )
        if torch.any(unfinished):
            duration_stack[:, -1] = duration_stack[:, -1] + torch.where(
                unfinished,
                1.0 - torch.sum(duration_stack, dim=1),
                torch.zeros_like(duration_stack[:, -1]),
            )
            presence_stack[:, -1] |= unfinished
        return {
            "state": torch.stack(states, dim=1),
            "presence": presence_stack,
            "duration": duration_stack,
            "coefficients": torch.stack(coefficients, dim=1),
        }

    def sample_rollouts(
        self,
        evidence: Mapping[str, torch.Tensor],
        zone_index: torch.Tensor,
        *,
        policy: GenerationPolicy,
        evidence_identity: str,
        x_m: np.ndarray,
        y_m: np.ndarray,
    ) -> tuple[dict[str, torch.Tensor], ...]:
        """Sample complete section EventTracks after one shared evidence encoding."""

        hidden, trace_feature, global_feature = self._encode(evidence)
        batch, width = evidence["lateral_valid"].shape
        if batch != 1 or zone_index.shape != (1,):
            raise ValueError("EventTrack ensemble sampling currently accepts one zone.")
        if np.asarray(x_m).shape != (width,) or np.asarray(y_m).shape != (width,):
            raise InputContractError("sampling coordinates must match the section width.")
        lateral_valid = evidence["lateral_valid"].bool()
        sample_capacity = evidence["highres_sample_capacity"]
        if sample_capacity.shape != lateral_valid.shape:
            raise InputContractError(
                "high-resolution sample capacity must match lateral validity."
            )
        if torch.any(lateral_valid & (sample_capacity < 1)):
            raise InputContractError(
                "every valid zone trace requires high-resolution sample capacity."
            )
        minimum_duration = torch.reciprocal(
            sample_capacity.clamp_min(1).to(hidden.dtype)
        )
        zone_start = evidence["zone_start_fraction"]
        zone_stop = evidence["zone_stop_fraction"]
        structure_temperature = float(policy.structure_sampling_temperature)
        profile_temperature = float(policy.profile_sampling_temperature)
        density = float(policy.event_density_multiplier)
        results: list[dict[str, torch.Tensor]] = []
        for member in range(policy.realization_count):
            cumulative = torch.zeros(
                (batch, width), dtype=hidden.dtype, device=hidden.device
            )
            remaining = lateral_valid.to(hidden.dtype)
            recurrent = global_feature.clone()
            previous_state = torch.full(
                (batch,), 3, dtype=torch.long, device=hidden.device
            )
            states: list[torch.Tensor] = []
            presences: list[torch.Tensor] = []
            durations: list[torch.Tensor] = []
            coefficients: list[torch.Tensor] = []
            for event in range(self.config.maximum_events):
                active = lateral_valid & (
                    remaining > self.config.minimum_remaining_fraction
                )
                if not torch.any(active):
                    break
                base = (
                    self.initial_log_probability[zone_index]
                    if event == 0
                    else self.transition_log_probability[zone_index, previous_state]
                )
                next_probability = torch.softmax(base, dim=-1)
                expected_duration = torch.sum(
                    next_probability
                    * torch.exp(self.duration_log_median[zone_index]),
                    dim=-1,
                ) / density
                zone_coordinate = torch.minimum(
                    cumulative + 0.5 * expected_duration[:, None],
                    torch.ones_like(cumulative),
                )
                query_coordinate = zone_start + zone_coordinate * (
                    zone_stop - zone_start
                )
                recurrent, step = self._step(
                    hidden,
                    trace_feature,
                    recurrent,
                    zone_index,
                    self.previous_state_embedding(previous_state),
                    query_coordinate,
                    zone_coordinate,
                    remaining,
                    active,
                    lateral_valid,
                )

                state_rng = np.random.default_rng(
                    _stable_random_seed(
                        policy.random_identity,
                        evidence_identity,
                        member,
                        event,
                        "state",
                    )
                )
                gumbel = torch.as_tensor(
                    state_rng.gumbel(size=3),
                    dtype=hidden.dtype,
                    device=hidden.device,
                )[None]
                state = torch.argmax(
                    (base + step["state_residual"]) / structure_temperature
                    + gumbel,
                    dim=-1,
                )
                state_event = state[:, None]
                single_output = {
                    "duration_residual": step["duration_residual"][:, None],
                    "coefficient_residual": step["coefficient_residual"][:, None],
                }

                presence_probability = torch.sigmoid(
                    step["presence_logits"] / structure_temperature
                )
                presence_gaussian = _coordinate_gaussian_field(
                    x_m,
                    y_m,
                    correlation_m=policy.lateral_correlation_m,
                    seed_parts=(
                        policy.random_identity,
                        evidence_identity,
                        member,
                        event,
                        "presence",
                    ),
                )
                presence_uniform = 0.5 * (
                    1.0
                    + torch.erf(
                        torch.as_tensor(
                            presence_gaussian,
                            dtype=hidden.dtype,
                            device=hidden.device,
                        )
                        / math.sqrt(2.0)
                    )
                )
                presence = active & (
                    presence_uniform[None] <= presence_probability
                )
                presence[0] = _longest_true_run_tensor(
                    presence[0], presence_probability[0], active[0]
                )
                if event == self.config.maximum_events - 1:
                    presence = active

                raw_duration = self.raw_duration_fraction(
                    single_output, zone_index, state_event
                )[:, 0]
                duration_sigma = self.duration_log_sigma[zone_index, state][:, None]
                duration_gaussian = _coordinate_gaussian_field(
                    x_m,
                    y_m,
                    correlation_m=policy.lateral_correlation_m,
                    seed_parts=(
                        policy.random_identity,
                        evidence_identity,
                        member,
                        event,
                        "duration",
                    ),
                )
                duration_noise = torch.as_tensor(
                    duration_gaussian,
                    dtype=hidden.dtype,
                    device=hidden.device,
                )[None]
                duration_log_scale = structure_temperature * duration_sigma
                sampled_duration = (
                    raw_duration
                    * torch.exp(
                        duration_log_scale * duration_noise
                        - 0.5 * duration_log_scale.square()
                    )
                    / density
                )
                sampled_duration = torch.maximum(
                    sampled_duration,
                    minimum_duration,
                )
                duration = torch.where(
                    presence,
                    torch.minimum(sampled_duration, remaining),
                    torch.zeros_like(sampled_duration),
                )

                coefficient = self.coefficients(
                    single_output, zone_index, state_event
                )[:, 0]
                coefficient_sigma = self.coefficient_sigma[zone_index, state][:, None, :]
                coefficient_noise = torch.stack(
                    tuple(
                        torch.as_tensor(
                            _coordinate_gaussian_field(
                                x_m,
                                y_m,
                                correlation_m=policy.lateral_correlation_m,
                                seed_parts=(
                                    policy.random_identity,
                                    evidence_identity,
                                    member,
                                    event,
                                    f"coefficient_{coefficient_index}",
                                ),
                            ),
                            dtype=hidden.dtype,
                            device=hidden.device,
                        )
                        for coefficient_index in range(3)
                    ),
                    dim=-1,
                )[None]
                coefficient = coefficient + (
                    profile_temperature * coefficient_sigma * coefficient_noise
                )
                lower = self.coefficient_lower[zone_index, state][:, None, :]
                upper = self.coefficient_upper[zone_index, state][:, None, :]
                coefficient = torch.maximum(lower, torch.minimum(upper, coefficient))

                states.append(state)
                presences.append(presence)
                durations.append(duration)
                coefficients.append(coefficient)
                cumulative = cumulative + duration
                remaining = torch.where(
                    lateral_valid,
                    (1.0 - cumulative).clamp_min(0.0),
                    torch.zeros_like(cumulative),
                )
                previous_state = state
            if not durations:
                raise InputContractError("EventTrack ensemble member produced no events.")
            duration_stack = torch.stack(durations, dim=1)
            presence_stack = torch.stack(presences, dim=1)
            unfinished = lateral_valid & (
                torch.sum(duration_stack, dim=1)
                < 1.0 - self.config.minimum_remaining_fraction
            )
            if torch.any(unfinished):
                duration_stack[:, -1] = duration_stack[:, -1] + torch.where(
                    unfinished,
                    1.0 - torch.sum(duration_stack, dim=1),
                    torch.zeros_like(duration_stack[:, -1]),
                )
                presence_stack[:, -1] |= unfinished
            results.append(
                {
                    "state": torch.stack(states, dim=1),
                    "presence": presence_stack,
                    "duration": duration_stack,
                    "coefficients": torch.stack(coefficients, dim=1),
                }
            )
        return tuple(results)


def _evidence_tensor_batch(
    evidences: Sequence[BandlimitedEvidence],
    observations: Sequence[ObservationTile],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not evidences or len(evidences) != len(observations):
        raise ValueError("event evidence batch cannot be empty or mismatched.")
    zone_start_fraction: list[np.ndarray] = []
    zone_stop_fraction: list[np.ndarray] = []
    highres_sample_capacity: list[np.ndarray] = []
    for evidence, observation in zip(evidences, observations, strict=True):
        axis = np.asarray(evidence.model_axis.coordinates, dtype=np.float64)
        axis_span = float(axis[-1] - axis[0])
        if not np.isfinite(axis_span) or axis_span <= 0.0:
            raise InputContractError("model axis must be finite and strictly increasing.")
        start = (observation.zone_top - axis[0]) / axis_span
        stop = (observation.zone_bottom - axis[0]) / axis_span
        valid = observation.lateral_valid
        if np.any(valid & ((start < -1.0e-6) | (stop > 1.0 + 1.0e-6))):
            raise InputContractError("zone extent lies outside the model axis.")
        zone_start_fraction.append(np.where(valid, np.clip(start, 0.0, 1.0), 0.0))
        zone_stop_fraction.append(np.where(valid, np.clip(stop, 0.0, 1.0), 0.0))
        capacity = np.count_nonzero(evidence.highres_support, axis=1).astype(
            np.int64
        )
        if np.any(valid & (capacity < 1)):
            raise InputContractError(
                "valid zone trace has no high-resolution sample capacity."
            )
        highres_sample_capacity.append(capacity)
    return {
        "increment_mean": torch.tensor(np.stack([item.projected_log_ai_increment_mean for item in evidences]), dtype=torch.float32, device=device),
        "increment_scale": torch.tensor(np.stack([item.projected_log_ai_increment_scale for item in evidences]), dtype=torch.float32, device=device),
        "reflectivity_mean": torch.tensor(np.stack([item.signed_reflectivity_mean for item in evidences]), dtype=torch.float32, device=device),
        "reflectivity_scale": torch.tensor(np.stack([item.signed_reflectivity_scale for item in evidences]), dtype=torch.float32, device=device),
        "state_fraction": torch.tensor(np.stack([item.state_fraction for item in evidences]), dtype=torch.float32, device=device),
        "background": torch.tensor(np.stack([item.background_lfm_linear for item in evidences]), dtype=torch.float32, device=device),
        "support": torch.tensor(np.stack([item.support for item in evidences]), dtype=torch.bool, device=device),
        "lateral_m": torch.tensor(np.stack([item.lateral_m for item in evidences]), dtype=torch.float32, device=device),
        "lateral_valid": torch.tensor(np.stack([item.lateral_valid for item in observations]), dtype=torch.bool, device=device),
        "zone_start_fraction": torch.tensor(np.stack(zone_start_fraction), dtype=torch.float32, device=device),
        "zone_stop_fraction": torch.tensor(np.stack(zone_stop_fraction), dtype=torch.float32, device=device),
        "highres_sample_capacity": torch.tensor(np.stack(highres_sample_capacity), dtype=torch.int64, device=device),
    }


def lfm_relative_truth_tracks(
    tile: StructuredTrainingTile,
    evidence: BandlimitedEvidence,
) -> tuple[EventTrack, ...]:
    """Rebase producer events exactly onto the discrete zone-linear LFM anchor."""

    results: list[EventTrack] = []
    for truth in tile.event_tracks:
        duration = np.zeros(tile.observation.width, dtype=np.float64)
        coefficients = np.full((tile.observation.width, 3), np.nan, dtype=np.float64)
        for trace in np.flatnonzero(truth.presence):
            event_mask = (
                tile.highres_zone_support[trace]
                & (tile.object_id_highres[trace] == truth.event_id)
            )
            zone_count = int(np.count_nonzero(tile.highres_zone_support[trace]))
            event_count_samples = int(np.count_nonzero(event_mask))
            if zone_count <= 0 or event_count_samples <= 0:
                raise InputContractError("present producer event has no high-resolution samples.")
            indices = np.flatnonzero(event_mask)
            if np.any(np.diff(indices) != 1):
                raise InputContractError("producer event samples must be contiguous.")
            duration[trace] = event_count_samples / zone_count
            profile = (
                tile.log_ai_highres[trace, event_mask]
                - evidence.background_lfm_linear_highres[trace, event_mask]
            )
            coefficients[trace] = fit_profile_coefficients(profile)
        results.append(
            EventTrack(
                event_id=str(truth.event_id),
                state_id=truth.state_id,
                presence=truth.presence,
                duration_fraction=duration,
                coefficients=coefficients,
            )
        )
    validate_event_track_order(results, width=tile.observation.width)
    return tuple(results)


def _event_target_batch(
    tiles: Sequence[StructuredTrainingTile],
    evidences: Sequence[BandlimitedEvidence],
    model: EventTrackNetwork,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = len(tiles)
    maximum = model.config.maximum_events
    width = tiles[0].observation.width
    samples = tiles[0].observation.model_axis.coordinates.size
    counts = np.zeros(batch, dtype=np.int64)
    states = np.full((batch, maximum), -1, dtype=np.int64)
    presence = np.zeros((batch, maximum, width), dtype=bool)
    duration = np.zeros((batch, maximum, width), dtype=np.float32)
    coefficients = np.zeros((batch, maximum, width, 3), dtype=np.float32)
    coefficient_valid = np.zeros((batch, maximum, width), dtype=bool)
    target_increment = np.zeros((batch, width, samples), dtype=np.float32)
    target_state = np.zeros((batch, width, samples, 3), dtype=np.float32)
    target_support = np.zeros((batch, width, samples), dtype=bool)
    normalized_coordinate = np.zeros((batch, width, samples), dtype=np.float32)
    zone_index = np.zeros(batch, dtype=np.int64)
    axis = tiles[0].observation.model_axis.coordinates
    for batch_index, (tile, evidence) in enumerate(zip(tiles, evidences, strict=True)):
        event_count = len(tile.event_tracks)
        if event_count > maximum:
            raise InputContractError(
                f"zone {tile.event_tracks[0].zone_id!r} has {event_count} events; maximum_events={maximum}."
            )
        counts[batch_index] = event_count
        zone_index[batch_index] = model.zone_index(tile.event_tracks[0].zone_id)
        rebased_tracks = lfm_relative_truth_tracks(tile, evidence)
        for event_index, (truth, track) in enumerate(
            zip(tile.event_tracks, rebased_tracks, strict=True)
        ):
            states[batch_index, event_index] = track.state_id
            presence[batch_index, event_index] = track.presence
            duration[batch_index, event_index] = track.duration_fraction
            coefficients[batch_index, event_index] = np.nan_to_num(track.coefficients)
            for trace in np.flatnonzero(track.presence):
                event_count_samples = int(
                    np.count_nonzero(
                        tile.highres_zone_support[trace]
                        & (tile.object_id_highres[trace] == truth.event_id)
                    )
                )
                coefficient_valid[batch_index, event_index, trace] = (
                    truth.segment_supervision_valid[trace] and event_count_samples >= 3
                )
        support = tile.model_zone_support & evidence.support
        target_support[batch_index] = support
        target_increment[batch_index] = np.where(
            support,
            tile.model_log_ai - evidence.background_lfm_linear,
            0.0,
        )
        target_state[batch_index] = np.where(
            support[..., None], tile.state_fraction_model, 0.0
        )
        for trace in range(width):
            if not tile.observation.lateral_valid[trace]:
                continue
            top = tile.observation.zone_top[trace]
            bottom = tile.observation.zone_bottom[trace]
            normalized_coordinate[batch_index, trace] = (axis - top) / (bottom - top)
    return {
        "count": torch.tensor(counts, dtype=torch.long, device=device),
        "state": torch.tensor(states, dtype=torch.long, device=device),
        "presence": torch.tensor(presence, dtype=torch.bool, device=device),
        "duration": torch.tensor(duration, dtype=torch.float32, device=device),
        "coefficients": torch.tensor(coefficients, dtype=torch.float32, device=device),
        "coefficient_valid": torch.tensor(coefficient_valid, dtype=torch.bool, device=device),
        "target_increment": torch.tensor(target_increment, dtype=torch.float32, device=device),
        "target_state": torch.tensor(target_state, dtype=torch.float32, device=device),
        "target_support": torch.tensor(target_support, dtype=torch.bool, device=device),
        "normalized_coordinate": torch.tensor(normalized_coordinate, dtype=torch.float32, device=device),
        "zone_index": torch.tensor(zone_index, dtype=torch.long, device=device),
    }


def _soft_rasterize(
    duration: torch.Tensor,
    coefficients: torch.Tensor,
    states: torch.Tensor,
    presence: torch.Tensor,
    normalized_coordinate: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    top = torch.cumsum(duration, dim=1) - duration
    bottom = torch.cumsum(duration, dim=1)
    coordinate = normalized_coordinate[:, None]
    left = torch.sigmoid((coordinate - top[..., None]) / temperature)
    right = torch.sigmoid((bottom[..., None] - coordinate) / temperature)
    membership = left * right * presence[..., None].to(left.dtype)
    membership = membership / membership.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    xi = ((coordinate - top[..., None]) / duration[..., None].clamp_min(1.0e-6)).clamp(0.0, 1.0)
    profile = (
        coefficients[..., 0, None]
        + coefficients[..., 1, None] * (2.0 * xi - 1.0)
        + coefficients[..., 2, None] * torch.sin(torch.pi * xi)
    )
    increment = torch.sum(membership * profile, dim=1)
    state_one_hot = F.one_hot(states.clamp_min(0), num_classes=3).to(membership.dtype)
    occupancy = torch.einsum("bewn,bec->bwnc", membership, state_one_hot)
    return increment, occupancy


def event_track_loss(
    model: EventTrackNetwork,
    output: Mapping[str, torch.Tensor],
    evidence: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    config: EventLearningConfig,
) -> dict[str, torch.Tensor]:
    events = int(output["state_residual"].shape[1])
    slot = torch.arange(events, device=target["count"].device)[None]
    event_valid = slot < target["count"][:, None]
    states = target["state"][:, :events]
    presence_target = target["presence"][:, :events]
    duration_target = target["duration"][:, :events]
    coefficient_target = target["coefficients"][:, :events]
    coefficient_valid_target = target["coefficient_valid"][:, :events]
    state_logits = output["state_logits"][:, :events]
    state_loss = F.cross_entropy(state_logits[event_valid], states[event_valid])
    lateral_valid = evidence["lateral_valid"][:, None, :]
    presence_valid = event_valid[..., None] & lateral_valid
    presence_loss = F.binary_cross_entropy_with_logits(
        output["presence_logits"][presence_valid],
        presence_target[presence_valid].to(output["presence_logits"].dtype),
    )
    duration = model.duration_fraction(
        output, target["zone_index"], states, presence_target
    )
    raw_duration = model.raw_duration_fraction(
        output, target["zone_index"], states
    )
    duration_valid = presence_target & presence_valid
    duration_loss = F.smooth_l1_loss(
        torch.log(raw_duration[duration_valid].clamp_min(1.0e-6)),
        torch.log(duration_target[duration_valid].clamp_min(1.0e-6)),
    )
    rollout_duration = output["rollout_duration"][:, :events]
    predicted_bottom = torch.cumsum(rollout_duration, dim=1)
    target_bottom = torch.cumsum(duration_target, dim=1)
    cumulative_boundary_loss = F.l1_loss(
        predicted_bottom[duration_valid], target_bottom[duration_valid]
    )
    predicted_fill = torch.sum(rollout_duration, dim=1)
    trace_valid = evidence["lateral_valid"].bool()
    renewal_density_loss = F.l1_loss(
        predicted_fill[trace_valid], torch.ones_like(predicted_fill[trace_valid])
    )
    coefficients = model.coefficients(
        output, target["zone_index"], states
    )
    xi = torch.linspace(0.0, 1.0, 9, device=coefficients.device, dtype=coefficients.dtype)
    basis = torch.stack((torch.ones_like(xi), 2.0 * xi - 1.0, torch.sin(torch.pi * xi)), dim=-1)
    predicted_profile = torch.einsum("kq,bewq->bewk", basis, coefficients)
    target_profile = torch.einsum("kq,bewq->bewk", basis, coefficient_target)
    profile_loss = F.smooth_l1_loss(
        predicted_profile[duration_valid], target_profile[duration_valid]
    )
    coefficient_valid = coefficient_valid_target & duration_valid
    coefficient_scale = model.coefficient_sigma[
        target["zone_index"][:, None].expand(-1, events),
        states.clamp_min(0),
    ][..., None, :]
    coefficient_loss = F.smooth_l1_loss(
        ((coefficients - coefficient_target) / coefficient_scale.clamp_min(1.0e-4))[coefficient_valid],
        torch.zeros_like(coefficients[coefficient_valid]),
    )
    raster_increment, raster_state = _soft_rasterize(
        duration,
        coefficients,
        states,
        presence_target,
        target["normalized_coordinate"],
        temperature=model.config.soft_raster_temperature_fraction,
    )
    support = target["target_support"]
    reconstruction_loss = F.smooth_l1_loss(
        (raster_increment - target["target_increment"])[support] / 0.1,
        torch.zeros_like(raster_increment[support]),
    )
    evidence_scale = evidence["increment_scale"].clamp_min(0.01)
    evidence_consistency = F.smooth_l1_loss(
        ((raster_increment - evidence["increment_mean"]) / evidence_scale)[support],
        torch.zeros_like(raster_increment[support]),
    )
    state_occupancy = F.mse_loss(raster_state[support], target["target_state"][support])
    if duration.shape[-1] > 1:
        duration_jump = torch.abs(duration[:, :, 1:] - duration[:, :, :-1])
        coefficient_jump = torch.abs(coefficients[:, :, 1:] - coefficients[:, :, :-1]).mean(dim=-1)
        pair_valid = presence_target[:, :, 1:] & presence_target[:, :, :-1]
        lateral_continuity = (duration_jump[pair_valid].mean() + coefficient_jump[pair_valid].mean())
    else:
        lateral_continuity = duration.new_zeros(())
    total = (
        config.state_loss_weight * state_loss
        + config.presence_loss_weight * presence_loss
        + config.duration_loss_weight * duration_loss
        + config.profile_loss_weight * profile_loss
        + config.coefficient_loss_weight * coefficient_loss
        + config.reconstruction_loss_weight * reconstruction_loss
        + config.evidence_consistency_weight * evidence_consistency
        + config.state_occupancy_weight * state_occupancy
        + config.lateral_continuity_weight * lateral_continuity
        + config.cumulative_boundary_loss_weight * cumulative_boundary_loss
        + config.renewal_density_loss_weight * renewal_density_loss
    )
    return {
        "loss": total,
        "state_cross_entropy": state_loss,
        "presence_binary_cross_entropy": presence_loss,
        "duration_log_huber": duration_loss,
        "profile_huber": profile_loss,
        "coefficient_huber": coefficient_loss,
        "model_grid_reconstruction_huber": reconstruction_loss,
        "evidence_consistency_huber": evidence_consistency,
        "state_occupancy_mse": state_occupancy,
        "lateral_continuity": lateral_continuity,
        "cumulative_boundary_l1": cumulative_boundary_loss,
        "renewal_fill_l1": renewal_density_loss,
    }


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    parsed = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(parsed)
    if indices.size == 0:
        return parsed
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split)
    selected = max(groups, key=lambda item: (item.size, -int(item[0])))
    result = np.zeros_like(parsed)
    result[selected] = True
    return result


def _longest_true_run_tensor(
    mask: torch.Tensor,
    probability: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Keep one laterally contiguous event extent during deterministic rollout."""

    parsed = mask.detach().cpu().numpy().astype(bool)
    selected = _longest_true_run(parsed)
    if not np.any(selected) and torch.any(active):
        valid_probability = torch.where(
            active, probability, torch.full_like(probability, -1.0)
        )
        selected[int(torch.argmax(valid_probability).item())] = True
    return torch.tensor(selected, dtype=torch.bool, device=mask.device)


def _hard_event_fields(
    model: EventTrackNetwork,
    evidence: Mapping[str, torch.Tensor],
    zone_index: int,
    lateral_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = evidence["increment_mean"].device
    rollout = model.rollout(
        evidence,
        torch.tensor([zone_index], dtype=torch.long, device=device),
    )
    return _numpy_event_fields(rollout, lateral_valid)


def _numpy_event_fields(
    rollout: Mapping[str, torch.Tensor],
    lateral_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize one rollout into the public ordered EventTrack field convention."""

    states = rollout["state"][0].detach().cpu().numpy().astype(np.int64)
    presence = rollout["presence"][0].detach().cpu().numpy().astype(bool)
    presence &= np.asarray(lateral_valid, dtype=bool)[None]
    duration = rollout["duration"][0].detach().cpu().numpy().astype(np.float64)
    duration[~presence] = 0.0
    totals = np.sum(duration, axis=0)
    valid = np.asarray(lateral_valid, dtype=bool)
    if np.any(valid & (~np.isfinite(totals) | (totals <= 0.0))):
        raise InputContractError("EventTrack rollout left a valid trace without duration.")
    duration[:, valid] /= totals[valid][None]
    coefficients = rollout["coefficients"][0].detach().cpu().numpy().astype(np.float64)
    coefficients[~presence] = np.nan
    return states, presence, duration, coefficients


def rasterize_event_tracks(
    evidence: BandlimitedEvidence,
    tracks: Sequence[EventTrack],
    *,
    identity: str | None = None,
) -> EventTrackRealization:
    """Rasterize one complete ordered track system through the shared decoder."""

    validate_event_track_order(tracks, width=evidence.lateral_m.size)
    segments: list[Segment] = []
    highres_support = np.asarray(evidence.highres_support, dtype=bool)
    for trace in range(highres_support.shape[0]):
        indices = np.flatnonzero(highres_support[trace])
        if indices.size == 0:
            continue
        if np.any(np.diff(indices) != 1):
            raise InputContractError("high-resolution zone support must be contiguous.")
        present = [track for track in tracks if track.presence[trace]]
        if not present:
            raise InputContractError("valid zone trace has no present events.")
        if len(present) > indices.size:
            raise InputContractError(
                f"{evidence.identity}: trace {trace} has {len(present)} events "
                f"for {indices.size} high-resolution zone samples."
            )
        weights = np.asarray([track.duration_fraction[trace] for track in present], dtype=np.float64)
        raw = weights / np.sum(weights) * indices.size
        counts = np.maximum(1, np.floor(raw).astype(np.int64))
        difference = int(indices.size - np.sum(counts))
        remainder = raw - np.floor(raw)
        if difference > 0:
            for item in np.argsort(-remainder, kind="stable")[:difference]:
                counts[item] += 1
        elif difference < 0:
            for item in np.argsort(remainder, kind="stable"):
                take = min(counts[item] - 1, -difference)
                counts[item] -= take
                difference += take
                if difference == 0:
                    break
        if np.sum(counts) != indices.size:
            raise InputContractError("integer event allocation cannot fill zone support.")
        start = int(indices[0])
        for track, sample_count in zip(present, counts, strict=True):
            stop = start + int(sample_count)
            c0, c1, c2 = track.coefficients[trace]
            segments.append(Segment(trace, track.state_id, start, stop, c0, c1, c2))
            start = stop
    log_ai_highres, state_highres = decode_segments_numpy(
        evidence.background_lfm_linear_highres, segments
    )
    model_log_ai, model_support = project_supported_highres_to_model(
        log_ai_highres,
        highres_support,
        highres_interval=evidence.highres_axis.sample_interval,
        model_interval=evidence.model_axis.sample_interval,
    )
    return EventTrackRealization(
        zone_id=evidence.identity.rsplit(":", 1)[-1],
        tracks=tuple(tracks),
        log_ai_highres=log_ai_highres,
        state_highres=state_highres,
        model_log_ai=model_log_ai,
        model_support=model_support,
        identity=identity or f"{evidence.identity}:map",
    )


def realize_event_tracks(
    model: EventTrackNetwork,
    producer_prior: ProducerPrior,
    evidence: BandlimitedEvidence,
    observation: ObservationTile,
) -> EventTrackRealization:
    """Create one deterministic joint EventTrack realization from evidence."""

    del producer_prior  # Prior tensors are frozen inside the model checkpoint.
    device = next(model.parameters()).device
    model.eval()
    zone_id = evidence.identity.rsplit(":", 1)[-1]
    zone_index = model.zone_index(zone_id)
    evidence_batch = _evidence_tensor_batch([evidence], [observation], device=device)
    with torch.no_grad():
        states, presence, duration, coefficients = _hard_event_fields(
            model, evidence_batch, zone_index, observation.lateral_valid
        )
    tracks = tuple(
        EventTrack(
            event_id=f"{zone_id}:event_{event:03d}",
            state_id=int(states[event]),
            presence=presence[event],
            duration_fraction=duration[event],
            coefficients=coefficients[event],
        )
        for event in range(states.size)
    )
    return rasterize_event_tracks(evidence, tracks)


def sample_event_track_realizations(
    model: EventTrackNetwork,
    producer_prior: ProducerPrior,
    evidence: BandlimitedEvidence,
    observation: ObservationTile,
    policy: GenerationPolicy,
) -> tuple[EventTrackRealization, ...]:
    """Sample complete, reproducible section members from one evidence encoding."""

    del producer_prior  # Frozen prior tensors are part of the fitted EventTrack model.
    device = next(model.parameters()).device
    model.eval()
    zone_id = evidence.identity.rsplit(":", 1)[-1]
    zone_index = model.zone_index(zone_id)
    evidence_batch = _evidence_tensor_batch([evidence], [observation], device=device)
    x_m = evidence.x_m if evidence.x_m is not None else evidence.lateral_m
    y_m = evidence.y_m if evidence.y_m is not None else np.zeros_like(evidence.lateral_m)
    with torch.no_grad():
        sampled = model.sample_rollouts(
            evidence_batch,
            torch.tensor([zone_index], dtype=torch.long, device=device),
            policy=policy,
            evidence_identity=evidence.identity,
            x_m=np.asarray(x_m, dtype=np.float64),
            y_m=np.asarray(y_m, dtype=np.float64),
        )
    results: list[EventTrackRealization] = []
    for member, rollout in enumerate(sampled):
        states, presence, duration, coefficients = _numpy_event_fields(
            rollout, observation.lateral_valid
        )
        tracks = tuple(
            EventTrack(
                event_id=f"{zone_id}:member_{member:03d}:event_{event:03d}",
                state_id=int(states[event]),
                presence=presence[event],
                duration_fraction=duration[event],
                coefficients=coefficients[event],
            )
            for event in range(states.size)
        )
        results.append(
            rasterize_event_tracks(
                evidence,
                tracks,
                identity=(
                    f"{evidence.identity}:member:{policy.random_identity}:{member:03d}"
                ),
            )
        )
    return tuple(results)


def realize_prior_event_tracks(
    model: EventTrackNetwork,
    evidence: BandlimitedEvidence,
    observation: ObservationTile,
) -> EventTrackRealization:
    """Create a deterministic ProducerPrior-only renewal baseline."""

    zone_id = evidence.identity.rsplit(":", 1)[-1]
    zone_index = model.zone_index(zone_id)
    valid = np.asarray(observation.lateral_valid, dtype=bool)
    if not np.any(valid):
        raise InputContractError("prior EventTrack baseline requires a valid lateral trace.")
    initial = model.initial_log_probability[zone_index].detach().cpu().numpy()
    transition = model.transition_log_probability[zone_index].detach().cpu().numpy()
    duration_median = np.exp(
        model.duration_log_median[zone_index].detach().cpu().numpy()
    )
    coefficient_median = model.coefficient_median[zone_index].detach().cpu().numpy()
    highres_capacity = np.count_nonzero(evidence.highres_support, axis=1)
    maximum_events = min(
        model.config.maximum_events,
        int(np.min(highres_capacity[valid])),
    )
    if maximum_events <= 0:
        raise InputContractError("ProducerPrior baseline has no high-resolution capacity.")
    remaining = 1.0
    previous_state: int | None = None
    tracks: list[EventTrack] = []
    for event_index in range(maximum_events):
        state = int(
            np.argmax(initial if previous_state is None else transition[previous_state])
        )
        fraction = min(float(duration_median[state]), remaining)
        if event_index == maximum_events - 1:
            fraction = remaining
        presence = valid.copy()
        duration = np.where(valid, fraction, 0.0)
        coefficients = np.full((valid.size, 3), np.nan, dtype=np.float64)
        coefficients[valid] = coefficient_median[state]
        tracks.append(
            EventTrack(
                event_id=f"{zone_id}:prior_{event_index:03d}",
                state_id=state,
                presence=presence,
                duration_fraction=duration,
                coefficients=coefficients,
            )
        )
        remaining -= fraction
        previous_state = state
        if remaining <= model.config.minimum_remaining_fraction:
            break
    if remaining > model.config.minimum_remaining_fraction:
        raise InputContractError("ProducerPrior baseline exceeded maximum_events.")
    if remaining != 0.0:
        last = tracks[-1]
        corrected = np.asarray(last.duration_fraction, dtype=np.float64).copy()
        corrected[valid] += remaining
        tracks[-1] = EventTrack(
            event_id=last.event_id,
            state_id=last.state_id,
            presence=last.presence,
            duration_fraction=corrected,
            coefficients=last.coefficients,
        )
    return rasterize_event_tracks(evidence, tracks)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".staging")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def _select_parents(corpus: Corpus, split: str, count: int) -> tuple[str, ...]:
    available = corpus.splits[split]
    if count > len(available):
        raise ValueError(f"requested {count} {split} parents, but only {len(available)} exist.")
    # The canonical index is already family-stratified; evenly spaced selection avoids prefix bias.
    indices = np.linspace(0, len(available) - 1, count, dtype=np.int64)
    return tuple(available[int(item)] for item in indices)


def _preflight_section_evaluation(
    corpus: Corpus,
    parent_ids: Sequence[str],
) -> dict[str, int]:
    """Validate all static section contracts before ensemble generation starts."""

    tile_count = 0
    valid_trace_count = 0
    singleton_anchor_count = 0
    minimum_highres_capacity: int | None = None
    for parent_id in parent_ids:
        for tile in parent_training_tiles(corpus, parent_id):
            observation = tile.observation
            anchor = build_lfm_anchor(observation)
            tile_count += 1
            model_axis = np.asarray(
                observation.model_axis.coordinates,
                dtype=np.float64,
            )
            for trace in np.flatnonzero(observation.lateral_valid):
                inside = (
                    observation.observed_valid[trace]
                    & (model_axis >= observation.zone_top[trace])
                    & (model_axis <= observation.zone_bottom[trace])
                )
                if np.count_nonzero(inside) == 1:
                    singleton_anchor_count += 1
                capacity = int(np.count_nonzero(anchor.highres_support[trace]))
                if capacity < 1:
                    raise InputContractError(
                        f"{observation.identity}: trace {trace} has no "
                        "high-resolution generation capacity."
                    )
                minimum_highres_capacity = (
                    capacity
                    if minimum_highres_capacity is None
                    else min(minimum_highres_capacity, capacity)
                )
                valid_trace_count += 1
    if tile_count == 0 or valid_trace_count == 0 or minimum_highres_capacity is None:
        raise InputContractError("section evaluation preflight found no valid tiles.")
    return {
        "parent_count": len(parent_ids),
        "tile_count": tile_count,
        "valid_trace_count": valid_trace_count,
        "singleton_anchor_count": singleton_anchor_count,
        "minimum_highres_sample_capacity": minimum_highres_capacity,
    }


def _parent_chunks(parent_ids: Sequence[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(parent_ids[start : start + size]) for start in range(0, len(parent_ids), size)]


def _load_evidence_generator(
    checkpoint_path: str | Path,
    corpus: Corpus,
    *,
    dominant_frequency_hz: float,
    device: torch.device,
):
    from ginn_v2.generator import ConditionalGenerator

    payload = load_checkpoint(checkpoint_path)
    metadata = payload["metadata"]
    if metadata.get("training_mode") not in {"full", "no_seismic"}:
        raise InputContractError(
            "Stage 2 requires a frozen full or no-seismic evidence checkpoint."
        )
    for name, expected in (
        ("sample_domain", corpus.benchmark.sample_domain),
        ("sample_unit", corpus.benchmark.sample_unit),
        ("depth_basis", corpus.benchmark.depth_basis),
    ):
        if metadata.get(name) != expected:
            raise InputContractError(f"evidence checkpoint {name} differs from the corpus.")
    network_config = EvidenceNetworkConfig.from_mapping(payload["model_config"])
    network = BandlimitedEvidenceNetwork(network_config)
    network.load_state_dict(payload["model_state"])
    network.requires_grad_(False)
    return ConditionalGenerator(
        network,
        dominant_frequency_hz=dominant_frequency_hz,
        sample_domain=corpus.benchmark.sample_domain,
        device=device,
    )


def _observed_tiles_for_view(
    tiles: Sequence[StructuredTrainingTile],
    *,
    view: str,
    profile: ObservationPerturbationProfile,
    random_identity: int,
) -> list[ObservationTile]:
    observations: list[ObservationTile] = []
    for tile in tiles:
        paired = build_paired_observation_views(
            tile.observation,
            profile=profile,
            random_identity=random_identity,
        )
        observations.append(getattr(paired, view))
    return observations


def _event_batch(
    corpus: Corpus,
    parent_ids: Sequence[str],
    evidence_generator: Any,
    event_model: EventTrackNetwork,
    *,
    view: str,
    profile: ObservationPerturbationProfile,
    random_identity: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tiles = [tile for parent_id in parent_ids for tile in parent_training_tiles(corpus, parent_id)]
    observations = _observed_tiles_for_view(
        tiles,
        view=view,
        profile=profile,
        random_identity=random_identity,
    )
    evidences = evidence_generator.observe_many(
        observations,
        vp_model_mps=tuple(tile.vp_model_mps for tile in tiles),
    )
    evidence_batch = _evidence_tensor_batch(evidences, observations, device=device)
    target_batch = _event_target_batch(tiles, evidences, event_model, device=device)
    return evidence_batch, target_batch


class _LossAccumulator:
    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.count = 0

    def add(self, values: Mapping[str, torch.Tensor]) -> None:
        for name, value in values.items():
            self.total[name] = self.total.get(name, 0.0) + float(value.detach().cpu())
        self.count += 1

    def finalize(self) -> dict[str, float]:
        if self.count == 0:
            raise ValueError("event loss accumulator is empty.")
        return {name: value / self.count for name, value in self.total.items()}


def _run_event_epoch(
    corpus: Corpus,
    parent_ids: Sequence[str],
    evidence_generator: Any,
    event_model: EventTrackNetwork,
    learning_config: EventLearningConfig,
    *,
    profile: ObservationPerturbationProfile,
    views: Sequence[str],
    random_identity: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    logger: Any,
    label: str,
    self_conditioning_fraction: float,
) -> dict[str, float]:
    training = optimizer is not None
    event_model.train(training)
    chunks = _parent_chunks(parent_ids, learning_config.parent_batch_size)
    accumulator = _LossAccumulator()
    started = time.perf_counter()
    for batch_index, parent_chunk in enumerate(chunks, start=1):
        for view in views:
            evidence, target = _event_batch(
                corpus,
                parent_chunk,
                evidence_generator,
                event_model,
                view=view,
                profile=profile,
                random_identity=random_identity,
                device=device,
            )
            with torch.set_grad_enabled(training):
                event_limit = int(torch.max(target["count"]).item())
                output = event_model(
                    evidence,
                    target["zone_index"],
                    teacher_duration=target["duration"][:, :event_limit],
                    teacher_presence=target["presence"][:, :event_limit],
                    teacher_state=target["state"][:, :event_limit],
                    self_conditioning_fraction=self_conditioning_fraction,
                )
                losses = event_track_loss(event_model, output, evidence, target, learning_config)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(event_model.parameters(), 5.0)
                    optimizer.step()
            accumulator.add(losses)
        if batch_index == 1 or batch_index % learning_config.log_every_batches == 0:
            logger.info(
                "%s | batches=%d/%d | parents=%d/%d | loss=%.6f | elapsed=%.1fs",
                label,
                batch_index,
                len(chunks),
                min(batch_index * learning_config.parent_batch_size, len(parent_ids)),
                len(parent_ids),
                accumulator.total.get("loss", 0.0) / max(accumulator.count, 1),
                time.perf_counter() - started,
            )
    metrics = accumulator.finalize()
    metrics["self_conditioning_fraction"] = float(self_conditioning_fraction)
    return metrics


def _event_self_conditioning_fraction(
    config: EventLearningConfig,
    epoch: int,
) -> float:
    if epoch < 1 or epoch > config.epochs:
        raise ValueError("event curriculum epoch lies outside the training budget.")
    if config.epochs == 1:
        return float(config.self_conditioning_final_fraction)
    progress = (epoch - 1) / (config.epochs - 1)
    return float(
        config.self_conditioning_initial_fraction
        + progress
        * (
            config.self_conditioning_final_fraction
            - config.self_conditioning_initial_fraction
        )
    )


def _event_preflight(
    corpus: Corpus,
    parent_ids: Sequence[str],
    maximum_events: int,
    evidence_generator: Any,
    parity_parent_count: int = 12,
) -> dict[str, Any]:
    counts: list[int] = []
    partial_presence = 0
    track_count = 0
    maximum_highres_error = 0.0
    maximum_model_error = 0.0
    parity_checked = 0
    for parent_index, parent_id in enumerate(parent_ids):
        tiles = parent_training_tiles(corpus, parent_id)
        evidences = (
            evidence_generator.observe_many(
                [tile.observation for tile in tiles],
                vp_model_mps=tuple(tile.vp_model_mps for tile in tiles),
            )
            if parent_index < parity_parent_count
            else tuple(None for _ in tiles)
        )
        for tile, evidence in zip(tiles, evidences, strict=True):
            counts.append(len(tile.event_tracks))
            for track in tile.event_tracks:
                track_count += 1
                partial_presence += int(not np.all(track.presence[tile.observation.lateral_valid]))
            if evidence is None:
                continue
            parity_checked += 1
            roundtrip = rasterize_event_tracks(evidence, lfm_relative_truth_tracks(tile, evidence))
            highres_support = tile.highres_zone_support
            model_support = tile.model_zone_support & roundtrip.model_support
            maximum_highres_error = max(
                maximum_highres_error,
                float(
                    np.max(
                        np.abs(
                            roundtrip.log_ai_highres[highres_support]
                            - tile.log_ai_highres[highres_support]
                        )
                    )
                ),
            )
            maximum_model_error = max(
                maximum_model_error,
                float(
                    np.max(
                        np.abs(
                            roundtrip.model_log_ai[model_support]
                            - tile.model_log_ai[model_support]
                        )
                    )
                ),
            )
    if max(counts) > maximum_events:
        raise InputContractError(
            f"event preflight found {max(counts)} events, above maximum_events={maximum_events}."
        )
    if maximum_highres_error > 5.0e-3 or maximum_model_error > 1.0e-3:
        raise InputContractError(
            "LFM-relative EventTrack rasterizer parity failed: "
            f"highres={maximum_highres_error:.6g}, model={maximum_model_error:.6g}."
        )
    return {
        "status": "passed",
        "parent_count": len(parent_ids),
        "zone_count": len(counts),
        "event_count_minimum": int(min(counts)),
        "event_count_median": float(np.median(counts)),
        "event_count_maximum": int(max(counts)),
        "track_count": int(track_count),
        "partial_presence_track_count": int(partial_presence),
        "parity_zone_count": int(parity_checked),
        "maximum_highres_roundtrip_error": maximum_highres_error,
        "maximum_model_roundtrip_error": maximum_model_error,
    }


def train_event_generator(
    corpus_root: str | Path,
    evidence_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    dominant_frequency_hz: float,
    generator_config: EventGeneratorConfig = EventGeneratorConfig(),
    learning_config: EventLearningConfig = EventLearningConfig(),
    perturbation_profile: ObservationPerturbationProfile = ObservationPerturbationProfile(),
    device_name: str = "auto",
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train the deterministic ordered EventTrack generator with frozen evidence."""

    output = Path(output_dir)
    report_path = output / "event_training_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(output, file_name="event_training.log")
    device, runtime = resolve_device(device_name)
    corpus = load_corpus(corpus_root)
    evidence_generator = _load_evidence_generator(
        evidence_checkpoint,
        corpus,
        dominant_frequency_hz=dominant_frequency_hz,
        device=device,
    )
    training_parents = _select_parents(corpus, "training", learning_config.training_parent_count)
    tuning_parents = _select_parents(corpus, "tuning", learning_config.tuning_parent_count)
    preflight = _event_preflight(
        corpus,
        tuple(training_parents) + tuple(tuning_parents),
        generator_config.maximum_events,
        evidence_generator,
    )
    random.seed(learning_config.random_seed)
    np.random.seed(learning_config.random_seed)
    torch.manual_seed(learning_config.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(learning_config.random_seed)
    event_model = EventTrackNetwork(generator_config, corpus.producer_prior).to(device)
    initialization: dict[str, Any] | None = None
    if initial_checkpoint is not None:
        initial_payload = load_checkpoint(initial_checkpoint)
        if initial_payload["metadata"].get("model_kind") != "autoregressive_event_track":
            raise InputContractError(
                "initial checkpoint is not an autoregressive EventTrack model."
            )
        if tuple(initial_payload["metadata"].get("zone_ids", ())) != tuple(
            event_model.zone_ids
        ):
            raise InputContractError(
                "initial EventTrack checkpoint zone identities differ from the corpus."
            )
        if EventGeneratorConfig.from_mapping(
            initial_payload["model_config"]
        ) != generator_config:
            raise InputContractError(
                "initial EventTrack checkpoint architecture differs from the requested model."
            )
        for name, expected in (
            ("sample_domain", corpus.benchmark.sample_domain),
            ("sample_unit", corpus.benchmark.sample_unit),
            ("depth_basis", corpus.benchmark.depth_basis),
        ):
            if initial_payload["metadata"].get(name) != expected:
                raise InputContractError(
                    f"initial EventTrack checkpoint {name} differs from the corpus."
                )
        event_model.load_state_dict(initial_payload["model_state"])
        initialization = {
            "checkpoint": Path(initial_checkpoint).as_posix(),
            "checkpoint_epoch": int(
                initial_payload["metadata"].get("epoch", -1)
            ),
            "optimizer_reset": True,
        }
    optimizer = torch.optim.AdamW(
        event_model.parameters(),
        lr=learning_config.learning_rate,
        weight_decay=learning_config.weight_decay,
    )
    report: dict[str, Any] = {
        "schema": EVENT_TRAINING_REPORT_SCHEMA,
        "status": "running",
        "runtime": runtime,
        "corpus": {
            "root": Path(corpus_root).as_posix(),
            "schema": corpus.manifest.get("schema_version"),
            "sample_domain": corpus.benchmark.sample_domain,
            "sample_unit": corpus.benchmark.sample_unit,
            "depth_basis": corpus.benchmark.depth_basis,
        },
        "evidence_checkpoint": Path(evidence_checkpoint).as_posix(),
        "generator_config": asdict(generator_config),
        "learning_config": asdict(learning_config),
        "initialization": initialization,
        "preflight": preflight,
        "training_parent_ids": list(training_parents),
        "tuning_parent_ids": list(tuning_parents),
        "history": [],
        "best_epoch": 0,
        "best_selection_loss": None,
    }
    _atomic_json(report_path, report)
    logger.info(
        "event training start | device=%s | training_parents=%d | tuning_parents=%d | max_events=%d",
        device,
        len(training_parents),
        len(tuning_parents),
        generator_config.maximum_events,
    )
    best_loss = math.inf
    epochs_without_improvement = 0
    training_views = ["clean", "peak_poor"] if learning_config.include_peak_poor else ["clean"]
    for epoch in range(1, learning_config.epochs + 1):
        training_self_conditioning = _event_self_conditioning_fraction(
            learning_config, epoch
        )
        shuffled = list(training_parents)
        random.Random(learning_config.random_seed + epoch).shuffle(shuffled)
        train_metrics = _run_event_epoch(
            corpus,
            shuffled,
            evidence_generator,
            event_model,
            learning_config,
            profile=perturbation_profile,
            views=training_views,
            random_identity=learning_config.dirty_random_identity + epoch,
            device=device,
            optimizer=optimizer,
            logger=logger,
            label=(
                f"event epoch {epoch}/{learning_config.epochs} train"
                f" | self_conditioning={training_self_conditioning:.2f}"
            ),
            self_conditioning_fraction=training_self_conditioning,
        )
        tuning_metrics = _run_event_epoch(
            corpus,
            tuning_parents,
            evidence_generator,
            event_model,
            learning_config,
            profile=perturbation_profile,
            views=("clean", "peak_poor"),
            random_identity=learning_config.dirty_random_identity,
            device=device,
            optimizer=None,
            logger=logger,
            label=(
                f"event epoch {epoch}/{learning_config.epochs} tuning"
                f" | self_conditioning="
                f"{learning_config.self_conditioning_final_fraction:.2f}"
            ),
            self_conditioning_fraction=learning_config.self_conditioning_final_fraction,
        )
        selection_loss = float(tuning_metrics["loss"])
        improved = selection_loss < best_loss
        if improved:
            best_loss = selection_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        metadata = {
            "model_kind": "autoregressive_event_track",
            "epoch": epoch,
            "selection_loss": selection_loss,
            "sample_domain": corpus.benchmark.sample_domain,
            "sample_unit": corpus.benchmark.sample_unit,
            "depth_basis": corpus.benchmark.depth_basis,
            "zone_ids": list(event_model.zone_ids),
            "evidence_checkpoint": Path(evidence_checkpoint).as_posix(),
            "training_parent_ids": list(training_parents),
            "tuning_parent_ids": list(tuning_parents),
            "learning_config": asdict(learning_config),
            "initialization": initialization,
        }
        save_checkpoint(
            output / "last.pt",
            model_state=event_model.state_dict(),
            model_config=asdict(generator_config),
            metadata=metadata,
            training_state={"optimizer": optimizer.state_dict()},
            overwrite=True,
        )
        if improved:
            save_checkpoint(
                output / "best.pt",
                model_state=event_model.state_dict(),
                model_config=asdict(generator_config),
                metadata=metadata,
                overwrite=True,
            )
        report["history"].append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "tuning": tuning_metrics,
                "selection_loss": selection_loss,
                "best": improved,
            }
        )
        report["best_epoch"] = epoch if improved else report["best_epoch"]
        report["best_selection_loss"] = best_loss
        _atomic_json(report_path, report)
        logger.info(
            "event epoch %d/%d complete | train_loss=%.6f | tuning_loss=%.6f | best=%s",
            epoch,
            learning_config.epochs,
            train_metrics["loss"],
            selection_loss,
            improved,
        )
        if epochs_without_improvement >= learning_config.early_stopping_patience:
            logger.info("event early stopping | epoch=%d | best_epoch=%d", epoch, report["best_epoch"])
            break
    report["status"] = "completed"
    _atomic_json(report_path, report)
    return report


def _event_metric_summary(
    values: list[dict[str, float]],
) -> dict[str, float | None]:
    if not values:
        raise ValueError("event evaluation produced no values.")
    result: dict[str, float | None] = {}
    for name in values[0]:
        parsed = np.asarray([item[name] for item in values], dtype=np.float64)
        finite = np.isfinite(parsed)
        result[name] = float(np.mean(parsed[finite])) if np.any(finite) else None
    return result


def _map_metrics(
    realization: EventTrackRealization,
    tile: StructuredTrainingTile,
    evidence: BandlimitedEvidence,
) -> dict[str, float]:
    truth_count = len(tile.event_tracks)
    predicted_count = len(realization.tracks)
    truth_tracks = lfm_relative_truth_tracks(tile, evidence)
    matched = min(truth_count, predicted_count)
    state_accuracy = float(
        np.mean(
            [realization.tracks[item].state_id == tile.event_tracks[item].state_id for item in range(matched)]
        )
    ) if matched else 0.0
    duration_error: list[float] = []
    coefficient_error: list[float] = []
    for item in range(matched):
        predicted = realization.tracks[item]
        truth = truth_tracks[item]
        valid = predicted.presence & truth.presence
        if np.any(valid):
            duration_error.extend(
                np.abs(
                    np.log(predicted.duration_fraction[valid].clip(1.0e-6))
                    - np.log(truth.duration_fraction[valid].clip(1.0e-6))
                ).tolist()
            )
            coefficient_error.extend(
                np.abs(predicted.coefficients[valid] - truth.coefficients[valid]).reshape(-1).tolist()
            )
    highres_support = tile.highres_zone_support & np.isfinite(realization.log_ai_highres)
    model_support = tile.model_zone_support & realization.model_support
    evidence_support = tile.model_zone_support & evidence.support
    evidence_log_ai = (
        evidence.background_lfm_linear
        + evidence.projected_log_ai_increment_mean
    )
    return {
        "event_count_bias": float(predicted_count - truth_count),
        "event_count_absolute_error": float(abs(predicted_count - truth_count)),
        "ordered_event_survival_fraction": float(matched / max(truth_count, predicted_count)),
        "ordered_state_accuracy": state_accuracy,
        "duration_log_absolute_error": float(np.mean(duration_error)) if duration_error else float("nan"),
        "coefficient_absolute_error": float(np.mean(coefficient_error)) if coefficient_error else float("nan"),
        "highres_log_ai_rmse": float(np.sqrt(np.mean((realization.log_ai_highres[highres_support] - tile.log_ai_highres[highres_support]) ** 2))),
        "projected_log_ai_rmse": float(np.sqrt(np.mean((realization.model_log_ai[model_support] - tile.model_log_ai[model_support]) ** 2))),
        "evidence_projected_log_ai_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        evidence_log_ai[evidence_support]
                        - tile.model_log_ai[evidence_support]
                    )
                    ** 2
                )
            )
        ),
        "anchor_projected_log_ai_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        evidence.background_lfm_linear[evidence_support]
                        - tile.model_log_ai[evidence_support]
                    )
                    ** 2
                )
            )
        ),
    }


def _empirical_crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Return the empirical ensemble CRPS averaged over supported samples."""

    values = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if values.ndim != 2 or target.shape != (values.shape[1],):
        raise ValueError("CRPS samples must have shape [member, supported_sample].")
    first = float(np.mean(np.abs(values - target[None])))
    pairwise = 0.0
    for left in range(values.shape[0]):
        for right in range(values.shape[0]):
            pairwise += float(np.mean(np.abs(values[left] - values[right])))
    pairwise /= float(values.shape[0] ** 2)
    return first - 0.5 * pairwise


def _ensemble_metrics(
    ensemble: StructuredEnsemble,
    tile: StructuredTrainingTile,
) -> dict[str, float]:
    """Score ensemble summaries without treating their mean as a legal realization."""

    if not ensemble.realizations:
        raise InputContractError("ensemble evaluation requires retained realizations.")
    highres_support = tile.highres_zone_support & ensemble.evidence.highres_support
    model_support = tile.model_zone_support & ensemble.evidence.support
    for realization in ensemble.realizations:
        highres_support &= np.isfinite(realization.log_ai_highres)
        model_support &= realization.model_support & np.isfinite(
            realization.model_log_ai
        )
    if not np.any(highres_support) or not np.any(model_support):
        raise InputContractError("ensemble members have no common evaluation support.")
    highres_samples = np.stack(
        [item.log_ai_highres[highres_support] for item in ensemble.realizations],
        axis=0,
    )
    model_samples = np.stack(
        [item.model_log_ai[model_support] for item in ensemble.realizations],
        axis=0,
    )
    highres_truth = tile.log_ai_highres[highres_support]
    model_truth = tile.model_log_ai[model_support]
    highres_mean = np.mean(highres_samples, axis=0)
    model_mean = np.mean(model_samples, axis=0)
    highres_lower, highres_upper = np.quantile(
        highres_samples, (0.05, 0.95), axis=0
    )
    model_lower, model_upper = np.quantile(model_samples, (0.05, 0.95), axis=0)
    event_counts = np.asarray(
        [len(item.tracks) for item in ensemble.realizations], dtype=np.float64
    )
    truth_count = float(len(tile.event_tracks))
    return {
        "event_count_mean_bias": float(np.mean(event_counts) - truth_count),
        "event_count_mean_absolute_error": float(
            abs(np.mean(event_counts) - truth_count)
        ),
        "event_count_standard_deviation": float(np.std(event_counts)),
        "event_count_range_coverage": float(
            np.min(event_counts) <= truth_count <= np.max(event_counts)
        ),
        "highres_log_ai_mean_rmse": float(
            np.sqrt(np.mean((highres_mean - highres_truth) ** 2))
        ),
        "projected_log_ai_mean_rmse": float(
            np.sqrt(np.mean((model_mean - model_truth) ** 2))
        ),
        "highres_log_ai_crps": _empirical_crps(highres_samples, highres_truth),
        "projected_log_ai_crps": _empirical_crps(model_samples, model_truth),
        "highres_log_ai_90pct_coverage": float(
            np.mean((highres_truth >= highres_lower) & (highres_truth <= highres_upper))
        ),
        "projected_log_ai_90pct_coverage": float(
            np.mean((model_truth >= model_lower) & (model_truth <= model_upper))
        ),
        "highres_log_ai_spread": float(np.mean(np.std(highres_samples, axis=0))),
        "projected_log_ai_spread": float(np.mean(np.std(model_samples, axis=0))),
    }


def _lateral_field_roughness(field: np.ndarray, support: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    valid = np.asarray(support, dtype=bool)
    pair = valid[1:] & valid[:-1]
    if not np.any(pair):
        return 0.0
    difference = np.abs(values[1:] - values[:-1])
    return float(np.mean(difference[pair]))


def _track_lateral_statistics(
    tracks: Sequence[EventTrack],
    lateral_valid: np.ndarray,
) -> dict[str, float]:
    valid = np.asarray(lateral_valid, dtype=bool)
    trace_count = np.sum(
        np.stack([track.presence for track in tracks], axis=0), axis=0
    ).astype(np.float64)
    pair_valid = valid[1:] & valid[:-1]
    count_jump = (
        float(np.mean(np.abs(np.diff(trace_count))[pair_valid]))
        if np.any(pair_valid)
        else 0.0
    )
    duration_jump: list[float] = []
    coefficient_jump: list[float] = []
    survival: list[float] = []
    for track in tracks:
        pair = track.presence[1:] & track.presence[:-1] & pair_valid
        if np.any(pair):
            duration_jump.extend(
                np.abs(np.diff(track.duration_fraction))[pair].tolist()
            )
            coefficient_jump.extend(
                np.linalg.norm(np.diff(track.coefficients, axis=0), axis=-1)[pair].tolist()
            )
        active_pair = pair_valid & (track.presence[1:] | track.presence[:-1])
        if np.any(active_pair):
            survival.append(float(np.mean(pair[active_pair])))
    return {
        "trace_event_count_jump": count_jump,
        "track_duration_jump": float(np.mean(duration_jump)) if duration_jump else 0.0,
        "track_coefficient_jump": (
            float(np.mean(coefficient_jump)) if coefficient_jump else 0.0
        ),
        "track_neighbor_survival": float(np.mean(survival)) if survival else 1.0,
    }


def _section_lateral_metrics(
    ensemble: StructuredEnsemble,
    tile: StructuredTrainingTile,
) -> dict[str, float]:
    truth_tracks = lfm_relative_truth_tracks(tile, ensemble.evidence)
    representative = ensemble.representative
    lateral_valid = tile.observation.lateral_valid
    truth_statistics = _track_lateral_statistics(truth_tracks, lateral_valid)
    representative_statistics = _track_lateral_statistics(
        representative.tracks, lateral_valid
    )
    member_statistics = [
        _track_lateral_statistics(member.tracks, lateral_valid)
        for member in ensemble.realizations
    ]
    model_support = tile.model_zone_support & ensemble.evidence.support
    for member in ensemble.realizations:
        model_support &= member.model_support
    truth_roughness = _lateral_field_roughness(tile.model_log_ai, model_support)
    representative_roughness = _lateral_field_roughness(
        representative.model_log_ai, model_support
    )
    ensemble_mean_roughness = _lateral_field_roughness(
        np.asarray(ensemble.summary["model_log_ai_mean"]), model_support
    )
    member_roughness = np.asarray(
        [
            _lateral_field_roughness(member.model_log_ai, model_support)
            for member in ensemble.realizations
        ],
        dtype=np.float64,
    )
    matched = min(len(truth_tracks), len(representative.tracks))
    presence_intersection = 0
    presence_union = 0
    false_presence = 0
    truth_absence = 0
    for index in range(matched):
        truth_presence = truth_tracks[index].presence & lateral_valid
        predicted_presence = representative.tracks[index].presence & lateral_valid
        presence_intersection += int(np.count_nonzero(truth_presence & predicted_presence))
        presence_union += int(np.count_nonzero(truth_presence | predicted_presence))
        false_presence += int(np.count_nonzero(~truth_presence & predicted_presence & lateral_valid))
        truth_absence += int(np.count_nonzero(~truth_presence & lateral_valid))
    result = {
        "truth_projected_lateral_roughness": truth_roughness,
        "representative_projected_lateral_roughness": representative_roughness,
        "ensemble_mean_projected_lateral_roughness": ensemble_mean_roughness,
        "member_projected_lateral_roughness_mean": float(np.mean(member_roughness)),
        "member_projected_lateral_roughness_std": float(np.std(member_roughness)),
        "representative_to_truth_roughness_ratio": float(
            representative_roughness / max(truth_roughness, 1.0e-8)
        ),
        "ensemble_mean_to_truth_roughness_ratio": float(
            ensemble_mean_roughness / max(truth_roughness, 1.0e-8)
        ),
        "representative_matched_presence_iou": float(
            presence_intersection / presence_union if presence_union else 1.0
        ),
        "representative_false_presence_rate": float(
            false_presence / truth_absence if truth_absence else 0.0
        ),
    }
    for prefix, statistics in (
        ("truth", truth_statistics),
        ("representative", representative_statistics),
    ):
        result.update(
            {f"{prefix}_{name}": value for name, value in statistics.items()}
        )
    for name in member_statistics[0]:
        result[f"member_{name}_mean"] = float(
            np.mean([item[name] for item in member_statistics])
        )
    return result


def _section_figure_parent_ids(parent_ids: Sequence[str]) -> set[str]:
    selected: set[str] = set()
    for family in ("none", "wedge", "pinchout"):
        for regime in ("iid", "combination_holdout"):
            token = f"section__{family}__{regime}__"
            match = next((item for item in parent_ids if token in item), None)
            if match is not None:
                selected.add(match)
    return selected


def _plot_section_ensemble(
    path: Path,
    tile: StructuredTrainingTile,
    ensemble: StructuredEnsemble,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    lateral = tile.observation.lateral_m
    model_axis = tile.observation.model_axis.coordinates
    highres_axis = tile.observation.highres_axis.coordinates
    members = ensemble.realizations
    if not members:
        raise InputContractError("section figures require retained ensemble members.")
    second_member = members[1] if len(members) > 1 else members[0]
    highres_support = tile.highres_zone_support & ensemble.evidence.highres_support

    def supported_highres(field: np.ndarray) -> np.ndarray:
        return np.where(highres_support, np.asarray(field, dtype=np.float64), np.nan)

    ai_fields = tuple(
        supported_highres(field)
        for field in (
            tile.log_ai_highres,
            ensemble.representative.log_ai_highres,
            members[0].log_ai_highres,
            second_member.log_ai_highres,
            np.asarray(ensemble.summary["highres_log_ai_mean"]),
        )
    )
    finite_ai = np.concatenate(
        [field[np.isfinite(field)] for field in ai_fields]
    )
    ai_lower, ai_upper = np.quantile(finite_ai, (0.02, 0.98))
    seismic = np.where(
        tile.observation.observed_valid, tile.observation.seismic, np.nan
    )
    lfm = np.where(tile.observation.observed_valid, tile.observation.lfm, np.nan)
    seismic_limit = float(np.nanquantile(np.abs(seismic), 0.98))
    figure, axes = plt.subplots(4, 2, figsize=(16, 18), constrained_layout=True)

    def draw(
        axis: Any,
        field: np.ndarray,
        vertical: np.ndarray,
        label: str,
        *,
        cmap: str,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        image = axis.pcolormesh(
            lateral,
            vertical,
            np.asarray(field).T,
            shading="auto",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
        )
        axis.invert_yaxis()
        axis.set_title(label)
        axis.set_xlabel("lateral distance (m)")
        axis.set_ylabel(f"{tile.observation.sample_domain} ({tile.observation.model_axis.unit})")
        figure.colorbar(image, ax=axis, shrink=0.82)

    draw(
        axes[0, 0],
        seismic,
        model_axis,
        "clean seismic",
        cmap="seismic",
        lower=-seismic_limit,
        upper=seismic_limit,
    )
    draw(axes[0, 1], lfm, model_axis, "full LFM", cmap="viridis")
    draw(axes[1, 0], ai_fields[0], highres_axis, "truth high-resolution log-AI", cmap="viridis", lower=ai_lower, upper=ai_upper)
    draw(axes[1, 1], ai_fields[1], highres_axis, "representative member", cmap="viridis", lower=ai_lower, upper=ai_upper)
    draw(axes[2, 0], ai_fields[2], highres_axis, "member 0", cmap="viridis", lower=ai_lower, upper=ai_upper)
    draw(axes[2, 1], ai_fields[3], highres_axis, "member 1", cmap="viridis", lower=ai_lower, upper=ai_upper)
    draw(axes[3, 0], ai_fields[4], highres_axis, "ensemble mean (summary only)", cmap="viridis", lower=ai_lower, upper=ai_upper)
    draw(axes[3, 1], supported_highres(np.asarray(ensemble.summary["highres_log_ai_std"])), highres_axis, "ensemble standard deviation", cmap="magma", lower=0.0)
    figure.suptitle(title)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def evaluate_event_generator(
    corpus_root: str | Path,
    evidence_checkpoint: str | Path,
    event_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    dominant_frequency_hz: float,
    config: EventEvaluationConfig = EventEvaluationConfig(),
    generation_policy: GenerationPolicy | None = None,
    perturbation_profile: ObservationPerturbationProfile = ObservationPerturbationProfile(),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate deterministic tracks or a K-member ensemble through one seam."""

    output = Path(output_dir)
    report_path = output / "event_evaluation_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(output, file_name="event_evaluation.log")
    device, runtime = resolve_device(device_name)
    corpus = load_corpus(corpus_root)
    evidence_generator = _load_evidence_generator(
        evidence_checkpoint,
        corpus,
        dominant_frequency_hz=dominant_frequency_hz,
        device=device,
    )
    payload = load_checkpoint(event_checkpoint)
    if payload["metadata"].get("model_kind") != "autoregressive_event_track":
        raise InputContractError("checkpoint is not an EventTrack model.")
    if tuple(payload["metadata"].get("zone_ids", ())) != tuple(
        zone.zone_id for zone in corpus.producer_prior.zones
    ):
        raise InputContractError("EventTrack checkpoint zone identities differ from the corpus.")
    for name, expected in (
        ("sample_domain", corpus.benchmark.sample_domain),
        ("sample_unit", corpus.benchmark.sample_unit),
        ("depth_basis", corpus.benchmark.depth_basis),
    ):
        if payload["metadata"].get(name) != expected:
            raise InputContractError(f"EventTrack checkpoint {name} differs from the corpus.")
    event_config = EventGeneratorConfig.from_mapping(payload["model_config"])
    event_model = EventTrackNetwork(event_config, corpus.producer_prior).to(device)
    event_model.load_state_dict(payload["model_state"])
    event_model.eval()
    ensemble_generator = None
    if generation_policy is not None:
        if not generation_policy.retain_realizations:
            raise InputContractError(
                "ensemble evaluation requires retain_realizations=True."
            )
        from ginn_v2.generator import ConditionalGenerator

        ensemble_generator = ConditionalGenerator(
            evidence_generator.network,
            dominant_frequency_hz=dominant_frequency_hz,
            sample_domain=corpus.benchmark.sample_domain,
            device=device,
            event_network=event_model,
            producer_prior=corpus.producer_prior,
        )
    parent_ids = _select_parents(corpus, config.split, config.parent_count)
    section_preflight = None
    if config.split == "section_gate":
        logger.info("section preflight start | parents=%d", len(parent_ids))
        section_preflight = _preflight_section_evaluation(corpus, parent_ids)
        logger.info(
            "section preflight completed | tiles=%d | valid_traces=%d | "
            "singleton_anchors=%d | minimum_highres_capacity=%d",
            section_preflight["tile_count"],
            section_preflight["valid_trace_count"],
            section_preflight["singleton_anchor_count"],
            section_preflight["minimum_highres_sample_capacity"],
        )
    figure_parent_ids = (
        _section_figure_parent_ids(parent_ids)
        if config.split == "section_gate" and generation_policy is not None
        else set()
    )
    values: dict[str, list[dict[str, float]]] = {
        "clean": [],
        "peak_poor": [],
        "prior_map": [],
    }
    ensemble_values: dict[str, list[dict[str, float]]] = {
        "clean": [],
        "peak_poor": [],
    }
    section_lateral_values: list[dict[str, float]] = []
    figure_paths: list[str] = []
    paired_count_difference: list[float] = []
    started = time.perf_counter()
    processed = 0
    for parent_chunk in _parent_chunks(parent_ids, config.parent_batch_size):
        parent_tiles = [
            (parent_id, tile)
            for parent_id in parent_chunk
            for tile in parent_training_tiles(corpus, parent_id)
        ]
        tile_parent_ids = [item[0] for item in parent_tiles]
        tiles = [item[1] for item in parent_tiles]
        per_condition_count: dict[str, list[int]] = {}
        for condition in ("clean", "peak_poor"):
            observations = _observed_tiles_for_view(
                tiles,
                view=condition,
                profile=perturbation_profile,
                random_identity=config.dirty_random_identity,
            )
            condition_counts: list[int] = []
            if ensemble_generator is None:
                evidences = evidence_generator.observe_many(
                    observations,
                    vp_model_mps=tuple(tile.vp_model_mps for tile in tiles),
                )
                predictions: tuple[StructuredEnsemble | None, ...] = tuple(
                    None for _ in tiles
                )
            else:
                generated = tuple(
                    ensemble_generator.generate(
                        observation,
                        generation_policy,
                        vp_model_mps=tile.vp_model_mps,
                    )
                    for tile, observation in zip(tiles, observations, strict=True)
                )
                evidences = tuple(item.evidence for item in generated)
                predictions = generated
            for parent_id, tile, evidence, observation, prediction in zip(
                tile_parent_ids,
                tiles,
                evidences,
                observations,
                predictions,
                strict=True,
            ):
                if prediction is None:
                    realization = realize_event_tracks(
                        event_model, corpus.producer_prior, evidence, observation
                    )
                else:
                    realization = prediction.representative
                    ensemble_values[condition].append(
                        _ensemble_metrics(prediction, tile)
                    )
                    if config.split == "section_gate" and condition == "clean":
                        section_lateral_values.append(
                            _section_lateral_metrics(prediction, tile)
                        )
                        if parent_id in figure_parent_ids:
                            identity = "".join(
                                character
                                if character.isalnum() or character in {"-", "_"}
                                else "_"
                                for character in tile.observation.identity
                            )
                            figure_path = (
                                output
                                / "figures"
                                / "section_gate"
                                / f"{identity}.png"
                            )
                            _plot_section_ensemble(
                                figure_path,
                                tile,
                                prediction,
                                title=f"{parent_id} | {tile.observation.identity}",
                            )
                            figure_paths.append(figure_path.as_posix())
                values[condition].append(_map_metrics(realization, tile, evidence))
                if condition == "clean":
                    prior_realization = realize_prior_event_tracks(
                        event_model, evidence, observation
                    )
                    values["prior_map"].append(
                        _map_metrics(prior_realization, tile, evidence)
                    )
                condition_counts.append(len(realization.tracks))
            per_condition_count[condition] = condition_counts
        paired_count_difference.extend(
            float(peak - clean)
            for clean, peak in zip(
                per_condition_count["clean"],
                per_condition_count["peak_poor"],
                strict=True,
            )
        )
        processed += len(parent_chunk)
        if processed == len(parent_ids) or processed % config.log_every_batches == 0:
            logger.info(
                "event evaluation | parents=%d/%d | elapsed=%.1fs",
                processed,
                len(parent_ids),
                time.perf_counter() - started,
            )
    metrics = {
        condition: _event_metric_summary(condition_values)
        for condition, condition_values in values.items()
    }
    ensemble_metrics = (
        {
            condition: _event_metric_summary(condition_values)
            for condition, condition_values in ensemble_values.items()
        }
        if generation_policy is not None
        else None
    )
    section_lateral_metrics = (
        _event_metric_summary(section_lateral_values)
        if section_lateral_values
        else None
    )
    robustness = {
        "peak_poor_minus_clean_event_count_mean": float(
            np.mean(paired_count_difference)
        ),
        "peak_poor_minus_clean_event_count_absolute_mean": float(
            np.mean(np.abs(paired_count_difference))
        ),
    }
    clean_rmse = metrics["clean"]["projected_log_ai_rmse"]
    evidence_rmse = metrics["clean"]["evidence_projected_log_ai_rmse"]
    anchor_rmse = metrics["clean"]["anchor_projected_log_ai_rmse"]
    prior_rmse = metrics["prior_map"]["projected_log_ai_rmse"]
    generation_gate = {
        "generator_minus_evidence_projected_rmse": float(clean_rmse - evidence_rmse),
        "generator_minus_anchor_projected_rmse": float(clean_rmse - anchor_rmse),
        "generator_minus_prior_projected_rmse": float(clean_rmse - prior_rmse),
        "maximum_evidence_rmse_degradation": float(
            config.maximum_evidence_rmse_degradation
        ),
        "improves_over_anchor": bool(clean_rmse < anchor_rmse),
        "improves_over_prior": bool(clean_rmse < prior_rmse),
        "preserves_bandlimited_evidence": bool(
            clean_rmse - evidence_rmse
            <= config.maximum_evidence_rmse_degradation
        ),
    }
    generation_gate["passed"] = bool(
        generation_gate["improves_over_prior"]
        and generation_gate["improves_over_anchor"]
        and generation_gate["preserves_bandlimited_evidence"]
    )
    if ensemble_metrics is not None:
        ensemble_mean_rmse = ensemble_metrics["clean"][
            "projected_log_ai_mean_rmse"
        ]
        generation_gate.update(
            {
                "ensemble_mean_minus_evidence_projected_rmse": float(
                    ensemble_mean_rmse - evidence_rmse
                ),
                "ensemble_mean_minus_anchor_projected_rmse": float(
                    ensemble_mean_rmse - anchor_rmse
                ),
                "ensemble_mean_minus_prior_projected_rmse": float(
                    ensemble_mean_rmse - prior_rmse
                ),
                "ensemble_mean_improves_over_anchor": bool(
                    ensemble_mean_rmse < anchor_rmse
                ),
                "ensemble_mean_improves_over_prior": bool(
                    ensemble_mean_rmse < prior_rmse
                ),
                "ensemble_mean_preserves_bandlimited_evidence": bool(
                    ensemble_mean_rmse - evidence_rmse
                    <= config.maximum_evidence_rmse_degradation
                ),
            }
        )
        generation_gate["passed"] = bool(
            generation_gate["improves_over_prior"]
            and generation_gate["improves_over_anchor"]
            and generation_gate["ensemble_mean_improves_over_prior"]
            and generation_gate["ensemble_mean_improves_over_anchor"]
            and generation_gate["ensemble_mean_preserves_bandlimited_evidence"]
        )
    report = {
        "schema": EVENT_EVALUATION_REPORT_SCHEMA,
        "status": "completed",
        "runtime": runtime,
        "split": config.split,
        "parent_ids": list(parent_ids),
        "section_preflight": section_preflight,
        "evidence_checkpoint": Path(evidence_checkpoint).as_posix(),
        "event_checkpoint": Path(event_checkpoint).as_posix(),
        "checkpoint_epoch": int(payload["metadata"].get("epoch", -1)),
        "generation_policy": (
            asdict(generation_policy) if generation_policy is not None else None
        ),
        "metrics": metrics,
        "ensemble_metrics": ensemble_metrics,
        "section_lateral_metrics": section_lateral_metrics,
        "figure_paths": figure_paths,
        "paired_robustness": robustness,
        (
            "ensemble_gate"
            if generation_policy is not None
            else "deterministic_gate"
        ): generation_gate,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _atomic_json(report_path, report)
    return report


def load_event_generation_policy(path: str | Path) -> GenerationPolicy:
    """Load one calibrated generation policy without fingerprint equality gates."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise InputContractError("event generation policy root must be a mapping.")
    if payload.get("schema") != EVENT_GENERATION_POLICY_SCHEMA:
        raise InputContractError("event generation policy schema is unsupported.")
    if payload.get("status") != "completed":
        raise InputContractError("event generation policy is not completed.")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise InputContractError("event generation policy payload is missing policy.")
    return GenerationPolicy.from_mapping(policy)


def _calibration_candidate_summary(
    report: Mapping[str, Any],
    policy: GenerationPolicy,
    report_path: Path,
) -> dict[str, Any]:
    ensemble = report["ensemble_metrics"]["clean"]
    representative = report["metrics"]["clean"]
    return {
        "policy": asdict(policy),
        "report": report_path.as_posix(),
        "event_count_mean_bias": ensemble["event_count_mean_bias"],
        "event_count_mean_absolute_error": ensemble[
            "event_count_mean_absolute_error"
        ],
        "highres_log_ai_90pct_coverage": ensemble[
            "highres_log_ai_90pct_coverage"
        ],
        "projected_log_ai_90pct_coverage": ensemble[
            "projected_log_ai_90pct_coverage"
        ],
        "projected_log_ai_crps": ensemble["projected_log_ai_crps"],
        "highres_log_ai_crps": ensemble["highres_log_ai_crps"],
        "representative_projected_log_ai_rmse": representative[
            "projected_log_ai_rmse"
        ],
        "ensemble_gate": report["ensemble_gate"],
    }


def _load_reusable_policy_candidates(
    source_path: str | Path,
    *,
    expected_config: EventPolicyCalibrationConfig,
    evidence_checkpoint: str | Path,
    event_checkpoint: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild candidate summaries from completed evaluator reports."""

    source = Path(source_path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise InputContractError("calibration candidate source must be a mapping.")
    if payload.get("schema") not in _REUSABLE_EVENT_POLICY_CALIBRATION_SCHEMAS:
        raise InputContractError("calibration candidate source schema is unsupported.")
    if payload.get("status") != "completed":
        raise InputContractError("calibration candidate source is not completed.")
    source_config = payload.get("config")
    if not isinstance(source_config, Mapping):
        raise InputContractError("calibration candidate source is missing config.")
    if EventPolicyCalibrationConfig.from_mapping(source_config) != expected_config:
        raise InputContractError(
            "calibration candidate source config differs from the requested config."
        )
    for name, expected in (
        ("evidence_checkpoint", evidence_checkpoint),
        ("event_checkpoint", event_checkpoint),
    ):
        published = payload.get(name)
        if not isinstance(published, str) or (
            Path(published).resolve() != Path(expected).resolve()
        ):
            raise InputContractError(
                f"calibration candidate source {name} differs from the request."
            )

    def rebuild(
        values: object,
        *,
        expected_count: int,
        include_coverage_error: bool,
    ) -> list[dict[str, Any]]:
        if not isinstance(values, list) or len(values) != expected_count:
            raise InputContractError(
                "calibration candidate source has an invalid candidate count."
            )
        summaries: list[dict[str, Any]] = []
        for value in values:
            if not isinstance(value, Mapping) or not isinstance(
                value.get("report"), str
            ):
                raise InputContractError(
                    "calibration candidate source has an invalid report reference."
                )
            candidate_path = Path(value["report"])
            with candidate_path.open("r", encoding="utf-8") as handle:
                candidate_report = json.load(handle)
            if not isinstance(candidate_report, Mapping):
                raise InputContractError(
                    "calibration candidate report must be a mapping."
                )
            if candidate_report.get("schema") != EVENT_EVALUATION_REPORT_SCHEMA:
                raise InputContractError(
                    "calibration candidate evaluation schema is unsupported."
                )
            if candidate_report.get("status") != "completed":
                raise InputContractError(
                    "calibration candidate evaluation is not completed."
                )
            policy_value = candidate_report.get("generation_policy")
            if not isinstance(policy_value, Mapping):
                raise InputContractError(
                    "calibration candidate evaluation is missing generation policy."
                )
            policy = GenerationPolicy.from_mapping(policy_value)
            summary = _calibration_candidate_summary(
                candidate_report,
                policy,
                candidate_path,
            )
            if include_coverage_error:
                summary["coverage_target_absolute_error"] = float(
                    abs(
                        float(summary["highres_log_ai_90pct_coverage"])
                        - expected_config.target_coverage
                    )
                    + abs(
                        float(summary["projected_log_ai_90pct_coverage"])
                        - expected_config.target_coverage
                    )
                )
            summaries.append(summary)
        return summaries

    return (
        rebuild(
            payload.get("density_candidates"),
            expected_count=len(expected_config.density_candidates),
            include_coverage_error=False,
        ),
        rebuild(
            payload.get("profile_temperature_candidates"),
            expected_count=len(expected_config.profile_temperature_candidates),
            include_coverage_error=True,
        ),
    )


def calibrate_event_generation_policy(
    corpus_root: str | Path,
    evidence_checkpoint: str | Path,
    event_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    dominant_frequency_hz: float,
    config: EventPolicyCalibrationConfig = EventPolicyCalibrationConfig(),
    perturbation_profile: ObservationPerturbationProfile = ObservationPerturbationProfile(),
    device_name: str = "auto",
    reuse_candidates_from: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate or reuse candidates, then publish one calibrated policy."""

    output = Path(output_dir)
    report_path = output / "event_policy_calibration_report.json"
    policy_path = output / "event_generation_policy.json"
    if report_path.exists() or policy_path.exists():
        raise FileExistsError(report_path if report_path.exists() else policy_path)
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_training_logger(output, file_name="event_policy_calibration.log")
    started = time.perf_counter()
    evaluation_config = EventEvaluationConfig(
        split="calibration",
        parent_count=config.parent_count,
        parent_batch_size=config.parent_batch_size,
        dirty_random_identity=config.random_identity,
        log_every_batches=config.log_every_batches,
        maximum_evidence_rmse_degradation=(
            config.maximum_evidence_rmse_degradation
        ),
    )
    profile_temperature_candidates: list[dict[str, Any]] | None = None
    if reuse_candidates_from is not None:
        density_candidates, profile_temperature_candidates = (
            _load_reusable_policy_candidates(
                reuse_candidates_from,
                expected_config=config,
                evidence_checkpoint=evidence_checkpoint,
                event_checkpoint=event_checkpoint,
            )
        )
        logger.info(
            "reused completed calibration candidates | source=%s",
            Path(reuse_candidates_from),
        )
    else:
        density_candidates = []
        for index, density in enumerate(config.density_candidates):
            policy = GenerationPolicy(
                realization_count=config.density_realization_count,
                random_identity=config.random_identity,
                retain_realizations=True,
                lateral_correlation_m=config.lateral_correlation_m,
                event_density_multiplier=float(density),
                structure_sampling_temperature=(
                    config.structure_sampling_temperature
                ),
                profile_sampling_temperature=(
                    config.baseline_profile_sampling_temperature
                ),
            )
            candidate_output = output / "candidates" / f"density_{index:02d}"
            candidate_report = evaluate_event_generator(
                corpus_root,
                evidence_checkpoint,
                event_checkpoint,
                candidate_output,
                dominant_frequency_hz=dominant_frequency_hz,
                config=evaluation_config,
                generation_policy=policy,
                perturbation_profile=perturbation_profile,
                device_name=device_name,
            )
            summary = _calibration_candidate_summary(
                candidate_report,
                policy,
                candidate_output / "event_evaluation_report.json",
            )
            density_candidates.append(summary)
            logger.info(
                "density calibration %d/%d | multiplier=%.4f | count_bias=%.4f | count_mae=%.4f",
                index + 1,
                len(config.density_candidates),
                density,
                summary["event_count_mean_bias"],
                summary["event_count_mean_absolute_error"],
            )
    selected_density = min(
        density_candidates,
        key=lambda item: (
            abs(float(item["event_count_mean_bias"])),
            float(item["event_count_mean_absolute_error"]),
            float(item["projected_log_ai_crps"]),
        ),
    )
    density_multiplier = float(
        selected_density["policy"]["event_density_multiplier"]
    )

    if profile_temperature_candidates is None:
        profile_temperature_candidates = []
        for index, profile_temperature in enumerate(
            config.profile_temperature_candidates
        ):
            policy = GenerationPolicy(
                realization_count=config.profile_realization_count,
                random_identity=config.random_identity,
                retain_realizations=True,
                lateral_correlation_m=config.lateral_correlation_m,
                event_density_multiplier=density_multiplier,
                structure_sampling_temperature=(
                    config.structure_sampling_temperature
                ),
                profile_sampling_temperature=float(profile_temperature),
            )
            candidate_output = (
                output / "candidates" / f"profile_temperature_{index:02d}"
            )
            candidate_report = evaluate_event_generator(
                corpus_root,
                evidence_checkpoint,
                event_checkpoint,
                candidate_output,
                dominant_frequency_hz=dominant_frequency_hz,
                config=evaluation_config,
                generation_policy=policy,
                perturbation_profile=perturbation_profile,
                device_name=device_name,
            )
            summary = _calibration_candidate_summary(
                candidate_report,
                policy,
                candidate_output / "event_evaluation_report.json",
            )
            summary["coverage_target_absolute_error"] = float(
                abs(
                    float(summary["highres_log_ai_90pct_coverage"])
                    - config.target_coverage
                )
                + abs(
                    float(summary["projected_log_ai_90pct_coverage"])
                    - config.target_coverage
                )
            )
            profile_temperature_candidates.append(summary)
            logger.info(
                "profile calibration %d/%d | temperature=%.4f | coverage_error=%.4f | crps=%.5f",
                index + 1,
                len(config.profile_temperature_candidates),
                profile_temperature,
                summary["coverage_target_absolute_error"],
                summary["projected_log_ai_crps"],
            )
    reused_profile_densities = {
        float(item["policy"]["event_density_multiplier"])
        for item in profile_temperature_candidates
    }
    if reused_profile_densities != {density_multiplier}:
        raise InputContractError(
            "profile candidates were not evaluated at the selected density."
        )
    count_bias_values = tuple(
        float(item["event_count_mean_bias"])
        for item in profile_temperature_candidates
    )
    count_bias_span = max(count_bias_values) - min(count_bias_values)
    if count_bias_span > 1.0e-9:
        raise RuntimeError(
            "profile temperature calibration changed event density; "
            f"event_count_mean_bias_span={count_bias_span:.6g}"
        )
    selected_profile_temperature = min(
        profile_temperature_candidates,
        key=lambda item: (
            float(item["projected_log_ai_crps"]),
            float(item["highres_log_ai_crps"]),
            float(item["coverage_target_absolute_error"]),
        ),
    )
    selected_policy = GenerationPolicy(
        realization_count=config.final_realization_count,
        random_identity=config.random_identity,
        retain_realizations=True,
        lateral_correlation_m=config.lateral_correlation_m,
        event_density_multiplier=density_multiplier,
        structure_sampling_temperature=config.structure_sampling_temperature,
        profile_sampling_temperature=float(
            selected_profile_temperature["policy"][
                "profile_sampling_temperature"
            ]
        ),
    )
    selected_profile_value = selected_policy.profile_sampling_temperature
    profile_temperature_at_grid_boundary = bool(
        np.isclose(
            selected_profile_value,
            min(config.profile_temperature_candidates),
        )
        or np.isclose(
            selected_profile_value,
            max(config.profile_temperature_candidates),
        )
    )
    profile_coverage_target_reached = bool(
        float(selected_profile_temperature["highres_log_ai_90pct_coverage"])
        >= config.target_coverage
        and float(
            selected_profile_temperature["projected_log_ai_90pct_coverage"]
        )
        >= config.target_coverage
    )
    calibration_warnings: list[str] = []
    if profile_temperature_at_grid_boundary:
        calibration_warnings.append(
            "selected profile temperature lies on the calibration grid boundary"
        )
    if not profile_coverage_target_reached:
        calibration_warnings.append(
            "selected profile temperature does not reach nominal coverage target"
        )
    policy_payload = {
        "schema": EVENT_GENERATION_POLICY_SCHEMA,
        "status": "completed",
        "policy": asdict(selected_policy),
        "calibration_split": "calibration",
        "calibration_parent_count": config.parent_count,
        "selection_rule": "projected_crps_then_highres_crps_then_coverage_v1",
        "calibration_warnings": calibration_warnings,
    }
    _atomic_json(policy_path, policy_payload)
    report = {
        "schema": EVENT_POLICY_CALIBRATION_REPORT_SCHEMA,
        "status": "completed",
        "config": asdict(config),
        "evidence_checkpoint": Path(evidence_checkpoint).as_posix(),
        "event_checkpoint": Path(event_checkpoint).as_posix(),
        "reuse_candidates_from": (
            Path(reuse_candidates_from).as_posix()
            if reuse_candidates_from is not None
            else None
        ),
        "selection_rule": "projected_crps_then_highres_crps_then_coverage_v1",
        "density_candidates": density_candidates,
        "selected_density_candidate": selected_density,
        "profile_temperature_candidates": profile_temperature_candidates,
        "selected_profile_temperature_candidate": (
            selected_profile_temperature
        ),
        "profile_calibration_event_count_bias_span": count_bias_span,
        "selected_profile_temperature_at_grid_boundary": (
            profile_temperature_at_grid_boundary
        ),
        "profile_coverage_target_reached": profile_coverage_target_reached,
        "warnings": calibration_warnings,
        "selected_policy": asdict(selected_policy),
        "policy_artifact": policy_path.as_posix(),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _atomic_json(report_path, report)
    return report


__all__ = [
    "EVENT_EVALUATION_REPORT_SCHEMA",
    "EVENT_GENERATION_POLICY_SCHEMA",
    "EVENT_POLICY_CALIBRATION_REPORT_SCHEMA",
    "EVENT_TRAINING_REPORT_SCHEMA",
    "EventEvaluationConfig",
    "EventGeneratorConfig",
    "EventLearningConfig",
    "EventPolicyCalibrationConfig",
    "EventTrack",
    "EventTrackNetwork",
    "ProducerPrior",
    "calibrate_event_generation_policy",
    "evaluate_event_generator",
    "event_track_loss",
    "load_event_generation_policy",
    "rasterize_event_tracks",
    "realize_event_tracks",
    "sample_event_track_realizations",
    "train_event_generator",
    "validate_event_track_order",
]
