"""Ordered event-track contracts and zone-fraction prior definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from ginn_v2.contracts import EventTrack, InputContractError


@dataclass(frozen=True)
class SemiMarkovPrior:
    """Transition and duration statistics expressed in zone fractions."""

    initial_probability: np.ndarray
    transition_probability: np.ndarray
    duration_fraction_mean: np.ndarray
    duration_fraction_std: np.ndarray
    maximum_duration_fraction: float = 1.0

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_probability, dtype=np.float64)
        transition = np.asarray(self.transition_probability, dtype=np.float64)
        mean = np.asarray(self.duration_fraction_mean, dtype=np.float64)
        std = np.asarray(self.duration_fraction_std, dtype=np.float64)
        states = int(initial.size)
        if states <= 0 or transition.shape != (states, states):
            raise InputContractError("prior transition shape must match state count.")
        if mean.shape != (states,) or std.shape != (states,):
            raise InputContractError("duration statistics must match state count.")
        if any(
            np.any(~np.isfinite(value)) or np.any(value <= 0.0)
            for value in (initial, transition, mean, std)
        ):
            raise InputContractError("prior probabilities and durations must be positive.")
        maximum = float(self.maximum_duration_fraction)
        if not np.isfinite(maximum) or not 0.0 < maximum <= 1.0:
            raise InputContractError("maximum_duration_fraction must lie in (0, 1].")
        object.__setattr__(self, "initial_probability", initial / np.sum(initial))
        object.__setattr__(
            self,
            "transition_probability",
            transition / np.sum(transition, axis=1, keepdims=True),
        )
        object.__setattr__(self, "duration_fraction_mean", mean)
        object.__setattr__(self, "duration_fraction_std", std)
        object.__setattr__(self, "maximum_duration_fraction", maximum)

    @classmethod
    def default(cls) -> "SemiMarkovPrior":
        return cls(
            initial_probability=np.full(3, 1.0 / 3.0),
            transition_probability=np.asarray(
                ((0.20, 0.45, 0.35), (0.40, 0.20, 0.40), (0.35, 0.45, 0.20)),
                dtype=np.float64,
            ),
            duration_fraction_mean=np.asarray((0.08, 0.10, 0.08)),
            duration_fraction_std=np.asarray((0.05, 0.06, 0.05)),
        )

    def duration_log_probability(self, sample_count: int) -> np.ndarray:
        if isinstance(sample_count, bool) or sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        maximum = max(
            1,
            min(sample_count, int(np.ceil(self.maximum_duration_fraction * sample_count))),
        )
        fractions = np.arange(1, maximum + 1, dtype=np.float64) / sample_count
        result = np.full(
            (self.initial_probability.size, maximum + 1), -np.inf, dtype=np.float64
        )
        for state in range(self.initial_probability.size):
            z = (fractions - self.duration_fraction_mean[state]) / self.duration_fraction_std[state]
            result[state, 1:] = -0.5 * z * z
            result[state, 1:] -= logsumexp(result[state, 1:])
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemiMarkovPrior":
        if (
            value.get("schema") != "structured_ginn_v2_semi_markov_prior_v1"
            or value.get("duration_unit") != "zone_fraction"
        ):
            raise ValueError("unsupported semi-Markov prior schema.")
        return cls(
            initial_probability=np.asarray(value["initial_probability"]),
            transition_probability=np.asarray(value["transition_probability"]),
            duration_fraction_mean=np.asarray(value["duration_fraction_mean"]),
            duration_fraction_std=np.asarray(value["duration_fraction_std"]),
            maximum_duration_fraction=float(value["maximum_duration_fraction"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "structured_ginn_v2_semi_markov_prior_v1",
            "duration_unit": "zone_fraction",
            "initial_probability": self.initial_probability.tolist(),
            "transition_probability": self.transition_probability.tolist(),
            "duration_fraction_mean": self.duration_fraction_mean.tolist(),
            "duration_fraction_std": self.duration_fraction_std.tolist(),
            "maximum_duration_fraction": self.maximum_duration_fraction,
        }


@dataclass(frozen=True)
class SemiMarkovConditioning:
    """Frozen relative weights used by the event decoder."""

    state_evidence_weight: float = 1.0
    duration_temperature: float = 1.0
    transition_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "state_evidence_weight",
            "duration_temperature",
            "transition_temperature",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise InputContractError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemiMarkovConditioning":
        if value.get("schema") != "structured_ginn_v2_hsmm_conditioning_v1":
            raise ValueError("unsupported event conditioning schema.")
        return cls(
            state_evidence_weight=float(value["state_evidence_weight"]),
            duration_temperature=float(value["duration_temperature"]),
            transition_temperature=float(value["transition_temperature"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "structured_ginn_v2_hsmm_conditioning_v1",
            "state_evidence_weight": self.state_evidence_weight,
            "duration_temperature": self.duration_temperature,
            "transition_temperature": self.transition_temperature,
        }


def validate_event_track_order(
    tracks: Sequence[EventTrack],
    *,
    width: int,
    tolerance: float = 1.0e-6,
) -> None:
    """Validate the topology-free part of an ordered section representation."""

    if width <= 0 or not tracks:
        raise InputContractError("an event section requires a positive width and tracks.")
    identities = [track.event_id for track in tracks]
    if len(set(identities)) != len(identities):
        raise InputContractError("event identities must be unique within a zone.")
    for track in tracks:
        if track.presence.size != width:
            raise InputContractError("event track width differs from section width.")
    duration = np.stack([track.duration_fraction for track in tracks], axis=0)
    active = np.stack([track.presence for track in tracks], axis=0)
    if np.any(duration[~active] > tolerance):
        raise InputContractError("inactive events must have zero duration.")
    totals = np.sum(duration, axis=0)
    if np.any(~np.isclose(totals, 1.0, rtol=0.0, atol=tolerance)):
        raise InputContractError("active event durations must fill each valid zone trace.")


__all__ = [
    "EventTrack",
    "SemiMarkovConditioning",
    "SemiMarkovPrior",
    "validate_event_track_order",
]
