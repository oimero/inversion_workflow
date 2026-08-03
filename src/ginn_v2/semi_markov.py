"""Conditional semi-Markov inference, sampling, and spatial random coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp, ndtr

from ginn_v2.contracts import NumericalFailure


@dataclass(frozen=True)
class SemiMarkovPrior:
    """Domain-neutral prior expressed in zone fractions."""

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
            raise ValueError("prior transition shape must match initial_probability.")
        if mean.shape != (states,) or std.shape != (states,):
            raise ValueError("duration statistics must have one value per state.")
        if (
            np.any(initial <= 0.0)
            or np.any(transition <= 0.0)
            or np.any(mean <= 0.0)
            or np.any(std <= 0.0)
        ):
            raise ValueError("semi-Markov probabilities and durations must be positive.")
        maximum = float(self.maximum_duration_fraction)
        if not np.isfinite(maximum) or not 0.0 < maximum <= 1.0:
            raise ValueError("maximum_duration_fraction must lie in (0, 1].")
        initial = initial / np.sum(initial)
        transition = transition / np.sum(transition, axis=1, keepdims=True)
        object.__setattr__(self, "initial_probability", initial)
        object.__setattr__(self, "transition_probability", transition)
        object.__setattr__(self, "duration_fraction_mean", mean)
        object.__setattr__(self, "duration_fraction_std", std)
        object.__setattr__(self, "maximum_duration_fraction", maximum)

    @classmethod
    def default(cls) -> "SemiMarkovPrior":
        return cls(
            initial_probability=np.full(3, 1.0 / 3.0),
            transition_probability=np.asarray(
                (
                    (0.20, 0.45, 0.35),
                    (0.40, 0.20, 0.40),
                    (0.35, 0.45, 0.20),
                ),
                dtype=np.float64,
            ),
            duration_fraction_mean=np.asarray((0.08, 0.10, 0.08)),
            duration_fraction_std=np.asarray((0.05, 0.06, 0.05)),
        )

    def duration_log_probability(self, sample_count: int) -> np.ndarray:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        maximum = max(
            1,
            min(
                sample_count,
                int(np.ceil(self.maximum_duration_fraction * sample_count)),
            ),
        )
        fraction = np.arange(1, maximum + 1, dtype=np.float64) / sample_count
        result = np.empty((self.initial_probability.size, maximum + 1), dtype=np.float64)
        result[:, 0] = -np.inf
        for state in range(self.initial_probability.size):
            z = (
                fraction - self.duration_fraction_mean[state]
            ) / self.duration_fraction_std[state]
            log_probability = -0.5 * z * z
            log_probability -= logsumexp(log_probability)
            result[state, 1:] = log_probability
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
            "maximum_duration_fraction": float(self.maximum_duration_fraction),
        }


@dataclass(frozen=True)
class SemiMarkovConditioning:
    """Frozen relative strengths for evidence, duration, and transition terms."""

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
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SemiMarkovConditioning":
        if value.get("schema") != "structured_ginn_v2_hsmm_conditioning_v1":
            raise ValueError("unsupported HSMM conditioning schema.")
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


def _conditioned_terms(
    state_probability: np.ndarray,
    renewal_probability: np.ndarray,
    prior: SemiMarkovPrior,
    conditioning: SemiMarkovConditioning,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    occupancy = np.asarray(state_probability, dtype=np.float64)
    activity = np.asarray(renewal_probability, dtype=np.float64).reshape(-1)
    if occupancy.ndim != 2 or occupancy.shape[0] != activity.size:
        raise ValueError("state_probability must be [sample, state].")
    if occupancy.shape[1] != prior.initial_probability.size:
        raise ValueError("state count differs between evidence and prior.")
    if np.any(~np.isfinite(occupancy)) or np.any(occupancy <= 0.0):
        raise ValueError("state occupancy must be finite and positive.")
    occupancy = occupancy / occupancy.sum(axis=1, keepdims=True)
    emission = conditioning.state_evidence_weight * np.log(occupancy)
    emission -= logsumexp(emission, axis=1, keepdims=True)
    activity = np.clip(activity, 1.0e-5, 1.0 - 1.0e-5)

    transition_log = (
        np.log(prior.transition_probability)
        / conditioning.transition_temperature
    )
    transition_log -= logsumexp(transition_log, axis=1, keepdims=True)
    transition = np.exp(transition_log)

    duration = prior.duration_log_probability(emission.shape[0])
    duration[:, 1:] /= conditioning.duration_temperature
    duration[:, 1:] -= logsumexp(duration[:, 1:], axis=1, keepdims=True)
    return emission, activity, transition, duration


@dataclass(frozen=True)
class SampledPath:
    segments: tuple[tuple[int, int, int], ...]
    state: np.ndarray
    log_score: float


@dataclass(frozen=True)
class SemiMarkovMarginals:
    state_probability: np.ndarray
    renewal_probability: np.ndarray
    same_state_renewal_probability: np.ndarray
    expected_segment_count: float


def _viterbi_path(
    *,
    emission_prefix: np.ndarray,
    boundary_probability: np.ndarray,
    initial_probability: np.ndarray,
    transition_probability: np.ndarray,
    duration_log_probability: np.ndarray,
    internal_no_boundary_prefix: np.ndarray,
) -> SampledPath:
    samples = int(emission_prefix.shape[0] - 1)
    states = int(emission_prefix.shape[1])
    delta = np.full((samples + 1, states), -np.inf, dtype=np.float64)
    previous_state = np.full((samples + 1, states), -1, dtype=np.int16)
    previous_start = np.full((samples + 1, states), -1, dtype=np.int32)
    transition_log = np.log(transition_probability)
    for stop in range(1, samples + 1):
        maximum = min(stop, duration_log_probability.shape[1] - 1)
        durations = np.arange(1, maximum + 1, dtype=np.int64)
        starts = stop - durations
        internal = (
            internal_no_boundary_prefix[stop]
            - internal_no_boundary_prefix[np.minimum(starts + 1, stop)]
        )
        renewal = np.zeros(maximum, dtype=np.float64)
        non_initial = starts > 0
        renewal[non_initial] = np.log(boundary_probability[starts[non_initial]])
        for state in range(states):
            segment_score = (
                emission_prefix[stop, state]
                - emission_prefix[starts, state]
                + internal
                + renewal
                + duration_log_probability[state, durations]
            )
            candidate_score = np.full(maximum, -np.inf, dtype=np.float64)
            candidate_previous = np.full(maximum, -1, dtype=np.int16)
            initial = starts == 0
            candidate_score[initial] = (
                np.log(initial_probability[state]) + segment_score[initial]
            )
            if np.any(non_initial):
                transitions = (
                    delta[starts[non_initial]]
                    + transition_log[:, state][None, :]
                )
                best_previous = np.argmax(transitions, axis=1)
                candidate_previous[non_initial] = best_previous
                candidate_score[non_initial] = (
                    transitions[np.arange(best_previous.size), best_previous]
                    + segment_score[non_initial]
                )
            best = int(np.argmax(candidate_score))
            delta[stop, state] = candidate_score[best]
            previous_state[stop, state] = candidate_previous[best]
            previous_start[stop, state] = starts[best]

    state = int(np.argmax(delta[samples]))
    score = float(delta[samples, state])
    if not np.isfinite(score):
        raise NumericalFailure("semi-Markov MAP path is non-finite.")
    stop = samples
    reversed_segments: list[tuple[int, int, int]] = []
    while stop > 0:
        start = int(previous_start[stop, state])
        if start < 0 or start >= stop:
            raise NumericalFailure("semi-Markov MAP backpointer is invalid.")
        reversed_segments.append((state, start, stop))
        state = int(previous_state[stop, state])
        stop = start
        if stop > 0 and state < 0:
            raise NumericalFailure("semi-Markov MAP path ended above the zone top.")
    segments = tuple(reversed(reversed_segments))
    state_values = np.full(samples, -1, dtype=np.int8)
    for state_id, start, stop in segments:
        state_values[start:stop] = int(state_id)
    if np.any(state_values < 0):
        raise NumericalFailure("semi-Markov MAP path does not cover every sample.")
    return SampledPath(segments=segments, state=state_values, log_score=score)


@dataclass
class SemiMarkovPosterior:
    emission_log_probability: np.ndarray
    boundary_probability: np.ndarray
    prior: SemiMarkovPrior
    transition_probability: np.ndarray
    duration_log_probability: np.ndarray
    alpha: np.ndarray
    emission_prefix: np.ndarray
    internal_no_boundary_prefix: np.ndarray
    log_evidence: float
    forward_recursions: int = 1

    @property
    def sample_count(self) -> int:
        return int(self.emission_log_probability.shape[0])

    @property
    def state_count(self) -> int:
        return int(self.emission_log_probability.shape[1])

    def _segment_score(self, state: int, start: int, stop: int) -> float:
        emission = (
            self.emission_prefix[stop, state]
            - self.emission_prefix[start, state]
        )
        duration = self.duration_log_probability[state, stop - start]
        internal = (
            self.internal_no_boundary_prefix[stop]
            - self.internal_no_boundary_prefix[min(start + 1, stop)]
        )
        renewal = (
            0.0
            if start == 0
            else float(np.log(self.boundary_probability[start]))
        )
        return float(emission + duration + internal + renewal)

    def _segment_score_vector(
        self,
        state: int,
        start: np.ndarray,
        stop: np.ndarray,
    ) -> np.ndarray:
        starts = np.asarray(start, dtype=np.int64)
        stops = np.asarray(stop, dtype=np.int64)
        if starts.shape != stops.shape:
            raise ValueError("segment start/stop vectors must share one shape.")
        emission = (
            self.emission_prefix[stops, state]
            - self.emission_prefix[starts, state]
        )
        duration = self.duration_log_probability[state, stops - starts]
        internal = (
            self.internal_no_boundary_prefix[stops]
            - self.internal_no_boundary_prefix[np.minimum(starts + 1, stops)]
        )
        renewal = np.zeros(starts.shape, dtype=np.float64)
        non_initial = starts > 0
        renewal[non_initial] = np.log(
            self.boundary_probability[starts[non_initial]]
        )
        return emission + duration + internal + renewal

    @staticmethod
    def _draw(log_weight: np.ndarray, uniform: float) -> int:
        finite = np.isfinite(log_weight)
        if not np.any(finite):
            raise NumericalFailure("semi-Markov sampling has no finite candidate.")
        normalized = np.zeros(log_weight.size, dtype=np.float64)
        normalized[finite] = np.exp(log_weight[finite] - logsumexp(log_weight[finite]))
        cumulative = np.cumsum(normalized)
        index = int(
            np.searchsorted(
                cumulative,
                min(max(float(uniform), 0.0), 1.0 - 1.0e-15),
            )
        )
        return min(index, int(log_weight.size) - 1)

    @staticmethod
    def _state_path(
        segments: tuple[tuple[int, int, int], ...],
        sample_count: int,
    ) -> np.ndarray:
        state_values = np.full(sample_count, -1, dtype=np.int8)
        for state_id, start, stop in segments:
            state_values[start:stop] = int(state_id)
        if np.any(state_values < 0):
            raise NumericalFailure("semi-Markov path does not cover every sample.")
        return state_values

    def map_path(self) -> SampledPath:
        """Return the globally optimal complete path under this posterior."""
        return _viterbi_path(
            emission_prefix=self.emission_prefix,
            boundary_probability=self.boundary_probability,
            initial_probability=self.prior.initial_probability,
            transition_probability=self.transition_probability,
            duration_log_probability=self.duration_log_probability,
            internal_no_boundary_prefix=self.internal_no_boundary_prefix,
        )

    def marginals(self) -> SemiMarkovMarginals:
        """Compute exact state and renewal marginals from the forward table."""

        samples = self.sample_count
        states = self.state_count
        transition_log = np.log(self.transition_probability)
        beta = np.full((samples + 1, states), -np.inf, dtype=np.float64)
        beta[samples] = 0.0
        for start in range(samples - 1, 0, -1):
            maximum = min(
                samples - start,
                self.duration_log_probability.shape[1] - 1,
            )
            stops = start + np.arange(1, maximum + 1, dtype=np.int64)
            starts = np.full(stops.shape, start, dtype=np.int64)
            future = np.empty((maximum, states), dtype=np.float64)
            for state in range(states):
                future[:, state] = (
                    self._segment_score_vector(state, starts, stops)
                    + beta[stops, state]
                )
            beta[start] = logsumexp(
                future[:, None, :] + transition_log[None, :, :],
                axis=(0, 2),
            )

        state_difference = np.zeros((samples + 1, states), dtype=np.float64)
        renewal = np.zeros(samples, dtype=np.float64)
        same_state_renewal = np.zeros(samples, dtype=np.float64)
        maximum_duration = self.duration_log_probability.shape[1] - 1
        for start in range(samples):
            maximum = min(samples - start, maximum_duration)
            stops = start + np.arange(1, maximum + 1, dtype=np.int64)
            starts = np.full(stops.shape, start, dtype=np.int64)
            for state in range(states):
                segment = self._segment_score_vector(state, starts, stops)
                suffix = beta[stops, state]
                if start == 0:
                    probability = np.exp(
                        np.log(self.prior.initial_probability[state])
                        + segment
                        + suffix
                        - self.log_evidence
                    )
                else:
                    log_weights = (
                        self.alpha[start][None, :]
                        + transition_log[:, state][None, :]
                        + segment[:, None]
                        + suffix[:, None]
                        - self.log_evidence
                    )
                    transition_probability = np.exp(log_weights)
                    probability = np.sum(transition_probability, axis=1)
                    renewal[start] += float(np.sum(probability))
                    same_state_renewal[start] += float(
                        np.sum(transition_probability[:, state])
                    )
                state_difference[start, state] += float(np.sum(probability))
                np.add.at(state_difference[:, state], stops, -probability)

        state_probability = np.cumsum(state_difference[:-1], axis=0)
        state_probability = np.clip(state_probability, 0.0, 1.0)
        normalizer = np.sum(state_probability, axis=1, keepdims=True)
        if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
            raise NumericalFailure("semi-Markov state marginals are non-finite.")
        state_probability /= normalizer
        renewal[0] = 1.0
        same_state_renewal[0] = 0.0
        renewal = np.clip(renewal, 0.0, 1.0)
        same_state_renewal = np.clip(same_state_renewal, 0.0, renewal)
        return SemiMarkovMarginals(
            state_probability=state_probability,
            renewal_probability=renewal,
            same_state_renewal_probability=same_state_renewal,
            expected_segment_count=float(1.0 + np.sum(renewal[1:])),
        )

    def sample(self, uniforms: Sequence[float]) -> SampledPath:
        random_values = np.asarray(uniforms, dtype=np.float64).reshape(-1)
        if random_values.size < self.sample_count + 1:
            raise ValueError("sampling requires at least sample_count + 1 uniforms.")
        final_state = self._draw(self.alpha[self.sample_count], random_values[0])
        stop = self.sample_count
        state = final_state
        reversed_segments: list[tuple[int, int, int]] = []
        cursor = 1
        while stop > 0:
            maximum = min(stop, self.duration_log_probability.shape[1] - 1)
            labels: list[tuple[int, int | None]] = []
            scores: list[float] = []
            for duration in range(1, maximum + 1):
                start = stop - duration
                segment_score = self._segment_score(state, start, stop)
                if start == 0:
                    labels.append((duration, None))
                    scores.append(
                        float(np.log(self.prior.initial_probability[state]) + segment_score)
                    )
                else:
                    for previous in range(self.state_count):
                        labels.append((duration, previous))
                        scores.append(
                            float(
                                self.alpha[start, previous]
                                + np.log(self.transition_probability[previous, state])
                                + segment_score
                            )
                        )
            choice = self._draw(np.asarray(scores), random_values[cursor])
            cursor += 1
            duration, previous = labels[choice]
            start = stop - duration
            reversed_segments.append((state, start, stop))
            stop = start
            if previous is None:
                break
            state = previous
        if stop != 0:
            raise NumericalFailure("sampled semi-Markov path does not reach the zone top.")
        segments = tuple(reversed(reversed_segments))
        state_values = self._state_path(segments, self.sample_count)
        total_score = 0.0
        for state_id, start, stop in segments:
            segment_score = self._segment_score(state_id, start, stop)
            if start == 0:
                total_score += float(
                    np.log(self.prior.initial_probability[state_id]) + segment_score
                )
            else:
                previous_state = int(state_values[start - 1])
                total_score += float(
                    np.log(self.transition_probability[previous_state, state_id])
                    + segment_score
                )
        return SampledPath(
            segments=segments,
            state=state_values,
            log_score=float(total_score),
        )


def viterbi_semi_markov_path(
    state_probability: np.ndarray,
    renewal_probability: np.ndarray,
    prior: SemiMarkovPrior,
    conditioning: SemiMarkovConditioning = SemiMarkovConditioning(),
) -> SampledPath:
    """Return the conditioned MAP path without running posterior recursion."""

    emission, activity, transition, duration = _conditioned_terms(
        state_probability,
        renewal_probability,
        prior,
        conditioning,
    )
    states = emission.shape[1]
    emission_prefix = np.vstack(
        (np.zeros((1, states), dtype=np.float64), np.cumsum(emission, axis=0))
    )
    no_boundary = np.log1p(-activity)
    no_boundary[0] = 0.0
    internal_prefix = np.r_[0.0, np.cumsum(no_boundary)]
    return _viterbi_path(
        emission_prefix=emission_prefix,
        boundary_probability=activity,
        initial_probability=prior.initial_probability,
        transition_probability=transition,
        duration_log_probability=duration,
        internal_no_boundary_prefix=internal_prefix,
    )


def exact_semi_markov_posterior(
    state_probability: np.ndarray,
    renewal_probability: np.ndarray,
    prior: SemiMarkovPrior,
    conditioning: SemiMarkovConditioning = SemiMarkovConditioning(),
) -> SemiMarkovPosterior:
    emission, activity, transition, duration = _conditioned_terms(
        state_probability,
        renewal_probability,
        prior,
        conditioning,
    )
    samples, states = emission.shape
    emission_prefix = np.vstack(
        (np.zeros((1, states), dtype=np.float64), np.cumsum(emission, axis=0))
    )
    no_boundary = np.log1p(-activity)
    no_boundary[0] = 0.0
    internal_prefix = np.r_[0.0, np.cumsum(no_boundary)]
    alpha = np.full((samples + 1, states), -np.inf, dtype=np.float64)
    transition_log = np.log(transition)

    for stop in range(1, samples + 1):
        maximum = min(stop, duration.shape[1] - 1)
        durations = np.arange(1, maximum + 1, dtype=np.int64)
        starts = stop - durations
        internal = (
            internal_prefix[stop]
            - internal_prefix[np.minimum(starts + 1, stop)]
        )
        renewal = np.zeros(maximum, dtype=np.float64)
        non_initial = starts > 0
        renewal[non_initial] = np.log(activity[starts[non_initial]])
        for state in range(states):
            segment_score = (
                emission_prefix[stop, state]
                - emission_prefix[starts, state]
                + internal
                + renewal
                + duration[state, durations]
            )
            prefix = np.full(maximum, -np.inf, dtype=np.float64)
            initial = starts == 0
            prefix[initial] = np.log(prior.initial_probability[state])
            if np.any(non_initial):
                prefix[non_initial] = logsumexp(
                    alpha[starts[non_initial]]
                    + transition_log[:, state][None, :],
                    axis=1,
                )
            alpha[stop, state] = logsumexp(prefix + segment_score)
    evidence = float(logsumexp(alpha[samples]))
    if not np.isfinite(evidence):
        raise NumericalFailure("semi-Markov forward recursion is non-finite.")
    return SemiMarkovPosterior(
        emission_log_probability=emission,
        boundary_probability=activity,
        prior=prior,
        transition_probability=transition,
        duration_log_probability=duration,
        alpha=alpha,
        emission_prefix=emission_prefix,
        internal_no_boundary_prefix=internal_prefix,
        log_evidence=evidence,
    )


def coordinate_stable_uniforms(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    realization_count: int,
    draw_count: int,
    random_identity: int,
    correlation_length_m: float,
    modes: int = 24,
) -> np.ndarray:
    """Evaluate deterministic spatial random Fourier fields at survey coordinates."""
    x = np.asarray(x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(y_m, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("x_m and y_m must be finite and share one shape.")
    if realization_count <= 0 or draw_count <= 0 or modes <= 0:
        raise ValueError("random field dimensions must be positive.")
    if not np.isfinite(correlation_length_m) or correlation_length_m <= 0.0:
        raise ValueError("correlation_length_m must be positive.")
    output = np.empty((realization_count, x.size, draw_count), dtype=np.float64)
    coordinates = np.column_stack((x, y))
    for realization in range(realization_count):
        for draw in range(draw_count):
            sequence = np.random.SeedSequence(
                [int(random_identity), int(realization), int(draw)]
            )
            rng = np.random.default_rng(sequence)
            frequency = rng.normal(
                0.0,
                1.0 / correlation_length_m,
                size=(modes, 2),
            )
            phase = rng.uniform(0.0, 2.0 * np.pi, size=modes)
            projection = coordinates @ frequency.T + phase
            field = np.sqrt(2.0 / modes) * np.sum(np.cos(projection), axis=1)
            output[realization, :, draw] = np.clip(
                ndtr(field),
                1.0e-12,
                1.0 - 1.0e-12,
            )
    return output


__all__ = [
    "SampledPath",
    "SemiMarkovConditioning",
    "SemiMarkovMarginals",
    "SemiMarkovPosterior",
    "SemiMarkovPrior",
    "coordinate_stable_uniforms",
    "exact_semi_markov_posterior",
    "viterbi_semi_markov_path",
]
