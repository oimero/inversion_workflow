"""Conditional semi-Markov inference, sampling, and spatial random coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
        initial = initial / np.sum(initial)
        transition = transition / np.sum(transition, axis=1, keepdims=True)
        object.__setattr__(self, "initial_probability", initial)
        object.__setattr__(self, "transition_probability", transition)
        object.__setattr__(self, "duration_fraction_mean", mean)
        object.__setattr__(self, "duration_fraction_std", std)

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


@dataclass(frozen=True)
class SampledPath:
    segments: tuple[tuple[int, int, int], ...]
    state: np.ndarray
    log_score: float


@dataclass
class SemiMarkovPosterior:
    emission_log_probability: np.ndarray
    boundary_probability: np.ndarray
    prior: SemiMarkovPrior
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

    @staticmethod
    def _draw(log_weight: np.ndarray, uniform: float) -> int:
        finite = np.isfinite(log_weight)
        if not np.any(finite):
            raise NumericalFailure("semi-Markov sampling has no finite candidate.")
        normalized = np.zeros(log_weight.size, dtype=np.float64)
        normalized[finite] = np.exp(log_weight[finite] - logsumexp(log_weight[finite]))
        cumulative = np.cumsum(normalized)
        return int(np.searchsorted(cumulative, min(max(float(uniform), 0.0), 1.0 - 1.0e-15)))

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
                                + np.log(self.prior.transition_probability[previous, state])
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
        state_values = np.full(self.sample_count, -1, dtype=np.int8)
        total_score = 0.0
        for state_id, start, stop in segments:
            state_values[start:stop] = int(state_id)
            segment_score = self._segment_score(state_id, start, stop)
            if start == 0:
                total_score += float(
                    np.log(self.prior.initial_probability[state_id]) + segment_score
                )
            else:
                previous_state = int(state_values[start - 1])
                total_score += float(
                    np.log(self.prior.transition_probability[previous_state, state_id])
                    + segment_score
                )
        return SampledPath(
            segments=segments,
            state=state_values,
            log_score=float(total_score),
        )


def exact_semi_markov_posterior(
    state_occupancy: np.ndarray,
    interface_activity: np.ndarray,
    prior: SemiMarkovPrior,
) -> SemiMarkovPosterior:
    occupancy = np.asarray(state_occupancy, dtype=np.float64)
    activity = np.asarray(interface_activity, dtype=np.float64).reshape(-1)
    if occupancy.ndim != 2 or occupancy.shape[0] != activity.size:
        raise ValueError("state_occupancy must be [sample, state].")
    if occupancy.shape[1] != prior.initial_probability.size:
        raise ValueError("state count differs between evidence and prior.")
    if np.any(~np.isfinite(occupancy)) or np.any(occupancy <= 0.0):
        raise ValueError("state occupancy must be finite and positive.")
    occupancy = occupancy / occupancy.sum(axis=1, keepdims=True)
    activity = np.clip(activity, 1.0e-5, 1.0 - 1.0e-5)
    emission = np.log(occupancy)
    samples, states = emission.shape
    duration = prior.duration_log_probability(samples)
    emission_prefix = np.vstack(
        (np.zeros((1, states), dtype=np.float64), np.cumsum(emission, axis=0))
    )
    no_boundary = np.log1p(-activity)
    no_boundary[0] = 0.0
    internal_prefix = np.r_[0.0, np.cumsum(no_boundary)]
    alpha = np.full((samples + 1, states), -np.inf, dtype=np.float64)

    def segment_score(state: int, start: int, stop: int) -> float:
        emission_score = emission_prefix[stop, state] - emission_prefix[start, state]
        internal = internal_prefix[stop] - internal_prefix[min(start + 1, stop)]
        renewal = 0.0 if start == 0 else np.log(activity[start])
        return float(emission_score + internal + renewal + duration[state, stop - start])

    for stop in range(1, samples + 1):
        maximum = min(stop, duration.shape[1] - 1)
        for state in range(states):
            candidates: list[float] = []
            for segment_duration in range(1, maximum + 1):
                start = stop - segment_duration
                score = segment_score(state, start, stop)
                if start == 0:
                    candidates.append(float(np.log(prior.initial_probability[state]) + score))
                else:
                    candidates.append(
                        float(
                            logsumexp(
                                alpha[start]
                                + np.log(prior.transition_probability[:, state])
                            )
                            + score
                        )
                    )
            alpha[stop, state] = logsumexp(candidates)
    evidence = float(logsumexp(alpha[samples]))
    if not np.isfinite(evidence):
        raise NumericalFailure("semi-Markov forward recursion is non-finite.")
    return SemiMarkovPosterior(
        emission_log_probability=emission,
        boundary_probability=activity,
        prior=prior,
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
    "SemiMarkovPosterior",
    "SemiMarkovPrior",
    "coordinate_stable_uniforms",
    "exact_semi_markov_posterior",
]
