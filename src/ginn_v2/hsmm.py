"""Exact three-state HSMM inference with zone-fraction duration priors.

The public seam intentionally accepts only one contiguous zone at a time.  It
hides duration discretization, dynamic programming, MAP backtracking, and
posterior marginal construction from model and training callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cup.synthetic.readers.structured import StructuredSyntheticBenchmark
from ginn_v2.data import ParentSplitManifest


HSMM_PRIOR_SCHEMA = "structured_ginn_v2_hsmm_prior_v1"
STATE_COUNT = 3


def _json_log_array(values: np.ndarray) -> list[Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return [float(value) if np.isfinite(value) else None for value in array]
    return [_json_log_array(row) for row in array]


def _log_array_from_json(values: Any) -> np.ndarray:
    def restore(item: Any) -> Any:
        if isinstance(item, list):
            return [restore(value) for value in item]
        return -np.inf if item is None else float(item)

    return np.asarray(restore(values), dtype=np.float64)


def _normalized_log_probabilities(counts: np.ndarray, smoothing: float) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("HSMM prior counts must be finite and non-negative.")
    values = values + float(smoothing)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("HSMM prior has no probability mass.")
    return np.log(values / total)


@dataclass(frozen=True)
class ZoneHsmmPrior:
    """One zone's fixed initial, transition, and duration distributions."""

    zone_id: str
    initial_log_probability: np.ndarray
    transition_log_probability: np.ndarray
    duration_fraction_edges: np.ndarray
    duration_log_probability: np.ndarray

    def __post_init__(self) -> None:
        initial = np.asarray(self.initial_log_probability, dtype=np.float64)
        transition = np.asarray(self.transition_log_probability, dtype=np.float64)
        edges = np.asarray(self.duration_fraction_edges, dtype=np.float64)
        duration = np.asarray(self.duration_log_probability, dtype=np.float64)
        if initial.shape != (STATE_COUNT,):
            raise ValueError("initial_log_probability must have shape [3].")
        if transition.shape != (STATE_COUNT, STATE_COUNT):
            raise ValueError("transition_log_probability must have shape [3, 3].")
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("duration_fraction_edges must contain at least two edges.")
        if duration.shape != (STATE_COUNT, edges.size - 1):
            raise ValueError("duration_log_probability has an invalid shape.")
        if not np.all(np.diff(edges) > 0.0) or not np.isclose(edges[0], 0.0):
            raise ValueError("duration fraction edges must increase from zero.")
        if not np.isclose(edges[-1], 1.0):
            raise ValueError("duration fraction edges must end at one.")
        if np.any(np.isfinite(np.diag(transition))):
            raise ValueError("HSMM self-transition probabilities must be -inf.")
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial log probabilities must be finite.")
        if np.any(np.isnan(duration)) or np.any(np.isposinf(duration)):
            raise ValueError("duration log probabilities contain invalid values.")
        if np.any(np.sum(np.isfinite(duration), axis=1) == 0):
            raise ValueError("every state needs at least one legal duration bin.")
        off_diagonal = transition[~np.eye(STATE_COUNT, dtype=bool)]
        if not np.all(np.isfinite(off_diagonal)):
            raise ValueError("off-diagonal transition probabilities must be finite.")
        object.__setattr__(self, "zone_id", str(self.zone_id))
        object.__setattr__(self, "initial_log_probability", initial)
        object.__setattr__(self, "transition_log_probability", transition)
        object.__setattr__(self, "duration_fraction_edges", edges)
        object.__setattr__(self, "duration_log_probability", duration)

    def duration_scores(
        self,
        sample_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Discretize the fraction prior over all legal durations for this zone."""
        if int(sample_count) <= 0:
            raise ValueError("sample_count must be positive.")
        durations = np.arange(1, int(sample_count) + 1, dtype=np.float64)
        fractions = durations / float(sample_count)
        bins = np.searchsorted(
            self.duration_fraction_edges,
            fractions,
            side="right",
        ) - 1
        bins = np.clip(bins, 0, self.duration_log_probability.shape[1] - 1)
        scores = self.duration_log_probability[:, bins]
        # The histogram stores fraction density.  Renormalizing over legal
        # integer durations makes each sampled zone a proper discrete prior.
        scores = scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)
        return torch.as_tensor(scores, dtype=dtype, device=device)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "initial_log_probability": _json_log_array(
                self.initial_log_probability
            ),
            "transition_log_probability": _json_log_array(
                self.transition_log_probability
            ),
            "duration_fraction_edges": self.duration_fraction_edges.tolist(),
            "duration_log_probability": _json_log_array(
                self.duration_log_probability
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ZoneHsmmPrior":
        required = {
            "zone_id",
            "initial_log_probability",
            "transition_log_probability",
            "duration_fraction_edges",
            "duration_log_probability",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(required))
        if missing or unknown:
            raise ValueError(
                f"zone HSMM prior mismatch; missing={missing}, unknown={unknown}"
            )
        return cls(
            zone_id=str(value["zone_id"]),
            initial_log_probability=_log_array_from_json(
                value["initial_log_probability"]
            ),
            transition_log_probability=_log_array_from_json(
                value["transition_log_probability"]
            ),
            duration_fraction_edges=np.asarray(
                value["duration_fraction_edges"], dtype=np.float64
            ),
            duration_log_probability=_log_array_from_json(
                value["duration_log_probability"]
            ),
        )


@dataclass(frozen=True)
class HsmmPrior:
    """Frozen priors calibrated only from development-training parents."""

    zones: Mapping[str, ZoneHsmmPrior]
    duration_bin_count: int
    smoothing: float
    split_manifest_fingerprint: str

    def __post_init__(self) -> None:
        zones = dict(self.zones)
        if not zones or any(key != value.zone_id for key, value in zones.items()):
            raise ValueError("HSMM prior zones are empty or inconsistently keyed.")
        if int(self.duration_bin_count) <= 1:
            raise ValueError("duration_bin_count must be greater than one.")
        if not np.isfinite(self.smoothing) or self.smoothing <= 0.0:
            raise ValueError("HSMM prior smoothing must be positive.")
        object.__setattr__(self, "zones", zones)
        object.__setattr__(self, "duration_bin_count", int(self.duration_bin_count))
        object.__setattr__(self, "smoothing", float(self.smoothing))
        object.__setattr__(
            self,
            "split_manifest_fingerprint",
            str(self.split_manifest_fingerprint),
        )

    def zone(self, zone_id: str) -> ZoneHsmmPrior:
        try:
            return self.zones[str(zone_id)]
        except KeyError as error:
            raise KeyError(f"HSMM prior has no zone {zone_id!r}.") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HSMM_PRIOR_SCHEMA,
            "duration_bin_count": self.duration_bin_count,
            "smoothing": self.smoothing,
            "split_manifest_fingerprint": self.split_manifest_fingerprint,
            "zones": [
                self.zones[name].to_dict() for name in sorted(self.zones)
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HsmmPrior":
        if value.get("schema") != HSMM_PRIOR_SCHEMA:
            raise ValueError("unsupported Structured GINN V2 HSMM prior.")
        required = {
            "schema",
            "duration_bin_count",
            "smoothing",
            "split_manifest_fingerprint",
            "zones",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(required))
        if missing or unknown:
            raise ValueError(
                f"HSMM prior mismatch; missing={missing}, unknown={unknown}"
            )
        zone_items = [ZoneHsmmPrior.from_dict(item) for item in value["zones"]]
        return cls(
            zones={item.zone_id: item for item in zone_items},
            duration_bin_count=int(value["duration_bin_count"]),
            smoothing=float(value["smoothing"]),
            split_manifest_fingerprint=str(value["split_manifest_fingerprint"]),
        )


def fit_hsmm_prior(
    benchmark: StructuredSyntheticBenchmark,
    split_manifest: ParentSplitManifest,
    *,
    duration_bin_count: int = 64,
    smoothing: float = 1.0,
) -> HsmmPrior:
    """Fit zone-fraction priors from every development-training trace."""
    if int(duration_bin_count) <= 1:
        raise ValueError("duration_bin_count must be greater than one.")
    grouped: dict[str, dict[str, np.ndarray]] = {}
    edges = np.linspace(0.0, 1.0, int(duration_bin_count) + 1)
    logger = logging.getLogger("ginn_v2")
    parent_count = len(split_manifest.training)
    for parent_index, parent_id in enumerate(split_manifest.training, start=1):
        parent = benchmark.read_parent(parent_id)
        sequences: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
        for row in parent.segments:
            if (
                int(row["duration_samples"]) <= 0
                or not bool(row["segment_supervision_valid"])
            ):
                continue
            key = (int(row["lateral_index"]), str(row["zone_id"]))
            sequences.setdefault(key, []).append(row)
        for (_, zone_id), rows in sequences.items():
            rows.sort(key=lambda item: (float(item["top"]), int(item["object_id"])))
            if zone_id not in grouped:
                grouped[zone_id] = {
                    "initial": np.zeros(STATE_COUNT, dtype=np.float64),
                    "transition": np.zeros(
                        (STATE_COUNT, STATE_COUNT), dtype=np.float64
                    ),
                    "duration": np.zeros(
                        (STATE_COUNT, int(duration_bin_count)), dtype=np.float64
                    ),
                }
            counts = grouped[zone_id]
            canonical_rows = _canonicalize_duration_rows(rows)
            states = [state for state, _, _, _ in canonical_rows]
            if not states:
                continue
            counts["initial"][states[0]] += 1.0
            for left, right in zip(states[:-1], states[1:], strict=True):
                if left == right:
                    raise ValueError(
                        f"producer emitted non-contiguous repeated state in zone {zone_id!r}."
                    )
                counts["transition"][left, right] += 1.0
            for state, fraction, _, _ in canonical_rows:
                if not 0.0 < fraction <= 1.0:
                    raise ValueError("training duration fraction lies outside (0, 1].")
                index = min(
                    int(np.searchsorted(edges, fraction, side="right") - 1),
                    int(duration_bin_count) - 1,
                )
                counts["duration"][state, index] += 1.0
        if parent_index % 25 == 0 or parent_index == parent_count:
            logger.info(
                "HSMM prior calibration | parents=%d/%d",
                parent_index,
                parent_count,
            )
    if not grouped:
        raise ValueError("training parents produced no HSMM prior observations.")
    zone_priors: dict[str, ZoneHsmmPrior] = {}
    for zone_id, counts in grouped.items():
        transition = np.full(
            (STATE_COUNT, STATE_COUNT), -np.inf, dtype=np.float64
        )
        for state in range(STATE_COUNT):
            allowed = np.arange(STATE_COUNT) != state
            transition[state, allowed] = _normalized_log_probabilities(
                counts["transition"][state, allowed],
                smoothing,
            )
        duration_rows: list[np.ndarray] = []
        for state in range(STATE_COUNT):
            observed = counts["duration"][state]
            occupied = np.flatnonzero(observed > 0.0)
            if occupied.size == 0:
                raise ValueError(
                    f"state {state} has no duration observations in zone {zone_id!r}."
                )
            lower = max(int(occupied[0]) - 1, 0)
            upper = min(int(occupied[-1]) + 2, int(duration_bin_count))
            row = np.full(int(duration_bin_count), -np.inf, dtype=np.float64)
            row[lower:upper] = _normalized_log_probabilities(
                observed[lower:upper],
                smoothing,
            )
            duration_rows.append(row)
        duration = np.stack(duration_rows, axis=0)
        zone_priors[zone_id] = ZoneHsmmPrior(
            zone_id=zone_id,
            initial_log_probability=_normalized_log_probabilities(
                counts["initial"], smoothing
            ),
            transition_log_probability=transition,
            duration_fraction_edges=edges,
            duration_log_probability=duration,
        )
    return HsmmPrior(
        zones=zone_priors,
        duration_bin_count=int(duration_bin_count),
        smoothing=float(smoothing),
        split_manifest_fingerprint=str(
            split_manifest.to_dict()["fingerprint_sha256"]
        ),
    )


def freeze_hsmm_prior(
    benchmark: StructuredSyntheticBenchmark,
    split_manifest: ParentSplitManifest,
    path: str | Path,
    *,
    duration_bin_count: int = 64,
    smoothing: float = 1.0,
) -> HsmmPrior:
    """Publish an immutable prior, accepting only an identical calibration."""
    target = Path(path)
    proposed = fit_hsmm_prior(
        benchmark,
        split_manifest,
        duration_bin_count=duration_bin_count,
        smoothing=smoothing,
    )
    if target.exists():
        existing = HsmmPrior.from_dict(json.loads(target.read_text(encoding="utf-8")))
        if existing.to_dict() != proposed.to_dict():
            raise FileExistsError(
                f"HSMM prior already exists with different content: {target}"
            )
        return existing
    temporary = target.with_name(f".{target.name}.staging")
    temporary.write_text(
        json.dumps(
            proposed.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    return proposed


@dataclass(frozen=True)
class HsmmSegment:
    """Half-open segment extent in a zone-local sample sequence."""

    state_id: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if int(self.state_id) not in range(STATE_COUNT):
            raise ValueError("HSMM segment state_id must be 0, 1, or 2.")
        if int(self.start) < 0 or int(self.stop) <= int(self.start):
            raise ValueError("HSMM segment extent must be non-empty and half-open.")

    @property
    def duration_samples(self) -> int:
        return int(self.stop) - int(self.start)


def canonicalize_hsmm_segments(
    segments: Sequence[HsmmSegment],
) -> tuple[HsmmSegment, ...]:
    """Merge contiguous equal-state runs into the HSMM state-path form.

    Structured event rows may contain two adjacent objects with the same
    producer state when an event between them pinches out on the current
    trace.  The event rows remain separate for profile supervision and
    lateral identity, but an HSMM state path cannot represent that event
    boundary.  This helper keeps the state path canonical without changing
    the artifact's event table.
    """
    if not segments:
        raise ValueError("HSMM segment path cannot be empty.")
    canonical: list[HsmmSegment] = []
    for segment in segments:
        if canonical:
            previous = canonical[-1]
            if int(segment.start) != int(previous.stop):
                raise ValueError("HSMM segment path must be contiguous.")
            if int(segment.state_id) == int(previous.state_id):
                canonical[-1] = HsmmSegment(
                    state_id=previous.state_id,
                    start=previous.start,
                    stop=segment.stop,
                )
                continue
        canonical.append(segment)
    return tuple(canonical)


def _canonicalize_duration_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, float, float, float], ...]:
    """Canonicalize contiguous same-state prior observations.

    ``duration_fraction`` is a zone fraction, so merging two adjacent
    same-state event rows adds their fractions.  Non-contiguous repeated
    states remain visible to the caller and are rejected as an invalid
    state-path topology.
    """
    canonical: list[tuple[int, float, float, float]] = []
    for row in rows:
        state = int(row["state_id"])
        fraction = float(row["duration_fraction"])
        top = float(row["top"])
        bottom = float(row["bottom"])
        if not 0.0 < fraction <= 1.0:
            raise ValueError("training duration fraction lies outside (0, 1].")
        if not np.isfinite(top) or not np.isfinite(bottom) or bottom <= top:
            raise ValueError("training duration endpoints must be finite and increasing.")
        if canonical:
            previous_state, previous_fraction, previous_top, previous_bottom = canonical[-1]
            if (
                state == previous_state
                and np.isclose(previous_bottom, top, rtol=1e-10, atol=1e-8)
            ):
                canonical[-1] = (
                    previous_state,
                    previous_fraction + fraction,
                    previous_top,
                    bottom,
                )
                continue
        canonical.append((state, fraction, top, bottom))
    return tuple(canonical)


@dataclass(frozen=True)
class HsmmResult:
    log_partition: torch.Tensor
    map_score: torch.Tensor
    map_segments: tuple[HsmmSegment, ...]
    consensus_segments: tuple[HsmmSegment, ...]
    state_marginal: torch.Tensor
    boundary_marginal: torch.Tensor


def posterior_consensus_segments(
    emission_log_potential: torch.Tensor,
    boundary_log_potential: torch.Tensor,
    state_marginal: torch.Tensor,
    boundary_marginal: torch.Tensor,
    prior: ZoneHsmmPrior,
    *,
    reward_bound: float = 8.0,
    search_steps: int = 12,
) -> tuple[HsmmSegment, ...]:
    """Decode a legal path whose count is closest to the posterior expectation.

    This is the Lagrangian form of count-constrained Viterbi decoding.  A
    per-segment reward is searched until joint MAP count matches the exact
    posterior expected count as closely as the discrete path family permits.
    """
    if (
        state_marginal.ndim != 2
        or state_marginal.shape[1] != STATE_COUNT
        or boundary_marginal.shape != state_marginal.shape[:1]
    ):
        raise ValueError("posterior consensus received invalid marginal shapes.")
    _validate_potentials(emission_log_potential, boundary_log_potential)
    if emission_log_potential.shape != state_marginal.shape:
        raise ValueError("posterior consensus potential and marginal shapes differ.")
    if not np.isfinite(reward_bound) or float(reward_bound) <= 0.0:
        raise ValueError("reward_bound must be positive.")
    if int(search_steps) <= 0:
        raise ValueError("search_steps must be positive.")
    sample_count = int(state_marginal.shape[0])
    target_count = max(
        1,
        min(sample_count, int(np.rint(float(torch.sum(boundary_marginal).item())))),
    )
    duration = prior.duration_scores(
        sample_count,
        device=state_marginal.device,
        dtype=state_marginal.dtype,
    )
    initial = torch.as_tensor(
        prior.initial_log_probability,
        dtype=state_marginal.dtype,
        device=state_marginal.device,
    )
    transition = torch.as_tensor(
        prior.transition_log_probability,
        dtype=state_marginal.dtype,
        device=state_marginal.device,
    )
    emission_prefix = torch.cat(
        (
            torch.zeros(
                (1, STATE_COUNT),
                dtype=emission_log_potential.dtype,
                device=emission_log_potential.device,
            ),
            torch.cumsum(emission_log_potential, dim=0),
        ),
        dim=0,
    )
    candidates: list[tuple[int, float, tuple[HsmmSegment, ...]]] = []

    def evaluate(reward: float) -> tuple[HsmmSegment, ...]:
        _, segments = _viterbi(
            emission_prefix,
            boundary_log_potential,
            initial,
            transition,
            duration,
            segment_reward=float(reward),
        )
        candidates.append((abs(len(segments) - target_count), abs(reward), segments))
        return segments

    lower = -float(reward_bound)
    upper = float(reward_bound)
    evaluate(lower)
    evaluate(upper)
    for _ in range(int(search_steps)):
        middle = 0.5 * (lower + upper)
        segments = evaluate(middle)
        if len(segments) < target_count:
            lower = middle
        elif len(segments) > target_count:
            upper = middle
        else:
            break
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _validate_potentials(
    emission_log_potential: torch.Tensor,
    boundary_log_potential: torch.Tensor,
) -> None:
    if emission_log_potential.ndim != 2 or emission_log_potential.shape[1] != 3:
        raise ValueError("emission_log_potential must have shape [sample, 3].")
    if boundary_log_potential.shape != emission_log_potential.shape[:1]:
        raise ValueError("boundary_log_potential must have shape [sample].")
    if emission_log_potential.shape[0] == 0:
        raise ValueError("HSMM requires a non-empty zone.")
    if not bool(torch.all(torch.isfinite(emission_log_potential)).item()):
        raise ValueError("emission potentials must be finite.")
    if not bool(torch.all(torch.isfinite(boundary_log_potential)).item()):
        raise ValueError("boundary potentials must be finite.")


def _segment_score(
    emission_prefix: torch.Tensor,
    boundary: torch.Tensor,
    duration: torch.Tensor,
    *,
    start: int,
    stop: int,
    state: int,
) -> torch.Tensor:
    score = emission_prefix[stop, state] - emission_prefix[start, state]
    score = score + duration[state, stop - start - 1]
    if start > 0:
        score = score + boundary[start]
    return score


def _forward_scores(
    emission_prefix: torch.Tensor,
    boundary: torch.Tensor,
    initial: torch.Tensor,
    transition: torch.Tensor,
    duration: torch.Tensor,
) -> list[torch.Tensor]:
    sample_count = boundary.shape[0]
    rows: list[torch.Tensor] = [
        torch.full(
            (STATE_COUNT,),
            -torch.inf,
            dtype=boundary.dtype,
            device=boundary.device,
        )
    ]
    maximum_duration = max(
        int(torch.count_nonzero(torch.isfinite(duration[state])).item())
        for state in range(STATE_COUNT)
    )
    for stop in range(1, sample_count + 1):
        first_start = max(0, stop - maximum_duration)
        starts = torch.arange(first_start, stop, device=boundary.device)
        duration_indices = stop - starts - 1
        prefix = torch.full(
            (starts.numel(), STATE_COUNT),
            -torch.inf,
            dtype=boundary.dtype,
            device=boundary.device,
        )
        interior_offset = 1 if first_start == 0 else 0
        if first_start == 0:
            prefix[0] = initial
        if starts.numel() > interior_offset:
            previous = torch.stack(
                rows[max(first_start, 1) : stop],
                dim=0,
            )
            prefix[interior_offset:] = torch.logsumexp(
                previous.unsqueeze(2) + transition.unsqueeze(0),
                dim=1,
            )
        boundary_scores = boundary[starts].clone()
        if first_start == 0:
            boundary_scores[0] = 0.0
        segment_scores = (
            emission_prefix[stop].unsqueeze(0)
            - emission_prefix[starts]
            + duration[:, duration_indices].transpose(0, 1)
            + boundary_scores.unsqueeze(1)
        )
        rows.append(torch.logsumexp(prefix + segment_scores, dim=0))
    return rows


def _backward_scores(
    emission_prefix: torch.Tensor,
    boundary: torch.Tensor,
    transition: torch.Tensor,
    duration: torch.Tensor,
) -> list[torch.Tensor]:
    sample_count = boundary.shape[0]
    rows: list[torch.Tensor | None] = [None] * (sample_count + 1)
    rows[sample_count] = torch.zeros(
        STATE_COUNT, dtype=boundary.dtype, device=boundary.device
    )
    for start in range(sample_count - 1, -1, -1):
        previous_values: list[torch.Tensor] = []
        for previous in range(STATE_COUNT):
            candidates: list[torch.Tensor] = []
            for state in range(STATE_COUNT):
                if state == previous:
                    continue
                maximum_duration = sample_count - start
                legal = torch.nonzero(
                    torch.isfinite(duration[state, :maximum_duration]),
                    as_tuple=False,
                ).flatten()
                if legal.numel() == 0:
                    continue
                stops = start + legal + 1
                suffix = torch.stack(
                    [rows[int(stop.item())][state] for stop in stops]
                )
                segment = (
                    emission_prefix[stops, state]
                    - emission_prefix[start, state]
                    + duration[state, legal]
                    + (boundary[start] if start > 0 else 0.0)
                )
                candidates.append(
                    torch.logsumexp(
                        transition[previous, state] + segment + suffix,
                        dim=0,
                    )
                )
            previous_values.append(
                torch.logsumexp(torch.stack(candidates), dim=0)
            )
        rows[start] = torch.stack(previous_values)
    return [item for item in rows if item is not None]


def _viterbi(
    emission_prefix: torch.Tensor,
    boundary: torch.Tensor,
    initial: torch.Tensor,
    transition: torch.Tensor,
    duration: torch.Tensor,
    *,
    segment_reward: float = 0.0,
) -> tuple[torch.Tensor, tuple[HsmmSegment, ...]]:
    sample_count = boundary.shape[0]
    scores = torch.full(
        (sample_count + 1, STATE_COUNT),
        -torch.inf,
        dtype=boundary.dtype,
        device=boundary.device,
    )
    back_start = torch.full(
        (sample_count + 1, STATE_COUNT), -1, dtype=torch.long, device=boundary.device
    )
    back_state = torch.full_like(back_start, -1)
    maximum_duration = max(
        int(torch.count_nonzero(torch.isfinite(duration[state])).item())
        for state in range(STATE_COUNT)
    )
    for stop in range(1, sample_count + 1):
        first_start = max(0, stop - maximum_duration)
        starts = torch.arange(first_start, stop, device=boundary.device)
        duration_indices = stop - starts - 1
        previous_score, previous_state = torch.max(
            scores[starts].unsqueeze(2) + transition.unsqueeze(0),
            dim=1,
        )
        if first_start == 0:
            previous_score[0] = initial
            previous_state[0] = -1
        boundary_scores = boundary[starts].clone()
        if first_start == 0:
            boundary_scores[0] = 0.0
        segment_scores = (
            emission_prefix[stop].unsqueeze(0)
            - emission_prefix[starts]
            + duration[:, duration_indices].transpose(0, 1)
            + boundary_scores.unsqueeze(1)
            + torch.where(
                starts.unsqueeze(1) > 0,
                torch.full(
                    (starts.numel(), 1),
                    float(segment_reward),
                    dtype=boundary.dtype,
                    device=boundary.device,
                ),
                torch.zeros(
                    (starts.numel(), 1),
                    dtype=boundary.dtype,
                    device=boundary.device,
                ),
            )
        )
        values = previous_score + segment_scores
        best_value, best_index = torch.max(values, dim=0)
        scores[stop] = best_value
        selected_starts = starts[best_index]
        back_start[stop] = selected_starts
        back_state[stop] = previous_state[best_index, torch.arange(STATE_COUNT)]
    map_score, final_state = torch.max(scores[sample_count], dim=0)
    segments: list[HsmmSegment] = []
    stop = sample_count
    state = int(final_state.item())
    while stop > 0:
        start = int(back_start[stop, state].item())
        if start < 0:
            raise RuntimeError("HSMM MAP backtracking encountered an invalid pointer.")
        segments.append(HsmmSegment(state_id=state, start=start, stop=stop))
        state = int(back_state[stop, state].item())
        stop = start
    segments.reverse()
    return map_score, tuple(segments)


def exact_hsmm(
    emission_log_potential: torch.Tensor,
    boundary_log_potential: torch.Tensor,
    prior: ZoneHsmmPrior,
) -> HsmmResult:
    """Run exact MAP and forward-backward inference for one contiguous zone."""
    _validate_potentials(emission_log_potential, boundary_log_potential)
    sample_count = int(emission_log_potential.shape[0])
    initial = torch.as_tensor(
        prior.initial_log_probability,
        dtype=emission_log_potential.dtype,
        device=emission_log_potential.device,
    )
    transition = torch.as_tensor(
        prior.transition_log_probability,
        dtype=emission_log_potential.dtype,
        device=emission_log_potential.device,
    )
    duration = prior.duration_scores(
        sample_count,
        device=emission_log_potential.device,
        dtype=emission_log_potential.dtype,
    )
    zero = torch.zeros(
        (1, STATE_COUNT),
        dtype=emission_log_potential.dtype,
        device=emission_log_potential.device,
    )
    emission_prefix = torch.cat(
        (zero, torch.cumsum(emission_log_potential, dim=0)), dim=0
    )
    # For a log-linear HSMM, derivatives of log Z with respect to emission and
    # boundary potentials are exactly their posterior marginals.  Autograd
    # computes the same forward-backward result without a second O(TD) Python
    # enumeration.
    with torch.enable_grad():
        marginal_emission = emission_log_potential.detach().requires_grad_(True)
        marginal_boundary = boundary_log_potential.detach().requires_grad_(True)
        marginal_partition = hsmm_log_partition(
            marginal_emission,
            marginal_boundary,
            prior,
        )
        state_mass, boundary_mass = torch.autograd.grad(
            marginal_partition,
            (marginal_emission, marginal_boundary),
        )
    log_partition = marginal_partition.detach()
    state_mass = state_mass / torch.clamp(
        state_mass.sum(dim=1, keepdim=True), min=torch.finfo(state_mass.dtype).tiny
    )
    state_mass = state_mass.detach()
    boundary_mass = boundary_mass.detach()
    boundary_mass[0] = 1.0
    map_score, map_segments = _viterbi(
        emission_prefix, boundary_log_potential, initial, transition, duration
    )
    consensus_segments = posterior_consensus_segments(
        emission_log_potential,
        boundary_log_potential,
        state_mass,
        boundary_mass,
        prior,
    )
    return HsmmResult(
        log_partition=log_partition,
        map_score=map_score,
        map_segments=map_segments,
        consensus_segments=consensus_segments,
        state_marginal=state_mass,
        boundary_marginal=torch.clamp(boundary_mass, min=0.0, max=1.0),
    )


def hsmm_log_partition(
    emission_log_potential: torch.Tensor,
    boundary_log_potential: torch.Tensor,
    prior: ZoneHsmmPrior,
) -> torch.Tensor:
    """Return the differentiable exact log partition without posterior work."""
    _validate_potentials(emission_log_potential, boundary_log_potential)
    sample_count = int(emission_log_potential.shape[0])
    initial = torch.as_tensor(
        prior.initial_log_probability,
        dtype=emission_log_potential.dtype,
        device=emission_log_potential.device,
    )
    transition = torch.as_tensor(
        prior.transition_log_probability,
        dtype=emission_log_potential.dtype,
        device=emission_log_potential.device,
    )
    duration = prior.duration_scores(
        sample_count,
        device=emission_log_potential.device,
        dtype=emission_log_potential.dtype,
    )
    emission_prefix = torch.cat(
        (
            torch.zeros(
                (1, STATE_COUNT),
                dtype=emission_log_potential.dtype,
                device=emission_log_potential.device,
            ),
            torch.cumsum(emission_log_potential, dim=0),
        ),
        dim=0,
    )
    forward = _forward_scores(
        emission_prefix, boundary_log_potential, initial, transition, duration
    )
    return torch.logsumexp(forward[sample_count], dim=0)


def hsmm_path_score(
    emission_log_potential: torch.Tensor,
    boundary_log_potential: torch.Tensor,
    prior: ZoneHsmmPrior,
    segments: Sequence[HsmmSegment],
) -> torch.Tensor:
    """Score one legal complete segmentation under the exact HSMM contract."""
    _validate_potentials(emission_log_potential, boundary_log_potential)
    sample_count = int(emission_log_potential.shape[0])
    if not segments or segments[0].start != 0 or segments[-1].stop != sample_count:
        raise ValueError("truth HSMM path must cover the complete zone.")
    for index, segment in enumerate(segments):
        if index and segment.start != segments[index - 1].stop:
            raise ValueError("truth HSMM path segments must be contiguous.")
        if index and segment.state_id == segments[index - 1].state_id:
            raise ValueError("truth HSMM path cannot contain adjacent equal states.")
    initial = torch.as_tensor(
        prior.initial_log_probability,
        dtype=emission_log_potential.dtype,
        device=emission_log_potential.device,
    )
    transition = torch.as_tensor(
        prior.transition_log_probability,
        dtype=emission_log_potential.dtype,
        device=emission_log_potential.device,
    )
    duration = prior.duration_scores(
        sample_count,
        device=emission_log_potential.device,
        dtype=emission_log_potential.dtype,
    )
    emission_prefix = torch.cat(
        (
            torch.zeros(
                (1, STATE_COUNT),
                dtype=emission_log_potential.dtype,
                device=emission_log_potential.device,
            ),
            torch.cumsum(emission_log_potential, dim=0),
        ),
        dim=0,
    )
    score = initial[segments[0].state_id]
    for index, segment in enumerate(segments):
        if index:
            score = score + transition[
                segments[index - 1].state_id, segment.state_id
            ]
        score = score + _segment_score(
            emission_prefix,
            boundary_log_potential,
            duration,
            start=segment.start,
            stop=segment.stop,
            state=segment.state_id,
        )
    return score


__all__ = [
    "HSMM_PRIOR_SCHEMA",
    "HsmmPrior",
    "HsmmResult",
    "HsmmSegment",
    "ZoneHsmmPrior",
    "canonicalize_hsmm_segments",
    "exact_hsmm",
    "fit_hsmm_prior",
    "freeze_hsmm_prior",
    "hsmm_log_partition",
    "hsmm_path_score",
    "posterior_consensus_segments",
]
