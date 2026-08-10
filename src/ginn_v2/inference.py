"""Evidence fusion, deterministic section/volume inference, and forward diagnosis."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from scipy.special import logsumexp

from ginn_v2.contracts import (
    ObservableEvidence,
    ObservationTile,
    StructuredPrediction,
    VolumeInferenceResult,
)
from ginn_v2.generator import ConditionalGenerator
from ginn_v2.semi_markov import (
    SemiMarkovConditioning,
    SemiMarkovDecodePolicy,
)


_SENSITIVITY_PARAMETERS = (
    "renewal_snr_threshold",
    "same_state_renewal_probability",
    "same_state_merge_scale_fraction",
    "duration_temperature",
    "transition_temperature",
)


@dataclass(frozen=True)
class DecodeSensitivityCase:
    """One predeclared one-at-a-time deterministic decode perturbation."""

    case_id: str
    changed_parameter: str | None
    changed_value: float | None
    policy: SemiMarkovDecodePolicy
    conditioning: SemiMarkovConditioning

    def to_mapping(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "changed_parameter": self.changed_parameter,
            "changed_value": self.changed_value,
            "decode_policy": self.policy.to_mapping(),
            "conditioning": self.conditioning.to_mapping(),
        }


@dataclass(frozen=True)
class DecodeSensitivityResult:
    """Predictions sharing one frozen evidence tensor."""

    cases: tuple[DecodeSensitivityCase, ...]
    predictions: Mapping[str, StructuredPrediction]

    def __post_init__(self) -> None:
        identities = tuple(case.case_id for case in self.cases)
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("decode sensitivity case identities must be unique.")
        if set(self.predictions) != set(identities):
            raise ValueError("decode sensitivity predictions do not match cases.")


def _case_value_token(value: float) -> str:
    token = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return token


def decode_sensitivity_cases(
    base_policy: SemiMarkovDecodePolicy,
    base_conditioning: SemiMarkovConditioning,
    specification: Mapping[str, Any],
) -> tuple[DecodeSensitivityCase, ...]:
    """Build a strict one-at-a-time policy plan around one reference."""

    if set(specification) != {"reference", "alternatives"}:
        raise ValueError(
            "decode sensitivity requires exactly reference and alternatives."
        )
    reference = specification["reference"]
    alternatives = specification["alternatives"]
    if not isinstance(reference, Mapping) or not isinstance(alternatives, Mapping):
        raise TypeError("decode sensitivity reference/alternatives must be mappings.")
    if set(reference) != set(_SENSITIVITY_PARAMETERS):
        raise ValueError(
            "decode sensitivity reference must define every swept parameter."
        )
    unknown = sorted(set(alternatives).difference(_SENSITIVITY_PARAMETERS))
    if unknown:
        raise ValueError(f"unknown decode sensitivity parameters: {unknown}")

    reference_policy = replace(
        base_policy,
        renewal_snr_threshold=float(reference["renewal_snr_threshold"]),
        same_state_renewal_probability=float(
            reference["same_state_renewal_probability"]
        ),
        same_state_merge_scale_fraction=float(
            reference["same_state_merge_scale_fraction"]
        ),
    )
    reference_conditioning = replace(
        base_conditioning,
        duration_temperature=float(reference["duration_temperature"]),
        transition_temperature=float(reference["transition_temperature"]),
    )
    cases: list[DecodeSensitivityCase] = [
        DecodeSensitivityCase(
            case_id="reference",
            changed_parameter=None,
            changed_value=None,
            policy=reference_policy,
            conditioning=reference_conditioning,
        )
    ]
    reference_values = {
        name: float(reference[name]) for name in _SENSITIVITY_PARAMETERS
    }
    for name in _SENSITIVITY_PARAMETERS:
        raw_values = alternatives.get(name, ())
        if not isinstance(raw_values, (list, tuple)):
            raise TypeError(f"decode sensitivity alternatives.{name} must be a list.")
        for raw_value in raw_values:
            value = float(raw_value)
            if np.isclose(value, reference_values[name], rtol=0.0, atol=1.0e-12):
                continue
            policy = reference_policy
            conditioning = reference_conditioning
            if name in {
                "renewal_snr_threshold",
                "same_state_renewal_probability",
                "same_state_merge_scale_fraction",
            }:
                policy = replace(policy, **{name: value})
            else:
                conditioning = replace(conditioning, **{name: value})
            cases.append(
                DecodeSensitivityCase(
                    case_id=f"{name}__{_case_value_token(value)}",
                    changed_parameter=name,
                    changed_value=value,
                    policy=policy,
                    conditioning=conditioning,
                )
            )
    if len(cases) < 2:
        raise ValueError("decode sensitivity requires at least one alternative.")
    return tuple(cases)


def run_decode_policy_sensitivity(
    generator: ConditionalGenerator,
    evidence: ObservableEvidence,
    specification: Mapping[str, Any],
    *,
    on_case_complete: Callable[
        [int, int, DecodeSensitivityCase, StructuredPrediction], None
    ]
    | None = None,
) -> DecodeSensitivityResult:
    """Decode one frozen evidence tensor under one-at-a-time policy cases."""

    if generator.semi_markov_conditioning is None:
        raise RuntimeError("decode sensitivity requires semi-Markov conditioning.")
    cases = decode_sensitivity_cases(
        generator.decode_policy,
        generator.semi_markov_conditioning,
        specification,
    )
    predictions: dict[str, StructuredPrediction] = {}
    for index, case in enumerate(cases, start=1):
        prediction = generator.decode(
            evidence,
            policy=case.policy,
            conditioning=case.conditioning,
        )
        predictions[case.case_id] = prediction
        if on_case_complete is not None:
            on_case_complete(index, len(cases), case, prediction)
    return DecodeSensitivityResult(cases=cases, predictions=predictions)


def summarize_structured_prediction(
    prediction: StructuredPrediction,
) -> dict[str, float]:
    """Report vertical resolution and stripe-risk proxies for one result."""

    segments_by_trace: dict[int, list[Any]] = {}
    for segment in prediction.realization.segments:
        segments_by_trace.setdefault(int(segment.trace_index), []).append(segment)
    counts: list[int] = []
    durations: list[float] = []
    same_state = 0
    adjacency = 0
    alternating = 0
    triplets = 0
    interval = float(prediction.evidence.highres_axis.sample_interval)
    for trace_segments in segments_by_trace.values():
        ordered = sorted(trace_segments, key=lambda item: int(item.start_index))
        states = np.asarray([int(item.state_id) for item in ordered], dtype=np.int64)
        counts.append(int(states.size))
        durations.extend(
            (int(item.stop_index) - int(item.start_index)) * interval
            for item in ordered
        )
        if states.size >= 2:
            adjacency += int(states.size - 1)
            same_state += int(np.count_nonzero(states[:-1] == states[1:]))
        if states.size >= 3:
            triplets += int(states.size - 2)
            alternating += int(
                np.count_nonzero(
                    (states[1:-1] == 1)
                    & (states[:-2] == states[2:])
                    & (states[:-2] != 1)
                )
            )
    if not counts or not durations:
        raise ValueError("structured sensitivity prediction has no segments.")
    count_values = np.asarray(counts, dtype=np.float64)
    duration_values = np.asarray(durations, dtype=np.float64)
    state = prediction.realization.state_highres
    support = prediction.evidence.highres_support
    supported_state = state[support]
    background_fraction = float(np.mean(supported_state == 1))
    highres = prediction.realization.log_ai_highres
    shared = support[:-1] & support[1:]
    lateral_difference = np.diff(highres, axis=0)
    finite = shared & np.isfinite(lateral_difference)
    lateral_rms = (
        float(np.sqrt(np.mean(lateral_difference[finite] ** 2)))
        if np.any(finite)
        else float("nan")
    )
    return {
        "supported_trace_count": float(len(counts)),
        "segment_count": float(duration_values.size),
        "segments_per_trace_p10": float(np.quantile(count_values, 0.10)),
        "segments_per_trace_median": float(np.median(count_values)),
        "segments_per_trace_p90": float(np.quantile(count_values, 0.90)),
        "segment_thickness_p10": float(np.quantile(duration_values, 0.10)),
        "segment_thickness_median": float(np.median(duration_values)),
        "segment_thickness_p90": float(np.quantile(duration_values, 0.90)),
        "same_state_adjacency_fraction": float(same_state / max(adjacency, 1)),
        "extreme_background_extreme_triplet_fraction": float(
            alternating / max(triplets, 1)
        ),
        "background_sample_fraction": background_fraction,
        "highres_lateral_rms": lateral_rms,
        "projection_consistency_rmse": float(
            prediction.diagnostics["projection_consistency_rmse"]
        ),
    }


def _axes_match(left: object, right: object) -> bool:
    return bool(
        left.sample_domain == right.sample_domain
        and left.unit == right.unit
        and left.depth_basis == right.depth_basis
        and left.coordinates.shape == right.coordinates.shape
        and np.allclose(
            left.coordinates,
            right.coordinates,
            rtol=0.0,
            atol=1.0e-10,
        )
    )


def fuse_directional_evidence(
    inline: ObservableEvidence,
    xline: ObservableEvidence,
) -> ObservableEvidence:
    """Fuse calibrated evidence before one deterministic structured decode."""

    coordinates_differ = (
        (inline.x_m is None) != (xline.x_m is None)
        or (
            inline.x_m is not None
            and (
                not np.array_equal(inline.x_m, xline.x_m)
                or not np.array_equal(inline.y_m, xline.y_m)
            )
        )
    )
    if (
        not _axes_match(inline.model_axis, xline.model_axis)
        or not _axes_match(inline.highres_axis, xline.highres_axis)
        or not np.array_equal(inline.lateral_m, xline.lateral_m)
        or coordinates_differ
    ):
        raise ValueError("directional evidence grids differ.")

    inline_support = inline.support.astype(np.float64)
    xline_support = xline.support.astype(np.float64)
    support_count = inline_support + xline_support
    support = support_count > 0.0
    denominator = np.maximum(support_count, 1.0)

    def average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        weights = inline_support
        other = xline_support
        if left.ndim == 3:
            weights = weights[..., None]
            other = other[..., None]
            divisor = denominator[..., None]
        else:
            divisor = denominator
        return (left * weights + right * other) / divisor

    def mixture(
        left_mean: np.ndarray,
        left_scale: np.ndarray,
        right_mean: np.ndarray,
        right_scale: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = average(left_mean, right_mean)
        within = average(left_scale**2, right_scale**2)
        between = average(
            (left_mean - mean) ** 2,
            (right_mean - mean) ** 2,
        )
        return mean, np.sqrt(np.maximum(within + between, 1.0e-12))

    increment_mean, increment_scale = mixture(
        inline.projected_log_ai_increment_mean,
        inline.projected_log_ai_increment_scale,
        xline.projected_log_ai_increment_mean,
        xline.projected_log_ai_increment_scale,
    )
    reflectivity_mean, reflectivity_scale = mixture(
        inline.signed_reflectivity_mean,
        inline.signed_reflectivity_scale,
        xline.signed_reflectivity_mean,
        xline.signed_reflectivity_scale,
    )
    background = average(
        inline.background_lfm_linear,
        xline.background_lfm_linear,
    )
    highres_support_count = (
        inline.highres_support.astype(np.float64)
        + xline.highres_support.astype(np.float64)
    )
    highres_support = highres_support_count > 0.0
    background_highres = (
        np.where(
            inline.highres_support,
            inline.background_lfm_linear_highres,
            0.0,
        )
        + np.where(
            xline.highres_support,
            xline.background_lfm_linear_highres,
            0.0,
        )
    ) / np.maximum(highres_support_count, 1.0)
    background_highres[~highres_support] = np.nan
    state_log_potential = average(
        inline.state_log_potential,
        xline.state_log_potential,
    )
    state_log_potential -= logsumexp(
        state_log_potential,
        axis=-1,
        keepdims=True,
    )
    tuning = average(inline.local_tuning_scale, xline.local_tuning_scale)
    return ObservableEvidence(
        model_axis=inline.model_axis,
        highres_axis=inline.highres_axis,
        background_lfm_linear=background,
        background_lfm_linear_highres=background_highres,
        projected_log_ai_increment_mean=increment_mean,
        projected_log_ai_increment_scale=increment_scale,
        signed_reflectivity_mean=reflectivity_mean,
        signed_reflectivity_scale=reflectivity_scale,
        state_log_potential=state_log_potential,
        local_tuning_scale=tuning,
        support=support,
        highres_support=highres_support,
        lateral_m=inline.lateral_m,
        x_m=inline.x_m,
        y_m=inline.y_m,
        identity=f"fused:{inline.identity}|{xline.identity}",
    )


def _center_evidence(
    evidence: ObservableEvidence,
    center_index: int | None,
) -> ObservableEvidence:
    width = evidence.lateral_m.size
    index = width // 2 if center_index is None else int(center_index)
    if index < 0 or index >= width:
        raise IndexError("directional center index is outside the evidence tile.")
    selection = slice(index, index + 1)
    return ObservableEvidence(
        model_axis=evidence.model_axis,
        highres_axis=evidence.highres_axis,
        background_lfm_linear=evidence.background_lfm_linear[selection],
        background_lfm_linear_highres=(
            evidence.background_lfm_linear_highres[selection]
        ),
        projected_log_ai_increment_mean=(
            evidence.projected_log_ai_increment_mean[selection]
        ),
        projected_log_ai_increment_scale=(
            evidence.projected_log_ai_increment_scale[selection]
        ),
        signed_reflectivity_mean=evidence.signed_reflectivity_mean[selection],
        signed_reflectivity_scale=evidence.signed_reflectivity_scale[selection],
        state_log_potential=evidence.state_log_potential[selection],
        local_tuning_scale=evidence.local_tuning_scale[selection],
        support=evidence.support[selection],
        highres_support=evidence.highres_support[selection],
        lateral_m=np.asarray([0.0], dtype=np.float64),
        x_m=None if evidence.x_m is None else evidence.x_m[selection],
        y_m=None if evidence.y_m is None else evidence.y_m[selection],
        identity=evidence.identity,
    )


def infer_section(
    generator: ConditionalGenerator,
    tile: ObservationTile,
) -> StructuredPrediction:
    """Run one section through the deterministic observation/decode seam."""

    return generator.predict(tile)


def infer_fused_section(
    generator: ConditionalGenerator,
    inline_tile: ObservationTile,
    xline_tile: ObservationTile,
    *,
    inline_center_index: int | None = None,
    xline_center_index: int | None = None,
) -> StructuredPrediction:
    """Fuse inline/xline evidence, then perform one deterministic decode."""

    inline = _center_evidence(
        generator.observe(inline_tile),
        inline_center_index,
    )
    xline = _center_evidence(
        generator.observe(xline_tile),
        xline_center_index,
    )
    shared = inline.support & xline.support
    if not np.any(shared):
        raise ValueError("directional evidence has no shared support.")
    prediction = generator.decode(fuse_directional_evidence(inline, xline))
    disagreement = float(
        np.sqrt(
            np.mean(
                (
                    inline.projected_log_ai_increment_mean[shared]
                    - xline.projected_log_ai_increment_mean[shared]
                )
                ** 2
            )
        )
    )
    return replace(
        prediction,
        diagnostics={
            **dict(prediction.diagnostics),
            "directional_increment_rmse": disagreement,
        },
    )


def infer_volume(
    generator: ConditionalGenerator,
    directional_tiles: Callable[
        [], Iterable[tuple[str, ObservationTile, ObservationTile]]
    ],
) -> VolumeInferenceResult:
    """Run deterministic directional fusion once per volume tile."""

    if not callable(directional_tiles):
        raise TypeError("volume inference requires a restartable tile factory.")
    predictions: dict[str, StructuredPrediction] = {}
    for tile_id, inline_tile, xline_tile in directional_tiles():
        identity = str(tile_id).strip()
        if not identity or identity in predictions:
            raise ValueError("volume tile identities must be non-empty and unique.")
        predictions[identity] = infer_fused_section(
            generator,
            inline_tile,
            xline_tile,
        )
    if not predictions:
        raise ValueError("volume inference received no tiles.")
    return VolumeInferenceResult(tiles=predictions)


def forward_diagnostic(
    prediction: StructuredPrediction,
    forward: Callable[[np.ndarray], np.ndarray],
    observed_seismic: np.ndarray,
    support: np.ndarray,
) -> Mapping[str, float]:
    """Evaluate one prediction without altering training or inference."""

    predicted = np.asarray(
        forward(prediction.realization.projected_log_ai),
        dtype=np.float64,
    )
    observed = np.asarray(observed_seismic, dtype=np.float64)
    valid = np.asarray(support, dtype=bool)
    if predicted.shape != observed.shape or valid.shape != observed.shape:
        raise ValueError("forward diagnostic arrays must share one shape.")
    valid &= np.isfinite(predicted) & np.isfinite(observed)
    if np.count_nonzero(valid) < 2:
        raise ValueError("forward diagnostic has fewer than two valid samples.")
    residual = predicted[valid] - observed[valid]
    correlation = (
        float(np.corrcoef(predicted[valid], observed[valid])[0, 1])
        if np.std(predicted[valid]) > 0.0 and np.std(observed[valid]) > 0.0
        else float("nan")
    )
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "correlation": correlation,
        "valid_sample_count": float(np.count_nonzero(valid)),
    }


__all__ = [
    "DecodeSensitivityCase",
    "DecodeSensitivityResult",
    "decode_sensitivity_cases",
    "forward_diagnostic",
    "fuse_directional_evidence",
    "infer_fused_section",
    "infer_section",
    "infer_volume",
    "run_decode_policy_sensitivity",
    "summarize_structured_prediction",
]
