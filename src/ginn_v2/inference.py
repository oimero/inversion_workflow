"""Evidence fusion, deterministic section/volume inference, and forward diagnosis."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Mapping

import numpy as np
from scipy.special import logsumexp

from ginn_v2.contracts import (
    ObservableEvidence,
    ObservationTile,
    StructuredPrediction,
    VolumeInferenceResult,
)
from ginn_v2.generator import ConditionalGenerator


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
    prediction = generator._decode_evidence(
        fuse_directional_evidence(inline, xline)
    )
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
    "forward_diagnostic",
    "fuse_directional_evidence",
    "infer_fused_section",
    "infer_section",
    "infer_volume",
]
