"""LFM anchoring and the shared three-parameter structured representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch

from cup.synthetic.core.signal import finite_support_fir, valid_filter_decimate

from ginn_v2.contracts import (
    InputContractError,
    ObservationTile,
    Segment,
)


@dataclass(frozen=True)
class LfmAnchor:
    model: np.ndarray
    highres: np.ndarray
    model_support: np.ndarray
    highres_support: np.ndarray
    intercept: np.ndarray
    slope: np.ndarray


@dataclass(frozen=True)
class ProfileSufficientStatistics:
    coefficient: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    target_square_mean: float
    identifiability_rank: int
    basis_condition: float


def _zone_coordinate(axis: np.ndarray, top: float, bottom: float) -> np.ndarray:
    return 2.0 * (np.asarray(axis, dtype=np.float64) - float(top)) / (
        float(bottom) - float(top)
    ) - 1.0


def build_lfm_anchor(tile: ObservationTile) -> LfmAnchor:
    """Fit one zone-linear LFM anchor per lateral trace."""
    model_axis = tile.model_axis.coordinates
    highres_axis = tile.highres_axis.coordinates
    model = np.full(tile.lfm.shape, np.nan, dtype=np.float64)
    highres = np.full(
        (tile.width, highres_axis.size),
        np.nan,
        dtype=np.float64,
    )
    model_support = np.zeros_like(tile.observed_valid)
    highres_support = np.zeros_like(highres, dtype=bool)
    intercept = np.full(tile.width, np.nan, dtype=np.float64)
    slope = np.full(tile.width, np.nan, dtype=np.float64)

    for trace in range(tile.width):
        if not tile.lateral_valid[trace]:
            continue
        top = float(tile.zone_top[trace])
        bottom = float(tile.zone_bottom[trace])
        inside = (
            tile.observed_valid[trace]
            & (model_axis >= top)
            & (model_axis <= bottom)
        )
        valid_sample_count = int(np.count_nonzero(inside))
        if valid_sample_count == 0:
            raise InputContractError(
                f"{tile.identity}: trace {trace} has {valid_sample_count} valid "
                f"model-grid LFM samples inside zone [{top:.6g}, {bottom:.6g}]."
            )
        coordinate = _zone_coordinate(model_axis[inside], top, bottom)
        if valid_sample_count == 1:
            coefficients = np.asarray(
                (float(tile.lfm[trace, inside][0]), 0.0),
                dtype=np.float64,
            )
        else:
            design = np.column_stack((np.ones(coordinate.size), coordinate))
            coefficients, _, rank, _ = np.linalg.lstsq(
                design,
                tile.lfm[trace, inside],
                rcond=None,
            )
            if rank != 2 or np.any(~np.isfinite(coefficients)):
                raise InputContractError(
                    f"{tile.identity}: trace {trace} LFM anchor is rank deficient."
                )
        if np.any(~np.isfinite(coefficients)):
            raise InputContractError(
                f"{tile.identity}: trace {trace} LFM anchor is non-finite."
            )
        intercept[trace], slope[trace] = coefficients
        model_coordinate = _zone_coordinate(model_axis, top, bottom)
        high_coordinate = _zone_coordinate(highres_axis, top, bottom)
        model_inside = (model_axis >= top) & (model_axis <= bottom)
        high_inside = (highres_axis >= top) & (highres_axis <= bottom)
        model[trace, model_inside] = coefficients[0] + coefficients[1] * model_coordinate[
            model_inside
        ]
        highres[trace, high_inside] = coefficients[0] + coefficients[1] * high_coordinate[
            high_inside
        ]
        model_support[trace] = model_inside & tile.observed_valid[trace]
        highres_support[trace] = high_inside

    return LfmAnchor(
        model=model,
        highres=highres,
        model_support=model_support,
        highres_support=highres_support,
        intercept=intercept,
        slope=slope,
    )


def lfm_residual_from_anchor(
    tile: ObservationTile,
    anchor: LfmAnchor,
) -> np.ndarray:
    """Return a finite model-grid LFM residual on the anchor support."""

    if anchor.model.shape != tile.lfm.shape or anchor.model_support.shape != tile.lfm.shape:
        raise InputContractError("LFM anchor shape differs from the observation tile.")
    support = anchor.model_support & tile.observed_valid
    values = np.asarray(tile.lfm, dtype=np.float64) - anchor.model
    if np.any(~np.isfinite(values[support])):
        raise InputContractError("LFM residual contains non-finite supported samples.")
    return np.where(support, values, 0.0)


def profile_basis(sample_count: int) -> np.ndarray:
    if isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if sample_count == 1:
        xi = np.zeros(1, dtype=np.float64)
    else:
        xi = np.linspace(0.0, 1.0, int(sample_count), dtype=np.float64)
    return np.column_stack(
        (
            np.ones(xi.size, dtype=np.float64),
            2.0 * xi - 1.0,
            np.sin(np.pi * xi),
        )
    )


def fit_profile_coefficients(values: np.ndarray) -> tuple[float, float, float]:
    target = np.asarray(values, dtype=np.float64).reshape(-1)
    if target.size == 0 or np.any(~np.isfinite(target)):
        raise InputContractError("profile fitting requires finite samples.")
    basis = profile_basis(target.size)
    if target.size == 1:
        return float(target[0]), 0.0, 0.0
    coefficients = np.linalg.pinv(basis, rcond=1.0e-8) @ target
    if np.any(~np.isfinite(coefficients)):
        raise FloatingPointError("profile fitting produced non-finite coefficients.")
    return tuple(float(value) for value in coefficients)


def profile_sufficient_statistics(values: np.ndarray) -> ProfileSufficientStatistics:
    target = np.asarray(values, dtype=np.float64).reshape(-1)
    if target.size == 0 or np.any(~np.isfinite(target)):
        raise InputContractError("profile statistics require finite samples.")
    basis = profile_basis(target.size)
    singular = np.linalg.svd(basis, compute_uv=False)
    rank = int(np.linalg.matrix_rank(basis, tol=1.0e-10))
    condition = (
        float(singular[0] / singular[-1])
        if rank == 3 and singular[-1] > 0.0
        else float("inf")
    )
    coefficient = np.asarray(fit_profile_coefficients(target), dtype=np.float64)
    count = float(target.size)
    return ProfileSufficientStatistics(
        coefficient=coefficient,
        gram=(basis.T @ basis) / count,
        cross=(basis.T @ target) / count,
        target_square_mean=float(np.mean(target**2)),
        identifiability_rank=rank,
        basis_condition=condition,
    )


def decode_segments_numpy(
    background_highres: np.ndarray,
    segments: Iterable[Segment],
) -> tuple[np.ndarray, np.ndarray]:
    background = np.asarray(background_highres, dtype=np.float64)
    if background.ndim != 2:
        raise InputContractError("background_highres must be [lateral, sample].")
    output = background.copy()
    state = np.full(background.shape, -1, dtype=np.int8)
    coverage = np.zeros(background.shape, dtype=bool)
    for segment in segments:
        if segment.trace_index < 0 or segment.trace_index >= background.shape[0]:
            raise InputContractError("segment trace_index is outside the tile.")
        if (
            segment.start_index < 0
            or segment.stop_index <= segment.start_index
            or segment.stop_index > background.shape[1]
        ):
            raise InputContractError("segment extent is outside the high-resolution axis.")
        index = slice(segment.start_index, segment.stop_index)
        if np.any(coverage[segment.trace_index, index]):
            raise InputContractError("segments overlap within a trace.")
        basis = profile_basis(segment.stop_index - segment.start_index)
        profile = basis @ np.asarray(
            [segment.c0, segment.c1, segment.c2],
            dtype=np.float64,
        )
        output[segment.trace_index, index] += profile
        state[segment.trace_index, index] = int(segment.state_id)
        coverage[segment.trace_index, index] = True
    valid_background = np.isfinite(background)
    if np.any(valid_background & ~coverage):
        raise InputContractError("segments do not cover every valid high-resolution sample.")
    if np.any(coverage & ~np.isfinite(output)):
        raise FloatingPointError("structured decoder produced non-finite values.")
    return output, state


def decode_segments_torch(
    background_highres: torch.Tensor,
    trace_index: torch.Tensor,
    start_index: torch.Tensor,
    stop_index: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Differentiable decoder used by training and parity smoke."""
    if background_highres.ndim != 2:
        raise ValueError("background_highres must be [lateral, sample].")
    if coefficients.ndim != 2 or coefficients.shape[1] != 3:
        raise ValueError("coefficients must have shape [segment, 3].")
    count = int(coefficients.shape[0])
    if any(item.shape != (count,) for item in (trace_index, start_index, stop_index)):
        raise ValueError("segment index tensors must have shape [segment].")
    output = background_highres.clone()
    coverage = torch.zeros_like(background_highres, dtype=torch.bool)
    for item in range(count):
        trace = int(trace_index[item].item())
        start = int(start_index[item].item())
        stop = int(stop_index[item].item())
        if trace < 0 or trace >= output.shape[0] or start < 0 or stop <= start or stop > output.shape[1]:
            raise ValueError("segment extent is outside decoder support.")
        xi = (
            torch.zeros(1, dtype=output.dtype, device=output.device)
            if stop - start == 1
            else torch.linspace(
                0.0,
                1.0,
                stop - start,
                dtype=output.dtype,
                device=output.device,
            )
        )
        basis = torch.stack(
            (
                torch.ones_like(xi),
                2.0 * xi - 1.0,
                torch.sin(torch.pi * xi),
            ),
            dim=-1,
        )
        output[trace, start:stop] = output[trace, start:stop] + (
            basis @ coefficients[item]
        )
        coverage[trace, start:stop] = True
    if torch.any(torch.isfinite(background_highres) & ~coverage):
        raise ValueError("segments do not cover valid decoder support.")
    return output


def project_highres_to_model(
    values: np.ndarray,
    *,
    highres_interval: float,
    model_interval: float,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = float(model_interval) / float(highres_interval)
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1.0e-10):
        raise InputContractError("projection intervals must have an integer ratio.")
    taps = finite_support_fir(factor)
    projected, support = valid_filter_decimate(
        np.asarray(values, dtype=np.float64),
        factor=factor,
        taps=taps,
    )
    return np.asarray(projected, dtype=np.float64), np.asarray(support, dtype=bool)


def project_supported_highres_to_model(
    values: np.ndarray,
    support: np.ndarray,
    *,
    highres_interval: float,
    model_interval: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project only model samples whose complete FIR window is supported."""

    data = np.asarray(values, dtype=np.float64)
    valid = np.asarray(support, dtype=bool)
    if data.shape != valid.shape or data.ndim != 2:
        raise InputContractError(
            "supported projection requires matching [lateral, highres_sample] arrays."
        )
    ratio = float(model_interval) / float(highres_interval)
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1.0e-10):
        raise InputContractError("projection intervals must have an integer ratio.")
    taps = finite_support_fir(factor)
    projected, finite_support = valid_filter_decimate(
        np.where(valid, data, 0.0),
        factor=factor,
        taps=taps,
    )
    projected_support = np.zeros(projected.shape, dtype=bool)
    half = taps.size // 2
    centers = np.arange(projected.shape[-1], dtype=np.int64) * factor
    convolution_indices = centers[finite_support] - half
    window = np.ones(taps.size, dtype=np.int64)
    for trace in range(data.shape[0]):
        count = np.convolve(valid[trace].astype(np.int64), window, mode="valid")
        projected_support[trace, finite_support] = (
            count[convolution_indices] == taps.size
        )
    projected[~projected_support] = np.nan
    return np.asarray(projected, dtype=np.float64), projected_support


__all__ = [
    "LfmAnchor",
    "ProfileSufficientStatistics",
    "build_lfm_anchor",
    "lfm_residual_from_anchor",
    "decode_segments_numpy",
    "decode_segments_torch",
    "fit_profile_coefficients",
    "profile_basis",
    "profile_sufficient_statistics",
    "project_highres_to_model",
    "project_supported_highres_to_model",
]
