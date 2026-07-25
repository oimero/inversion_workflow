"""Zone-linear LFM anchoring and direct effective-parameter decoding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cup.synthetic.core.records import SampleAxis
from ginn_v2.truth import SegmentTruth, StructuredSample


PARAMETER_NAMES = ("c0", "c1", "c2")


@dataclass(frozen=True)
class AnchoredSegment:
    """One truth segment expressed relative to the zone-linear LFM anchor."""

    source: SegmentTruth
    sample_indices: np.ndarray
    basis: np.ndarray
    effective_parameters_lfm: np.ndarray
    profile_supervision_valid: bool
    parameter_supervision_valid: bool
    parameter_identifiability_rank: int
    parameter_basis_condition: float
    rebase_residual: float

    def __post_init__(self) -> None:
        indices = np.asarray(self.sample_indices, dtype=np.int64).reshape(-1)
        basis = np.asarray(self.basis, dtype=np.float64)
        parameters = np.asarray(
            self.effective_parameters_lfm, dtype=np.float64
        ).reshape(-1)
        if indices.size == 0:
            raise ValueError("AnchoredSegment requires at least one sample.")
        if basis.shape != (indices.size, 3):
            raise ValueError("AnchoredSegment.basis must have shape [samples, 3].")
        if parameters.shape != (3,) or np.any(~np.isfinite(parameters)):
            raise ValueError("AnchoredSegment parameters must contain three finite values.")
        if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
            raise ValueError("AnchoredSegment.sample_indices must be ordered and unique.")
        rank = int(self.parameter_identifiability_rank)
        if rank < 1 or rank > 3:
            raise ValueError("AnchoredSegment identifiability rank must be in [1, 3].")
        condition = float(self.parameter_basis_condition)
        if not np.isfinite(condition) and rank == 3:
            raise ValueError("Rank-3 AnchoredSegment must have finite basis condition.")
        residual = float(self.rebase_residual)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("AnchoredSegment rebase residual must be finite and non-negative.")
        object.__setattr__(self, "sample_indices", indices)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "effective_parameters_lfm", parameters)
        object.__setattr__(self, "profile_supervision_valid", bool(self.profile_supervision_valid))
        object.__setattr__(
            self, "parameter_supervision_valid", bool(self.parameter_supervision_valid)
        )
        object.__setattr__(self, "parameter_identifiability_rank", rank)
        object.__setattr__(self, "parameter_basis_condition", condition)
        object.__setattr__(self, "rebase_residual", residual)


@dataclass(frozen=True)
class LfmAnchoredStructuredSample:
    """A StructuredSample plus one authoritative LFM-relative supervision view."""

    source: StructuredSample
    background_a_lfm: float
    background_b_lfm: float
    background_lfm_model: np.ndarray
    background_lfm_highres: np.ndarray
    ai_bounds: tuple[float, float]
    segments: tuple[AnchoredSegment, ...]
    decoder_max_abs_error: float

    def __post_init__(self) -> None:
        if not isinstance(self.source, StructuredSample):
            raise TypeError("LfmAnchoredStructuredSample.source must be StructuredSample.")
        model = np.asarray(self.background_lfm_model, dtype=np.float64).reshape(-1)
        highres = np.asarray(self.background_lfm_highres, dtype=np.float64).reshape(-1)
        if model.shape != self.source.observed.sample_axis.coordinates.shape:
            raise ValueError("background_lfm_model must match the observed axis.")
        if highres.shape != self.source.latent.latent_axis.coordinates.shape:
            raise ValueError("background_lfm_highres must match the latent axis.")
        lower, upper = (float(value) for value in self.ai_bounds)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("LfmAnchoredStructuredSample.ai_bounds are invalid.")
        segments = tuple(self.segments)
        if not segments or not all(isinstance(item, AnchoredSegment) for item in segments):
            raise TypeError("LfmAnchoredStructuredSample.segments are invalid.")
        error = float(self.decoder_max_abs_error)
        if not np.isfinite(error) or error < 0.0:
            raise ValueError("decoder_max_abs_error must be finite and non-negative.")
        object.__setattr__(self, "background_a_lfm", float(self.background_a_lfm))
        object.__setattr__(self, "background_b_lfm", float(self.background_b_lfm))
        object.__setattr__(self, "background_lfm_model", model)
        object.__setattr__(self, "background_lfm_highres", highres)
        object.__setattr__(self, "ai_bounds", (lower, upper))
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "decoder_max_abs_error", error)


def load_zone_ai_bounds(path: str | Path) -> dict[str, tuple[float, float]]:
    """Read only the zone clipping bounds needed by the anchored decoder."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    models = payload.get("zone_models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError(f"calibration lacks zone_models: {source}")
    result: dict[str, tuple[float, float]] = {}
    for zone_id, model in models.items():
        if not isinstance(model, Mapping) or not isinstance(model.get("ai_bounds"), Mapping):
            raise ValueError(f"calibration lacks ai_bounds for zone {zone_id!r}")
        bounds = model["ai_bounds"]
        lower = float(bounds["p01"])
        upper = float(bounds["p99"])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"calibration ai_bounds are invalid for zone {zone_id!r}")
        result[str(zone_id)] = (lower, upper)
    return result


def _zone_coordinate(axis: np.ndarray, sample: StructuredSample) -> np.ndarray:
    return (axis - sample.zone.top) / (sample.zone.bottom - sample.zone.top)


def _fit_lfm_anchor(sample: StructuredSample) -> tuple[float, float, np.ndarray, np.ndarray]:
    model_axis = np.asarray(sample.observed.sample_axis.coordinates, dtype=np.float64)
    zeta_model = _zone_coordinate(model_axis, sample)
    valid = (
        np.asarray(sample.observed.observed_valid, dtype=bool)
        & (model_axis >= sample.zone.top)
        & (model_axis <= sample.zone.bottom)
        & np.isfinite(sample.observed.lfm)
    )
    design = np.column_stack(
        (np.ones(np.count_nonzero(valid), dtype=np.float64), 2.0 * zeta_model[valid] - 1.0)
    )
    if design.shape[0] < 2 or np.linalg.matrix_rank(design) != 2:
        raise ValueError("zone-linear LFM anchor requires two independent valid samples.")
    coefficients, *_ = np.linalg.lstsq(
        design,
        np.asarray(sample.observed.lfm, dtype=np.float64)[valid],
        rcond=None,
    )
    a_lfm, b_lfm = (float(value) for value in coefficients)
    highres_axis = np.asarray(sample.latent.latent_axis.coordinates, dtype=np.float64)
    background_model = a_lfm + b_lfm * (2.0 * zeta_model - 1.0)
    background_highres = a_lfm + b_lfm * (
        2.0 * _zone_coordinate(highres_axis, sample) - 1.0
    )
    return a_lfm, b_lfm, background_model, background_highres


def _segment_basis(
    sample: StructuredSample,
    segment: SegmentTruth,
) -> tuple[np.ndarray, np.ndarray]:
    object_ids = np.asarray(sample.latent.object_id)
    zone_valid = np.asarray(sample.zone.zone_valid, dtype=bool)
    indices = np.flatnonzero(zone_valid & (object_ids == int(segment.object_id)))
    if indices.size == 0:
        raise ValueError(
            f"segment {segment.zone_id}/{segment.object_id} has no high-resolution samples."
        )
    xi = np.asarray(sample.latent.object_xi, dtype=np.float64)[indices]
    if np.any(~np.isfinite(xi)) or np.any((xi < 0.0) | (xi > 1.0)):
        raise ValueError("segment object_xi contains invalid values.")
    basis = np.column_stack(
        (
            np.ones(indices.size, dtype=np.float64),
            2.0 * xi - 1.0,
            np.sin(np.pi * xi),
        )
    )
    return indices, basis


def _rebase_segment(
    sample: StructuredSample,
    segment: SegmentTruth,
    *,
    background_lfm_highres: np.ndarray,
    condition_limit: float,
    residual_tolerance: float,
) -> AnchoredSegment:
    indices, basis = _segment_basis(sample, segment)
    highres_axis = np.asarray(sample.latent.latent_axis.coordinates, dtype=np.float64)
    zeta = _zone_coordinate(highres_axis[indices], sample)
    original_background = sample.zone.background_a + sample.zone.background_b * (
        2.0 * zeta - 1.0
    )
    delta_background = original_background - background_lfm_highres[indices]
    linear_basis = basis[:, :2]
    linear_rank = int(np.linalg.matrix_rank(linear_basis))
    if indices.size == 1:
        adjustment = np.asarray([delta_background[0], 0.0], dtype=np.float64)
    elif linear_rank == 2:
        adjustment, *_ = np.linalg.lstsq(linear_basis, delta_background, rcond=None)
    else:
        raise ValueError("multi-sample segment has rank-deficient linear rebase basis.")
    residual = float(np.max(np.abs(linear_basis @ adjustment - delta_background)))
    if residual > float(residual_tolerance):
        raise ValueError(
            f"segment {segment.zone_id}/{segment.object_id} rebase residual "
            f"{residual:.3e} exceeds {residual_tolerance:.3e}."
        )
    parameters = np.asarray(
        [
            float(segment.c0_effective[0]) + float(adjustment[0]),
            float(segment.c1_effective[0]) + float(adjustment[1]),
            float(segment.c2_effective[0]),
        ],
        dtype=np.float64,
    )
    rank = int(np.linalg.matrix_rank(basis))
    condition = float(np.linalg.cond(basis)) if rank == 3 else float("inf")
    canonical_valid = bool(segment.segment_supervision_valid)
    profile_valid = canonical_valid and bool(
        np.all(np.isfinite(sample.latent.log_ai_highres_truth[indices]))
    )
    clipped = bool(np.any(np.asarray(sample.latent.clipping_mask, dtype=bool)[indices]))
    parameter_valid = (
        profile_valid
        and rank == 3
        and condition <= float(condition_limit)
        and not clipped
    )
    return AnchoredSegment(
        source=segment,
        sample_indices=indices,
        basis=basis,
        effective_parameters_lfm=parameters,
        profile_supervision_valid=profile_valid,
        parameter_supervision_valid=parameter_valid,
        parameter_identifiability_rank=rank,
        parameter_basis_condition=condition,
        rebase_residual=residual,
    )


def decode_lfm_anchored_numpy(
    sample: LfmAnchoredStructuredSample,
    parameters: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Decode effective LFM-relative coefficients without generator projection."""
    coefficients = (
        tuple(item.effective_parameters_lfm for item in sample.segments)
        if parameters is None
        else tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in parameters)
    )
    if len(coefficients) != len(sample.segments):
        raise ValueError("parameter count differs from anchored segment count.")
    output = np.asarray(sample.background_lfm_highres, dtype=np.float64).copy()
    zone_valid = np.asarray(sample.source.zone.zone_valid, dtype=bool)
    for segment, values in zip(sample.segments, coefficients, strict=True):
        if values.shape != (3,) or np.any(~np.isfinite(values)):
            raise ValueError("each anchored decoder parameter vector must be finite [3].")
        output[segment.sample_indices] += segment.basis @ values
    output[zone_valid] = np.clip(
        output[zone_valid],
        sample.ai_bounds[0],
        sample.ai_bounds[1],
    )
    output[~zone_valid] = np.nan
    return output


def decode_lfm_anchored_torch(
    background_highres: Any,
    segment_basis: Any,
    segment_mask: Any,
    parameters: Any,
    zone_valid: Any,
    ai_bounds: Any,
) -> Any:
    """Vectorized differentiable decoder for padded teacher-forcing batches."""
    import torch

    if background_highres.ndim != 2:
        raise ValueError("background_highres must have shape [batch, highres].")
    if segment_basis.ndim != 4 or segment_basis.shape[-1] != 3:
        raise ValueError("segment_basis must have shape [batch, segment, highres, 3].")
    if segment_mask.shape != segment_basis.shape[:-1]:
        raise ValueError("segment_mask must match segment_basis without its coefficient axis.")
    if parameters.shape != (*segment_basis.shape[:2], 3):
        raise ValueError("parameters must have shape [batch, segment, 3].")
    profile = torch.einsum("bshc,bsc->bsh", segment_basis, parameters)
    profile = torch.where(segment_mask, profile, torch.zeros_like(profile))
    values = background_highres + torch.sum(profile, dim=1)
    lower = ai_bounds[:, 0].unsqueeze(-1)
    upper = ai_bounds[:, 1].unsqueeze(-1)
    values = torch.minimum(torch.maximum(values, lower), upper)
    return torch.where(zone_valid, values, torch.full_like(values, float("nan")))


def anchor_to_lfm(
    sample: StructuredSample,
    *,
    ai_bounds: Mapping[str, tuple[float, float]],
    condition_limit: float = 100.0,
    residual_tolerance: float = 1e-8,
    decoder_atol: float = 1e-5,
) -> LfmAnchoredStructuredSample:
    """Create the sole Stage-1 LFM-relative supervision representation."""
    if sample.zone.zone_id not in ai_bounds:
        raise KeyError(f"missing AI bounds for zone {sample.zone.zone_id!r}")
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("condition_limit must be finite and greater than one.")
    a_lfm, b_lfm, background_model, background_highres = _fit_lfm_anchor(sample)
    anchored_segments = tuple(
        _rebase_segment(
            sample,
            segment,
            background_lfm_highres=background_highres,
            condition_limit=condition_limit,
            residual_tolerance=residual_tolerance,
        )
        for segment in sample.segments
    )
    provisional = LfmAnchoredStructuredSample(
        source=sample,
        background_a_lfm=a_lfm,
        background_b_lfm=b_lfm,
        background_lfm_model=background_model,
        background_lfm_highres=background_highres,
        ai_bounds=ai_bounds[sample.zone.zone_id],
        segments=anchored_segments,
        decoder_max_abs_error=0.0,
    )
    decoded = decode_lfm_anchored_numpy(provisional)
    valid = np.asarray(sample.zone.zone_valid, dtype=bool)
    truth = np.asarray(sample.latent.log_ai_highres_truth, dtype=np.float64)
    max_error = float(np.max(np.abs(decoded[valid] - truth[valid])))
    if max_error > float(decoder_atol):
        raise ValueError(
            f"LFM-anchored decoder parity error {max_error:.3e} exceeds "
            f"{float(decoder_atol):.3e}."
        )
    return LfmAnchoredStructuredSample(
        source=sample,
        background_a_lfm=a_lfm,
        background_b_lfm=b_lfm,
        background_lfm_model=background_model,
        background_lfm_highres=background_highres,
        ai_bounds=ai_bounds[sample.zone.zone_id],
        segments=anchored_segments,
        decoder_max_abs_error=max_error,
    )


__all__ = [
    "AnchoredSegment",
    "LfmAnchoredStructuredSample",
    "PARAMETER_NAMES",
    "anchor_to_lfm",
    "decode_lfm_anchored_numpy",
    "decode_lfm_anchored_torch",
    "load_zone_ai_bounds",
]
