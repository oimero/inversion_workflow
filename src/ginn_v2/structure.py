"""Single-trace structured inference built around the exact HSMM seam."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ginn_v2.anchor import decode_lfm_anchored_torch
from ginn_v2.hsmm import (
    HsmmPrior,
    HsmmResult,
    HsmmSegment,
    canonicalize_hsmm_segments,
    exact_hsmm,
    hsmm_log_partition,
    hsmm_path_score,
)
from ginn_v2.model import (
    DirectionalEvidence,
    SingleTraceStructuredModel,
    TeacherForcingLoss,
    TeacherForcingLossConfig,
    TeacherForcingOutput,
    TorchTeacherForcingBatch,
    project_highres_torch,
    teacher_forcing_loss,
)


@dataclass(frozen=True)
class StructuredLossConfig:
    """Weights for Step-2 truth structure and existing profile objectives."""

    emission_weight: float = 1.0
    boundary_weight: float = 0.5
    hsmm_nll_weight: float = 1.0
    teacher_forcing_weight: float = 0.0
    teacher_forcing: TeacherForcingLossConfig = TeacherForcingLossConfig()

    def __post_init__(self) -> None:
        for name in (
            "emission_weight",
            "boundary_weight",
            "hsmm_nll_weight",
            "teacher_forcing_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not any(
            float(getattr(self, name)) > 0.0
            for name in ("emission_weight", "boundary_weight", "hsmm_nll_weight")
        ):
            raise ValueError("at least one structure loss weight must be positive.")


@dataclass(frozen=True)
class StructuredLoss:
    total: torch.Tensor
    emission_cross_entropy: torch.Tensor
    boundary_binary_cross_entropy: torch.Tensor
    hsmm_negative_log_likelihood: torch.Tensor
    teacher_forcing: TeacherForcingLoss
    zone_count: int


@dataclass(frozen=True)
class CenterTracePosterior:
    """One batch's posterior-consensus structures and profile parameterization."""

    evidence: DirectionalEvidence
    hsmm_results: tuple[HsmmResult, ...]
    map_state_highres: torch.Tensor
    state_marginal_highres: torch.Tensor
    boundary_marginal_highres: torch.Tensor
    predicted_segment_batch: TorchTeacherForcingBatch
    predicted_profile: TeacherForcingOutput


def _zone_indices(mask: torch.Tensor) -> torch.Tensor:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("structured inference received an empty zone mask.")
    expected = torch.arange(
        int(indices[0].item()),
        int(indices[-1].item()) + 1,
        device=indices.device,
    )
    if not torch.equal(indices, expected):
        raise ValueError("HSMM requires one contiguous zone support.")
    return indices


def truth_hsmm_segments(
    batch: TorchTeacherForcingBatch,
    batch_index: int,
) -> tuple[HsmmSegment, ...]:
    """Convert truth masks to one canonical zone-local HSMM state path.

    Event rows remain separate in the teacher-forcing tensors.  When a
    pinchout removes an intervening event on one trace, two adjacent event
    rows can have the same state; the HSMM path merges that run because it
    models state runs rather than event identity.
    """
    zone = _zone_indices(batch.zone_valid[batch_index])
    first_zone = int(zone[0].item())
    output: list[HsmmSegment] = []
    valid_segments = torch.nonzero(
        batch.segment_valid[batch_index],
        as_tuple=False,
    ).flatten()
    for segment_index in valid_segments.tolist():
        indices = torch.nonzero(
            batch.segment_mask[batch_index, segment_index]
            & batch.zone_valid[batch_index],
            as_tuple=False,
        ).flatten()
        if indices.numel() == 0:
            raise ValueError("truth segment is valid but has no zone samples.")
        output.append(
            HsmmSegment(
                state_id=int(batch.state_id[batch_index, segment_index].item()),
                start=int(indices[0].item()) - first_zone,
                stop=int(indices[-1].item()) - first_zone + 1,
            )
        )
    if not output:
        raise ValueError("truth batch contains no valid structured segments.")
    canonical = canonicalize_hsmm_segments(output)
    if canonical[0].start != 0 or canonical[-1].stop != zone.numel():
        raise ValueError("truth segments do not cover the complete HSMM zone.")
    return canonical


def _teacher_output_from_evidence(
    model: SingleTraceStructuredModel,
    evidence: DirectionalEvidence,
    batch: TorchTeacherForcingBatch,
) -> TeacherForcingOutput:
    mean, std = model.parameterize_segments(
        evidence.center_feature_sequence,
        batch,
    )
    decoded = decode_lfm_anchored_torch(
        batch.background_highres,
        batch.segment_basis,
        batch.segment_mask,
        mean,
        batch.zone_valid,
        batch.ai_bounds,
    )
    projected, support = project_highres_torch(
        decoded,
        batch.zone_valid,
        factor=batch.projection_factor,
    )
    return TeacherForcingOutput(
        parameter_mean=mean,
        parameter_std=std,
        decoded_highres=decoded,
        projected_log_ai=projected,
        projection_support=support,
        interface_jump_mean=evidence.interface_jump_mean,
        interface_jump_std=evidence.interface_jump_std,
        interface_polarity_logits=evidence.interface_polarity_log_potential.transpose(
            1, 2
        ),
    )


def balanced_emission_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Give each state present in the batch equal weight."""
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError("emission logits must have shape [batch, sample, 3].")
    if target.shape != logits.shape[:2] or valid.shape != target.shape:
        raise ValueError("emission target/mask shapes do not match logits.")
    terms = F.cross_entropy(
        logits.transpose(1, 2),
        torch.clamp(target, min=0, max=2),
        reduction="none",
    )
    state_losses: list[torch.Tensor] = []
    for state_id in range(3):
        state_mask = valid & (target == state_id)
        if bool(torch.any(state_mask).item()):
            state_losses.append(torch.mean(terms[state_mask]))
    if not state_losses:
        raise ValueError("balanced emission supervision has no valid state samples.")
    return torch.mean(torch.stack(state_losses))


def soft_boundary_supervision(
    batch: TorchTeacherForcingBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build triangular boundary targets over one model-grid interval."""
    target = torch.zeros_like(batch.background_highres)
    valid = batch.zone_valid.clone()
    radius = int(batch.projection_factor)
    if radius <= 0:
        raise ValueError("projection_factor must be positive.")
    for batch_index in range(batch.background_highres.shape[0]):
        zone_indices = _zone_indices(batch.zone_valid[batch_index])
        for segment in truth_hsmm_segments(batch, batch_index)[1:]:
            center = int(zone_indices[segment.start].item())
            for offset in range(-radius, radius + 1):
                sample = center + offset
                if (
                    sample < 0
                    or sample >= target.shape[1]
                    or not bool(batch.zone_valid[batch_index, sample].item())
                ):
                    continue
                weight = 1.0 - abs(offset) / float(radius + 1)
                target[batch_index, sample] = torch.maximum(
                    target[batch_index, sample],
                    target.new_tensor(weight),
                )
        # The first sample is a forced zone start, not an interior boundary.
        valid[batch_index, int(zone_indices[0].item())] = False
    return target, valid


def structured_training_loss(
    model: SingleTraceStructuredModel,
    batch: TorchTeacherForcingBatch,
    prior: HsmmPrior,
    config: StructuredLossConfig,
) -> tuple[StructuredLoss, DirectionalEvidence, TeacherForcingOutput]:
    """Train evidence on truth paths while parameterizing only truth+jitter segments."""
    evidence = model.encode_patch(batch)
    teacher_output = _teacher_output_from_evidence(model, evidence, batch)
    teacher_loss = teacher_forcing_loss(
        teacher_output,
        batch,
        config.teacher_forcing,
    )
    state_mask = batch.zone_valid & (batch.truth_state_highres >= 0)
    emission_loss = balanced_emission_cross_entropy(
        evidence.emission_log_potential,
        batch.truth_state_highres,
        state_mask,
    )

    boundary_target, boundary_valid = soft_boundary_supervision(batch)
    zone_losses: list[torch.Tensor] = []
    for batch_index, zone_id in enumerate(batch.zone_ids):
        zone_indices = _zone_indices(batch.zone_valid[batch_index])
        truth_segments = truth_hsmm_segments(batch, batch_index)
        local_emission = evidence.emission_log_potential[
            batch_index, zone_indices
        ]
        # The auxiliary boundary classifier is not a calibrated likelihood
        # ratio.  Feeding its class-imbalanced raw logits into the HSMM
        # double-counts segment frequency already represented by the duration
        # prior.  Step 2 therefore conditions structure on emission evidence
        # and the fixed semi-Markov prior only.
        local_boundary = torch.zeros_like(
            evidence.boundary_log_potential[batch_index, zone_indices]
        )
        zone_prior = prior.zone(zone_id)
        zone_losses.append(
            (
                hsmm_log_partition(local_emission, local_boundary, zone_prior)
                - hsmm_path_score(
                    local_emission,
                    local_boundary,
                    zone_prior,
                    truth_segments,
                )
            )
            / float(zone_indices.numel())
        )
    positive = boundary_valid & (boundary_target > 0.0)
    negative = boundary_valid & (boundary_target == 0.0)
    if not bool(torch.any(positive).item()) or not bool(torch.any(negative).item()):
        raise ValueError("balanced boundary supervision requires both classes.")
    boundary_terms = F.binary_cross_entropy_with_logits(
        evidence.boundary_log_potential,
        boundary_target,
        reduction="none",
    )
    positive_loss = boundary_terms[positive].mean()
    negative_loss = boundary_terms[negative].mean()
    boundary_loss = 0.5 * (positive_loss + negative_loss)
    hsmm_nll = torch.mean(torch.stack(zone_losses))
    total = (
        config.teacher_forcing_weight * teacher_loss.total
        + config.emission_weight * emission_loss
        + config.boundary_weight * boundary_loss
        + config.hsmm_nll_weight * hsmm_nll
    )
    return (
        StructuredLoss(
            total=total,
            emission_cross_entropy=emission_loss,
            boundary_binary_cross_entropy=boundary_loss,
            hsmm_negative_log_likelihood=hsmm_nll,
            teacher_forcing=teacher_loss,
            zone_count=len(zone_losses),
        ),
        evidence,
        teacher_output,
    )


def _basis_for_duration(
    duration: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if int(duration) <= 0:
        raise ValueError("predicted segment duration must be positive.")
    if int(duration) == 1:
        xi = torch.full((1,), 0.5, dtype=dtype, device=device)
    else:
        xi = torch.linspace(0.0, 1.0, int(duration), dtype=dtype, device=device)
    return torch.stack(
        (
            torch.ones_like(xi),
            2.0 * xi - 1.0,
            torch.sin(torch.pi * xi),
        ),
        dim=-1,
    )


def build_predicted_segment_batch(
    batch: TorchTeacherForcingBatch,
    segmentations: Sequence[Sequence[HsmmSegment]],
) -> TorchTeacherForcingBatch:
    """Replace truth segment descriptors with one legal MAP table per trace."""
    batch_size, highres_size = batch.zone_valid.shape
    if len(segmentations) != batch_size:
        raise ValueError("predicted segmentation count differs from batch size.")
    maximum_segments = max(len(items) for items in segmentations)
    if maximum_segments <= 0:
        raise ValueError("predicted segmentations must be non-empty.")
    shape = (batch_size, maximum_segments, highres_size)
    basis = torch.zeros(
        (*shape, 3),
        dtype=batch.background_highres.dtype,
        device=batch.background_highres.device,
    )
    mask = torch.zeros(shape, dtype=torch.bool, device=batch.zone_valid.device)
    valid = torch.zeros(
        (batch_size, maximum_segments),
        dtype=torch.bool,
        device=batch.zone_valid.device,
    )
    state = torch.zeros(
        (batch_size, maximum_segments),
        dtype=torch.long,
        device=batch.zone_valid.device,
    )
    duration = torch.zeros(
        (batch_size, maximum_segments),
        dtype=batch.background_highres.dtype,
        device=batch.background_highres.device,
    )
    extent = torch.zeros(
        (batch_size, maximum_segments, 2),
        dtype=batch.background_highres.dtype,
        device=batch.background_highres.device,
    )
    for batch_index, segments in enumerate(segmentations):
        zone = _zone_indices(batch.zone_valid[batch_index])
        if (
            not segments
            or segments[0].start != 0
            or segments[-1].stop != zone.numel()
        ):
            raise ValueError("predicted segments do not cover their complete zone.")
        previous_stop = 0
        previous_state: int | None = None
        for segment_index, segment in enumerate(segments):
            if segment.start != previous_stop:
                raise ValueError("predicted segments are not contiguous.")
            if previous_state == segment.state_id:
                raise ValueError("predicted HSMM contains adjacent equal states.")
            global_indices = zone[segment.start : segment.stop]
            count = int(global_indices.numel())
            mask[batch_index, segment_index, global_indices] = True
            basis[
                batch_index, segment_index, global_indices
            ] = _basis_for_duration(
                count,
                dtype=basis.dtype,
                device=basis.device,
            )
            valid[batch_index, segment_index] = True
            state[batch_index, segment_index] = segment.state_id
            duration[batch_index, segment_index] = count / float(zone.numel())
            extent[batch_index, segment_index] = torch.as_tensor(
                (
                    segment.start / float(zone.numel()),
                    segment.stop / float(zone.numel()),
                ),
                dtype=extent.dtype,
                device=extent.device,
            )
            previous_stop = segment.stop
            previous_state = segment.state_id
    return replace(
        batch,
        segment_basis=basis,
        segment_mask=mask,
        pooling_mask=mask,
        segment_valid=valid,
        state_id=state,
        duration_fraction=duration,
        extent_fraction=extent,
        target_parameters=torch.zeros(
            (batch_size, maximum_segments, 3),
            dtype=batch.target_parameters.dtype,
            device=batch.target_parameters.device,
        ),
        parameter_supervision_valid=torch.zeros_like(valid),
        profile_supervision_valid=torch.zeros_like(valid),
    )


def infer_center_trace(
    model: SingleTraceStructuredModel,
    batch: TorchTeacherForcingBatch,
    prior: HsmmPrior,
    *,
    evidence: DirectionalEvidence | None = None,
) -> CenterTracePosterior:
    """Produce one posterior-consensus segmentation for every center trace."""
    directional = model.encode_patch(batch) if evidence is None else evidence
    batch_size, highres_size = batch.zone_valid.shape
    map_state = torch.full(
        (batch_size, highres_size),
        -1,
        dtype=torch.long,
        device=batch.zone_valid.device,
    )
    state_marginal = torch.zeros(
        (batch_size, highres_size, 3),
        dtype=directional.emission_log_potential.dtype,
        device=directional.emission_log_potential.device,
    )
    boundary_marginal = torch.zeros(
        (batch_size, highres_size),
        dtype=directional.emission_log_potential.dtype,
        device=directional.emission_log_potential.device,
    )
    results: list[HsmmResult] = []
    for batch_index, zone_id in enumerate(batch.zone_ids):
        zone = _zone_indices(batch.zone_valid[batch_index])
        result = exact_hsmm(
            directional.emission_log_potential[batch_index, zone],
            torch.zeros_like(
                directional.boundary_log_potential[batch_index, zone]
            ),
            prior.zone(zone_id),
        )
        results.append(result)
        state_marginal[batch_index, zone] = result.state_marginal
        boundary_marginal[batch_index, zone] = result.boundary_marginal
        for segment in result.consensus_segments:
            map_state[
                batch_index, zone[segment.start : segment.stop]
            ] = segment.state_id
    predicted_batch = build_predicted_segment_batch(
        batch,
        [result.consensus_segments for result in results],
    )
    predicted_output = _teacher_output_from_evidence(
        model,
        directional,
        predicted_batch,
    )
    return CenterTracePosterior(
        evidence=directional,
        hsmm_results=tuple(results),
        map_state_highres=map_state,
        state_marginal_highres=state_marginal,
        boundary_marginal_highres=boundary_marginal,
        predicted_segment_batch=predicted_batch,
        predicted_profile=predicted_output,
    )


__all__ = [
    "balanced_emission_cross_entropy",
    "CenterTracePosterior",
    "StructuredLoss",
    "StructuredLossConfig",
    "build_predicted_segment_batch",
    "infer_center_trace",
    "soft_boundary_supervision",
    "structured_training_loss",
    "truth_hsmm_segments",
]
