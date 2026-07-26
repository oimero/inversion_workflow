"""Teacher-forced single-trace parameter model for Structured GINN V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cup.synthetic.core.signal import finite_support_fir
from ginn_v2.anchor import decode_lfm_anchored_torch
from ginn_v2.data import TeacherForcingBatch


@dataclass(frozen=True)
class TeacherForcingModelConfig:
    feature_channels: int = 48
    encoder_blocks: int = 4
    state_embedding_channels: int = 8
    hidden_channels: int = 96
    kernel_size: int = 7
    seismic_scale: float = 0.1
    lfm_residual_scale: float = 0.1
    minimum_parameter_std: float = 1e-3
    maximum_parameter_std: float = 0.5
    segment_evidence_mode: str = "mean"
    predict_interface_evidence: bool = False
    maximum_interface_jump_magnitude: float = 0.5
    minimum_interface_jump_std: float = 1e-3
    maximum_interface_jump_std: float = 0.5
    use_seismic: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.use_seismic, bool):
            raise TypeError("use_seismic must be boolean.")
        if not isinstance(self.predict_interface_evidence, bool):
            raise TypeError("predict_interface_evidence must be boolean.")
        if self.segment_evidence_mode not in {"mean", "boundary_aware"}:
            raise ValueError(
                "segment_evidence_mode must be 'mean' or 'boundary_aware'."
            )
        for name in (
            "feature_channels",
            "encoder_blocks",
            "state_embedding_channels",
            "hidden_channels",
            "kernel_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        if self.seismic_scale <= 0.0 or self.lfm_residual_scale <= 0.0:
            raise ValueError("input scales must be positive.")
        if (
            self.minimum_parameter_std <= 0.0
            or self.maximum_parameter_std <= self.minimum_parameter_std
        ):
            raise ValueError("parameter standard-deviation bounds are invalid.")
        if (
            not np.isfinite(self.maximum_interface_jump_magnitude)
            or self.maximum_interface_jump_magnitude <= 0.0
        ):
            raise ValueError("maximum_interface_jump_magnitude must be positive.")
        if (
            self.minimum_interface_jump_std <= 0.0
            or self.maximum_interface_jump_std <= self.minimum_interface_jump_std
        ):
            raise ValueError("interface-jump standard-deviation bounds are invalid.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TeacherForcingModelConfig":
        names = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(names))
        if unknown:
            raise ValueError(f"unknown teacher-forcing model fields: {unknown}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TeacherForcingLossConfig:
    parameter_nll_weight: float = 1.0
    highres_mse_weight: float = 1.0
    projected_mse_weight: float = 1.0
    interface_jump_nll_weight: float = 0.0
    interface_polarity_weight: float = 0.0
    interface_jump_zero_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        values = (
            self.parameter_nll_weight,
            self.highres_mse_weight,
            self.projected_mse_weight,
            self.interface_jump_nll_weight,
            self.interface_polarity_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("teacher-forcing loss weights must be finite and non-negative.")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one teacher-forcing loss weight must be positive.")
        if (
            not np.isfinite(self.interface_jump_zero_tolerance)
            or self.interface_jump_zero_tolerance < 0.0
        ):
            raise ValueError("interface_jump_zero_tolerance must be non-negative.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TeacherForcingLossConfig":
        names = set(cls.__dataclass_fields__)
        unknown = sorted(set(value).difference(names))
        if unknown:
            raise ValueError(f"unknown teacher-forcing loss fields: {unknown}")
        return cls(**dict(value))


@dataclass(frozen=True)
class TorchTeacherForcingBatch:
    seismic: torch.Tensor
    lfm_residual: torch.Tensor
    observed_valid: torch.Tensor
    background_highres: torch.Tensor
    truth_highres: torch.Tensor
    zone_valid: torch.Tensor
    segment_basis: torch.Tensor
    segment_mask: torch.Tensor
    pooling_mask: torch.Tensor
    segment_valid: torch.Tensor
    state_id: torch.Tensor
    duration_fraction: torch.Tensor
    extent_fraction: torch.Tensor
    target_parameters: torch.Tensor
    parameter_supervision_valid: torch.Tensor
    profile_supervision_valid: torch.Tensor
    interface_jump_target: torch.Tensor
    interface_jump_valid: torch.Tensor
    ai_bounds: torch.Tensor
    projected_truth: torch.Tensor
    projected_support: torch.Tensor
    projection_factor: int


@dataclass(frozen=True)
class TeacherForcingOutput:
    parameter_mean: torch.Tensor
    parameter_std: torch.Tensor
    decoded_highres: torch.Tensor
    projected_log_ai: torch.Tensor
    projection_support: torch.Tensor
    interface_jump_mean: torch.Tensor | None
    interface_jump_std: torch.Tensor | None
    interface_polarity_logits: torch.Tensor | None


@dataclass(frozen=True)
class TeacherForcingLoss:
    total: torch.Tensor
    parameter_nll: torch.Tensor
    highres_mse: torch.Tensor
    projected_mse: torch.Tensor
    interface_jump_nll: torch.Tensor
    interface_polarity_loss: torch.Tensor
    parameter_count: int
    highres_count: int
    projected_count: int
    interface_jump_count: int


def batch_to_torch(
    batch: TeacherForcingBatch,
    *,
    device: torch.device,
) -> TorchTeacherForcingBatch:
    def floating(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def boolean(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.bool, device=device)

    return TorchTeacherForcingBatch(
        seismic=floating(batch.seismic),
        lfm_residual=floating(batch.lfm_residual),
        observed_valid=boolean(batch.observed_valid),
        background_highres=floating(batch.background_highres),
        truth_highres=floating(batch.truth_highres),
        zone_valid=boolean(batch.zone_valid),
        segment_basis=floating(batch.segment_basis),
        segment_mask=boolean(batch.segment_mask),
        pooling_mask=boolean(batch.pooling_mask),
        segment_valid=boolean(batch.segment_valid),
        state_id=torch.as_tensor(batch.state_id, dtype=torch.long, device=device),
        duration_fraction=floating(batch.duration_fraction),
        extent_fraction=floating(batch.extent_fraction),
        target_parameters=floating(batch.target_parameters),
        parameter_supervision_valid=boolean(batch.parameter_supervision_valid),
        profile_supervision_valid=boolean(batch.profile_supervision_valid),
        interface_jump_target=floating(batch.interface_jump_target),
        interface_jump_valid=boolean(batch.interface_jump_valid),
        ai_bounds=floating(batch.ai_bounds),
        projected_truth=floating(batch.projected_truth),
        projected_support=boolean(batch.projected_support),
        projection_factor=int(batch.projection_factor),
    )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.normalization = nn.GroupNorm(1, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + F.gelu(self.normalization(self.convolution(values)))


class TeacherForcedParameterModel(nn.Module):
    """Encode one trace and parameterize externally supplied truth/jitter segments."""

    def __init__(self, config: TeacherForcingModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(3, config.feature_channels, 1)
        self.encoder = nn.Sequential(
            *(
                _ResidualBlock(
                    config.feature_channels,
                    config.kernel_size,
                    dilation=2 ** (index % 4),
                )
                for index in range(config.encoder_blocks)
            )
        )
        self.state_embedding = nn.Embedding(3, config.state_embedding_channels)
        if config.segment_evidence_mode == "mean":
            segment_feature_channels = config.feature_channels
        else:
            segment_feature_channels = 8 * config.feature_channels + 4
        head_inputs = (
            segment_feature_channels + config.state_embedding_channels + 3
        )
        self.parameter_head = nn.Sequential(
            nn.Linear(head_inputs, config.hidden_channels),
            nn.GELU(),
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.GELU(),
            nn.Linear(config.hidden_channels, 6),
        )
        self.interface_evidence_head = (
            nn.Conv1d(config.feature_channels, 5, 1)
            if config.predict_interface_evidence
            else None
        )

    def encode_trace(self, batch: TorchTeacherForcingBatch) -> torch.Tensor:
        mask = batch.observed_valid.unsqueeze(1)
        seismic = (
            batch.seismic
            if self.config.use_seismic
            else torch.zeros_like(batch.seismic)
        )
        inputs = torch.stack(
            (
                seismic / self.config.seismic_scale,
                batch.lfm_residual / self.config.lfm_residual_scale,
                batch.observed_valid.to(dtype=batch.seismic.dtype),
            ),
            dim=1,
        )
        inputs = torch.where(mask, inputs, torch.zeros_like(inputs))
        features = self.encoder(self.input_projection(inputs))
        features = torch.where(mask, features, torch.zeros_like(features))
        return F.interpolate(
            features,
            size=batch.background_highres.shape[-1],
            mode="linear",
            align_corners=True,
        )

    def parameterize_segments(
        self,
        feature_sequence: torch.Tensor,
        batch: TorchTeacherForcingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.segment_evidence_mode == "mean":
            pooled = self._mean_segment_evidence(feature_sequence, batch)
        else:
            pooled = self._boundary_aware_segment_evidence(
                feature_sequence,
                batch,
            )
        state = self.state_embedding(torch.clamp(batch.state_id, min=0, max=2))
        descriptors = torch.cat(
            (
                pooled,
                state,
                batch.duration_fraction.unsqueeze(-1),
                batch.extent_fraction,
            ),
            dim=-1,
        )
        raw = self.parameter_head(descriptors)
        mean = raw[..., :3]
        fraction = torch.sigmoid(raw[..., 3:])
        std = self.config.minimum_parameter_std + fraction * (
            self.config.maximum_parameter_std
            - self.config.minimum_parameter_std
        )
        valid = batch.segment_valid.unsqueeze(-1)
        return (
            torch.where(valid, mean, torch.zeros_like(mean)),
            torch.where(valid, std, torch.ones_like(std)),
        )

    @staticmethod
    def _mean_segment_evidence(
        feature_sequence: torch.Tensor,
        batch: TorchTeacherForcingBatch,
    ) -> torch.Tensor:
        weights = batch.pooling_mask.to(dtype=feature_sequence.dtype)
        denominator = torch.clamp(weights.sum(dim=-1, keepdim=True), min=1.0)
        pooled = torch.einsum("bch,bsh->bsc", feature_sequence, weights)
        return pooled / denominator

    @staticmethod
    def _gather_features(
        feature_sequence: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        channels = feature_sequence.shape[1]
        gather_index = indices.unsqueeze(1).expand(-1, channels, -1)
        return torch.gather(feature_sequence, dim=2, index=gather_index).transpose(
            1, 2
        )

    def _boundary_aware_segment_evidence(
        self,
        feature_sequence: torch.Tensor,
        batch: TorchTeacherForcingBatch,
    ) -> torch.Tensor:
        mask = batch.pooling_mask
        weights = mask.to(dtype=feature_sequence.dtype)
        counts = weights.sum(dim=-1, keepdim=True)
        safe_counts = torch.clamp(counts, min=1.0)
        mean = torch.einsum("bch,bsh->bsc", feature_sequence, weights)
        mean = mean / safe_counts

        ranks = torch.cumsum(weights, dim=-1) - 1.0
        denominator = torch.clamp(counts - 1.0, min=1.0)
        xi = ranks / denominator
        singleton = counts <= 1.0
        xi = torch.where(singleton, torch.full_like(xi, 0.5), xi)
        xi = torch.where(mask, xi, torch.zeros_like(xi))

        linear = (2.0 * xi - 1.0) * weights
        linear_norm = torch.clamp(
            torch.sum(linear.square(), dim=-1, keepdim=True),
            min=1.0,
        )
        linear_moment = (
            torch.einsum("bch,bsh->bsc", feature_sequence, linear)
            / linear_norm
        )

        sine = torch.sin(torch.pi * xi) * weights
        sine_mean = torch.sum(sine, dim=-1, keepdim=True) / safe_counts
        centered_sine = sine - sine_mean * weights
        linear_projection = (
            torch.sum(centered_sine * linear, dim=-1, keepdim=True)
            / linear_norm
        )
        orthogonal_sine = centered_sine - linear_projection * linear
        sine_norm = torch.clamp(
            torch.sum(orthogonal_sine.square(), dim=-1, keepdim=True),
            min=1.0,
        )
        sine_moment = (
            torch.einsum("bch,bsh->bsc", feature_sequence, orthogonal_sine)
            / sine_norm
        )

        highres_size = mask.shape[-1]
        valid = batch.segment_valid
        first = torch.argmax(mask.to(dtype=torch.int64), dim=-1)
        last = highres_size - 1 - torch.argmax(
            torch.flip(mask, dims=(-1,)).to(dtype=torch.int64),
            dim=-1,
        )
        center = torch.round(0.5 * (first + last).to(dtype=torch.float32)).to(
            dtype=torch.long
        )
        top_outside_index = torch.clamp(first - 1, min=0)
        bottom_outside_index = torch.clamp(last + 1, max=highres_size - 1)

        top_inside = self._gather_features(feature_sequence, first)
        bottom_inside = self._gather_features(feature_sequence, last)
        center_feature = self._gather_features(feature_sequence, center)
        top_outside = self._gather_features(
            feature_sequence,
            top_outside_index,
        )
        bottom_outside = self._gather_features(
            feature_sequence,
            bottom_outside_index,
        )

        top_outside_valid = valid & (first > 0)
        bottom_outside_valid = valid & (last + 1 < highres_size)
        top_zone_valid = torch.gather(
            batch.zone_valid,
            dim=1,
            index=top_outside_index,
        )
        bottom_zone_valid = torch.gather(
            batch.zone_valid,
            dim=1,
            index=bottom_outside_index,
        )
        top_outside_valid &= top_zone_valid
        bottom_outside_valid &= bottom_zone_valid
        top_outside = torch.where(
            top_outside_valid.unsqueeze(-1),
            top_outside,
            torch.zeros_like(top_outside),
        )
        bottom_outside = torch.where(
            bottom_outside_valid.unsqueeze(-1),
            bottom_outside,
            torch.zeros_like(bottom_outside),
        )
        inside_valid = valid.unsqueeze(-1)
        summaries = (
            mean,
            linear_moment,
            sine_moment,
            center_feature,
            torch.where(inside_valid, top_inside, torch.zeros_like(top_inside)),
            top_outside,
            torch.where(
                inside_valid,
                bottom_inside,
                torch.zeros_like(bottom_inside),
            ),
            bottom_outside,
        )
        validity = torch.stack(
            (
                valid,
                top_outside_valid,
                valid,
                bottom_outside_valid,
            ),
            dim=-1,
        ).to(dtype=feature_sequence.dtype)
        return torch.cat((*summaries, validity), dim=-1)

    def _interface_evidence(
        self,
        feature_sequence: torch.Tensor,
        batch: TorchTeacherForcingBatch,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.interface_evidence_head is None:
            return None, None, None
        raw = self.interface_evidence_head(feature_sequence)
        mean = self.config.maximum_interface_jump_magnitude * torch.tanh(raw[:, 0])
        fraction = torch.sigmoid(raw[:, 1])
        std = self.config.minimum_interface_jump_std + fraction * (
            self.config.maximum_interface_jump_std
            - self.config.minimum_interface_jump_std
        )
        logits = raw[:, 2:5]
        valid = batch.zone_valid
        return (
            torch.where(valid, mean, torch.zeros_like(mean)),
            torch.where(valid, std, torch.ones_like(std)),
            torch.where(valid.unsqueeze(1), logits, torch.zeros_like(logits)),
        )

    def forward(self, batch: TorchTeacherForcingBatch) -> TeacherForcingOutput:
        features = self.encode_trace(batch)
        mean, std = self.parameterize_segments(features, batch)
        jump_mean, jump_std, polarity_logits = self._interface_evidence(
            features,
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
            interface_jump_mean=jump_mean,
            interface_jump_std=jump_std,
            interface_polarity_logits=polarity_logits,
        )


def project_highres_torch(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the canonical finite-support FIR at nested model-grid centers."""
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("project_highres_torch expects matching [batch, highres] tensors.")
    taps = torch.as_tensor(
        finite_support_fir(int(factor)),
        dtype=values.dtype,
        device=values.device,
    )
    half = taps.numel() // 2
    clean = torch.where(valid, values, torch.zeros_like(values)).unsqueeze(1)
    filtered = F.conv1d(clean, taps.view(1, 1, -1), padding=half).squeeze(1)
    support_mass = F.conv1d(
        valid.to(dtype=values.dtype).unsqueeze(1),
        torch.ones_like(taps).view(1, 1, -1),
        padding=half,
    ).squeeze(1)
    supported = support_mass >= float(taps.numel()) - 0.5
    global_support = torch.zeros_like(valid)
    if values.shape[-1] > 2 * half:
        global_support[:, half : values.shape[-1] - half] = True
    supported &= global_support
    return filtered[:, :: int(factor)], supported[:, :: int(factor)]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    count = int(torch.count_nonzero(mask).item())
    if count == 0:
        return values.sum() * 0.0, 0
    return torch.mean(values[mask]), count


def interface_polarity_classes(
    jump: torch.Tensor,
    *,
    zero_tolerance: float,
) -> torch.Tensor:
    """Map signed downward log-AI jumps to decrease/neutral/increase classes."""
    target = torch.ones_like(jump, dtype=torch.long)
    target = torch.where(jump < -float(zero_tolerance), torch.zeros_like(target), target)
    return torch.where(
        jump > float(zero_tolerance),
        torch.full_like(target, 2),
        target,
    )


def teacher_forcing_loss(
    output: TeacherForcingOutput,
    batch: TorchTeacherForcingBatch,
    config: TeacherForcingLossConfig,
) -> TeacherForcingLoss:
    variance = output.parameter_std.square()
    parameter_terms = 0.5 * (
        (output.parameter_mean - batch.target_parameters).square() / variance
        + torch.log(variance)
    )
    parameter_mask = batch.parameter_supervision_valid.unsqueeze(-1).expand_as(
        parameter_terms
    )
    parameter_nll, parameter_count = _masked_mean(parameter_terms, parameter_mask)
    profile_segments = (
        batch.segment_mask
        & batch.profile_supervision_valid.unsqueeze(-1)
    )
    profile_mask = torch.any(profile_segments, dim=1) & batch.zone_valid
    highres_terms = (
        torch.nan_to_num(output.decoded_highres, nan=0.0) - batch.truth_highres
    ).square()
    highres_mse, highres_count = _masked_mean(highres_terms, profile_mask)
    projection_mask = (
        output.projection_support
        & batch.projected_support
        & torch.isfinite(output.projected_log_ai)
    )
    projected_terms = (
        output.projected_log_ai - batch.projected_truth
    ).square()
    projected_mse, projected_count = _masked_mean(projected_terms, projection_mask)
    jump_nll = output.decoded_highres.sum() * 0.0
    polarity_loss = output.decoded_highres.sum() * 0.0
    jump_count = (
        int(torch.count_nonzero(batch.interface_jump_valid).item())
        if output.interface_jump_mean is not None
        else 0
    )
    if (
        config.interface_jump_nll_weight > 0.0
        or config.interface_polarity_weight > 0.0
    ):
        if (
            output.interface_jump_mean is None
            or output.interface_jump_std is None
            or output.interface_polarity_logits is None
        ):
            raise ValueError(
                "interface evidence loss requires predict_interface_evidence=true."
            )
        jump_variance = output.interface_jump_std.square()
        jump_terms = 0.5 * (
            (
                output.interface_jump_mean - batch.interface_jump_target
            ).square()
            / jump_variance
            + torch.log(jump_variance)
        )
        jump_nll, _ = _masked_mean(jump_terms, batch.interface_jump_valid)
        polarity_target = interface_polarity_classes(
            batch.interface_jump_target,
            zero_tolerance=config.interface_jump_zero_tolerance,
        )
        polarity_terms = F.cross_entropy(
            output.interface_polarity_logits,
            polarity_target,
            reduction="none",
        )
        polarity_loss, _ = _masked_mean(
            polarity_terms,
            batch.interface_jump_valid,
        )
    total = (
        config.parameter_nll_weight * parameter_nll
        + config.highres_mse_weight * highres_mse
        + config.projected_mse_weight * projected_mse
        + config.interface_jump_nll_weight * jump_nll
        + config.interface_polarity_weight * polarity_loss
    )
    return TeacherForcingLoss(
        total=total,
        parameter_nll=parameter_nll,
        highres_mse=highres_mse,
        projected_mse=projected_mse,
        interface_jump_nll=jump_nll,
        interface_polarity_loss=polarity_loss,
        parameter_count=parameter_count,
        highres_count=highres_count,
        projected_count=projected_count,
        interface_jump_count=jump_count,
    )


__all__ = [
    "TeacherForcedParameterModel",
    "TeacherForcingLoss",
    "TeacherForcingLossConfig",
    "TeacherForcingModelConfig",
    "TeacherForcingOutput",
    "TorchTeacherForcingBatch",
    "batch_to_torch",
    "interface_polarity_classes",
    "project_highres_torch",
    "teacher_forcing_loss",
]
