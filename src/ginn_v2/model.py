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
    use_seismic: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.use_seismic, bool):
            raise TypeError("use_seismic must be boolean.")
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

    def __post_init__(self) -> None:
        values = (
            self.parameter_nll_weight,
            self.highres_mse_weight,
            self.projected_mse_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("teacher-forcing loss weights must be finite and non-negative.")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one teacher-forcing loss weight must be positive.")

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


@dataclass(frozen=True)
class TeacherForcingLoss:
    total: torch.Tensor
    parameter_nll: torch.Tensor
    highres_mse: torch.Tensor
    projected_mse: torch.Tensor
    parameter_count: int
    highres_count: int
    projected_count: int


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
        head_inputs = (
            config.feature_channels
            + config.state_embedding_channels
            + 3
        )
        self.parameter_head = nn.Sequential(
            nn.Linear(head_inputs, config.hidden_channels),
            nn.GELU(),
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.GELU(),
            nn.Linear(config.hidden_channels, 6),
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
        weights = batch.pooling_mask.to(dtype=feature_sequence.dtype)
        denominator = torch.clamp(weights.sum(dim=-1, keepdim=True), min=1.0)
        pooled = torch.einsum("bch,bsh->bsc", feature_sequence, weights)
        pooled = pooled / denominator
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

    def forward(self, batch: TorchTeacherForcingBatch) -> TeacherForcingOutput:
        features = self.encode_trace(batch)
        mean, std = self.parameterize_segments(features, batch)
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
    total = (
        config.parameter_nll_weight * parameter_nll
        + config.highres_mse_weight * highres_mse
        + config.projected_mse_weight * projected_mse
    )
    return TeacherForcingLoss(
        total=total,
        parameter_nll=parameter_nll,
        highres_mse=highres_mse,
        projected_mse=projected_mse,
        parameter_count=parameter_count,
        highres_count=highres_count,
        projected_count=projected_count,
    )


__all__ = [
    "TeacherForcedParameterModel",
    "TeacherForcingLoss",
    "TeacherForcingLossConfig",
    "TeacherForcingModelConfig",
    "TeacherForcingOutput",
    "TorchTeacherForcingBatch",
    "batch_to_torch",
    "project_highres_torch",
    "teacher_forcing_loss",
]
