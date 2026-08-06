"""Observable evidence construction and the reusable vertical encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cup.physics.numpy_backend import reflectivity_from_log_ai
from ginn_v2.contracts import (
    InputContractError,
    ObservationTile,
)


@dataclass(frozen=True)
class ObservableTargets:
    """Physically defined model-grid targets used by the audit seam."""

    projected_log_ai_increment: np.ndarray
    signed_reflectivity: np.ndarray
    state_id: np.ndarray
    support: np.ndarray


def build_observable_targets(
    tile: ObservationTile,
    *,
    model_log_ai: np.ndarray,
    state_highres: np.ndarray,
    background_lfm_linear: np.ndarray,
    anchor_support: np.ndarray,
) -> ObservableTargets:
    """Build targets without per-trace normalization or invented activity labels.

    Reflectivity follows the same lower-interface convention as ``cup.physics``:
    target sample ``j`` represents the interface between samples ``j-1`` and
    ``j``; sample zero has no lower-interface predecessor and is unsupported.
    """

    log_ai = np.asarray(model_log_ai, dtype=np.float64)
    state = np.asarray(state_highres)
    background = np.asarray(background_lfm_linear, dtype=np.float64)
    support = np.asarray(anchor_support, dtype=bool)
    if log_ai.shape != tile.seismic.shape:
        raise InputContractError("model_log_ai must match the observation tile.")
    if background.shape != log_ai.shape or support.shape != log_ai.shape:
        raise InputContractError("LFM anchor arrays must match model_log_ai.")
    expected_highres = (tile.width, tile.highres_axis.coordinates.size)
    if state.shape != expected_highres:
        raise InputContractError("state_highres must match the high-resolution axis.")

    ratio = tile.model_axis.sample_interval / tile.highres_axis.sample_interval
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1.0e-10):
        raise InputContractError("observable targets require integer-nested axes.")
    nested_coordinates = tile.highres_axis.coordinates[::factor]
    if nested_coordinates.shape != tile.model_axis.coordinates.shape or not np.allclose(
        nested_coordinates,
        tile.model_axis.coordinates,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise InputContractError(
            "observable targets require coincident model/high-resolution axes."
        )
    state_model = state[:, ::factor]
    if state_model.shape != log_ai.shape:
        raise InputContractError("model-grid state sampling changed the target shape.")

    base_support = (
        support
        & tile.observed_valid
        & tile.lateral_valid[:, None]
        & np.isfinite(log_ai)
        & np.isfinite(background)
        & (state_model >= 0)
        & (state_model <= 2)
    )
    increment = np.where(base_support, log_ai - background, 0.0)
    reflectivity = np.zeros_like(log_ai)
    reflectivity[:, 1:] = reflectivity_from_log_ai(log_ai)
    reflectivity_support = base_support.copy()
    reflectivity_support[:, 0] = False
    reflectivity_support[:, 1:] &= base_support[:, :-1]
    common_support = base_support & reflectivity_support
    state_model = np.where(common_support, state_model, -1).astype(np.int64)
    return ObservableTargets(
        projected_log_ai_increment=np.where(common_support, increment, 0.0),
        signed_reflectivity=np.where(common_support, reflectivity, 0.0),
        state_id=state_model,
        support=common_support,
    )


def dominant_frequency_hz(
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
) -> float:
    time = np.asarray(wavelet_time_s, dtype=np.float64).reshape(-1)
    amplitude = np.asarray(wavelet_amplitude, dtype=np.float64).reshape(-1)
    if (
        time.size < 3
        or time.size != amplitude.size
        or np.any(~np.isfinite(time))
        or np.any(~np.isfinite(amplitude))
    ):
        raise InputContractError("wavelet arrays must be finite and share a length >= 3.")
    intervals = np.diff(time)
    if np.any(intervals <= 0.0) or not np.allclose(
        intervals,
        intervals[0],
        rtol=1.0e-6,
        atol=1.0e-12,
    ):
        raise InputContractError("wavelet_time_s must be regularly increasing.")
    spectrum = np.abs(np.fft.rfft(amplitude - np.mean(amplitude)))
    frequency = np.fft.rfftfreq(amplitude.size, d=float(intervals[0]))
    if spectrum.size <= 1 or not np.any(spectrum[1:] > 0.0):
        raise InputContractError("wavelet has no non-zero frequency content.")
    index = 1 + int(np.argmax(spectrum[1:]))
    result = float(frequency[index])
    if not np.isfinite(result) or result <= 0.0:
        raise InputContractError("dominant wavelet frequency is invalid.")
    return result


def tuning_scale_on_model_axis(
    tile: ObservationTile,
    *,
    dominant_frequency: float,
    vp_model_mps: np.ndarray | None = None,
) -> np.ndarray:
    if not np.isfinite(dominant_frequency) or dominant_frequency <= 0.0:
        raise ValueError("dominant_frequency must be finite and positive.")
    shape = tile.seismic.shape
    if tile.sample_domain == "time":
        return np.full(shape, 1.0 / (2.0 * dominant_frequency), dtype=np.float64)
    if vp_model_mps is None:
        raise InputContractError("depth tuning scale requires vp_model_mps.")
    velocity = np.asarray(vp_model_mps, dtype=np.float64)
    if velocity.shape != shape or np.any(tile.observed_valid & (~np.isfinite(velocity) | (velocity <= 0.0))):
        raise InputContractError("vp_model_mps must be finite, positive, and match observations.")
    return velocity / (4.0 * dominant_frequency)


@dataclass(frozen=True)
class EvidenceNetworkConfig:
    hidden_channels: int = 48
    vertical_layers: int = 3
    lateral_distance_scale_m: float = 250.0
    use_lateral_context: bool = False
    input_mode: str = "full"
    minimum_scale_fraction: float = 0.05
    seismic_scale: float = 1.0
    lfm_residual_scale: float = 1.0
    projected_log_ai_increment_scale: float = 1.0
    signed_reflectivity_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_channels <= 0 or self.vertical_layers <= 0:
            raise ValueError("network dimensions must be positive.")
        if self.lateral_distance_scale_m <= 0.0:
            raise ValueError("lateral_distance_scale_m must be positive.")
        if not isinstance(self.use_lateral_context, bool):
            raise TypeError("use_lateral_context must be boolean.")
        if self.input_mode not in {"full", "no_seismic"}:
            raise ValueError("input_mode must be full or no_seismic.")
        for name in (
            "minimum_scale_fraction",
            "seismic_scale",
            "lfm_residual_scale",
            "projected_log_ai_increment_scale",
            "signed_reflectivity_scale",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "EvidenceNetworkConfig":
        payload = dict(value)
        return cls(**payload)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
        )
        self.normalization = nn.GroupNorm(1, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.normalization(self.convolution(values))
        return values + F.gelu(update)


class ObservableEvidenceNetwork(nn.Module):
    """Single-trace audited evidence model with an optional later lateral mixer."""

    def __init__(self, config: EvidenceNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(3, config.hidden_channels, 1)
        self.vertical = nn.Sequential(
            *(
                _ResidualBlock(
                    config.hidden_channels,
                    dilation=2 ** (index % 4),
                )
                for index in range(config.vertical_layers)
            )
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(2 * config.hidden_channels, config.hidden_channels, 1),
            nn.GroupNorm(1, config.hidden_channels),
            nn.GELU(),
        )
        self.increment_mean = nn.Conv1d(config.hidden_channels, 1, 1)
        self.increment_raw_scale = nn.Conv1d(config.hidden_channels, 1, 1)
        self.reflectivity_mean = nn.Conv1d(config.hidden_channels, 1, 1)
        self.reflectivity_raw_scale = nn.Conv1d(config.hidden_channels, 1, 1)
        self.state_logits = nn.Conv1d(config.hidden_channels, 3, 1)
        for head in (self.increment_mean, self.reflectivity_mean):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        initial_ratio = max(0.5 - config.minimum_scale_fraction, 1.0e-4)
        raw_bias = float(np.log(np.expm1(initial_ratio)))
        for head in (self.increment_raw_scale, self.reflectivity_raw_scale):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, raw_bias)

    def _positive_scale(self, raw: torch.Tensor, global_scale: float) -> torch.Tensor:
        ratio = self.config.minimum_scale_fraction + F.softplus(raw)
        return ratio * float(global_scale)

    def forward(
        self,
        seismic: torch.Tensor,
        lfm_residual: torch.Tensor,
        observed_valid: torch.Tensor,
        lateral_m: torch.Tensor,
        lateral_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            seismic.ndim != 3
            or lfm_residual.shape != seismic.shape
            or observed_valid.shape != seismic.shape
        ):
            raise ValueError("observations must have shape [batch, lateral, sample].")
        batch, width, samples = seismic.shape
        if lateral_m.shape != (batch, width) or lateral_valid.shape != (batch, width):
            raise ValueError("lateral geometry must match [batch, lateral].")
        seismic_scaled = torch.where(
            observed_valid,
            seismic / self.config.seismic_scale,
            torch.zeros_like(seismic),
        )
        lfm_residual_scaled = torch.where(
            observed_valid,
            lfm_residual / self.config.lfm_residual_scale,
            torch.zeros_like(lfm_residual),
        )
        seismic_input = (
            torch.zeros_like(seismic_scaled)
            if self.config.input_mode == "no_seismic"
            else seismic_scaled
        )
        values = torch.stack(
            (
                seismic_input,
                lfm_residual_scaled,
                observed_valid.to(dtype=seismic.dtype),
            ),
            dim=2,
        ).reshape(
            batch * width,
            3,
            samples,
        )
        feature = self.vertical(self.input_projection(values)).reshape(
            batch,
            width,
            self.config.hidden_channels,
            samples,
        )
        distance = torch.abs(lateral_m[:, :, None] - lateral_m[:, None, :])
        weight = torch.exp(-distance / self.config.lateral_distance_scale_m)
        pair_valid = lateral_valid[:, :, None] & lateral_valid[:, None, :]
        weight = weight * pair_valid.to(weight.dtype)
        if self.config.use_lateral_context:
            weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            mixed = torch.einsum("bij,bjcn->bicn", weight, feature)
        else:
            mixed = feature
        combined = torch.cat((feature, mixed), dim=2).reshape(
            batch * width,
            2 * self.config.hidden_channels,
            samples,
        )
        hidden = self.fuse(combined)
        support = observed_valid & lateral_valid[:, :, None]
        increment_mean = self.increment_mean(hidden).reshape(batch, width, samples)
        reflectivity_mean = self.reflectivity_mean(hidden).reshape(
            batch, width, samples
        )
        state_logits = self.state_logits(hidden).reshape(
            batch, width, 3, samples
        ).permute(0, 1, 3, 2)
        return {
            "projected_log_ai_increment_mean": increment_mean,
            "projected_log_ai_increment_scale": self._positive_scale(
                self.increment_raw_scale(hidden).reshape(batch, width, samples),
                self.config.projected_log_ai_increment_scale,
            ),
            "signed_reflectivity_mean": reflectivity_mean,
            "signed_reflectivity_scale": self._positive_scale(
                self.reflectivity_raw_scale(hidden).reshape(batch, width, samples),
                self.config.signed_reflectivity_scale,
            ),
            "state_logits": state_logits,
            "state_log_potential": F.log_softmax(state_logits, dim=-1),
            "support": support,
        }


def evidence_loss(
    output: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    config: EvidenceNetworkConfig,
    increment_weight: float,
    reflectivity_weight: float,
    state_weight: float,
    scale_weight: float,
) -> dict[str, torch.Tensor]:
    support = targets["support"].bool() & output["support"].bool()
    if not torch.any(support):
        raise ValueError("evidence loss has no supported samples.")
    increment_residual = (
        output["projected_log_ai_increment_mean"]
        - targets["projected_log_ai_increment"]
    ) / config.projected_log_ai_increment_scale
    increment_mean_huber = F.smooth_l1_loss(
        increment_residual[support],
        torch.zeros_like(increment_residual[support]),
        beta=1.0,
    )
    increment_scale_target = (
        torch.abs(increment_residual.detach()) + config.minimum_scale_fraction
    )
    increment_scale_ratio = (
        output["projected_log_ai_increment_scale"]
        / config.projected_log_ai_increment_scale
    )
    increment_scale_huber = F.smooth_l1_loss(
        torch.log(increment_scale_ratio[support]),
        torch.log(increment_scale_target[support]),
    )

    reflectivity_residual = (
        output["signed_reflectivity_mean"] - targets["signed_reflectivity"]
    ) / config.signed_reflectivity_scale
    reflectivity_mean_huber = F.smooth_l1_loss(
        reflectivity_residual[support],
        torch.zeros_like(reflectivity_residual[support]),
        beta=1.0,
    )
    reflectivity_scale_target = (
        torch.abs(reflectivity_residual.detach()) + config.minimum_scale_fraction
    )
    reflectivity_scale_ratio = (
        output["signed_reflectivity_scale"] / config.signed_reflectivity_scale
    )
    reflectivity_scale_huber = F.smooth_l1_loss(
        torch.log(reflectivity_scale_ratio[support]),
        torch.log(reflectivity_scale_target[support]),
    )

    state_target = targets["state_emission"].long()
    represented = state_target[support]
    counts = torch.bincount(represented, minlength=3).to(dtype=torch.float32)
    present = counts > 0.0
    class_weight = torch.zeros_like(counts)
    class_weight[present] = counts[present].sum() / (
        present.sum() * counts[present]
    )
    state_cross_entropy = F.cross_entropy(
        output["state_logits"][support],
        represented,
        weight=class_weight,
    )
    total = (
        increment_weight * increment_mean_huber
        + reflectivity_weight * reflectivity_mean_huber
        + state_weight * state_cross_entropy
        + scale_weight * (increment_scale_huber + reflectivity_scale_huber)
    )
    return {
        "loss": total,
        "increment_mean_huber": increment_mean_huber,
        "increment_scale_huber": increment_scale_huber,
        "reflectivity_mean_huber": reflectivity_mean_huber,
        "reflectivity_scale_huber": reflectivity_scale_huber,
        "state_cross_entropy": state_cross_entropy,
    }


__all__ = [
    "ObservableEvidenceNetwork",
    "EvidenceNetworkConfig",
    "ObservableTargets",
    "build_observable_targets",
    "dominant_frequency_hz",
    "evidence_loss",
    "tuning_scale_on_model_axis",
]
