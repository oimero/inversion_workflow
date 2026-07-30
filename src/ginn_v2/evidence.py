"""Tuning-window supervision and the band-limited evidence network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ginn_v2.contracts import InputContractError, ObservationTile


@dataclass(frozen=True)
class TuningTargets:
    increment: np.ndarray
    state_occupancy: np.ndarray
    interface_activity: np.ndarray
    local_tuning_scale: np.ndarray
    support: np.ndarray


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


def _gaussian_average(
    source_axis: np.ndarray,
    values: np.ndarray,
    target_axis: np.ndarray,
    fwhm: np.ndarray,
    source_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_axis, dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    result = np.zeros(target.size, dtype=np.float64)
    support = np.zeros(target.size, dtype=bool)
    for index, center in enumerate(target):
        sigma = float(fwhm[index]) / 2.354820045
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise InputContractError("tuning-window FWHM must be finite and positive.")
        within = source_valid & (np.abs(source - center) <= 3.0 * sigma)
        if not np.any(within):
            continue
        weight = np.exp(-0.5 * ((source[within] - center) / sigma) ** 2)
        total = float(np.sum(weight))
        if total <= 0.0:
            continue
        result[index] = float(np.sum(weight * values[within]) / total)
        support[index] = True
    return result, support


def build_tuning_targets(
    tile: ObservationTile,
    *,
    log_ai_highres: np.ndarray,
    state_highres: np.ndarray,
    background_lfm_highres: np.ndarray,
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    vp_model_mps: np.ndarray | None = None,
) -> TuningTargets:
    """Build domain-neutral tuning-window targets on the model grid."""
    log_ai = np.asarray(log_ai_highres, dtype=np.float64)
    state = np.asarray(state_highres)
    background = np.asarray(background_lfm_highres, dtype=np.float64)
    expected = (tile.width, tile.highres_axis.coordinates.size)
    if log_ai.shape != expected or state.shape != expected or background.shape != expected:
        raise InputContractError("high-resolution truth arrays must match the tile axes.")
    frequency = dominant_frequency_hz(wavelet_time_s, wavelet_amplitude)
    tuning = tuning_scale_on_model_axis(
        tile,
        dominant_frequency=frequency,
        vp_model_mps=vp_model_mps,
    )
    increment = np.zeros(tile.seismic.shape, dtype=np.float64)
    occupancy = np.full(tile.seismic.shape + (3,), 1.0 / 3.0, dtype=np.float64)
    activity = np.zeros(tile.seismic.shape, dtype=np.float64)
    support = np.zeros(tile.seismic.shape, dtype=bool)
    high_axis = tile.highres_axis.coordinates
    model_axis = tile.model_axis.coordinates
    for trace in range(tile.width):
        valid = np.isfinite(log_ai[trace]) & np.isfinite(background[trace])
        if not np.any(valid):
            continue
        residual = log_ai[trace] - background[trace]
        target, target_support = _gaussian_average(
            high_axis,
            residual,
            model_axis,
            tuning[trace],
            valid,
        )
        increment[trace] = target
        support[trace] = target_support & tile.observed_valid[trace]
        state_columns: list[np.ndarray] = []
        for state_id in range(3):
            value, _ = _gaussian_average(
                high_axis,
                (state[trace] == state_id).astype(np.float64),
                model_axis,
                tuning[trace],
                valid & (state[trace] >= 0),
            )
            state_columns.append(value)
        stacked = np.stack(state_columns, axis=-1)
        total = np.sum(stacked, axis=-1, keepdims=True)
        good = total[..., 0] > 0.0
        stacked[good] /= total[good]
        stacked[~good] = 1.0 / 3.0
        occupancy[trace] = stacked

        jump = np.zeros(high_axis.size, dtype=np.float64)
        pair_valid = valid[1:] & valid[:-1]
        jump_indices = np.flatnonzero(pair_valid) + 1
        jump[jump_indices] = np.abs(np.diff(log_ai[trace])[pair_valid])
        smoothed_jump, _ = _gaussian_average(
            high_axis,
            jump,
            model_axis,
            tuning[trace],
            valid,
        )
        robust = float(np.quantile(smoothed_jump[support[trace]], 0.95)) if np.any(
            support[trace]
        ) else 0.0
        if robust > 0.0:
            activity[trace] = np.clip(smoothed_jump / robust, 0.0, 1.0)
    return TuningTargets(
        increment=increment,
        state_occupancy=occupancy,
        interface_activity=activity,
        local_tuning_scale=tuning,
        support=support,
    )


@dataclass(frozen=True)
class EvidenceNetworkConfig:
    hidden_channels: int = 48
    vertical_layers: int = 3
    lateral_distance_scale_m: float = 250.0
    input_mode: str = "full"

    def __post_init__(self) -> None:
        if self.hidden_channels <= 0 or self.vertical_layers <= 0:
            raise ValueError("network dimensions must be positive.")
        if self.lateral_distance_scale_m <= 0.0:
            raise ValueError("lateral_distance_scale_m must be positive.")
        if self.input_mode not in {"full", "no_seismic", "single_trace"}:
            raise ValueError("input_mode must be full, no_seismic, or single_trace.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EvidenceNetworkConfig":
        return cls(**dict(value))


class BandlimitedEvidenceNetwork(nn.Module):
    """Variable-width 1D lateral model with an explicit evidence bottleneck."""

    def __init__(self, config: EvidenceNetworkConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        channels = 2
        for _ in range(config.vertical_layers):
            layers.extend(
                (
                    nn.Conv1d(channels, config.hidden_channels, kernel_size=5, padding=2),
                    nn.GELU(),
                )
            )
            channels = config.hidden_channels
        self.vertical = nn.Sequential(*layers)
        self.fuse = nn.Sequential(
            nn.Conv1d(2 * config.hidden_channels, config.hidden_channels, 1),
            nn.GELU(),
        )
        self.increment_mean = nn.Conv1d(config.hidden_channels, 1, 1)
        self.increment_scale = nn.Conv1d(config.hidden_channels, 1, 1)
        self.state_logits = nn.Conv1d(config.hidden_channels, 3, 1)
        self.interface_logit = nn.Conv1d(config.hidden_channels, 1, 1)

    def forward(
        self,
        seismic: torch.Tensor,
        lfm: torch.Tensor,
        observed_valid: torch.Tensor,
        lateral_m: torch.Tensor,
        lateral_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if seismic.ndim != 3 or lfm.shape != seismic.shape or observed_valid.shape != seismic.shape:
            raise ValueError("observations must have shape [batch, lateral, sample].")
        batch, width, samples = seismic.shape
        if lateral_m.shape != (batch, width) or lateral_valid.shape != (batch, width):
            raise ValueError("lateral geometry must match [batch, lateral].")
        seismic_input = (
            torch.zeros_like(seismic)
            if self.config.input_mode == "no_seismic"
            else seismic
        )
        values = torch.stack((seismic_input, lfm), dim=2).reshape(
            batch * width,
            2,
            samples,
        )
        feature = self.vertical(values).reshape(
            batch,
            width,
            self.config.hidden_channels,
            samples,
        )
        distance = torch.abs(lateral_m[:, :, None] - lateral_m[:, None, :])
        weight = torch.exp(-distance / self.config.lateral_distance_scale_m)
        pair_valid = lateral_valid[:, :, None] & lateral_valid[:, None, :]
        weight = weight * pair_valid.to(weight.dtype)
        if self.config.input_mode == "single_trace":
            weight = torch.eye(width, device=weight.device, dtype=weight.dtype)[None].expand(
                batch,
                -1,
                -1,
            )
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        mixed = torch.einsum("bij,bjcn->bicn", weight, feature)
        combined = torch.cat((feature, mixed), dim=2).reshape(
            batch * width,
            2 * self.config.hidden_channels,
            samples,
        )
        hidden = self.fuse(combined)
        support = observed_valid & lateral_valid[:, :, None]
        return {
            "increment_mean": self.increment_mean(hidden).reshape(batch, width, samples),
            "increment_scale": (
                F.softplus(self.increment_scale(hidden)) + 1.0e-4
            ).reshape(batch, width, samples),
            "state_occupancy": torch.softmax(
                self.state_logits(hidden).reshape(batch, width, 3, samples).permute(
                    0,
                    1,
                    3,
                    2,
                ),
                dim=-1,
            ),
            "interface_activity": torch.sigmoid(
                self.interface_logit(hidden).reshape(batch, width, samples)
            ),
            "support": support,
        }


def evidence_loss(
    output: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    support = targets["support"].bool() & output["support"].bool()
    if not torch.any(support):
        raise ValueError("evidence loss has no supported samples.")
    residual = output["increment_mean"] - targets["increment"]
    scale = output["increment_scale"]
    increment_nll = (
        0.5 * (residual / scale).square() + torch.log(scale)
    )[support].mean()
    state_target = targets["state_occupancy"]
    state_ce = -(
        state_target
        * torch.log(output["state_occupancy"].clamp_min(1.0e-8))
    ).sum(dim=-1)[support].mean()
    interface = F.binary_cross_entropy(
        output["interface_activity"][support],
        targets["interface_activity"][support],
    )
    total = increment_nll + state_ce + interface
    return {
        "loss": total,
        "increment_nll": increment_nll,
        "state_cross_entropy": state_ce,
        "interface_bce": interface,
    }


__all__ = [
    "BandlimitedEvidenceNetwork",
    "EvidenceNetworkConfig",
    "TuningTargets",
    "build_tuning_targets",
    "dominant_frequency_hz",
    "evidence_loss",
    "tuning_scale_on_model_axis",
]
