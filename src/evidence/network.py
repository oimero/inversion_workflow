"""Band-limited evidence network and its public inference interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from evidence.contracts import (
    BandlimitedEvidence,
    EvidenceInput,
    EvidenceTargetContract,
    InputContractError,
)
from evidence.features import build_lfm_anchor, lfm_residual_from_anchor


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
    result = float(frequency[1 + int(np.argmax(spectrum[1:]))])
    if not np.isfinite(result) or result <= 0.0:
        raise InputContractError("dominant wavelet frequency is invalid.")
    return result


def tuning_scale_on_sample_axis(
    observation: EvidenceInput,
    *,
    dominant_frequency: float,
) -> np.ndarray:
    if not np.isfinite(dominant_frequency) or dominant_frequency <= 0.0:
        raise ValueError("dominant_frequency must be finite and positive.")
    if observation.sample_domain == "time":
        return np.full(
            observation.seismic.shape,
            1.0 / (2.0 * dominant_frequency),
            dtype=np.float64,
        )
    if observation.vp_model_mps is None:
        raise InputContractError("depth tuning scale requires vp_model_mps.")
    return observation.vp_model_mps / (4.0 * dominant_frequency)


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
        *,
        target_contract: EvidenceTargetContract | None = None,
    ) -> "EvidenceNetworkConfig":
        payload = dict(value)
        if target_contract is not None:
            scale_fields = {
                "seismic_scale",
                "lfm_residual_scale",
                "projected_log_ai_increment_scale",
                "signed_reflectivity_scale",
            }
            configured = sorted(scale_fields.intersection(payload))
            if configured:
                raise ValueError(
                    "network scales come from the target contract: "
                    f"{configured}"
                )
            scales = target_contract.global_scales
            payload.update(
                {
                    "seismic_scale": scales["seismic"],
                    "lfm_residual_scale": scales["lfm_residual"],
                    "projected_log_ai_increment_scale": scales[
                        "projected_log_ai_increment"
                    ],
                    "signed_reflectivity_scale": scales["signed_reflectivity"],
                }
            )
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
        return values + F.gelu(self.normalization(self.convolution(values)))


class BandlimitedEvidenceNetwork(nn.Module):
    """Vertical evidence encoder with an optional metric lateral mixer."""

    def __init__(self, config: EvidenceNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Conv1d(3, config.hidden_channels, 1)
        self.vertical = nn.Sequential(
            *(
                _ResidualBlock(config.hidden_channels, dilation=2 ** (index % 4))
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
        return (
            self.config.minimum_scale_fraction + F.softplus(raw)
        ) * float(global_scale)

    def forward(
        self,
        seismic: torch.Tensor,
        lfm_residual: torch.Tensor,
        observed_valid: torch.Tensor,
        lateral_m: torch.Tensor,
        lateral_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if seismic.ndim != 3 or lfm_residual.shape != seismic.shape or observed_valid.shape != seismic.shape:
            raise ValueError("observations must have shape [batch, lateral, sample].")
        batch, width, samples = seismic.shape
        if lateral_m.shape != (batch, width) or lateral_valid.shape != (batch, width):
            raise ValueError("lateral geometry must match [batch, lateral].")
        seismic_scaled = torch.where(
            observed_valid,
            seismic / self.config.seismic_scale,
            torch.zeros_like(seismic),
        )
        lfm_scaled = torch.where(
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
            (seismic_input, lfm_scaled, observed_valid.to(dtype=seismic.dtype)),
            dim=2,
        ).reshape(batch * width, 3, samples)
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
        hidden = self.fuse(
            torch.cat((feature, mixed), dim=2).reshape(
                batch * width,
                2 * self.config.hidden_channels,
                samples,
            )
        )
        support = observed_valid & lateral_valid[:, :, None]
        increment_mean = self.increment_mean(hidden).reshape(batch, width, samples)
        reflectivity_mean = self.reflectivity_mean(hidden).reshape(batch, width, samples)
        state_logits = self.state_logits(hidden).reshape(
            batch,
            width,
            3,
            samples,
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


class EvidenceModel:
    """Deep module for converting one observation tile into band-limited evidence."""

    PAYLOAD_SCHEMA = "bandlimited_evidence_model_v1"

    def __init__(
        self,
        network: BandlimitedEvidenceNetwork,
        *,
        target_contract: EvidenceTargetContract,
        dominant_frequency: float,
        device: str | torch.device = "cpu",
    ) -> None:
        if not np.isfinite(dominant_frequency) or dominant_frequency <= 0.0:
            raise ValueError("dominant_frequency must be finite and positive.")
        self.network = network
        self.target_contract = target_contract
        self.dominant_frequency = float(dominant_frequency)
        self.device = torch.device(device)
        self.network.to(self.device)

    @property
    def network_config(self) -> EvidenceNetworkConfig:
        return self.network.config

    @property
    def sample_domain(self) -> str:
        return self.target_contract.sample_domain

    def predict(self, observation: EvidenceInput) -> BandlimitedEvidence:
        contract = self.target_contract
        axis = observation.sample_axis
        if (
            observation.sample_domain != contract.sample_domain
            or axis.unit != contract.sample_unit
            or axis.depth_basis != contract.depth_basis
        ):
            raise InputContractError(
                "observation domain, unit, or depth basis differs from the model contract."
            )
        anchor = build_lfm_anchor(observation)
        lfm_residual = lfm_residual_from_anchor(observation, anchor)
        tuning = tuning_scale_on_sample_axis(
            observation,
            dominant_frequency=self.dominant_frequency,
        )
        self.network.eval()
        with torch.no_grad():
            output = self.network(
                torch.as_tensor(observation.seismic[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(lfm_residual[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(observation.observed_valid[None], dtype=torch.bool, device=self.device),
                torch.as_tensor(observation.lateral_m[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(observation.lateral_valid[None], dtype=torch.bool, device=self.device),
            )
        support = output["support"][0].cpu().numpy().astype(bool) & anchor.support
        increment_mean = output["projected_log_ai_increment_mean"][0].cpu().numpy().astype(np.float64)
        increment_scale = output["projected_log_ai_increment_scale"][0].cpu().numpy().astype(np.float64)
        reflectivity_mean = output["signed_reflectivity_mean"][0].cpu().numpy().astype(np.float64)
        reflectivity_scale = output["signed_reflectivity_scale"][0].cpu().numpy().astype(np.float64)
        state_log_potential = output["state_log_potential"][0].cpu().numpy().astype(np.float64)
        increment_mean[~support] = 0.0
        increment_scale[~support] = self.network_config.projected_log_ai_increment_scale
        reflectivity_mean[~support] = 0.0
        reflectivity_scale[~support] = self.network_config.signed_reflectivity_scale
        state_log_potential[~support] = -np.log(3.0)
        tuning[~support] = axis.sample_interval
        background = np.where(support, anchor.values, 0.0)
        return BandlimitedEvidence(
            sample_axis=axis,
            background_lfm_linear=background,
            projected_log_ai_increment_mean=increment_mean,
            projected_log_ai_increment_scale=increment_scale,
            signed_reflectivity_mean=reflectivity_mean,
            signed_reflectivity_scale=reflectivity_scale,
            state_log_potential=state_log_potential,
            local_tuning_scale=tuning,
            support=support,
            lateral_m=observation.lateral_m,
            x_m=observation.x_m,
            y_m=observation.y_m,
            identity=observation.identity,
        )

    def state_dict_payload(self) -> dict[str, Any]:
        return {
            "schema": self.PAYLOAD_SCHEMA,
            "network_config": asdict(self.network_config),
            "network_state": self.network.state_dict(),
            "target_contract": self.target_contract.to_mapping(),
            "dominant_frequency": self.dominant_frequency,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
    ) -> "EvidenceModel":
        if payload.get("schema") != cls.PAYLOAD_SCHEMA:
            raise ValueError("unsupported evidence model payload.")
        target_contract = EvidenceTargetContract.from_mapping(payload["target_contract"])
        config = EvidenceNetworkConfig.from_mapping(payload["network_config"])
        network = BandlimitedEvidenceNetwork(config)
        network.load_state_dict(payload["network_state"], strict=True)
        return cls(
            network,
            target_contract=target_contract,
            dominant_frequency=float(payload["dominant_frequency"]),
            device=device,
        )


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
    increment_scale_target = torch.abs(increment_residual.detach()) + config.minimum_scale_fraction
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
    reflectivity_scale_target = torch.abs(reflectivity_residual.detach()) + config.minimum_scale_fraction
    reflectivity_scale_ratio = (
        output["signed_reflectivity_scale"] / config.signed_reflectivity_scale
    )
    reflectivity_scale_huber = F.smooth_l1_loss(
        torch.log(reflectivity_scale_ratio[support]),
        torch.log(reflectivity_scale_target[support]),
    )
    represented = targets["state_emission"].long()[support]
    counts = torch.bincount(represented, minlength=3).to(dtype=torch.float32)
    present = counts > 0.0
    class_weight = torch.zeros_like(counts)
    class_weight[present] = counts[present].sum() / (present.sum() * counts[present])
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
    "BandlimitedEvidenceNetwork",
    "EvidenceModel",
    "EvidenceNetworkConfig",
    "dominant_frequency_hz",
    "evidence_loss",
    "tuning_scale_on_sample_axis",
]
