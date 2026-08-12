"""Small shared neural module for center-trace body prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BodyNetworkConfig:
    """Architecture contract persisted in every body-inversion checkpoint."""

    input_channels: int = 6
    hidden_channels: int = 32
    residual_blocks: int = 4
    lateral_kernel: int = 3
    sample_kernel: int = 7

    def __post_init__(self) -> None:
        for name in ("input_channels", "hidden_channels", "residual_blocks", "lateral_kernel", "sample_kernel"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.lateral_kernel % 2 == 0 or self.sample_kernel % 2 == 0:
            raise ValueError("lateral_kernel and sample_kernel must be odd.")


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, lateral_kernel: int, sample_kernel: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(lateral_kernel, sample_kernel),
            padding=(lateral_kernel // 2, sample_kernel // 2),
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(lateral_kernel, sample_kernel),
            padding=(lateral_kernel // 2, sample_kernel // 2),
        )
        self.activation = nn.GELU()

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.activation(self.conv1(value))
        value = self.conv2(value)
        return self.activation(value + residual)


class CenterTraceBodyNet(nn.Module):
    """Predict one body-scale center trace from an oriented 2-D patch.

    The network is orientation agnostic: inline and xline profiles use the
    same weights.  The center index is supplied at inference time so the
    network never infers a missing center from a fixed channel value.
    """

    def __init__(self, config: BodyNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or BodyNetworkConfig()
        cfg = self.config
        self.input_layer = nn.Conv2d(
            cfg.input_channels,
            cfg.hidden_channels,
            kernel_size=(cfg.lateral_kernel, cfg.sample_kernel),
            padding=(cfg.lateral_kernel // 2, cfg.sample_kernel // 2),
        )
        self.blocks = nn.Sequential(
            *(
                _ResidualBlock(
                    cfg.hidden_channels,
                    lateral_kernel=cfg.lateral_kernel,
                    sample_kernel=cfg.sample_kernel,
                )
                for _ in range(cfg.residual_blocks)
            )
        )
        self.output_layer = nn.Conv2d(
            cfg.hidden_channels,
            1,
            kernel_size=(1, cfg.sample_kernel),
            padding=(0, cfg.sample_kernel // 2),
        )

        # A zero correction starts from the LFM skip connection.  This makes
        # the LFM-only baseline an explicit model state in the same network.
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        features: Tensor,
        *,
        center_index: int,
        center_lfm_log_ai: Tensor | None = None,
    ) -> Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (batch, channels, lateral, samples).")
        if features.shape[1] != self.config.input_channels:
            raise ValueError(
                f"features has {features.shape[1]} channels; expected {self.config.input_channels}."
            )
        if not torch.is_floating_point(features) or not bool(torch.all(torch.isfinite(features)).item()):
            raise ValueError("features must be finite floating tensors.")
        width = features.shape[2]
        if isinstance(center_index, bool) or not 0 <= int(center_index) < width:
            raise ValueError("center_index must address a feature lateral row.")
        value = self.output_layer(self.blocks(self.input_layer(features)))
        correction = value[:, 0, int(center_index), :]
        if center_lfm_log_ai is None:
            return correction
        if center_lfm_log_ai.shape != correction.shape:
            raise ValueError("center_lfm_log_ai must match the predicted center trace shape.")
        if not torch.is_floating_point(center_lfm_log_ai) or not bool(torch.all(torch.isfinite(center_lfm_log_ai)).item()):
            raise ValueError("center_lfm_log_ai must be finite and floating.")
        return center_lfm_log_ai + correction


__all__ = ["BodyNetworkConfig", "CenterTraceBodyNet"]
