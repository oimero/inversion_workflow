"""Small reference models for the model-ablation gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelInfo:
    architecture_id: str
    parameter_count: int
    trainable_parameter_count: int
    lateral_receptive_field: int
    vertical_receptive_field: int
    input_channels: int
    output_channels: int

    @property
    def receptive_field_lateral(self) -> int:
        return self.lateral_receptive_field

    @property
    def receptive_field_twt(self) -> int:
        return self.vertical_receptive_field


class Patch2DNet(nn.Module):
    def __init__(self, *, in_channels: int = 3, hidden_channels: int = 32, depth: int = 5) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("Patch2DNet depth must be >= 2.")
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.depth = depth
        _zero_initialize(self.net[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Trace1DNet(nn.Module):
    def __init__(self, *, in_channels: int = 3, hidden_channels: int = 32, depth: int = 5) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("Trace1DNet depth must be >= 2.")
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv1d(hidden_channels, 1, kernel_size=5, padding=2))
        self.net = nn.Sequential(*layers)
        self.depth = depth
        _zero_initialize(self.net[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, lateral, twt = x.shape
        traces = x.permute(0, 2, 1, 3).reshape(b * lateral, c, twt)
        out = self.net(traces)
        return out.reshape(b, lateral, 1, twt).permute(0, 2, 1, 3)


class Trace1DDilatedTCN(nn.Module):
    def __init__(self, *, in_channels: int = 3, hidden_channels: int = 32, depth: int = 5) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("Trace1DDilatedTCN depth must be >= 2.")
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        ]
        for block in range(depth - 1):
            dilation = 2 ** block
            padding = 2 * dilation
            layers.extend(
                [
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=5,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv1d(hidden_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.depth = depth
        _zero_initialize(self.net[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, lateral, twt = x.shape
        traces = x.permute(0, 2, 1, 3).reshape(b * lateral, c, twt)
        out = self.net(traces)
        return out.reshape(b, lateral, 1, twt).permute(0, 2, 1, 3)


class Trace1DTCNShallowLateralMixer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_channels: int = 32,
        depth: int = 5,
        lateral_kernel: int = 3,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("Trace1DTCNShallowLateralMixer depth must be >= 2.")
        if lateral_kernel < 1 or lateral_kernel % 2 == 0:
            raise ValueError("lateral_kernel must be an odd positive integer.")
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        ]
        for block in range(depth - 1):
            dilation = 2 ** block
            padding = 2 * dilation
            layers.extend(
                [
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=5,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                ]
            )
        self.temporal_encoder = nn.Sequential(*layers)
        self.lateral_mixer = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(lateral_kernel, 1),
                padding=(lateral_kernel // 2, 0),
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.output = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.depth = depth
        self.lateral_kernel = lateral_kernel
        _zero_initialize(self.output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, lateral, twt = x.shape
        traces = x.permute(0, 2, 1, 3).reshape(b * lateral, c, twt)
        encoded = self.temporal_encoder(traces)
        encoded = encoded.reshape(b, lateral, encoded.shape[1], twt).permute(0, 2, 1, 3)
        mixed = encoded + self.lateral_mixer(encoded)
        return self.output(mixed)


def _zero_initialize(layer: nn.Module) -> None:
    if not isinstance(layer, (nn.Conv1d, nn.Conv2d)):
        raise TypeError("GINN-v2 output layer must be convolutional.")
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


ARCHITECTURE_IDS = (
    "trace_conv1d",
    "trace_dilated_tcn",
    "trace_lateral_mixer",
    "patch_conv2d",
)


def build_model(
    architecture_id: str,
    *,
    hidden_channels: int = 32,
    depth: int = 5,
    lateral_kernel: int | None = None,
) -> tuple[nn.Module, ModelInfo]:
    if architecture_id not in ARCHITECTURE_IDS:
        raise ValueError(
            f"Unsupported GINN-v2 architecture id {architecture_id!r}. "
            f"Use one of {list(ARCHITECTURE_IDS)}; legacy model IDs are not accepted."
        )
    if architecture_id == "patch_conv2d":
        if lateral_kernel is not None:
            raise ValueError("patch_conv2d does not accept lateral_kernel.")
        model = Patch2DNet(hidden_channels=hidden_channels, depth=depth)
        rf_lateral = 1 + 2 * depth
        rf_vertical = 1 + 2 * depth
    elif architecture_id == "trace_conv1d":
        if lateral_kernel is not None:
            raise ValueError("trace_conv1d does not accept lateral_kernel.")
        model = Trace1DNet(hidden_channels=hidden_channels, depth=depth)
        rf_lateral = 1
        rf_vertical = 1 + 4 * depth
    elif architecture_id == "trace_dilated_tcn":
        if lateral_kernel is not None:
            raise ValueError("trace_dilated_tcn does not accept lateral_kernel.")
        model = Trace1DDilatedTCN(hidden_channels=hidden_channels, depth=depth)
        rf_lateral = 1
        rf_vertical = 1 + 4 * (2 ** (depth - 1) - 1)
    else:
        lateral_kernel = 3 if lateral_kernel is None else int(lateral_kernel)
        model = Trace1DTCNShallowLateralMixer(
            hidden_channels=hidden_channels,
            depth=depth,
            lateral_kernel=lateral_kernel,
        )
        rf_lateral = lateral_kernel
        rf_vertical = 1 + 4 * (2 ** (depth - 1) - 1)
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return model, ModelInfo(
        architecture_id=architecture_id,
        parameter_count=int(params),
        trainable_parameter_count=int(trainable),
        lateral_receptive_field=int(rf_lateral),
        vertical_receptive_field=int(rf_vertical),
        input_channels=3,
        output_channels=1,
    )
