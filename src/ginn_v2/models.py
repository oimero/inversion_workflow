"""Small reference models for the model-ablation gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    parameter_count: int
    trainable_parameter_count: int
    receptive_field_lateral: int
    receptive_field_twt: int
    input_channels: int
    output_channels: int


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, lateral, twt = x.shape
        traces = x.permute(0, 2, 1, 3).reshape(b * lateral, c, twt)
        out = self.net(traces)
        return out.reshape(b, lateral, 1, twt).permute(0, 2, 1, 3)


def build_model(
    model_id: str,
    *,
    hidden_channels: int = 32,
    depth: int = 5,
) -> tuple[nn.Module, ModelInfo]:
    if model_id in {
        "patch_2d_supervised",
        "patch_2d_mismatch_training",
        "patch_2d_with_physics_loss",
    }:
        model = Patch2DNet(hidden_channels=hidden_channels, depth=depth)
        rf_lateral = 1 + 2 * depth
        rf_twt = 1 + 2 * depth
    elif model_id == "trace_1d":
        model = Trace1DNet(hidden_channels=hidden_channels, depth=depth)
        rf_lateral = 1
        rf_twt = 1 + 4 * depth
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return model, ModelInfo(
        model_id=model_id,
        parameter_count=int(params),
        trainable_parameter_count=int(trainable),
        receptive_field_lateral=int(rf_lateral),
        receptive_field_twt=int(rf_twt),
        input_channels=3,
        output_channels=1,
    )
