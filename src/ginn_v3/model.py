"""The independent PIAI network used by :mod:`ginn_v3`.

The implementation deliberately keeps its own architecture.  In particular,
it does not import the v2 body network or its output construction.  The input
is a pair of one-dimensional traces (standardised seismic and standardised
log-LFM), and the two heads follow the original Mariana design: a three-layer
bidirectional GRU for the impedance output and another three-layer
bidirectional GRU followed by a whole-trace wavelet projection.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ginn_v3.config import NetworkConfig
from ginn_v3.types import Prediction


class _TemporalBlock(nn.Module):
    """A Mariana-style residual TCN block with Tanh activations."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.skip = nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
        self.activation = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Keep the small random convolution initialisation used by Mariana.
        # The correction head is zero-initialised separately below so that the
        # complete network starts at the supplied initial model.
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.skip.weight, mean=0.0, std=0.01)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.skip(value)
        value = self.activation(self.conv1(value))
        value = self.dropout1(value)
        value = self.activation(self.conv2(value))
        value = self.dropout2(value)
        return self.activation(value + residual)


class _BiGRUTraceHead(nn.Module):
    """Three-layer bidirectional GRU producing one value per sample."""

    def __init__(self, channels: int, *, layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(channels * 2, 1)

    def forward(self, features: Tensor) -> Tensor:
        # Conv1d features are (B, C, N); recurrent layers consume (B, N, C).
        sequence, _ = self.gru(features.transpose(1, 2))
        return self.output(sequence).squeeze(-1)


class _WaveletHead(nn.Module):
    """BiGRU followed by Mariana's whole-trace wavelet projection."""

    def __init__(
        self,
        channels: int,
        *,
        sample_count: int,
        wavelet_samples: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=channels,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.per_sample = nn.Linear(channels * 2, 1)
        self.wavelet_projection = nn.Linear(sample_count, wavelet_samples)

    def forward(self, features: Tensor) -> Tensor:
        sequence, _ = self.gru(features.transpose(1, 2))
        sequence = self.per_sample(sequence).squeeze(-1)
        return self.wavelet_projection(sequence)


class PIAINetwork(nn.Module):
    """Predict a raw log-AI correction and a wavelet per trace.

    Wavelet coefficients are expressed in standardized seismic units.  Their
    amplitudes are unconstrained, following the original PIAI head; the
    physical adapter owns any conversion to stored seismic units.

    ``log_ai`` is intentionally defined as ``initial_log_ai + raw_correction``
    with no Gaussian smoothing or low-frequency projection.  This is the v3
    model boundary: physical validity and support masking belong to the data
    and physics modules, while this network is only a predictor.
    """

    input_channels = 2

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        if not isinstance(config, NetworkConfig):
            raise TypeError("config must be a NetworkConfig instance.")
        self.config = config

        blocks: list[nn.Module] = []
        in_channels = self.input_channels
        for out_channels in config.tcn_channels:
            blocks.append(
                _TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size=config.kernel_size,
                    dilation=config.dilation,
                    dropout=config.dropout,
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.projection = nn.Conv1d(
            in_channels,
            config.hidden_channels,
            kernel_size=3,
            padding=1,
            dilation=1,
        )
        # The original PIAI model uses the two input channels as GN groups.
        self.group_norm = nn.GroupNorm(self.input_channels, config.hidden_channels)

        self.correction_head = _BiGRUTraceHead(
            config.hidden_channels,
            layers=config.gru_layers,
            dropout=config.dropout,
        )
        self.wavelet_head = _WaveletHead(
            config.hidden_channels,
            sample_count=config.sample_count,
            wavelet_samples=config.wavelet_samples,
            layers=config.gru_layers,
            dropout=config.dropout,
        )

        # Preserve the useful identity-at-start property without any v2
        # smoother or low-frequency anchor: raw correction is exactly zero.
        nn.init.zeros_(self.correction_head.output.weight)
        nn.init.zeros_(self.correction_head.output.bias)

    def forward(self, features: Tensor, initial_log_ai: Tensor) -> Prediction:
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError("features must have shape (batch, 2, samples).")
        if not torch.is_floating_point(features):
            raise ValueError("features must be a floating tensor.")
        if initial_log_ai.ndim != 2 or initial_log_ai.shape != features.shape[:1] + features.shape[2:]:
            raise ValueError("initial_log_ai must have shape (batch, samples).")
        if not torch.is_floating_point(initial_log_ai):
            raise ValueError("initial_log_ai must be a floating tensor.")
        if features.shape[2] != self.config.sample_count:
            raise ValueError(
                f"features has {features.shape[2]} samples; expected {self.config.sample_count}."
            )
        if initial_log_ai.device != features.device:
            raise ValueError("features and initial_log_ai must be on the same device.")

        encoded = self.encoder(features)
        encoded = self.projection(encoded)
        encoded = self.group_norm(encoded)
        raw_correction = self.correction_head(encoded)
        wavelets = self.wavelet_head(encoded)
        return Prediction(
            log_ai=initial_log_ai + raw_correction,
            raw_correction=raw_correction,
            wavelets=wavelets,
        )


__all__ = ["PIAINetwork"]
