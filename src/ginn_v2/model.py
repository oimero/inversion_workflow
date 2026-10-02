"""Center-trace body network and final-curve physical smoothing."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor, nn

from cup.lfm.math import LowpassSpec
from cup.utils.masks import true_runs
from ginn_v2.loss import _gaussian_weights, masked_lfm_lowpass

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

        # A zero raw correction starts from the initial-model state after the
        # final-curve smoother is applied by the training/inference module.
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        features: Tensor,
        *,
        center_index: int,
    ) -> Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape (batch, channels, lateral, samples).")
        if features.shape[1] != self.config.input_channels:
            raise ValueError(
                f"features has {features.shape[1]} channels; expected {self.config.input_channels}."
            )
        if not torch.is_floating_point(features):
            raise ValueError("features must be floating tensors.")
        width = features.shape[2]
        if isinstance(center_index, bool) or not 0 <= int(center_index) < width:
            raise ValueError("center_index must address a feature lateral row.")
        value = self.output_layer(self.blocks(self.input_layer(features)))
        correction = value[:, 0, int(center_index), :]
        return correction


@dataclass(frozen=True)
class BodySmoother:
    """Apply one normalized physical-coordinate Gaussian to a complete curve.

    The smoother deliberately has no Step-7 low-pass dependency.  A network
    correction is first added to the initial model and this operation is then
    applied to the resulting curve::

        body = G_fwhm(initial_log_ai + raw_network_correction)

    ``support_mask`` identifies both source samples allowed into the Gaussian
    and output samples for which a smoothed value is requested.  Samples
    outside support are returned as zero by :meth:`smooth` and as NaN by
    :meth:`smooth_numpy`, matching the tensor/array conventions of the
    training and well-target paths.
    """

    smoothing_fwhm_m: float

    def __post_init__(self) -> None:
        width = float(self.smoothing_fwhm_m)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("smoothing_fwhm_m must be finite and positive.")

    def smooth(
        self,
        values: Tensor,
        coordinates_m: Tensor,
        support_mask: Tensor,
    ) -> Tensor:
        """Smooth a batch of curves and return zeros outside ``support_mask``."""

        if values.ndim != 2 or not torch.is_floating_point(values):
            raise ValueError("values must be a floating (batch, samples) tensor.")
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise ValueError("values must contain only finite values.")
        if coordinates_m.ndim not in {1, 2} or not torch.is_floating_point(coordinates_m):
            raise ValueError("coordinates_m must be a floating one- or two-dimensional tensor.")
        if support_mask.shape != values.shape or support_mask.dtype != torch.bool:
            raise ValueError("support_mask must be boolean and match values.")
        if coordinates_m.ndim == 1 and coordinates_m.shape[0] != values.shape[1]:
            raise ValueError("coordinates_m sample count differs from values.")
        if coordinates_m.ndim == 2 and coordinates_m.shape != values.shape:
            raise ValueError("coordinates_m batch shape differs from values.")
        coordinates = coordinates_m.to(device=values.device, dtype=values.dtype)
        if not bool(torch.all(torch.isfinite(coordinates)).item()):
            raise ValueError("coordinates_m must contain only finite values.")
        output = torch.zeros_like(values)
        support_cpu = support_mask.detach().cpu().numpy()
        runs: dict[tuple[int, int], list[int]] = {}
        for row, row_support in enumerate(support_cpu):
            for start, stop in true_runs(row_support):
                runs.setdefault((start, stop), []).append(row)
        # A 601-sample trace needs a 601x601 exact Gaussian matrix.  Reuse it
        # for identical 1-D axes, and keep the 2-D case in small batches so
        # variable per-row physical coordinates do not materialize a whole
        # volume-sized matrix at once.
        max_coordinate_rows = 16
        for (start, stop), row_values in runs.items():
            if coordinates.ndim == 1:
                weights = _gaussian_weights(
                    coordinates[start:stop],
                    fwhm_m=self.smoothing_fwhm_m,
                )[0].to(device=values.device, dtype=values.dtype)
                row_indices = torch.as_tensor(row_values, device=values.device, dtype=torch.long)
                output[row_indices, start:stop] = values[row_indices, start:stop] @ weights.T
                continue
            for batch_start in range(0, len(row_values), max_coordinate_rows):
                selected = row_values[batch_start : batch_start + max_coordinate_rows]
                row_indices = torch.as_tensor(selected, device=values.device, dtype=torch.long)
                segment_coordinates = coordinates[row_indices, start:stop]
                weights = _gaussian_weights(
                    segment_coordinates,
                    fwhm_m=self.smoothing_fwhm_m,
                ).to(device=values.device, dtype=values.dtype)
                output[row_indices, start:stop] = torch.bmm(
                    weights,
                    values[row_indices, start:stop].unsqueeze(-1),
                ).squeeze(-1)
        return output

    def smooth_numpy(
        self,
        values: np.ndarray,
        coordinates_m: np.ndarray,
        support_mask: np.ndarray,
    ) -> np.ndarray:
        """Smooth one NumPy curve and return NaN outside ``support_mask``."""

        array = np.asarray(values, dtype=np.float64)
        coordinates = np.asarray(coordinates_m, dtype=np.float64)
        support = np.asarray(support_mask, dtype=bool)
        if array.ndim != 1 or coordinates.ndim != 1 or support.shape != array.shape:
            raise ValueError("NumPy smoother inputs must be matching one-dimensional arrays.")
        if np.any(~np.isfinite(coordinates)) or np.any(support & ~np.isfinite(array)):
            raise ValueError("Supported NumPy smoother values and coordinates must be finite.")
        safe = np.where(support, array, 0.0)
        with torch.no_grad():
            result = self.smooth(
                torch.as_tensor(safe, dtype=torch.float64)[None, :],
                torch.as_tensor(coordinates, dtype=torch.float64),
                torch.as_tensor(support, dtype=torch.bool)[None, :],
            )[0].cpu().numpy()
        result = np.asarray(result, dtype=np.float64)
        result[~support] = np.nan
        return result

    def construct(
        self,
        initial_log_ai: Tensor,
        raw_correction: Tensor,
        coordinates_m: Tensor,
        support_mask: Tensor,
        *,
        sample_step: float,
        lfm_lowpass_spec: LowpassSpec,
    ) -> "BodyConstruction":
        """Construct the shared body output from one raw network correction.

        The model always has a smoothed initial baseline ``b0``.  The raw
        correction is smoothed once, differenced against that baseline, and
        only its low-frequency part is removed with the exact upstream LFM
        operator::

            b0 = G(initial)
            d = G(initial + raw) - b0
            body = b0 + (I - L)d

        Keeping this operation here makes training, direct trace inference,
        and volume fusion use the same output semantics.  ``support_mask``
        describes the complete finite LFM run used by both physical filters.
        """

        if initial_log_ai.ndim != 2 or not torch.is_floating_point(initial_log_ai):
            raise ValueError("initial_log_ai must be a floating (batch, samples) tensor.")
        if raw_correction.shape != initial_log_ai.shape or not torch.is_floating_point(raw_correction):
            raise ValueError("raw_correction must be a floating tensor matching initial_log_ai.")
        if not bool(torch.all(torch.isfinite(initial_log_ai)).item()) or not bool(
            torch.all(torch.isfinite(raw_correction)).item()
        ):
            raise ValueError("initial_log_ai and raw_correction must contain only finite values.")
        if support_mask.shape != initial_log_ai.shape or support_mask.dtype != torch.bool:
            raise ValueError("support_mask must be boolean and match initial_log_ai.")
        step = float(sample_step)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("sample_step must be finite and positive.")
        if not isinstance(lfm_lowpass_spec, LowpassSpec):
            raise TypeError("lfm_lowpass_spec must be a LowpassSpec.")
        baseline = self.smooth(initial_log_ai, coordinates_m, support_mask)
        smoothed_raw = self.smooth(initial_log_ai + raw_correction, coordinates_m, support_mask)
        correction = smoothed_raw - baseline
        if lfm_lowpass_spec.enabled:
            low_frequency, low_support = masked_lfm_lowpass(
                correction,
                support_mask,
                sample_step=step,
                spec=lfm_lowpass_spec,
            )
        else:
            low_frequency = torch.zeros_like(correction)
            low_support = support_mask
        body = baseline + correction - low_frequency
        valid = support_mask & low_support
        return BodyConstruction(
            baseline_log_ai=baseline,
            smoothed_raw_log_ai=smoothed_raw,
            correction_log_ai=correction,
            low_frequency_increment=low_frequency,
            body_log_ai=body,
            valid_mask=valid,
        )


@dataclass(frozen=True)
class BodyConstruction:
    """Shared baseline/correction decomposition for a body prediction."""

    baseline_log_ai: Tensor
    smoothed_raw_log_ai: Tensor
    correction_log_ai: Tensor
    low_frequency_increment: Tensor
    body_log_ai: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        shape = self.body_log_ai.shape
        for name in (
            "baseline_log_ai",
            "smoothed_raw_log_ai",
            "correction_log_ai",
            "low_frequency_increment",
        ):
            value = getattr(self, name)
            if value.shape != shape or not torch.is_floating_point(value):
                raise ValueError(f"{name} must be a floating tensor matching body_log_ai.")
        if self.valid_mask.dtype != torch.bool or self.valid_mask.shape != shape:
            raise ValueError("valid_mask must be boolean and match body_log_ai.")


__all__ = [
    "BodyConstruction",
    "BodyNetworkConfig",
    "BodySmoother",
    "CenterTraceBodyNet",
]
