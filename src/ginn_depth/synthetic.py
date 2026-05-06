"""Synthetic depth-domain samples for GINN residual pretraining."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

SyntheticVelocityMode = Literal["lfm_vp", "from_ai_linear", "blend"]


class _BaseDepthDataset(Protocol):
    seis_rms: float
    lfm_scale: float
    dynamic_gain_median: float | None

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...


class SyntheticDepthTraceDataset(Dataset):
    """Generate model-ready synthetic depth traces from real LFM/Vp backgrounds."""

    def __init__(
        self,
        base_dataset: _BaseDepthDataset,
        *,
        num_examples: int,
        residual_max_abs: float,
        thin_bed_min_samples: int,
        thin_bed_max_samples: int,
        ai_min: float,
        ai_max: float,
        velocity_mode: SyntheticVelocityMode = "lfm_vp",
        vp_ai_slope: float | None = None,
        vp_ai_intercept: float | None = None,
        vp_blend_alpha: float = 0.5,
        vp_smooth_samples: int = 3,
    ) -> None:
        if num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {num_examples}.")
        if residual_max_abs <= 0.0:
            raise ValueError(f"residual_max_abs must be positive, got {residual_max_abs}.")
        if thin_bed_min_samples <= 0:
            raise ValueError(f"thin_bed_min_samples must be positive, got {thin_bed_min_samples}.")
        if thin_bed_max_samples < thin_bed_min_samples:
            raise ValueError(
                "thin_bed_max_samples must be >= thin_bed_min_samples, "
                f"got {thin_bed_max_samples} < {thin_bed_min_samples}."
            )
        if ai_min <= 0.0 or ai_max <= ai_min:
            raise ValueError(f"Invalid AI bounds: ai_min={ai_min}, ai_max={ai_max}.")
        if velocity_mode not in ("lfm_vp", "from_ai_linear", "blend"):
            raise ValueError(f"Unsupported synthetic velocity_mode={velocity_mode!r}.")
        if velocity_mode in ("from_ai_linear", "blend"):
            if vp_ai_slope is None or vp_ai_slope <= 0.0:
                raise ValueError(f"vp_ai_slope must be positive for velocity_mode={velocity_mode!r}.")
            if vp_ai_intercept is None:
                raise ValueError(f"vp_ai_intercept is required for velocity_mode={velocity_mode!r}.")
        if not 0.0 <= vp_blend_alpha <= 1.0:
            raise ValueError(f"vp_blend_alpha must be within [0, 1], got {vp_blend_alpha}.")
        if vp_smooth_samples < 1:
            raise ValueError(f"vp_smooth_samples must be >= 1, got {vp_smooth_samples}.")

        self.base_dataset = base_dataset
        self.num_examples = int(num_examples)
        self.residual_max_abs = float(residual_max_abs)
        self.thin_bed_min_samples = int(thin_bed_min_samples)
        self.thin_bed_max_samples = int(thin_bed_max_samples)
        self.ai_min = float(ai_min)
        self.ai_max = float(ai_max)
        self.velocity_mode = velocity_mode
        self.vp_ai_slope = vp_ai_slope
        self.vp_ai_intercept = vp_ai_intercept
        self.vp_blend_alpha = float(vp_blend_alpha)
        self.vp_smooth_samples = int(vp_smooth_samples)

        self._vp_clip_min, self._vp_clip_max = self._estimate_velocity_clip()

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        base_idx = int(np.random.randint(0, len(self.base_dataset)))
        item = dict(self.base_dataset[base_idx])

        lfm = item["lfm_raw"].squeeze(0).numpy().astype(np.float32, copy=False)
        base_vp = item["velocity_raw"].squeeze(0).numpy().astype(np.float32, copy=False)
        taper = item["taper_weight"].squeeze(0).numpy().astype(np.float32, copy=False)

        residual = random_log_ai_residual(
            lfm.size,
            max_abs=self.residual_max_abs,
            thin_bed_min_samples=self.thin_bed_min_samples,
            thin_bed_max_samples=self.thin_bed_max_samples,
        )
        residual = (residual * taper).astype(np.float32, copy=False)

        safe_lfm = np.maximum(lfm, 1e-6)
        target_ai = safe_lfm * np.exp(residual)
        target_ai = np.clip(target_ai, self.ai_min, self.ai_max).astype(np.float32, copy=False)
        residual = np.log(np.maximum(target_ai, 1e-6) / safe_lfm).astype(np.float32, copy=False)
        target_vp = self._derive_velocity(target_ai, base_vp)

        item["target_residual"] = torch.from_numpy(residual[np.newaxis]).float()
        item["target_ai"] = torch.from_numpy(target_ai[np.newaxis]).float()
        item["velocity_raw"] = torch.from_numpy(target_vp[np.newaxis]).float()
        return item

    def _derive_velocity(self, ai: np.ndarray, base_vp: np.ndarray) -> np.ndarray:
        if self.velocity_mode == "lfm_vp":
            vp = base_vp.astype(np.float32, copy=True)
        else:
            assert self.vp_ai_slope is not None
            assert self.vp_ai_intercept is not None
            vp_from_ai = (ai.astype(np.float64) - float(self.vp_ai_intercept)) / float(self.vp_ai_slope)
            if self.velocity_mode == "blend":
                vp = self.vp_blend_alpha * vp_from_ai + (1.0 - self.vp_blend_alpha) * base_vp.astype(np.float64)
            else:
                vp = vp_from_ai
            vp = _moving_average(vp.astype(np.float32), self.vp_smooth_samples)

        vp = np.clip(vp, self._vp_clip_min, self._vp_clip_max)
        return np.maximum(vp, 1.0).astype(np.float32, copy=False)

    def _estimate_velocity_clip(self) -> tuple[float, float]:
        values = []
        n_probe = min(len(self.base_dataset), 64)
        for idx in range(n_probe):
            vp = self.base_dataset[idx]["velocity_raw"].detach().cpu().numpy()
            valid = vp[np.isfinite(vp) & (vp > 0.0)]
            if valid.size:
                values.append(valid.reshape(-1))
        if not values:
            return 500.0, 8000.0
        merged = np.concatenate(values)
        lo, hi = np.percentile(merged, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 500.0, 8000.0
        return float(max(lo, 1.0)), float(hi)


def random_log_ai_residual(
    n_sample: int,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> np.ndarray:
    """Create a bounded mixed blocky/thin-bed log-AI residual."""
    if n_sample <= 0:
        raise ValueError(f"n_sample must be positive, got {n_sample}.")

    residual = np.zeros((n_sample,), dtype=np.float32)
    generators = [_add_blocky_component, _add_thin_bed_component, _add_spike_component, _add_smooth_component]
    n_components = int(np.random.randint(1, 4))
    for _ in range(n_components):
        generator = generators[int(np.random.randint(0, len(generators)))]
        generator(
            residual,
            max_abs=max_abs,
            thin_bed_min_samples=thin_bed_min_samples,
            thin_bed_max_samples=thin_bed_max_samples,
        )

    return np.clip(residual, -max_abs, max_abs).astype(np.float32, copy=False)


def _add_blocky_component(
    residual: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    del thin_bed_min_samples
    n = residual.size
    segment_max = max(thin_bed_max_samples * 4, 8)
    pos = 0
    while pos < n:
        width = int(np.random.randint(thin_bed_max_samples, segment_max + 1))
        amp = float(np.random.uniform(-0.55 * max_abs, 0.55 * max_abs))
        residual[pos : min(n, pos + width)] += amp
        pos += width


def _add_thin_bed_component(
    residual: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    n = residual.size
    n_beds = int(np.random.randint(1, max(2, n // max(thin_bed_max_samples * 6, 1))))
    for _ in range(n_beds):
        start = int(np.random.randint(0, n))
        width = int(np.random.randint(thin_bed_min_samples, thin_bed_max_samples + 1))
        amp = float(np.random.uniform(-max_abs, max_abs))
        residual[start : min(n, start + width)] += amp


def _add_spike_component(
    residual: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    del thin_bed_min_samples, thin_bed_max_samples
    n = residual.size
    center = int(np.random.randint(0, n))
    sigma = float(np.random.uniform(1.0, 3.0))
    amp = float(np.random.uniform(-max_abs, max_abs))
    x = np.arange(n, dtype=np.float32)
    residual += (amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)).astype(np.float32)


def _add_smooth_component(
    residual: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    del thin_bed_min_samples, thin_bed_max_samples
    n = residual.size
    n_knots = int(np.random.randint(4, 9))
    x = np.linspace(0, n - 1, n_knots)
    y = np.random.uniform(-0.35 * max_abs, 0.35 * max_abs, size=n_knots)
    residual += np.interp(np.arange(n), x, y).astype(np.float32)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32, copy=False)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.full((window,), 1.0 / float(window), dtype=np.float32)
    smoothed = np.convolve(padded, kernel, mode="valid")
    if smoothed.size != values.size:
        raise RuntimeError(f"Moving average changed sample count: {values.size} -> {smoothed.size}.")
    if not np.all(np.isfinite(smoothed)):
        raise ValueError("Smoothed velocity contains non-finite values.")
    return smoothed.astype(np.float32, copy=False)
