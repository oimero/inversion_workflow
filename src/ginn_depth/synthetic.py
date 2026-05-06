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
        log_ai_highpass_samples: int,
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
        if log_ai_highpass_samples < 3:
            raise ValueError(f"log_ai_highpass_samples must be >= 3, got {log_ai_highpass_samples}.")
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
        self.log_ai_highpass_samples = int(log_ai_highpass_samples)
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

        safe_lfm = np.maximum(lfm, 1e-6)
        reflectivity = random_reflectivity(
            max(lfm.size - 1, 1),
            max_abs=min(0.35, float(np.tanh(0.5 * self.residual_max_abs))),
            thin_bed_min_samples=self.thin_bed_min_samples,
            thin_bed_max_samples=self.thin_bed_max_samples,
        )
        log_ai_raw = reflectivity_to_log_ai(reflectivity, initial_log_ai=float(np.log(safe_lfm[0])))
        residual = highpass_log_ai_residual(
            log_ai_raw,
            window=self.log_ai_highpass_samples,
            max_abs=self.residual_max_abs,
        )
        residual = _fit_residual_to_lfm_bounds(
            residual * taper,
            safe_lfm=safe_lfm,
            ai_min=self.ai_min,
            ai_max=self.ai_max,
            max_abs=self.residual_max_abs,
        )
        target_ai = safe_lfm * np.exp(residual)
        target_ai = np.clip(target_ai, self.ai_min, self.ai_max).astype(np.float32, copy=False)
        residual = np.log(np.maximum(target_ai, 1e-6) / safe_lfm).astype(np.float32, copy=False)
        target_vp = self._derive_velocity(target_ai, base_vp)

        item["target_residual"] = torch.from_numpy(residual[np.newaxis]).float()
        item["target_ai"] = torch.from_numpy(target_ai[np.newaxis]).float()
        item["target_reflectivity"] = torch.from_numpy(reflectivity[np.newaxis]).float()
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


def random_reflectivity(
    n_interface: int,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> np.ndarray:
    """Create bounded sparse reflectivity interfaces for synthetic AI construction."""
    if n_interface <= 0:
        raise ValueError(f"n_interface must be positive, got {n_interface}.")
    if not 0.0 < max_abs < 1.0:
        raise ValueError(f"max_abs must be in (0, 1), got {max_abs}.")

    reflectivity = np.zeros((n_interface,), dtype=np.float32)
    generators = [_add_sparse_interfaces, _add_bed_boundaries, _add_thin_bed_pairs, _add_reflectivity_spike]
    n_components = int(np.random.randint(1, 3))
    for _ in range(n_components):
        generator = generators[int(np.random.randint(0, len(generators)))]
        generator(
            reflectivity,
            max_abs=max_abs,
            thin_bed_min_samples=thin_bed_min_samples,
            thin_bed_max_samples=thin_bed_max_samples,
        )

    if not np.any(reflectivity):
        reflectivity[int(np.random.randint(0, n_interface))] = float(np.random.uniform(-max_abs, max_abs))
    return np.clip(reflectivity, -max_abs, max_abs).astype(np.float32, copy=False)


def reflectivity_to_log_ai(reflectivity: np.ndarray, *, initial_log_ai: float = 0.0) -> np.ndarray:
    """Integrate normal-incidence reflectivity into a log-AI trace."""
    r = np.asarray(reflectivity, dtype=np.float32).reshape(-1)
    if r.size <= 0:
        raise ValueError("reflectivity must contain at least one interface.")
    if np.any(~np.isfinite(r)):
        raise ValueError("reflectivity contains non-finite values.")
    r = np.clip(r, -0.95, 0.95).astype(np.float64)
    increments = np.log1p(r) - np.log1p(-r)
    log_ai = np.empty((r.size + 1,), dtype=np.float64)
    log_ai[0] = float(initial_log_ai)
    log_ai[1:] = float(initial_log_ai) + np.cumsum(increments)
    return log_ai.astype(np.float32)


def highpass_log_ai_residual(log_ai_raw: np.ndarray, *, window: int, max_abs: float) -> np.ndarray:
    """Keep only the high-frequency part of a raw log-AI trace."""
    if max_abs <= 0.0:
        raise ValueError(f"max_abs must be positive, got {max_abs}.")
    log_ai = np.asarray(log_ai_raw, dtype=np.float32).reshape(-1)
    if log_ai.size <= 0:
        raise ValueError("log_ai_raw must contain at least one sample.")
    if np.any(~np.isfinite(log_ai)):
        raise ValueError("log_ai_raw contains non-finite values.")

    low = _moving_average(log_ai, int(window))
    residual = log_ai - low
    scale_ref = float(np.percentile(np.abs(residual), 99.0)) if residual.size else 0.0
    if scale_ref > max_abs:
        residual = residual * (float(max_abs) / scale_ref)
    return np.clip(residual, -max_abs, max_abs).astype(np.float32, copy=False)


def _fit_residual_to_lfm_bounds(
    residual: np.ndarray,
    *,
    safe_lfm: np.ndarray,
    ai_min: float,
    ai_max: float,
    max_abs: float,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float32)
    lower = np.log(float(ai_min) / safe_lfm)
    upper = np.log(float(ai_max) / safe_lfm)
    lower = np.maximum(lower, -float(max_abs))
    upper = np.minimum(upper, float(max_abs))
    clipped = np.clip(residual, lower, upper)
    impossible = lower > upper
    if np.any(impossible):
        clipped[impossible] = np.clip(
            residual[impossible],
            np.log(float(ai_min) / safe_lfm[impossible]),
            np.log(float(ai_max) / safe_lfm[impossible]),
        )
    return clipped.astype(np.float32, copy=False)


def _add_sparse_interfaces(
    reflectivity: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    del thin_bed_min_samples
    n = reflectivity.size
    n_events = int(np.random.randint(1, max(2, n // max(thin_bed_max_samples * 3, 1))))
    for _ in range(n_events):
        idx = int(np.random.randint(0, n))
        reflectivity[idx] += float(np.random.uniform(-0.7 * max_abs, 0.7 * max_abs))


def _add_bed_boundaries(
    reflectivity: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    n = reflectivity.size
    pos = int(np.random.randint(0, max(1, thin_bed_max_samples)))
    segment_max = max(thin_bed_max_samples * 5, thin_bed_min_samples + 1)
    while pos < n:
        reflectivity[pos] += float(np.random.uniform(-max_abs, max_abs))
        width = int(np.random.randint(thin_bed_min_samples, segment_max + 1))
        pos += width


def _add_thin_bed_pairs(
    reflectivity: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    n = reflectivity.size
    n_beds = int(np.random.randint(1, max(2, n // max(thin_bed_max_samples * 8, 1))))
    for _ in range(n_beds):
        start = int(np.random.randint(0, n))
        width = int(np.random.randint(thin_bed_min_samples, thin_bed_max_samples + 1))
        stop = start + width
        if stop >= n:
            continue
        amp = float(np.random.uniform(-max_abs, max_abs))
        reflectivity[start] += amp
        reflectivity[stop] -= amp * float(np.random.uniform(0.5, 1.0))


def _add_reflectivity_spike(
    reflectivity: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    del thin_bed_min_samples, thin_bed_max_samples
    idx = int(np.random.randint(0, reflectivity.size))
    reflectivity[idx] += float(np.random.uniform(-max_abs, max_abs))


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
