"""Synthetic depth-domain samples for GINN resolution-prior experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from ginn.enhance import (
    WellResolutionPriorBundle,
    ai_to_reflectivity,
    edge_taper,
    fit_residual_to_lfm_bounds,
    highpass_log_ai_residual,
    load_well_resolution_prior_npz,
    moving_average,
    random_reflectivity_in_taper,
    reflectivity_to_log_ai,
    true_runs,
    validate_ai_bounds,
)
from ginn_depth.physics import DepthForwardModel

SyntheticVelocityMode = Literal["lfm_vp", "from_ai_linear", "blend"]
WellGuidedMode = Literal["well_patch", "unresolved_cluster"]


class _BaseDepthDataset(Protocol):
    seis_rms: float
    lfm_scale: float
    dynamic_gain_median: float | None

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...


class SyntheticDepthTraceDataset(Dataset):
    """Legacy exact-residual synthetic dataset kept for helper compatibility."""

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
        validate_ai_bounds(ai_min, ai_max)
        _validate_velocity_mode(velocity_mode, vp_ai_slope, vp_ai_intercept, vp_blend_alpha, vp_smooth_samples)

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
        self._vp_clip_min, self._vp_clip_max = _estimate_velocity_clip(base_dataset)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        base_idx = int(np.random.randint(0, len(self.base_dataset)))
        item = dict(self.base_dataset[base_idx])

        lfm = item["lfm_raw"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        base_vp = item["velocity_raw"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        taper = item["taper_weight"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        safe_lfm = np.maximum(lfm, 1e-6)
        reflectivity = random_reflectivity_in_taper(
            max(lfm.size - 1, 1),
            taper=taper,
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
        residual = fit_residual_to_lfm_bounds(
            residual * taper,
            safe_lfm=safe_lfm,
            ai_min=self.ai_min,
            ai_max=self.ai_max,
            max_abs=self.residual_max_abs,
        )
        target_ai = safe_lfm * np.exp(residual)
        target_ai = np.clip(target_ai, self.ai_min, self.ai_max).astype(np.float32, copy=False)
        residual = np.log(np.maximum(target_ai, 1e-6) / safe_lfm).astype(np.float32, copy=False)
        target_vp = _derive_velocity(
            target_ai,
            base_vp,
            velocity_mode=self.velocity_mode,
            vp_ai_slope=self.vp_ai_slope,
            vp_ai_intercept=self.vp_ai_intercept,
            vp_blend_alpha=self.vp_blend_alpha,
            vp_smooth_samples=self.vp_smooth_samples,
            vp_clip_min=self._vp_clip_min,
            vp_clip_max=self._vp_clip_max,
        )

        item["target_residual"] = torch.from_numpy(residual[np.newaxis]).float()
        item["target_ai"] = torch.from_numpy(target_ai[np.newaxis]).float()
        item["raw_reflectivity"] = torch.from_numpy(reflectivity[np.newaxis]).float()
        item["velocity_raw"] = torch.from_numpy(target_vp[np.newaxis]).float()
        return item


class WellGuidedSyntheticDepthTraceDataset(Dataset):
    """Well-prior guided synthetic traces with unresolved high-frequency structure.

    The dataset is diagnostic-first: it generates target AI/residual traces and a
    matched synthetic seismic trace, but it does not define a training loss.
    """

    def __init__(
        self,
        base_dataset: _BaseDepthDataset,
        prior: WellResolutionPriorBundle | str | Path,
        forward_model: DepthForwardModel,
        *,
        num_examples: int,
        ai_min: float,
        ai_max: float,
        patch_fraction: float = 0.70,
        unresolved_fraction: float = 0.30,
        well_patch_scale_min: float = 0.35,
        well_patch_scale_max: float = 0.80,
        cluster_min_events: int = 2,
        cluster_max_events: int = 5,
        cluster_amp_abs_p95_min: float = 0.45,
        cluster_amp_abs_p99_max: float = 1.00,
        cluster_main_lobe_samples: int | None = None,
        default_main_lobe_samples: int = 12,
        residual_max_abs: float | None = None,
        residual_highpass_samples: int = 31,
        seismic_rms_match: bool = True,
        seismic_rms_target: float = 1.0,
        seismic_rms_scale_min: float = 0.5,
        seismic_rms_scale_max: float = 2.0,
        quality_gate_enabled: bool = True,
        max_residual_near_clip_fraction: float | None = 0.02,
        max_seismic_rms_ratio: float | None = 2.0,
        max_seismic_abs_p99_ratio: float | None = 2.5,
        max_resample_attempts: int = 8,
        velocity_mode: SyntheticVelocityMode = "lfm_vp",
        vp_ai_slope: float | None = None,
        vp_ai_intercept: float | None = None,
        vp_blend_alpha: float = 0.5,
        vp_smooth_samples: int = 3,
    ) -> None:
        if num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {num_examples}.")
        validate_ai_bounds(ai_min, ai_max)
        if patch_fraction < 0.0 or unresolved_fraction < 0.0 or patch_fraction + unresolved_fraction <= 0.0:
            raise ValueError("patch_fraction and unresolved_fraction must be non-negative with positive sum.")
        if well_patch_scale_min < 0.0 or well_patch_scale_max < well_patch_scale_min:
            raise ValueError("well patch scale bounds are invalid.")
        if cluster_min_events < 1 or cluster_max_events < cluster_min_events:
            raise ValueError("cluster event bounds are invalid.")
        if cluster_amp_abs_p95_min < 0.0 or cluster_amp_abs_p99_max < cluster_amp_abs_p95_min:
            raise ValueError("cluster amplitude factors are invalid.")
        if default_main_lobe_samples < 1:
            raise ValueError("default_main_lobe_samples must be positive.")
        if residual_highpass_samples < 3:
            raise ValueError("residual_highpass_samples must be >= 3.")
        if seismic_rms_target <= 0.0:
            raise ValueError("seismic_rms_target must be positive.")
        if seismic_rms_scale_min <= 0.0 or seismic_rms_scale_max < seismic_rms_scale_min:
            raise ValueError("seismic RMS scale bounds are invalid.")
        if max_resample_attempts < 1:
            raise ValueError("max_resample_attempts must be >= 1.")
        for name, value in {
            "max_residual_near_clip_fraction": max_residual_near_clip_fraction,
            "max_seismic_rms_ratio": max_seismic_rms_ratio,
            "max_seismic_abs_p99_ratio": max_seismic_abs_p99_ratio,
        }.items():
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when provided.")
        _validate_velocity_mode(velocity_mode, vp_ai_slope, vp_ai_intercept, vp_blend_alpha, vp_smooth_samples)

        self.base_dataset = base_dataset
        self.prior = load_well_resolution_prior_npz(prior) if isinstance(prior, (str, Path)) else prior
        self.forward_model = forward_model
        self.num_examples = int(num_examples)
        self.ai_min = float(ai_min)
        self.ai_max = float(ai_max)
        self.patch_fraction = float(patch_fraction)
        self.unresolved_fraction = float(unresolved_fraction)
        self.well_patch_scale_min = float(well_patch_scale_min)
        self.well_patch_scale_max = float(well_patch_scale_max)
        self.cluster_min_events = int(cluster_min_events)
        self.cluster_max_events = int(cluster_max_events)
        self.cluster_amp_abs_p95_min = float(cluster_amp_abs_p95_min)
        self.cluster_amp_abs_p99_max = float(cluster_amp_abs_p99_max)
        self.cluster_main_lobe_samples = int(
            cluster_main_lobe_samples if cluster_main_lobe_samples is not None else default_main_lobe_samples
        )
        self.residual_highpass_samples = int(residual_highpass_samples)
        self.seismic_rms_match = bool(seismic_rms_match)
        self.seismic_rms_target = float(seismic_rms_target)
        self.seismic_rms_scale_min = float(seismic_rms_scale_min)
        self.seismic_rms_scale_max = float(seismic_rms_scale_max)
        self.quality_gate_enabled = bool(quality_gate_enabled)
        self.max_residual_near_clip_fraction = max_residual_near_clip_fraction
        self.max_seismic_rms_ratio = max_seismic_rms_ratio
        self.max_seismic_abs_p99_ratio = max_seismic_abs_p99_ratio
        self.max_resample_attempts = int(max_resample_attempts)
        self.velocity_mode = velocity_mode
        self.vp_ai_slope = vp_ai_slope
        self.vp_ai_intercept = vp_ai_intercept
        self.vp_blend_alpha = float(vp_blend_alpha)
        self.vp_smooth_samples = int(vp_smooth_samples)

        self._vp_clip_min, self._vp_clip_max = _estimate_velocity_clip(base_dataset)
        self._prior_values = self.prior.residual_log_ai[self.prior.well_mask]
        self._prior_values = self._prior_values[np.isfinite(self._prior_values)]
        if self._prior_values.size == 0:
            raise ValueError("Well resolution prior contains no finite residual values.")
        self.residual_abs_p95 = _summary_or_percentile(self.prior, "abs_p95", 95.0)
        self.residual_abs_p99 = _summary_or_percentile(self.prior, "abs_p99", 99.0)
        self.residual_max_abs = float(residual_max_abs or max(self.residual_abs_p99, self.residual_abs_p95, 1e-3))
        if self.residual_max_abs <= 0.0:
            raise ValueError("residual_max_abs must be positive.")

        self._well_rows = _valid_prior_rows(self.prior)
        if not self._well_rows:
            raise ValueError("Well resolution prior contains no valid well rows.")

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        for attempt in range(self.max_resample_attempts):
            item = self._sample_item(idx)
            if not self.quality_gate_enabled or self._passes_quality_gate(item):
                item["synthetic_resample_attempts"] = torch.tensor(attempt + 1, dtype=torch.int64)
                return item
        item["synthetic_resample_attempts"] = torch.tensor(self.max_resample_attempts, dtype=torch.int64)
        return item

    def _sample_item(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        base_idx = int(np.random.randint(0, len(self.base_dataset)))
        item = dict(self.base_dataset[base_idx])

        lfm = item["lfm_raw"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        base_vp = item["velocity_raw"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        taper = item["taper_weight"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        core_mask = item["mask"].squeeze(0).detach().cpu().numpy().astype(bool, copy=False)
        loss_mask = item.get("loss_mask", item["mask"]).squeeze(0).detach().cpu().numpy().astype(bool, copy=False)

        safe_lfm = np.maximum(lfm, 1e-6)
        mode = self._sample_mode()
        if mode == "well_patch":
            residual = self._sample_well_patch_residual(lfm.size, taper)
            mode_code = 0
        else:
            residual = self._sample_unresolved_cluster_residual(lfm.size, taper)
            mode_code = 1

        residual = highpass_log_ai_residual(
            residual,
            window=self.residual_highpass_samples,
            max_abs=self.residual_max_abs,
        )
        residual = fit_residual_to_lfm_bounds(
            residual * taper,
            safe_lfm=safe_lfm,
            ai_min=self.ai_min,
            ai_max=self.ai_max,
            max_abs=self.residual_max_abs,
        )
        target_ai = np.clip(safe_lfm * np.exp(residual), self.ai_min, self.ai_max).astype(np.float32)
        residual = np.log(np.maximum(target_ai, 1e-6) / safe_lfm).astype(np.float32, copy=False)
        target_vp = _derive_velocity(
            target_ai,
            base_vp,
            velocity_mode=self.velocity_mode,
            vp_ai_slope=self.vp_ai_slope,
            vp_ai_intercept=self.vp_ai_intercept,
            vp_blend_alpha=self.vp_blend_alpha,
            vp_smooth_samples=self.vp_smooth_samples,
            vp_clip_min=self._vp_clip_min,
            vp_clip_max=self._vp_clip_max,
        )

        target_seismic, target_seismic_raw, rms_scale = self._forward_target_seismic(
            target_ai,
            target_vp,
            item.get("dynamic_gain"),
            loss_mask,
        )
        if "dynamic_gain" in item and float(rms_scale) != 1.0:
            item["dynamic_gain"] = item["dynamic_gain"] * float(rms_scale)
        reflectivity = ai_to_reflectivity(target_ai)

        item["target_residual"] = torch.from_numpy(residual[np.newaxis]).float()
        item["target_ai"] = torch.from_numpy(target_ai[np.newaxis]).float()
        item["target_seismic"] = torch.from_numpy(target_seismic[np.newaxis]).float()
        item["target_seismic_raw"] = torch.from_numpy(target_seismic_raw[np.newaxis]).float()
        item["synthetic_rms_scale"] = torch.tensor(float(rms_scale), dtype=torch.float32)
        item["raw_reflectivity"] = torch.from_numpy(reflectivity[np.newaxis]).float()
        item["velocity_raw"] = torch.from_numpy(target_vp[np.newaxis]).float()
        item["synthetic_mode"] = torch.tensor(mode_code, dtype=torch.int64)
        item["mask"] = torch.from_numpy(core_mask[np.newaxis]).bool()
        item["loss_mask"] = torch.from_numpy(loss_mask[np.newaxis]).bool()
        return item

    def _passes_quality_gate(self, item: dict[str, torch.Tensor]) -> bool:
        loss_mask = item["loss_mask"].squeeze(0).detach().cpu().numpy().astype(bool, copy=False)
        residual = item["target_residual"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        target_seismic = item["target_seismic"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        source_seismic = item["obs"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        valid = loss_mask & np.isfinite(residual) & np.isfinite(target_seismic) & np.isfinite(source_seismic)
        if not np.any(valid):
            return False

        if self.max_residual_near_clip_fraction is not None:
            near_clip = float(np.mean(np.abs(residual[valid]) >= 0.98 * self.residual_max_abs))
            if near_clip > float(self.max_residual_near_clip_fraction):
                return False

        if self.max_seismic_rms_ratio is not None:
            source_rms = _rms(source_seismic[valid])
            target_rms = _rms(target_seismic[valid])
            if source_rms > 0.0 and np.isfinite(source_rms):
                if target_rms / source_rms > float(self.max_seismic_rms_ratio):
                    return False

        if self.max_seismic_abs_p99_ratio is not None:
            source_p99 = _abs_percentile(source_seismic[valid], 99.0)
            target_p99 = _abs_percentile(target_seismic[valid], 99.0)
            if source_p99 > 0.0 and np.isfinite(source_p99):
                if target_p99 / source_p99 > float(self.max_seismic_abs_p99_ratio):
                    return False

        return True

    def _sample_mode(self) -> WellGuidedMode:
        p_patch = self.patch_fraction / (self.patch_fraction + self.unresolved_fraction)
        return "well_patch" if float(np.random.random()) < p_patch else "unresolved_cluster"

    def _sample_well_patch_residual(self, n_sample: int, taper: np.ndarray) -> np.ndarray:
        residual = np.zeros((n_sample,), dtype=np.float32)
        active_runs = true_runs(np.asarray(taper) > 0.0)
        if not active_runs:
            return residual

        row_idx = int(np.random.choice(self._well_rows))
        well_mask = np.asarray(self.prior.well_mask[row_idx], dtype=bool)
        well_residual = np.asarray(self.prior.residual_log_ai[row_idx], dtype=np.float32)
        well_runs = true_runs(well_mask)
        if not well_runs:
            return residual

        src_start, src_stop = well_runs[int(np.random.randint(0, len(well_runs)))]
        dst_start, dst_stop = active_runs[int(np.random.randint(0, len(active_runs)))]
        max_len = min(src_stop - src_start, dst_stop - dst_start, n_sample)
        if max_len <= 0:
            return residual
        min_len = min(max_len, max(8, self.cluster_main_lobe_samples))
        length = int(np.random.randint(min_len, max_len + 1)) if max_len > min_len else int(max_len)
        src0 = int(np.random.randint(src_start, src_stop - length + 1))
        dst0 = int(np.random.randint(dst_start, dst_stop - length + 1))
        patch = well_residual[src0 : src0 + length].astype(np.float32, copy=True)
        patch -= float(np.mean(patch))
        patch *= float(np.random.uniform(self.well_patch_scale_min, self.well_patch_scale_max))
        patch = np.clip(patch, -self.residual_max_abs, self.residual_max_abs)
        residual[dst0 : dst0 + length] = patch * edge_taper(length)
        return residual

    def _sample_unresolved_cluster_residual(self, n_sample: int, taper: np.ndarray) -> np.ndarray:
        residual = np.zeros((n_sample,), dtype=np.float32)
        active = np.asarray(taper, dtype=np.float32).reshape(-1) > 0.0
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            return residual

        n_clusters = int(np.random.randint(1, 4))
        for _ in range(n_clusters):
            center = int(np.random.choice(active_indices))
            width = max(1, min(self.cluster_main_lobe_samples, n_sample))
            n_events = int(np.random.randint(self.cluster_min_events, self.cluster_max_events + 1))
            offsets = np.random.randint(-(width // 2), width // 2 + 1, size=n_events)
            amp_min = self.cluster_amp_abs_p95_min * self.residual_abs_p95
            amp_max = self.cluster_amp_abs_p99_max * self.residual_abs_p99
            amp_scale = float(np.random.uniform(amp_min, max(amp_min, amp_max)))
            signs = np.random.choice([-1.0, 1.0], size=n_events)
            for offset, sign in zip(offsets, signs):
                pos = center + int(offset)
                if 0 <= pos < n_sample and active[pos]:
                    residual[pos] += float(sign) * amp_scale * float(np.random.uniform(0.5, 1.0))

        residual -= moving_average(residual, max(3, self.cluster_main_lobe_samples * 2 + 1))
        return np.clip(residual, -self.residual_max_abs, self.residual_max_abs).astype(np.float32)

    def _forward_target_seismic(
        self,
        target_ai: np.ndarray,
        target_vp: np.ndarray,
        dynamic_gain: torch.Tensor | None,
        loss_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        ai_tensor = torch.from_numpy(target_ai[np.newaxis, np.newaxis]).float()
        vp_tensor = torch.from_numpy(target_vp[np.newaxis, np.newaxis]).float()
        gain_tensor = dynamic_gain.float().unsqueeze(0) if dynamic_gain is not None and dynamic_gain.ndim == 2 else dynamic_gain
        with torch.no_grad():
            seismic = self.forward_model(ai_tensor, vp_tensor, gain=gain_tensor)
        raw = seismic.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        scale = 1.0
        if self.seismic_rms_match and dynamic_gain is not None:
            valid = np.asarray(loss_mask, dtype=bool) & np.isfinite(raw)
            rms = float(np.sqrt(np.mean(raw[valid] ** 2))) if np.any(valid) else 0.0
            if rms > 0.0 and np.isfinite(rms):
                scale = float(np.clip(self.seismic_rms_target / rms, self.seismic_rms_scale_min, self.seismic_rms_scale_max))
        matched = (raw * scale).astype(np.float32)
        return matched, raw, scale


def _derive_velocity(
    ai: np.ndarray,
    base_vp: np.ndarray,
    *,
    velocity_mode: SyntheticVelocityMode,
    vp_ai_slope: float | None,
    vp_ai_intercept: float | None,
    vp_blend_alpha: float,
    vp_smooth_samples: int,
    vp_clip_min: float,
    vp_clip_max: float,
) -> np.ndarray:
    if velocity_mode == "lfm_vp":
        vp = base_vp.astype(np.float32, copy=True)
    else:
        assert vp_ai_slope is not None
        assert vp_ai_intercept is not None
        vp_from_ai = (ai.astype(np.float64) - float(vp_ai_intercept)) / float(vp_ai_slope)
        if velocity_mode == "blend":
            vp = vp_blend_alpha * vp_from_ai + (1.0 - vp_blend_alpha) * base_vp.astype(np.float64)
        else:
            vp = vp_from_ai
        vp = moving_average(vp.astype(np.float32), vp_smooth_samples)

    vp = np.clip(vp, vp_clip_min, vp_clip_max)
    return np.maximum(vp, 1.0).astype(np.float32, copy=False)


def _estimate_velocity_clip(base_dataset: _BaseDepthDataset) -> tuple[float, float]:
    values = []
    n_probe = min(len(base_dataset), 64)
    for idx in range(n_probe):
        vp = base_dataset[idx]["velocity_raw"].detach().cpu().numpy()
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


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def _abs_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.percentile(np.abs(values), percentile))


def _valid_prior_rows(prior: WellResolutionPriorBundle) -> list[int]:
    return [int(row) for row in range(prior.n_wells) if int(np.asarray(prior.well_mask[row]).sum()) >= 2]


def _summary_or_percentile(prior: WellResolutionPriorBundle, key: str, percentile: float) -> float:
    summary_value = prior.summary.get("residual", {}).get(key)
    if summary_value is not None and np.isfinite(float(summary_value)):
        return float(summary_value)
    values = prior.residual_log_ai[prior.well_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-3
    return float(np.percentile(np.abs(values), percentile))


def _validate_velocity_mode(
    velocity_mode: SyntheticVelocityMode,
    vp_ai_slope: float | None,
    vp_ai_intercept: float | None,
    vp_blend_alpha: float,
    vp_smooth_samples: int,
) -> None:
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
