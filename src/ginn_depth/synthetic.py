"""Standalone synthetic depth-domain samples for resolution enhancement experiments.

This module is intentionally not wired into ``ginn_depth.trainer``.  It keeps the
well-prior and thin-bed sample generation utilities that may be reused by a
future second-stage enhancement workflow built on top of a stage-1 GINN base AI.
"""

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
        unresolved_oversample_factor: int = 6,
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
        if unresolved_oversample_factor < 1:
            raise ValueError("unresolved_oversample_factor must be >= 1.")
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
        self.unresolved_oversample_factor = int(unresolved_oversample_factor)
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
        self._depth_axis_m = _extract_forward_depth_axis(forward_model)
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
            residual = self._sample_well_patch_residual(lfm.size, taper, loss_mask)
            mode_code = 0
            highres_residual = None
            highres_ai = None
            highres_reflectivity = None
            highres_depth = None
        else:
            residual, highres_residual, highres_depth = self._sample_unresolved_cluster_residual(
                lfm.size,
                taper,
                loss_mask,
                safe_lfm,
            )
            mode_code = 1

        if mode == "well_patch":
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
        else:
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

        if mode == "unresolved_cluster" and highres_residual is not None and highres_depth is not None:
            highres_lfm = np.interp(highres_depth, self._sample_depth_axis(lfm.size), safe_lfm).astype(np.float32)
            highres_residual = fit_residual_to_lfm_bounds(
                highres_residual,
                safe_lfm=np.maximum(highres_lfm, 1e-6),
                ai_min=self.ai_min,
                ai_max=self.ai_max,
                max_abs=self.residual_max_abs,
            )
            highres_ai = np.clip(highres_lfm * np.exp(highres_residual), self.ai_min, self.ai_max).astype(np.float32)
            highres_vp = np.interp(highres_depth, self._sample_depth_axis(lfm.size), target_vp).astype(np.float32)
            highres_reflectivity = ai_to_reflectivity(highres_ai).astype(np.float32)
            target_seismic, target_seismic_raw, rms_scale = self._forward_highres_target_seismic(
                highres_ai,
                highres_vp,
                highres_depth,
                item.get("dynamic_gain"),
                loss_mask,
                lfm.size,
            )
        else:
            target_seismic, target_seismic_raw, rms_scale = self._forward_target_seismic(
                target_ai,
                target_vp,
                item.get("dynamic_gain"),
                loss_mask,
            )
        if "dynamic_gain" in item and float(rms_scale) != 1.0:
            item["dynamic_gain"] = item["dynamic_gain"] * float(rms_scale)
        reflectivity = ai_to_reflectivity(target_ai)
        if highres_residual is None or highres_ai is None or highres_reflectivity is None or highres_depth is None:
            depth = self._sample_depth_axis(lfm.size)
            highres_depth = _make_highres_depth_axis(depth, self.unresolved_oversample_factor)
            highres_residual = np.interp(highres_depth, depth, residual).astype(np.float32)
            highres_ai = np.interp(highres_depth, depth, target_ai).astype(np.float32)
            highres_reflectivity = ai_to_reflectivity(highres_ai).astype(np.float32)

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
        item["target_residual_highres"] = torch.from_numpy(highres_residual[np.newaxis]).float()
        item["target_ai_highres"] = torch.from_numpy(highres_ai[np.newaxis]).float()
        item["raw_reflectivity_highres"] = torch.from_numpy(highres_reflectivity[np.newaxis]).float()
        item["depth_highres"] = torch.from_numpy(highres_depth[np.newaxis]).float()
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

    def _sample_well_patch_residual(
        self,
        n_sample: int,
        taper: np.ndarray,
        support_mask: np.ndarray,
    ) -> np.ndarray:
        residual = np.zeros((n_sample,), dtype=np.float32)
        patch_window = self._sample_well_patch_window(
            n_sample,
            taper,
            support_mask,
            min_len_samples=max(8, self.cluster_main_lobe_samples),
        )
        if patch_window is None:
            return residual

        patch, dst0 = patch_window
        length = patch.size
        patch = patch.astype(np.float32, copy=True)
        patch -= float(np.mean(patch))
        patch *= float(np.random.uniform(self.well_patch_scale_min, self.well_patch_scale_max))
        patch = np.clip(patch, -self.residual_max_abs, self.residual_max_abs)
        residual[dst0 : dst0 + length] = patch * edge_taper(length)
        return residual

    def _sample_well_patch_window(
        self,
        n_sample: int,
        taper: np.ndarray,
        support_mask: np.ndarray | None = None,
        *,
        min_len_samples: int,
    ) -> tuple[np.ndarray, int] | None:
        active = np.asarray(taper) > 0.0
        if support_mask is not None:
            support = np.asarray(support_mask, dtype=bool).reshape(-1)
            if support.shape == active.shape and np.any(active & support):
                active = active & support
        active_runs = true_runs(active)
        if not active_runs:
            return None

        row_idx = int(np.random.choice(self._well_rows))
        well_mask = np.asarray(self.prior.well_mask[row_idx], dtype=bool)
        well_residual = np.asarray(self.prior.residual_log_ai[row_idx], dtype=np.float32)
        well_runs = true_runs(well_mask)
        if not well_runs:
            return None

        src_start, src_stop = well_runs[int(np.random.randint(0, len(well_runs)))]
        dst_start, dst_stop = active_runs[int(np.random.randint(0, len(active_runs)))]
        max_len = min(src_stop - src_start, dst_stop - dst_start, n_sample)
        if max_len <= 0:
            return None
        min_len = min(max_len, max(4, int(min_len_samples)))
        length = int(np.random.randint(min_len, max_len + 1)) if max_len > min_len else int(max_len)
        src0 = int(np.random.randint(src_start, src_stop - length + 1))
        dst0 = int(np.random.randint(dst_start, dst_stop - length + 1))
        patch = well_residual[src0 : src0 + length].astype(np.float32, copy=True)
        return patch, dst0

    def _sample_unresolved_cluster_residual(
        self,
        n_sample: int,
        taper: np.ndarray,
        support_mask: np.ndarray,
        safe_lfm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        coarse = np.zeros((n_sample,), dtype=np.float32)
        taper_1d = np.asarray(taper, dtype=np.float32).reshape(-1)
        support = np.asarray(support_mask, dtype=bool).reshape(-1)
        active = taper_1d > 0.0
        if support.shape == active.shape and np.any(active & support):
            active = active & support
        active_runs = true_runs(active)
        if not active_runs:
            return coarse, None, None

        main_lobe = max(4, min(self.cluster_main_lobe_samples, n_sample))
        patch_window = self._sample_well_patch_window(
            n_sample,
            taper_1d,
            support,
            min_len_samples=max(main_lobe * 2, 16),
        )
        if patch_window is None:
            return coarse, None, None

        source_patch, dst0 = patch_window
        length = source_patch.size
        if length <= 0:
            return coarse, None, None

        source_patch = source_patch.astype(np.float32, copy=True)
        source_patch -= float(np.mean(source_patch))
        patch_scale = float(np.random.uniform(self.well_patch_scale_min, self.well_patch_scale_max))
        source_patch *= patch_scale

        depth = self._sample_depth_axis(n_sample)
        factor = max(1, self.unresolved_oversample_factor)
        highres_depth = _make_highres_depth_axis(depth, factor)
        highres = np.zeros((highres_depth.size,), dtype=np.float32)

        patch_depth = depth[dst0 : dst0 + length]
        if patch_depth.size < 2:
            return coarse, None, None
        hi_start = int(dst0 * factor)
        hi_stop = int((dst0 + length - 1) * factor + 1)
        hi_stop = min(hi_stop, highres.size)
        if hi_stop <= hi_start:
            return coarse, None, None

        local_depth = highres_depth[hi_start:hi_stop]
        local = np.zeros((local_depth.size,), dtype=np.float32)
        source_hi = np.interp(local_depth, patch_depth, source_patch).astype(np.float32)
        taper_hi = np.interp(highres_depth, depth, taper_1d).astype(np.float32)
        safe_lfm_hi = np.interp(highres_depth, depth, safe_lfm).astype(np.float32)

        background_scale = float(np.random.uniform(0.20, 0.45))
        local += background_scale * source_hi

        amp_min = self.cluster_amp_abs_p95_min * self.residual_abs_p95
        amp_max = self.cluster_amp_abs_p99_max * self.residual_abs_p99
        amp_hi = max(amp_min, amp_max)
        abs_patch = np.abs(source_hi)
        if np.any(np.isfinite(abs_patch)) and float(np.max(abs_patch)) > 0.0:
            envelope = moving_average(abs_patch, max(3, (main_lobe * factor) // 2))
            env_max = float(np.max(envelope))
            if env_max > 0.0 and np.isfinite(env_max):
                envelope = envelope / env_max
            else:
                envelope = np.ones_like(abs_patch)
        else:
            envelope = np.ones_like(abs_patch)

        main_lobe_hi = max(4, main_lobe * factor)
        local_len = local.size
        packet_spacing = max(main_lobe_hi * 2, 1)
        n_packets = max(1, int(np.ceil(local_len / packet_spacing)))
        n_packets = min(n_packets, max(1, local_len // max(4, main_lobe_hi // 2)))
        centers = np.linspace(main_lobe_hi / 2.0, max(main_lobe_hi / 2.0, local_len - main_lobe_hi / 2.0), n_packets)
        centers += np.random.uniform(-0.35 * main_lobe_hi, 0.35 * main_lobe_hi, size=n_packets)
        centers = np.clip(centers, 0, local_len - 1)

        for packet_center in centers:
            packet_width = int(round(main_lobe_hi * float(np.random.uniform(0.85, 1.45))))
            packet_width = max(4, min(packet_width, local_len))
            packet_start = int(round(float(packet_center))) - packet_width // 2
            packet_start = max(0, min(packet_start, local_len - packet_width))
            packet_stop = packet_start + packet_width

            n_events = int(np.random.randint(self.cluster_min_events, self.cluster_max_events + 1))
            n_events = max(2, min(n_events, max(2, packet_width // 2)))
            center_idx = int(np.clip(round(float(packet_center)), 0, local_len - 1))
            sign0 = 1.0 if source_hi[center_idx] >= 0.0 else -1.0

            local_env = float(np.clip(envelope[center_idx], 0.35, 1.0))
            amp_scale = float(np.random.uniform(amp_min, amp_hi)) * local_env
            source_amp = float(np.percentile(abs_patch[max(0, packet_start) : packet_stop], 75)) if packet_stop > packet_start else 0.0
            if np.isfinite(source_amp) and source_amp > 0.0:
                amp_scale = 0.5 * amp_scale + 0.5 * source_amp

            packet = _markov_thin_bed_packet(
                packet_width,
                n_events=n_events,
                amp_scale=amp_scale,
                sign0=sign0,
            )
            packet *= edge_taper(packet_width)
            local[packet_start:packet_stop] += packet

        local -= float(np.mean(local))
        local *= edge_taper(local_len)
        highres[hi_start:hi_stop] = local
        highres *= taper_hi
        highres = fit_residual_to_lfm_bounds(
            highres,
            safe_lfm=np.maximum(safe_lfm_hi, 1e-6),
            ai_min=self.ai_min,
            ai_max=self.ai_max,
            max_abs=self.residual_max_abs,
        )

        coarse = _downsample_highres_to_samples(highres, factor, n_sample)
        coarse = np.clip(coarse, -self.residual_max_abs, self.residual_max_abs).astype(np.float32, copy=False)
        return coarse, highres.astype(np.float32, copy=False), highres_depth.astype(np.float32, copy=False)

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

    def _forward_highres_target_seismic(
        self,
        highres_ai: np.ndarray,
        highres_vp: np.ndarray,
        highres_depth: np.ndarray,
        dynamic_gain: torch.Tensor | None,
        loss_mask: np.ndarray,
        n_sample: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        ai_tensor = torch.from_numpy(highres_ai[np.newaxis, np.newaxis]).float()
        vp_tensor = torch.from_numpy(highres_vp[np.newaxis, np.newaxis]).float()
        with torch.no_grad():
            seismic = self.forward_model(ai_tensor, vp_tensor, depth_axis_m=highres_depth)
        raw_hi = seismic.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        depth = self._sample_depth_axis(n_sample)
        raw = np.interp(depth, highres_depth, raw_hi).astype(np.float32)
        if dynamic_gain is not None:
            gain = dynamic_gain.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            if gain.shape == raw.shape:
                raw = (raw * gain).astype(np.float32)

        scale = 1.0
        if self.seismic_rms_match and dynamic_gain is not None:
            valid = np.asarray(loss_mask, dtype=bool) & np.isfinite(raw)
            rms = float(np.sqrt(np.mean(raw[valid] ** 2))) if np.any(valid) else 0.0
            if rms > 0.0 and np.isfinite(rms):
                scale = float(np.clip(self.seismic_rms_target / rms, self.seismic_rms_scale_min, self.seismic_rms_scale_max))
        matched = (raw * scale).astype(np.float32)
        return matched, raw, scale

    def _sample_depth_axis(self, n_sample: int) -> np.ndarray:
        if self._depth_axis_m is not None and self._depth_axis_m.size == n_sample:
            return self._depth_axis_m
        return np.arange(n_sample, dtype=np.float32)


def _extract_forward_depth_axis(forward_model: DepthForwardModel) -> np.ndarray | None:
    depth_axis = getattr(getattr(forward_model, "matrix_builder", None), "depth_axis_m", None)
    if depth_axis is None:
        return None
    try:
        values = depth_axis.detach().cpu().numpy().astype(np.float32, copy=True)
    except AttributeError:
        values = np.asarray(depth_axis, dtype=np.float32)
    values = values.reshape(-1)
    if values.size < 2:
        return None
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        return None
    return values


def _make_highres_depth_axis(depth_axis: np.ndarray, factor: int) -> np.ndarray:
    depth = np.asarray(depth_axis, dtype=np.float32).reshape(-1)
    if factor <= 1 or depth.size < 2:
        return depth.copy()
    segments = [np.linspace(float(depth[i]), float(depth[i + 1]), factor + 1, dtype=np.float32)[:-1] for i in range(depth.size - 1)]
    return np.concatenate([*segments, depth[-1:].astype(np.float32)])


def _downsample_highres_to_samples(values: np.ndarray, factor: int, n_sample: int) -> np.ndarray:
    highres = np.asarray(values, dtype=np.float32).reshape(-1)
    if factor <= 1:
        return highres[:n_sample].astype(np.float32, copy=True)
    coarse = np.zeros((n_sample,), dtype=np.float32)
    for idx in range(n_sample):
        center = idx * factor
        start = max(0, center - factor // 2)
        stop = min(highres.size, center + (factor + 1) // 2 + 1)
        if stop <= start:
            coarse[idx] = highres[min(center, highres.size - 1)]
        else:
            coarse[idx] = float(np.mean(highres[start:stop]))
    return coarse


def _markov_thin_bed_packet(width: int, *, n_events: int, amp_scale: float, sign0: float) -> np.ndarray:
    width = int(max(1, width))
    target_transitions = max(1, int(n_events))
    sand_fraction = float(np.random.uniform(0.35, 0.65))
    avg_run = max(1.0, width / float(target_transitions + 1))
    beta = float(np.clip(1.0 / avg_run, 0.05, 0.95))
    alpha = float(np.clip(sand_fraction / (avg_run * max(1e-3, 1.0 - sand_fraction)), 0.05, 0.95))
    transition = np.array([[1.0 - alpha, alpha], [beta, 1.0 - beta]], dtype=np.float64)

    best_states = None
    best_score = float("inf")
    for _ in range(16):
        states = np.empty((width,), dtype=np.int8)
        states[0] = int(np.random.random() < sand_fraction)
        for idx in range(1, width):
            states[idx] = int(np.random.choice([0, 1], p=transition[int(states[idx - 1])]))
        transitions = int(np.count_nonzero(states[1:] != states[:-1]))
        score = abs(transitions - target_transitions)
        if score < best_score:
            best_score = score
            best_states = states
        if score == 0:
            break

    assert best_states is not None
    levels = np.where(best_states > 0, 1.0, -1.0).astype(np.float32)
    levels -= float(np.mean(levels))
    peak = float(np.max(np.abs(levels)))
    if peak > 0.0:
        levels /= peak
    return (float(sign0) * float(amp_scale) * levels).astype(np.float32)


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
