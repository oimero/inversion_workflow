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

from cup.utils.raw_trace import centered_moving_average
from enhance.prior import (
    WellResolutionPriorBundle,
    ai_to_reflectivity,
    load_well_resolution_prior_npz,
    validate_ai_bounds,
)
from enhance.synthetic import (
    downsample_highres_to_samples,
    edge_taper,
    fit_delta_to_base_ai_bounds,
    fit_residual_to_lfm_bounds,
    highpass_log_ai_residual,
    make_highres_axis,
    markov_thin_bed_packet,
    random_reflectivity_in_taper,
    reflectivity_to_log_ai,
    sample_highres_prior_patch,
    summary_or_percentile,
    true_runs,
    valid_prior_rows,
)
from ginn_depth.physics import DepthForwardModel

SyntheticVelocityMode = Literal["lfm_vp", "from_ai_linear", "blend"]
DeltaSupervisionMask = Literal["core", "loss"]


class _BaseDepthDataset(Protocol):
    seis_rms: float
    lfm_scale: float
    dynamic_gain_median: float | None
    input_channel_names: tuple[str, ...]

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
            velocity_mode=self.velocity_mode,  # type: ignore
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
        seismic_rms_match: bool = True,
        seismic_rms_target: float = 1.0,
        seismic_rms_scale_min: float = 0.5,
        seismic_rms_scale_max: float = 2.0,
        quality_gate_enabled: bool = True,
        max_residual_near_clip_fraction: float | None = 0.02,
        max_seismic_rms_ratio: float | None = 2.0,
        max_seismic_abs_p99_ratio: float | None = 2.5,
        min_target_obs_waveform_corr: float | None = 0.0,
        min_base_target_waveform_corr: float | None = None,
        input_augmentation_enabled: bool = True,
        input_phase_deg_max: float = 12.0,
        input_amp_jitter: tuple[float, float] = (0.9, 1.1),
        input_noise_rms_fraction: tuple[float, float] = (0.02, 0.06),
        input_spectral_tilt_max: float = 0.12,
        max_resample_attempts: int = 8,
        delta_supervision_mask: DeltaSupervisionMask = "core",
        velocity_mode: SyntheticVelocityMode = "lfm_vp",
        vp_ai_slope: float | None = None,
        vp_ai_intercept: float | None = None,
        vp_blend_alpha: float = 0.5,
        vp_smooth_samples: int = 3,
    ) -> None:
        if num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {num_examples}.")
        validate_ai_bounds(ai_min, ai_max)
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
        if min_base_target_waveform_corr is not None and not (-1.0 <= min_base_target_waveform_corr <= 1.0):
            raise ValueError("min_base_target_waveform_corr must be within [-1, 1] when provided.")
        if min_target_obs_waveform_corr is not None and not (-1.0 <= min_target_obs_waveform_corr <= 1.0):
            raise ValueError("min_target_obs_waveform_corr must be within [-1, 1] when provided.")
        if input_phase_deg_max < 0.0:
            raise ValueError("input_phase_deg_max must be non-negative.")
        if input_spectral_tilt_max < 0.0:
            raise ValueError("input_spectral_tilt_max must be non-negative.")
        input_amp_jitter = tuple(float(v) for v in input_amp_jitter)  # type: ignore
        if len(input_amp_jitter) != 2 or input_amp_jitter[0] <= 0.0 or input_amp_jitter[1] < input_amp_jitter[0]:
            raise ValueError("input_amp_jitter must be a positive (min, max) range.")
        input_noise_rms_fraction = tuple(float(v) for v in input_noise_rms_fraction)  # type: ignore
        if (
            len(input_noise_rms_fraction) != 2
            or input_noise_rms_fraction[0] < 0.0
            or input_noise_rms_fraction[1] < input_noise_rms_fraction[0]
        ):
            raise ValueError("input_noise_rms_fraction must be a non-negative (min, max) range.")
        if delta_supervision_mask not in ("core", "loss"):
            raise ValueError("delta_supervision_mask must be one of ['core', 'loss'].")
        _validate_velocity_mode(velocity_mode, vp_ai_slope, vp_ai_intercept, vp_blend_alpha, vp_smooth_samples)

        self.base_dataset = base_dataset
        self.prior = load_well_resolution_prior_npz(prior) if isinstance(prior, (str, Path)) else prior
        self.forward_model = forward_model
        self.num_examples = int(num_examples)
        self.ai_min = float(ai_min)
        self.ai_max = float(ai_max)
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
        self.seismic_rms_match = bool(seismic_rms_match)
        self.seismic_rms_target = float(seismic_rms_target)
        self.seismic_rms_scale_min = float(seismic_rms_scale_min)
        self.seismic_rms_scale_max = float(seismic_rms_scale_max)
        self.quality_gate_enabled = bool(quality_gate_enabled)
        self.max_residual_near_clip_fraction = max_residual_near_clip_fraction
        self.max_seismic_rms_ratio = max_seismic_rms_ratio
        self.max_seismic_abs_p99_ratio = max_seismic_abs_p99_ratio
        self.min_target_obs_waveform_corr = min_target_obs_waveform_corr
        self.min_base_target_waveform_corr = min_base_target_waveform_corr
        self.input_augmentation_enabled = bool(input_augmentation_enabled)
        self.input_phase_deg_max = float(input_phase_deg_max)
        self.input_amp_jitter = input_amp_jitter
        self.input_noise_rms_fraction = input_noise_rms_fraction
        self.input_spectral_tilt_max = float(input_spectral_tilt_max)
        self.max_resample_attempts = int(max_resample_attempts)
        self.delta_supervision_mask = delta_supervision_mask
        self.velocity_mode = velocity_mode
        self.vp_ai_slope = vp_ai_slope
        self.vp_ai_intercept = vp_ai_intercept
        self.vp_blend_alpha = float(vp_blend_alpha)
        self.vp_smooth_samples = int(vp_smooth_samples)

        self._vp_clip_min, self._vp_clip_max = _estimate_velocity_clip(base_dataset)
        self._depth_axis_m = _extract_forward_depth_axis(forward_model)
        self._prior_values = self.prior.highres_well_high_log_ai[self.prior.highres_well_mask]
        self._prior_values = self._prior_values[np.isfinite(self._prior_values)]
        if self._prior_values.size == 0:
            raise ValueError("Well resolution prior contains no finite high-frequency values.")
        self.residual_abs_p95 = summary_or_percentile(self.prior, "abs_p95", 95.0)
        self.residual_abs_p99 = summary_or_percentile(self.prior, "abs_p99", 99.0)
        self.residual_max_abs = float(residual_max_abs or max(self.residual_abs_p99, self.residual_abs_p95, 1e-3))
        if self.residual_max_abs <= 0.0:
            raise ValueError("residual_max_abs must be positive.")

        self._well_rows = valid_prior_rows(self.prior)
        if not self._well_rows:
            raise ValueError("Well resolution prior contains no valid well rows.")

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        for attempt in range(self.max_resample_attempts):
            item = self._sample_item(idx)
            gate_passed = not self.quality_gate_enabled or self._passes_quality_gate(item)
            if gate_passed:
                item["synthetic_resample_attempts"] = torch.tensor(attempt + 1, dtype=torch.int64)
                item["synthetic_quality_gate_passed"] = torch.tensor(True, dtype=torch.bool)
                item["synthetic_quality_gate_forced_accept"] = torch.tensor(False, dtype=torch.bool)
                item["synthetic_quality_gate_max_attempt_reached"] = torch.tensor(
                    self.quality_gate_enabled and attempt + 1 >= self.max_resample_attempts,
                    dtype=torch.bool,
                )
                return item
        item["synthetic_resample_attempts"] = torch.tensor(self.max_resample_attempts, dtype=torch.int64)
        item["synthetic_quality_gate_passed"] = torch.tensor(False, dtype=torch.bool)
        item["synthetic_quality_gate_forced_accept"] = torch.tensor(True, dtype=torch.bool)
        item["synthetic_quality_gate_max_attempt_reached"] = torch.tensor(True, dtype=torch.bool)
        return item

    def _sample_item(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        base_idx = int(np.random.randint(0, len(self.base_dataset)))
        item = dict(self.base_dataset[base_idx])

        base_ai = item["lfm_raw"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        base_vp = item["velocity_raw"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        taper = item["taper_weight"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        core_mask = item["mask"].squeeze(0).detach().cpu().numpy().astype(bool, copy=False)
        has_base_loss_mask = "loss_mask" in item
        if self.delta_supervision_mask == "loss":
            if not has_base_loss_mask:
                raise KeyError(
                    "Base depth dataset item must contain 'loss_mask' "
                    "when delta_supervision_mask='loss'."
                )
            loss_mask = item["loss_mask"].squeeze(0).detach().cpu().numpy().astype(bool, copy=False)
            delta_loss_mask = loss_mask
        else:
            loss_mask = (
                item["loss_mask"].squeeze(0).detach().cpu().numpy().astype(bool, copy=False)
                if has_base_loss_mask
                else core_mask
            )
            delta_loss_mask = core_mask
        waveform_qc_mask = core_mask

        safe_base_ai = np.maximum(base_ai, 1e-6)
        depth = self._sample_depth_axis(base_ai.size)
        factor = max(1, self.unresolved_oversample_factor)
        highres_depth = make_highres_axis(depth, factor)
        highres_base_ai = np.interp(highres_depth, depth, safe_base_ai).astype(np.float32)
        taper_hi = np.interp(highres_depth, depth, taper).astype(np.float32)
        support_hi = np.interp(highres_depth, depth, delta_loss_mask.astype(np.float32)) >= 0.5
        highres_residual = self._sample_unresolved_cluster_residual(
            highres_depth,
            taper_hi,
            support_hi,
            factor,
        )
        mode_code = 1
        synthetic_empty_residual = highres_residual is None
        if highres_residual is None:
            highres_residual = np.zeros_like(highres_depth, dtype=np.float32)

        highres_residual = fit_delta_to_base_ai_bounds(
            highres_residual * taper_hi,
            safe_base_ai=np.maximum(highres_base_ai, 1e-6),
            ai_min=self.ai_min,
            ai_max=self.ai_max,
            max_abs=self.residual_max_abs,
        )
        highres_ai = np.clip(highres_base_ai * np.exp(highres_residual), self.ai_min, self.ai_max).astype(np.float32)
        highres_residual = np.log(np.maximum(highres_ai, 1e-6) / np.maximum(highres_base_ai, 1e-6)).astype(
            np.float32,
            copy=False,
        )
        delta_log_ai = downsample_highres_to_samples(highres_residual, factor, base_ai.size)
        target_ai = np.clip(safe_base_ai * np.exp(delta_log_ai), self.ai_min, self.ai_max).astype(np.float32)
        delta_log_ai = np.log(np.maximum(target_ai, 1e-6) / safe_base_ai).astype(np.float32, copy=False)
        target_vp = _derive_velocity(
            target_ai,
            base_vp,
            velocity_mode=self.velocity_mode,  # type: ignore
            vp_ai_slope=self.vp_ai_slope,
            vp_ai_intercept=self.vp_ai_intercept,
            vp_blend_alpha=self.vp_blend_alpha,
            vp_smooth_samples=self.vp_smooth_samples,
            vp_clip_min=self._vp_clip_min,
            vp_clip_max=self._vp_clip_max,
        )

        highres_vp = np.interp(highres_depth, depth, target_vp).astype(np.float32)
        highres_reflectivity = ai_to_reflectivity(highres_ai).astype(np.float32)
        target_seismic, target_seismic_raw, rms_scale = self._forward_highres_target_seismic(
            highres_ai,
            highres_vp,
            highres_depth,
            item.get("dynamic_gain"),
            waveform_qc_mask,
            base_ai.size,
        )
        base_seismic_raw = self._forward_raw_seismic(
            safe_base_ai,
            base_vp,
            item.get("dynamic_gain"),
        )
        base_seismic = (base_seismic_raw * float(rms_scale)).astype(np.float32)
        if "dynamic_gain" in item and float(rms_scale) != 1.0:
            item["dynamic_gain"] = item["dynamic_gain"] * float(rms_scale)
        reflectivity = ai_to_reflectivity(target_ai)

        item["base_ai_raw"] = torch.from_numpy(base_ai[np.newaxis]).float()
        item["target_delta_log_ai"] = torch.from_numpy(delta_log_ai[np.newaxis]).float()
        item["target_residual"] = item["target_delta_log_ai"]
        item["target_ai"] = torch.from_numpy(target_ai[np.newaxis]).float()
        item["target_seismic"] = torch.from_numpy(target_seismic[np.newaxis]).float()
        item["target_seismic_raw"] = torch.from_numpy(target_seismic_raw[np.newaxis]).float()
        item["base_seismic"] = torch.from_numpy(base_seismic[np.newaxis]).float()
        item["base_seismic_raw"] = torch.from_numpy(base_seismic_raw[np.newaxis]).float()
        item["synthetic_rms_scale"] = torch.tensor(float(rms_scale), dtype=torch.float32)
        item["raw_reflectivity"] = torch.from_numpy(reflectivity[np.newaxis]).float()
        item["velocity_raw"] = torch.from_numpy(target_vp[np.newaxis]).float()
        item["synthetic_mode"] = torch.tensor(mode_code, dtype=torch.int64)
        item["synthetic_empty_residual"] = torch.tensor(synthetic_empty_residual, dtype=torch.bool)
        item["mask"] = torch.from_numpy(core_mask[np.newaxis]).bool()
        if has_base_loss_mask:
            item["loss_mask"] = torch.from_numpy(loss_mask[np.newaxis]).bool()
        item["delta_loss_mask"] = torch.from_numpy(delta_loss_mask[np.newaxis]).bool()
        input_seismic_clean = target_seismic.astype(np.float32, copy=True)
        input_seismic_augmented = self._augment_input_seismic(
            input_seismic_clean,
            item["obs"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False),
            core_mask,
        )
        item["input_seismic_clean"] = torch.from_numpy(input_seismic_clean[np.newaxis]).float()
        item["input_seismic_augmented"] = torch.from_numpy(input_seismic_augmented[np.newaxis]).float()
        item["input_augmentation_enabled"] = torch.tensor(self.input_augmentation_enabled, dtype=torch.bool)
        item["input"] = self._compose_enhancement_input(item)
        item["target_residual_highres"] = torch.from_numpy(highres_residual[np.newaxis]).float()
        item["target_ai_highres"] = torch.from_numpy(highres_ai[np.newaxis]).float()
        item["raw_reflectivity_highres"] = torch.from_numpy(highres_reflectivity[np.newaxis]).float()
        item["depth_highres"] = torch.from_numpy(highres_depth[np.newaxis]).float()
        return item

    def _compose_enhancement_input(self, item: dict[str, torch.Tensor]) -> torch.Tensor:
        input_seismic = item.get("input_seismic_augmented", item["target_seismic"])
        channels = [input_seismic.squeeze(0).detach().cpu().numpy().astype(np.float32)]
        channel_names = getattr(self.base_dataset, "input_channel_names", ())
        base_ai = item["base_ai_raw"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        if "ai_lfm" in channel_names or "base_ai" in channel_names:
            channels.append((base_ai / float(self.base_dataset.lfm_scale)).astype(np.float32))
        if "mask" in channel_names:
            channels.append(item["mask"].squeeze(0).detach().cpu().numpy().astype(np.float32))
        if "dynamic_gain_log_ratio" in channel_names:
            dynamic_gain = item.get("dynamic_gain")
            if dynamic_gain is None:
                channels.append(np.zeros_like(channels[0], dtype=np.float32))
            else:
                gain = dynamic_gain.squeeze(0).detach().cpu().numpy().astype(np.float32)
                median = float(self.base_dataset.dynamic_gain_median or 1.0)
                gain_channel = np.log(np.maximum(gain, 1e-6) / median)
                channels.append(np.clip(gain_channel, -3.0, 3.0).astype(np.float32))
        return torch.from_numpy(np.stack(channels, axis=0)).float()

    def _augment_input_seismic(self, clean: np.ndarray, source_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        augmented = np.asarray(clean, dtype=np.float32).reshape(-1).copy()
        if not self.input_augmentation_enabled:
            return augmented

        if self.input_phase_deg_max > 0.0:
            angle = np.deg2rad(np.random.uniform(-self.input_phase_deg_max, self.input_phase_deg_max))
            augmented = _constant_phase_rotation(augmented, float(angle)).astype(np.float32, copy=False)

        if self.input_spectral_tilt_max > 0.0:
            tilt = float(np.random.uniform(-self.input_spectral_tilt_max, self.input_spectral_tilt_max))
            augmented = _apply_spectral_tilt(augmented, tilt).astype(np.float32, copy=False)

        amp_min, amp_max = self.input_amp_jitter
        if amp_max > 0.0 and (amp_min != 1.0 or amp_max != 1.0):
            augmented = (augmented * float(np.random.uniform(amp_min, amp_max))).astype(np.float32, copy=False)

        noise_min, noise_max = self.input_noise_rms_fraction
        if noise_max > 0.0:
            obs_rms = _masked_rms(np.asarray(source_obs, dtype=np.float32).reshape(-1), mask)
            signal_rms = _masked_rms(augmented, mask)
            reference_rms = obs_rms if obs_rms > 0.0 and np.isfinite(obs_rms) else signal_rms
            if reference_rms > 0.0 and np.isfinite(reference_rms):
                noise_fraction = float(np.random.uniform(noise_min, noise_max))
                noise = np.random.normal(0.0, reference_rms * noise_fraction, size=augmented.shape)
                augmented = (augmented + noise.astype(np.float32)).astype(np.float32, copy=False)

        return augmented

    def _passes_quality_gate(self, item: dict[str, torch.Tensor]) -> bool:
        waveform_mask = item["mask"].squeeze(0).detach().cpu().numpy().astype(bool, copy=False)
        delta_mask = item["delta_loss_mask"].squeeze(0).detach().cpu().numpy().astype(bool, copy=False)
        residual = item["target_delta_log_ai"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        target_seismic = item["target_seismic"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        base_seismic = item["base_seismic"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        source_seismic = item["obs"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        valid_delta = delta_mask & np.isfinite(residual)
        valid_waveform = (
            waveform_mask & np.isfinite(target_seismic) & np.isfinite(source_seismic) & np.isfinite(base_seismic)
        )
        if not np.any(valid_delta) or not np.any(valid_waveform):
            return False
        if bool(item.get("synthetic_empty_residual", torch.tensor(False)).item()):
            return False
        if not np.any(np.abs(residual[valid_delta]) > 1e-8):
            return False

        if self.max_residual_near_clip_fraction is not None:
            near_clip = float(np.mean(np.abs(residual[valid_delta]) >= 0.98 * self.residual_max_abs))
            if near_clip > float(self.max_residual_near_clip_fraction):
                return False

        if self.max_seismic_rms_ratio is not None:
            source_rms = _rms(source_seismic[valid_waveform])
            target_rms = _rms(target_seismic[valid_waveform])
            if source_rms > 0.0 and np.isfinite(source_rms):
                if target_rms / source_rms > float(self.max_seismic_rms_ratio):
                    return False

        if self.max_seismic_abs_p99_ratio is not None:
            source_p99 = _abs_percentile(source_seismic[valid_waveform], 99.0)
            target_p99 = _abs_percentile(target_seismic[valid_waveform], 99.0)
            if source_p99 > 0.0 and np.isfinite(source_p99):
                if target_p99 / source_p99 > float(self.max_seismic_abs_p99_ratio):
                    return False

        if self.min_target_obs_waveform_corr is not None:
            corr = _normalized_cross_correlation(target_seismic[valid_waveform], source_seismic[valid_waveform])
            if np.isfinite(corr) and corr < float(self.min_target_obs_waveform_corr):
                return False

        if self.min_base_target_waveform_corr is not None:
            corr = _normalized_cross_correlation(base_seismic[valid_waveform], target_seismic[valid_waveform])
            if np.isfinite(corr) and corr < float(self.min_base_target_waveform_corr):
                return False

        return True

    def _sample_unresolved_cluster_residual(
        self,
        highres_depth: np.ndarray,
        taper_hi: np.ndarray,
        support_hi: np.ndarray,
        factor: int,
    ) -> np.ndarray | None:
        highres = np.zeros_like(highres_depth, dtype=np.float32)
        n_highres = highres.size
        taper_1d = np.asarray(taper_hi, dtype=np.float32).reshape(-1)
        support = np.asarray(support_hi, dtype=bool).reshape(-1)
        active = taper_1d > 0.0
        if support.shape == active.shape and np.any(active & support):
            active = active & support
        active_runs = true_runs(active)
        if not active_runs:
            return None

        main_lobe = max(4, min(self.cluster_main_lobe_samples, max(1, n_highres // max(1, int(factor)))))
        patch_window = sample_highres_prior_patch(
            self.prior,
            self._well_rows,
            highres_depth,
            taper_1d,
            support,
            min_len_samples=max(main_lobe * 2, 16) * max(1, int(factor)),
            values_key="highres_well_high_log_ai",
        )
        if patch_window is None:
            return None

        placed, dst0, dst1, source_patch = patch_window
        length = int(dst1) - int(dst0)
        if length <= 0:
            return None

        source_patch = source_patch.astype(np.float32, copy=True)
        source_patch -= float(np.mean(source_patch))
        patch_scale = float(np.random.uniform(self.well_patch_scale_min, self.well_patch_scale_max))
        source_patch *= patch_scale

        hi_start = int(dst0)
        hi_stop = int(dst1)
        local = np.zeros((length,), dtype=np.float32)
        source_hi = source_patch

        background_scale = float(np.random.uniform(0.20, 0.45))
        local += background_scale * source_hi

        amp_min = self.cluster_amp_abs_p95_min * self.residual_abs_p95
        amp_max = self.cluster_amp_abs_p99_max * self.residual_abs_p99
        amp_hi = max(amp_min, amp_max)
        abs_patch = np.abs(source_hi)
        if np.any(np.isfinite(abs_patch)) and float(np.max(abs_patch)) > 0.0:
            envelope = centered_moving_average(abs_patch, max(3, (main_lobe * factor) // 2))
            env_max = float(np.max(envelope))
            if env_max > 0.0 and np.isfinite(env_max):
                envelope = envelope / env_max
            else:
                envelope = np.ones_like(abs_patch)
        else:
            envelope = np.ones_like(abs_patch)

        main_lobe_hi = max(4, main_lobe * max(1, int(factor)))
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
            source_amp = (
                float(np.percentile(abs_patch[max(0, packet_start) : packet_stop], 75))
                if packet_stop > packet_start
                else 0.0
            )
            if np.isfinite(source_amp) and source_amp > 0.0:
                amp_scale = 0.5 * amp_scale + 0.5 * source_amp

            packet = markov_thin_bed_packet(
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
        return highres.astype(np.float32, copy=False)

    def _forward_raw_seismic(
        self,
        ai: np.ndarray,
        vp: np.ndarray,
        dynamic_gain: torch.Tensor | None,
    ) -> np.ndarray:
        ai_tensor = torch.from_numpy(ai[np.newaxis, np.newaxis]).float()
        vp_tensor = torch.from_numpy(vp[np.newaxis, np.newaxis]).float()
        gain_tensor = (
            dynamic_gain.float().unsqueeze(0) if dynamic_gain is not None and dynamic_gain.ndim == 2 else dynamic_gain
        )
        with torch.no_grad():
            seismic = self.forward_model(ai_tensor, vp_tensor, gain=gain_tensor)
        return seismic.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _forward_target_seismic(
        self,
        target_ai: np.ndarray,
        target_vp: np.ndarray,
        dynamic_gain: torch.Tensor | None,
        waveform_qc_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        raw = self._forward_raw_seismic(target_ai, target_vp, dynamic_gain)
        scale = 1.0
        if self.seismic_rms_match and dynamic_gain is not None:
            valid = np.asarray(waveform_qc_mask, dtype=bool) & np.isfinite(raw)
            rms = float(np.sqrt(np.mean(raw[valid] ** 2))) if np.any(valid) else 0.0
            if rms > 0.0 and np.isfinite(rms):
                scale = float(
                    np.clip(self.seismic_rms_target / rms, self.seismic_rms_scale_min, self.seismic_rms_scale_max)
                )
        matched = (raw * scale).astype(np.float32)
        return matched, raw, scale

    def _forward_highres_target_seismic(
        self,
        highres_ai: np.ndarray,
        highres_vp: np.ndarray,
        highres_depth: np.ndarray,
        dynamic_gain: torch.Tensor | None,
        waveform_qc_mask: np.ndarray,
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
            valid = np.asarray(waveform_qc_mask, dtype=bool) & np.isfinite(raw)
            rms = float(np.sqrt(np.mean(raw[valid] ** 2))) if np.any(valid) else 0.0
            if rms > 0.0 and np.isfinite(rms):
                scale = float(
                    np.clip(self.seismic_rms_target / rms, self.seismic_rms_scale_min, self.seismic_rms_scale_max)
                )
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
        vp = centered_moving_average(vp.astype(np.float32), vp_smooth_samples)

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


def _masked_rms(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        return _rms(values)
    valid = mask & np.isfinite(values)
    if not np.any(valid):
        return 0.0
    return float(np.sqrt(np.mean(values[valid] * values[valid])))


def _constant_phase_rotation(values: np.ndarray, angle_rad: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 2 or abs(angle_rad) <= 0.0:
        return values.copy()
    quadrature = _hilbert_imag(values)
    rotated = values.astype(np.float64) * np.cos(angle_rad) - quadrature.astype(np.float64) * np.sin(angle_rad)
    original_rms = _rms(values)
    rotated_rms = _rms(rotated)
    if original_rms > 0.0 and rotated_rms > 0.0:
        rotated *= original_rms / rotated_rms
    return rotated.astype(np.float32, copy=False)


def _hilbert_imag(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    spectrum = np.fft.fft(values)
    multiplier = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        multiplier[0] = 1.0
        multiplier[n // 2] = 1.0
        multiplier[1 : n // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * multiplier)
    return np.imag(analytic).astype(np.float32, copy=False)


def _apply_spectral_tilt(values: np.ndarray, tilt: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 3 or abs(tilt) <= 0.0:
        return values.copy()
    original_rms = _rms(values)
    spectrum = np.fft.rfft(values.astype(np.float64))
    weights = np.linspace(1.0 - float(tilt), 1.0 + float(tilt), spectrum.size, dtype=np.float64)
    weights = np.clip(weights, 0.05, None)
    tilted = np.fft.irfft(spectrum * weights, n=values.size)
    tilted_rms = _rms(tilted)
    if original_rms > 0.0 and tilted_rms > 0.0:
        tilted *= original_rms / tilted_rms
    return tilted.astype(np.float32, copy=False)


def _abs_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.percentile(np.abs(values), percentile))


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan")
    a = a[valid] - float(np.mean(a[valid]))
    b = b[valid] - float(np.mean(b[valid]))
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.sum(a * b) / denom)


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
