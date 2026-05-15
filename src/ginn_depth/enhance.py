"""Depth-domain adapter for stage-2 resolution enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from enhance.config import EnhancementConfig
from ginn_depth.config import DepthGINNConfig
from ginn_depth.data import DatasetBundle, DepthSeismicTraceDataset, build_dataset, load_lfm_depth_npz
from ginn_depth.physics import DepthForwardModel
from ginn_depth.synthetic import WellGuidedSyntheticDepthTraceDataset


@dataclass
class DepthEnhancementBundle:
    """Depth adapter outputs for enhancement training and inference."""

    depth_cfg: DepthGINNConfig
    dataset_bundle: DatasetBundle
    synthetic_dataset: WellGuidedSyntheticDepthTraceDataset
    metadata: dict[str, Any]


@dataclass
class DepthEnhancementDataBundle:
    """Depth datasets using stage-1 base AI as the enhancement base."""

    depth_cfg: DepthGINNConfig
    dataset_bundle: DatasetBundle
    metadata: dict[str, Any]


def build_depth_enhancement_data_bundle(cfg: EnhancementConfig) -> DepthEnhancementDataBundle:
    """Build depth datasets with ``base_ai_file`` wired through the existing AI-base slot."""
    repo_root = Path(__file__).resolve().parents[2]
    depth_cfg = DepthGINNConfig.from_yaml(cfg.depth_config_file, base_dir=repo_root)
    depth_cfg.include_lfm_input = cfg.include_base_ai_input
    depth_cfg.include_mask_input = cfg.include_mask_input
    depth_cfg.include_dynamic_gain_input = cfg.include_dynamic_gain_input
    depth_cfg.in_channels = cfg.in_channels

    dataset_bundle = build_dataset(depth_cfg)
    _replace_dataset_bundle_base_ai(dataset_bundle, cfg.base_ai_file)
    metadata = {
        "domain": "depth",
        "depth_config_file": cfg.depth_config_file,
        "base_ai_file": cfg.base_ai_file,
        "geometry": dataset_bundle.geometry,
        "split_metadata": dataset_bundle.split_metadata,
        "input_channel_names": dataset_bundle.train_dataset.input_channel_names,
    }
    return DepthEnhancementDataBundle(depth_cfg=depth_cfg, dataset_bundle=dataset_bundle, metadata=metadata)


def _replace_dataset_bundle_base_ai(dataset_bundle: DatasetBundle, base_ai_file: str | Path) -> None:
    """Use stage-1 base AI as the enhancement base while preserving depth mask metadata."""
    base_ai = load_lfm_depth_npz(base_ai_file)
    expected_shape = (
        int(dataset_bundle.geometry["n_il"]),
        int(dataset_bundle.geometry["n_xl"]),
        int(dataset_bundle.geometry["n_sample"]),
    )
    if base_ai.shape != expected_shape:
        raise ValueError(f"Base AI shape {base_ai.shape} does not match depth dataset shape {expected_shape}.")
    if not np.allclose(base_ai.samples, dataset_bundle.depth_axis_m):
        raise ValueError("Base AI depth samples do not match the depth dataset samples.")

    base_flat = np.asarray(base_ai.volume, dtype=np.float32).reshape(-1, expected_shape[-1])
    for dataset in (dataset_bundle.train_dataset, dataset_bundle.val_dataset, dataset_bundle.inference_dataset):
        if dataset is not None:
            _replace_dataset_base_ai(dataset, base_flat)


def _replace_dataset_base_ai(dataset: DepthSeismicTraceDataset, base_flat: np.ndarray) -> None:
    if base_flat.shape != dataset.ai_lfm_flat.shape:
        raise ValueError(
            f"Base AI flat shape {base_flat.shape} does not match dataset AI shape {dataset.ai_lfm_flat.shape}."
        )
    dataset._ai_lfm_flat = base_flat  # type: ignore[attr-defined]
    selected_mask = dataset._mask_flat[dataset.valid_indices]  # type: ignore[attr-defined]
    selected_base_ai = base_flat[dataset.valid_indices]
    valid_base_ai = selected_base_ai[selected_mask]
    dataset._lfm_scale = float(np.abs(valid_base_ai).max()) + 1e-10  # type: ignore[attr-defined]


def build_depth_enhancement_bundle(cfg: EnhancementConfig) -> DepthEnhancementBundle:
    """Build a depth synthetic dataset for stage-2 enhancement training."""
    data_bundle = build_depth_enhancement_data_bundle(cfg)
    depth_cfg = data_bundle.depth_cfg
    dataset_bundle = data_bundle.dataset_bundle
    forward_model = DepthForwardModel(
        dataset_bundle.wavelet_time_s,
        dataset_bundle.wavelet_amp,
        depth_axis_m=dataset_bundle.depth_axis_m,
        amplitude_threshold=depth_cfg.wavelet_amplitude_threshold,
    )
    synthetic_dataset = WellGuidedSyntheticDepthTraceDataset(
        dataset_bundle.train_dataset, # type: ignore
        cfg.resolution_prior_file,
        forward_model,
        num_examples=cfg.synthetic_traces_per_epoch,
        ai_min=cfg.ai_min,
        ai_max=cfg.ai_max,
        well_patch_scale_min=cfg.synthetic_well_patch_scale_min,
        well_patch_scale_max=cfg.synthetic_well_patch_scale_max,
        cluster_min_events=cfg.synthetic_cluster_min_events,
        cluster_max_events=cfg.synthetic_cluster_max_events,
        cluster_amp_abs_p95_min=cfg.synthetic_cluster_amp_abs_p95_min,
        cluster_amp_abs_p99_max=cfg.synthetic_cluster_amp_abs_p99_max,
        cluster_main_lobe_samples=cfg.synthetic_cluster_main_lobe_samples,
        unresolved_oversample_factor=cfg.synthetic_unresolved_oversample_factor,
        seismic_rms_match=cfg.synthetic_seismic_rms_match,
        seismic_rms_target=cfg.synthetic_seismic_rms_target,
        quality_gate_enabled=cfg.synthetic_quality_gate_enabled,
        max_residual_near_clip_fraction=cfg.synthetic_max_residual_near_clip_fraction,
        max_seismic_rms_ratio=cfg.synthetic_max_seismic_rms_ratio,
        max_seismic_abs_p99_ratio=cfg.synthetic_max_seismic_abs_p99_ratio,
        min_target_obs_waveform_corr=cfg.synthetic_min_target_obs_waveform_corr,
        min_base_target_waveform_corr=cfg.synthetic_min_base_target_waveform_corr,
        input_augmentation_enabled=cfg.synthetic_input_augmentation_enabled,
        input_phase_deg_max=cfg.synthetic_input_phase_deg_max,
        input_amp_jitter=cfg.synthetic_input_amp_jitter,
        input_noise_rms_fraction=cfg.synthetic_input_noise_rms_fraction,
        input_spectral_tilt_max=cfg.synthetic_input_spectral_tilt_max,
        max_resample_attempts=cfg.synthetic_max_resample_attempts,
        delta_supervision_mask=cfg.delta_supervision_mask,
    )
    metadata = dict(data_bundle.metadata)
    metadata["resolution_prior_file"] = cfg.resolution_prior_file
    return DepthEnhancementBundle(
        depth_cfg=depth_cfg,
        dataset_bundle=dataset_bundle,
        synthetic_dataset=synthetic_dataset,
        metadata=metadata,
    )


def load_enhancement_model(checkpoint_path: str, cfg: EnhancementConfig) -> torch.nn.Module:
    """Load an enhancement model from a checkpoint."""
    from enhance.model import DilatedResNet1D

    model = DilatedResNet1D(
        in_channels=cfg.in_channels,
        hidden_channels=cfg.hidden_channels,
        out_channels=cfg.out_channels,
        dilations=cfg.dilations,
        kernel_size=cfg.kernel_size,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model
