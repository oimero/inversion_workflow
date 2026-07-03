"""Real-field zero-shot prediction and forward diagnostics for GINN-v2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.wavelet import (
    DEFAULT_ACTIVE_SUPPORT_THRESHOLD,
    compute_wavelet_active_half_support_s,
    infer_wavelet_dt,
)
from cup.synthetic.dataset import SynthoseisBenchmark
from cup.utils.io import (
    repo_relative_path,
    require_contract_fingerprint,
    resolve_artifact_path,
    resolve_relative_path,
    write_json,
)
from ginn_v2.data import denormalize_delta
from ginn_v2.training import load_checkpoint, resolve_device


@dataclass(frozen=True)
class RealFieldSection:
    seismic: np.ndarray
    lfm: np.ndarray
    valid_mask: np.ndarray
    ilines: np.ndarray
    xlines: np.ndarray
    twt_s: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RealFieldVolume:
    seismic: np.ndarray
    lfm: np.ndarray
    valid_mask: np.ndarray
    ilines: np.ndarray
    xlines: np.ndarray
    twt_s: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PatchGeometry:
    lateral_start: int
    lateral_stop: int
    twt_start: int
    twt_stop: int
    valid_fraction: float


def finite_summary_stats(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "robust_rms": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "abs_p95": float("nan"),
            "abs_p99": float("nan"),
        }
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "rms": float(np.sqrt(np.mean(data * data))),
        "robust_rms": float(1.4826 * mad),
        "p01": float(np.quantile(data, 0.01)),
        "p05": float(np.quantile(data, 0.05)),
        "p50": median,
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "abs_p95": float(np.quantile(np.abs(data), 0.95)),
        "abs_p99": float(np.quantile(np.abs(data), 0.99)),
    }


def load_model_manifest(model_cfg: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    model_run_dir = resolve_relative_path(_required_text(model_cfg, "model_run_dir"), root=root)
    manifest_path = model_run_dir / "model_run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model_run_manifest.json not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if str(manifest.get("schema_version") or "") != "ginn_v2_model_run_v3":
        raise ValueError(f"Unsupported model run manifest schema: {manifest_path}")
    require_contract_fingerprint(manifest, label=f"model run {model_run_dir}")
    return manifest


def _model_manifest_and_dir(model_cfg: Mapping[str, Any], *, root: Path) -> tuple[Path, dict[str, Any]]:
    extra = sorted(set(model_cfg) - {"model_run_dir"})
    if extra:
        raise ValueError(f"R0 model entries only accept model_run_dir; got {extra}.")
    model_run_dir = resolve_relative_path(_required_text(model_cfg, "model_run_dir"), root=root)
    manifest_path = model_run_dir / "model_run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model_run_manifest.json not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if str(manifest.get("schema_version") or "") != "ginn_v2_model_run_v3":
        raise ValueError(f"Unsupported model run manifest schema: {manifest_path}")
    require_contract_fingerprint(manifest, label=f"model run {model_run_dir}")
    gate = manifest.get("synthetic_gate_evidence")
    if not isinstance(gate, Mapping):
        raise ValueError(f"Model run manifest lacks synthetic_gate_evidence: {manifest_path}")
    required_gate = ("report_dir", "report_card", "is_current_frozen_candidate")
    missing_gate = [key for key in required_gate if key not in gate]
    if missing_gate:
        raise ValueError(f"synthetic_gate_evidence missing {missing_gate}: {manifest_path}")
    return model_run_dir, manifest


def _manifest_model_role(manifest: Mapping[str, Any]) -> str:
    explicit = str(manifest.get("model_role") or "").strip()
    if explicit:
        return explicit
    model_id = str(manifest.get("model_id") or "").strip()
    if "lateral_mixer" in model_id:
        return "lateral"
    if "trace_1d" in model_id or "trace1d" in model_id:
        return "no_lateral"
    if not model_id:
        raise ValueError("Model manifest must contain model_id or model_role.")
    return model_id.replace("-", "_")


def _primary_checkpoint_path(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> Path:
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("GINN-v2 v3 manifest lacks checkpoints mapping.")
    primary = str(checkpoints.get("primary") or "")
    if primary not in {"best", "final"}:
        raise ValueError(f"Invalid GINN-v2 primary checkpoint: {primary!r}")
    record = checkpoints.get(primary)
    if not isinstance(record, Mapping):
        raise ValueError(f"GINN-v2 manifest lacks primary checkpoint record: {primary}")
    path = resolve_relative_path(str(record.get("path") or ""), root=root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def synthetic_train_values(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    input_name: str = "seismic",
    max_patches: int = 4096,
    seed: int = 20260620,
) -> tuple[np.ndarray, dict[str, object]]:
    benchmark_dir = resolve_relative_path(_required_text(manifest, "benchmark_dir"), root=root)
    patch_index_path = resolve_relative_path(_required_text(manifest, "patch_index"), root=root)
    benchmark = SynthoseisBenchmark(benchmark_dir)
    patch_index = pd.read_csv(patch_index_path)
    train = patch_index[patch_index["split"].astype(str).eq("train")]
    total_train = int(train.shape[0])
    if max_patches > 0 and total_train > max_patches:
        train = train.sample(n=int(max_patches), random_state=int(seed)).sort_index()
    chunks: list[np.ndarray] = []
    for _, row in train.iterrows():
        sample = benchmark.load_sample(str(row["sample_id"]))
        target = np.asarray(sample.target_log_ai, dtype=np.float32)
        seismic = np.asarray(sample.seismic_input, dtype=np.float32)
        lfm = np.asarray(sample.priors["lfm_controlled_degraded"], dtype=np.float32)
        valid = np.asarray(sample.valid_mask, dtype=bool)
        if seismic.shape[1] == target.shape[1] - 1:
            target = target[:, 1:]
            lfm = lfm[:, 1:]
            valid = valid[:, 1:]
        elif seismic.shape[1] != target.shape[1]:
            raise ValueError(f"Unsupported seismic/target time shape for {sample.sample_id}.")
        effective_valid = valid & np.isfinite(target) & np.isfinite(seismic) & np.isfinite(lfm)
        lateral = slice(int(row["lateral_start"]), int(row["lateral_stop"]))
        twt = slice(int(row["twt_start"]), int(row["twt_stop"]))
        patch_valid = effective_valid[lateral, twt]
        if input_name == "seismic":
            values = seismic[lateral, twt]
        elif input_name == "lfm":
            values = lfm[lateral, twt]
        elif input_name == "target":
            values = target[lateral, twt]
        else:
            raise ValueError(f"Unsupported synthetic input: {input_name}")
        finite = np.asarray(values, dtype=np.float64)[patch_valid & np.isfinite(values)]
        if finite.size:
            chunks.append(finite)
    if not chunks:
        raise ValueError(f"No finite synthetic train values for {input_name}.")
    metadata = {
        "input_name": input_name,
        "total_train_patches": total_train,
        "sampled_train_patches": int(train.shape[0]),
        "sampling_seed": int(seed),
        "sampling_policy": (
            "deterministic_random_sample_without_replacement"
            if train.shape[0] < total_train
            else "all_train_patches"
        ),
    }
    return np.concatenate(chunks), metadata


def transform_real_seismic(
    seismic: np.ndarray,
    valid_mask: np.ndarray,
    *,
    transform: str,
    reference_stats: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply an auditable real-field seismic input transform.

    The transform is deterministic and diagnostic; it does not modify the model
    checkpoint or normalization.  Matching is done against synthetic train
    seismic summary statistics supplied by the caller.
    """

    mode = str(transform or "identity").strip()
    values = np.asarray(seismic, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    real = finite_summary_stats(values[valid])
    if mode in {"identity", "raw", "none"}:
        return np.asarray(values, dtype=np.float32), {
            "seismic_value_transform": "identity",
            "real_input_stats": real,
            "reference_stats": dict(reference_stats or {}),
            "scale": 1.0,
            "center": 0.0,
            "polarity": 1.0,
        }
    if reference_stats is None:
        raise ValueError(f"seismic_value_transform={mode} requires synthetic reference stats.")
    ref = dict(reference_stats)
    center = float(real["p50"])
    ref_mean = float(ref["mean"])
    polarity = -1.0 if mode.endswith("_polarity_flip") else 1.0
    base_mode = mode.removesuffix("_polarity_flip")
    if base_mode == "robust_rms_matched":
        numerator = float(ref["robust_rms"])
        denominator = float(real["robust_rms"])
    elif base_mode == "p95_abs_matched":
        numerator = float(ref["abs_p95"])
        denominator = float(real["abs_p95"])
    elif base_mode == "p99_abs_matched":
        numerator = float(ref["abs_p99"])
        denominator = float(real["abs_p99"])
    else:
        raise ValueError(f"Unsupported seismic_value_transform: {mode}")
    if denominator <= 0.0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        raise ValueError(f"Cannot apply seismic_value_transform={mode}: invalid scale denominator/reference.")
    scale = numerator / denominator
    transformed = polarity * (values - center) * scale + ref_mean
    metadata = {
        "seismic_value_transform": mode,
        "real_input_stats": real,
        "reference_stats": ref,
        "scale": float(scale),
        "center": center,
        "polarity": polarity,
    }
    return np.asarray(transformed, dtype=np.float32), metadata


def load_real_field_section(
    *,
    config: Mapping[str, Any],
    root: Path,
    data_root: Path,
) -> RealFieldSection:
    inputs = _mapping(config.get("real_field_inputs"), "real_field_zero_shot.real_field_inputs")
    if str(inputs.get("target_mask_file") or "").strip():
        raise ValueError("Unified LFM v2 does not accept an external target_mask_file.")
    section_cfg = _mapping(config.get("section"), "real_field_zero_shot.section")
    from cup.seismic.lfm.artifacts import resolve_lfm_variant

    selected_variant = resolve_lfm_variant(inputs, repo_root=root)
    lfm_path = selected_variant.lfm_path
    sample_domain = str(selected_variant.variant_metadata.get("sample_domain") or "")
    if sample_domain not in {"time", "depth"}:
        raise ValueError(f"Unsupported selected LFM sample_domain={sample_domain!r}.")
    lfm_transform = str(inputs.get("lfm_value_transform") or "identity").casefold()
    _validate_lfm_npz_contract(lfm_path, lfm_transform=lfm_transform, expected_variant_id=selected_variant.variant_id)
    lfm_values, ilines, xlines, twt_s = _load_npz_grid_section(
        lfm_path,
        section_cfg=section_cfg,
        value_keys=("log_ai", "lfm", "volume", "pred_log_ai"),
    )
    if lfm_transform == "log":
        if np.any(lfm_values <= 0.0):
            raise ValueError("lfm_value_transform=log requires positive LFM/AI values.")
        lfm_values = np.log(lfm_values)
    elif lfm_transform not in {"identity", "none"}:
        raise ValueError(f"Unsupported lfm_value_transform: {lfm_transform}")
    sample_start = float(twt_s[0])
    sample_end = float(twt_s[-1])
    _validate_section_axes_against_config(section_cfg, ilines, xlines)

    seismic_path = resolve_relative_path(_required_text(inputs, "seismic_file"), root=data_root)
    seismic_type = str(inputs.get("seismic_type", inputs.get("type", "zgy"))).casefold()
    if seismic_path.suffix.casefold() == ".npz":
        seismic, seismic_ilines, seismic_xlines, seismic_twt = _load_npz_grid_section(
            seismic_path,
            section_cfg=section_cfg,
            value_keys=("seismic", "volume", "seismic_input"),
        )
        _assert_axes_close("seismic ilines", seismic_ilines, ilines)
        _assert_axes_close("seismic xlines", seismic_xlines, xlines)
        seismic, lfm_values, twt_s = _align_time_arrays(seismic, seismic_twt, lfm_values, twt_s)
    else:
        seismic, seismic_twt = _load_survey_section(
            seismic_path,
            seismic_type=seismic_type,
            section_cfg=section_cfg,
            ilines=ilines,
            xlines=xlines,
            sample_start=sample_start,
            sample_end=sample_end,
            seismic_options=dict(inputs.get("segy_options") or {}),
            sample_domain=sample_domain,
        )
        seismic, lfm_values, twt_s = _align_time_arrays(seismic, seismic_twt, lfm_values, twt_s)

    mask_path_text = str(inputs.get("target_mask_file") or "").strip()
    if mask_path_text:
        mask_path = resolve_relative_path(mask_path_text, root=root)
        mask, mask_ilines, mask_xlines, mask_twt = _load_npz_grid_section(
            mask_path,
            section_cfg=section_cfg,
            value_keys=("valid_mask_model", "mask", "target_mask", "volume"),
        )
        _assert_axes_close("mask ilines", mask_ilines, ilines)
        _assert_axes_close("mask xlines", mask_xlines, xlines)
        mask, _, _ = _align_time_arrays(mask.astype(float), mask_twt, lfm_values, twt_s)
        valid_mask = mask > 0.5
    elif _npz_has_any(lfm_path, ("valid_mask_model", "mask", "target_mask")):
        mask, mask_ilines, mask_xlines, mask_twt = _load_npz_grid_section(
            lfm_path,
            section_cfg=section_cfg,
            value_keys=("valid_mask_model", "mask", "target_mask"),
        )
        _assert_axes_close("lfm mask ilines", mask_ilines, ilines)
        _assert_axes_close("lfm mask xlines", mask_xlines, xlines)
        mask, _, _ = _align_time_arrays(mask.astype(float), mask_twt, lfm_values, twt_s)
        valid_mask = mask > 0.5
    else:
        valid_mask = np.isfinite(lfm_values) & np.isfinite(seismic)

    valid_mask = valid_mask & np.isfinite(lfm_values) & np.isfinite(seismic)
    if seismic.shape != lfm_values.shape or valid_mask.shape != lfm_values.shape:
        raise ValueError(
            f"Real-field shape mismatch: seismic={seismic.shape}, lfm={lfm_values.shape}, "
            f"mask={valid_mask.shape}"
        )
    if not np.any(valid_mask):
        raise ValueError("Real-field section has no finite valid samples.")
    _validate_lfm_log_domain(lfm_values, valid_mask, lfm_transform=lfm_transform, path=lfm_path)
    seismic_transform = inputs.get("seismic_value_transform") or inputs.get("seismic_transform") or "identity"
    reference_stats = inputs.get("seismic_reference_stats")
    seismic, seismic_transform_metadata = transform_real_seismic(
        seismic,
        valid_mask,
        transform=str(seismic_transform),
        reference_stats=reference_stats if isinstance(reference_stats, Mapping) else None,
    )
    return RealFieldSection(
        seismic=np.asarray(seismic, dtype=np.float32),
        lfm=np.asarray(lfm_values, dtype=np.float32),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        ilines=np.asarray(ilines, dtype=np.float64),
        xlines=np.asarray(xlines, dtype=np.float64),
        twt_s=np.asarray(twt_s, dtype=np.float64),
        metadata={
            "lfm_file": str(lfm_path),
            "seismic_file": str(seismic_path),
            "seismic_type": seismic_type,
            "target_mask_file": mask_path_text,
            "lfm_value_transform": lfm_transform,
            "lfm_run_dir": str(selected_variant.run_dir),
            "variant_id": selected_variant.variant_id,
            "well_control_run_dir": str(selected_variant.well_control_run_dir),
            "lfm_contract_fingerprint_sha256": selected_variant.contract_fingerprint_sha256,
            "sample_domain": sample_domain,
            "sample_unit": selected_variant.variant_metadata.get("sample_unit"),
            "seismic_value_transform": str(seismic_transform),
            "seismic_reference_stats_file": str(inputs.get("seismic_reference_stats_file") or ""),
            "seismic_transform_metadata": seismic_transform_metadata,
            "section": dict(section_cfg),
        },
    )


def load_real_field_volume(
    *,
    config: Mapping[str, Any],
    root: Path,
    data_root: Path,
) -> RealFieldVolume:
    inputs = _mapping(config.get("real_field_inputs"), "real_field_zero_shot.real_field_inputs")
    if str(inputs.get("target_mask_file") or "").strip():
        raise ValueError("Unified LFM v2 does not accept an external target_mask_file.")
    volume_cfg = dict(config.get("volume") or {})
    from cup.seismic.lfm.artifacts import resolve_lfm_variant

    selected_variant = resolve_lfm_variant(inputs, repo_root=root)
    lfm_path = selected_variant.lfm_path
    sample_domain = str(selected_variant.variant_metadata.get("sample_domain") or "")
    if sample_domain not in {"time", "depth"}:
        raise ValueError(f"Unsupported selected LFM sample_domain={sample_domain!r}.")
    lfm_transform = str(inputs.get("lfm_value_transform") or "identity").casefold()
    _validate_lfm_npz_contract(lfm_path, lfm_transform=lfm_transform, expected_variant_id=selected_variant.variant_id)
    lfm_values, ilines, xlines, twt_s = _load_npz_grid_volume(
        lfm_path,
        volume_cfg=volume_cfg,
        value_keys=("log_ai", "lfm", "volume", "pred_log_ai"),
    )
    if lfm_transform == "log":
        if np.any(lfm_values <= 0.0):
            raise ValueError("lfm_value_transform=log requires positive LFM/AI values.")
        lfm_values = np.log(lfm_values)
    elif lfm_transform not in {"identity", "none"}:
        raise ValueError(f"Unsupported lfm_value_transform: {lfm_transform}")

    seismic_path = resolve_relative_path(_required_text(inputs, "seismic_file"), root=data_root)
    seismic_type = str(inputs.get("seismic_type", inputs.get("type", "zgy"))).casefold()
    if seismic_path.suffix.casefold() == ".npz":
        seismic, seismic_ilines, seismic_xlines, seismic_twt = _load_npz_grid_volume(
            seismic_path,
            volume_cfg=volume_cfg,
            value_keys=("seismic", "volume", "seismic_input"),
        )
        _assert_axes_close("seismic ilines", seismic_ilines, ilines)
        _assert_axes_close("seismic xlines", seismic_xlines, xlines)
        seismic, lfm_values, twt_s = _align_volume_time_arrays(seismic, seismic_twt, lfm_values, twt_s)
    else:
        seismic, seismic_twt = _load_survey_volume(
            seismic_path,
            seismic_type=seismic_type,
            ilines=ilines,
            xlines=xlines,
            sample_start=float(twt_s[0]),
            sample_end=float(twt_s[-1]),
            seismic_options=dict(inputs.get("segy_options") or {}),
            sample_domain=sample_domain,
        )
        seismic, lfm_values, twt_s = _align_volume_time_arrays(seismic, seismic_twt, lfm_values, twt_s)

    mask_path_text = str(inputs.get("target_mask_file") or "").strip()
    if mask_path_text:
        mask_path = resolve_relative_path(mask_path_text, root=root)
        mask, mask_ilines, mask_xlines, mask_twt = _load_npz_grid_volume(
            mask_path,
            volume_cfg=volume_cfg,
            value_keys=("valid_mask_model", "mask", "target_mask", "volume"),
        )
        _assert_axes_close("mask ilines", mask_ilines, ilines)
        _assert_axes_close("mask xlines", mask_xlines, xlines)
        mask, _, _ = _align_volume_time_arrays(mask.astype(float), mask_twt, lfm_values, twt_s)
        valid_mask = mask > 0.5
    elif _npz_has_any(lfm_path, ("valid_mask_model", "mask", "target_mask")):
        mask, mask_ilines, mask_xlines, mask_twt = _load_npz_grid_volume(
            lfm_path,
            volume_cfg=volume_cfg,
            value_keys=("valid_mask_model", "mask", "target_mask"),
        )
        _assert_axes_close("lfm mask ilines", mask_ilines, ilines)
        _assert_axes_close("lfm mask xlines", mask_xlines, xlines)
        mask, _, _ = _align_volume_time_arrays(mask.astype(float), mask_twt, lfm_values, twt_s)
        valid_mask = mask > 0.5
    else:
        valid_mask = np.isfinite(lfm_values) & np.isfinite(seismic)

    valid_mask = valid_mask & np.isfinite(lfm_values) & np.isfinite(seismic)
    if seismic.shape != lfm_values.shape or valid_mask.shape != lfm_values.shape:
        raise ValueError(
            f"Real-field volume shape mismatch: seismic={seismic.shape}, lfm={lfm_values.shape}, "
            f"mask={valid_mask.shape}"
        )
    if not np.any(valid_mask):
        raise ValueError("Real-field volume has no finite valid samples.")
    _validate_lfm_log_domain(lfm_values, valid_mask, lfm_transform=lfm_transform, path=lfm_path)
    seismic_transform = inputs.get("seismic_value_transform") or inputs.get("seismic_transform") or "identity"
    reference_stats = inputs.get("seismic_reference_stats")
    seismic, seismic_transform_metadata = transform_real_seismic(
        seismic,
        valid_mask,
        transform=str(seismic_transform),
        reference_stats=reference_stats if isinstance(reference_stats, Mapping) else None,
    )
    return RealFieldVolume(
        seismic=np.asarray(seismic, dtype=np.float32),
        lfm=np.asarray(lfm_values, dtype=np.float32),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        ilines=np.asarray(ilines, dtype=np.float64),
        xlines=np.asarray(xlines, dtype=np.float64),
        twt_s=np.asarray(twt_s, dtype=np.float64),
        metadata={
            "lfm_file": str(lfm_path),
            "seismic_file": str(seismic_path),
            "seismic_type": seismic_type,
            "target_mask_file": mask_path_text,
            "lfm_value_transform": lfm_transform,
            "lfm_run_dir": str(selected_variant.run_dir),
            "variant_id": selected_variant.variant_id,
            "well_control_run_dir": str(selected_variant.well_control_run_dir),
            "lfm_contract_fingerprint_sha256": selected_variant.contract_fingerprint_sha256,
            "sample_domain": sample_domain,
            "sample_unit": selected_variant.variant_metadata.get("sample_unit"),
            "seismic_value_transform": str(seismic_transform),
            "seismic_reference_stats_file": str(inputs.get("seismic_reference_stats_file") or ""),
            "seismic_transform_metadata": seismic_transform_metadata,
            "volume": volume_cfg,
        },
    )


def run_zero_shot_model(
    *,
    section: RealFieldSection,
    model_cfg: Mapping[str, Any],
    output_dir: Path,
    root: Path,
    device_name: str,
    stitch_strategy: str,
) -> dict[str, Any]:
    model_run_dir, manifest = _model_manifest_and_dir(model_cfg, root=root)
    model_role = _manifest_model_role(manifest)
    model_id_label = _required_text(manifest, "model_id")
    synthetic_gate_evidence = dict(manifest["synthetic_gate_evidence"])
    checkpoint_path = _primary_checkpoint_path(manifest, root=root)
    model, checkpoint = load_checkpoint(checkpoint_path)
    normalization_path = None
    normalization_ref = manifest.get("normalization_path") or manifest.get("normalization_file")
    if normalization_ref:
        normalization_path = _resolve_model_artifact(
            normalization_ref,
            root=root,
            model_run_dir=model_run_dir,
            fallback_name="normalization.json",
        )
        with normalization_path.open("r", encoding="utf-8") as handle:
            normalization_file_payload = json.load(handle)
        if normalization_file_payload != dict(checkpoint["normalization"]):
            raise ValueError(f"normalization_file does not match checkpoint normalization: {normalization_path}")
    device, device_metadata = resolve_device(device_name)
    model.to(device)
    model.eval()
    normalization = dict(checkpoint["normalization"])
    patch_spec = _mapping(manifest.get("patch_spec"), f"{model_role}.patch_spec")
    patches = build_real_field_patch_index(
        section.valid_mask,
        lateral_samples=int(patch_spec["lateral_samples"]),
        twt_samples=int(patch_spec["twt_samples"]),
        lateral_stride=int(patch_spec["lateral_stride"]),
        twt_stride=int(patch_spec["twt_stride"]),
        min_valid_fraction=float(patch_spec["min_valid_fraction"]),
    )
    if not patches:
        raise ValueError(f"No valid real-field patches for model role={model_role}.")

    pred_sum = np.zeros_like(section.lfm, dtype=np.float64)
    weight = np.zeros_like(section.lfm, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    patch_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for patch_idx, patch in enumerate(patches):
            sl = (slice(patch.lateral_start, patch.lateral_stop), slice(patch.twt_start, patch.twt_stop))
            seismic_patch = section.seismic[sl]
            lfm_patch = section.lfm[sl]
            valid_patch = section.valid_mask[sl]
            inputs = np.stack(
                [
                    _normalize_with_mask(seismic_patch, valid_patch, normalization["seismic"]),
                    _normalize_with_mask(lfm_patch, valid_patch, normalization["lfm"]),
                    valid_patch.astype(np.float32),
                ],
                axis=0,
            )[None, ...]
            tensor = torch.as_tensor(inputs, dtype=torch.float32, device=device)
            pred_delta_n = model(tensor).detach().cpu().numpy()[0, 0]
            pred_delta = denormalize_delta(pred_delta_n, normalization)
            prediction = np.asarray(lfm_patch + pred_delta, dtype=np.float32)
            patch_predictions.append(prediction)
            local, dest = _stitch_slices(patch, strategy=stitch_strategy)
            local_valid = valid_patch[local]
            pred_sum[dest] += np.where(local_valid, prediction[local], 0.0)
            weight[dest] += local_valid.astype(np.float64)
            rows.append(
                {
                    "patch_id": f"{model_role}__p{patch_idx:05d}",
                    "model_role": model_role,
                    "model_id": model_id_label,
                    "checkpoint_model_id": str(checkpoint["model_id"]),
                    "lateral_start": patch.lateral_start,
                    "lateral_stop": patch.lateral_stop,
                    "twt_start": patch.twt_start,
                    "twt_stop": patch.twt_stop,
                    "inline_start": float(section.ilines[patch.lateral_start]),
                    "inline_stop": float(section.ilines[patch.lateral_stop - 1]),
                    "xline_start": float(section.xlines[patch.lateral_start]),
                    "xline_stop": float(section.xlines[patch.lateral_stop - 1]),
                    "twt_start_s": float(section.twt_s[patch.twt_start]),
                    "twt_stop_s": float(section.twt_s[patch.twt_stop - 1]),
                    "valid_fraction": patch.valid_fraction,
                    "stitch_strategy": stitch_strategy,
                }
            )
    stitched = np.divide(pred_sum, weight, out=np.full_like(pred_sum, np.nan), where=weight > 0.0)
    pred_delta_vs_lfm = stitched - section.lfm
    model_dir = output_dir / model_role
    model_dir.mkdir(parents=True, exist_ok=False)
    npz_path = model_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        stitched_pred_log_ai=stitched.astype(np.float32),
        pred_delta_vs_lfm=pred_delta_vs_lfm.astype(np.float32),
        stitching_weight=weight.astype(np.float32),
        lfm_input=section.lfm.astype(np.float32),
        seismic_input=section.seismic.astype(np.float32),
        valid_mask_model=section.valid_mask.astype(bool),
        ilines=section.ilines,
        xlines=section.xlines,
        twt_s=section.twt_s,
        patch_pred_log_ai=np.asarray(patch_predictions, dtype=np.float32),
    )
    index_path = model_dir / "prediction_index.csv"
    pd.DataFrame.from_records(rows).to_csv(index_path, index=False)
    summary = {
        "schema_version": "real_field_zero_shot_model_v2",
        "status": "ok",
        "input_contracts": {
            "model_run": {
                "path": repo_relative_path(model_run_dir / "model_run_manifest.json", root=root),
                "contract_fingerprint_sha256": require_contract_fingerprint(
                    manifest, label=f"model run {model_run_dir}"
                ),
            }
        },
        "model_role": model_role,
        "model_id": model_id_label,
        "checkpoint_model_id": str(checkpoint["model_id"]),
        "model_run_dir": repo_relative_path(model_run_dir, root=root),
        "checkpoint": repo_relative_path(checkpoint_path, root=root),
        "normalization_file": (
            repo_relative_path(normalization_path, root=root) if normalization_path is not None else ""
        ),
        "synthetic_gate_evidence": synthetic_gate_evidence,
        "device": device_metadata,
        "normalization": normalization,
        "patch_spec": patch_spec,
        "stitch_strategy": stitch_strategy,
        "n_patches": int(len(patches)),
        "outputs": {
            "predictions": repo_relative_path(npz_path, root=root),
            "prediction_index": repo_relative_path(index_path, root=root),
        },
        "prediction_coverage_fraction": float(np.mean(weight > 0.0)),
    }
    write_json(model_dir / "real_field_zero_shot_model_summary.json", summary)
    return summary


def run_zero_shot_volume_model(
    *,
    volume: RealFieldVolume,
    model_cfg: Mapping[str, Any],
    output_dir: Path,
    root: Path,
    device_name: str,
    stitch_strategy: str,
) -> dict[str, Any]:
    model_run_dir, manifest = _model_manifest_and_dir(model_cfg, root=root)
    model_role = _manifest_model_role(manifest)
    model_id_label = _required_text(manifest, "model_id")
    synthetic_gate_evidence = dict(manifest["synthetic_gate_evidence"])
    checkpoint_path = _primary_checkpoint_path(manifest, root=root)
    model, checkpoint = load_checkpoint(checkpoint_path)
    normalization_path = None
    normalization_ref = manifest.get("normalization_path") or manifest.get("normalization_file")
    if normalization_ref:
        normalization_path = _resolve_model_artifact(
            normalization_ref,
            root=root,
            model_run_dir=model_run_dir,
            fallback_name="normalization.json",
        )
        with normalization_path.open("r", encoding="utf-8") as handle:
            normalization_file_payload = json.load(handle)
        if normalization_file_payload != dict(checkpoint["normalization"]):
            raise ValueError(f"normalization_file does not match checkpoint normalization: {normalization_path}")
    device, device_metadata = resolve_device(device_name)
    model.to(device)
    model.eval()
    normalization = dict(checkpoint["normalization"])
    patch_spec = _mapping(manifest.get("patch_spec"), f"{model_role}.patch_spec")

    pred_sum = np.zeros_like(volume.lfm, dtype=np.float64)
    weight = np.zeros_like(volume.lfm, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    n_patches = 0
    with torch.no_grad():
        for inline_idx, inline_no in enumerate(volume.ilines):
            section_valid = volume.valid_mask[inline_idx]
            if not np.any(section_valid):
                continue
            patches = build_real_field_patch_index(
                section_valid,
                lateral_samples=int(patch_spec["lateral_samples"]),
                twt_samples=int(patch_spec["twt_samples"]),
                lateral_stride=int(patch_spec["lateral_stride"]),
                twt_stride=int(patch_spec["twt_stride"]),
                min_valid_fraction=float(patch_spec["min_valid_fraction"]),
            )
            for patch in patches:
                sl = (
                    inline_idx,
                    slice(patch.lateral_start, patch.lateral_stop),
                    slice(patch.twt_start, patch.twt_stop),
                )
                seismic_patch = volume.seismic[sl]
                lfm_patch = volume.lfm[sl]
                valid_patch = volume.valid_mask[sl]
                inputs = np.stack(
                    [
                        _normalize_with_mask(seismic_patch, valid_patch, normalization["seismic"]),
                        _normalize_with_mask(lfm_patch, valid_patch, normalization["lfm"]),
                        valid_patch.astype(np.float32),
                    ],
                    axis=0,
                )[None, ...]
                tensor = torch.as_tensor(inputs, dtype=torch.float32, device=device)
                pred_delta_n = model(tensor).detach().cpu().numpy()[0, 0]
                pred_delta = denormalize_delta(pred_delta_n, normalization)
                prediction = np.asarray(lfm_patch + pred_delta, dtype=np.float32)
                local, dest_2d = _stitch_slices(patch, strategy=stitch_strategy)
                local_valid = valid_patch[local]
                dest = (inline_idx, dest_2d[0], dest_2d[1])
                pred_sum[dest] += np.where(local_valid, prediction[local], 0.0)
                weight[dest] += local_valid.astype(np.float64)
                rows.append(
                    {
                        "patch_id": f"{model_role}__il{inline_idx:04d}__p{n_patches:07d}",
                        "model_role": model_role,
                        "model_id": model_id_label,
                        "checkpoint_model_id": str(checkpoint["model_id"]),
                        "inline_index": int(inline_idx),
                        "inline": float(inline_no),
                        "xline_start_index": patch.lateral_start,
                        "xline_stop_index": patch.lateral_stop,
                        "twt_start": patch.twt_start,
                        "twt_stop": patch.twt_stop,
                        "xline_start": float(volume.xlines[patch.lateral_start]),
                        "xline_stop": float(volume.xlines[patch.lateral_stop - 1]),
                        "twt_start_s": float(volume.twt_s[patch.twt_start]),
                        "twt_stop_s": float(volume.twt_s[patch.twt_stop - 1]),
                        "valid_fraction": patch.valid_fraction,
                        "stitch_strategy": stitch_strategy,
                    }
                )
                n_patches += 1
    if n_patches <= 0:
        raise ValueError(f"No valid real-field volume patches for model role={model_role}.")
    stitched = np.divide(pred_sum, weight, out=np.full_like(pred_sum, np.nan), where=weight > 0.0)
    pred_delta_vs_lfm = stitched - volume.lfm
    model_dir = output_dir / model_role
    model_dir.mkdir(parents=True, exist_ok=False)
    npz_path = model_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        stitched_pred_log_ai=stitched.astype(np.float32),
        pred_delta_vs_lfm=pred_delta_vs_lfm.astype(np.float32),
        stitching_weight=weight.astype(np.float32),
        lfm_input=volume.lfm.astype(np.float32),
        seismic_input=volume.seismic.astype(np.float32),
        valid_mask_model=volume.valid_mask.astype(bool),
        ilines=volume.ilines,
        xlines=volume.xlines,
        twt_s=volume.twt_s,
        output_mode=np.asarray("volume"),
    )
    index_path = model_dir / "prediction_index.csv"
    pd.DataFrame.from_records(rows).to_csv(index_path, index=False)
    summary = {
        "schema_version": "real_field_zero_shot_model_v2",
        "status": "ok",
        "input_contracts": {
            "model_run": {
                "path": repo_relative_path(model_run_dir / "model_run_manifest.json", root=root),
                "contract_fingerprint_sha256": require_contract_fingerprint(
                    manifest, label=f"model run {model_run_dir}"
                ),
            }
        },
        "output_mode": "volume",
        "model_role": model_role,
        "model_id": model_id_label,
        "checkpoint_model_id": str(checkpoint["model_id"]),
        "model_run_dir": repo_relative_path(model_run_dir, root=root),
        "checkpoint": repo_relative_path(checkpoint_path, root=root),
        "normalization_file": (
            repo_relative_path(normalization_path, root=root) if normalization_path is not None else ""
        ),
        "synthetic_gate_evidence": synthetic_gate_evidence,
        "device": device_metadata,
        "normalization": normalization,
        "patch_spec": patch_spec,
        "stitch_strategy": stitch_strategy,
        "n_patches": int(n_patches),
        "n_inline_rows": int(volume.ilines.size),
        "outputs": {
            "predictions": repo_relative_path(npz_path, root=root),
            "prediction_index": repo_relative_path(index_path, root=root),
        },
        "prediction_coverage_fraction": float(np.mean(weight > 0.0)),
    }
    write_json(model_dir / "real_field_zero_shot_model_summary.json", summary)
    return summary


def build_real_field_patch_index(
    valid_mask: np.ndarray,
    *,
    lateral_samples: int,
    twt_samples: int,
    lateral_stride: int,
    twt_stride: int,
    min_valid_fraction: float,
) -> list[PatchGeometry]:
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("valid_mask must be 2D [lateral, twt].")
    lateral_starts = _window_starts(valid.shape[0], int(lateral_samples), int(lateral_stride))
    twt_starts = _window_starts(valid.shape[1], int(twt_samples), int(twt_stride))
    rows: list[PatchGeometry] = []
    for lateral_start in lateral_starts:
        for twt_start in twt_starts:
            patch_mask = valid[
                lateral_start : lateral_start + lateral_samples,
                twt_start : twt_start + twt_samples,
            ]
            fraction = float(np.mean(patch_mask))
            if fraction >= float(min_valid_fraction):
                rows.append(
                    PatchGeometry(
                        lateral_start=lateral_start,
                        lateral_stop=lateral_start + lateral_samples,
                        twt_start=twt_start,
                        twt_stop=twt_start + twt_samples,
                        valid_fraction=fraction,
                    )
                )
    return rows


def input_qc_frame(section: RealFieldSection, normalization: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name, values, stats_key in [
        ("seismic", section.seismic, "seismic"),
        ("lfm", section.lfm, "lfm"),
    ]:
        valid = section.valid_mask & np.isfinite(values)
        data = values[valid].astype(np.float64)
        row = {"input": name, "finite_fraction": float(np.mean(valid)), **_summary_stats(data)}
        if stats_key in normalization:
            normalized = (data - float(normalization[stats_key]["mean"])) / float(normalization[stats_key]["std"])
            row.update(
                {
                    "normalized_mean": float(np.mean(normalized)) if normalized.size else float("nan"),
                    "normalized_std": float(np.std(normalized)) if normalized.size else float("nan"),
                    "fraction_abs_normalized_gt_3": float(np.mean(np.abs(normalized) > 3.0)) if normalized.size else 0.0,
                    "fraction_abs_normalized_gt_5": float(np.mean(np.abs(normalized) > 5.0)) if normalized.size else 0.0,
                }
            )
        rows.append(row)
    rows.append(
        {
            "input": "valid_mask_model",
            "finite_fraction": 1.0,
            "mean": float(np.mean(section.valid_mask)),
            "rms": float(np.sqrt(np.mean(section.valid_mask.astype(float) ** 2))),
            "robust_rms": float("nan"),
            "p01": float("nan"),
            "p50": float("nan"),
            "p99": float("nan"),
        }
    )
    return pd.DataFrame.from_records(rows)


def load_selected_wavelet(
    wavelet_generation_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    csv_path = wavelet_generation_dir / "selected_wavelet.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"selected_wavelet.csv not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "amplitude" not in frame:
        raise ValueError(f"selected_wavelet.csv lacks amplitude column: {csv_path}")
    if "time_s" not in frame:
        raise ValueError(f"selected_wavelet.csv lacks time_s column: {csv_path}")
    time_s = frame["time_s"].to_numpy(dtype=np.float64)
    wavelet = frame["amplitude"].to_numpy(dtype=np.float64)
    dt_s = infer_wavelet_dt(time_s)
    active_threshold = DEFAULT_ACTIVE_SUPPORT_THRESHOLD
    active_half_support_s = compute_wavelet_active_half_support_s(
        time_s,
        wavelet,
        active_threshold=active_threshold,
    )
    return time_s, wavelet, {
        "selected_wavelet_csv": str(csv_path),
        "n_samples": int(wavelet.size),
        "dt_s": float(dt_s),
        "active_support_threshold": float(active_threshold),
        "active_half_support_s": float(active_half_support_s),
    }


def diagnostic_metrics(
    *,
    observed: np.ndarray,
    synthetic: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(observed) & np.isfinite(synthetic)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 8:
        return {"status": "insufficient_valid_samples", "n_valid": n_valid}
    obs = np.asarray(observed, dtype=np.float64)[valid]
    syn = np.asarray(synthetic, dtype=np.float64)[valid]
    syn_energy = float(np.dot(syn, syn))
    obs_rms = float(np.sqrt(np.mean(obs * obs)))
    syn_rms = float(np.sqrt(np.mean(syn * syn)))
    raw = _residual_stats(obs, syn)
    if syn_energy <= 0.0 or not np.isfinite(syn_energy):
        return {
            "status": "invalid_synthetic_energy",
            "n_valid": n_valid,
            "observed_rms": obs_rms,
            "synthetic_rms_before_scale": syn_rms,
            **raw,
        }
    scale = float(np.dot(obs, syn) / syn_energy)
    scale_status = "ok" if scale > 0.0 else "invalid_nonpositive_scale"
    scaled = _residual_stats(obs, scale * syn) if scale > 0.0 else {}
    syn_centered = syn - float(np.mean(syn))
    obs_centered = obs - float(np.mean(obs))
    centered_energy = float(np.dot(syn_centered, syn_centered))
    slope = float(np.dot(obs_centered, syn_centered) / centered_energy) if centered_energy > 0.0 else float("nan")
    intercept = float(np.mean(obs) - slope * np.mean(syn)) if np.isfinite(slope) else float("nan")
    slope_status = "ok" if np.isfinite(slope) and slope > 0.0 else "invalid_nonpositive_scale"
    scaled_intercept = _residual_stats(obs, slope * syn + intercept) if slope_status == "ok" else {}
    return {
        "status": "ok",
        "n_valid": n_valid,
        "observed_rms": obs_rms,
        "synthetic_rms_before_scale": syn_rms,
        "scale_positive": scale,
        "scale_status": scale_status,
        "intercept": intercept,
        "scale_intercept_status": slope_status,
        "residual_rms_raw": raw["rmse"],
        "residual_corr_raw": raw["corr"],
        "residual_rms_scaled": scaled.get("rmse", float("nan")),
        "residual_corr_scaled": scaled.get("corr", float("nan")),
        "residual_rms_scaled_intercept": scaled_intercept.get("rmse", float("nan")),
        "residual_corr_scaled_intercept": scaled_intercept.get("corr", float("nan")),
    }


def phase_shift_scan(
    *,
    observed: np.ndarray,
    synthetic: np.ndarray,
    valid_mask: np.ndarray,
    phase_degrees: Sequence[float],
    fractional_shift_samples: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for phase in phase_degrees:
        candidate = _constant_phase_rotate(synthetic, float(phase))
        row = diagnostic_metrics(observed=observed, synthetic=candidate, valid_mask=valid_mask)
        rows.append({"scan_type": "phase", "phase_deg": float(phase), "fractional_shift_samples": 0.0, **row})
    for shift in fractional_shift_samples:
        candidate = _fractional_shift(synthetic, float(shift))
        row = diagnostic_metrics(observed=observed, synthetic=candidate, valid_mask=valid_mask)
        rows.append({"scan_type": "fractional_shift", "phase_deg": 0.0, "fractional_shift_samples": float(shift), **row})
    return pd.DataFrame.from_records(rows)


def load_zero_shot_predictions(run_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        npz_path = child / "predictions.npz"
        summary_path = child / "real_field_zero_shot_model_summary.json"
        if not npz_path.is_file():
            continue
        arrays = np.load(npz_path, allow_pickle=True)
        summary = {}
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        role = str(summary.get("model_role") or child.name)
        out[role] = {"dir": child, "arrays": arrays, "summary": summary}
    if not out:
        raise FileNotFoundError(f"No R0 model predictions found under {run_dir}")
    return out


def _validate_lfm_npz_contract(
    path: Path,
    *,
    lfm_transform: str,
    expected_variant_id: str,
) -> None:
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        required = {"log_ai", "valid_mask_model", "ilines", "xlines", "samples", "metadata_json"}
        if files != required:
            raise ValueError(f"R0 primary LFM must use the minimal real_field_lfm_variant_v3 keys {sorted(required)}: {path}")
        if str(lfm_transform).casefold() not in {"identity", "none"}:
            raise ValueError("real_field_lfm_variant_v3/log_ai is already log(AI); use identity transform.")
        if "metadata_json" not in files:
            raise ValueError(f"R0 primary LFM lacks metadata_json: {path}")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if str(metadata.get("schema_version")) != "real_field_lfm_variant_v3":
        raise ValueError(f"Unsupported LFM schema_version={metadata.get('schema_version')!r}: {path}")
    if str(metadata.get("variant_id")) != str(expected_variant_id):
        raise ValueError(f"LFM variant_id mismatch: expected {expected_variant_id!r}, got {metadata.get('variant_id')!r}")
    if str(metadata.get("value_key")) != "log_ai":
        raise ValueError(f"LFM metadata value_key must be log_ai, got {metadata.get('value_key')!r}: {path}")
    if str(metadata.get("value_domain")) != "log(AI)":
        raise ValueError(f"LFM metadata value_domain must be log(AI), got {metadata.get('value_domain')!r}: {path}")
    if str(metadata.get("valid_mask_key")) != "valid_mask_model":
        raise ValueError(
            f"LFM metadata valid_mask_key must be valid_mask_model, got {metadata.get('valid_mask_key')!r}: {path}"
        )


def _validate_section_axes_against_config(
    section_cfg: Mapping[str, Any],
    ilines: np.ndarray,
    xlines: np.ndarray,
) -> None:
    if "path" not in section_cfg:
        return
    expected_ilines, expected_xlines = _section_line_path(section_cfg)
    if expected_ilines.shape != ilines.shape or expected_xlines.shape != xlines.shape:
        raise ValueError(
            "R0 section axis mismatch: LFM section axes do not match configured section.n_traces/path."
        )
    if not np.allclose(expected_ilines, ilines, rtol=0.0, atol=1e-6):
        raise ValueError("R0 section axis mismatch: LFM ilines do not match configured section.path.")
    if not np.allclose(expected_xlines, xlines, rtol=0.0, atol=1e-6):
        raise ValueError("R0 section axis mismatch: LFM xlines do not match configured section.path.")


def _load_npz_grid_section(
    path: Path,
    *,
    section_cfg: Mapping[str, Any],
    value_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    key = next((candidate for candidate in value_keys if candidate in data.files), None)
    if key is None:
        raise ValueError(f"{path} lacks any value key from {list(value_keys)}; found {data.files}")
    values = np.asarray(data[key])
    if "samples" in data.files:
        twt = np.asarray(data["samples"], dtype=np.float64)
    elif "twt_s" in data.files:
        twt = np.asarray(data["twt_s"], dtype=np.float64)
    else:
        raise ValueError(f"{path} lacks samples/twt_s axis.")
    if twt.ndim != 1:
        raise ValueError(f"{path} sample axis must be 1D.")
    if values.ndim == 2:
        lateral = values.shape[0]
        sample_slice, cropped_twt = _crop_twt_axis(twt, section_cfg)
        ilines, xlines = _npz_lateral_axes(data, path=path, lateral=lateral)
        return (
            np.asarray(values[:, sample_slice], dtype=np.float32),
            ilines,
            xlines,
            cropped_twt,
        )
    if values.ndim != 3:
        raise ValueError(f"{path} values must be 2D or 3D, got shape {values.shape}.")
    il_axis = np.asarray(data["ilines"], dtype=np.float64)
    xl_axis = np.asarray(data["xlines"], dtype=np.float64)
    ilines, xlines = _section_line_path(section_cfg)
    il_idx = _nearest_axis_indices(il_axis, ilines, axis_name="inline")
    xl_idx = _nearest_axis_indices(xl_axis, xlines, axis_name="xline")
    sample_slice, cropped_twt = _crop_twt_axis(twt, section_cfg)
    section = values[il_idx, xl_idx, :][:, sample_slice]
    return np.asarray(section, dtype=np.float32), ilines, xlines, cropped_twt


def _load_npz_grid_volume(
    path: Path,
    *,
    volume_cfg: Mapping[str, Any],
    value_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        key = next((candidate for candidate in value_keys if candidate in data.files), None)
        if key is None:
            raise ValueError(f"{path} lacks any value key from {list(value_keys)}; found {data.files}")
        values = np.asarray(data[key])
        if values.ndim != 3:
            raise ValueError(f"{path} volume values must be 3D [inline, xline, twt], got shape {values.shape}.")
        if "samples" in data.files:
            twt = np.asarray(data["samples"], dtype=np.float64)
        elif "twt_s" in data.files:
            twt = np.asarray(data["twt_s"], dtype=np.float64)
        else:
            raise ValueError(f"{path} lacks samples/twt_s axis.")
        if "ilines" not in data.files or "xlines" not in data.files:
            raise ValueError(f"{path} volume must provide ilines and xlines axes.")
        ilines = np.asarray(data["ilines"], dtype=np.float64).reshape(-1)
        xlines = np.asarray(data["xlines"], dtype=np.float64).reshape(-1)
    if values.shape[:2] != (ilines.size, xlines.size):
        raise ValueError(
            f"{path} volume axis mismatch: values={values.shape}, ilines={ilines.size}, xlines={xlines.size}."
        )
    il_slice = _axis_slice_from_cfg(ilines, volume_cfg, start_keys=("inline_start", "iline_start"), stop_keys=("inline_stop", "iline_stop"))
    xl_slice = _axis_slice_from_cfg(xlines, volume_cfg, start_keys=("xline_start",), stop_keys=("xline_stop",))
    sample_slice, cropped_twt = _crop_twt_axis(twt, volume_cfg)
    return (
        np.asarray(values[il_slice, xl_slice, sample_slice], dtype=np.float32),
        np.asarray(ilines[il_slice], dtype=np.float64),
        np.asarray(xlines[xl_slice], dtype=np.float64),
        cropped_twt,
    )


def _npz_has_any(path: Path, keys: Sequence[str]) -> bool:
    with np.load(path, allow_pickle=False) as data:
        return any(key in data.files for key in keys)


def _validate_lfm_log_domain(
    lfm_values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lfm_transform: str,
    path: Path,
) -> None:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(lfm_values)
    values = np.asarray(lfm_values, dtype=np.float64)[valid]
    if values.size == 0:
        raise ValueError(f"LFM has no finite valid log(AI) samples after transform: {path}")
    stats = finite_summary_stats(values)
    p01 = float(stats["p01"])
    p99 = float(stats["p99"])
    median = float(stats["p50"])
    if not (0.0 < median < 30.0 and -5.0 < p01 < 30.0 and 0.0 < p99 < 30.0):
        raise ValueError(
            "LFM values do not look like log(AI). "
            f"lfm_value_transform={lfm_transform!r}, p01={p01:.6g}, "
            f"median={median:.6g}, p99={p99:.6g}, path={path}. "
            "Unified LFM v3 requires real_field_lfm_variant_v3/log_ai with identity transform."
        )


def _npz_lateral_axes(data: np.lib.npyio.NpzFile, *, path: Path, lateral: int) -> tuple[np.ndarray, np.ndarray]:
    il_key = next((key for key in ("ilines", "inline", "section_ilines") if key in data.files), None)
    xl_key = next((key for key in ("xlines", "xline", "section_xlines") if key in data.files), None)
    if il_key is None and xl_key is None:
        axis = np.arange(lateral, dtype=np.float64)
        return axis, axis.copy()
    if il_key is None or xl_key is None:
        raise ValueError(f"{path} 2D section must provide both ilines and xlines axes when either is present.")
    ilines = np.asarray(data[il_key], dtype=np.float64).reshape(-1)
    xlines = np.asarray(data[xl_key], dtype=np.float64).reshape(-1)
    if ilines.size != lateral or xlines.size != lateral:
        raise ValueError(
            f"{path} 2D section axis length mismatch: values lateral={lateral}, "
            f"ilines={ilines.size}, xlines={xlines.size}."
        )
    return ilines, xlines


def _load_survey_section(
    path: Path,
    *,
    seismic_type: str,
    section_cfg: Mapping[str, Any],
    ilines: np.ndarray,
    xlines: np.ndarray,
    sample_start: float,
    sample_end: float,
    seismic_options: Mapping[str, Any],
    sample_domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    options = segy_options_from_config(dict(seismic_options)) if seismic_type == "segy" else None
    survey = open_survey(path, seismic_type, segy_options=options)
    indices = []
    for il, xl in zip(ilines, xlines):
        i_float, j_float = survey.line_geometry.line_to_index(float(il), float(xl))
        indices.append((int(round(i_float)), int(round(j_float))))
    traces = survey.read_traces_at_indices(
        indices,
        sample_start=float(sample_start),
        sample_end=float(sample_end),
        domain=sample_domain,
    )
    first_trace = traces[indices[0]]
    twt_s = np.asarray(first_trace.basis, dtype=np.float64)
    values = [np.asarray(traces[key].values, dtype=np.float32) for key in indices]
    arr = np.asarray(values, dtype=np.float32)
    expected = int(np.asarray(section_cfg.get("expected_twt_samples", arr.shape[1])).item())
    if "expected_twt_samples" in section_cfg and arr.shape[1] != expected:
        raise ValueError(f"Survey section sample count {arr.shape[1]} != expected {expected}.")
    return arr, twt_s


def _load_survey_volume(
    path: Path,
    *,
    seismic_type: str,
    ilines: np.ndarray,
    xlines: np.ndarray,
    sample_start: float,
    sample_end: float,
    seismic_options: Mapping[str, Any],
    sample_domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    options = segy_options_from_config(dict(seismic_options)) if seismic_type == "segy" else None
    survey = open_survey(path, seismic_type, segy_options=options)
    il_indices = _line_axis_indices(survey.line_geometry.inline_axis.values(), ilines, axis_name="inline")
    xl_indices = _line_axis_indices(survey.line_geometry.xline_axis.values(), xlines, axis_name="xline")
    rows: list[np.ndarray] = []
    twt_s: np.ndarray | None = None
    for il_idx in il_indices:
        trace_keys = [(int(il_idx), int(xl_idx)) for xl_idx in xl_indices]
        traces = survey.read_traces_at_indices(
            trace_keys,
            sample_start=float(sample_start),
            sample_end=float(sample_end),
            domain=sample_domain,
        )
        if not traces:
            raise ValueError(f"No survey traces read for inline index {il_idx}.")
        first_trace = traces[trace_keys[0]]
        row_twt = np.asarray(first_trace.basis, dtype=np.float64)
        if twt_s is None:
            twt_s = row_twt
        elif not np.allclose(twt_s, row_twt, rtol=0.0, atol=1e-8):
            raise ValueError("Survey volume row time axes are inconsistent.")
        rows.append(np.asarray([np.asarray(traces[key].values, dtype=np.float32) for key in trace_keys], dtype=np.float32))
    if twt_s is None:
        raise ValueError("No survey volume rows were read.")
    return np.asarray(rows, dtype=np.float32), twt_s


def _section_line_path(section_cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = section_cfg.get("path")
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError("section.path must contain at least two inline/xline points.")
    first = _mapping(path[0], "section.path[0]")
    last = _mapping(path[-1], "section.path[-1]")
    n_traces = int(section_cfg.get("n_traces") or _infer_trace_count(first, last))
    if n_traces <= 0:
        raise ValueError("section.n_traces must be positive.")
    ilines = np.linspace(float(first["inline"]), float(last["inline"]), n_traces)
    xlines = np.linspace(float(first["xline"]), float(last["xline"]), n_traces)
    return ilines, xlines


def _infer_trace_count(first: Mapping[str, Any], last: Mapping[str, Any]) -> int:
    return int(max(abs(float(last["inline"]) - float(first["inline"])), abs(float(last["xline"]) - float(first["xline"]))) + 1)


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray, *, axis_name: str) -> np.ndarray:
    axis_values = np.asarray(axis, dtype=np.float64)
    indices = np.searchsorted(axis_values, values)
    indices = np.clip(indices, 0, axis_values.size - 1)
    left = np.maximum(indices - 1, 0)
    choose_left = np.abs(axis_values[left] - values) < np.abs(axis_values[indices] - values)
    nearest = np.where(choose_left, left, indices)
    tolerance = float(np.median(np.abs(np.diff(axis_values)))) * 0.51 if axis_values.size > 1 else 1e-6
    if np.any(np.abs(axis_values[nearest] - values) > tolerance):
        raise ValueError(f"Requested {axis_name} values are not supported by the source grid.")
    return nearest.astype(np.int64)


def _line_axis_indices(axis: np.ndarray, values: np.ndarray, *, axis_name: str) -> np.ndarray:
    axis_values = np.asarray(axis, dtype=np.float64)
    requested = np.asarray(values, dtype=np.float64)
    nearest = _nearest_axis_indices(axis_values, requested, axis_name=axis_name)
    return nearest.astype(np.int64)


def _axis_slice_from_cfg(
    axis: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    start_keys: Sequence[str],
    stop_keys: Sequence[str],
) -> slice:
    values = np.asarray(axis, dtype=np.float64)
    start = next((cfg[key] for key in start_keys if key in cfg and cfg[key] is not None), None)
    stop = next((cfg[key] for key in stop_keys if key in cfg and cfg[key] is not None), None)
    if start is None and stop is None:
        return slice(None)
    lo = float(values[0] if start is None else start)
    hi = float(values[-1] if stop is None else stop)
    if lo > hi:
        lo, hi = hi, lo
    i0 = int(np.searchsorted(values, lo, side="left"))
    i1 = int(np.searchsorted(values, hi, side="right"))
    i0 = max(0, i0)
    i1 = min(values.size, i1)
    if i0 >= i1:
        raise ValueError(f"Selected axis window is empty: start={start}, stop={stop}.")
    return slice(i0, i1)


def _crop_twt_axis(twt: np.ndarray, section_cfg: Mapping[str, Any]) -> tuple[slice, np.ndarray]:
    start = section_cfg.get("sample_start_s")
    end = section_cfg.get("sample_end_s")
    i0 = 0 if start is None else int(np.searchsorted(twt, float(start), side="left"))
    i1 = twt.size if end is None else int(np.searchsorted(twt, float(end), side="right"))
    i0 = max(0, i0)
    i1 = min(twt.size, i1)
    if i0 >= i1:
        raise ValueError("Selected TWT window is empty.")
    return slice(i0, i1), np.asarray(twt[i0:i1], dtype=np.float64)


def _align_time_arrays(
    first: np.ndarray,
    first_twt: np.ndarray,
    second: np.ndarray,
    second_twt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if first.shape[1] == second.shape[1] and np.allclose(first_twt, second_twt, rtol=0.0, atol=1e-8):
        return first, second, first_twt
    dt_first = float(np.median(np.diff(first_twt)))
    dt_second = float(np.median(np.diff(second_twt)))
    if not np.isclose(dt_first, dt_second, rtol=0.0, atol=1e-8):
        raise ValueError(f"Cannot align time axes with different dt: {dt_first} vs {dt_second}")
    start = max(float(first_twt[0]), float(second_twt[0]))
    end = min(float(first_twt[-1]), float(second_twt[-1]))
    if start > end:
        raise ValueError("Time axes do not overlap.")
    dt = dt_first
    n = int(np.floor((end - start) / dt + 0.5)) + 1
    if n <= 0:
        raise ValueError("Time axes have no common samples.")
    common = start + np.arange(n, dtype=np.float64) * dt
    first_idx = _nearest_time_indices(first_twt, common, dt=dt, name="first_twt")
    second_idx = _nearest_time_indices(second_twt, common, dt=dt, name="second_twt")
    return first[:, first_idx], second[:, second_idx], common


def _align_volume_time_arrays(
    first: np.ndarray,
    first_twt: np.ndarray,
    second: np.ndarray,
    second_twt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if first.shape[2] == second.shape[2] and np.allclose(first_twt, second_twt, rtol=0.0, atol=1e-8):
        return first, second, first_twt
    dt_first = float(np.median(np.diff(first_twt)))
    dt_second = float(np.median(np.diff(second_twt)))
    if not np.isclose(dt_first, dt_second, rtol=0.0, atol=1e-8):
        raise ValueError(f"Cannot align volume time axes with different dt: {dt_first} vs {dt_second}")
    start = max(float(first_twt[0]), float(second_twt[0]))
    end = min(float(first_twt[-1]), float(second_twt[-1]))
    if start > end:
        raise ValueError("Volume time axes do not overlap.")
    dt = dt_first
    n = int(np.floor((end - start) / dt + 0.5)) + 1
    if n <= 0:
        raise ValueError("Volume time axes have no common samples.")
    common = start + np.arange(n, dtype=np.float64) * dt
    first_idx = _nearest_time_indices(first_twt, common, dt=dt, name="first_twt")
    second_idx = _nearest_time_indices(second_twt, common, dt=dt, name="second_twt")
    return first[:, :, first_idx], second[:, :, second_idx], common


def _nearest_time_indices(axis: np.ndarray, values: np.ndarray, *, dt: float, name: str) -> np.ndarray:
    indices = np.searchsorted(axis, values)
    indices = np.clip(indices, 0, axis.size - 1)
    left = np.maximum(indices - 1, 0)
    choose_left = np.abs(axis[left] - values) < np.abs(axis[indices] - values)
    nearest = np.where(choose_left, left, indices)
    delta = np.abs(axis[nearest] - values)
    if np.any(delta > max(abs(dt) * 0.25, 1e-8)):
        raise ValueError(f"{name} cannot be aligned to common TWT axis.")
    return nearest.astype(np.int64)


def _assert_axes_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} mismatch.")


def _resolve_model_artifact(value: Any, *, root: Path, model_run_dir: Path, fallback_name: str) -> Path:
    resolved = resolve_artifact_path(value, root=root, run_dir=model_run_dir) if value else None
    if resolved is not None and resolved.is_file():
        return resolved
    fallback = model_run_dir / fallback_name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Cannot resolve model artifact {value!r}; fallback missing: {fallback}")


def _normalize_with_mask(values: np.ndarray, mask: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    out = (np.asarray(values, dtype=np.float32) - float(stats["mean"])) / float(stats["std"])
    return np.where(mask & np.isfinite(out), out, 0.0).astype(np.float32)


def _window_starts(size: int, window: int, stride: int) -> list[int]:
    if window <= 0 or stride <= 0:
        raise ValueError("Patch window and stride must be positive.")
    if size < window:
        raise ValueError(f"Section size {size} is smaller than patch window {window}.")
    starts = list(range(0, size - window + 1, stride))
    last = size - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _stitch_slices(patch: PatchGeometry, *, strategy: str) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    lateral = patch.lateral_stop - patch.lateral_start
    twt = patch.twt_stop - patch.twt_start
    if strategy == "uniform":
        l0, l1 = 0, lateral
        t0, t1 = 0, twt
    elif strategy == "center_crop":
        l0, l1 = _center_crop_bounds(lateral)
        t0, t1 = _center_crop_bounds(twt)
    else:
        raise ValueError(f"Unsupported stitch strategy: {strategy}")
    return (
        (slice(l0, l1), slice(t0, t1)),
        (slice(patch.lateral_start + l0, patch.lateral_start + l1), slice(patch.twt_start + t0, patch.twt_start + t1)),
    )


def _center_crop_bounds(size: int) -> tuple[int, int]:
    if size <= 4:
        return 0, size
    margin = max(1, size // 4)
    return margin, size - margin


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    stats = finite_summary_stats(values)
    return {
        "mean": stats["mean"],
        "rms": stats["rms"],
        "robust_rms": stats["robust_rms"],
        "p01": stats["p01"],
        "p50": stats["p50"],
        "p99": stats["p99"],
    }


def _residual_stats(observed: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    residual = observed - synthetic
    rmse = float(np.sqrt(np.mean(residual * residual)))
    if observed.size < 2 or np.std(observed) <= 0.0 or np.std(synthetic) <= 0.0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(observed, synthetic)[0, 1])
    return {"rmse": rmse, "corr": corr}


def _constant_phase_rotate(values: np.ndarray, phase_deg: float) -> np.ndarray:
    analytic = _analytic_signal(values)
    radians = np.deg2rad(float(phase_deg))
    return np.real(analytic * np.exp(1j * radians))


def _analytic_signal(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[-1]
    spectrum = np.fft.fft(x, axis=-1)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h, axis=-1)


def _fractional_shift(values: np.ndarray, shift_samples: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    original_shape = x.shape
    flat = x.reshape((-1, x.shape[-1]))
    coords = np.arange(x.shape[-1], dtype=np.float64)
    shifted_coords = coords - float(shift_samples)
    out = np.empty_like(flat)
    for idx in range(flat.shape[0]):
        out[idx] = np.interp(shifted_coords, coords, flat[idx], left=np.nan, right=np.nan)
    return out.reshape(original_shape)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _required_text(config: Mapping[str, Any], key: str) -> str:
    text = "" if config.get(key) is None else str(config.get(key)).strip()
    if not text:
        raise ValueError(f"Missing required config value: {key}")
    return text
