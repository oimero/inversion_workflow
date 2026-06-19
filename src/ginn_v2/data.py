"""Patch data access for the model-ablation gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cup.synthetic.dataset import SynthoseisBenchmark


SPLIT_VALUES = {"train", "validation", "test", "benchmark"}


@dataclass(frozen=True)
class PatchSpec:
    lateral_samples: int = 32
    twt_samples: int = 128
    lateral_stride: int = 16
    twt_stride: int = 64
    min_valid_fraction: float = 0.50


def default_train_kinds(model_id: str) -> set[str]:
    if model_id in {
        "trace_1d_tcn_lateral_mixer_mismatch_training",
        "trace_1d_dilated_tcn_mismatch_training",
        "trace_1d_mismatch_training",
        "patch_2d_mismatch_training",
    }:
        return {"base", "seismic_variant"}
    if model_id in {
        "trace_1d",
        "trace_1d_dilated_tcn",
        "patch_2d_supervised",
        "patch_2d_with_physics_loss",
    }:
        return {"base"}
    raise ValueError(f"Unsupported model_id: {model_id}")


def default_eval_kinds() -> set[str]:
    return {"base", "frequency_probe", "seismic_variant", "frequency_probe_seismic_variant"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _parent_id(row: Mapping[str, Any]) -> str:
    for key in ("parent_realization_id", "source_sample_id", "realization_id", "sample_id"):
        value = _clean_text(row.get(key))
        if value:
            return value
    raise ValueError("Cannot derive parent realization id from sample_index row.")


def _derive_split(parent: str, *, validation_fraction: float, test_fraction: float) -> str:
    digest = hashlib.sha256(parent.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64)
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def _row_split(
    row: Mapping[str, Any],
    *,
    split_policy: str,
    validation_fraction: float,
    test_fraction: float,
) -> str:
    text = _clean_text(row.get("split")).lower()
    if text in SPLIT_VALUES:
        return text
    if split_policy == "strict":
        raise ValueError(
            f"Sample {row.get('sample_id')} has non-training split {text!r}; "
            "use --split-policy derive to create a research split."
        )
    if split_policy != "derive":
        raise ValueError(f"Unsupported split_policy: {split_policy}")
    return _derive_split(
        _parent_id(row),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def _aligned_arrays(sample: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    target = np.asarray(sample.target_log_ai, dtype=np.float32)
    seismic = np.asarray(sample.seismic_input, dtype=np.float32)
    lfm = np.asarray(sample.priors["lfm_controlled_degraded"], dtype=np.float32)
    valid = np.asarray(sample.valid_mask, dtype=bool)
    if target.ndim != 2 or seismic.ndim != 2 or lfm.ndim != 2 or valid.ndim != 2:
        raise ValueError(f"Expected 2D arrays for sample {sample.sample_id}.")
    if target.shape[0] != seismic.shape[0] or lfm.shape != target.shape or valid.shape != target.shape:
        raise ValueError(
            f"Shape mismatch for {sample.sample_id}: target={target.shape}, "
            f"seismic={seismic.shape}, lfm={lfm.shape}, valid={valid.shape}"
        )
    raw_twt_offset_samples = 0
    if seismic.shape[1] == target.shape[1] - 1:
        target = target[:, 1:]
        lfm = lfm[:, 1:]
        valid = valid[:, 1:]
        raw_twt_offset_samples = 1
    elif seismic.shape[1] != target.shape[1]:
        raise ValueError(
            f"Unsupported seismic/target time shape for {sample.sample_id}: "
            f"{seismic.shape[1]} vs {target.shape[1]}"
        )
    effective_valid = valid & np.isfinite(target) & np.isfinite(seismic) & np.isfinite(lfm)
    return target, seismic, lfm, effective_valid, raw_twt_offset_samples


def _window_starts(size: int, window: int, stride: int) -> list[int]:
    if window <= 0 or stride <= 0:
        raise ValueError("Patch window and stride must be positive.")
    if size < window:
        return []
    starts = list(range(0, size - window + 1, stride))
    last = size - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_patch_index(
    benchmark: SynthoseisBenchmark,
    *,
    patch_spec: PatchSpec,
    sample_kinds: set[str],
    split_policy: str = "derive",
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    max_patches: int | None = None,
) -> pd.DataFrame:
    probe_kinds = {"frequency_probe", "frequency_probe_seismic_variant"}
    if max_patches is not None and sample_kinds.intersection(probe_kinds):
        raise ValueError("max_patches is smoke-only and cannot be used when probe sample kinds are indexed.")
    rows: list[dict[str, Any]] = []
    sample_ids = benchmark.sample_ids(kinds=sample_kinds, status="ok")
    for sample_id in sample_ids:
        sample = benchmark.load_sample(sample_id)
        target, _, _, valid, raw_twt_offset_samples = _aligned_arrays(sample)
        n_lateral, n_twt = target.shape
        lateral_starts = _window_starts(n_lateral, patch_spec.lateral_samples, patch_spec.lateral_stride)
        twt_starts = _window_starts(n_twt, patch_spec.twt_samples, patch_spec.twt_stride)
        if not lateral_starts or not twt_starts:
            continue
        row = sample.row
        split = _row_split(
            row,
            split_policy=split_policy,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        parent = _parent_id(row)
        for lateral_start in lateral_starts:
            for twt_start in twt_starts:
                patch_mask = valid[
                    lateral_start : lateral_start + patch_spec.lateral_samples,
                    twt_start : twt_start + patch_spec.twt_samples,
                ]
                valid_fraction = float(np.mean(patch_mask))
                if valid_fraction < patch_spec.min_valid_fraction:
                    continue
                patch_id = (
                    f"{sample_id}__l{lateral_start:04d}_{lateral_start + patch_spec.lateral_samples:04d}"
                    f"__t{twt_start:04d}_{twt_start + patch_spec.twt_samples:04d}"
                )
                rows.append(
                    {
                        "patch_id": patch_id,
                        "sample_id": sample_id,
                        "source_sample_id": row.get("source_sample_id", ""),
                        "sample_kind": row.get("sample_kind", "base"),
                        "parent_realization_id": parent,
                        "split": split,
                        "hdf5_group": row.get("hdf5_group", ""),
                        "lateral_start": lateral_start,
                        "lateral_stop": lateral_start + patch_spec.lateral_samples,
                        "twt_start": twt_start,
                        "twt_stop": twt_start + patch_spec.twt_samples,
                        "model_twt_start": twt_start,
                        "model_twt_stop": twt_start + patch_spec.twt_samples,
                        "raw_twt_offset_samples": raw_twt_offset_samples,
                        "time_alignment_mode": (
                            "target_lfm_mask_drop_first_sample_when_seismic_uses_forward_axis"
                            if raw_twt_offset_samples
                            else "native_model_grid"
                        ),
                        "patch_lateral_samples": patch_spec.lateral_samples,
                        "patch_twt_samples": patch_spec.twt_samples,
                        "valid_fraction": valid_fraction,
                        "paired_zero_sample_id": row.get("paired_zero_sample_id", ""),
                        "probe_group_id": row.get("probe_group_id", ""),
                        "probe_frequency_hz": row.get("probe_frequency_hz", ""),
                        "probe_phase": row.get("probe_phase", ""),
                        "probe_amplitude_multiplier": row.get("probe_amplitude_multiplier", ""),
                        "probe_lateral_shape": row.get("probe_lateral_shape", ""),
                        "seismic_variant_id": row.get("seismic_variant_id", ""),
                        "seismic_mismatch_family": row.get("seismic_mismatch_family", ""),
                        "suite": row.get("suite", ""),
                        "section_id": row.get("section_id", ""),
                        "scenario_id": row.get("scenario_id", ""),
                        "geometry_family": row.get("geometry_family", ""),
                        "duration_mode": row.get("duration_mode", ""),
                        "status": row.get("status", "ok"),
                    }
                )
                if max_patches is not None and len(rows) >= max_patches:
                    frame = pd.DataFrame.from_records(rows)
                    return _attach_paired_zero_patch_ids(frame)
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("No valid patches were produced.")
    return _attach_paired_zero_patch_ids(frame)


def _attach_paired_zero_patch_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    keys = {}
    variant_keys = {}
    for _, row in frame.iterrows():
        window = (
            int(row["lateral_start"]),
            int(row["twt_start"]),
            int(row["lateral_stop"]),
            int(row["twt_stop"]),
        )
        key = (str(row.get("sample_id", "")), *window)
        keys[key] = str(row["patch_id"])
        source_sample = _clean_text(row.get("source_sample_id"))
        seismic_variant = _clean_text(row.get("seismic_variant_id"))
        if source_sample and seismic_variant:
            variant_keys[(source_sample, seismic_variant, *window)] = str(row["patch_id"])
    paired = []
    for _, row in frame.iterrows():
        pair_sample = _clean_text(row.get("paired_zero_sample_id"))
        if not pair_sample:
            paired.append("")
            continue
        window = (
            int(row["lateral_start"]),
            int(row["twt_start"]),
            int(row["lateral_stop"]),
            int(row["twt_stop"]),
        )
        direct = keys.get((pair_sample, *window), "")
        if direct:
            paired.append(direct)
            continue
        seismic_variant = _clean_text(row.get("seismic_variant_id"))
        paired.append(variant_keys.get((pair_sample, seismic_variant, *window), ""))
    frame["paired_zero_patch_id"] = paired
    return frame


def _finite_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    return np.asarray(values, dtype=np.float64)[valid]


def compute_normalization(
    benchmark: SynthoseisBenchmark,
    patch_index: pd.DataFrame,
) -> dict[str, Any]:
    train = patch_index[patch_index["split"].eq("train")]
    if train.empty:
        raise ValueError("Cannot compute normalization without train patches.")
    buckets = {"seismic": [], "lfm": [], "target": [], "delta": []}
    for _, row in train.iterrows():
        sample = benchmark.load_sample(str(row["sample_id"]))
        target, seismic, lfm, valid, _ = _aligned_arrays(sample)
        sl = _row_slice(row)
        patch_valid = valid[sl]
        patch_target = target[sl]
        patch_seismic = seismic[sl]
        patch_lfm = lfm[sl]
        buckets["seismic"].append(_finite_values(patch_seismic, patch_valid))
        buckets["lfm"].append(_finite_values(patch_lfm, patch_valid))
        buckets["target"].append(_finite_values(patch_target, patch_valid))
        buckets["delta"].append(_finite_values(patch_target - patch_lfm, patch_valid))
    stats = {
        "normalization_scope": "train_only",
        "normalization_mask": "valid_mask_model",
        "time_alignment": "target_lfm_mask_drop_first_sample_when_seismic_uses_forward_axis",
    }
    for key, chunks in buckets.items():
        values = np.concatenate([chunk for chunk in chunks if chunk.size])
        if values.size < 2:
            raise ValueError(f"Insufficient finite train values for normalization: {key}")
        mean = float(np.mean(values))
        std = float(np.std(values))
        stats[key] = {"mean": mean, "std": std if std > 0.0 else 1.0}
    return stats


def _row_slice(row: Mapping[str, Any]) -> tuple[slice, slice]:
    return (
        slice(int(row["lateral_start"]), int(row["lateral_stop"])),
        slice(int(row["twt_start"]), int(row["twt_stop"])),
    )


class PatchDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        benchmark: SynthoseisBenchmark,
        patch_index: pd.DataFrame,
        *,
        split: str | Iterable[str],
        normalization: Mapping[str, Any],
    ) -> None:
        if isinstance(split, str):
            splits = {split}
        else:
            splits = set(split)
        frame = patch_index[patch_index["split"].astype(str).isin(splits)].copy()
        if frame.empty:
            raise ValueError(f"No patches selected for split={split}.")
        self.benchmark = benchmark
        self.frame = frame.reset_index(drop=True)
        self.normalization = normalization

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[int(index)].to_dict()
        sample = self.benchmark.load_sample(str(row["sample_id"]))
        target, seismic, lfm, valid, _ = _aligned_arrays(sample)
        sl = _row_slice(row)
        target_patch = target[sl]
        seismic_patch = seismic[sl]
        lfm_patch = lfm[sl]
        valid_patch = valid[sl]
        delta = target_patch - lfm_patch
        seismic_n = _norm(seismic_patch, self.normalization["seismic"])
        lfm_n = _norm(lfm_patch, self.normalization["lfm"])
        delta_n = _norm(delta, self.normalization["delta"])
        seismic_n = np.where(valid_patch & np.isfinite(seismic_n), seismic_n, 0.0)
        lfm_n = np.where(valid_patch & np.isfinite(lfm_n), lfm_n, 0.0)
        delta_n = np.where(valid_patch & np.isfinite(delta_n), delta_n, 0.0)
        target_patch = np.where(valid_patch & np.isfinite(target_patch), target_patch, 0.0)
        lfm_patch = np.where(valid_patch & np.isfinite(lfm_patch), lfm_patch, 0.0)
        lfm_ideal = np.asarray(sample.priors["lfm_ideal"], dtype=np.float32)
        if lfm_ideal.shape[0] != target.shape[0]:
            raise ValueError(
                f"lfm_ideal lateral shape mismatch for {sample.sample_id}: "
                f"{lfm_ideal.shape} vs {target.shape}"
            )
        if lfm_ideal.shape[1] == target.shape[1] + 1:
            lfm_ideal = lfm_ideal[:, 1:]
        if lfm_ideal.shape != target.shape:
            raise ValueError(
                f"lfm_ideal/target shape mismatch for {sample.sample_id}: "
                f"{lfm_ideal.shape} vs {target.shape}"
            )
        lfm_ideal_patch = np.where(
            valid_patch & np.isfinite(lfm_ideal[sl]),
            lfm_ideal[sl],
            0.0,
        )
        inputs = np.stack(
            [
                seismic_n,
                lfm_n,
                valid_patch.astype(np.float32),
            ],
            axis=0,
        )
        return {
            "input": torch.from_numpy(inputs.astype(np.float32)),
            "target_delta": torch.from_numpy(delta_n.astype(np.float32))[None, :, :],
            "target_log_ai": torch.from_numpy(target_patch.astype(np.float32))[None, :, :],
            "seismic": torch.from_numpy(
                np.where(valid_patch & np.isfinite(seismic_patch), seismic_patch, 0.0).astype(np.float32)
            )[None, :, :],
            "lfm": torch.from_numpy(lfm_patch.astype(np.float32))[None, :, :],
            "lfm_ideal": torch.from_numpy(lfm_ideal_patch.astype(np.float32))[None, :, :],
            "valid_mask": torch.from_numpy(valid_patch.astype(np.float32))[None, :, :],
            "patch_id": str(row["patch_id"]),
            "sample_id": str(row["sample_id"]),
            "sample_kind": str(row.get("sample_kind", "")),
        }


def _norm(values: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) - float(stats["mean"])) / float(stats["std"])


def denormalize_delta(values: np.ndarray, normalization: Mapping[str, Any]) -> np.ndarray:
    stats = normalization["delta"]
    return np.asarray(values, dtype=np.float32) * float(stats["std"]) + float(stats["mean"])
