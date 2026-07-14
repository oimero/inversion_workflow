"""Patch data access for the model-ablation gate."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cup.synthetic.benchmark import SynthoseisBenchmark


SPLIT_VALUES = {"train", "validation", "test", "benchmark"}


@dataclass(frozen=True)
class PatchSpec:
    lateral_samples: int = 32
    twt_samples: int = 128
    lateral_stride: int = 16
    twt_stride: int = 64


def default_eval_kinds() -> set[str]:
    return {
        "base",
        "seismic_variant",
    }


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
    for key in (
        "parent_realization_id",
        "source_sample_id",
        "realization_id",
        "sample_id",
    ):
        value = _clean_text(row.get(key))
        if value:
            return value
    raise ValueError("Cannot derive parent realization id from sample_index row.")


def _derive_split(
    parent: str, *, validation_fraction: float, test_fraction: float
) -> str:
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
    held_out_geometry_family: str = "",
) -> str:
    text = _clean_text(row.get("split")).lower()
    if split_policy == "strict":
        if text in SPLIT_VALUES:
            return text
        raise ValueError(
            f"Sample {row.get('sample_id')} has non-training split {text!r}; "
            "use --split-policy derive to create a research split."
        )
    if split_policy != "derive":
        raise ValueError(f"Unsupported split_policy: {split_policy}")
    evaluation_role = _clean_text(row.get("evaluation_role")).lower()
    geometry_family = _clean_text(row.get("geometry_family"))
    if evaluation_role == "geometry_holdout" or (
        held_out_geometry_family and geometry_family == held_out_geometry_family
    ):
        return "test"
    return _derive_split(
        _parent_id(row),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def _aligned_arrays(
    sample: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(sample.target_log_ai, dtype=np.float32)
    increment = np.asarray(sample.target_increment_log_ai, dtype=np.float32)
    seismic = np.asarray(sample.seismic_input, dtype=np.float32)
    lfm = np.asarray(sample.input_lfm_log_ai, dtype=np.float32)
    valid = np.asarray(sample.valid_mask, dtype=bool)
    sample_domain = str(getattr(sample, "sample_domain", ""))
    if sample_domain not in {"time", "depth"}:
        raise ValueError(
            f"Unsupported sample domain for {sample.sample_id}: {sample_domain!r}"
        )
    if seismic.shape != target.shape:
        raise ValueError(
            f"{sample_domain.capitalize()} v4 requires N-point seismic/target alignment "
            f"for {sample.sample_id}: {seismic.shape} vs {target.shape}"
        )
    if (
        target.ndim != 2
        or increment.ndim != 2
        or seismic.ndim != 2
        or lfm.ndim != 2
        or valid.ndim != 2
    ):
        raise ValueError(f"Expected 2D arrays for sample {sample.sample_id}.")
    if (
        target.shape[0] != seismic.shape[0]
        or increment.shape != target.shape
        or lfm.shape != target.shape
        or valid.shape != target.shape
    ):
        raise ValueError(
            f"Shape mismatch for {sample.sample_id}: target={target.shape}, "
            f"seismic={seismic.shape}, lfm={lfm.shape}, valid={valid.shape}"
        )
    if np.any(valid & (
        ~np.isfinite(target)
        | ~np.isfinite(increment)
        | ~np.isfinite(seismic)
        | ~np.isfinite(lfm)
    )):
        raise ValueError(
            f"Sample {sample.sample_id} has non-finite values inside valid_mask."
        )
    return target, seismic, lfm, valid


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
    min_valid_samples: int = 1,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sample_ids = benchmark.sample_ids(kinds=sample_kinds, status="ok")
    held_out_geometry_family = str(
        dict(getattr(benchmark, "manifest", {}).get("split_policy") or {}).get(
            "held_out_geometry_family"
        )
        or ""
    )
    for sample_id in sample_ids:
        sample = benchmark.load_sample(sample_id)
        target, _, _, valid = _aligned_arrays(sample)
        n_lateral, n_twt = target.shape
        lateral_starts = _window_starts(
            n_lateral, patch_spec.lateral_samples, patch_spec.lateral_stride
        )
        twt_starts = _window_starts(
            n_twt, patch_spec.twt_samples, patch_spec.twt_stride
        )
        if not lateral_starts or not twt_starts:
            continue
        row = sample.row
        split = _row_split(
            row,
            split_policy=split_policy,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            held_out_geometry_family=held_out_geometry_family,
        )
        parent = _parent_id(row)
        for lateral_start in lateral_starts:
            for twt_start in twt_starts:
                patch_mask = valid[
                    lateral_start : lateral_start + patch_spec.lateral_samples,
                    twt_start : twt_start + patch_spec.twt_samples,
                ]
                valid_count = int(np.count_nonzero(patch_mask))
                if valid_count < int(min_valid_samples):
                    continue
                valid_fraction = float(np.mean(patch_mask))
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
                        "evaluation_role": row.get("evaluation_role", ""),
                        "hdf5_group": row.get("hdf5_group", ""),
                        "lateral_start": lateral_start,
                        "lateral_stop": lateral_start + patch_spec.lateral_samples,
                        "twt_start": twt_start,
                        "twt_stop": twt_start + patch_spec.twt_samples,
                        "model_twt_start": twt_start,
                        "model_twt_stop": twt_start + patch_spec.twt_samples,
                        "time_alignment_mode": "explicit_lower_interface_N_point_operator",
                        "patch_lateral_samples": patch_spec.lateral_samples,
                        "patch_twt_samples": patch_spec.twt_samples,
                        "valid_fraction": valid_fraction,
                        "valid_samples": valid_count,
                        "paired_zero_sample_id": row.get("paired_zero_sample_id", ""),
                        "seismic_variant_id": row.get("seismic_variant_id", ""),
                        "seismic_mismatch_family": row.get(
                            "seismic_mismatch_family", ""
                        ),
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
            variant_keys[(source_sample, seismic_variant, *window)] = str(
                row["patch_id"]
            )
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
    buckets = {"seismic": [], "lfm": []}
    samples: dict[str, Any] = {}
    for _, row in train.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id not in samples:
            samples[sample_id] = benchmark.load_sample(sample_id)
        sample = samples[sample_id]
        target, seismic, lfm, valid = _aligned_arrays(sample)
        sl = _row_slice(row)
        patch_valid = valid[sl]
        patch_target = target[sl]
        patch_seismic = seismic[sl]
        patch_lfm = lfm[sl]
        input_valid = np.isfinite(patch_seismic) & np.isfinite(patch_lfm)
        buckets["seismic"].append(_finite_values(patch_seismic, input_valid))
        buckets["lfm"].append(_finite_values(patch_lfm, input_valid))
    stats = {
        "normalization_scope": "train_only",
        "normalization_mask": "finite_non_padding",
        "time_alignment": "explicit_lower_interface_N_point_operator",
    }
    for key, chunks in buckets.items():
        values = np.concatenate([chunk for chunk in chunks if chunk.size])
        if values.size < 2:
            raise ValueError(
                f"Insufficient finite train values for normalization: {key}"
            )
        mean = float(np.mean(values))
        std = float(np.std(values))
        stats[key] = {"mean": mean, "std": std if std > 0.0 else 1.0}
    return stats


def compute_input_reference_stats(
    benchmark: SynthoseisBenchmark,
    patch_index: pd.DataFrame,
    *,
    input_name: str = "seismic",
) -> dict[str, Any]:
    train = patch_index[patch_index["split"].eq("train")]
    if train.empty:
        raise ValueError("Cannot compute input reference stats without train patches.")
    chunks: list[np.ndarray] = []
    samples: dict[str, Any] = {}
    for _, row in train.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id not in samples:
            samples[sample_id] = benchmark.load_sample(sample_id)
        sample = samples[sample_id]
        target, seismic, lfm, valid = _aligned_arrays(sample)
        sl = _row_slice(row)
        patch_valid = np.isfinite(seismic[sl]) & np.isfinite(lfm[sl])
        if input_name == "seismic":
            chunks.append(_finite_values(seismic[sl], patch_valid))
        elif input_name == "lfm":
            chunks.append(_finite_values(lfm[sl], patch_valid))
        elif input_name == "target":
            chunks.append(_finite_values(target[sl], patch_valid))
        else:
            raise ValueError(
                f"Unsupported input reference stats input_name: {input_name}"
            )
    values = np.concatenate([chunk for chunk in chunks if chunk.size])
    if values.size < 2:
        raise ValueError(
            f"Insufficient finite train values for input reference stats: {input_name}"
        )
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "robust_rms": float(1.4826 * mad),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": median,
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "abs_p95": float(np.quantile(np.abs(values), 0.95)),
        "abs_p99": float(np.quantile(np.abs(values), 0.99)),
    }


def _row_slice(row: Mapping[str, Any]) -> tuple[slice, slice]:
    return (
        slice(int(row["lateral_start"]), int(row["lateral_stop"])),
        slice(int(row["twt_start"]), int(row["twt_stop"])),
    )


class PatchDataset(Dataset[dict[str, torch.Tensor | str]]):
    _DEFAULT_SAMPLE_CACHE_SIZE = 8

    def __init__(
        self,
        benchmark: SynthoseisBenchmark,
        patch_index: pd.DataFrame,
        *,
        split: str | Iterable[str],
        normalization: Mapping[str, Any],
        sample_cache_size: int = _DEFAULT_SAMPLE_CACHE_SIZE,
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
        if int(sample_cache_size) <= 0:
            raise ValueError("sample_cache_size must be positive.")
        self._sample_cache_size = int(sample_cache_size)
        self._sample_cache: OrderedDict[str, Any] = OrderedDict()

    def __len__(self) -> int:
        return int(len(self.frame))

    def _load_sample(self, sample_id: str) -> Any:
        cached = self._sample_cache.pop(sample_id, None)
        if cached is None:
            cached = self.benchmark.load_sample(sample_id)
        self._sample_cache[sample_id] = cached
        while len(self._sample_cache) > self._sample_cache_size:
            self._sample_cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[int(index)].to_dict()
        sample = self._load_sample(str(row["sample_id"]))
        target, seismic, lfm, valid = _aligned_arrays(sample)
        sl = _row_slice(row)
        target_patch = target[sl]
        increment = np.asarray(sample.target_increment_log_ai, dtype=np.float32)[sl]
        seismic_patch = seismic[sl]
        lfm_patch = lfm[sl]
        valid_patch = valid[sl]
        input_valid_patch = valid_patch
        forward_lfm_patch = np.where(
            valid_patch & np.isfinite(lfm_patch),
            lfm_patch,
            0.0,
        ).astype(np.float32)
        seismic_n = _norm(seismic_patch, self.normalization["seismic"])
        lfm_n = _norm(lfm_patch, self.normalization["lfm"])
        seismic_n = np.where(input_valid_patch & np.isfinite(seismic_n), seismic_n, 0.0)
        lfm_n = np.where(input_valid_patch & np.isfinite(lfm_n), lfm_n, 0.0)
        increment = np.where(
            valid_patch & np.isfinite(increment), increment, 0.0
        )
        target_patch = np.where(
            valid_patch & np.isfinite(target_patch), target_patch, 0.0
        )
        lfm_patch = np.where(valid_patch & np.isfinite(lfm_patch), lfm_patch, 0.0)
        lfm_ideal = np.asarray(sample.priors["lfm_ideal"], dtype=np.float32)
        if lfm_ideal.shape[0] != target.shape[0]:
            raise ValueError(
                f"lfm_ideal lateral shape mismatch for {sample.sample_id}: "
                f"{lfm_ideal.shape} vs {target.shape}"
            )
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
        seismic_model_consistent = np.asarray(
            sample.seismic_model_consistent, dtype=np.float32
        )
        if seismic_model_consistent.shape != target.shape:
            raise ValueError(
                f"Physics target shape mismatch for {sample.sample_id}."
            )
        if np.any(valid & ~np.isfinite(seismic_model_consistent)):
            raise ValueError(
                f"Physics target has non-finite values inside valid_mask for {sample.sample_id}."
            )
        physics_seismic_patch = np.where(
            valid_patch, seismic_model_consistent[sl], 0.0
        )
        sample_domain = str(getattr(sample, "sample_domain", ""))
        if sample_domain == "depth":
            sample_axis = np.asarray(sample.tvdss_model_m, dtype=np.float64)
        elif sample_domain == "time":
            sample_axis = np.asarray(sample.twt_model_s, dtype=np.float64)
        else:
            raise ValueError(
                f"Unsupported sample domain for {sample.sample_id}: {sample_domain!r}"
            )
        vertical_slice = sl[1]
        sample_axis_patch = sample_axis[vertical_slice]
        if sample_axis_patch.size != target_patch.shape[-1]:
            raise ValueError(
                f"Sample axis/patch mismatch for {sample.sample_id}: "
                f"{sample_axis_patch.size} vs {target_patch.shape[-1]}"
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
            "target_increment_log_ai": torch.from_numpy(
                increment.astype(np.float32)
            )[None, :, :],
            "target_log_ai": torch.from_numpy(target_patch.astype(np.float32))[
                None, :, :
            ],
            "seismic": torch.from_numpy(
                np.where(
                    valid_patch & np.isfinite(seismic_patch), seismic_patch, 0.0
                ).astype(np.float32)
            )[None, :, :],
            "seismic_model_consistent": torch.from_numpy(
                physics_seismic_patch.astype(np.float32)
            )[None, :, :],
            "sample_axis": torch.from_numpy(sample_axis_patch.astype(np.float64)),
            "input_lfm_log_ai": torch.from_numpy(
                lfm_patch.astype(np.float32)
            )[None, :, :],
            "forward_lfm": torch.from_numpy(forward_lfm_patch)[None, :, :],
            "lfm_ideal": torch.from_numpy(lfm_ideal_patch.astype(np.float32))[
                None, :, :
            ],
            "valid_mask": torch.from_numpy(valid_patch.astype(np.float32))[None, :, :],
            "patch_id": str(row["patch_id"]),
            "sample_id": str(row["sample_id"]),
            "sample_kind": str(row.get("sample_kind", "")),
            "sample_domain": sample_domain,
        }


def _norm(values: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) - float(stats["mean"])) / float(
        stats["std"]
    )


def denormalize_increment(
    values: np.ndarray, normalization: Mapping[str, Any]
) -> np.ndarray:
    if "delta" in normalization:
        raise ValueError(
            "Normalized-increment checkpoints are obsolete; GINN-v2 v5 outputs physical predicted_increment_log_ai."
        )
    return np.asarray(values, dtype=np.float32)
