"""Composable sources, losses, and stage runner for GINN-v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from cup.impedance import validate_increment_contract
from cup.synthetic.benchmark import SynthoseisBenchmark
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    write_json,
)
from ginn_v2.contracts import CHECKPOINT_SCHEMA_VERSION, EXPERIMENT_SCHEMA_VERSION
from ginn_v2.data import (
    PatchDataset, PatchSpec, build_patch_index, compute_input_reference_stats,
    compute_normalization,
)
from ginn_v2.experiment import ExperimentConfig
from ginn_v2.models import build_model
from ginn_v2.real_field import (
    RealFieldVolume, build_real_field_patch_index, finite_summary_stats,
    load_real_field_volume,
)
from ginn_v2.real_well_supervised import prepare_real_well_supervised_support
from ginn_v2.runtime import forward_physics_batch, load_benchmark_wavelet, masked_mse, resolve_device


@dataclass
class PreparedBlock:
    config: dict[str, Any]
    train_dataset: Dataset[Any] | None
    validation_dataset: Dataset[Any] | None
    wavelet: tuple[torch.Tensor, torch.Tensor, dict[str, float] | None] | None = None
    real_well_support: Any | None = None
    sample_axis_contract: dict[str, Any] | None = None


class RealFieldPatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self, model_volume: RealFieldVolume, physics_volume: RealFieldVolume,
        rows: pd.DataFrame, normalization: Mapping[str, Any], patch_spec: PatchSpec,
    ) -> None:
        _validate_aligned_real_volumes(model_volume, physics_volume)
        self.model_volume = model_volume
        self.physics_volume = physics_volume
        self.rows = rows.reset_index(drop=True)
        self.normalization = normalization
        self.patch_spec = patch_spec

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[int(index)]
        inline = int(row["inline_index"])
        lateral = slice(int(row["lateral_start"]), int(row["lateral_stop"]))
        vertical = slice(int(row["vertical_start"]), int(row["vertical_stop"]))
        seismic = self.model_volume.seismic[inline, lateral, vertical]
        target = self.physics_volume.seismic[inline, lateral, vertical]
        lfm = self.model_volume.lfm[inline, lateral, vertical]
        valid = self.model_volume.valid_mask[inline, lateral, vertical]
        forward_lfm = np.where(
            valid & np.isfinite(lfm),
            lfm,
            0.0,
        ).astype(np.float32)
        input_valid = valid
        if np.any(valid & (~np.isfinite(seismic) | ~np.isfinite(lfm) | ~np.isfinite(target))):
            raise ValueError("Real-field patch has non-finite values inside valid_mask.")
        shape = (self.patch_spec.lateral_samples, self.patch_spec.twt_samples)
        padding = ((0, shape[0] - valid.shape[0]), (0, shape[1] - valid.shape[1]))
        seismic = np.pad(seismic, padding, constant_values=0.0)
        target = np.pad(target, padding, constant_values=0.0)
        lfm = np.pad(lfm, padding, constant_values=0.0)
        forward_lfm = np.pad(forward_lfm, padding, constant_values=0.0)
        valid = np.pad(valid, padding, constant_values=False)
        input_valid = np.pad(input_valid, padding, constant_values=False)
        seismic_n = np.where(input_valid, (seismic - float(self.normalization["seismic"]["mean"])) / float(self.normalization["seismic"]["std"]), 0.0)
        lfm_n = np.where(input_valid, (lfm - float(self.normalization["lfm"]["mean"])) / float(self.normalization["lfm"]["std"]), 0.0)
        inputs = np.stack([seismic_n, lfm_n, valid.astype(np.float32)], axis=0).astype(np.float32)
        axis = self.model_volume.sample_axis.values[vertical]
        if len(axis) < shape[1]:
            step = self.model_volume.sample_axis.step
            axis = np.concatenate([axis, axis[-1] + step * np.arange(1, shape[1] - len(axis) + 1)])
        return {
            "input": torch.from_numpy(inputs),
            "input_lfm_log_ai": torch.from_numpy(
                np.where(valid, lfm, 0.0).astype(np.float32)
            )[None],
            "forward_lfm": torch.from_numpy(forward_lfm)[None],
            "valid_mask": torch.from_numpy(valid.astype(np.float32))[None],
            "seismic_model_consistent": torch.from_numpy(np.where(valid, target, 0.0).astype(np.float32))[None],
            "sample_axis": torch.from_numpy(np.asarray(axis, dtype=np.float64)),
            "patch_id": str(row["patch_id"]),
        }


def _validate_aligned_real_volumes(left: RealFieldVolume, right: RealFieldVolume) -> None:
    for name in ("ilines", "xlines"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            raise ValueError(f"Real-field model-input and physics-target {name} axes differ.")
    if not np.array_equal(left.sample_axis.values, right.sample_axis.values):
        raise ValueError("Real-field model-input and physics-target sample axes differ.")
    if (
        left.sample_axis.domain != right.sample_axis.domain
        or left.sample_axis.unit != right.sample_axis.unit
        or left.depth_basis != right.depth_basis
    ):
        raise ValueError("Real-field model-input and physics-target sample contracts differ.")
    if not np.array_equal(left.valid_mask, right.valid_mask):
        raise ValueError("Real-field model-input and physics-target valid masks differ.")
    if not np.array_equal(left.lfm, right.lfm, equal_nan=True):
        raise ValueError("Real-field model-input and physics-target LFM volumes differ.")


class DeterministicCycler:
    def __init__(self, dataset: Dataset[Any], *, batch_size: int, seed: int) -> None:
        if len(dataset) == 0:
            raise ValueError("Loss block dataset is empty.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        rng = np.random.default_rng(int(seed))
        self.order = rng.permutation(len(dataset)).tolist()
        self.position = 0

    def next(self) -> Any:
        indices = []
        for _ in range(self.batch_size):
            indices.append(self.order[self.position])
            self.position = (self.position + 1) % len(self.order)
        return next(iter(DataLoader(Subset(self.dataset, indices), batch_size=len(indices), shuffle=False)))


class DeterministicParentBalancedViewCycler:
    """Choose parent, base/view family, view, then patch in that order."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        batch_size: int,
        num_batches: int,
        seed: int,
        parent_weights: Mapping[str, float],
        view_weights: Mapping[str, float],
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("Loss block dataset is empty.")
        if int(batch_size) <= 0 or int(num_batches) <= 0:
            raise ValueError("Balanced sampler batch_size and num_batches must be positive.")
        frame = getattr(dataset, "frame", None)
        if not isinstance(frame, pd.DataFrame) or "sample_kind" not in frame:
            raise ValueError("parent_balanced_seismic_view requires patch metadata.")
        if "parent_realization_id" not in frame or "sample_kind" not in frame:
            raise ValueError("parent_balanced_seismic_view requires parent and sample-kind columns.")
        parent_weights = {str(key): float(value) for key, value in parent_weights.items()}
        # Mapping order is presentation, not part of the scientific random
        # contract.  Canonicalize view IDs before passing them to the RNG so
        # reordering equivalent YAML keys cannot change a sampled sequence.
        view_weights = {
            str(key): float(value) for key, value in view_weights.items()
        }
        view_weights = dict(sorted(view_weights.items()))
        if set(parent_weights) != {"base", "variant"}:
            raise ValueError("parent_balanced_seismic_view requires explicit base/variant weights.")
        if any(value < 0.0 for value in parent_weights.values()) or not np.isclose(
            sum(parent_weights.values()), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("parent_balanced_seismic_view parent weights must be nonnegative and sum to one.")
        if "base" in view_weights:
            raise ValueError("parent_balanced_seismic_view view weights must not contain base.")
        if parent_weights["variant"] > 0.0 and (
            any(value <= 0.0 for value in view_weights.values())
            or not view_weights
            or not np.isclose(sum(view_weights.values()), 1.0, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError("parent_balanced_seismic_view view weights must be positive and sum to one.")
        if parent_weights["variant"] == 0.0 and view_weights:
            raise ValueError("parent_balanced_seismic_view view weights must be empty for clean-only training.")
        dynamic_views = callable(getattr(dataset, "get_item_for_view", None))
        parents: dict[str, dict[str, list[int]]] = {}
        for index, raw in frame.iterrows():
            parent = str(raw["parent_realization_id"])
            kind = "base" if str(raw["sample_kind"]) == "base" else "variant"
            view_id = "base" if kind == "base" else str(raw.get("view_id") or "")
            parents.setdefault(parent, {}).setdefault(view_id, []).append(int(index))
        if dynamic_views:
            # Every configured view reuses the same parent/window rows.  The
            # view is materialized only after the sampler has selected it.
            for parent, views in parents.items():
                base_indices = list(views.get("base", []))
                if not base_indices:
                    raise ValueError(
                        f"parent_balanced_seismic_view requires a base patch for parent {parent!r}."
                    )
                benchmark = getattr(dataset, "benchmark", None)
                available_view_reader = getattr(benchmark, "available_view_ids", None)
                if not callable(available_view_reader):
                    # Keep the sampler useful for small in-memory contract
                    # tests and adapters which expose view materialization but
                    # deliberately do not expose a reader object.  Production
                    # PatchDataset takes the strict reader path above.
                    available = [
                        str(view_id)
                        for view_id in views
                        if str(view_id) != "base"
                    ]
                else:
                    available = [
                        str(value)
                        for value in available_view_reader(parent)
                    ]
                for view_id in available:
                    views.setdefault(view_id, list(base_indices))
        if not parents:
            raise ValueError("parent_balanced_seismic_view found no parent realizations.")
        if parent_weights["variant"] > 0.0 and not any(
            any(key != "base" for key in views) for views in parents.values()
        ):
            raise ValueError("variant parent weight is positive but benchmark has no seismic views.")
        requested_view_ids = {key for key, value in view_weights.items() if value > 0.0}
        available_view_ids = {view_id for views in parents.values() for view_id in views if view_id != "base"}
        if not requested_view_ids.issubset(available_view_ids):
            raise ValueError(
                "Configured seismic view weights reference views absent from benchmark: "
                f"{sorted(requested_view_ids - available_view_ids)}"
            )
        if parent_weights["variant"] > 0.0:
            incomplete = sorted(
                parent for parent, views in parents.items()
                if not requested_view_ids.issubset({key for key in views if key != "base"})
            )
            if incomplete:
                raise ValueError(
                    "Configured seismic view weights are not available for every parent: "
                    f"{incomplete[:5]}"
                )
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.parent_weights = parent_weights
        self.view_weights = view_weights
        self.parents = parents
        self.parent_ids = tuple(sorted(parents))
        rng = np.random.default_rng(int(seed))
        total_items = self.batch_size * self.num_batches
        self._schedule = list(range(total_items))
        self._cursor = 0
        self._rng = rng
        self._patch_rngs = {}
        self._patch_orders = {}
        for parent, views in parents.items():
            for view_id, indices in views.items():
                key_digest = hashlib.sha256(
                    f"parent_balanced_seismic_view\0{seed}\0{parent}\0{view_id}".encode("utf-8")
                ).digest()
                key_seed = int.from_bytes(key_digest[:8], "big", signed=False)
                key_rng = np.random.default_rng(key_seed)
                self._patch_rngs[(parent, view_id)] = key_rng
                self._patch_orders[(parent, view_id)] = key_rng.permutation(indices).tolist()
        self._patch_positions = {key: 0 for key in self._patch_orders}
        self.last_batch_counts: dict[str, int] = {}
        self.last_batch_view_counts: dict[str, int] = {}

    def next(self) -> Any:
        if self._cursor + self.batch_size > len(self._schedule):
            raise RuntimeError("Balanced sampler exhausted its configured epoch batch count.")
        indices: list[int] = []
        selected_keys: list[tuple[str, str]] = []
        counts: dict[str, int] = {}
        view_counts: dict[str, int] = {}
        for _ in self._schedule[self._cursor : self._cursor + self.batch_size]:
            parent = self.parent_ids[int(self._rng.integers(0, len(self.parent_ids)))]
            family = str(self._rng.choice(["base", "variant"], p=np.asarray([self.parent_weights["base"], self.parent_weights["variant"]]) / sum(self.parent_weights.values())))
            available = self.parents[parent]
            if family == "base":
                view_id = "base"
            else:
                choices = [key for key in self.view_weights if key in available and self.view_weights[key] > 0.0]
                if not choices:
                    raise ValueError(f"No configured seismic view is available for parent {parent!r}.")
                else:
                    weights = np.asarray([self.view_weights[key] for key in choices], dtype=float)
                    weights /= weights.sum()
                    view_id = str(self._rng.choice(choices, p=weights))
            key = (parent, view_id)
            order = self._patch_orders[key]
            position = self._patch_positions[key]
            if position >= len(order):
                order = self._patch_rngs[key].permutation(order).tolist()
                self._patch_orders[key] = order
                position = 0
            indices.append(order[position])
            selected_keys.append((parent, view_id))
            self._patch_positions[key] = position + 1
            kind_label = "base" if view_id == "base" else "seismic_view"
            counts[kind_label] = counts.get(kind_label, 0) + 1
            view_counts[view_id] = view_counts.get(view_id, 0) + 1
        self._cursor += self.batch_size
        self.last_batch_counts = counts
        self.last_batch_view_counts = view_counts
        if callable(getattr(self.dataset, "get_item_for_view", None)):
            items = [
                self.dataset.get_item_for_view(
                    index,
                    view_id,
                )
                for index, (_, view_id) in zip(indices, selected_keys)
            ]
            return default_collate(items)
        return next(iter(DataLoader(Subset(self.dataset, indices), batch_size=len(indices), shuffle=False)))


def _seed(base: int, stage_index: int, block_index: int, epoch: int) -> int:
    return int(np.random.SeedSequence([base, stage_index, block_index, epoch]).generate_state(1)[0])


def validate_source_axis_contracts(
    source_contracts: Mapping[str, Mapping[str, Any]],
    consumed_sources: set[str],
) -> dict[str, Any]:
    contracts = {
        tuple(dict(source_contracts[source_id]).items())
        for source_id in consumed_sources
    }
    if len(contracts) != 1:
        details = {source_id: dict(source_contracts[source_id]) for source_id in sorted(consumed_sources)}
        raise ValueError(
            "A GINN-v2 experiment cannot mix time/depth or incompatible sample-axis contracts "
            f"without an explicit resampling adapter: {details}"
        )
    return dict(source_contracts[next(iter(consumed_sources))])


def _sample_axis_contract(
    values: np.ndarray, *, domain: str, unit: str, depth_basis: str | None,
) -> dict[str, Any]:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError("Sample axis must be a finite one-dimensional array with at least two samples.")
    differences = np.diff(axis)
    step = float(differences[0])
    if step <= 0 or not np.allclose(differences, step, rtol=0.0, atol=max(1e-12, abs(step) * 1e-9)):
        raise ValueError("GINN-v2 requires a regular increasing sample axis.")
    return {
        "sample_domain": domain,
        "sample_unit": unit,
        "depth_basis": depth_basis,
        "sample_step": step,
        "axis_direction": "increasing",
        "axis_regularity": "regular",
    }


def _synthetic_axis_contract(benchmark: SynthoseisBenchmark) -> dict[str, Any]:
    sample_ids = benchmark.sample_ids(status="ok")
    if not sample_ids:
        raise ValueError("Synthoseis-lite benchmark has no usable samples.")
    sample = benchmark.load_sample(sample_ids[0])
    domain = str(sample.sample_domain)
    values = sample.twt_model_s if domain == "time" else sample.tvdss_model_m
    return _sample_axis_contract(
        values,
        domain=domain,
        unit="s" if domain == "time" else "m",
        depth_basis=benchmark.manifest.get("depth_basis"),
    )


def _increment_axis_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    resolved = validate_increment_contract(contract)
    return {
        "sample_domain": resolved.sample_domain,
        "sample_unit": resolved.sample_unit,
        "depth_basis": resolved.depth_basis,
        "sample_step": resolved.sample_interval,
        "axis_direction": "increasing",
        "axis_regularity": "regular",
    }


def _validate_increment_contract_against_axis(
    increment_contract: Mapping[str, Any], axis_contract: Mapping[str, Any], *, label: str,
) -> None:
    expected = _increment_axis_contract(increment_contract)
    actual = dict(axis_contract)
    for key in ("sample_domain", "sample_unit", "depth_basis", "axis_direction", "axis_regularity"):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"{label} sample-axis field {key!r} is incompatible with increment_contract: "
                f"expected {expected.get(key)!r}, got {actual.get(key)!r}."
            )
    if not np.isclose(float(actual.get("sample_step")), float(expected["sample_step"]), rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"{label} sample interval is incompatible with increment_contract: "
            f"expected {expected['sample_step']}, got {actual.get('sample_step')}."
        )


def _resolve_auto_benchmark(root: Path) -> Path:
    results = root / "experiments" / "synthoseis_lite" / "results"
    candidates = [
        path / "generate_field_conditioned"
        for path in results.glob("*")
        if (path / "generate_field_conditioned" / "benchmark_manifest.json").is_file()
        and (path / "generate_field_conditioned" / "realization_index.csv").is_file()
        and (path / "generate_field_conditioned" / "synthetic_benchmark.h5").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete Synthoseis-lite benchmark under {results}.")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _benchmark_forward_model_inputs_path(
    benchmark: SynthoseisBenchmark, benchmark_dir: Path,
) -> Path | None:
    if str(benchmark.manifest.get("sample_domain") or "") != "depth":
        return None
    raw_path = str(benchmark.manifest.get("forward_model_inputs_path") or "").strip()
    if not raw_path:
        raise ValueError(
            "Depth Synthoseis benchmark lacks forward_model_inputs_path; "
            "R0/R1 require the frozen depth forward contract."
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = benchmark_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Depth forward_model_inputs_path not found: {path}")
    return path


def _source_path(value: Any, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _synthetic_sample_kinds(source: Mapping[str, Any]) -> set[str]:
    """The v5 patch catalog is parent-owned; views are selected dynamically."""
    return {"base"}


def _materialize_parent_split_assignment(
    benchmark: SynthoseisBenchmark, *, path: Path,
) -> dict[str, str]:
    """Materialize the GINN-owned parent split with a versioned hash contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    seed = 20260714
    split_contract = {
        "version": "parent_hash_split_v1",
        "owner": "ginn_v2_experiment_suite",
        "seed": seed,
        "hash_algorithm": "sha256",
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "geometry_holdout_role": "test",
    }
    split_identity = hashlib.sha256(
        json.dumps(split_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    seen: set[str] = set()
    for _, raw in benchmark.realizations.iterrows():
        parent = str(raw["realization_id"])
        if not parent or parent in seen:
            raise ValueError(f"Benchmark realization IDs must be unique and non-empty: {parent!r}")
        seen.add(parent)
        role = str(raw.get("evaluation_role") or "")
        if role == "geometry_holdout":
            split = "test"
            hash_value = ""
        else:
            digest = hashlib.sha256(
                f"parent_hash_split_v1\0{seed}\0{parent}".encode("utf-8")
            ).digest()
            value = int.from_bytes(digest[:8], "big", signed=False) / float(2**64)
            hash_value = f"{value:.17g}"
            split = "test" if value < 0.15 else "validation" if value < 0.30 else "train"
        rows.append({
            "realization_id": parent,
            "evaluation_role": role,
            "hash_value": hash_value,
            "split": split,
            "split_contract": "parent_hash_split_v1",
            "split_contract_identity": split_identity,
            "seed": str(seed),
            "hash_algorithm": "sha256",
            "validation_fraction": "0.15",
            "test_fraction": "0.15",
        })
    frame = pd.DataFrame.from_records(rows).sort_values("realization_id")
    frame.to_csv(path, index=False)
    return {str(row["realization_id"]): str(row["split"]) for _, row in frame.iterrows()}


def _prepare_synthetic_indices(
    *, benchmark: SynthoseisBenchmark, source: Mapping[str, Any], patch_spec: PatchSpec,
    blocks: list[Mapping[str, Any]], split_assignment: Mapping[str, str],
    logger: logging.Logger | None = None,
    allow_smoke_validation_duplication: bool = False,
) -> dict[str, pd.DataFrame]:
    result = {}
    for block in blocks:
        block_id = str(block["block_id"])
        build_started = time.perf_counter()
        if logger is not None:
            logger.info("patch_index_build_start block_id=%s kind=%s", block_id, block["kind"])
        index = build_patch_index(
            benchmark,
            patch_spec=patch_spec,
            sample_kinds=_synthetic_sample_kinds(source),
            min_valid_samples=(1 if block["kind"] == "physics" else int(block["min_valid_samples"])),
            max_patches=(int(source["max_patches"]) if source.get("max_patches") is not None else None),
            split_assignment=split_assignment,
        )
        if logger is not None:
            logger.info(
                "patch_index_build_end block_id=%s rows=%d elapsed_s=%.1f",
                block_id,
                len(index),
                time.perf_counter() - build_started,
            )
        if block["kind"] == "physics":
            counts = []
            cache: dict[str, Any] = {}
            cache_limit = 8
            loaded_sample_count = 0
            progress_interval = max(1, len(index) // 10)
            index_started = time.perf_counter()
            for row_number, (_, row) in enumerate(index.iterrows(), start=1):
                sample_id = str(row["sample_id"])
                sample = cache.get(sample_id)
                if sample is None:
                    sample = benchmark.load_sample(sample_id)
                    if len(cache) >= cache_limit:
                        cache.pop(next(iter(cache)))
                    cache[sample_id] = sample
                    loaded_sample_count += 1
                lfm_context = np.asarray(sample.input_lfm_log_ai, dtype=np.float64)[
                    int(row["lateral_start"]):int(row["lateral_stop"]),
                    int(row["twt_start"]):int(row["twt_stop"]),
                ]
                valid_context = np.asarray(sample.valid_mask, dtype=bool)[
                    int(row["lateral_start"]):int(row["lateral_stop"]),
                    int(row["twt_start"]):int(row["twt_stop"]),
                ]
                physics_target = np.asarray(sample.seismic_model_consistent, dtype=np.float64)[
                    int(row["lateral_start"]):int(row["lateral_stop"]),
                    int(row["twt_start"]):int(row["twt_stop"]),
                ]
                counts.append(int(np.count_nonzero(
                    valid_context
                    & np.isfinite(physics_target)
                    & np.isfinite(lfm_context)
                )))
                if logger is not None and (
                    row_number == 1
                    or row_number % progress_interval == 0
                    or row_number == len(index)
                ):
                    logger.info(
                        "physics_index_progress block_id=%s rows=%d/%d loaded_samples=%d cached_samples=%d elapsed_s=%.1f",
                        block_id,
                        row_number,
                        len(index),
                        loaded_sample_count,
                        len(cache),
                        time.perf_counter() - index_started,
                    )
            index["supervision_valid_samples"] = counts
            index = index[index["supervision_valid_samples"] >= int(block["min_valid_samples"])].copy()
        if source.get("max_patches") is not None and "validation" not in set(index["split"].astype(str)):
            if not allow_smoke_validation_duplication:
                raise ValueError(
                    "Capped synthetic patch indices produced no independent validation parent; "
                    "use run_mode=smoke with development_limited=true for a duplicated validation patch."
                )
            validation = index.head(min(4, len(index))).copy()
            validation["patch_id"] = validation["patch_id"].astype(str) + "__smoke_validation"
            validation["split"] = "validation"
            index = pd.concat([index, validation], ignore_index=True)
        path = output_dir / f"{block_id}_patch_index.csv"
        index.to_csv(path, index=False)
        result[block_id] = index
    return result


_SYNTHETIC_PREDICTION_INDEX_COLUMNS = {
    "patch_id",
    "sample_id",
    "sample_kind",
    "split",
    "lateral_start",
    "lateral_stop",
    "twt_start",
    "twt_stop",
    "patch_lateral_samples",
    "patch_twt_samples",
}


def _deployment_patch_index_contract(
    *,
    stage: Mapping[str, Any],
    stage_dir: Path,
    sources: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> tuple[Path | None, dict[str, dict[str, Any]], str | None]:
    """Resolve the synthetic evaluation index from the selection block."""
    blocks = {
        str(block.get("block_id") or ""): dict(block)
        for block in stage.get("loss_blocks", [])
    }
    catalog: dict[str, dict[str, Any]] = {}
    for block_id, block in blocks.items():
        path = stage_dir / f"{block_id}_patch_index.csv"
        if not path.is_file():
            continue
        source_id = str(block.get("source") or "")
        source_kind = str(dict(sources.get(source_id) or {}).get("kind") or "")
        catalog[block_id] = {
            "source": source_id,
            "source_kind": source_kind,
            "path": repo_relative_path(path, root=root),
        }
    selection_metric = str(dict(stage.get("validation") or {}).get("selection_metric") or "")
    selection_block_id = selection_metric.split(".", 1)[0] if selection_metric else ""
    selection_block = blocks.get(selection_block_id)
    if selection_block is None:
        raise ValueError(
            f"Deployment stage selection block is missing: {selection_block_id!r}."
        )
    if str(selection_block.get("kind") or "") != "synthetic_supervised":
        return None, catalog, None
    source_id = str(selection_block.get("source") or "")
    if str(dict(sources.get(source_id) or {}).get("kind") or "") != "synthoseis_lite":
        raise ValueError(
            "Synthetic deployment selection block must consume a synthoseis_lite source."
        )
    path = stage_dir / f"{selection_block_id}_patch_index.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Deployment selection patch index is missing: {path}")
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(_SYNTHETIC_PREDICTION_INDEX_COLUMNS - columns)
    if missing:
        raise ValueError(
            f"Deployment synthetic patch index has incompatible columns: missing={missing}, "
            f"available={sorted(columns)}."
        )
    return path, catalog, source_id


def _normalization_for_synthetic(benchmark: SynthoseisBenchmark, index: pd.DataFrame) -> dict[str, Any]:
    normalization = compute_normalization(benchmark, index)
    if "delta" in normalization:
        raise AssertionError("Composable normalization must not contain delta statistics.")
    return normalization


def _real_field_config(
    source: Mapping[str, Any], *, root: Path, transform: str,
    reference_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lfm_run_dir = _source_path(source.get("lfm_run_dir"), root)
    with (lfm_run_dir / "lfm_run_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    seismic = dict(summary.get("seismic") or {})
    inputs = {
            "lfm_run_dir": str(lfm_run_dir),
            "variant_id": str(source.get("variant_id") or ""),
            "seismic_file": str(_source_path(seismic.get("path"), root)),
            "seismic_type": str(seismic.get("type") or ""),
            "segy_options": dict(source.get("segy_options") or {}),
            "seismic_value_transform": transform,
            "seismic_reference_stats": dict(reference_stats or {}),
            "lfm_value_transform": "identity",
    }
    if source.get("forward_model_inputs_path") is not None:
        inputs["forward_model_inputs_path"] = str(
            _source_path(source["forward_model_inputs_path"], root)
        )
    if source.get("well_control_run_dir") is not None:
        inputs["well_control_run_dir"] = str(_source_path(source["well_control_run_dir"], root))
    return {
        "real_field_inputs": inputs,
        "volume": dict(source.get("volume") or {}),
    }


def _load_real_source(
    source: Mapping[str, Any], *, root: Path,
    reference_stats: Mapping[str, Any] | None = None,
) -> tuple[RealFieldVolume, RealFieldVolume]:
    model_transform = str(source.get("model_input_seismic_transform") or "identity")
    model_volume = load_real_field_volume(config=_real_field_config(source, root=root, transform=model_transform, reference_stats=reference_stats), root=root, data_root=root)
    physics_transform = str(source.get("physics_target_seismic_transform") or "identity")
    physics_volume = load_real_field_volume(
        config=_real_field_config(source, root=root, transform=physics_transform, reference_stats=reference_stats),
        root=root,
        data_root=root,
    )
    return model_volume, physics_volume


def _input_reference_stats_for_real(
    volume: RealFieldVolume, train_inline_indices: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(volume.seismic, dtype=np.float64)[train_inline_indices]
    valid = np.isfinite(values) & np.isfinite(volume.lfm[train_inline_indices])
    return finite_summary_stats(values[valid])


def _normalization_for_real(volume: RealFieldVolume, train_inline_indices: np.ndarray) -> dict[str, Any]:
    if train_inline_indices.size == 0:
        raise ValueError("Real normalization reference has no training spatial support.")
    valid = (
        np.isfinite(volume.seismic[train_inline_indices])
        & np.isfinite(volume.lfm[train_inline_indices])
    )
    result: dict[str, Any] = {"normalization_scope": "reference_source_train_partition", "normalization_mask": "finite_non_padding"}
    for name, values in (("seismic", volume.seismic), ("lfm", volume.lfm)):
        selected = np.asarray(values, dtype=np.float64)[train_inline_indices][valid]
        std = float(np.std(selected))
        if selected.size < 2 or not np.isfinite(std) or std <= 0:
            raise ValueError(f"Real normalization reference has invalid {name} values.")
        result[name] = {"mean": float(np.mean(selected)), "std": std}
    return result


def _real_patch_rows(
    volume: RealFieldVolume, *, patch_spec: PatchSpec, min_valid_samples: int,
    validation: Mapping[str, Any], geometry: Any,
) -> pd.DataFrame:
    fraction = float(validation.get("fraction", 0.1))
    if not 0 < fraction < 1:
        raise ValueError("real_field.validation_split.fraction must be between zero and one.")
    n_validation = max(1, int(np.ceil(volume.ilines.size * fraction)))
    gap_m = float(validation.get("gap_m", 0.0))
    validation_start = volume.ilines.size - n_validation
    validation_inline = float(volume.ilines[validation_start])
    validation_segment = np.asarray(
        [
            geometry.line_to_coord(validation_inline, float(volume.xlines[0])),
            geometry.line_to_coord(validation_inline, float(volume.xlines[-1])),
        ],
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    for inline_index in range(volume.ilines.size):
        if inline_index >= validation_start:
            split = "validation"
            distance_to_validation_m = 0.0
        else:
            inline_segment = np.asarray(
                [
                    geometry.line_to_coord(float(volume.ilines[inline_index]), float(volume.xlines[0])),
                    geometry.line_to_coord(float(volume.ilines[inline_index]), float(volume.xlines[-1])),
                ],
                dtype=np.float64,
            )
            distance_to_validation_m = _segment_distance_m(inline_segment, validation_segment)
            split = "train" if distance_to_validation_m >= gap_m else "gap"
        if split == "gap":
            continue
        data_valid = np.isfinite(volume.seismic[inline_index]) & np.isfinite(volume.lfm[inline_index])
        for patch_index, patch in enumerate(build_real_field_patch_index(
            data_valid, lateral_samples=patch_spec.lateral_samples,
            vertical_samples=patch_spec.twt_samples, lateral_stride=patch_spec.lateral_stride,
            vertical_stride=patch_spec.twt_stride,
        )):
            lfm_context = volume.lfm[
                inline_index, patch.lateral_start:patch.lateral_stop,
                patch.sample_start:patch.sample_stop,
            ]
            if (
                lfm_context.shape != (patch_spec.lateral_samples, patch_spec.twt_samples)
                or not np.all(np.isfinite(lfm_context))
            ):
                continue
            mask = data_valid[patch.lateral_start:patch.lateral_stop, patch.sample_start:patch.sample_stop]
            count = int(np.count_nonzero(mask))
            if count < int(min_valid_samples):
                continue
            rows.append({
                "patch_id": f"il{inline_index:05d}_p{patch_index:06d}", "split": split,
                "inline_index": inline_index, "lateral_start": patch.lateral_start,
                "lateral_stop": patch.lateral_stop, "vertical_start": patch.sample_start,
                "vertical_stop": patch.sample_stop, "supervision_valid_samples": count,
                "distance_to_validation_m": float(distance_to_validation_m),
            })
    frame = pd.DataFrame(rows)
    if frame.empty or not {"train", "validation"}.issubset(set(frame["split"])):
        raise ValueError("Real-field spatial split did not produce both train and validation patches.")
    return frame


def _point_segment_distance(point: np.ndarray, segment: np.ndarray) -> float:
    vector = segment[1] - segment[0]
    denominator = float(np.dot(vector, vector))
    if denominator == 0.0:
        return float(np.linalg.norm(point - segment[0]))
    fraction = float(np.dot(point - segment[0], vector) / denominator)
    projection = segment[0] + min(1.0, max(0.0, fraction)) * vector
    return float(np.linalg.norm(point - projection))


def _segment_distance_m(left: np.ndarray, right: np.ndarray) -> float:
    return min(
        _point_segment_distance(left[0], right),
        _point_segment_distance(left[1], right),
        _point_segment_distance(right[0], left),
        _point_segment_distance(right[1], left),
    )


def _load_real_wavelet(
    source: Mapping[str, Any], volume: RealFieldVolume, *, root: Path, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
    if volume.sample_axis.domain == "time":
        directory = _source_path(source.get("wavelet_generation_dir"), root)
        path = directory / "selected_wavelet.csv"
        relation = None
    else:
        contract_path = _source_path(source.get("forward_model_inputs_path"), root)
        if not contract_path.is_file():
            raise FileNotFoundError(
                "Depth real-field physics requires source.forward_model_inputs_path."
            )
        with contract_path.open("r", encoding="utf-8") as handle:
            contract = json.load(handle)
        wavelet_ref = str(dict(contract.get("wavelet") or {}).get("path") or "")
        path = _source_path(wavelet_ref, root)
        raw_relation = dict(contract.get("ai_velocity_relation") or {})
        relation = {"a": float(raw_relation["a"]), "b": float(raw_relation["b"])}
    frame = pd.read_csv(path)
    return (
        torch.as_tensor(frame["time_s"].to_numpy(), dtype=torch.float32, device=device),
        torch.as_tensor(frame["amplitude"].to_numpy(), dtype=torch.float32, device=device),
        relation,
    )


def masked_centered_rms(
    values: torch.Tensor, mask: torch.Tensor, *, epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return item-wise standardized tensors and centered RMS values."""
    if values.shape != mask.shape:
        raise ValueError("values/mask shape mismatch for centered RMS.")
    flat_x = values.reshape(values.shape[0], -1)
    flat_m = (mask > 0.5).reshape(mask.shape[0], -1)
    count = flat_m.sum(dim=1)
    if bool(torch.any(count == 0)):
        raise ValueError("Centered RMS item has no valid waveform samples.")
    weights = flat_m.to(values.dtype)
    mean = (flat_x * weights).sum(dim=1) / count
    centered = flat_x - mean[:, None]
    rms = torch.sqrt((centered.square() * weights).sum(dim=1) / count)
    denominator = torch.clamp(rms, min=float(epsilon) ** 0.5)
    standardized = torch.where(flat_m, centered / denominator[:, None], torch.zeros_like(centered))
    return standardized.reshape_as(values), rms


def _block_loss(
    prepared: PreparedBlock, batch: Any, model: torch.nn.Module, *, device: torch.device,
    normalization: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = prepared.config
    kind = str(cfg["kind"])
    if kind == "real_well_supervised":
        loss, counts = prepared.real_well_support.training_loss(model, device=device)
        if int(counts["selected_real_samples"]) < int(cfg["min_valid_samples"]):
            raise ValueError(
                f"Real-well block {cfg['block_id']} selected "
                f"{counts['selected_real_samples']} valid samples, below "
                f"min_valid_samples={cfg['min_valid_samples']}."
            )
        return loss, {"mse": float(loss.detach().cpu()), **{key: float(value) for key, value in counts.items()}}
    x = batch["input"].to(device)
    prediction = model(x)
    if kind == "synthetic_supervised":
        supervised_mask = batch["valid_mask"].to(device)
        loss = masked_mse(
            prediction,
            batch["target_increment_log_ai"].to(device),
            supervised_mask,
        )
        return loss, {
            "mse": float(loss.detach().cpu()),
            "valid_sample_count": float(torch.count_nonzero(supervised_mask).item()),
        }
    if kind != "physics" or prepared.wavelet is None:
        raise ValueError(f"Unsupported prepared loss block: {kind}")
    forward_lfm = batch["forward_lfm"].to(device)
    physics_valid = batch["valid_mask"].to(device)
    if str(cfg.get("source_kind")) == "synthoseis_lite":
        physics_background = batch["canonical_background_log_ai"].to(device)
    else:
        physics_background = forward_lfm
    pred_log_ai = physics_background + prediction
    wavelet_time, wavelet_amp, relation = prepared.wavelet
    if prepared.sample_axis_contract is None:
        raise ValueError(f"Loss block {cfg['block_id']} lacks a sample-axis contract.")
    if not bool(torch.all(torch.isfinite(pred_log_ai))):
        raise ValueError(
            "Physics requires a complete finite LFM patch; padded or non-finite "
            "logAI context is not accepted."
        )
    synthetic = forward_physics_batch(
        pred_log_ai[:, 0], sample_axes=batch["sample_axis"].to(device),
        sample_domain=str(prepared.sample_axis_contract["sample_domain"]), wavelet_time_s=wavelet_time,
        wavelet_amp=wavelet_amp, ai_velocity_relation=relation,
    )
    target = batch["seismic_model_consistent"].to(device)[:, 0]
    mask = physics_valid[:, 0] > 0.5
    enough_waveform = (
        mask.reshape(mask.shape[0], -1).sum(dim=1)
        >= int(cfg["min_valid_samples"])
    )
    skipped_item_count = int(mask.shape[0] - torch.count_nonzero(enough_waveform).item())
    if not bool(torch.any(enough_waveform)):
        raise AssertionError(
            "Physics patch index admitted a batch with no item meeting min_valid_samples "
            "on finite, non-padding waveform samples."
        )
    synthetic = synthetic[enough_waveform]
    target = target[enough_waveform]
    mask = mask[enough_waveform]
    prediction_for_l2 = prediction[enough_waveform]
    valid_for_l2 = physics_valid[enough_waveform]
    if str(cfg.get("source_kind")) == "real_field":
        pred_n, pred_rms = masked_centered_rms(synthetic, mask, epsilon=float(cfg.get("centered_rms_epsilon", 1e-12)))
        target_n, target_rms = masked_centered_rms(target, mask, epsilon=float(cfg.get("centered_rms_epsilon", 1e-12)))
        minimum = float(cfg.get("min_centered_rms", 1e-6))
        usable = (pred_rms >= minimum) & (target_rms >= minimum)
        skipped_item_count += int(usable.numel() - torch.count_nonzero(usable).item())
        if not bool(torch.any(usable)):
            raise ValueError("Real physics batch has no item above min_centered_rms.")
        waveform = masked_mse(pred_n[usable], target_n[usable], mask[usable])
        waveform_mask = mask[usable]
        diagnostics = {
            "observed_centered_rms": float(target_rms[usable].mean().detach().cpu()),
            "predicted_centered_rms": float(pred_rms[usable].mean().detach().cpu()),
            "predicted_to_observed_rms_ratio": float((pred_rms[usable] / target_rms[usable]).mean().detach().cpu()),
        }
    else:
        waveform = masked_mse(synthetic, target, mask)
        waveform_mask = mask
        diagnostics = {}
    delta_l2 = masked_mse(prediction_for_l2, torch.zeros_like(prediction_for_l2), valid_for_l2)
    diagnostics["waveform"] = float(waveform.detach().cpu())
    diagnostics["increment_l2"] = float(delta_l2.detach().cpu())
    total = waveform + float(cfg["increment_l2_weight"]) * delta_l2
    diagnostics["total"] = float(total.detach().cpu())
    diagnostics["valid_sample_count"] = float(torch.count_nonzero(waveform_mask).item())
    diagnostics["increment_valid_sample_count"] = float(torch.count_nonzero(valid_for_l2).item())
    diagnostics["skipped_item_count"] = float(skipped_item_count)
    return total, diagnostics


def _checkpoint_payload(
    *, model: torch.nn.Module, config: ExperimentConfig, model_info: Mapping[str, Any],
    normalization: Mapping[str, Any], stage_id: str, kind: str, epoch: int,
    metric_name: str, metric_value: float, sample_axis: Mapping[str, Any],
    increment_contract: Mapping[str, Any],
    training_sources: Mapping[str, Any],
    stage_lineage: list[Mapping[str, Any]],
    stage_deployment_eligible: bool,
    stage_deployment_eligibility_reason: str,
    physics_closures: list[Mapping[str, Any]],
    stage_loss_blocks: list[Mapping[str, Any]],
    sampling_statistics: Mapping[str, Any] | None = None,
    forward_model_inputs_path: str = "",
) -> dict[str, Any]:
    synthetic_sampling_contract: list[dict[str, Any]] = []
    for stage in config.stages:
        for block in stage.get("loss_blocks", []):
            if str(block.get("sampling", {}).get("kind") or "") != "parent_balanced_seismic_view":
                continue
            synthetic_sampling_contract.append({
                "stage_id": str(stage.get("stage_id") or ""),
                "block_id": str(block.get("block_id") or ""),
                "source": str(block.get("source") or ""),
                "parent_weights": dict(block["sampling"]["parent_weights"]),
                "view_weights": dict(block["sampling"]["view_weights"]),
                "validation": dict(stage.get("validation", {}).get("seismic_views") or {}),
            })
    benchmark_identity = {
        source_id: {
            key: source.get(key, "")
            for key in (
                "benchmark_dir", "schema_version", "science_revision",
                "projection_contract_version", "seismic_view_contract_version",
                "seismic_operator_contract_version", "random_stream_contract_version",
                "contract_fingerprint_sha256", "materialized_view_ids",
                "materialized_view_spec_hashes", "split_assignment_path",
            )
            if key in source
        }
        for source_id, source in training_sources.items()
        if str(source.get("kind") or "") == "synthoseis_lite"
    }
    split_contract = {
        "version": "parent_hash_split_v1",
        "owner": "ginn_v2_experiment_suite",
        "seed": 20260714,
        "hash_algorithm": "sha256",
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
        "geometry_holdout_role": "test",
    }
    split_identity = hashlib.sha256(
        json.dumps(split_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "stage_id": stage_id,
        "checkpoint_kind": kind,
        "epoch": int(epoch),
        "selection_metric": metric_name,
        "selection_metric_value": float(metric_value),
        "architecture": config.architecture.as_dict(),
        "model_info": dict(model_info),
        "input_channels": ["seismic", "input_lfm_log_ai", "valid_mask"],
        "output_semantics": "predicted_increment_log_ai",
        "normalization": dict(normalization),
        "increment_contract": validate_increment_contract(increment_contract).as_dict(),
        "training_sources": {key: dict(value) for key, value in training_sources.items()},
        "benchmark_identity": benchmark_identity,
        "synthetic_sampling_contract": synthetic_sampling_contract,
        "split_assignment_contract": split_contract | {"identity_sha256": split_identity},
        "normalization_identity": str(normalization.get("identity_sha256") or ""),
        "stage_lineage": [dict(value) for value in stage_lineage],
        "run_mode": config.run_mode,
        "development_limited": config.development_limited,
        "deployment_eligible": bool(stage_deployment_eligible),
        "deployment_eligibility_reason": str(stage_deployment_eligibility_reason),
        "stage_deployment_eligible": bool(stage_deployment_eligible),
        "stage_deployment_eligibility_reason": str(stage_deployment_eligibility_reason),
        "physics_closures": [dict(item) for item in physics_closures],
        "stage_loss_blocks": [
            {
                "block_id": str(value.get("block_id") or ""),
                "kind": str(value.get("kind") or ""),
                "source": str(value.get("source") or ""),
                "weight": float(value.get("weight", 0.0)),
                "update_interval": int(value.get("update_interval", 0)),
                "sampling": dict(value.get("sampling") or {}),
            }
            for value in stage_loss_blocks
        ],
        "sampling_statistics": dict(sampling_statistics or {}),
        "stage_selection_metric": metric_name,
        "sample_axis_contract": dict(sample_axis),
        "patch_deployment_contract": config.patching.as_dict() | {
            "axis_end_rule": "append_axis_length_minus_window",
            "short_axis_padding": "right",
            "padding_values": [0.0, 0.0, 0.0],
            "stitching": "uniform",
        },
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    if forward_model_inputs_path:
        payload["forward_model_inputs_path"] = forward_model_inputs_path
    return payload


def _evaluate(
    *, prepared: PreparedBlock, model: torch.nn.Module, device: torch.device,
    normalization: Mapping[str, Any], steps: int | None,
) -> dict[str, float]:
    if prepared.config["kind"] == "real_well_supervised":
        support = prepared.real_well_support
        held_out = support.samples[
            support.samples["supervision_role"].astype(str).str.startswith("held_out")
            & support.samples["valid_for_fit"].astype(bool)
        ]
        if held_out.empty:
            raise ValueError("real_well_supervised validation has no held-out well samples.")
        groups = [group.sort_values("sample_index") for _, group in held_out.groupby("well_name", sort=True)]
        with torch.no_grad():
            predictions = support.predictor.predict_delta_n_groups(
                model, groups, device=device, canonical_full_patch=support.canonical_full_patch,
            )
        losses = []
        for rows, prediction in zip(groups, predictions):
            target = torch.as_tensor(
                rows["well_target_increment_log_ai"].to_numpy(dtype=np.float32),
                device=device,
            )
            losses.append(torch.mean((prediction - target) ** 2))
        return {"mse": float(torch.stack(losses).mean().cpu())}
    assert prepared.validation_dataset is not None
    validation_views = dict(
        dict(prepared.config.get("validation") or {}).get("seismic_views") or {}
    )
    if (
        str(prepared.config.get("source_kind") or "") == "synthoseis_lite"
        and validation_views
        and str(prepared.config.get("kind") or "") in {"synthetic_supervised", "physics"}
    ):
        return _evaluate_synthetic_view_weighted(
            prepared=prepared,
            model=model,
            device=device,
            normalization=normalization,
            steps=steps,
            validation_views=validation_views,
        )
    batch_size = int(prepared.config["batch_size"])
    if steps is None:
        batches: Iterator[Any] = iter(DataLoader(
            prepared.validation_dataset, batch_size=batch_size, shuffle=False,
        ))
    else:
        cycler = DeterministicCycler(prepared.validation_dataset, batch_size=batch_size, seed=0)
        batches = (cycler.next() for _ in range(int(steps)))
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            loss, diagnostics = _block_loss(prepared, batch, model, device=device, normalization=normalization)
            rows.append({"total": float(loss.detach().cpu()), **diagnostics})
    if prepared.config["kind"] != "physics":
        count = float(np.sum([row["valid_sample_count"] for row in rows]))
        return {"mse": float(np.sum([row["mse"] * row["valid_sample_count"] for row in rows]) / count)}
    waveform_count = float(np.sum([row["valid_sample_count"] for row in rows]))
    increment_count = float(np.sum([row["increment_valid_sample_count"] for row in rows]))
    waveform_mse = float(np.sum([row["waveform"] * row["valid_sample_count"] for row in rows]) / waveform_count)
    increment_l2 = float(
        np.sum([row["increment_l2"] * row["increment_valid_sample_count"] for row in rows])
        / increment_count
    )
    return {
        "waveform_mse": waveform_mse,
        "increment_l2": increment_l2,
        "total": waveform_mse + float(prepared.config["increment_l2_weight"]) * increment_l2,
        "valid_sample_count": waveform_count,
        "skipped_item_count": float(np.sum([row["skipped_item_count"] for row in rows])),
    }


def _evaluate_synthetic_view_weighted(
    *,
    prepared: PreparedBlock,
    model: torch.nn.Module,
    device: torch.device,
    normalization: Mapping[str, Any],
    steps: int | None,
    validation_views: Mapping[str, Any],
) -> dict[str, float]:
    """Aggregate validation by explicit parent/view weights, never row counts."""
    dataset = prepared.validation_dataset
    if dataset is None:
        raise ValueError("Synthetic view validation requires a validation dataset.")
    parent_weights = {
        str(key): float(value)
        for key, value in dict(validation_views.get("parent_weights") or {}).items()
    }
    view_weights = {
        str(key): float(value)
        for key, value in dict(validation_views.get("view_weights") or {}).items()
    }
    if set(parent_weights) != {"base", "variant"}:
        raise ValueError("Validation parent weights must contain base and variant.")
    if not np.isclose(sum(parent_weights.values()), 1.0):
        raise ValueError("Validation parent weights must sum to one.")
    if parent_weights["variant"] > 0.0 and (
        not view_weights or not np.isclose(sum(view_weights.values()), 1.0)
    ):
        raise ValueError("Validation seismic view weights must sum to one.")
    metric_name = (
        "mse"
        if str(prepared.config.get("kind") or "") == "synthetic_supervised"
        else str(dict(prepared.config.get("validation") or {}).get("selection_metric") or "").split(".", 1)[-1]
    )
    if metric_name == "":
        metric_name = "total"
    values: dict[str, list[float]] = {}
    patch_values: dict[tuple[str, str], float] = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        dynamic = callable(getattr(dataset, "get_item_for_view", None))
        if dynamic:
            frame = dataset.frame.reset_index(drop=True)
            patch_indices = list(range(len(frame)))
            if steps is not None:
                patch_indices = patch_indices[: int(steps)]
            requested_views = ["base", *sorted(view_weights)]
            for view_id in requested_views:
                for row_index in patch_indices:
                    item = dataset.get_item_for_view(row_index, view_id)
                    batch = default_collate([item])
                    loss, diagnostics = _block_loss(
                        prepared,
                        batch,
                        model,
                        device=device,
                        normalization=normalization,
                    )
                    value = float(diagnostics.get(metric_name, float(loss.detach().cpu())))
                    patch_id = str(item["patch_id"])
                    values.setdefault(view_id, []).append(value)
                    patch_values[(patch_id, view_id)] = value
        else:
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            for index, batch in enumerate(loader):
                if steps is not None and index >= int(steps):
                    break
                loss, diagnostics = _block_loss(
                    prepared, batch, model, device=device, normalization=normalization
                )
                view_values = batch.get("view_id", ["base"])
                view_id = str(view_values[0] if isinstance(view_values, (list, tuple)) else view_values)
                value = float(
                    diagnostics.get(metric_name, float(loss.detach().cpu()))
                )
                values.setdefault(view_id, []).append(value)
                patch_values[(str(batch.get("patch_id", [""])[0]), view_id)] = value
    if was_training:
        model.train()
    if not values:
        raise ValueError("Synthetic view validation produced no batches.")
    means = {view: float(np.mean(items)) for view, items in values.items()}
    if parent_weights["base"] > 0.0 and "base" not in means:
        raise ValueError("Validation benchmark lacks the configured base view.")
    missing = sorted(view for view, weight in view_weights.items() if weight > 0.0 and view not in means)
    if missing:
        raise ValueError(f"Validation benchmark lacks configured seismic views: {missing}")
    weighted = parent_weights["base"] * means.get("base", 0.0)
    weighted += parent_weights["variant"] * sum(
        view_weights[view] * means[view] for view in view_weights
    )
    result: dict[str, float] = {
        metric_name: float(weighted),
        f"weighted_{metric_name}": float(weighted),
        "validation_parent_count": float(len({str(v) for v in dataset.frame["parent_realization_id"]})),
    }
    for view_id, value in sorted(means.items()):
        result[f"view_{view_id}_{metric_name}"] = float(value)
        result[f"view_{view_id}_count"] = float(len(values[view_id]))
    for view_id in sorted(view_weights):
        paired: list[float] = []
        for (patch_id, current_view), current in patch_values.items():
            if current_view != view_id:
                continue
            base_id = patch_id.replace(f"__view__{view_id}", "")
            base_value = patch_values.get((base_id, "base"))
            if base_value is not None:
                paired.append(current - base_value)
        if paired:
            result[f"paired_{view_id}_delta_{metric_name}"] = float(np.mean(paired))
            result[f"paired_{view_id}_count"] = float(len(paired))
    return result


def _collect_validation_increment_predictions(
    *, prepared: PreparedBlock, model: torch.nn.Module, device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Capture a deterministic validation prediction reference for physics drift QC."""
    if prepared.validation_dataset is None:
        return []
    was_training = model.training
    model.eval()
    result: list[tuple[np.ndarray, np.ndarray]] = []
    loader = DataLoader(
        prepared.validation_dataset,
        batch_size=int(prepared.config["batch_size"]),
        shuffle=False,
    )
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["input"].to(device)).detach().cpu().numpy()
            mask = batch["valid_mask"].detach().cpu().numpy() > 0.5
            result.append((prediction, mask))
    if was_training:
        model.train()
    return result


def _validation_increment_drift_mse(
    *, prepared: PreparedBlock, model: torch.nn.Module, device: torch.device,
    reference: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Measure prediction drift from the checkpoint entering a physics stage."""
    if prepared.validation_dataset is None or not reference:
        return float("nan")
    was_training = model.training
    model.eval()
    numerator = 0.0
    denominator = 0
    ref_index = 0
    loader = DataLoader(
        prepared.validation_dataset,
        batch_size=int(prepared.config["batch_size"]),
        shuffle=False,
    )
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["input"].to(device)).detach().cpu().numpy()
            mask = batch["valid_mask"].detach().cpu().numpy() > 0.5
            if ref_index >= len(reference):
                raise AssertionError("Physics drift reference and validation batches differ.")
            reference_prediction, reference_mask = reference[ref_index]
            if prediction.shape != reference_prediction.shape or not np.array_equal(mask, reference_mask):
                raise AssertionError("Physics drift reference and validation batch shapes differ.")
            finite = mask & np.isfinite(prediction) & np.isfinite(reference_prediction)
            if np.any(finite):
                difference = prediction[finite] - reference_prediction[finite]
                numerator += float(np.sum(difference * difference, dtype=np.float64))
                denominator += int(np.count_nonzero(finite))
            ref_index += 1
    if was_training:
        model.train()
    if ref_index != len(reference):
        raise AssertionError("Physics drift reference contains unused validation batches.")
    return float(numerator / denominator) if denominator else float("nan")


def run_experiment(
    *, config: ExperimentConfig, root: Path, output_dir: Path, logger: logging.Logger,
) -> dict[str, Any]:
    """Run a strict canonical-increment experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "experiment_manifest.json").exists() or (output_dir / "stages").exists():
        raise FileExistsError(f"GINN-v2 experiment outputs already exist: {output_dir}")
    device, device_metadata = resolve_device(config.device)
    model, info = build_model(**({
        "architecture_id": config.architecture.id,
        "hidden_channels": config.architecture.hidden_channels,
        "depth": config.architecture.depth,
        "lateral_kernel": config.architecture.lateral_kernel,
    }))
    model.to(device)
    run_started = time.perf_counter()
    logger.info(
        "experiment_start experiment_id=%s architecture=%s device=%s output_dir=%s",
        config.experiment_id,
        config.architecture.id,
        device,
        output_dir,
    )
    logger.info("device_metadata=%s", device_metadata)
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    patch_spec = PatchSpec(
        lateral_samples=config.patching.lateral_samples,
        twt_samples=config.patching.vertical_samples,
        lateral_stride=config.patching.lateral_stride,
        twt_stride=config.patching.vertical_stride,
    )
    synthetic: dict[str, tuple[SynthoseisBenchmark, Path]] = {}
    real_fields: dict[str, tuple[RealFieldVolume, RealFieldVolume]] = {}
    real_geometries: dict[str, Any] = {}
    source_contracts: dict[str, dict[str, Any]] = {}
    resolved_sources: dict[str, dict[str, Any]] = {}
    forward_model_inputs_by_source: dict[str, Path] = {}
    experiment_increment_contract = validate_increment_contract(
        config.increment_contract
    ).as_dict()
    for source_id, source in config.sources.items():
        if source["kind"] == "synthoseis_lite":
            raw_dir = source.get("benchmark_dir")
            benchmark_dir = _resolve_auto_benchmark(root) if str(raw_dir).casefold() == "auto" else _source_path(raw_dir, root)
            synthetic[source_id] = (SynthoseisBenchmark(benchmark_dir), benchmark_dir)
            manifest = synthetic[source_id][0].manifest
            domain = str(manifest.get("sample_domain") or "")
            source_contracts[source_id] = _synthetic_axis_contract(synthetic[source_id][0])
            benchmark_increment_contract = synthetic[source_id][0].increment_contract.as_dict()
            if benchmark_increment_contract != experiment_increment_contract:
                raise ValueError(
                    f"Synthetic source {source_id} increment_contract does not exactly match "
                    "ginn_v2.increment_contract."
                )
            forward_inputs_path = _benchmark_forward_model_inputs_path(
                synthetic[source_id][0], benchmark_dir,
            )
            resolved_sources[source_id] = {
                "kind": "synthoseis_lite",
                "benchmark_dir": repo_relative_path(benchmark_dir, root=root),
                "schema_version": manifest.get("schema_version"),
                "science_revision": manifest.get("science_revision"),
                "projection_contract_version": manifest.get("projection_contract_version"),
                "seismic_view_contract_version": manifest.get("seismic_view_contract_version"),
                "seismic_operator_contract_version": manifest.get("seismic_operator_contract_version"),
                "random_stream_contract_version": manifest.get("random_stream_contract_version"),
                "contract_fingerprint_sha256": manifest.get("contract_fingerprint_sha256", ""),
                "realization_index_path": repo_relative_path(
                    benchmark_dir / "realization_index.csv", root=root
                ),
                "seismic_view_index_path": repo_relative_path(
                    benchmark_dir / "seismic_view_index.csv", root=root
                ),
                "increment_contract": benchmark_increment_contract,
                "sample_axis_contract": dict(source_contracts[source_id]),
                "physics_target_variant": source.get("physics_target_variant", "model_consistent"),
                "configured_seismic_view_weights": dict(
                    next(
                        (block.get("sampling", {}).get("view_weights", {})
                         for stage in config.stages
                         for block in stage.get("loss_blocks", [])
                         if block.get("source") == source_id and block.get("sampling", {}).get("kind") == "parent_balanced_seismic_view"),
                        {}
                    )
                ),
                "configured_seismic_parent_weights": dict(
                    next(
                        (block.get("sampling", {}).get("parent_weights", {})
                         for stage in config.stages
                         for block in stage.get("loss_blocks", [])
                         if block.get("source") == source_id and block.get("sampling", {}).get("kind") == "parent_balanced_seismic_view"),
                        {}
                    )
                ),
                "configured_validation_seismic_view_weights": dict(
                    next(
                        (stage.get("validation", {}).get("seismic_views", {}).get("view_weights", {})
                         for stage in config.stages
                         if any(block.get("source") == source_id for block in stage.get("loss_blocks", []))
                         and stage.get("validation", {}).get("seismic_views")),
                        {}
                    )
                ),
                "materialized_view_ids": sorted(
                    str(value) for value in synthetic[source_id][0].views.get("view_id", pd.Series(dtype=str)).tolist()
                ),
                "materialized_view_spec_hashes": sorted(
                    str(value)
                    for value in synthetic[source_id][0].views.get(
                        "view_spec_sha256", pd.Series(dtype=str)
                    ).tolist()
                    if str(value).strip()
                ),
            }
            if forward_inputs_path is not None:
                forward_model_inputs_by_source[source_id] = forward_inputs_path
                resolved_sources[source_id]["forward_model_inputs_path"] = repo_relative_path(
                    forward_inputs_path, root=root,
                )
        elif source["kind"] == "real_field":
            raw_source = dict(source) | {
                "model_input_seismic_transform": "identity",
                "physics_target_seismic_transform": "identity",
            }
            real_fields[source_id] = _load_real_source(raw_source, root=root)
            real_cfg = _real_field_config(source, root=root, transform="identity")
            seismic_cfg = dict(real_cfg["real_field_inputs"])
            real_geometries[source_id] = open_survey(
                Path(str(seismic_cfg["seismic_file"])),
                str(seismic_cfg["seismic_type"]),
                segy_options=segy_options_from_config(dict(source.get("segy_options") or {})),
            ).line_geometry
            volume = real_fields[source_id][0]
            source_contracts[source_id] = _sample_axis_contract(
                volume.sample_axis.values,
                domain=volume.sample_axis.domain,
                unit=volume.sample_axis.unit,
                depth_basis=volume.depth_basis,
            )
            resolved_sources[source_id] = {
                "kind": "real_field",
                "lfm_run_dir": repo_relative_path(_source_path(source["lfm_run_dir"], root), root=root),
                "variant_id": source["variant_id"],
                "sample_axis_contract": dict(source_contracts[source_id]),
                "model_input_seismic_transform": source.get("model_input_seismic_transform", "identity"),
                "physics_target_seismic_transform": source.get("physics_target_seismic_transform", "identity"),
                "validation_split": dict(source["validation_split"]),
            }
            if source.get("forward_model_inputs_path") is not None:
                forward_inputs_path = _source_path(source["forward_model_inputs_path"], root).resolve()
                if not forward_inputs_path.is_file():
                    raise FileNotFoundError(f"Depth forward_model_inputs_path not found: {forward_inputs_path}")
                forward_model_inputs_by_source[source_id] = forward_inputs_path
                resolved_sources[source_id]["forward_model_inputs_path"] = repo_relative_path(
                    forward_inputs_path, root=root,
                )
            if source.get("well_control_run_dir") is not None:
                resolved_sources[source_id]["well_control_run_dir"] = repo_relative_path(
                    _source_path(source["well_control_run_dir"], root), root=root
                )
    for source_id, source in config.sources.items():
        if source["kind"] == "real_wells":
            source_contracts[source_id] = dict(source_contracts[str(source["field_source"])])
            resolved_sources[source_id] = {
                "kind": "real_wells",
                "field_source": source["field_source"],
                "well_control_run_dir": repo_relative_path(_source_path(source["well_control_run_dir"], root), root=root),
                "sample_axis_contract": dict(source_contracts[source_id]),
                "held_out_well": source["held_out_well"],
                "exclude_same_cluster": source.get("exclude_same_cluster", True),
                "supervision_excluded_well_names": list(
                    source.get("supervision_excluded_well_names") or []
                ),
            }
    for source_id, axis_contract in source_contracts.items():
        _validate_increment_contract_against_axis(
            experiment_increment_contract,
            axis_contract,
            label=f"source {source_id}",
        )
    logger.info(
        "sources_ready source_ids=%s consumed_source_ids=%s",
        sorted(resolved_sources),
        sorted({
            str(block["source"])
            for stage in config.stages
            for block in stage["loss_blocks"]
        }),
    )
    consumed_sources = {
        str(block["source"])
        for stage in config.stages
        for block in stage["loss_blocks"]
    }
    training_source_ids = consumed_sources | {config.normalization_reference}
    training_sources = {
        source_id: dict(resolved_sources[source_id])
        for source_id in sorted(training_source_ids)
    }
    experiment_axis_contract = validate_source_axis_contracts(source_contracts, consumed_sources)
    forward_model_inputs_paths = {
        path.resolve() for path in forward_model_inputs_by_source.values()
    }
    if len(forward_model_inputs_paths) > 1:
        raise ValueError(
            "A GINN-v2 experiment must use one depth forward_model_inputs contract."
        )
    forward_model_inputs_path = (
        repo_relative_path(next(iter(forward_model_inputs_paths)), root=root)
        if forward_model_inputs_paths else ""
    )
    split_assignment_paths: dict[str, Path] = {}
    split_assignments: dict[str, dict[str, str]] = {}
    for source_id, (benchmark, _) in synthetic.items():
        assignment_path = output_dir / f"split_assignment_{source_id}.csv"
        split_assignments[source_id] = _materialize_parent_split_assignment(
            benchmark, path=assignment_path
        )
        split_assignment_paths[source_id] = assignment_path
        resolved_sources[source_id]["split_assignment_path"] = repo_relative_path(
            assignment_path, root=root
        )
        if source_id in training_sources:
            training_sources[source_id] = dict(resolved_sources[source_id])
    reference_index: pd.DataFrame | None = None
    reference_rows: pd.DataFrame | None = None
    if config.normalization_reference in synthetic:
        reference_benchmark, _ = synthetic[config.normalization_reference]
        reference_split_assignment = split_assignments[config.normalization_reference]
        reference_index = build_patch_index(
            reference_benchmark,
            patch_spec=patch_spec,
            sample_kinds=_synthetic_sample_kinds(config.sources[config.normalization_reference]),
            min_valid_samples=1,
            max_patches=(
                int(config.sources[config.normalization_reference]["max_patches"])
                if config.sources[config.normalization_reference].get("max_patches") is not None
                else None
            ),
            split_assignment=reference_split_assignment,
        )
        input_reference_stats = compute_input_reference_stats(
            reference_benchmark, reference_index, input_name="seismic"
        )
    else:
        reference_volume = real_fields[config.normalization_reference][0]
        reference_rows = _real_patch_rows(
            reference_volume,
            patch_spec=patch_spec,
            min_valid_samples=1,
            validation=dict(config.sources[config.normalization_reference]["validation_split"]),
            geometry=real_geometries[config.normalization_reference],
        )
        train_inline_indices = np.sort(
            reference_rows.loc[reference_rows["split"] == "train", "inline_index"].unique().astype(int)
        )
        input_reference_stats = _input_reference_stats_for_real(
            reference_volume, train_inline_indices
        )
    for source_id, source in config.sources.items():
        if source["kind"] == "real_field":
            real_fields[source_id] = _load_real_source(
                source, root=root, reference_stats=input_reference_stats
            )
    if config.normalization_reference in synthetic:
        assert reference_index is not None
        normalization = _normalization_for_synthetic(reference_benchmark, reference_index)
    else:
        assert reference_rows is not None
        reference_volume = real_fields[config.normalization_reference][0]
        train_inline_indices = np.sort(
            reference_rows.loc[reference_rows["split"] == "train", "inline_index"].unique().astype(int)
        )
        normalization = _normalization_for_real(reference_volume, train_inline_indices)
    write_json(output_dir / "normalization.json", normalization)
    input_stats_path = output_dir / "input_reference_stats.json"
    write_json(input_stats_path, {"stats": input_reference_stats})
    reference_patch_count = len(reference_index) if reference_index is not None else len(reference_rows)
    logger.info(
        "normalization_ready reference_source=%s reference_patch_count=%d input_reference_stats_path=%s",
        config.normalization_reference,
        reference_patch_count,
        input_stats_path,
    )
    manifest_stages: list[dict[str, Any]] = []
    checkpoints: dict[tuple[str, str], Path] = {}
    sample_axis_contract: dict[str, Any] | None = dict(experiment_axis_contract)
    for stage_index, stage in enumerate(config.stages):
        stage_id = str(stage["stage_id"])
        stage_dir = output_dir / "stages" / stage_id
        stage_dir.mkdir(parents=True, exist_ok=False)
        stage_started = time.perf_counter()
        initialize = str(stage["initialize_from"])
        logger.info(
            "stage_start stage_id=%s stage_index=%d initialize_from=%s epochs=%d steps_per_epoch=%d loss_blocks=%s",
            stage_id,
            stage_index,
            initialize,
            int(stage["epochs"]),
            int(stage["steps_per_epoch"]),
            [str(block["block_id"]) for block in stage["loss_blocks"]],
        )
        if initialize == "zero":
            model.load_state_dict(initial_state)
        else:
            parent_stage, parent_kind = initialize.rsplit(".", 1)
            payload = torch.load(checkpoints[(parent_stage, parent_kind)], map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(stage["optimizer"]["learning_rate"]),
            weight_decay=float(stage["optimizer"]["weight_decay"]),
        )
        prepared_blocks: list[PreparedBlock] = []
        for block in stage["loss_blocks"]:
            block_started = time.perf_counter()
            source_id = str(block["source"])
            source = config.sources[source_id]
            cfg = dict(block) | {
                "source_kind": source["kind"],
                "validation": dict(stage.get("validation") or {}),
            }
            logger.info(
                "block_prepare_start stage_id=%s block_id=%s kind=%s source=%s",
                stage_id,
                block["block_id"],
                block["kind"],
                source_id,
            )
            if source["kind"] == "synthoseis_lite":
                benchmark, benchmark_dir = synthetic[source_id]
                wavelet = load_benchmark_wavelet(
                    benchmark_dir, device=device, artifact_root=root
                ) if block["kind"] == "physics" else None
                indices = _prepare_synthetic_indices(
                    benchmark=benchmark, source=source, patch_spec=patch_spec,
                    blocks=[block],
                    split_assignment=split_assignments[source_id],
                    logger=logger,
                    allow_smoke_validation_duplication=(
                        config.run_mode == "smoke"
                        and config.validation_semantics == "duplicated_training_patch"
                    ),
                )[str(block["block_id"])]
                train_dataset = PatchDataset(benchmark, indices, split="train", normalization=normalization)
                validation_dataset = PatchDataset(benchmark, indices, split="validation", normalization=normalization)
                prepared_blocks.append(PreparedBlock(cfg, train_dataset, validation_dataset, wavelet, sample_axis_contract=dict(source_contracts[source_id])))
                if sample_axis_contract is None:
                    domain = str(benchmark.manifest.get("sample_domain"))
                    sample_axis_contract = {"sample_domain": domain, "sample_unit": "s" if domain == "time" else "m", "depth_basis": benchmark.manifest.get("depth_basis")}
            elif source["kind"] == "real_field":
                model_volume, physics_volume = real_fields[source_id]
                wavelet = _load_real_wavelet(source, model_volume, root=root, device=device)
                rows = _real_patch_rows(
                    model_volume, patch_spec=patch_spec,
                    min_valid_samples=int(block["min_valid_samples"]),
                    validation=dict(source.get("validation_split") or {}),
                    geometry=real_geometries[source_id],
                )
                rows.to_csv(stage_dir / f"{block['block_id']}_patch_index.csv", index=False)
                train_dataset = RealFieldPatchDataset(model_volume, physics_volume, rows[rows["split"] == "train"], normalization, patch_spec)
                validation_dataset = RealFieldPatchDataset(model_volume, physics_volume, rows[rows["split"] == "validation"], normalization, patch_spec)
                prepared_blocks.append(PreparedBlock(cfg, train_dataset, validation_dataset, wavelet, sample_axis_contract=dict(source_contracts[source_id])))
                if sample_axis_contract is None:
                    sample_axis_contract = {"sample_domain": model_volume.sample_axis.domain, "sample_unit": model_volume.sample_axis.unit, "depth_basis": model_volume.depth_basis}
            else:
                field_source_id = str(source["field_source"])
                if field_source_id not in real_fields:
                    raise ValueError(f"real_wells source {source_id} references missing real_field source {field_source_id}.")
                support_cfg = {
                    "lfm_run_dir": config.sources[field_source_id]["lfm_run_dir"],
                    "variant_id": config.sources[field_source_id]["variant_id"],
                    "well_control_run_dir": source["well_control_run_dir"],
                    "held_out_well": source["held_out_well"],
                    "exclude_same_cluster": bool(source.get("exclude_same_cluster", True)),
                    "supervision_excluded_well_names": list(
                        source.get("supervision_excluded_well_names") or []
                    ),
                    "clusters_per_step": int(block["batch_size"]),
                    "cluster_radius_m": float(source.get("cluster_radius_m", 500.0)),
                    "diagnostic_max_hz": float(source.get("diagnostic_max_hz", 80.0)),
                    "reconstruction_tolerance_log_ai": float(source.get("reconstruction_tolerance_log_ai", 1e-5)),
                    "seismic_value_transform": str(config.sources[field_source_id].get("model_input_seismic_transform") or "identity"),
                    "lfm_value_transform": "identity",
                }
                support = prepare_real_well_supervised_support(
                    config=support_cfg, repo_root=root, output_dir=stage_dir,
                    normalization=normalization, patch_spec=config.patching.as_dict(),
                    input_reference_stats_path=input_stats_path, lambda_real_well_supervised=1.0,
                    seed=_seed(config.seed, stage_index, len(prepared_blocks), 0), logger=logger,
                )
                support.configure_model(receptive_field_lateral=info.lateral_receptive_field)
                prepared_blocks.append(PreparedBlock(cfg, None, None, real_well_support=support, sample_axis_contract=dict(source_contracts[source_id])))
                volume = real_fields[field_source_id][0]
                if sample_axis_contract is None:
                    sample_axis_contract = {"sample_domain": volume.sample_axis.domain, "sample_unit": volume.sample_axis.unit, "depth_basis": volume.depth_basis}
            prepared = prepared_blocks[-1]
            train_size = len(prepared.train_dataset) if prepared.train_dataset is not None else 0
            validation_size = len(prepared.validation_dataset) if prepared.validation_dataset is not None else 0
            logger.info(
                "block_ready stage_id=%s block_id=%s train_patches=%d validation_patches=%d real_well_support=%s elapsed_s=%.1f",
                stage_id,
                block["block_id"],
                train_size,
                validation_size,
                prepared.real_well_support is not None,
                time.perf_counter() - block_started,
            )
        stage_lineage = [
            {
                "stage_id": str(item["stage_id"]),
                "initialize_from": str(item.get("initialize_from") or ""),
                "checkpoints": dict(item.get("checkpoints") or {}),
                "loss_blocks": [
                    {
                        "block_id": str(block.get("block_id") or ""),
                        "kind": str(block.get("kind") or ""),
                        "source": str(block.get("source") or ""),
                    }
                    for block in list(item.get("loss_blocks") or [])
                ],
                "deployment_eligible": bool(item.get("deployment_eligible", False)),
                "deployment_eligibility_reason": str(
                    item.get("deployment_eligibility_reason") or ""
                ),
            }
            for item in manifest_stages
        ]
        stage_lineage.append({
            "stage_id": stage_id,
            "initialize_from": initialize,
            "loss_blocks": [
                {
                    "block_id": str(block.get("block_id") or ""),
                    "kind": str(block.get("kind") or ""),
                    "source": str(block.get("source") or ""),
                }
                for block in list(stage.get("loss_blocks") or [])
            ],
            "deployment_eligible": bool(stage.get("deployment_eligible", False)),
            "deployment_eligibility_reason": str(
                stage.get("deployment_eligibility_reason") or ""
            ),
        })
        physics_reference_predictions = {
            str(prepared.config["block_id"]): _collect_validation_increment_predictions(
                prepared=prepared, model=model, device=device
            )
            for prepared in prepared_blocks
            if str(prepared.config.get("kind") or "") == "physics"
        }
        physics_drift_after = float("nan")
        physics_drift_blocks: dict[str, float] = {}
        best_value = float("inf")
        best_epoch = 0
        history: list[dict[str, Any]] = []
        metric_name = str(stage["validation"]["selection_metric"])
        steps_per_epoch = int(stage["steps_per_epoch"])
        for epoch in range(1, int(stage["epochs"]) + 1):
            epoch_started = time.perf_counter()
            logger.info("epoch_start stage_id=%s epoch=%d/%d", stage_id, epoch, int(stage["epochs"]))
            cyclers = []
            for index, block in enumerate(prepared_blocks):
                if block.train_dataset is None:
                    cyclers.append(None)
                    continue
                batch_size = int(block.config["batch_size"])
                interval = int(block.config["update_interval"])
                num_batches = (steps_per_epoch + interval - 1) // interval
                sampling_kind = str(dict(block.config.get("sampling") or {}).get("kind") or "uniform_patch")
                seed = _seed(config.seed, stage_index, index, epoch)
                if sampling_kind == "parent_balanced_seismic_view":
                    if str(block.config.get("source_kind")) != "synthoseis_lite":
                        raise ValueError(
                            f"Loss block {block.config['block_id']} uses parent_balanced_seismic_view "
                            "with a non-synthetic source."
                        )
                    sampling = dict(block.config.get("sampling") or {})
                    cycler = DeterministicParentBalancedViewCycler(
                        block.train_dataset,
                        batch_size=batch_size,
                        num_batches=num_batches,
                        seed=seed,
                        parent_weights=dict(sampling["parent_weights"]),
                        view_weights=dict(sampling["view_weights"]),
                    )
                elif sampling_kind == "uniform_patch":
                    cycler = DeterministicCycler(
                        block.train_dataset,
                        batch_size=batch_size,
                        seed=seed,
                    )
                else:
                    raise ValueError(
                        f"Unsupported sampling.kind for block {block.config['block_id']}: "
                        f"{sampling_kind!r}"
                    )
                cyclers.append(cycler)
            model.train()
            totals: dict[str, list[dict[str, float]]] = {
                str(block.config["block_id"]): [] for block in prepared_blocks
            }
            batch_counts = {str(block.config["block_id"]): 0 for block in prepared_blocks}
            sampled_kind_counts: dict[str, dict[str, int]] = {
                str(block.config["block_id"]): {} for block in prepared_blocks
            }
            sampled_view_counts: dict[str, dict[str, int]] = {
                str(block.config["block_id"]): {} for block in prepared_blocks
            }
            progress_interval = max(1, steps_per_epoch // 10)
            last_combined = float("nan")
            for step in range(steps_per_epoch):
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                combined: torch.Tensor | None = None
                for block_index, prepared in enumerate(prepared_blocks):
                    if step % int(prepared.config["update_interval"]) != 0:
                        continue
                    cycler = cyclers[block_index]
                    batch = cycler.next() if cycler is not None else None
                    if cycler is not None and hasattr(cycler, "last_batch_counts"):
                        block_counts = sampled_kind_counts[str(prepared.config["block_id"])]
                        for sample_kind, count in cycler.last_batch_counts.items():
                            block_counts[sample_kind] = block_counts.get(sample_kind, 0) + int(count)
                        if hasattr(cycler, "last_batch_view_counts"):
                            view_counts = sampled_view_counts[str(prepared.config["block_id"])]
                            for view_id, count in cycler.last_batch_view_counts.items():
                                view_counts[view_id] = view_counts.get(view_id, 0) + int(count)
                    elif isinstance(batch, dict) and "sample_kind" in batch:
                        block_counts = sampled_kind_counts[str(prepared.config["block_id"])]
                        values = batch["sample_kind"]
                        if isinstance(values, str):
                            values = [values]
                        for sample_kind in values:
                            key = str(sample_kind)
                            block_counts[key] = block_counts.get(key, 0) + 1
                    loss, diagnostics = _block_loss(prepared, batch, model, device=device, normalization=normalization)
                    weighted = float(prepared.config["weight"]) * loss
                    combined = weighted if combined is None else combined + weighted
                    totals[str(prepared.config["block_id"])].append(
                        {"total": float(loss.detach().cpu()), **diagnostics}
                    )
                    batch_counts[str(prepared.config["block_id"])] += 1
                if combined is None or not bool(torch.isfinite(combined)):
                    raise FloatingPointError(f"Invalid combined loss at {stage_id} epoch={epoch} step={step}.")
                last_combined = float(combined.detach().cpu())
                combined.backward()
                optimizer.step()
                if step == 0 or (step + 1) % progress_interval == 0 or step + 1 == steps_per_epoch:
                    logger.info(
                        "step_progress stage_id=%s epoch=%d/%d step=%d/%d loss=%.6g step_elapsed_s=%.1f epoch_elapsed_s=%.1f",
                        stage_id,
                        epoch,
                        int(stage["epochs"]),
                        step + 1,
                        steps_per_epoch,
                        last_combined,
                        time.perf_counter() - step_started,
                        time.perf_counter() - epoch_started,
                    )
            validation_started = time.perf_counter()
            logger.info("validation_start stage_id=%s epoch=%d/%d", stage_id, epoch, int(stage["epochs"]))
            validation_values: dict[str, float] = {}
            for prepared in prepared_blocks:
                metrics = _evaluate(prepared=prepared, model=model, device=device, normalization=normalization, steps=None)
                for name, value in metrics.items():
                    validation_values[f"{prepared.config['block_id']}.{name}"] = value
            logger.info(
                "validation_end stage_id=%s epoch=%d/%d metrics=%s elapsed_s=%.1f",
                stage_id,
                epoch,
                int(stage["epochs"]),
                validation_values,
                time.perf_counter() - validation_started,
            )
            if physics_reference_predictions:
                physics_drift_blocks = {
                    block_id: _validation_increment_drift_mse(
                        prepared=prepared,
                        model=model,
                        device=device,
                        reference=physics_reference_predictions[block_id],
                    )
                    for block_id, prepared in (
                        (str(prepared.config["block_id"]), prepared)
                        for prepared in prepared_blocks
                        if str(prepared.config.get("kind") or "") == "physics"
                    )
                }
                finite_drifts = [
                    value for value in physics_drift_blocks.values() if np.isfinite(value)
                ]
                physics_drift_after = (
                    float(np.mean(finite_drifts)) if finite_drifts else float("nan")
                )
            selected = float(validation_values[metric_name])
            if not np.isfinite(selected):
                raise FloatingPointError(
                    f"Selection metric {metric_name} is non-finite at "
                    f"{stage_id} epoch={epoch}: {selected}"
                )
            training_values: dict[str, float] = {}
            for block_id, block_rows in totals.items():
                metric_keys = sorted({key for values in block_rows for key in values})
                for key in metric_keys:
                    values = [row[key] for row in block_rows if key in row]
                    aggregate = np.sum if key in {"valid_sample_count", "skipped_item_count"} else np.mean
                    training_values[f"{block_id}.train_{key}"] = float(aggregate(values))
                raw_ratio = training_values.get(
                    f"{block_id}.train_predicted_to_observed_rms_ratio"
                )
                if raw_ratio is not None and np.isfinite(raw_ratio) and not (
                    1.0e-3 <= raw_ratio <= 1.0e3
                ):
                    logger.warning(
                        "physics_raw_amplitude_scale_mismatch stage_id=%s epoch=%d "
                        "block_id=%s predicted_to_observed_rms_ratio=%.6g "
                        "waveform_standardization=masked_centered_rms interpretation=shape_only",
                        stage_id,
                        epoch,
                        block_id,
                        raw_ratio,
                    )
            row = {
                "epoch": epoch,
                "selection_metric": metric_name,
                "selection_metric_value": selected,
                **{f"{block_id}.batch_count": count for block_id, count in batch_counts.items()},
                **training_values,
                **validation_values,
            }
            if physics_reference_predictions:
                row["physics_increment_drift_before_mse"] = 0.0
                row["physics_increment_drift_after_mse"] = physics_drift_after
                for block_id, value in physics_drift_blocks.items():
                    row[f"{block_id}.increment_drift_after_mse"] = float(value)
            for block_id, kind_counts in sampled_kind_counts.items():
                for sample_kind in ("base", "seismic_view"):
                    row[f"{block_id}.sampled_{sample_kind}_count"] = int(kind_counts.get(sample_kind, 0))
                row[f"{block_id}.sampled_other_kind_count"] = int(
                    sum(count for sample_kind, count in kind_counts.items()
                        if sample_kind not in {"base", "seismic_view"})
                )
            for block_id, view_counts in sampled_view_counts.items():
                for view_id, count in view_counts.items():
                    row[f"{block_id}.sampled_view_{view_id}_count"] = int(count)
            history.append(row)
            logger.info(
                "epoch_end stage_id=%s epoch=%d/%d last_train_loss=%.6g selected_metric=%s selected_value=%.6g training=%s validation=%s sampled_kind_counts=%s elapsed_s=%.1f",
                stage_id,
                epoch,
                int(stage["epochs"]),
                last_combined,
                metric_name,
                selected,
                training_values,
                validation_values,
                sampled_kind_counts,
                time.perf_counter() - epoch_started,
            )
            if selected < best_value:
                best_value, best_epoch = selected, epoch
                path = stage_dir / "checkpoint_best.pt"
                torch.save(_checkpoint_payload(model=model, config=config, model_info=info.__dict__, normalization=normalization, stage_id=stage_id, kind="best", epoch=epoch, metric_name=metric_name, metric_value=selected, sample_axis=sample_axis_contract, increment_contract=experiment_increment_contract, training_sources=training_sources, stage_lineage=stage_lineage, stage_deployment_eligible=bool(stage.get("deployment_eligible", False)), stage_deployment_eligibility_reason=str(stage.get("deployment_eligibility_reason") or ""), physics_closures=list(stage.get("physics_closures") or []), stage_loss_blocks=list(stage.get("loss_blocks") or []), sampling_statistics={"epoch": epoch, "kind_counts": sampled_kind_counts, "view_counts": sampled_view_counts}, forward_model_inputs_path=forward_model_inputs_path), path)
                checkpoints[(stage_id, "best")] = path
                logger.info(
                    "checkpoint_best stage_id=%s epoch=%d metric=%s value=%.6g path=%s",
                    stage_id,
                    epoch,
                    metric_name,
                    selected,
                    path,
                )
        final_path = stage_dir / "checkpoint_final.pt"
        final_has_physics = bool(stage.get("physics_closures"))
        final_eligible = bool(stage.get("deployment_eligible", False)) and not final_has_physics
        final_reason = (
            "physics_stage_final_checkpoint_diagnostic_only"
            if final_has_physics
            else str(stage.get("deployment_eligibility_reason") or "")
        )
        torch.save(_checkpoint_payload(model=model, config=config, model_info=info.__dict__, normalization=normalization, stage_id=stage_id, kind="final", epoch=int(stage["epochs"]), metric_name=metric_name, metric_value=float(history[-1]["selection_metric_value"]), sample_axis=sample_axis_contract, increment_contract=experiment_increment_contract, training_sources=training_sources, stage_lineage=stage_lineage, stage_deployment_eligible=final_eligible, stage_deployment_eligibility_reason=final_reason, physics_closures=list(stage.get("physics_closures") or []), stage_loss_blocks=list(stage.get("loss_blocks") or []), sampling_statistics={"epoch": int(stage["epochs"]), "kind_counts": sampled_kind_counts, "view_counts": sampled_view_counts}, forward_model_inputs_path=forward_model_inputs_path), final_path)
        checkpoints[(stage_id, "final")] = final_path
        pd.DataFrame(history).to_csv(stage_dir / "training_history.csv", index=False)
        logger.info(
            "stage_end stage_id=%s best_epoch=%d best_value=%.6g final_checkpoint=%s elapsed_s=%.1f",
            stage_id,
            best_epoch,
            best_value,
            final_path,
            time.perf_counter() - stage_started,
        )
        manifest_stages.append({
            **stage, "best_epoch": best_epoch, "best_selection_metric_value": best_value,
            "checkpoints": {kind: repo_relative_path(checkpoints[(stage_id, kind)], root=root) for kind in ("best", "final")},
            "training_history": repo_relative_path(stage_dir / "training_history.csv", root=root),
            "physics_diagnostics": {
                "selection_metric": metric_name,
                "increment_drift_before_mse": 0.0,
                "increment_drift_after_mse": physics_drift_after,
                "increment_drift_by_block": dict(physics_drift_blocks),
            } if physics_reference_predictions else {},
        })
    deployment_path = checkpoints[(config.deployment_stage_id, config.deployment_checkpoint_kind)]
    deployment_stage_record = next(
        stage for stage in manifest_stages
        if str(stage.get("stage_id") or "") == config.deployment_stage_id
    )
    deployment_eligible = bool(deployment_stage_record.get("deployment_eligible", False))
    deployment_eligibility_reason = str(
        deployment_stage_record.get("deployment_eligibility_reason") or ""
    )
    deployment_stage_dir = output_dir / "stages" / config.deployment_stage_id
    deployment_eval_index, deployment_index_catalog, deployment_eval_source_id = (
        _deployment_patch_index_contract(
            stage=deployment_stage_record,
            stage_dir=deployment_stage_dir,
            sources=resolved_sources,
            root=root,
        )
    )
    manifest = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "status": "ok",
        "experiment_id": config.experiment_id,
        "run_mode": config.run_mode,
        "development_limited": config.development_limited,
        "validation_semantics": config.validation_semantics,
        "deployment_eligible": deployment_eligible,
        "deployment_eligibility_reason": deployment_eligibility_reason,
        "architecture": config.architecture.as_dict(),
        "model_info": info.__dict__,
        "normalization_reference": {"source": config.normalization_reference},
        "normalization": normalization,
        "normalization_path": repo_relative_path(output_dir / "normalization.json", root=root),
        "split_assignment": (
            repo_relative_path(
                split_assignment_paths[config.normalization_reference], root=root
            )
            if config.normalization_reference in split_assignment_paths
            else ""
        ),
        "split_assignments": {
            source_id: repo_relative_path(path, root=root)
            for source_id, path in sorted(split_assignment_paths.items())
        },
        "patch_catalog_role": "parent_window_without_view_replication",
        "sources": resolved_sources,
        "increment_contract": experiment_increment_contract,
        "patching": config.patching.as_dict(),
        "input_channels": ["seismic", "input_lfm_log_ai", "valid_mask"],
        "output_semantics": (
            "predicted_log_ai = input_lfm_log_ai + predicted_increment_log_ai"
        ),
        "sample_axis_contract": sample_axis_contract,
        "forward_model_inputs_path": forward_model_inputs_path,
        "stages": manifest_stages,
        "deployment_checkpoint": {
            "stage_id": config.deployment_stage_id,
            "kind": config.deployment_checkpoint_kind,
            "path": repo_relative_path(deployment_path, root=root),
            "eligible": deployment_eligible,
            "reason": deployment_eligibility_reason,
        },
        "device": device_metadata,
    }
    manifest["input_contracts"] = {
        source_id: {
            "kind": str(source.get("kind") or ""),
            "schema_version": source.get("schema_version"),
            "increment_contract": dict(source.get("increment_contract") or {}),
            "sample_axis_contract": dict(source.get("sample_axis_contract") or {}),
            "contract_fingerprint_sha256": str(
                source.get("contract_fingerprint_sha256") or ""
            ),
        }
        for source_id, source in resolved_sources.items()
        if source_id in training_source_ids
    }
    if deployment_index_catalog:
        manifest["patch_indices"] = deployment_index_catalog
    if deployment_eval_index is not None and deployment_eval_source_id is not None:
        selected_synthetic = synthetic[deployment_eval_source_id]
        manifest["benchmark_dir"] = repo_relative_path(selected_synthetic[1], root=root)
        manifest["patch_index"] = repo_relative_path(deployment_eval_index, root=root)
        manifest["deployment_evaluation_patch_index"] = manifest["patch_index"]
    manifest["contract_fingerprint_schema"] = CONTRACT_FINGERPRINT_SCHEMA
    manifest["contract_fingerprint_sha256"] = contract_fingerprint_sha256(
        contract_schema_version=EXPERIMENT_SCHEMA_VERSION,
        semantics={
            "experiment_id": config.experiment_id,
            "increment_contract": experiment_increment_contract,
            "sample_axis_contract": sample_axis_contract,
            "output_semantics": manifest["output_semantics"],
        },
        business_config={
            "architecture": config.architecture.as_dict(),
            "patching": config.patching.as_dict(),
            "run_mode": config.run_mode,
            "development_limited": config.development_limited,
            "validation_semantics": config.validation_semantics,
            "deployment_checkpoint": manifest["deployment_checkpoint"],
        },
        input_contracts={
            f"source:{source_id}": str(source["contract_fingerprint_sha256"])
            for source_id, source in resolved_sources.items()
            if str(source.get("contract_fingerprint_sha256") or "")
        },
        primary_artifacts={
            "deployment_checkpoint": deployment_path,
            "normalization": output_dir / "normalization.json",
            "input_reference_stats": input_stats_path,
        },
    )
    write_json(output_dir / "experiment_manifest.json", manifest)
    write_json(output_dir / "model_run_manifest.json", manifest)
    logger.info(
        "experiment_end experiment_id=%s deployment_checkpoint=%s elapsed_s=%.1f",
        config.experiment_id,
        deployment_path,
        time.perf_counter() - run_started,
    )
    return manifest


__all__ = [
    "DeterministicCycler", "DeterministicSampleKindCycler", "masked_centered_rms", "run_experiment",
    "validate_source_axis_contracts",
]
