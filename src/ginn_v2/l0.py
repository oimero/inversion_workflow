"""L0 synthetic + real-well delta-anchor validation."""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from cup.config.sources import load_summary
from cup.seismic.viz import plot_well_waveform_qc, waveform_qc_metrics
from cup.synthetic.dataset import SynthoseisBenchmark
from cup.synthetic.metrics import regression_metrics
from cup.utils.io import repo_relative_path, resolve_relative_path, sha256_file, write_json
from cup.well.anchor import build_well_anchor_samples, sample_volume_trilinear
from ginn_v2.data import PatchDataset, denormalize_delta
from ginn_v2.models import build_model
from ginn_v2.real_field import (
    RealFieldVolume,
    build_real_field_patch_index,
    forward_log_ai,
    load_real_field_volume,
    load_selected_wavelet,
)
from ginn_v2.training import masked_mse, resolve_device
from wtie.optimize.similarity import dynamic_normalized_xcorr, normalized_xcorr
from wtie.processing import grid


SCHEMA_VERSION = "l0_real_delta_anchor_v1"
FROZEN_MODEL_ROLE = "no_lateral"
FROZEN_LAMBDA_ANCHOR = 0.1
FROZEN_CLUSTERS_PER_STEP = 4
SYNTHETIC_SCOPES = {
    "validation_base": ("validation", "base"),
    "validation_mismatch": ("validation", "seismic_variant"),
    "test_base": ("test", "base"),
}


@dataclass(frozen=True)
class L0Sources:
    reference_dir: Path
    reference_manifest_path: Path
    reference_manifest: dict[str, Any]
    benchmark_dir: Path
    patch_index_path: Path
    lfm_dir: Path
    lfm_path: Path
    lfm_summary_path: Path
    lfm_summary: dict[str, Any]
    well_auto_tie_dir: Path
    well_inventory_file: Path
    seismic_path: Path


class BalancedAnchorSampler:
    """Deterministic cluster and within-cluster shuffled cycles."""

    def __init__(self, samples: pd.DataFrame, *, clusters_per_step: int, seed: int) -> None:
        valid = samples[samples["valid_for_fit"].astype(bool)].copy()
        if valid.empty:
            raise ValueError("Anchor sampler received no valid samples.")
        self.wells_by_cluster = {
            int(cluster): sorted(group["well_name"].astype(str).unique().tolist())
            for cluster, group in valid.groupby("spatial_cluster_id", sort=True)
        }
        self.clusters = sorted(self.wells_by_cluster)
        self.k = min(int(clusters_per_step), len(self.clusters))
        if self.k <= 0:
            raise ValueError("Anchor sampler requires at least one training cluster.")
        self.rng = np.random.default_rng(int(seed))
        self.cluster_queue: deque[int] = deque()
        self.well_queues: dict[int, deque[str]] = {cluster: deque() for cluster in self.clusters}
        self.counts: dict[tuple[int, str], int] = defaultdict(int)

    def _refill_clusters(self) -> None:
        order = np.asarray(self.clusters, dtype=np.int64)
        self.rng.shuffle(order)
        self.cluster_queue.extend(int(value) for value in order)

    def _next_clusters(self) -> list[int]:
        selected: list[int] = []
        deferred: list[int] = []
        while len(selected) < self.k:
            if not self.cluster_queue:
                self._refill_clusters()
            cluster = self.cluster_queue.popleft()
            if cluster in selected:
                deferred.append(cluster)
                continue
            selected.append(cluster)
        self.cluster_queue.extendleft(reversed(deferred))
        return selected

    def _next_well(self, cluster: int) -> str:
        queue = self.well_queues[cluster]
        if not queue:
            wells = np.asarray(self.wells_by_cluster[cluster], dtype=object)
            self.rng.shuffle(wells)
            queue.extend(str(value) for value in wells)
        return queue.popleft()

    def select(self) -> list[tuple[int, str]]:
        selected = [(cluster, self._next_well(cluster)) for cluster in self._next_clusters()]
        for item in selected:
            self.counts[item] += 1
        return selected


def resolve_holdout_selection(
    samples: pd.DataFrame,
    *,
    held_out_well: str,
    exclude_same_cluster: bool,
) -> tuple[int, list[str], pd.DataFrame]:
    """Resolve the auditable training exclusion set for one configured well."""

    available_wells = set(samples["well_name"].astype(str).unique())
    if held_out_well not in available_wells:
        raise ValueError(
            f"Configured held_out_well={held_out_well!r} is not a valid anchor well. "
            f"Available wells: {sorted(available_wells)}"
        )
    cluster_values = samples.loc[
        samples["well_name"].astype(str).eq(held_out_well),
        "spatial_cluster_id",
    ].astype(int).unique()
    if cluster_values.size != 1:
        raise ValueError(f"Held-out well must belong to exactly one spatial cluster: {held_out_well}")
    cluster = int(cluster_values[0])
    if exclude_same_cluster:
        excluded_wells = sorted(
            samples.loc[
                samples["spatial_cluster_id"].astype(int).eq(cluster),
                "well_name",
            ].astype(str).unique()
        )
    else:
        excluded_wells = [held_out_well]
    training = samples[~samples["well_name"].astype(str).isin(excluded_wells)].copy()
    if training.empty:
        raise ValueError("Manual holdout leaves no real-well anchors for training.")
    return cluster, excluded_wells, training


class DifferentiableWellPredictor:
    """Canonical no-lateral patch/stitch inference sampled on well trajectories."""

    def __init__(
        self,
        *,
        volume: RealFieldVolume,
        samples: pd.DataFrame,
        patch_spec: Mapping[str, Any],
        normalization: Mapping[str, Any],
        forward_batch_size: int = 256,
    ) -> None:
        self.volume = volume
        self.samples = samples
        self.patch_spec = dict(patch_spec)
        self.normalization = dict(normalization)
        self.forward_batch_size = int(forward_batch_size)
        self._patches: dict[int, list[Any]] = {}
        self._trace_task_cache: dict[tuple[int, int], list[tuple[Any, torch.Tensor]]] = {}
        self._geometry_cache: dict[tuple[int, ...], tuple[list[list[tuple[tuple[int, int], int, int, float, float]]], set[tuple[int, int]]]] = {}

    def prepare(self, wells: pd.DataFrame) -> dict[str, int]:
        """Precompute static geometry and normalized support tensors once."""

        all_nodes: set[tuple[int, int]] = set()
        for _, rows in wells.groupby("well_name", sort=True):
            _geometry, nodes = self._cached_point_geometry(rows)
            all_nodes.update(nodes)
        task_count = 0
        for node in sorted(all_nodes):
            task_count += len(self._trace_tasks_for_node(node))
        return {
            "n_prepared_wells": int(wells["well_name"].nunique()),
            "n_support_nodes": int(len(all_nodes)),
            "n_support_trace_patches": int(task_count),
        }

    def patches(self, inline_index: int) -> list[Any]:
        if inline_index not in self._patches:
            self._patches[inline_index] = build_real_field_patch_index(
                self.volume.valid_mask[inline_index],
                lateral_samples=int(self.patch_spec["lateral_samples"]),
                twt_samples=int(self.patch_spec["twt_samples"]),
                lateral_stride=int(self.patch_spec["lateral_stride"]),
                twt_stride=int(self.patch_spec["twt_stride"]),
                min_valid_fraction=float(self.patch_spec["min_valid_fraction"]),
            )
        return self._patches[inline_index]

    def predict_delta_n(
        self,
        model: torch.nn.Module,
        rows: pd.DataFrame,
        *,
        device: torch.device,
        canonical_full_patch: bool = False,
    ) -> torch.Tensor:
        return self.predict_delta_n_groups(
            model,
            [rows],
            device=device,
            canonical_full_patch=canonical_full_patch,
        )[0]

    def predict_delta_n_groups(
        self,
        model: torch.nn.Module,
        groups: Sequence[pd.DataFrame],
        *,
        device: torch.device,
        canonical_full_patch: bool = False,
    ) -> list[torch.Tensor]:
        """Predict several selected wells with one batched support forward."""

        if not groups or any(rows.empty for rows in groups):
            raise ValueError("Cannot predict an empty well sample group.")
        group_geometry = []
        nodes: set[tuple[int, int]] = set()
        for rows in groups:
            geometry, group_nodes = self._cached_point_geometry(rows)
            group_geometry.append(geometry)
            nodes.update(group_nodes)
        node_predictions = self._predict_nodes(
            model,
            nodes,
            device=device,
            canonical_full_patch=canonical_full_patch,
        )
        return [self._interpolate_geometry(geometry, node_predictions) for geometry in group_geometry]

    @staticmethod
    def _interpolate_geometry(
        point_geometry: list[list[tuple[tuple[int, int], int, int, float, float]]],
        node_predictions: Mapping[tuple[int, int], torch.Tensor],
    ) -> torch.Tensor:
        predictions: list[torch.Tensor] = []
        for geometry in point_geometry:
            value: torch.Tensor | None = None
            interpolation_weight = 0.0
            for node, twt0, twt1, time_weight, spatial_weight in geometry:
                trace = node_predictions[node]
                for twt_index, temporal_weight in (
                    (twt0, 1.0 - time_weight),
                    (twt1, time_weight),
                ):
                    weight = spatial_weight * temporal_weight
                    if weight <= 0.0 or not bool(torch.isfinite(trace[twt_index])):
                        continue
                    term = trace[twt_index] * weight
                    value = term if value is None else value + term
                    interpolation_weight += weight
            if value is None or interpolation_weight <= 0.0:
                raise ValueError("Canonical sparse predictor has an uncovered or non-finite well sample.")
            predictions.append(value / interpolation_weight)
        return torch.stack(predictions)

    def _cached_point_geometry(
        self,
        rows: pd.DataFrame,
    ) -> tuple[list[list[tuple[tuple[int, int], int, int, float, float]]], set[tuple[int, int]]]:
        if "_l0_row_id" not in rows:
            return self._point_geometry(rows.reset_index(drop=True))
        key = tuple(pd.to_numeric(rows["_l0_row_id"], errors="raise").astype(int).tolist())
        if key not in self._geometry_cache:
            self._geometry_cache[key] = self._point_geometry(rows.reset_index(drop=True))
        return self._geometry_cache[key]

    def _point_geometry(
        self,
        rows: pd.DataFrame,
    ) -> tuple[list[list[tuple[tuple[int, int], int, int, float, float]]], set[tuple[int, int]]]:
        axes = (self.volume.ilines, self.volume.xlines, self.volume.twt_s)
        columns = ("inline", "xline", "twt_s")
        fractional = []
        for axis, column in zip(axes, columns):
            values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float64)
            frac = np.interp(values, axis, np.arange(axis.size), left=np.nan, right=np.nan)
            fractional.append(frac)
        output: list[list[tuple[tuple[int, int], int, int, float, float]]] = []
        nodes: set[tuple[int, int]] = set()
        for point in range(len(rows)):
            if not all(np.isfinite(frac[point]) for frac in fractional):
                raise ValueError("Well sample is outside a canonical real-field axis.")
            positions = [float(frac[point]) for frac in fractional]
            lower = [min(int(np.floor(value)), axes[dim].size - 2) for dim, value in enumerate(positions)]
            weights = [value - index for value, index in zip(positions, lower)]
            terms: list[tuple[tuple[int, int], int, int, float, float]] = []
            for di in (0, 1):
                for dj in (0, 1):
                    spatial_weight = (weights[0] if di else 1.0 - weights[0]) * (
                        weights[1] if dj else 1.0 - weights[1]
                    )
                    if spatial_weight <= 0.0:
                        continue
                    node = (lower[0] + di, lower[1] + dj)
                    terms.append((node, lower[2], lower[2] + 1, weights[2], spatial_weight))
                    nodes.add(node)
            output.append(terms)
        return output, nodes

    def _predict_nodes(
        self,
        model: torch.nn.Module,
        nodes: set[tuple[int, int]],
        *,
        device: torch.device,
        canonical_full_patch: bool,
    ) -> dict[tuple[int, int], torch.Tensor]:
        if canonical_full_patch:
            return self._predict_nodes_full_patch(model, nodes, device=device)
        tasks: list[tuple[tuple[int, int], Any, torch.Tensor]] = []
        for node in sorted(nodes):
            for patch, tensor in self._trace_tasks_for_node(node):
                tasks.append((node, patch, tensor))
        if not tasks:
            raise ValueError("No canonical real-field patches cover the requested well support nodes.")
        outputs: list[torch.Tensor] = []
        for start in range(0, len(tasks), self.forward_batch_size):
            batch = torch.stack([task[2] for task in tasks[start : start + self.forward_batch_size]]).to(device)
            outputs.extend(model(batch)[:, 0, 0, :].unbind(0))
        return self._stitch_node_tasks(nodes, tasks, outputs, device=device)

    def _trace_tasks_for_node(self, node: tuple[int, int]) -> list[tuple[Any, torch.Tensor]]:
        if node in self._trace_task_cache:
            return self._trace_task_cache[node]
        inline_index, xline_index = node
        tasks: list[tuple[Any, torch.Tensor]] = []
        for patch in self.patches(inline_index):
            if not patch.lateral_start <= xline_index < patch.lateral_stop:
                continue
            sl = (inline_index, xline_index, slice(patch.twt_start, patch.twt_stop))
            valid = self.volume.valid_mask[sl]
            inputs = self._input_tensor(self.volume.seismic[sl], self.volume.lfm[sl], valid)
            tasks.append((patch, inputs[:, None, :].contiguous()))
        self._trace_task_cache[node] = tasks
        return tasks

    def _predict_nodes_full_patch(
        self,
        model: torch.nn.Module,
        nodes: set[tuple[int, int]],
        *,
        device: torch.device,
    ) -> dict[tuple[int, int], torch.Tensor]:
        patch_keys: dict[tuple[int, int, int], Any] = {}
        for inline_index, xline_index in nodes:
            for patch in self.patches(inline_index):
                if patch.lateral_start <= xline_index < patch.lateral_stop:
                    patch_keys[(inline_index, patch.lateral_start, patch.twt_start)] = patch
        patch_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        items = sorted(patch_keys.items())
        for start in range(0, len(items), max(1, self.forward_batch_size // int(self.patch_spec["lateral_samples"]))):
            chunk = items[start : start + max(1, self.forward_batch_size // int(self.patch_spec["lateral_samples"]))]
            tensors = []
            for (inline_index, _, _), patch in chunk:
                sl = (
                    inline_index,
                    slice(patch.lateral_start, patch.lateral_stop),
                    slice(patch.twt_start, patch.twt_stop),
                )
                tensors.append(self._input_tensor(self.volume.seismic[sl], self.volume.lfm[sl], self.volume.valid_mask[sl]))
            predicted = model(torch.stack(tensors).to(device))[:, 0]
            for (key, _patch), value in zip(chunk, predicted.unbind(0)):
                patch_outputs[key] = value
        tasks: list[tuple[tuple[int, int], Any, torch.Tensor]] = []
        outputs: list[torch.Tensor] = []
        for node in sorted(nodes):
            inline_index, xline_index = node
            for patch in self.patches(inline_index):
                if patch.lateral_start <= xline_index < patch.lateral_stop:
                    key = (inline_index, patch.lateral_start, patch.twt_start)
                    tasks.append((node, patch, torch.empty(0)))
                    outputs.append(patch_outputs[key][xline_index - patch.lateral_start])
        return self._stitch_node_tasks(nodes, tasks, outputs, device=device)

    def _stitch_node_tasks(
        self,
        nodes: set[tuple[int, int]],
        tasks: Sequence[tuple[tuple[int, int], Any, torch.Tensor]],
        outputs: Sequence[torch.Tensor],
        *,
        device: torch.device,
    ) -> dict[tuple[int, int], torch.Tensor]:
        nt = self.volume.twt_s.size
        by_node: dict[tuple[int, int], list[tuple[Any, torch.Tensor]]] = defaultdict(list)
        for (node, patch, _), output in zip(tasks, outputs):
            by_node[node].append((patch, output))
        stitched: dict[tuple[int, int], torch.Tensor] = {}
        for node in nodes:
            indices: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            weight = torch.zeros(nt, dtype=torch.float32, device=device)
            inline_index, xline_index = node
            for patch, output in by_node[node]:
                valid = torch.as_tensor(
                    self.volume.valid_mask[
                        inline_index,
                        xline_index,
                        patch.twt_start : patch.twt_stop,
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                destination = torch.arange(patch.twt_start, patch.twt_stop, device=device)[valid]
                indices.append(destination)
                values.append(output[valid])
                weight.index_add_(
                    0,
                    destination,
                    torch.ones(destination.shape, dtype=torch.float32, device=device),
                )
            if not indices:
                raise ValueError(f"No patch contributions for real-field node {node}.")
            total = torch.zeros(nt, dtype=torch.float32, device=device).index_add(
                0,
                torch.cat(indices),
                torch.cat(values),
            )
            stitched[node] = torch.where(weight > 0.0, total / weight, torch.full_like(total, torch.nan))
        return stitched

    def _input_tensor(self, seismic: np.ndarray, lfm: np.ndarray, valid: np.ndarray) -> torch.Tensor:
        seismic_n = (np.asarray(seismic, dtype=np.float32) - float(self.normalization["seismic"]["mean"])) / float(
            self.normalization["seismic"]["std"]
        )
        lfm_n = (np.asarray(lfm, dtype=np.float32) - float(self.normalization["lfm"]["mean"])) / float(
            self.normalization["lfm"]["std"]
        )
        mask = np.asarray(valid, dtype=bool)
        values = np.stack(
            [
                np.where(mask & np.isfinite(seismic_n), seismic_n, 0.0),
                np.where(mask & np.isfinite(lfm_n), lfm_n, 0.0),
                mask.astype(np.float32),
            ],
            axis=0,
        )
        return torch.from_numpy(values.astype(np.float32))


def run_l0(
    *,
    raw_config: Mapping[str, Any],
    repo_root: Path,
    data_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute the complete frozen L0 experiment."""

    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = _validate_config(raw_config)
    sources = _resolve_sources(cfg, raw_config=raw_config, repo_root=repo_root, data_root=data_root)
    manifest = sources.reference_manifest
    normalization = dict(manifest["normalization"])
    patch_spec = dict(manifest["patch_spec"])
    training_cfg = dict(manifest["training"])
    input_stats_path = resolve_relative_path(str(manifest["input_reference_stats"]), root=repo_root)
    with input_stats_path.open("r", encoding="utf-8") as handle:
        input_stats = json.load(handle)
    real_cfg = {
        "real_field_inputs": {
            "lfm_file": repo_relative_path(sources.lfm_path, root=repo_root),
            "seismic_file": str(dict(raw_config["seismic"])["file"]),
            "seismic_type": str(dict(raw_config["seismic"]).get("type", "zgy")),
            "segy_options": dict(dict(raw_config["seismic"]).get("segy_options") or {}),
            "seismic_value_transform": str(dict(cfg["real_field_inputs"])["seismic_value_transform"]),
            "lfm_value_transform": str(dict(cfg["real_field_inputs"])["lfm_value_transform"]),
            "seismic_reference_stats": dict(input_stats["stats"]),
            "seismic_reference_stats_file": repo_relative_path(input_stats_path, root=repo_root),
            "seismic_reference_stats_sha256": sha256_file(input_stats_path),
        },
        "volume": {},
    }
    volume = load_real_field_volume(config=real_cfg, root=repo_root, data_root=data_root)
    samples, sample_metadata = build_well_anchor_samples(
        well_auto_tie_dir=sources.well_auto_tie_dir,
        well_inventory_file=sources.well_inventory_file,
        lfm=volume.lfm,
        valid_mask=volume.valid_mask,
        ilines=volume.ilines,
        xlines=volume.xlines,
        twt_s=volume.twt_s,
        repo_root=repo_root,
        cluster_radius_m=float(dict(raw_config.get("spatial_debias") or {}).get("cluster_radius_m", 600.0)),
    )
    samples_path = output_dir / "l0_well_anchor_samples.csv"
    samples.to_csv(samples_path, index=False)
    valid_samples = samples[samples["valid_for_fit"].astype(bool)].copy()
    valid_samples["_l0_row_id"] = np.arange(len(valid_samples), dtype=np.int64)
    held_out_well = str(cfg["held_out_well"]).strip()
    exclude_same_cluster = bool(cfg.get("exclude_same_cluster", False))
    held_out_cluster, excluded_wells, training_samples = resolve_holdout_selection(
        valid_samples,
        held_out_well=held_out_well,
        exclude_same_cluster=exclude_same_cluster,
    )

    device, device_metadata = resolve_device(str(cfg["device"]))
    model_id = str(manifest["model_id"])
    reference_checkpoint = resolve_relative_path(str(manifest["checkpoint"]), root=repo_root)
    if sha256_file(reference_checkpoint) != str(manifest["checkpoint_sha256"]):
        raise ValueError("Reference checkpoint SHA-256 mismatch.")
    # The checkpoint is read only for architecture metadata. Its weights and
    # optimizer state never enter L0 initialization.
    reference_payload = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
    architecture = dict(reference_payload.get("architecture") or {})
    hidden_channels = int(architecture["hidden_channels"])
    depth = int(architecture["depth"])
    torch.manual_seed(int(training_cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(training_cfg["seed"]))
    initial_model, model_info = build_model(model_id, hidden_channels=hidden_channels, depth=depth)
    initial_state = deepcopy(initial_model.state_dict())
    initial_hash = _state_dict_sha256(initial_state)
    predictor = DifferentiableWellPredictor(
        volume=volume,
        samples=valid_samples,
        patch_spec=patch_spec,
        normalization=normalization,
    )
    precompute_summary = predictor.prepare(valid_samples)
    initial_model.to(device).eval()
    with torch.no_grad():
        sparse = predictor.predict_delta_n(initial_model, valid_samples, device=device)
        canonical = predictor.predict_delta_n(
            initial_model,
            valid_samples,
            device=device,
            canonical_full_patch=True,
        )
    reconstruction_error = float(torch.max(torch.abs(sparse - canonical)).cpu()) * float(normalization["delta"]["std"])
    tolerance = float(cfg["reconstruction_tolerance_log_ai"])
    if not np.isfinite(reconstruction_error) or reconstruction_error > tolerance:
        raise ValueError(
            "real_field_well_reconstruction_mismatch: "
            f"max_abs_log_ai={reconstruction_error:.9g}, tolerance={tolerance:.9g}"
        )
    del initial_model, sparse, canonical
    if device.type == "cuda":
        torch.cuda.empty_cache()

    benchmark = SynthoseisBenchmark(sources.benchmark_dir)
    patch_index = pd.read_csv(sources.patch_index_path)
    run_results: dict[str, dict[str, Any]] = {}
    control_dir = output_dir / "control"
    print("L0 training: paired synthetic control")
    run_results["control"] = _train_one(
        run_id="control",
        output_dir=control_dir,
        benchmark=benchmark,
        patch_index=patch_index,
        normalization=normalization,
        initial_state=initial_state,
        model_id=model_id,
        model_info=model_info.__dict__,
        hidden_channels=hidden_channels,
        depth=depth,
        training_cfg=training_cfg,
        lambda_physics=float(dict(manifest["loss"]).get("lambda_physics", 0.0)),
        lambda_anchor=0.0,
        predictor=None,
        anchor_samples=None,
        held_out_well=None,
        held_out_cluster=None,
        clusters_per_step=FROZEN_CLUSTERS_PER_STEP,
        device=device,
        device_metadata=device_metadata,
    )
    expected_sequence = run_results["control"]["synthetic_sequence_sha256"]
    print("L0 training: configured manual holdout")
    holdout_result = _train_one(
        run_id="holdout",
        output_dir=output_dir / "holdout",
        benchmark=benchmark,
        patch_index=patch_index,
        normalization=normalization,
        initial_state=initial_state,
        model_id=model_id,
        model_info=model_info.__dict__,
        hidden_channels=hidden_channels,
        depth=depth,
        training_cfg=training_cfg,
        lambda_physics=float(dict(manifest["loss"]).get("lambda_physics", 0.0)),
        lambda_anchor=FROZEN_LAMBDA_ANCHOR,
        predictor=predictor,
        anchor_samples=training_samples,
        held_out_well=held_out_well,
        held_out_cluster=held_out_cluster,
        clusters_per_step=FROZEN_CLUSTERS_PER_STEP,
        device=device,
        device_metadata=device_metadata,
    )
    if holdout_result["synthetic_sequence_sha256"] != expected_sequence:
        raise RuntimeError("Synthetic RNG sequence changed in the holdout run.")
    run_results["holdout"] = holdout_result

    history = pd.concat([result["history_frame"] for result in run_results.values()], ignore_index=True)
    history.to_csv(output_dir / "l0_training_history.csv", index=False)
    sampling_frames = [result["sampling_frame"] for result in run_results.values() if not result["sampling_frame"].empty]
    sampling = pd.concat(sampling_frames, ignore_index=True) if sampling_frames else pd.DataFrame()
    held_out_sampling_rows = []
    excluded_samples = valid_samples[valid_samples["well_name"].astype(str).isin(excluded_wells)]
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        for well_name, well_rows in excluded_samples.groupby("well_name", sort=True):
            held_out_sampling_rows.append(
                {
                    "run_id": "holdout",
                    "epoch": epoch,
                    "held_out_well": held_out_well,
                    "held_out_cluster_id": held_out_cluster,
                    "spatial_cluster_id": int(well_rows["spatial_cluster_id"].iloc[0]),
                    "well_name": well_name,
                    "excluded_from_anchor_training": True,
                    "selected_count": 0,
                    "n_valid_samples": int(len(well_rows)),
                }
            )
    if held_out_sampling_rows:
        sampling = pd.concat(
            [sampling, pd.DataFrame.from_records(held_out_sampling_rows)],
            ignore_index=True,
        )
    sampling.to_csv(output_dir / "l0_anchor_sampling_qc.csv", index=False)

    holdout_metrics, holdout_summary, synthetic_preservation, figure_status = _evaluate_runs(
        output_dir=output_dir,
        run_results=run_results,
        samples=valid_samples,
        held_out_well=held_out_well,
        excluded_wells=excluded_wells,
        held_out_cluster=held_out_cluster,
        predictor=predictor,
        normalization=normalization,
        benchmark=benchmark,
        patch_index=patch_index,
        volume=volume,
        sources=sources,
        cfg=cfg,
        repo_root=repo_root,
        device=device,
    )
    holdout_metrics.to_csv(output_dir / "l0_holdout_metrics.csv", index=False)
    holdout_summary.to_csv(output_dir / "l0_holdout_summary.csv", index=False)
    synthetic_preservation.to_csv(output_dir / "l0_synthetic_preservation.csv", index=False)
    decision, decision_table = _decide(
        holdout_summary,
        holdout_metrics,
        synthetic_preservation,
        cfg,
    )
    decision_table.to_csv(output_dir / "l0_decision_table.csv", index=False)
    output_files = [
        samples_path,
        output_dir / "l0_holdout_metrics.csv",
        output_dir / "l0_holdout_summary.csv",
        output_dir / "l0_training_history.csv",
        output_dir / "l0_anchor_sampling_qc.csv",
        output_dir / "l0_synthetic_preservation.csv",
        output_dir / "l0_decision_table.csv",
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": decision["status"],
        "eligible_for_l1": bool(decision["eligible_for_l1"]),
        "holdout_scope": "configured single-well anchor-label holdout",
        "held_out_well": held_out_well,
        "held_out_cluster_id": held_out_cluster,
        "exclude_same_cluster": exclude_same_cluster,
        "excluded_wells": excluded_wells,
        "same_cluster_training_leakage_risk": not exclude_same_cluster,
        "config": cfg,
        "initial_state_sha256": initial_hash,
        "model_id": model_id,
        "model_info": model_info.__dict__,
        "reference_training": {
            "run_dir": repo_relative_path(sources.reference_dir, root=repo_root),
            "manifest": repo_relative_path(sources.reference_manifest_path, root=repo_root),
            "manifest_sha256": sha256_file(sources.reference_manifest_path),
            "benchmark_dir": repo_relative_path(sources.benchmark_dir, root=repo_root),
            "patch_index": repo_relative_path(sources.patch_index_path, root=repo_root),
            "patch_index_sha256": sha256_file(sources.patch_index_path),
            "checkpoint_metadata_source": repo_relative_path(reference_checkpoint, root=repo_root),
            "checkpoint_metadata_source_sha256": sha256_file(reference_checkpoint),
            "checkpoint_weights_used_for_initialization": False,
        },
        "real_field_sources": _source_summary(sources, repo_root=repo_root),
        "real_field_input": {
            "metadata": volume.metadata,
            "shape": [int(value) for value in volume.lfm.shape],
            "valid_fraction": float(np.mean(volume.valid_mask)),
            "inline_range": [float(volume.ilines[0]), float(volume.ilines[-1])],
            "xline_range": [float(volume.xlines[0]), float(volume.xlines[-1])],
            "twt_range_s": [float(volume.twt_s[0]), float(volume.twt_s[-1])],
        },
        "well_anchor_samples": {**sample_metadata, "path": repo_relative_path(samples_path, root=repo_root), "sha256": sha256_file(samples_path)},
        "real_support_precompute": precompute_summary,
        "real_field_reconstruction_max_abs_log_ai": reconstruction_error,
        "device": device_metadata,
        "runs": {
            run_id: {
                "final_checkpoint": repo_relative_path(result["checkpoint"], root=repo_root),
                "final_checkpoint_sha256": sha256_file(result["checkpoint"]),
                "synthetic_sequence_sha256": result["synthetic_sequence_sha256"],
            }
            for run_id, result in run_results.items()
        },
        "decision": decision,
        "figure_status": figure_status,
        "outputs": {
            path.name: {
                "path": repo_relative_path(path, root=repo_root),
                "sha256": sha256_file(path),
            }
            for path in output_files
        },
    }
    write_json(output_dir / "l0_real_delta_anchor_summary.json", summary)
    return summary


def _train_one(
    *,
    run_id: str,
    output_dir: Path,
    benchmark: SynthoseisBenchmark,
    patch_index: pd.DataFrame,
    normalization: Mapping[str, Any],
    initial_state: Mapping[str, torch.Tensor],
    model_id: str,
    model_info: Mapping[str, Any],
    hidden_channels: int,
    depth: int,
    training_cfg: Mapping[str, Any],
    lambda_physics: float,
    lambda_anchor: float,
    predictor: DifferentiableWellPredictor | None,
    anchor_samples: pd.DataFrame | None,
    held_out_well: str | None,
    held_out_cluster: int | None,
    clusters_per_step: int,
    device: torch.device,
    device_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    seed = int(training_cfg["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model, _ = build_model(model_id, hidden_channels=hidden_channels, depth=depth)
    model.load_state_dict(initial_state)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg["learning_rate"]))
    train_ds = PatchDataset(benchmark, patch_index, split="train", normalization=normalization)
    kinds = train_ds.frame["sample_kind"].astype(str)
    counts = kinds.value_counts().to_dict()
    weights = torch.as_tensor([1.0 / float(counts[kind]) for kind in kinds], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    loader = DataLoader(train_ds, batch_size=int(training_cfg["batch_size"]), sampler=sampler, num_workers=0)
    anchor_sampler = None
    if lambda_anchor > 0.0:
        if predictor is None or anchor_samples is None:
            raise ValueError("Anchor run requires predictor and training samples.")
        anchor_sampler = BalancedAnchorSampler(
            anchor_samples,
            clusters_per_step=clusters_per_step,
            seed=seed + 104729 + int(held_out_cluster or 0),
        )
    history: list[dict[str, Any]] = []
    sequence = hashlib.sha256()
    sampling_rows: list[dict[str, Any]] = []
    fixed_synthetic_batch = next(iter(DataLoader(train_ds, batch_size=min(2, len(train_ds)), shuffle=False)))
    fixed_anchor = _fixed_anchor_rows(anchor_samples) if anchor_samples is not None else None
    audit = _gradient_audit(
        model=model,
        synthetic_batch=fixed_synthetic_batch,
        anchor_rows=fixed_anchor,
        predictor=predictor,
        normalization=normalization,
        lambda_anchor=lambda_anchor,
        device=device,
    )
    history.append({"run_id": run_id, "epoch": 0, "synthetic_loss": np.nan, "anchor_loss": np.nan, "weighted_anchor_loss": np.nan, "total_loss": np.nan, "synthetic_step_count": 0, "anchor_microbatch_count": 0, "anchor_time_block_count": 0, "anchor_covered_sample_count": 0, **audit, "is_final_checkpoint": False})
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        losses: list[tuple[float, float, float]] = []
        synthetic_steps = 0
        anchor_microbatches = 0
        anchor_covered_samples = 0
        before_counts = dict(anchor_sampler.counts) if anchor_sampler is not None else {}
        for batch in loader:
            synthetic_steps += 1
            for patch_id in batch["patch_id"]:
                sequence.update(str(patch_id).encode("utf-8"))
                sequence.update(b"\0")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["input"].to(device))
            synthetic_loss = masked_mse(prediction, batch["target_delta"].to(device), batch["valid_mask"].to(device))
            total = synthetic_loss
            if lambda_physics != 0.0:
                raise NotImplementedError("L0 currently requires the reference recipe to use lambda_physics=0.")
            anchor_value = torch.zeros((), device=device)
            if anchor_sampler is not None and predictor is not None and anchor_samples is not None:
                selected_anchors = anchor_sampler.select()
                selected_rows = [
                    anchor_samples[
                        anchor_samples["spatial_cluster_id"].astype(int).eq(cluster)
                        & anchor_samples["well_name"].astype(str).eq(well_name)
                    ]
                    for cluster, well_name in selected_anchors
                ]
                selected_predictions = predictor.predict_delta_n_groups(
                    model,
                    selected_rows,
                    device=device,
                )
                cluster_losses = []
                for rows, pred_delta_n in zip(selected_rows, selected_predictions):
                    target_delta = pd.to_numeric(rows["filtered_log_ai"], errors="coerce").to_numpy(dtype=np.float32) - pd.to_numeric(rows["lfm_log_ai"], errors="coerce").to_numpy(dtype=np.float32)
                    target_n = (target_delta - float(normalization["delta"]["mean"])) / float(normalization["delta"]["std"])
                    target = torch.as_tensor(target_n, dtype=torch.float32, device=device)
                    cluster_losses.append(torch.mean((pred_delta_n - target) ** 2))
                    anchor_microbatches += 1
                    anchor_covered_samples += int(len(rows))
                anchor_value = torch.stack(cluster_losses).mean()
                total = total + float(lambda_anchor) * anchor_value
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite L0 loss in {run_id}, epoch {epoch}.")
            total.backward()
            if any(parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()):
                raise FloatingPointError(f"Non-finite L0 gradient in {run_id}, epoch {epoch}.")
            optimizer.step()
            losses.append((float(synthetic_loss.detach().cpu()), float(anchor_value.detach().cpu()), float(total.detach().cpu())))
        audit = _gradient_audit(
            model=model,
            synthetic_batch=fixed_synthetic_batch,
            anchor_rows=fixed_anchor,
            predictor=predictor,
            normalization=normalization,
            lambda_anchor=lambda_anchor,
            device=device,
        )
        values = np.asarray(losses, dtype=np.float64)
        history.append(
            {
                "run_id": run_id,
                "epoch": epoch,
                "synthetic_loss": float(np.mean(values[:, 0])),
                "anchor_loss": float(np.mean(values[:, 1])) if lambda_anchor > 0.0 else np.nan,
                "weighted_anchor_loss": float(lambda_anchor * np.mean(values[:, 1])) if lambda_anchor > 0.0 else np.nan,
                "total_loss": float(np.mean(values[:, 2])),
                "synthetic_step_count": synthetic_steps,
                "anchor_microbatch_count": anchor_microbatches,
                "anchor_time_block_count": anchor_microbatches,
                "anchor_covered_sample_count": anchor_covered_samples,
                **audit,
                "is_final_checkpoint": epoch == int(training_cfg["epochs"]),
            }
        )
        if anchor_sampler is not None and anchor_samples is not None:
            for (cluster, well), total_count in sorted(anchor_sampler.counts.items()):
                selected_count = total_count - before_counts.get((cluster, well), 0)
                n_valid = int(
                    np.count_nonzero(
                        anchor_samples["spatial_cluster_id"].astype(int).eq(cluster)
                        & anchor_samples["well_name"].astype(str).eq(well)
                    )
                )
                sampling_rows.append(
                    {"run_id": run_id, "epoch": epoch, "held_out_well": held_out_well, "held_out_cluster_id": held_out_cluster, "spatial_cluster_id": cluster, "well_name": well, "excluded_from_anchor_training": False, "selected_count": selected_count, "n_valid_samples": n_valid}
                )
    if anchor_sampler is not None:
        missing = sorted(set(anchor_sampler.wells_by_cluster[cluster][index] for cluster in anchor_sampler.clusters for index in range(len(anchor_sampler.wells_by_cluster[cluster]))) - {well for (_cluster, well), count in anchor_sampler.counts.items() if count > 0})
        if missing:
            raise RuntimeError(f"anchor_sampling_incomplete in {run_id}: {missing}")
    checkpoint = output_dir / "final_checkpoint.pt"
    torch.save(
        {
            "schema_version": "l0_final_checkpoint_v1",
            "run_id": run_id,
            "model_id": model_id,
            "state_dict": model.state_dict(),
            "normalization": dict(normalization),
            "model_info": dict(model_info),
            "architecture": {"hidden_channels": hidden_channels, "depth": depth},
            "device": dict(device_metadata),
            "final_epoch": int(training_cfg["epochs"]),
            "held_out_cluster_id": held_out_cluster,
            "held_out_well": held_out_well,
            "lambda_anchor": float(lambda_anchor),
        },
        checkpoint,
    )
    return {
        "model": model,
        "checkpoint": checkpoint,
        "history_frame": pd.DataFrame.from_records(history),
        "sampling_frame": pd.DataFrame.from_records(sampling_rows),
        "synthetic_sequence_sha256": sequence.hexdigest(),
    }


def _fixed_anchor_rows(samples: pd.DataFrame | None) -> pd.DataFrame | None:
    if samples is None or samples.empty:
        return None
    parts = []
    for _, cluster in samples.groupby("spatial_cluster_id", sort=True):
        well = sorted(cluster["well_name"].astype(str).unique())[0]
        parts.append(cluster[cluster["well_name"].astype(str).eq(well)])
    return pd.concat(parts, ignore_index=True)


def _gradient_audit(
    *,
    model: torch.nn.Module,
    synthetic_batch: Mapping[str, Any],
    anchor_rows: pd.DataFrame | None,
    predictor: DifferentiableWellPredictor | None,
    normalization: Mapping[str, Any],
    lambda_anchor: float,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    pred = model(synthetic_batch["input"].to(device))
    synth_loss = masked_mse(pred, synthetic_batch["target_delta"].to(device), synthetic_batch["valid_mask"].to(device))
    synth_grads = torch.autograd.grad(synth_loss, tuple(model.parameters()), retain_graph=False, allow_unused=True)
    synthetic_norm = _grad_norm(synth_grads)
    result = {
        "synthetic_grad_norm": synthetic_norm,
        "anchor_grad_norm": np.nan,
        "weighted_anchor_grad_norm": np.nan,
        "anchor_to_synthetic_grad_norm_ratio": np.nan,
    }
    if anchor_rows is not None and predictor is not None and lambda_anchor > 0.0:
        pred_anchor = predictor.predict_delta_n(model, anchor_rows, device=device)
        target_delta = pd.to_numeric(anchor_rows["filtered_log_ai"], errors="coerce").to_numpy(dtype=np.float32) - pd.to_numeric(anchor_rows["lfm_log_ai"], errors="coerce").to_numpy(dtype=np.float32)
        target_n = (target_delta - float(normalization["delta"]["mean"])) / float(normalization["delta"]["std"])
        anchor_loss = torch.mean((pred_anchor - torch.as_tensor(target_n, device=device)) ** 2)
        anchor_grads = torch.autograd.grad(anchor_loss, tuple(model.parameters()), retain_graph=False, allow_unused=True)
        anchor_norm = _grad_norm(anchor_grads)
        result.update(
            {
                "anchor_grad_norm": anchor_norm,
                "weighted_anchor_grad_norm": float(lambda_anchor * anchor_norm),
                "anchor_to_synthetic_grad_norm_ratio": float(lambda_anchor * anchor_norm / synthetic_norm) if synthetic_norm > 0.0 else np.nan,
            }
        )
    model.train(was_training)
    return result


def _grad_norm(grads: Iterable[torch.Tensor | None]) -> float:
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(torch.sum(grad.detach() ** 2).cpu())
    return float(np.sqrt(total))


def _evaluate_runs(
    *,
    output_dir: Path,
    run_results: Mapping[str, Mapping[str, Any]],
    samples: pd.DataFrame,
    held_out_well: str,
    excluded_wells: Sequence[str],
    held_out_cluster: int,
    predictor: DifferentiableWellPredictor,
    normalization: Mapping[str, Any],
    benchmark: SynthoseisBenchmark,
    patch_index: pd.DataFrame,
    volume: RealFieldVolume,
    sources: L0Sources,
    cfg: Mapping[str, Any],
    repo_root: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    held = samples[samples["well_name"].astype(str).isin(excluded_wells)].copy()
    if held.empty:
        raise ValueError("No valid samples are available for configured holdout evaluation.")
    control_model = run_results["control"]["model"]
    holdout_model = run_results["holdout"]["model"]
    control_model.eval()
    holdout_model.eval()
    with torch.no_grad():
        control_n = predictor.predict_delta_n(control_model, held, device=device).cpu().numpy()
        anchor_n = predictor.predict_delta_n(holdout_model, held, device=device).cpu().numpy()
    delta_mean = float(normalization["delta"]["mean"])
    delta_std = float(normalization["delta"]["std"])
    held["control_delta"] = control_n * delta_std + delta_mean
    held["anchor_delta"] = anchor_n * delta_std + delta_mean

    synthetic_rows: list[dict[str, Any]] = []
    control_synthetic = _synthetic_metrics(control_model, benchmark, patch_index, normalization, device=device)
    for scope, metrics in control_synthetic.items():
        synthetic_rows.append({"run_id": "control", "held_out_cluster_id": np.nan, "scope": scope, **metrics, "rmse_relative_increase": 0.0, "nrmse_relative_increase": 0.0, "error_relative_increase": 0.0, "corr_drop": 0.0, "warning": False, "catastrophic": False})
    holdout_synthetic = _synthetic_metrics(holdout_model, benchmark, patch_index, normalization, device=device)
    thresholds = dict(cfg["thresholds"])
    for scope, metrics in holdout_synthetic.items():
        base = control_synthetic[scope]
        rmse_relative = (metrics["rmse"] - base["rmse"]) / base["rmse"] if base["rmse"] > 0.0 else np.nan
        nrmse_relative = (metrics["nrmse"] - base["nrmse"]) / base["nrmse"] if base["nrmse"] > 0.0 else np.nan
        relative = float(np.nanmax([rmse_relative, nrmse_relative]))
        corr_drop = base["corr"] - metrics["corr"]
        warning = bool(
            relative > float(thresholds["synthetic_warning_error_relative_increase"])
            or corr_drop > float(thresholds["synthetic_warning_corr_drop"])
        )
        catastrophic = bool(
            relative > float(thresholds["maximum_synthetic_error_relative_increase"])
            or corr_drop > float(thresholds["maximum_synthetic_corr_drop"])
        )
        synthetic_rows.append(
            {
                "run_id": "holdout",
                "held_out_well": held_out_well,
                "held_out_cluster_id": held_out_cluster,
                "scope": scope,
                **metrics,
                "rmse_relative_increase": rmse_relative,
                "nrmse_relative_increase": nrmse_relative,
                "error_relative_increase": relative,
                "corr_drop": corr_drop,
                "warning": warning,
                "catastrophic": catastrophic,
            }
        )

    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    figure_status: dict[str, Any] = {}
    wavelet_dir = resolve_relative_path(str(json.loads((sources.benchmark_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))["source_runs"]["wavelet_generation_dir"]), root=repo_root)
    wavelet, _wavelet_meta = load_selected_wavelet(wavelet_dir)
    tie_metrics = pd.read_csv(sources.well_auto_tie_dir / "well_tie_metrics.csv").set_index("well_name", drop=False)
    for well_name, well in held.groupby("well_name", sort=True):
            cluster = int(well["spatial_cluster_id"].iloc[0])
            well = well.sort_values("sample_index").copy()
            target_ai = well["filtered_log_ai"].to_numpy(dtype=np.float64)
            lfm_ai = well["lfm_log_ai"].to_numpy(dtype=np.float64)
            target_delta = target_ai - lfm_ai
            twt = well["twt_s"].to_numpy(dtype=np.float64)
            control_pred_delta = well["control_delta"].to_numpy(dtype=np.float64)
            anchor_pred_delta = well["anchor_delta"].to_numpy(dtype=np.float64)
            control_metrics = _well_metrics(target_ai, target_delta, lfm_ai + control_pred_delta, control_pred_delta, twt)
            anchor_metrics = _well_metrics(target_ai, target_delta, lfm_ai + anchor_pred_delta, anchor_pred_delta, twt)
            control_metrics.update(
                _well_band_metrics(
                    target_ai=target_ai,
                    target_delta=target_delta,
                    pred_ai=lfm_ai + control_pred_delta,
                    pred_delta=control_pred_delta,
                    twt=twt,
                    diagnostic_max_hz=float(cfg["diagnostic_max_hz"]),
                )
            )
            anchor_metrics.update(
                _well_band_metrics(
                    target_ai=target_ai,
                    target_delta=target_delta,
                    pred_ai=lfm_ai + anchor_pred_delta,
                    pred_delta=anchor_pred_delta,
                    twt=twt,
                    diagnostic_max_hz=float(cfg["diagnostic_max_hz"]),
                )
            )
            lfm_metrics = _basic_metrics(target_ai, lfm_ai)
            gains = {
                "delta_corr_gain": anchor_metrics["delta_corr"] - control_metrics["delta_corr"],
                "delta_energy_error_change": anchor_metrics["delta_energy_error"] - control_metrics["delta_energy_error"],
                "gradient_energy_error_change": anchor_metrics["gradient_energy_error"] - control_metrics["gradient_energy_error"],
                "full_ai_corr_gain": anchor_metrics["full_ai_corr"] - control_metrics["full_ai_corr"],
                "full_ai_rmse_delta": anchor_metrics["full_ai_rmse"] - control_metrics["full_ai_rmse"],
            }
            good = bool(control_metrics["full_ai_corr"] > lfm_metrics["corr"] and control_metrics["full_ai_rmse"] < lfm_metrics["rmse"])
            rmse_relative = (anchor_metrics["full_ai_rmse"] - control_metrics["full_ai_rmse"]) / control_metrics["full_ai_rmse"] if control_metrics["full_ai_rmse"] > 0.0 else np.inf
            good_pass = not good or not (
                -gains["full_ai_corr_gain"] > float(thresholds["maximum_good_well_corr_drop"])
                or rmse_relative > float(thresholds["maximum_good_well_rmse_relative_increase"])
                or -gains["delta_corr_gain"] > float(thresholds["maximum_good_well_delta_corr_drop"])
                or gains["delta_energy_error_change"] > 0.0
                or gains["gradient_energy_error_change"] > 0.0
            )
            figure_paths, qc_status, waveform_metrics = _write_well_figures(
                output_dir=output_dir,
                cluster=cluster,
                well_name=str(well_name),
                well=well,
                target_ai=target_ai,
                lfm_ai=lfm_ai,
                control_delta=control_pred_delta,
                anchor_delta=anchor_pred_delta,
                volume=volume,
                wavelet=wavelet,
                tie=tie_metrics.loc[str(well_name)],
                repo_root=repo_root,
            )
            figure_status[str(well_name)] = qc_status
            for variant, metrics in (("control", control_metrics), ("anchor", anchor_metrics)):
                rows.append(
                    {
                        "held_out_cluster_id": cluster,
                        "configured_held_out_well": held_out_well,
                        "is_primary_holdout": str(well_name) == held_out_well,
                        "excluded_from_anchor_training": True,
                        "well_name": well_name,
                        "model_variant": variant,
                        **metrics,
                        **gains,
                        "lfm_full_ai_corr": lfm_metrics["corr"],
                        "lfm_full_ai_rmse": lfm_metrics["rmse"],
                        "control_good_well": good,
                        "good_well_preservation_passed": good_pass,
                        **waveform_metrics.get(variant, {}),
                        **figure_paths,
                        "status": "ok" if qc_status == "ok" else qc_status,
                        "reason": "" if qc_status == "ok" else qc_status,
                    }
                )
            formal_columns = [
                "delta_corr_gain",
                "full_ai_corr_gain",
                "full_ai_rmse_delta",
                "delta_energy_error_change",
                "gradient_energy_error_change",
            ]
            formal = {column: float(gains[column]) for column in formal_columns}
            formal_valid = all(np.isfinite(value) for value in formal.values())
            summary_rows.append(
                {
                    "configured_held_out_well": held_out_well,
                    "well_name": well_name,
                    "is_primary_holdout": str(well_name) == held_out_well,
                    "spatial_cluster_id": cluster,
                    **formal,
                    "good_well_preservation_passed": good_pass,
                    "status": "ok" if formal_valid else "invalid_held_out_metric",
                }
            )
    return (
        pd.DataFrame.from_records(rows),
        pd.DataFrame.from_records(summary_rows),
        pd.DataFrame.from_records(synthetic_rows),
        figure_status,
    )


def _synthetic_metrics(
    model: torch.nn.Module,
    benchmark: SynthoseisBenchmark,
    patch_index: pd.DataFrame,
    normalization: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    model.eval()
    for scope, (split, kind) in SYNTHETIC_SCOPES.items():
        selected = patch_index[
            patch_index["split"].astype(str).eq(split)
            & patch_index["sample_kind"].astype(str).eq(kind)
        ].copy()
        if selected.empty:
            raise ValueError(f"incomplete_synthetic_evidence: missing scope {scope}")
        dataset = PatchDataset(benchmark, selected, split=split, normalization=normalization)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        patch_metrics: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                pred_n = model(batch["input"].to(device))[:, 0].cpu().numpy()
                pred_delta = denormalize_delta(pred_n, normalization)
                lfm = batch["lfm"][:, 0].numpy()
                target = batch["target_log_ai"][:, 0].numpy()
                valid = batch["valid_mask"][:, 0].numpy() > 0.5
                for index in range(target.shape[0]):
                    patch_metrics.append(
                        regression_metrics(
                            target[index],
                            lfm[index] + pred_delta[index],
                            valid_mask=valid[index],
                        )
                    )
        frame = pd.DataFrame.from_records(patch_metrics)
        ok = frame[frame["status"].eq("ok")]
        if ok.empty:
            raise ValueError(f"Synthetic scope has no valid patch metrics: {scope}")
        result[scope] = {
            "n_valid": int(ok["n_valid"].sum()),
            "n_patches": int(len(ok)),
            "rmse": float(ok["rmse"].mean()),
            "nrmse": float(ok["nrmse"].mean()),
            "corr": float(ok["corr"].median()),
        }
    return result


def _well_metrics(target_ai: np.ndarray, target_delta: np.ndarray, pred_ai: np.ndarray, pred_delta: np.ndarray, twt: np.ndarray) -> dict[str, float | int]:
    delta = _basic_metrics(target_delta, pred_delta)
    full = _basic_metrics(target_ai, pred_ai)
    target_delta_rms = _rms(target_delta)
    pred_delta_rms = _rms(pred_delta)
    target_gradient, pred_gradient = _paired_gradients(target_delta, pred_delta, twt)
    target_gradient_rms = _rms(target_gradient)
    gradient_rms = _rms(pred_gradient)
    return {
        "n_valid": int(delta["n_valid"]),
        "delta_corr": delta["corr"], "delta_rmse": delta["rmse"], "delta_bias": delta["bias"],
        "full_ai_corr": full["corr"], "full_ai_rmse": full["rmse"], "full_ai_bias": full["bias"],
        "delta_rms": pred_delta_rms, "target_delta_rms": target_delta_rms,
        "delta_energy_error": _energy_error(pred_delta_rms, target_delta_rms),
        "gradient_rms": gradient_rms, "target_gradient_rms": target_gradient_rms,
        "gradient_energy_error": _energy_error(gradient_rms, target_gradient_rms),
    }


def _basic_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    n = int(np.count_nonzero(valid))
    if n < 2:
        return {"n_valid": n, "corr": np.nan, "rmse": np.nan, "bias": np.nan}
    x, y = np.asarray(target)[valid], np.asarray(prediction)[valid]
    residual = y - x
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0.0 and np.std(y) > 0.0 else np.nan
    return {"n_valid": n, "corr": corr, "rmse": float(np.sqrt(np.mean(residual ** 2))), "bias": float(np.mean(residual))}


def _well_band_metrics(
    *,
    target_ai: np.ndarray,
    target_delta: np.ndarray,
    pred_ai: np.ndarray,
    pred_delta: np.ndarray,
    twt: np.ndarray,
    diagnostic_max_hz: float,
) -> dict[str, float | int]:
    dt = float(np.nanmedian(np.diff(twt)))
    if not np.isfinite(dt) or dt <= 0.0:
        return {}
    high = min(float(diagnostic_max_hz), 0.45 * (0.5 / dt))
    bands = (
        ("lowfreq", 0.0, 0.2 * high),
        ("observable_band", 0.2 * high, 0.4 * high),
        ("highfreq_or_nullspace", 0.4 * high, high),
    )
    output: dict[str, float | int] = {}
    for name, low_hz, high_hz in bands:
        for prefix, target, prediction in (
            ("delta", target_delta, pred_delta),
            ("full_ai", target_ai, pred_ai),
        ):
            target_band, prediction_band = _fft_band_pair(target, prediction, dt=dt, low_hz=low_hz, high_hz=high_hz)
            metrics = _basic_metrics(target_band, prediction_band)
            output[f"{prefix}_{name}_n_valid"] = metrics["n_valid"]
            output[f"{prefix}_{name}_corr"] = metrics["corr"]
            output[f"{prefix}_{name}_rmse"] = metrics["rmse"]
    return output


def _fft_band_pair(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    dt: float,
    low_hz: float,
    high_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    run = _largest_true_run(valid)
    if run is None or run[1] - run[0] < 8:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    sl = slice(*run)
    output = []
    for values in (np.asarray(target, dtype=np.float64)[sl], np.asarray(prediction, dtype=np.float64)[sl]):
        centered = values - float(np.mean(values))
        spectrum = np.fft.rfft(centered)
        frequency = np.fft.rfftfreq(values.size, d=dt)
        keep = (frequency >= float(low_hz)) & (frequency < float(high_hz))
        if low_hz <= 0.0:
            keep[0] = True
            spectrum[0] = np.fft.rfft(values)[0]
        output.append(np.fft.irfft(np.where(keep, spectrum, 0.0), n=values.size))
    return output[0], output[1]


def _paired_gradients(target: np.ndarray, prediction: np.ndarray, twt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(twt)
    nominal = float(np.nanmedian(dt))
    valid = np.isfinite(target[:-1]) & np.isfinite(target[1:]) & np.isfinite(prediction[:-1]) & np.isfinite(prediction[1:]) & np.isfinite(dt) & (dt > 0.0) & (dt <= 1.5 * nominal)
    return np.diff(target)[valid] / dt[valid], np.diff(prediction)[valid] / dt[valid]


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else np.nan


def _energy_error(prediction_rms: float, target_rms: float) -> float:
    if not (np.isfinite(prediction_rms) and np.isfinite(target_rms) and prediction_rms > 0.0 and target_rms > 0.0):
        return np.nan
    return float(abs(np.log(prediction_rms / target_rms)))


def _write_well_figures(
    *,
    output_dir: Path,
    cluster: int,
    well_name: str,
    well: pd.DataFrame,
    target_ai: np.ndarray,
    lfm_ai: np.ndarray,
    control_delta: np.ndarray,
    anchor_delta: np.ndarray,
    volume: RealFieldVolume,
    wavelet: np.ndarray,
    tie: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, str], str, dict[str, dict[str, float | int]]]:
    figures_dir = output_dir / "figures" / "wells" / str(cluster)
    figures_dir.mkdir(parents=True, exist_ok=True)
    twt = well["twt_s"].to_numpy(dtype=np.float64)
    control_ai = lfm_ai + control_delta
    anchor_ai = lfm_ai + anchor_delta
    ai_path = figures_dir / f"{well_name}_ai_delta_qc.png"
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 7.5), sharey=True, constrained_layout=True)
    for values, label, color in ((target_ai, "Filtered logAI", "black"), (lfm_ai, "LFM", "tab:blue"), (control_ai, "Control", "tab:orange"), (anchor_ai, "Anchor", "tab:red")):
        axes[0].plot(values, twt, label=label, color=color, lw=1.2)
    target_delta = target_ai - lfm_ai
    for values, label, color in ((target_delta, "Well delta", "black"), (control_delta, "Control delta", "tab:orange"), (anchor_delta, "Anchor delta", "tab:red")):
        axes[1].plot(values, twt, label=label, color=color, lw=1.2)
    for axis in axes:
        axis.invert_yaxis(); axis.grid(True, alpha=0.25); axis.legend(fontsize=8)
    axes[0].set_ylabel("TWT (s)"); axes[0].set_xlabel("logAI"); axes[1].set_xlabel("delta logAI")
    fig.suptitle(f"L0 held-out well QC | {well_name} | cluster {cluster}")
    fig.savefig(ai_path, dpi=180); plt.close(fig)
    paths = {"ai_delta_qc_figure": repo_relative_path(ai_path, root=repo_root)}
    statuses = []
    waveform_metrics: dict[str, dict[str, float | int]] = {}
    for variant, pred_ai in (("control", control_ai), ("anchor", anchor_ai)):
        path = figures_dir / f"{well_name}_{variant}_forward_qc.png"
        status, metrics = _write_forward_figure(path=path, title=f"L0 well QC | {well_name} | cluster {cluster} | {variant}", pred_log_ai=pred_ai, filtered_log_ai=target_ai, twt=twt, well=well, volume=volume, wavelet=wavelet, tie=tie)
        paths[f"{variant}_forward_qc_figure"] = repo_relative_path(path, root=repo_root) if status == "ok" else ""
        statuses.append(status)
        waveform_metrics[variant] = metrics
    return paths, "ok" if all(status == "ok" for status in statuses) else "insufficient_forward_qc_support", waveform_metrics


def _write_forward_figure(*, path: Path, title: str, pred_log_ai: np.ndarray, filtered_log_ai: np.ndarray, twt: np.ndarray, well: pd.DataFrame, volume: RealFieldVolume, wavelet: np.ndarray, tie: Mapping[str, Any]) -> tuple[str, dict[str, float | int]]:
    if twt.size < 9:
        return "insufficient_forward_qc_support", {}
    filled = _fill_nonfinite(pred_log_ai)
    synthetic = forward_log_ai(filled[None, :], wavelet)[0]
    observed, inside = sample_volume_trilinear(
        volume.seismic,
        ilines=volume.ilines,
        xlines=volume.xlines,
        twt_s=volume.twt_s,
        inline_values=well["inline"].to_numpy(dtype=np.float64)[1:],
        xline_values=well["xline"].to_numpy(dtype=np.float64)[1:],
        sample_twt_s=twt[1:],
    )
    valid = inside & np.isfinite(synthetic) & np.isfinite(observed)
    start = _number(tie.get("tie_window_start_s")); stop = _number(tie.get("tie_window_end_s"))
    if np.isfinite(start) and np.isfinite(stop):
        valid &= (twt[1:] >= start) & (twt[1:] <= stop)
    run = _largest_true_run(valid)
    if run is None or run[1] - run[0] < 8:
        return "insufficient_forward_qc_support", {}
    sl = slice(*run); basis = twt[1:][sl]
    pred_ai = grid.Log(np.exp(filled[1:][sl]), basis, "twt", name="Predicted AI")
    filtered_ai = grid.Log(np.exp(_fill_nonfinite(filtered_log_ai)[1:][sl]), basis, "twt", name="Filtered LAS AI")
    reflectivity = grid.Reflectivity(np.tanh(0.5 * np.diff(filled))[sl], basis, "twt", name="Reflectivity")
    synthetic_trace = grid.Seismic(synthetic[sl], basis, "twt", name="Synthetic")
    observed_trace = grid.Seismic(observed[sl], basis, "twt", name="Seismic")
    xcorr_values = normalized_xcorr(observed_trace.values, synthetic_trace.values)
    xcorr_basis = synthetic_trace.sampling_rate * np.arange(-(synthetic_trace.size - 1), synthetic_trace.size)
    xcorr = grid.XCorr(xcorr_values, xcorr_basis, "tlag", name="XCorr")
    dxcorr = dynamic_normalized_xcorr(observed_trace, synthetic_trace)
    fig, _ = plot_well_waveform_qc([pred_ai, filtered_ai], reflectivity, synthetic_trace, observed_trace, xcorr, dxcorr, figsize=(12.0, 7.5), synthetic_ai=pred_ai, title=title)
    fig.savefig(path, dpi=180); plt.close(fig)
    raw = waveform_qc_metrics(observed_trace.values, synthetic_trace.values)
    denominator = float(np.dot(synthetic_trace.values, synthetic_trace.values))
    positive_scale = max(0.0, float(np.dot(observed_trace.values, synthetic_trace.values) / denominator)) if denominator > 0.0 else np.nan
    scaled = waveform_qc_metrics(observed_trace.values, positive_scale * synthetic_trace.values) if np.isfinite(positive_scale) else {}
    metrics: dict[str, float | int] = {
        **{f"waveform_raw_{key}": value for key, value in raw.items()},
        **{f"waveform_scaled_{key}": value for key, value in scaled.items()},
        "waveform_positive_scale": positive_scale,
    }
    return "ok", metrics


def _fill_nonfinite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Trace has insufficient finite samples.")
    return np.interp(np.arange(values.size), np.flatnonzero(valid), values[valid])


def _largest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    best = None
    start = None
    for index, value in enumerate(np.r_[np.asarray(mask, dtype=bool), False]):
        if value and start is None: start = index
        elif not value and start is not None:
            if best is None or index - start > best[1] - best[0]: best = (start, index)
            start = None
    return best


def _decide(
    holdout_summary: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    synthetic: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    thresholds = dict(cfg["thresholds"])
    primary = holdout_summary[holdout_summary["is_primary_holdout"].astype(bool)]
    primary_anchor = holdout_metrics[
        holdout_metrics["is_primary_holdout"].astype(bool)
        & holdout_metrics["model_variant"].eq("anchor")
    ]
    complete = len(primary) == 1 and primary["status"].eq("ok").all()
    row = primary.iloc[0] if len(primary) == 1 else pd.Series(dtype=object)
    checks = [
        ("complete_primary_holdout", complete, len(primary), 1),
        ("held_out_delta_corr_gain", complete and float(row["delta_corr_gain"]) > float(thresholds["minimum_held_out_delta_corr_gain"]), _series_number(row, "delta_corr_gain"), thresholds["minimum_held_out_delta_corr_gain"]),
        ("held_out_full_ai_corr_gain", complete and float(row["full_ai_corr_gain"]) >= float(thresholds["minimum_held_out_full_ai_corr_gain"]), _series_number(row, "full_ai_corr_gain"), thresholds["minimum_held_out_full_ai_corr_gain"]),
        ("held_out_full_ai_rmse_delta", complete and float(row["full_ai_rmse_delta"]) <= float(thresholds["maximum_held_out_full_ai_rmse_delta"]), _series_number(row, "full_ai_rmse_delta"), thresholds["maximum_held_out_full_ai_rmse_delta"]),
        ("held_out_delta_energy", complete and float(row["delta_energy_error_change"]) <= 0.0, _series_number(row, "delta_energy_error_change"), 0.0),
        ("held_out_gradient_energy", complete and float(row["gradient_energy_error_change"]) <= 0.0, _series_number(row, "gradient_energy_error_change"), 0.0),
        ("good_well_preservation", len(primary_anchor) == 1 and bool(primary_anchor["good_well_preservation_passed"].iloc[0]), bool(primary_anchor["good_well_preservation_passed"].iloc[0]) if len(primary_anchor) == 1 else False, True),
        ("synthetic_catastrophic_guard", not bool(synthetic["catastrophic"].any()), not bool(synthetic["catastrophic"].any()), True),
    ]
    table = pd.DataFrame.from_records([{"rule": name, "passed": passed, "observed": observed, "threshold": threshold} for name, passed, observed, threshold in checks])
    passed = bool(table["passed"].all())
    warning = bool(synthetic["warning"].any())
    if passed:
        status = "l0_positive_with_synthetic_warning" if warning else "l0_positive"
    elif not checks[0][1]: status = "incomplete_holdout_evidence"
    elif not checks[-1][1]: status = "synthetic_catastrophic_regression"
    elif not checks[-2][1]: status = "good_well_preservation_failed"
    elif not checks[4][1] or not checks[5][1]: status = "delta_or_gradient_collapse"
    else: status = "no_anchor_transfer_signal"
    return {"status": status, "eligible_for_l1": passed, "synthetic_warning": warning, "rules": table.to_dict(orient="records")}, table


def _validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(raw.get("l0_real_delta_anchor") or {})
    required = {"reference_training_run_dir", "real_field_lfm_dir", "model_role", "lambda_anchor", "anchor_clusters_per_step", "held_out_well", "device", "diagnostic_max_hz", "reconstruction_tolerance_log_ai", "real_field_inputs", "thresholds"}
    missing = sorted(required - set(cfg))
    if missing: raise ValueError(f"l0_real_delta_anchor is missing required keys: {missing}")
    if str(cfg["model_role"]) != FROZEN_MODEL_ROLE: raise ValueError("L0 model_role is frozen to no_lateral.")
    if float(cfg["lambda_anchor"]) != FROZEN_LAMBDA_ANCHOR: raise ValueError("L0 lambda_anchor is frozen to 0.1.")
    if int(cfg["anchor_clusters_per_step"]) != FROZEN_CLUSTERS_PER_STEP: raise ValueError("L0 anchor_clusters_per_step is frozen to 4.")
    if not str(cfg["held_out_well"]).strip(): raise ValueError("L0 held_out_well must be a non-empty well name.")
    if "exclude_same_cluster" in cfg and not isinstance(cfg["exclude_same_cluster"], bool):
        raise ValueError("L0 exclude_same_cluster must be a YAML boolean.")
    cfg["exclude_same_cluster"] = bool(cfg.get("exclude_same_cluster", False))
    threshold_keys = {
        "minimum_held_out_delta_corr_gain",
        "minimum_held_out_full_ai_corr_gain",
        "maximum_held_out_full_ai_rmse_delta",
        "maximum_good_well_corr_drop",
        "maximum_good_well_rmse_relative_increase",
        "maximum_good_well_delta_corr_drop",
        "synthetic_warning_error_relative_increase",
        "synthetic_warning_corr_drop",
        "maximum_synthetic_error_relative_increase",
        "maximum_synthetic_corr_drop",
    }
    missing_thresholds = sorted(threshold_keys - set(dict(cfg["thresholds"])))
    if missing_thresholds:
        raise ValueError(f"L0 thresholds are missing required keys: {missing_thresholds}")
    return cfg


def _series_number(row: pd.Series, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _resolve_sources(cfg: Mapping[str, Any], *, raw_config: Mapping[str, Any], repo_root: Path, data_root: Path) -> L0Sources:
    reference_dir = resolve_relative_path(str(cfg["reference_training_run_dir"]), root=repo_root)
    manifest_path = reference_dir / "model_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "ginn_v2_model_run_v1" or manifest.get("status") != "ok": raise ValueError("Reference model manifest is not a successful ginn_v2_model_run_v1.")
    if str(manifest.get("model_role")) != FROZEN_MODEL_ROLE or "lateral_mixer" in str(manifest.get("model_id")): raise ValueError("L0 reference recipe must be no_lateral.")
    benchmark_dir = resolve_relative_path(str(manifest["benchmark_dir"]), root=repo_root)
    for name, expected in dict(manifest["benchmark_hashes"]).items():
        path = benchmark_dir / name
        if not path.is_file() or sha256_file(path) != expected: raise ValueError(f"Reference benchmark hash mismatch: {path}")
    patch_index_path = resolve_relative_path(str(manifest["patch_index"]), root=repo_root)
    if sha256_file(patch_index_path) != str(manifest["patch_index_sha256"]): raise ValueError("Reference patch_index SHA-256 mismatch.")
    normalization_path = resolve_relative_path(str(manifest["normalization_path"]), root=repo_root)
    if sha256_file(normalization_path) != str(manifest["normalization_sha256"]):
        raise ValueError("Reference normalization SHA-256 mismatch.")
    normalization_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    if normalization_payload != dict(manifest["normalization"]):
        raise ValueError("Reference normalization file does not match the manifest payload.")
    input_stats_path = resolve_relative_path(str(manifest["input_reference_stats"]), root=repo_root)
    if sha256_file(input_stats_path) != str(manifest["input_reference_stats_sha256"]):
        raise ValueError("Reference input statistics SHA-256 mismatch.")
    lfm_dir = resolve_relative_path(str(cfg["real_field_lfm_dir"]), root=repo_root)
    lfm_summary_path = lfm_dir / "real_field_lfm_summary.json"
    lfm_summary = load_summary(lfm_summary_path, schema_version="real_field_lfm_v1", allowed_status={"ok"}, label="real_field_lfm_summary.json")
    lfm_path = lfm_dir / "real_field_lfm.npz"
    well_auto_tie_dir = resolve_relative_path(str(dict(lfm_summary["source_runs"])["well_auto_tie_dir"]), root=repo_root)
    well_inventory_file = resolve_relative_path(str(dict(lfm_summary["inputs"])["well_inventory_file"]), root=repo_root)
    seismic_cfg = dict(raw_config.get("seismic") or {})
    seismic_path = resolve_relative_path(str(seismic_cfg["file"]), root=data_root)
    recorded = resolve_relative_path(str(dict(lfm_summary["inputs"])["seismic_file"]), root=repo_root)
    if seismic_path.resolve() != recorded.resolve() or sha256_file(seismic_path) != str(dict(lfm_summary["inputs"])["seismic_sha256"]): raise ValueError("Top-level seismic does not match the frozen Step 7 source.")
    for path in (lfm_path, well_auto_tie_dir / "well_tie_metrics.csv", well_inventory_file):
        if not path.exists(): raise FileNotFoundError(path)
    return L0Sources(reference_dir, manifest_path, manifest, benchmark_dir, patch_index_path, lfm_dir, lfm_path, lfm_summary_path, lfm_summary, well_auto_tie_dir, well_inventory_file, seismic_path)


def _source_summary(sources: L0Sources, *, repo_root: Path) -> dict[str, Any]:
    paths = {"lfm": sources.lfm_path, "lfm_summary": sources.lfm_summary_path, "well_tie_metrics": sources.well_auto_tie_dir / "well_tie_metrics.csv", "well_inventory": sources.well_inventory_file, "seismic": sources.seismic_path}
    return {name: {"path": repo_relative_path(path, root=repo_root), "sha256": sha256_file(path)} for name, path in paths.items()}


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    buffer = io.BytesIO(); torch.save(dict(state), buffer); return hashlib.sha256(buffer.getvalue()).hexdigest()


def _number(value: Any) -> float:
    try: return float(value)
    except (TypeError, ValueError): return float("nan")


__all__ = [
    "BalancedAnchorSampler",
    "DifferentiableWellPredictor",
    "resolve_holdout_selection",
    "run_l0",
]
