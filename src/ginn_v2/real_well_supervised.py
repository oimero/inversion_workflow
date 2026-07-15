"""Sparse real-well canonical-increment supervision for GINN-v2."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cup.physics.calibration import AIVelocityRelation
from cup.physics.numpy_backend import forward_depth, forward_time, reflectivity_from_log_ai
from cup.synthetic.schemas import FORWARD_MODEL_INPUTS_SCHEMA_VERSION
from cup.seismic.lfm.artifacts import resolve_lfm_variant
from cup.seismic.viz import plot_well_waveform_qc, waveform_qc_metrics
from cup.utils.io import repo_relative_path, resolve_relative_path
from cup.well.anchor import build_well_anchor_samples, sample_volume_trilinear
from ginn_v2.real_field import (
    RealFieldVolume,
    build_real_field_patch_index,
    load_real_field_volume,
    load_selected_wavelet,
)
from wtie.optimize.similarity import dynamic_normalized_xcorr, normalized_xcorr
from wtie.processing import grid


@dataclass(frozen=True)
class RealWellSupervisedSources:
    lfm_run_dir: Path
    variant_id: str
    lfm_path: Path
    lfm_run_summary_path: Path
    lfm_run_summary: dict[str, Any]
    lfm_contract_fingerprint_sha256: str
    well_control_run_dir: Path
    well_control_contract_fingerprint_sha256: str
    seismic_path: Path


class BalancedRealWellSampler:
    """Deterministic cluster and within-cluster shuffled cycles."""

    def __init__(self, samples: pd.DataFrame, *, clusters_per_step: int, seed: int) -> None:
        valid = samples[samples["valid_for_fit"].astype(bool)].copy()
        if valid.empty:
            raise ValueError("Real-well sampler received no valid samples.")
        self.wells_by_cluster = {
            int(cluster): sorted(group["well_name"].astype(str).unique().tolist())
            for cluster, group in valid.groupby("spatial_cluster_id", sort=True)
        }
        self.clusters = sorted(self.wells_by_cluster)
        self.k = min(int(clusters_per_step), len(self.clusters))
        if self.k <= 0:
            raise ValueError("Real-well sampler requires at least one training cluster.")
        self.rng = np.random.default_rng(int(seed))
        self.cluster_queue: deque[int] = deque()
        self.well_queues: dict[int, deque[str]] = {
            cluster: deque() for cluster in self.clusters
        }
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
        selected = [
            (cluster, self._next_well(cluster))
            for cluster in self._next_clusters()
        ]
        for item in selected:
            self.counts[item] += 1
        return selected


class DifferentiableWellPredictor:
    """Canonical no-lateral patch/stitch inference sampled on well trajectories."""

    def __init__(
        self,
        *,
        volume: RealFieldVolume,
        patch_spec: Mapping[str, Any],
        normalization: Mapping[str, Any],
        forward_batch_size: int = 256,
    ) -> None:
        self.volume = volume
        self.patch_spec = dict(patch_spec)
        self.normalization = dict(normalization)
        self.forward_batch_size = int(forward_batch_size)
        self._patches: dict[int, list[Any]] = {}
        self._trace_task_cache: dict[tuple[int, int], list[tuple[Any, torch.Tensor]]] = {}
        self._geometry_cache: dict[
            tuple[int, ...],
            tuple[
                list[list[tuple[tuple[int, int], int, int, float, float]]],
                set[tuple[int, int]],
            ],
        ] = {}

    def prepare(
        self,
        wells: pd.DataFrame,
        *,
        logger: logging.Logger | None = None,
    ) -> dict[str, int]:
        all_nodes: set[tuple[int, int]] = set()
        groups = list(wells.groupby("well_name", sort=True))
        for index, (well_name, rows) in enumerate(groups, start=1):
            _geometry, nodes = self._cached_point_geometry(rows)
            all_nodes.update(nodes)
            if logger is not None:
                logger.info(
                    "real-well precompute: geometry well=%s (%d/%d) cumulative_nodes=%d",
                    well_name,
                    index,
                    len(groups),
                    len(all_nodes),
                )
        task_count = 0
        ordered_nodes = sorted(all_nodes)
        node_log_interval = max(1, len(ordered_nodes) // 10)
        for index, node in enumerate(ordered_nodes, start=1):
            task_count += len(self._trace_tasks_for_node(node))
            if logger is not None and (
                index == 1
                or index == len(ordered_nodes)
                or index % node_log_interval == 0
            ):
                logger.info(
                    "real-well precompute: support node=%d/%d trace_patches=%d",
                    index,
                    len(ordered_nodes),
                    task_count,
                )
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
                vertical_samples=int(self.patch_spec["vertical_samples"]),
                lateral_stride=int(self.patch_spec["lateral_stride"]),
                vertical_stride=int(self.patch_spec["vertical_stride"]),
            )
        return self._patches[inline_index]

    def predict_increment_n(
        self,
        model: torch.nn.Module,
        rows: pd.DataFrame,
        *,
        device: torch.device,
        canonical_full_patch: bool = False,
    ) -> torch.Tensor:
        return self.predict_increment_n_groups(
            model,
            [rows],
            device=device,
            canonical_full_patch=canonical_full_patch,
        )[0]

    def predict_increment_n_groups(
        self,
        model: torch.nn.Module,
        groups: Sequence[pd.DataFrame],
        *,
        device: torch.device,
        canonical_full_patch: bool = False,
    ) -> list[torch.Tensor]:
        if not groups or any(rows.empty for rows in groups):
            raise ValueError("Cannot predict an empty real-well sample group.")
        geometries = []
        nodes: set[tuple[int, int]] = set()
        for rows in groups:
            geometry, group_nodes = self._cached_point_geometry(rows)
            geometries.append(geometry)
            nodes.update(group_nodes)
        node_predictions = self._predict_nodes(
            model,
            nodes,
            device=device,
            canonical_full_patch=canonical_full_patch,
        )
        return [
            self._interpolate_geometry(geometry, node_predictions)
            for geometry in geometries
        ]

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
                raise ValueError("Canonical-increment predictor found an uncovered well sample.")
            predictions.append(value / interpolation_weight)
        return torch.stack(predictions)

    def _cached_point_geometry(
        self,
        rows: pd.DataFrame,
    ) -> tuple[
        list[list[tuple[tuple[int, int], int, int, float, float]]],
        set[tuple[int, int]],
    ]:
        if "_real_well_row_id" not in rows:
            return self._point_geometry(rows.reset_index(drop=True))
        key = tuple(
            pd.to_numeric(rows["_real_well_row_id"], errors="raise")
            .astype(int)
            .tolist()
        )
        if key not in self._geometry_cache:
            self._geometry_cache[key] = self._point_geometry(rows.reset_index(drop=True))
        return self._geometry_cache[key]

    def _point_geometry(
        self,
        rows: pd.DataFrame,
    ) -> tuple[
        list[list[tuple[tuple[int, int], int, int, float, float]]],
        set[tuple[int, int]],
    ]:
        axes = (self.volume.ilines, self.volume.xlines, self.volume.sample_axis.values)
        columns = ("inline", "xline", "sample")
        fractional = []
        for axis, column in zip(axes, columns):
            values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float64)
            fractional.append(
                np.interp(values, axis, np.arange(axis.size), left=np.nan, right=np.nan)
            )
        output: list[list[tuple[tuple[int, int], int, int, float, float]]] = []
        nodes: set[tuple[int, int]] = set()
        for point in range(len(rows)):
            if not all(np.isfinite(frac[point]) for frac in fractional):
                raise ValueError("Well sample is outside a canonical real-field axis.")
            positions = [float(frac[point]) for frac in fractional]
            lower = [
                min(int(np.floor(value)), axes[dim].size - 2)
                for dim, value in enumerate(positions)
            ]
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
                    terms.append(
                        (node, lower[2], lower[2] + 1, weights[2], spatial_weight)
                    )
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
            raise ValueError("No canonical real-field patches cover the well support nodes.")
        outputs: list[torch.Tensor] = []
        for start in range(0, len(tasks), self.forward_batch_size):
            batch = torch.stack(
                [task[2] for task in tasks[start : start + self.forward_batch_size]]
            ).to(device)
            outputs.extend(model(batch)[:, 0, 0, :].unbind(0))
        return self._stitch_node_tasks(nodes, tasks, outputs, device=device)

    def _trace_tasks_for_node(
        self,
        node: tuple[int, int],
    ) -> list[tuple[Any, torch.Tensor]]:
        if node in self._trace_task_cache:
            return self._trace_task_cache[node]
        inline_index, xline_index = node
        tasks: list[tuple[Any, torch.Tensor]] = []
        for patch in self.patches(inline_index):
            if not patch.lateral_start <= xline_index < patch.lateral_stop:
                continue
            sl = (inline_index, xline_index, slice(patch.sample_start, patch.sample_stop))
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
                    patch_keys[(inline_index, patch.lateral_start, patch.sample_start)] = patch
        patch_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        items = sorted(patch_keys.items())
        chunk_size = max(
            1,
            self.forward_batch_size // int(self.patch_spec["lateral_samples"]),
        )
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            tensors = []
            for (inline_index, _, _), patch in chunk:
                sl = (
                    inline_index,
                    slice(patch.lateral_start, patch.lateral_stop),
                    slice(patch.sample_start, patch.sample_stop),
                )
                tensors.append(
                    self._input_tensor(
                        self.volume.seismic[sl],
                        self.volume.lfm[sl],
                        self.volume.valid_mask[sl],
                    )
                )
            predicted = model(torch.stack(tensors).to(device))[:, 0]
            for (key, _patch), value in zip(chunk, predicted.unbind(0)):
                patch_outputs[key] = value
        tasks: list[tuple[tuple[int, int], Any, torch.Tensor]] = []
        outputs: list[torch.Tensor] = []
        for node in sorted(nodes):
            inline_index, xline_index = node
            for patch in self.patches(inline_index):
                if patch.lateral_start <= xline_index < patch.lateral_stop:
                    key = (inline_index, patch.lateral_start, patch.sample_start)
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
        nt = self.volume.sample_axis.values.size
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
                        patch.sample_start : patch.sample_stop,
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                destination = torch.arange(
                    patch.sample_start,
                    patch.sample_stop,
                    device=device,
                )[valid]
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
            stitched[node] = torch.where(
                weight > 0.0,
                total / weight,
                torch.full_like(total, torch.nan),
            )
        return stitched

    def _input_tensor(
        self,
        seismic: np.ndarray,
        lfm: np.ndarray,
        valid: np.ndarray,
    ) -> torch.Tensor:
        seismic_n = (
            np.asarray(seismic, dtype=np.float32)
            - float(self.normalization["seismic"]["mean"])
        ) / float(self.normalization["seismic"]["std"])
        lfm_n = (
            np.asarray(lfm, dtype=np.float32)
            - float(self.normalization["lfm"]["mean"])
        ) / float(self.normalization["lfm"]["std"])
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


class RealWellSupervisedSupport:
    """Prepared real-well labels and differentiable support for one model run."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        sources: RealWellSupervisedSources,
        volume: RealFieldVolume,
        samples: pd.DataFrame,
        training_samples: pd.DataFrame,
        excluded_wells: Sequence[str],
        supervision_excluded_wells: Sequence[str],
        held_out_cluster: int,
        predictor: DifferentiableWellPredictor,
        sampler: BalancedRealWellSampler | None,
        normalization: Mapping[str, Any],
        precompute_summary: Mapping[str, Any],
        source_summary: Mapping[str, Any],
    ) -> None:
        self.config = dict(config)
        self.sources = sources
        self.volume = volume
        self.samples = samples
        self.training_samples = training_samples
        self.excluded_wells = list(excluded_wells)
        self.supervision_excluded_wells = list(supervision_excluded_wells)
        self.held_out_cluster = int(held_out_cluster)
        self.predictor = predictor
        self.sampler = sampler
        self.normalization = dict(normalization)
        self.precompute_summary = dict(precompute_summary)
        self.source_summary = dict(source_summary)
        self.canonical_full_patch = False
        self.reconstruction_max_abs_log_ai: float | None = None

    def configure_model(self, *, receptive_field_lateral: int) -> None:
        self.canonical_full_patch = int(receptive_field_lateral) != 1

    def validate_reconstruction(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
    ) -> float:
        was_training = model.training
        model.eval()
        if self.canonical_full_patch:
            with torch.no_grad():
                prediction = self.predictor.predict_increment_n(
                    model,
                    self.samples,
                    device=device,
                    canonical_full_patch=True,
                )
            model.train(was_training)
            if not bool(torch.all(torch.isfinite(prediction))):
                raise ValueError("Canonical full-patch real-well prediction is non-finite.")
            return float("nan")
        with torch.no_grad():
            sparse = self.predictor.predict_increment_n(
                model,
                self.samples,
                device=device,
            )
            canonical = self.predictor.predict_increment_n(
                model,
                self.samples,
                device=device,
                canonical_full_patch=True,
            )
        model.train(was_training)
        error = float(torch.max(torch.abs(sparse - canonical)).cpu())
        tolerance = float(self.config["reconstruction_tolerance_log_ai"])
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(
                "real_field_well_reconstruction_mismatch: "
                f"max_abs_log_ai={error:.9g}, tolerance={tolerance:.9g}"
            )
        self.reconstruction_max_abs_log_ai = error
        return error

    def training_loss(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        if self.sampler is None:
            raise RuntimeError("Real-well supervised loss requested without an active sampler.")
        selected = self.sampler.select()
        groups = [
            self.training_samples[
                self.training_samples["spatial_cluster_id"].astype(int).eq(cluster)
                & self.training_samples["well_name"].astype(str).eq(well_name)
            ].sort_values("sample_index")
            for cluster, well_name in selected
        ]
        predictions = self.predictor.predict_increment_n_groups(model, groups, device=device)
        losses = []
        n_samples = 0
        for rows, prediction in zip(groups, predictions):
            target = rows["well_target_increment_log_ai"].to_numpy(dtype=np.float32)
            target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
            losses.append(torch.mean((prediction - target_tensor) ** 2))
            n_samples += int(len(rows))
        return torch.stack(losses).mean(), {
            "selected_real_clusters": int(len(selected)),
            "selected_real_wells": int(len(selected)),
            "selected_real_samples": int(n_samples),
        }

    def sampling_qc_frame(self) -> pd.DataFrame:
        rows = []
        counts = self.sampler.counts if self.sampler is not None else {}
        wells = self.samples.drop_duplicates("well_name").sort_values("well_name")
        for _, row in wells.iterrows():
            cluster = int(row["spatial_cluster_id"])
            well_name = str(row["well_name"])
            rows.append(
                {
                    "well_name": well_name,
                    "spatial_cluster_id": cluster,
                    "supervision_role": str(row["supervision_role"]),
                    "used_for_real_well_training": bool(
                        row["used_for_real_well_training"]
                    ),
                    "exclusion_reason": str(row["exclusion_reason"]),
                    "selected_count": int(counts.get((cluster, well_name), 0)),
                    "n_valid_samples": int(
                        self.samples[
                            self.samples["well_name"].astype(str).eq(well_name)
                            & self.samples["valid_for_fit"].astype(bool)
                        ].shape[0]
                    ),
                }
            )
        return pd.DataFrame.from_records(rows)

    def manifest_payload(self, *, repo_root: Path) -> dict[str, Any]:
        public_config = {
            key: value
            for key, value in self.config.items()
            if key != "samples_path"
        }
        return {
            "config": public_config,
            "held_out_cluster_id": self.held_out_cluster,
            "holdout_excluded_wells": list(self.excluded_wells),
            "supervision_excluded_well_names": list(
                self.supervision_excluded_wells
            ),
            "same_cluster_training_leakage_risk": bool(
                float(self.config["lambda_real_well_supervised"]) > 0.0
                and not self.config["exclude_same_cluster"]
            ),
            "n_wells": int(self.samples["well_name"].nunique()),
            "n_training_wells": int(
                self.samples.loc[
                    self.samples["used_for_real_well_training"].astype(bool),
                    "well_name",
                ].nunique()
            ),
            "n_clusters": int(self.samples["spatial_cluster_id"].nunique()),
            "precompute": dict(self.precompute_summary),
            "support_prediction_mode": (
                "canonical_full_patch"
                if self.canonical_full_patch
                else "sparse_no_lateral"
            ),
            "reconstruction_max_abs_log_ai": self.reconstruction_max_abs_log_ai,
            "sources": dict(self.source_summary),
            "well_samples": {
                "path": repo_relative_path(
                    Path(self.config["samples_path"]),
                    root=repo_root,
                ),
            },
        }


def prepare_real_well_supervised_support(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    output_dir: Path,
    normalization: Mapping[str, Any],
    patch_spec: Mapping[str, Any],
    input_reference_stats_path: Path,
    lambda_real_well_supervised: float,
    seed: int,
    logger: logging.Logger,
) -> RealWellSupervisedSupport:
    cfg = _validate_config(config)
    cfg["lambda_real_well_supervised"] = float(lambda_real_well_supervised)
    sources = _resolve_sources(cfg, repo_root=repo_root)
    logger.info("real-well source: %s variant=%s", sources.lfm_run_dir, sources.variant_id)
    with input_reference_stats_path.open("r", encoding="utf-8") as handle:
        input_stats = json.load(handle)
    seismic_type = str(dict(sources.lfm_run_summary.get("seismic") or {}).get("type") or "").casefold()
    if seismic_type not in {"segy", "zgy"}:
        raise ValueError(f"Unified LFM run has unsupported seismic type: {seismic_type!r}")
    real_cfg = {
        "real_field_inputs": {
            "lfm_run_dir": repo_relative_path(sources.lfm_run_dir, root=repo_root),
            "variant_id": sources.variant_id,
            "well_control_run_dir": repo_relative_path(sources.well_control_run_dir, root=repo_root),
            "seismic_file": repo_relative_path(sources.seismic_path, root=repo_root),
            "seismic_type": seismic_type,
            "seismic_value_transform": cfg["seismic_value_transform"],
            "lfm_value_transform": cfg["lfm_value_transform"],
            "seismic_reference_stats": dict(input_stats["stats"]),
            "seismic_reference_stats_file": repo_relative_path(
                input_reference_stats_path,
                root=repo_root,
            ),
        },
        "volume": {},
    }
    logger.info("real-well: loading real-field seismic and LFM volume")
    volume = load_real_field_volume(
        config=real_cfg,
        root=repo_root,
        data_root=repo_root,
    )
    logger.info("real-well: building canonical increment labels")
    samples, label_metadata = build_well_anchor_samples(
        well_control_run_dir=sources.well_control_run_dir,
        lfm=volume.lfm,
        valid_mask=volume.valid_mask,
        ilines=volume.ilines,
        xlines=volume.xlines,
        samples=volume.sample_axis.values,
        repo_root=repo_root,
        cluster_radius_m=float(cfg["cluster_radius_m"]),
        variant_id=sources.variant_id,
        lfm_contract_fingerprint_sha256=sources.lfm_contract_fingerprint_sha256,
        expected_well_control_contract_fingerprint_sha256=(
            sources.well_control_contract_fingerprint_sha256
        ),
    )
    valid = samples[samples["valid_for_fit"].astype(bool)].copy()
    if valid.empty:
        raise ValueError("Real-well label builder produced no valid samples.")
    valid["_real_well_row_id"] = np.arange(len(valid), dtype=np.int64)
    held_out_cluster, excluded_wells, training = assign_supervision_roles(
        valid,
        held_out_well=cfg["held_out_well"],
        exclude_same_cluster=bool(cfg["exclude_same_cluster"]),
        supervision_excluded_well_names=cfg["supervision_excluded_well_names"],
        require_training=lambda_real_well_supervised > 0.0,
    )
    if lambda_real_well_supervised <= 0.0:
        valid["used_for_real_well_training"] = False
        training["used_for_real_well_training"] = False
    role_lookup = (
        valid.drop_duplicates("well_name")
        .set_index("well_name")[
            ["supervision_role", "used_for_real_well_training", "exclusion_reason"]
        ]
        .to_dict(orient="index")
    )
    for column in (
        "supervision_role",
        "used_for_real_well_training",
        "exclusion_reason",
    ):
        samples[column] = samples["well_name"].astype(str).map(
            {well: values[column] for well, values in role_lookup.items()}
        )
    samples_path = output_dir / "real_well_supervised_samples.csv"
    samples.to_csv(samples_path, index=False)
    cfg["samples_path"] = str(samples_path)
    cfg["label_metadata"] = label_metadata
    predictor = DifferentiableWellPredictor(
        volume=volume,
        patch_spec=patch_spec,
        normalization=normalization,
    )
    logger.info(
        "real-well: precomputing support for %d wells",
        valid["well_name"].nunique(),
    )
    precompute = predictor.prepare(valid, logger=logger)
    logger.info(
        "real-well: prepared %d nodes and %d trace patches",
        precompute["n_support_nodes"],
        precompute["n_support_trace_patches"],
    )
    sampler = (
        BalancedRealWellSampler(
            training,
            clusters_per_step=int(cfg["clusters_per_step"]),
            seed=int(seed) + 1_000_003,
        )
        if lambda_real_well_supervised > 0.0
        else None
    )
    return RealWellSupervisedSupport(
        config=cfg,
        sources=sources,
        volume=volume,
        samples=valid,
        training_samples=training,
        excluded_wells=excluded_wells,
        supervision_excluded_wells=cfg["supervision_excluded_well_names"],
        held_out_cluster=held_out_cluster,
        predictor=predictor,
        sampler=sampler,
        normalization=normalization,
        precompute_summary=precompute,
        source_summary=_source_summary(sources, repo_root=repo_root),
    )


def assign_supervision_roles(
    samples: pd.DataFrame,
    *,
    held_out_well: str,
    exclude_same_cluster: bool,
    supervision_excluded_well_names: Sequence[str],
    require_training: bool,
) -> tuple[int, list[str], pd.DataFrame]:
    wells = set(samples["well_name"].astype(str).unique())
    supervision_excluded = [str(name).strip() for name in supervision_excluded_well_names]
    duplicates = sorted(
        {name for name in supervision_excluded if supervision_excluded.count(name) > 1}
    )
    if duplicates:
        raise ValueError(
            f"Duplicate supervision_excluded_well_names: {duplicates}"
        )
    unknown = sorted(set(supervision_excluded) - wells)
    if unknown:
        raise ValueError(
            "Configured supervision-excluded wells are unavailable; "
            f"unknown wells: {unknown}; available wells: {sorted(wells)}"
        )
    if held_out_well in supervision_excluded:
        raise ValueError(
            f"held_out_well={held_out_well!r} overlaps "
            "supervision_excluded_well_names."
        )
    if held_out_well not in wells:
        raise ValueError(
            f"Configured held_out_well={held_out_well!r} is unavailable; "
            f"available wells: {sorted(wells)}"
        )
    cluster_values = samples.loc[
        samples["well_name"].astype(str).eq(held_out_well),
        "spatial_cluster_id",
    ].astype(int).unique()
    if cluster_values.size != 1:
        raise ValueError(f"Held-out well has ambiguous cluster: {held_out_well}")
    cluster = int(cluster_values[0])
    if exclude_same_cluster:
        excluded = sorted(
            samples.loc[
                samples["spatial_cluster_id"].astype(int).eq(cluster),
                "well_name",
            ].astype(str).unique()
        )
    else:
        excluded = [held_out_well]
    samples["supervision_role"] = "training"
    samples["exclusion_reason"] = ""
    held_mask = samples["well_name"].astype(str).eq(held_out_well)
    samples.loc[held_mask, "supervision_role"] = "held_out"
    samples.loc[held_mask, "exclusion_reason"] = "configured_holdout"
    same_cluster_mask = samples["well_name"].astype(str).isin(
        set(excluded) - {held_out_well}
    )
    samples.loc[same_cluster_mask, "supervision_role"] = "same_cluster_excluded"
    samples.loc[same_cluster_mask, "exclusion_reason"] = "same_cluster_as_holdout"
    configured_exclusion_mask = samples["well_name"].astype(str).isin(
        supervision_excluded
    )
    samples.loc[
        configured_exclusion_mask, "supervision_role"
    ] = "configured_supervision_excluded"
    samples.loc[
        configured_exclusion_mask, "exclusion_reason"
    ] = "configured_supervision_exclusion"
    samples["used_for_real_well_training"] = samples["supervision_role"].eq("training")
    training = samples[samples["used_for_real_well_training"]].copy()
    if require_training and training.empty:
        raise ValueError("Configured holdout leaves no real-well training wells.")
    return cluster, excluded, training


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    retired = sorted({"lfm_file", "well_auto_tie_dir", "well_inventory_file"} & set(cfg))
    if retired:
        raise ValueError(
            f"train.real_well_supervised contains retired v1 source keys {retired}; "
            "use lfm_run_dir + variant_id + well_control_run_dir."
        )
    required = {
        "lfm_run_dir",
        "variant_id",
        "well_control_run_dir",
        "held_out_well",
        "supervision_excluded_well_names",
        "exclude_same_cluster",
        "clusters_per_step",
        "cluster_radius_m",
        "diagnostic_max_hz",
        "reconstruction_tolerance_log_ai",
        "seismic_value_transform",
        "lfm_value_transform",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"train.real_well_supervised is missing required keys: {missing}")
    if not str(cfg["held_out_well"]).strip():
        raise ValueError("train.real_well_supervised.held_out_well must be non-empty.")
    cfg["held_out_well"] = str(cfg["held_out_well"]).strip()
    excluded_value = cfg["supervision_excluded_well_names"]
    if excluded_value is None:
        excluded_names: list[str] = []
    elif isinstance(excluded_value, list):
        excluded_names = []
        for index, item in enumerate(excluded_value):
            name = str(item).strip()
            if not name:
                raise ValueError(
                    "train.real_well_supervised."
                    f"supervision_excluded_well_names[{index}] must be non-empty."
                )
            excluded_names.append(name)
    else:
        raise ValueError(
            "train.real_well_supervised.supervision_excluded_well_names "
            "must be a YAML list or null."
        )
    duplicates = sorted(
        {name for name in excluded_names if excluded_names.count(name) > 1}
    )
    if duplicates:
        raise ValueError(
            "train.real_well_supervised.supervision_excluded_well_names "
            f"contains duplicates: {duplicates}"
        )
    if cfg["held_out_well"] in excluded_names:
        raise ValueError(
            "train.real_well_supervised.held_out_well overlaps "
            "supervision_excluded_well_names."
        )
    cfg["supervision_excluded_well_names"] = excluded_names
    for key in ("lfm_run_dir", "variant_id", "well_control_run_dir"):
        value = str(cfg[key]).strip()
        if not value or value.casefold() == "auto":
            raise ValueError(f"train.real_well_supervised.{key} must be explicit and non-auto.")
        cfg[key] = value
    if str(cfg["lfm_value_transform"]).casefold() not in {"identity", "none"}:
        raise ValueError("train.real_well_supervised.lfm_value_transform must be identity for the canonical LFM contract.")
    if not isinstance(cfg["exclude_same_cluster"], bool):
        raise ValueError("train.real_well_supervised.exclude_same_cluster must be a YAML boolean.")
    if int(cfg["clusters_per_step"]) <= 0:
        raise ValueError("train.real_well_supervised.clusters_per_step must be positive.")
    for key in (
        "cluster_radius_m",
        "diagnostic_max_hz",
        "reconstruction_tolerance_log_ai",
    ):
        value = float(cfg[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"train.real_well_supervised.{key} must be finite and positive.")
    return cfg


def _resolve_sources(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> RealWellSupervisedSources:
    selected = resolve_lfm_variant(config, repo_root=repo_root)
    seismic = dict(selected.run_summary.get("seismic") or {})
    seismic_path = resolve_relative_path(str(seismic.get("path") or ""), root=repo_root)
    for path in (
        selected.lfm_path,
        selected.well_control_run_dir / "well_control_manifest.csv",
        seismic_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    return RealWellSupervisedSources(
        lfm_run_dir=selected.run_dir,
        variant_id=selected.variant_id,
        lfm_path=selected.lfm_path,
        lfm_run_summary_path=selected.run_summary_path,
        lfm_run_summary=dict(selected.run_summary),
        lfm_contract_fingerprint_sha256=selected.contract_fingerprint_sha256,
        well_control_run_dir=selected.well_control_run_dir,
        well_control_contract_fingerprint_sha256=(
            selected.well_control_contract_fingerprint_sha256
        ),
        seismic_path=seismic_path,
    )


def _source_summary(
    sources: RealWellSupervisedSources,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    paths = {
        "lfm": sources.lfm_path,
        "lfm_run_summary": sources.lfm_run_summary_path,
        "well_control_summary": sources.well_control_run_dir / "run_summary.json",
        "well_control_manifest": sources.well_control_run_dir / "well_control_manifest.csv",
        "seismic": sources.seismic_path,
    }
    summary = {
        name: {"path": repo_relative_path(path, root=repo_root)}
        for name, path in paths.items()
    }
    summary["lfm"]["contract_fingerprint_sha256"] = (
        sources.lfm_contract_fingerprint_sha256
    )
    summary["well_control_summary"]["contract_fingerprint_sha256"] = (
        sources.well_control_contract_fingerprint_sha256
    )
    return summary


def _load_depth_forward_inputs(
    benchmark_manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[np.ndarray, np.ndarray, AIVelocityRelation]:
    reference = str(benchmark_manifest.get("forward_model_inputs_path") or "").strip()
    if not reference:
        raise ValueError("Depth real-well benchmark lacks forward_model_inputs_path.")
    path = resolve_relative_path(reference, root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"Depth real-well forward inputs not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != FORWARD_MODEL_INPUTS_SCHEMA_VERSION:
        raise ValueError(
            "Depth real-well expected "
            f"{FORWARD_MODEL_INPUTS_SCHEMA_VERSION}, got {payload.get('schema')!r}; "
            "rebuild the depth forward-model inputs."
        )
    if payload.get("sample_domain") != "depth" or payload.get("depth_basis") != "tvdss":
        raise ValueError("Depth real-well forward inputs must declare depth/TVDSS.")
    relation = AIVelocityRelation.from_mapping(
        dict(payload.get("ai_velocity_relation") or {})
    )
    wavelet_ref = dict(payload.get("wavelet") or {})
    if wavelet_ref.get("time_unit") != "s":
        raise ValueError("Depth real-well wavelet time_unit must be 's'.")
    wavelet_path = resolve_relative_path(str(wavelet_ref.get("path") or ""), root=repo_root)
    if not wavelet_path.is_file():
        raise FileNotFoundError(f"Depth real-well wavelet not found: {wavelet_path}")
    frame = pd.read_csv(wavelet_path)
    required = {"time_s", "amplitude"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Depth real-well wavelet must contain columns {sorted(required)}.")
    return (
        frame["time_s"].to_numpy(dtype=np.float64),
        frame["amplitude"].to_numpy(dtype=np.float64),
        relation,
    )


def _validate_well_sample_contract(well: pd.DataFrame, *, volume: RealFieldVolume) -> None:
    domains = set(well["sample_domain"].astype(str))
    units = set(well["sample_unit"].astype(str))
    expected_units = {volume.sample_axis.unit}
    if domains != {volume.sample_axis.domain} or units != expected_units:
        raise ValueError(
            "Real-well/LFM sample-axis mismatch: "
            f"well_domain={sorted(domains)}, well_unit={sorted(units)}, "
            f"lfm_domain={volume.sample_axis.domain!r}, lfm_unit={volume.sample_axis.unit!r}."
        )
    if volume.sample_axis.domain == "depth" and (
        volume.depth_basis != "tvdss" or volume.sample_axis.unit != "m"
    ):
        raise ValueError("Depth real-well QC requires LFM TVDSS in metres.")
    if volume.sample_axis.domain == "time" and volume.sample_axis.unit != "s":
        raise ValueError("Time real-well QC requires TWT in seconds.")


def evaluate_real_wells(
    *,
    support: RealWellSupervisedSupport,
    models: Mapping[str, torch.nn.Module],
    output_dir: Path,
    benchmark_dir: Path,
    repo_root: Path,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Path]:
    """Evaluate best/final checkpoints on every valid real well."""

    manifest_path = benchmark_dir / "benchmark_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        benchmark_manifest = json.load(handle)
    sample_domain = support.volume.sample_axis.domain
    relation: AIVelocityRelation | None = None
    if sample_domain == "time":
        wavelet_dir = resolve_relative_path(
            str(dict(benchmark_manifest["source_runs"])["wavelet_generation_dir"]),
            root=repo_root,
        )
        wavelet_time_s, wavelet, _wavelet_metadata = load_selected_wavelet(wavelet_dir)
    elif sample_domain == "depth":
        wavelet_time_s, wavelet, relation = _load_depth_forward_inputs(
            benchmark_manifest,
            repo_root=repo_root,
        )
    else:
        raise ValueError(f"Unsupported real-well sample domain: {sample_domain!r}.")
    groups = [
        group.sort_values("sample_index").copy()
        for _, group in support.samples.groupby("well_name", sort=True)
    ]
    metrics_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    waveform_rows: list[dict[str, Any]] = []

    for checkpoint_name, model in models.items():
        logger.info(
            "real-well QC: checkpoint=%s, wells=%d",
            checkpoint_name,
            len(groups),
        )
        model.eval()
        with torch.no_grad():
            predictions_n = support.predictor.predict_increment_n_groups(
                model,
                groups,
                device=device,
                canonical_full_patch=support.canonical_full_patch,
            )
        for index, (well, prediction_n) in enumerate(zip(groups, predictions_n), start=1):
            well_name = str(well["well_name"].iloc[0])
            cluster = int(well["spatial_cluster_id"].iloc[0])
            pred_increment = prediction_n.detach().cpu().numpy()
            target_ai = well["well_log_ai"].to_numpy(dtype=np.float64)
            lfm_ai = well["lfm_log_ai"].to_numpy(dtype=np.float64)
            target_increment = well["well_target_increment_log_ai"].to_numpy(dtype=np.float64)
            pred_ai = lfm_ai + pred_increment
            sample_axis = well["sample"].to_numpy(dtype=np.float64)
            _validate_well_sample_contract(well, volume=support.volume)
            common = {
                "checkpoint": checkpoint_name,
                "well_name": well_name,
                "sample_domain": sample_domain,
                "sample_unit": support.volume.sample_axis.unit,
                "depth_basis": support.volume.depth_basis or "",
                "spatial_cluster_id": cluster,
                "supervision_role": str(well["supervision_role"].iloc[0]),
                "used_for_real_well_training": bool(
                    well["used_for_real_well_training"].iloc[0]
                ),
                "exclusion_reason": str(well["exclusion_reason"].iloc[0]),
            }
            metrics_rows.append(
                {
                    **common,
                    **_well_metrics(
                        target_ai,
                        target_increment,
                        pred_ai,
                        pred_increment,
                        sample_axis,
                    ),
                }
            )
            band_rows.append(
                {
                    **common,
                    **_well_band_metrics(
                        target_ai=target_ai,
                        target_increment=target_increment,
                        pred_ai=pred_ai,
                        pred_increment=pred_increment,
                        sample_axis=sample_axis,
                        sample_domain=sample_domain,
                        diagnostic_max_hz=float(
                            support.config["diagnostic_max_hz"]
                        ),
                    ),
                }
            )
            figures, waveform_status, waveform = _write_well_figures(
                output_dir=output_dir,
                checkpoint_name=checkpoint_name,
                cluster=cluster,
                well_name=well_name,
                well=well,
                target_ai=target_ai,
                target_increment=target_increment,
                lfm_ai=lfm_ai,
                pred_increment=pred_increment,
                volume=support.volume,
                wavelet_time_s=wavelet_time_s,
                wavelet=wavelet,
                relation=relation,
                tie={},
                repo_root=repo_root,
            )
            waveform_rows.append(
                {
                    **common,
                    "status": waveform_status,
                    **waveform,
                    **figures,
                }
            )
            logger.info(
                "real-well QC: checkpoint=%s well=%s (%d/%d) status=%s",
                checkpoint_name,
                well_name,
                index,
                len(groups),
                waveform_status,
            )

    outputs = {
        "real_well_metrics": output_dir / "real_well_metrics.csv",
        "real_well_band_metrics": output_dir / "real_well_band_metrics.csv",
        "real_well_waveform_metrics": output_dir / "real_well_waveform_metrics.csv",
        "real_well_supervised_sampling_qc": output_dir / "real_well_supervised_sampling_qc.csv",
    }
    pd.DataFrame.from_records(metrics_rows).to_csv(
        outputs["real_well_metrics"],
        index=False,
    )
    pd.DataFrame.from_records(band_rows).to_csv(
        outputs["real_well_band_metrics"],
        index=False,
    )
    pd.DataFrame.from_records(waveform_rows).to_csv(
        outputs["real_well_waveform_metrics"],
        index=False,
    )
    support.sampling_qc_frame().to_csv(
        outputs["real_well_supervised_sampling_qc"],
        index=False,
    )
    return outputs


def _well_metrics(
    target_ai: np.ndarray,
    target_increment: np.ndarray,
    pred_ai: np.ndarray,
    pred_increment: np.ndarray,
    twt: np.ndarray,
) -> dict[str, float | int]:
    increment = _basic_metrics(target_increment, pred_increment)
    full = _basic_metrics(target_ai, pred_ai)
    target_increment_rms = _rms(target_increment)
    pred_increment_rms = _rms(pred_increment)
    target_gradient, pred_gradient = _paired_gradients(
        target_increment,
        pred_increment,
        twt,
    )
    return {
        "n_valid": int(increment["n_valid"]),
        "increment_corr": increment["corr"],
        "increment_rmse": increment["rmse"],
        "increment_bias": increment["bias"],
        "full_ai_corr": full["corr"],
        "full_ai_rmse": full["rmse"],
        "full_ai_bias": full["bias"],
        "predicted_increment_rms": pred_increment_rms,
        "target_increment_rms": target_increment_rms,
        "increment_target_relative_log_error": _energy_error(
            pred_increment_rms,
            target_increment_rms,
        ),
        "gradient_rms": _rms(pred_gradient),
        "target_gradient_rms": _rms(target_gradient),
        "gradient_target_relative_log_error": _energy_error(
            _rms(pred_gradient),
            _rms(target_gradient),
        ),
    }


def _basic_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 2:
        return {
            "n_valid": n_valid,
            "corr": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }
    target_valid = np.asarray(target)[valid]
    prediction_valid = np.asarray(prediction)[valid]
    residual = prediction_valid - target_valid
    corr = (
        float(np.corrcoef(target_valid, prediction_valid)[0, 1])
        if np.std(target_valid) > 0.0 and np.std(prediction_valid) > 0.0
        else np.nan
    )
    return {
        "n_valid": n_valid,
        "corr": corr,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
    }


def _well_band_metrics(
    *,
    target_ai: np.ndarray,
    target_increment: np.ndarray,
    pred_ai: np.ndarray,
    pred_increment: np.ndarray,
    sample_axis: np.ndarray,
    sample_domain: str,
    diagnostic_max_hz: float,
) -> dict[str, float | int]:
    if sample_domain == "depth":
        return {}
    if sample_domain != "time":
        raise ValueError(f"Unsupported real-well sample domain: {sample_domain!r}.")
    dt = float(np.nanmedian(np.diff(sample_axis)))
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
            ("increment", target_increment, pred_increment),
            ("full_ai", target_ai, pred_ai),
        ):
            target_band, prediction_band = _fft_band_pair(
                target,
                prediction,
                dt=dt,
                low_hz=low_hz,
                high_hz=high_hz,
            )
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
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    sl = slice(*run)
    output = []
    for values in (
        np.asarray(target, dtype=np.float64)[sl],
        np.asarray(prediction, dtype=np.float64)[sl],
    ):
        centered = values - float(np.mean(values))
        spectrum = np.fft.rfft(centered)
        frequency = np.fft.rfftfreq(values.size, d=dt)
        keep = (frequency >= float(low_hz)) & (frequency < float(high_hz))
        if low_hz <= 0.0:
            keep[0] = True
            spectrum[0] = np.fft.rfft(values)[0]
        output.append(
            np.fft.irfft(np.where(keep, spectrum, 0.0), n=values.size)
        )
    return output[0], output[1]


def _paired_gradients(
    target: np.ndarray,
    prediction: np.ndarray,
    twt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(twt)
    nominal = float(np.nanmedian(dt))
    valid = (
        np.isfinite(target[:-1])
        & np.isfinite(target[1:])
        & np.isfinite(prediction[:-1])
        & np.isfinite(prediction[1:])
        & np.isfinite(dt)
        & (dt > 0.0)
        & (dt <= 1.5 * nominal)
    )
    return np.diff(target)[valid] / dt[valid], np.diff(prediction)[valid] / dt[valid]


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else np.nan


def _energy_error(prediction_rms: float, target_rms: float) -> float:
    if not (
        np.isfinite(prediction_rms)
        and np.isfinite(target_rms)
        and prediction_rms > 0.0
        and target_rms > 0.0
    ):
        return np.nan
    return float(abs(np.log(prediction_rms / target_rms)))


def _write_well_figures(
    *,
    output_dir: Path,
    checkpoint_name: str,
    cluster: int,
    well_name: str,
    well: pd.DataFrame,
    target_ai: np.ndarray,
    target_increment: np.ndarray,
    lfm_ai: np.ndarray,
    pred_increment: np.ndarray,
    volume: RealFieldVolume,
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
    relation: AIVelocityRelation | None,
    tie: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, str], str, dict[str, float | int]]:
    figures_dir = output_dir / "figures" / "wells" / str(cluster)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sample_axis = well["sample"].to_numpy(dtype=np.float64)
    sample_domain = volume.sample_axis.domain
    sample_label = "TVDSS (m)" if sample_domain == "depth" else "TWT (s)"
    pred_ai = lfm_ai + pred_increment
    ai_path = figures_dir / f"{well_name}_{checkpoint_name}_ai_increment_qc.png"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.5, 7.5),
        sharey=True,
        constrained_layout=True,
    )
    for values, label, color in (
        (target_ai, "Canonical well logAI", "black"),
        (lfm_ai, "LFM", "tab:blue"),
        (pred_ai, f"{checkpoint_name} prediction", "tab:red"),
    ):
        axes[0].plot(values, sample_axis, label=label, color=color, lw=1.2)
    for values, label, color in (
        (target_increment, "Well canonical increment", "black"),
        (pred_increment, f"{checkpoint_name} increment", "tab:red"),
    ):
        axes[1].plot(values, sample_axis, label=label, color=color, lw=1.2)
    for axis in axes:
        axis.invert_yaxis()
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(sample_label)
    axes[0].set_xlabel("logAI")
    axes[1].set_xlabel("canonical increment logAI")
    fig.suptitle(
        f"GINN-v2 real-well QC | {well_name} | cluster {cluster} | {checkpoint_name}"
    )
    fig.savefig(ai_path, dpi=180)
    plt.close(fig)

    forward_path = figures_dir / f"{well_name}_{checkpoint_name}_forward_qc.png"
    status, waveform = _write_forward_figure(
        path=forward_path,
        title=(
            f"GINN-v2 real-well forward QC | {well_name} | "
            f"cluster {cluster} | {checkpoint_name}"
        ),
        pred_log_ai=pred_ai,
        well_log_ai=target_ai,
        sample_axis=sample_axis,
        well=well,
        volume=volume,
        wavelet_time_s=wavelet_time_s,
        wavelet=wavelet,
        relation=relation,
        tie=tie,
    )
    return (
        {
            "ai_increment_qc_figure": repo_relative_path(ai_path, root=repo_root),
            "forward_qc_figure": (
                repo_relative_path(forward_path, root=repo_root)
                if status == "ok"
                else ""
            ),
        },
        status,
        waveform,
    )


def _write_forward_figure(
    *,
    path: Path,
    title: str,
    pred_log_ai: np.ndarray,
    well_log_ai: np.ndarray,
    sample_axis: np.ndarray,
    well: pd.DataFrame,
    volume: RealFieldVolume,
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
    relation: AIVelocityRelation | None,
    tie: Mapping[str, Any],
) -> tuple[str, dict[str, float | int]]:
    if sample_axis.size < 9:
        return "insufficient_forward_qc_support", {}
    filled = _fill_nonfinite(pred_log_ai)
    sample_domain = volume.sample_axis.domain
    if sample_domain == "time":
        if relation is not None:
            raise ValueError("Time real-well forward QC rejects an AI--Vp relation.")
        synthetic = forward_time(filled[None, :], wavelet_time_s, wavelet)[0]
    elif sample_domain == "depth":
        if volume.depth_basis != "tvdss" or volume.sample_axis.unit != "m":
            raise ValueError("Depth real-well forward QC requires TVDSS in metres.")
        if relation is None:
            raise ValueError("Depth real-well forward QC requires a frozen AI--Vp relation.")
        velocity = relation.velocity_from_ai(np.exp(filled))
        synthetic = forward_depth(
            filled[None, :],
            velocity[None, :],
            sample_axis,
            wavelet_time_s,
            wavelet,
        )[0]
    else:
        raise ValueError(f"Unsupported real-well sample domain: {sample_domain!r}.")
    observed, inside = sample_volume_trilinear(
        volume.seismic,
        ilines=volume.ilines,
        xlines=volume.xlines,
        twt_s=volume.sample_axis.values,
        inline_values=well["inline"].to_numpy(dtype=np.float64),
        xline_values=well["xline"].to_numpy(dtype=np.float64),
        sample_twt_s=sample_axis,
    )
    valid = inside & np.isfinite(synthetic) & np.isfinite(observed)
    window_suffix = "m" if sample_domain == "depth" else "s"
    start = _number(tie.get(f"tie_window_start_{window_suffix}"))
    stop = _number(tie.get(f"tie_window_end_{window_suffix}"))
    if np.isfinite(start) and np.isfinite(stop):
        valid &= (sample_axis >= start) & (sample_axis <= stop)
    run = _largest_true_run(valid)
    if run is None or run[1] - run[0] < 8:
        return "insufficient_forward_qc_support", {}
    metric_slice = slice(*run)
    plot_start = max(run[0], 1)
    plot_stop = run[1]
    if plot_stop - plot_start < 8:
        return "insufficient_forward_qc_support", {}
    sample_slice = slice(plot_start, plot_stop)
    interface_slice = slice(plot_start - 1, plot_stop - 1)
    basis = sample_axis[sample_slice]
    grid_basis = "tvdss" if sample_domain == "depth" else "twt"
    pred_ai = grid.Log(np.exp(filled[sample_slice]), basis, grid_basis, name="Predicted AI")
    filtered_ai = grid.Log(
        np.exp(_fill_nonfinite(well_log_ai)[sample_slice]),
        basis,
        grid_basis,
        name="Filtered LAS AI",
    )
    reflectivity = grid.Reflectivity(
        reflectivity_from_log_ai(filled)[interface_slice],
        basis,
        grid_basis,
        name="Reflectivity",
    )
    synthetic_trace = grid.Seismic(synthetic[sample_slice], basis, grid_basis, name="Synthetic")
    observed_trace = grid.Seismic(observed[sample_slice], basis, grid_basis, name="Seismic")
    xcorr_values = normalized_xcorr(observed_trace.values, synthetic_trace.values)
    xcorr_basis = synthetic_trace.sampling_rate * np.arange(
        -(synthetic_trace.size - 1),
        synthetic_trace.size,
    )
    lag_basis = "zlag" if sample_domain == "depth" else "tlag"
    xcorr = grid.XCorr(xcorr_values, xcorr_basis, lag_basis, name="XCorr")
    dxcorr = dynamic_normalized_xcorr(observed_trace, synthetic_trace)
    fig, _ = plot_well_waveform_qc(
        [pred_ai, filtered_ai],
        reflectivity,
        synthetic_trace,
        observed_trace,
        xcorr,
        dxcorr,
        figsize=(12.0, 7.5),
        synthetic_ai=pred_ai,
        title=title,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    observed_metrics = observed[metric_slice]
    synthetic_metrics = synthetic[metric_slice]
    raw = waveform_qc_metrics(observed_metrics, synthetic_metrics)
    denominator = float(np.dot(synthetic_metrics, synthetic_metrics))
    positive_scale = (
        max(
            0.0,
            float(
                np.dot(observed_metrics, synthetic_metrics) / denominator
            ),
        )
        if denominator > 0.0
        else np.nan
    )
    scaled = (
        waveform_qc_metrics(
            observed_metrics,
            positive_scale * synthetic_metrics,
        )
        if np.isfinite(positive_scale)
        else {}
    )
    return "ok", {
        **{f"waveform_raw_{key}": value for key, value in raw.items()},
        **{f"waveform_scaled_{key}": value for key, value in scaled.items()},
        "waveform_positive_scale": positive_scale,
    }


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
        if value and start is None:
            start = index
        elif not value and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    return best


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


__all__ = [
    "BalancedRealWellSampler",
    "DifferentiableWellPredictor",
    "RealWellSupervisedSupport",
    "assign_supervision_roles",
    "evaluate_real_wells",
    "prepare_real_well_supervised_support",
]
