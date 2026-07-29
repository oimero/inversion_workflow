"""Stage 1 Step 3 lateral patch contract and structured model.

The module keeps the single-trace HSMM as the only sequence seam.  A lateral
patch contributes masked, metre-based context before that seam; it never
creates a second HSMM or a second segmentation for a neighbour direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cup.synthetic.readers.structured import StructuredParent, StructuredSyntheticBenchmark
from ginn_v2.anchor import (
    LfmAnchoredStructuredSample,
    anchor_to_lfm,
    decode_lfm_anchored_torch,
    load_zone_ai_bounds,
)
from ginn_v2.augmentation import (
    ObservationAugmentationProfile,
    apply_observation_augmentation,
    stable_random_identity,
)
from ginn_v2.data import (
    ParentSplitManifest,
    TeacherForcingBatch,
    TeacherForcingDataModule,
    collate_teacher_forcing_samples,
)
from ginn_v2.hsmm import HsmmPrior, HsmmResult, HsmmSegment, exact_hsmm, hsmm_log_partition, hsmm_path_score
from ginn_v2.model import (
    DirectionalEvidence,
    SingleTraceStructuredModel,
    TeacherForcingLoss,
    TeacherForcingModelConfig,
    TeacherForcingOutput,
    TorchTeacherForcingBatch,
    batch_to_torch,
    project_highres_torch,
    teacher_forcing_loss,
)
from ginn_v2.structure import (
    CenterTracePosterior,
    StructuredLoss,
    StructuredLossConfig,
    _zone_indices,
    balanced_emission_cross_entropy,
    build_predicted_segment_batch,
    soft_boundary_supervision,
    truth_hsmm_segments,
)
from ginn_v2.truth import StructuredTruthAdapter


LATERAL_PATCH_SCHEMA = "structured_ginn_v2_lateral_patch_v1"
LATERAL_RUN_SCHEMA = "structured_ginn_v2_stage1_step3_v1"


def _as_bool_array(value: Any, *, label: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}, got {array.shape}.")
    return np.array(array, dtype=bool, copy=True)


def _validate_lateral_coordinates(parent: StructuredParent) -> np.ndarray:
    values = np.asarray(parent.lateral_m, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("canonical lateral_m must be finite and non-empty.")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    if np.any(np.diff(sorted_values) <= 0.0):
        raise ValueError("canonical lateral_m must be strictly increasing after sorting.")
    if not np.isfinite(float(parent.xline_step)) or float(parent.xline_step) == 0.0:
        raise ValueError("canonical xline_step must be finite and non-zero.")
    return order.astype(np.int64)


def validate_parent_event_identity(parent: StructuredParent) -> dict[str, Any]:
    """Validate producer event identity before any lateral consistency loss."""
    order = _validate_lateral_coordinates(parent)
    rows_by_zone: dict[str, list[Mapping[str, Any]]] = {}
    for row in parent.segments:
        zone_id = str(row["zone_id"])
        rows_by_zone.setdefault(zone_id, []).append(row)
    if not rows_by_zone:
        raise ValueError("parent has no structured segment rows.")
    zone_reports: dict[str, Any] = {}
    for zone_id, rows in sorted(rows_by_zone.items()):
        event_rows: dict[int, list[Mapping[str, Any]]] = {}
        lateral_rows: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            object_id = int(row["object_id"])
            if object_id < 0:
                raise ValueError(f"zone {zone_id!r} contains a negative object_id.")
            event_rows.setdefault(object_id, []).append(row)
            lateral_rows.setdefault(int(row["lateral_index"]), []).append(row)
        if not event_rows:
            raise ValueError(f"zone {zone_id!r} has no explicit event rows.")
        state_by_event: dict[int, int] = {}
        for object_id, event_items in event_rows.items():
            states = {int(item["state_id"]) for item in event_items}
            if len(states) != 1:
                raise ValueError(
                    f"event identity {(zone_id, object_id)!r} changes producer state across lateral."
                )
            state_by_event[object_id] = next(iter(states))
        # object_id is assigned before the producer's lateral loop and is the
        # event-order contract.  Reconstructing order from min(top) is wrong
        # for laterally varying surfaces: a deeper event can have a smaller
        # top coordinate at one lateral than a shallower event elsewhere.
        canonical_order = tuple(sorted(event_rows))
        canonical_position = {object_id: index for index, object_id in enumerate(canonical_order)}
        topology_transition_rows = 0
        checked_laterals = 0
        for lateral_index, lateral_items in lateral_rows.items():
            active = [
                item
                for item in lateral_items
                if float(item["bottom"]) > float(item["top"])
                and float(item["duration_fraction"]) > 0.0
                and int(item.get("duration_samples", 0)) > 0
            ]
            active.sort(key=lambda item: (float(item["top"]), int(item["object_id"])))
            active_ids = [int(item["object_id"]) for item in active]
            if len(active_ids) != len(set(active_ids)):
                raise ValueError(f"zone {zone_id!r} lateral {lateral_index} repeats an object_id.")
            positions = [canonical_position.get(object_id, -1) for object_id in active_ids]
            if any(position < 0 for position in positions) or positions != sorted(positions):
                raise ValueError(
                    f"zone {zone_id!r} object order reverses at lateral {lateral_index}."
                )
            for item in active:
                object_id = int(item["object_id"])
                if int(item["state_id"]) != state_by_event[object_id]:
                    raise ValueError("event state is not stable across lateral.")
            active_set = set(active_ids)
            topology_transition_rows += len(set(canonical_order) - active_set)
            checked_laterals += 1
        zone_reports[zone_id] = {
            "event_count": len(canonical_order),
            "checked_lateral_count": checked_laterals,
            "canonical_event_order": list(canonical_order),
            "state_by_event": {str(key): value for key, value in sorted(state_by_event.items())},
            "birth_death_rows": int(topology_transition_rows),
        }
    return {
        "schema": "structured_ginn_v2_event_identity_v1",
        "parent_id": parent.identity.realization_id,
        "lateral_count": int(order.size),
        "xline_step": float(parent.xline_step),
        "lateral_coordinate_min_m": float(np.min(parent.lateral_m)),
        "lateral_coordinate_max_m": float(np.max(parent.lateral_m)),
        "zones": zone_reports,
    }


def _copy_teacher_batch_rows(
    batches: Sequence[TeacherForcingBatch],
    row_maps: Sequence[np.ndarray],
    *,
    total_rows: int,
) -> TeacherForcingBatch:
    """Stack trace batches while explicitly placing invalid patch rows."""
    if len(batches) != len(row_maps) or not batches:
        raise ValueError("teacher batch and row-map counts must match and be non-empty.")
    model_size = batches[0].seismic.shape[1]
    highres_size = batches[0].background_highres.shape[1]
    factor = int(batches[0].projection_factor)
    maximum_segments = max(batch.segment_valid.shape[1] for batch in batches)
    if total_rows <= 0:
        raise ValueError("total_rows must be positive.")
    for batch, mapping in zip(batches, row_maps, strict=True):
        mapping = np.asarray(mapping, dtype=np.int64).reshape(-1)
        if mapping.size != batch.batch_size or np.any(mapping < 0) or np.any(mapping >= total_rows):
            raise ValueError("teacher row map is incompatible with total_rows.")
        if np.unique(mapping).size != mapping.size:
            raise ValueError("teacher row map contains duplicate target rows.")
        if batch.seismic.shape[1] != model_size or batch.background_highres.shape[1] != highres_size:
            raise ValueError("all lateral traces must share vertical axis sizes.")
        if int(batch.projection_factor) != factor:
            raise ValueError("all lateral traces must share projection_factor.")

    seismic = np.zeros((total_rows, model_size), dtype=np.float32)
    lfm_residual = np.zeros_like(seismic)
    observed_valid = np.zeros((total_rows, model_size), dtype=bool)
    background = np.zeros((total_rows, highres_size), dtype=np.float32)
    truth = np.zeros_like(background)
    zone_valid = np.zeros((total_rows, highres_size), dtype=bool)
    truth_state = np.full((total_rows, highres_size), -1, dtype=np.int64)
    basis = np.zeros((total_rows, maximum_segments, highres_size, 3), dtype=np.float32)
    segment_mask = np.zeros((total_rows, maximum_segments, highres_size), dtype=bool)
    pooling_mask = np.zeros_like(segment_mask)
    segment_valid = np.zeros((total_rows, maximum_segments), dtype=bool)
    state_id = np.zeros((total_rows, maximum_segments), dtype=np.int64)
    duration = np.zeros((total_rows, maximum_segments), dtype=np.float32)
    extent = np.zeros((total_rows, maximum_segments, 2), dtype=np.float32)
    parameters = np.zeros((total_rows, maximum_segments, 3), dtype=np.float32)
    parameter_valid = np.zeros((total_rows, maximum_segments), dtype=bool)
    profile_valid = np.zeros_like(parameter_valid)
    jump_target = np.zeros((total_rows, highres_size), dtype=np.float32)
    jump_valid = np.zeros((total_rows, highres_size), dtype=bool)
    ai_bounds = np.zeros((total_rows, 2), dtype=np.float32)
    projected_truth = np.zeros((total_rows, model_size), dtype=np.float32)
    projected_support = np.zeros((total_rows, model_size), dtype=bool)
    keys = [""] * total_rows
    zone_ids = [""] * total_rows

    def put(target: np.ndarray, source: np.ndarray, mapping: np.ndarray) -> None:
        target[mapping] = source

    for batch, mapping in zip(batches, row_maps, strict=True):
        mapping = np.asarray(mapping, dtype=np.int64)
        put(seismic, batch.seismic, mapping)
        put(lfm_residual, batch.lfm_residual, mapping)
        put(observed_valid, batch.observed_valid, mapping)
        put(background, batch.background_highres, mapping)
        put(truth, batch.truth_highres, mapping)
        put(zone_valid, batch.zone_valid, mapping)
        put(truth_state, batch.truth_state_highres, mapping)
        put(segment_mask[:, : batch.segment_mask.shape[1]], batch.segment_mask, mapping)
        put(pooling_mask[:, : batch.pooling_mask.shape[1]], batch.pooling_mask, mapping)
        put(segment_valid[:, : batch.segment_valid.shape[1]], batch.segment_valid, mapping)
        put(state_id[:, : batch.state_id.shape[1]], batch.state_id, mapping)
        put(duration[:, : batch.duration_fraction.shape[1]], batch.duration_fraction, mapping)
        put(extent[:, : batch.extent_fraction.shape[1]], batch.extent_fraction, mapping)
        put(parameters[:, : batch.target_parameters.shape[1]], batch.target_parameters, mapping)
        put(parameter_valid[:, : batch.parameter_supervision_valid.shape[1]], batch.parameter_supervision_valid, mapping)
        put(profile_valid[:, : batch.profile_supervision_valid.shape[1]], batch.profile_supervision_valid, mapping)
        put(jump_target, batch.interface_jump_target, mapping)
        put(jump_valid, batch.interface_jump_valid, mapping)
        put(ai_bounds, batch.ai_bounds, mapping)
        put(projected_truth, batch.projected_truth, mapping)
        put(projected_support, batch.projected_support, mapping)
        put(basis[:, : batch.segment_basis.shape[1]], batch.segment_basis, mapping)
        for source_index, target_index in enumerate(mapping.tolist()):
            keys[target_index] = str(batch.sample_keys[source_index])
            zone_ids[target_index] = str(batch.zone_ids[source_index])
    return TeacherForcingBatch(
        seismic=seismic,
        lfm_residual=lfm_residual,
        observed_valid=observed_valid,
        background_highres=background,
        truth_highres=truth,
        zone_valid=zone_valid,
        truth_state_highres=truth_state,
        segment_basis=basis,
        segment_mask=segment_mask,
        pooling_mask=pooling_mask,
        segment_valid=segment_valid,
        state_id=state_id,
        duration_fraction=duration,
        extent_fraction=extent,
        target_parameters=parameters,
        parameter_supervision_valid=parameter_valid,
        profile_supervision_valid=profile_valid,
        interface_jump_target=jump_target,
        interface_jump_valid=jump_valid,
        ai_bounds=ai_bounds,
        projected_truth=projected_truth,
        projected_support=projected_support,
        projection_factor=factor,
        sample_keys=tuple(keys),
        zone_ids=tuple(zone_ids),
    )


@dataclass(frozen=True)
class LateralPatch:
    """One fixed-width patch with explicit support and topology masks."""

    trace_batch: TeacherForcingBatch
    relative_lateral_m: np.ndarray
    lateral_valid: np.ndarray
    topology_mask: np.ndarray
    center_index: int
    parent_id: str
    zone_id: str
    augmentation_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        width = self.trace_batch.batch_size
        relative = np.asarray(self.relative_lateral_m, dtype=np.float64).reshape(-1)
        lateral_valid = _as_bool_array(self.lateral_valid, label="lateral_valid")
        topology = _as_bool_array(self.topology_mask, label="topology_mask")
        if relative.shape != (width,) or lateral_valid.shape != (width,):
            raise ValueError("lateral patch coordinate and support shape mismatch.")
        if topology.shape != (width, self.trace_batch.background_highres.shape[1]):
            raise ValueError("topology_mask must be [patch_width, highres].")
        center = int(self.center_index)
        if center < 0 or center >= width or not bool(lateral_valid[center]):
            raise ValueError("lateral patch center must be a valid trace.")
        if not np.isclose(relative[center], 0.0, rtol=0.0, atol=1e-8):
            raise ValueError("relative_lateral_m must use the center trace as zero.")
        if not str(self.parent_id).strip() or not str(self.zone_id).strip():
            raise ValueError("lateral patch identity is incomplete.")
        if not isinstance(self.augmentation_identity, Mapping):
            raise TypeError("augmentation_identity must be a mapping.")
        object.__setattr__(self, "relative_lateral_m", relative)
        object.__setattr__(self, "lateral_valid", lateral_valid)
        object.__setattr__(self, "topology_mask", topology)
        object.__setattr__(self, "center_index", center)
        object.__setattr__(self, "parent_id", str(self.parent_id))
        object.__setattr__(self, "zone_id", str(self.zone_id))
        object.__setattr__(self, "augmentation_identity", dict(self.augmentation_identity))


@dataclass(frozen=True)
class LateralPatchBatch:
    """A batch of patches; trace tensors remain flat for vertical encoder reuse."""

    trace_batch: TeacherForcingBatch
    relative_lateral_m: np.ndarray
    lateral_valid: np.ndarray
    topology_mask: np.ndarray
    center_index: int
    parent_ids: tuple[str, ...]
    zone_ids: tuple[str, ...]
    augmentation_identities: tuple[Mapping[str, Any], ...]

    @property
    def batch_size(self) -> int:
        return len(self.parent_ids)

    @property
    def patch_width(self) -> int:
        return self.trace_batch.batch_size // self.batch_size

    def __post_init__(self) -> None:
        batch_size = len(self.parent_ids)
        width = int(self.trace_batch.batch_size // max(batch_size, 1))
        if batch_size <= 0 or self.trace_batch.batch_size != batch_size * width:
            raise ValueError("lateral trace batch size is not divisible by patch count.")
        relative = np.asarray(self.relative_lateral_m, dtype=np.float64)
        valid = _as_bool_array(self.lateral_valid, label="lateral_valid")
        topology = _as_bool_array(self.topology_mask, label="topology_mask")
        if relative.shape != (batch_size, width) or valid.shape != relative.shape:
            raise ValueError("lateral patch batch coordinate/support shape mismatch.")
        if topology.shape != (batch_size, width, self.trace_batch.background_highres.shape[1]):
            raise ValueError("lateral patch batch topology shape mismatch.")
        if int(self.center_index) < 0 or int(self.center_index) >= width:
            raise ValueError("lateral patch batch center index is invalid.")
        if len(self.zone_ids) != batch_size or len(self.augmentation_identities) != batch_size:
            raise ValueError("lateral patch batch metadata length mismatch.")
        object.__setattr__(self, "relative_lateral_m", relative)
        object.__setattr__(self, "lateral_valid", valid)
        object.__setattr__(self, "topology_mask", topology)
        object.__setattr__(self, "center_index", int(self.center_index))


def collate_lateral_patches(patches: Sequence[LateralPatch]) -> LateralPatchBatch:
    """Collate fixed-width patches without copying invalid edge traces."""
    if not patches:
        raise ValueError("cannot collate an empty lateral patch list.")
    width = patches[0].trace_batch.batch_size
    center = patches[0].center_index
    if any(item.trace_batch.batch_size != width or item.center_index != center for item in patches):
        raise ValueError("lateral patches in one batch must share width and center index.")
    row_maps = [
        np.arange(index * width, (index + 1) * width, dtype=np.int64)
        for index in range(len(patches))
    ]
    trace_batch = _copy_teacher_batch_rows(
        [item.trace_batch for item in patches],
        row_maps,
        total_rows=len(patches) * width,
    )
    return LateralPatchBatch(
        trace_batch=trace_batch,
        relative_lateral_m=np.stack([item.relative_lateral_m for item in patches]),
        lateral_valid=np.stack([item.lateral_valid for item in patches]),
        topology_mask=np.stack([item.topology_mask for item in patches]),
        center_index=center,
        parent_ids=tuple(item.parent_id for item in patches),
        zone_ids=tuple(item.zone_id for item in patches),
        augmentation_identities=tuple(item.augmentation_identity for item in patches),
    )


@dataclass(frozen=True)
class TorchLateralPatchBatch:
    trace_batch: TorchTeacherForcingBatch
    relative_lateral_m: torch.Tensor
    lateral_valid: torch.Tensor
    topology_mask: torch.Tensor
    center_index: int
    parent_ids: tuple[str, ...]
    zone_ids: tuple[str, ...]
    augmentation_identities: tuple[Mapping[str, Any], ...]

    @property
    def batch_size(self) -> int:
        return len(self.parent_ids)

    @property
    def patch_width(self) -> int:
        return self.trace_batch.seismic.shape[0] // self.batch_size


def lateral_patch_to_torch(
    batch: LateralPatchBatch,
    *,
    device: torch.device,
) -> TorchLateralPatchBatch:
    return TorchLateralPatchBatch(
        trace_batch=batch_to_torch(batch.trace_batch, device=device),
        relative_lateral_m=torch.as_tensor(batch.relative_lateral_m, dtype=torch.float32, device=device),
        lateral_valid=torch.as_tensor(batch.lateral_valid, dtype=torch.bool, device=device),
        topology_mask=torch.as_tensor(batch.topology_mask, dtype=torch.bool, device=device),
        center_index=batch.center_index,
        parent_ids=batch.parent_ids,
        zone_ids=batch.zone_ids,
        augmentation_identities=batch.augmentation_identities,
    )


def center_trace_batch(
    batch: TorchLateralPatchBatch,
) -> TorchTeacherForcingBatch:
    """Select the unique center trace for the existing single-trace seam."""
    width = batch.patch_width
    rows = torch.arange(batch.batch_size, device=batch.trace_batch.seismic.device) * width + batch.center_index
    values = {}
    for name in (
        "seismic",
        "lfm_residual",
        "observed_valid",
        "background_highres",
        "truth_highres",
        "zone_valid",
        "truth_state_highres",
        "segment_basis",
        "segment_mask",
        "pooling_mask",
        "segment_valid",
        "state_id",
        "duration_fraction",
        "extent_fraction",
        "target_parameters",
        "parameter_supervision_valid",
        "profile_supervision_valid",
        "interface_jump_target",
        "interface_jump_valid",
        "ai_bounds",
        "projected_truth",
        "projected_support",
    ):
        values[name] = getattr(batch.trace_batch, name).index_select(0, rows)
    values["sample_keys"] = tuple(
        batch.trace_batch.sample_keys[int(row)] for row in rows.detach().cpu().tolist()
    )
    values["zone_ids"] = batch.zone_ids
    values["projection_factor"] = batch.trace_batch.projection_factor
    return replace(batch.trace_batch, **values)


class LateralPatchDataModule:
    """Parent-atomic 21-trace patch reader with explicit topology masks."""

    def __init__(
        self,
        benchmark_dir: str | Path,
        calibration_path: str | Path,
        split_manifest: ParentSplitManifest,
        *,
        patch_width: int = 21,
        augmentation_profile: ObservationAugmentationProfile | None = None,
        dirty_probability: float = 0.5,
        condition_limit: float = 100.0,
    ) -> None:
        if int(patch_width) <= 0 or int(patch_width) % 2 != 1:
            raise ValueError("patch_width must be a positive odd integer.")
        if not 0.0 <= float(dirty_probability) <= 1.0:
            raise ValueError("dirty_probability must be in [0, 1].")
        self.benchmark = StructuredSyntheticBenchmark(benchmark_dir)
        self.ai_bounds = load_zone_ai_bounds(calibration_path)
        self.split_manifest = split_manifest
        self.patch_width = int(patch_width)
        self.center_index = self.patch_width // 2
        self.augmentation_profile = augmentation_profile
        self.dirty_probability = float(dirty_probability)
        self.condition_limit = float(condition_limit)
        benchmark_ids = {item.realization_id for item in self.benchmark.list_parents()}
        manifest_ids = set().union(*(set(split_manifest.parent_ids(name)) for name in ("training", "tuning_validation", "calibration", "geometry_holdout")))
        if benchmark_ids != manifest_ids:
            raise ValueError("split manifest parent set differs from benchmark.")
        if self.augmentation_profile is None and self.dirty_probability > 0.0:
            raise ValueError("dirty patch generation requires an explicit augmentation profile.")

    def _zone_rows(self, parent: StructuredParent) -> dict[str, dict[int, Mapping[str, Any]]]:
        rows: dict[str, dict[int, Mapping[str, Any]]] = {}
        for row in parent.zones:
            zone_id = str(row["zone_id"])
            lateral_index = int(row["lateral_index"])
            if lateral_index in rows.setdefault(zone_id, {}):
                raise ValueError("canonical parent repeats a zone/lateral key.")
            rows[zone_id][lateral_index] = row
        return rows

    def _center_keys(
        self,
        parent: StructuredParent,
        *,
        rng: np.random.Generator,
        samples_per_zone: int | None,
    ) -> list[tuple[int, str]]:
        order = _validate_lateral_coordinates(parent)
        position = {int(lateral): index for index, lateral in enumerate(order.tolist())}
        rows = self._zone_rows(parent)
        keys: list[tuple[int, str]] = []
        for zone_id, by_lateral in sorted(rows.items()):
            candidates = sorted(
                (
                    lateral_index
                    for lateral_index, row in by_lateral.items()
                    if bool(row["zone_valid"])
                    and int(lateral_index) in position
                    and self._trace_is_constructable(
                        parent,
                        zone_id=zone_id,
                        lateral_index=int(lateral_index),
                    )
                ),
                key=lambda index: position[int(index)],
            )
            if not candidates:
                continue
            if samples_per_zone is None or int(samples_per_zone) >= len(candidates):
                selected = candidates
            else:
                count = int(samples_per_zone)
                if count <= 0:
                    raise ValueError("samples_per_zone must be positive when supplied.")
                phase = float(rng.random())
                positions = np.floor(
                    (np.arange(count, dtype=np.float64) + phase)
                    * len(candidates)
                    / count
                ).astype(np.int64)
                selected = [candidates[int(index)] for index in positions]
            keys.extend((int(index), zone_id) for index in selected)
        return keys

    @staticmethod
    def _trace_is_constructable(
        parent: StructuredParent,
        *,
        zone_id: str,
        lateral_index: int,
    ) -> bool:
        """Check the explicit adapter precondition for one zone/lateral trace."""
        zone_rows = [
            row
            for row in parent.zones
            if str(row["zone_id"]) == str(zone_id)
            and int(row["lateral_index"]) == int(lateral_index)
        ]
        if len(zone_rows) != 1 or not bool(zone_rows[0]["zone_valid"]):
            return False
        zone_mask = (
            np.asarray(parent.zone_id_highres[int(lateral_index)])
            == int(zone_rows[0]["zone_grid_value"])
        )
        object_grid = np.asarray(parent.object_id_highres[int(lateral_index)])
        rows = [
            row
            for row in parent.segments
            if str(row["zone_id"]) == str(zone_id)
            and int(row["lateral_index"]) == int(lateral_index)
            and float(row["bottom"]) > float(row["top"])
            and float(row["duration_fraction"]) > 0.0
        ]
        if not rows:
            return False
        return all(
            bool(np.any(zone_mask & (object_grid == int(row["object_id"]))))
            for row in rows
        )

    def _patch(
        self,
        parent: StructuredParent,
        *,
        center_lateral_index: int,
        zone_id: str,
        condition: str,
        random_seed: int,
        boundary_jitter_samples: int = 0,
    ) -> LateralPatch:
        if condition not in {"clean", "dirty"}:
            raise ValueError("patch condition must be clean or dirty.")
        if condition == "dirty" and self.augmentation_profile is None:
            raise ValueError("dirty patch generation requires an augmentation profile.")
        order = _validate_lateral_coordinates(parent)
        position = {int(lateral): index for index, lateral in enumerate(order.tolist())}
        center_position = position[int(center_lateral_index)]
        zone_rows = self._zone_rows(parent).get(str(zone_id), {})
        center_row = zone_rows.get(int(center_lateral_index))
        if center_row is None or not bool(center_row["zone_valid"]):
            raise ValueError("lateral patch center does not have a valid selected zone.")
        relative = np.zeros(self.patch_width, dtype=np.float64)
        lateral_valid = np.zeros(self.patch_width, dtype=bool)
        topology = np.zeros(
            (self.patch_width, parent.highres_axis.coordinates.size),
            dtype=bool,
        )
        samples: list[LfmAnchoredStructuredSample] = []
        sample_positions: list[int] = []
        identity: dict[str, Any] = {
            "schema": LATERAL_PATCH_SCHEMA,
            "condition": condition,
            "seed": int(random_seed),
            "trace_identities": [],
        }
        rng = np.random.default_rng(int(random_seed))
        center_zone_grid = np.asarray(parent.zone_id_highres[int(center_lateral_index)])
        center_object_grid = np.asarray(parent.object_id_highres[int(center_lateral_index)])
        center_state_grid = np.asarray(parent.state_id_highres[int(center_lateral_index)])
        center_zone_mask = center_zone_grid == int(center_row["zone_grid_value"])
        for patch_index, offset in enumerate(
            range(-self.center_index, self.center_index + 1)
        ):
            current_position = center_position + offset
            if current_position < 0 or current_position >= order.size:
                continue
            lateral_index = int(order[current_position])
            row = zone_rows.get(lateral_index)
            if (
                row is None
                or not bool(row["zone_valid"])
                or not self._trace_is_constructable(
                    parent,
                    zone_id=zone_id,
                    lateral_index=lateral_index,
                )
            ):
                continue
            sample = anchor_to_lfm(
                StructuredTruthAdapter.from_structured_parent(
                    parent,
                    zone_id=zone_id,
                    lateral_index=lateral_index,
                ),
                ai_bounds=self.ai_bounds,
                condition_limit=self.condition_limit,
            )
            identity["trace_identities"].append(
                {"lateral_index": lateral_index, "condition": "clean"}
            )
            samples.append(sample)
            sample_positions.append(patch_index)
            relative[patch_index] = float(parent.lateral_m[lateral_index]) - float(parent.lateral_m[center_lateral_index])
            lateral_valid[patch_index] = True
            neighbor_zone_grid = np.asarray(parent.zone_id_highres[lateral_index])
            neighbor_object_grid = np.asarray(parent.object_id_highres[lateral_index])
            neighbor_state_grid = np.asarray(parent.state_id_highres[lateral_index])
            if patch_index != self.center_index:
                topology[patch_index] = (
                    center_zone_mask
                    & (neighbor_zone_grid == int(row["zone_grid_value"]))
                    & np.asarray(parent.truth_valid_highres[int(center_lateral_index)], dtype=bool)
                    & np.asarray(parent.truth_valid_highres[lateral_index], dtype=bool)
                    & (center_object_grid >= 0)
                    & (center_object_grid == neighbor_object_grid)
                    & (center_state_grid == neighbor_state_grid)
                )
        if not lateral_valid[self.center_index]:
            raise RuntimeError("lateral patch lost its center trace.")
        source_batch = collate_teacher_forcing_samples(
            samples,
            boundary_jitter_samples=int(boundary_jitter_samples),
            random_seed=int(rng.integers(0, 2**31 - 1)),
        )
        trace_batch = _copy_teacher_batch_rows(
            [source_batch],
            [np.asarray(sample_positions, dtype=np.int64)],
            total_rows=self.patch_width,
        )
        if condition == "dirty":
            augmented = apply_observation_augmentation(
                trace_batch.seismic,
                trace_batch.observed_valid,
                profile=self.augmentation_profile,
                rng=rng,
                relative_lateral_m=relative,
            )
            trace_batch = replace(trace_batch, seismic=augmented.seismic)
            identity["patch_augmentation"] = dict(augmented.identity)
        return LateralPatch(
            trace_batch=trace_batch,
            relative_lateral_m=relative,
            lateral_valid=lateral_valid,
            topology_mask=topology,
            center_index=self.center_index,
            parent_id=parent.identity.realization_id,
            zone_id=zone_id,
            augmentation_identity=identity,
        )

    def iter_batches(
        self,
        split: str,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        condition: str = "clean",
        samples_per_zone_per_parent: int | None = 1,
        maximum_parents: int | None = None,
        maximum_patches_per_parent: int | None = None,
        boundary_jitter_samples: int = 0,
    ) -> Iterator[LateralPatchBatch]:
        if split not in {"training", "tuning_validation", "calibration", "geometry_holdout"}:
            raise KeyError(f"unknown lateral split {split!r}")
        if condition not in {"clean", "dirty", "mixed"}:
            raise ValueError("condition must be clean, dirty or mixed.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(boundary_jitter_samples) < 0:
            raise ValueError("boundary_jitter_samples must be non-negative.")
        if (
            maximum_patches_per_parent is not None
            and int(maximum_patches_per_parent) <= 0
        ):
            raise ValueError("maximum_patches_per_parent must be positive when supplied.")
        parent_ids = list(self.split_manifest.parent_ids(split))
        if maximum_parents is not None:
            parent_ids = parent_ids[: int(maximum_parents)]
        rng = np.random.default_rng(int(seed))
        if shuffle:
            rng.shuffle(parent_ids)
        for parent_id in parent_ids:
            parent = self.benchmark.read_parent(parent_id)
            validate_parent_event_identity(parent)
            keys = self._center_keys(
                parent,
                rng=rng,
                samples_per_zone=samples_per_zone_per_parent,
            )
            if maximum_patches_per_parent is not None:
                keys = keys[: int(maximum_patches_per_parent)]
            patches: list[LateralPatch] = []
            for lateral_index, zone_id in keys:
                patch_condition = condition
                if condition == "mixed":
                    patch_condition = "dirty" if rng.random() < self.dirty_probability else "clean"
                patch_seed = stable_random_identity(
                    int(seed), parent_id, zone_id, lateral_index, patch_condition
                )
                patches.append(
                    self._patch(
                        parent,
                        center_lateral_index=lateral_index,
                        zone_id=zone_id,
                        condition=patch_condition,
                        random_seed=patch_seed,
                        boundary_jitter_samples=int(boundary_jitter_samples),
                    )
                )
            if shuffle:
                rng.shuffle(patches)
            for start in range(0, len(patches), int(batch_size)):
                group = patches[start : start + int(batch_size)]
                if group:
                    yield collate_lateral_patches(group)

    def iter_parent_batches(
        self,
        split: str,
        *,
        seed: int,
        condition: str = "clean",
        samples_per_zone_per_parent: int | None = 1,
        maximum_parents: int | None = None,
        maximum_patches_per_parent: int | None = None,
    ) -> Iterator[LateralPatchBatch]:
        """Yield one complete patch batch per parent for paired interventions."""
        if split not in {"training", "tuning_validation", "calibration", "geometry_holdout"}:
            raise KeyError(f"unknown lateral split {split!r}")
        if (
            maximum_patches_per_parent is not None
            and int(maximum_patches_per_parent) <= 0
        ):
            raise ValueError("maximum_patches_per_parent must be positive when supplied.")
        parent_ids = list(self.split_manifest.parent_ids(split))
        if maximum_parents is not None:
            parent_ids = parent_ids[: int(maximum_parents)]
        rng = np.random.default_rng(int(seed))
        for parent_id in parent_ids:
            parent = self.benchmark.read_parent(parent_id)
            validate_parent_event_identity(parent)
            keys = self._center_keys(
                parent,
                rng=rng,
                samples_per_zone=samples_per_zone_per_parent,
            )
            if maximum_patches_per_parent is not None:
                keys = keys[: int(maximum_patches_per_parent)]
            patches: list[LateralPatch] = []
            for lateral_index, zone_id in keys:
                patch_condition = condition
                if condition == "mixed":
                    patch_condition = "dirty" if rng.random() < self.dirty_probability else "clean"
                patch_seed = stable_random_identity(
                    int(seed), parent_id, zone_id, lateral_index, patch_condition
                )
                patches.append(
                    self._patch(
                        parent,
                        center_lateral_index=lateral_index,
                        zone_id=zone_id,
                        condition=patch_condition,
                        random_seed=patch_seed,
                        boundary_jitter_samples=0,
                    )
                )
            if patches:
                yield collate_lateral_patches(patches)


@dataclass(frozen=True)
class LateralModelConfig:
    base: TeacherForcingModelConfig
    patch_width: int = 21
    mixer_hidden_channels: int = 64
    mixer_layers: int = 2
    lateral_distance_scale_m: float = 250.0
    feature_consistency_weight: float = 0.05
    state_consistency_weight: float = 0.10

    def __post_init__(self) -> None:
        if int(self.patch_width) <= 0 or int(self.patch_width) % 2 != 1:
            raise ValueError("lateral patch_width must be a positive odd integer.")
        for name in ("mixer_hidden_channels", "mixer_layers"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not np.isfinite(self.lateral_distance_scale_m) or self.lateral_distance_scale_m <= 0.0:
            raise ValueError("lateral_distance_scale_m must be positive.")
        for name in ("feature_consistency_weight", "state_consistency_weight"):
            if not np.isfinite(getattr(self, name)) or float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LateralModelConfig":
        required = {"base"}
        allowed = required | {
            "patch_width",
            "mixer_hidden_channels",
            "mixer_layers",
            "lateral_distance_scale_m",
            "feature_consistency_weight",
            "state_consistency_weight",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(allowed))
        if missing or unknown:
            raise ValueError(f"lateral model mismatch; missing={missing}, unknown={unknown}")
        base = value["base"]
        if not isinstance(base, Mapping):
            raise TypeError("lateral model base must be a mapping.")
        return cls(
            base=TeacherForcingModelConfig.from_mapping(base),
            patch_width=int(value.get("patch_width", 21)),
            mixer_hidden_channels=int(value.get("mixer_hidden_channels", 64)),
            mixer_layers=int(value.get("mixer_layers", 2)),
            lateral_distance_scale_m=float(value.get("lateral_distance_scale_m", 250.0)),
            feature_consistency_weight=float(value.get("feature_consistency_weight", 0.05)),
            state_consistency_weight=float(value.get("state_consistency_weight", 0.10)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["base"] = self.base.to_dict()
        return payload


class MaskedLateralMixer(nn.Module):
    """A distance-aware attention mixer with explicit lateral support masking."""

    def __init__(self, channels: int, hidden_channels: int, layers: int, distance_scale_m: float) -> None:
        super().__init__()
        self.distance_scale_m = float(distance_scale_m)
        self.query = nn.Linear(channels, hidden_channels)
        self.key = nn.Linear(channels, hidden_channels)
        self.value = nn.Linear(channels, hidden_channels)
        self.output = nn.Linear(hidden_channels, channels)
        self.distance_bias = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )
        self.norm = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
            for _ in range(int(layers))
        )

    def forward(
        self,
        features: torch.Tensor,
        relative_lateral_m: torch.Tensor,
        lateral_valid: torch.Tensor,
        *,
        center_index: int,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("lateral mixer features must be [batch, width, channels, vertical].")
        batch_size, width, channels, vertical = features.shape
        if relative_lateral_m.shape != (batch_size, width) or lateral_valid.shape != (batch_size, width):
            raise ValueError("lateral mixer coordinate/support shapes do not match features.")
        if not bool(torch.all(lateral_valid[:, int(center_index)]).item()):
            raise ValueError("lateral mixer requires a valid center trace in every patch.")
        tokens = features.permute(0, 1, 3, 2)
        center = tokens[:, int(center_index)]
        query = self.query(center)
        key = self.key(tokens)
        value = self.value(tokens)
        scores = torch.einsum("bvh,bwvh->bwv", query, key)
        scores = scores / float(np.sqrt(max(query.shape[-1], 1)))
        distance = (relative_lateral_m / float(self.distance_scale_m)).unsqueeze(-1)
        scores = scores + self.distance_bias(distance).squeeze(-1).unsqueeze(-1)
        scores = scores.masked_fill(~lateral_valid.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        attended = torch.einsum("bwv,bwvh->bvh", weights, value)
        mixed = self.norm(center + self.output(attended))
        mixed = mixed + self.feed_forward(mixed)
        for layer in self.layers:
            mixed = mixed + layer(mixed)
        output = mixed.permute(0, 2, 1)
        if output.shape != (batch_size, channels, vertical):
            raise RuntimeError("lateral mixer returned an unexpected shape.")
        return output


class LateralStructuredModel(nn.Module):
    """Step-3 model: vertical evidence, masked lateral context, one center HSMM."""

    def __init__(self, config: LateralModelConfig) -> None:
        super().__init__()
        if not config.base.predict_interface_evidence:
            raise ValueError("lateral model requires the Step-2 interface evidence head.")
        self.config = config
        self.vertical_model = SingleTraceStructuredModel(config.base)
        self.lateral_mixer = MaskedLateralMixer(
            config.base.feature_channels,
            config.mixer_hidden_channels,
            config.mixer_layers,
            config.lateral_distance_scale_m,
        )

    @classmethod
    def from_step2_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
        patch_width: int = 21,
        mixer_hidden_channels: int = 64,
        mixer_layers: int = 2,
        lateral_distance_scale_m: float = 250.0,
        feature_consistency_weight: float = 0.05,
        state_consistency_weight: float = 0.10,
    ) -> tuple["LateralStructuredModel", Mapping[str, Any], Mapping[str, Any]]:
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
        if checkpoint.get("schema") != "structured_ginn_v2_stage1_step2_v3":
            raise ValueError("lateral model requires a complete Stage-1 Step-2 v3 checkpoint.")
        base_config = TeacherForcingModelConfig.from_mapping(checkpoint["model_config"])
        model = cls(
            LateralModelConfig(
                base=base_config,
                patch_width=patch_width,
                mixer_hidden_channels=mixer_hidden_channels,
                mixer_layers=mixer_layers,
                lateral_distance_scale_m=lateral_distance_scale_m,
                feature_consistency_weight=feature_consistency_weight,
                state_consistency_weight=state_consistency_weight,
            )
        ).to(device)
        model.vertical_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.freeze_profile_modules()
        model.eval()
        return model, checkpoint, base_config.to_dict()

    @classmethod
    def from_step3_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device,
    ) -> tuple["LateralStructuredModel", Mapping[str, Any]]:
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
        if checkpoint.get("schema") != LATERAL_RUN_SCHEMA:
            raise ValueError("unsupported or incomplete Stage-1 Step-3 checkpoint.")
        model = cls(LateralModelConfig.from_mapping(checkpoint["model_config"])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.freeze_profile_modules()
        model.eval()
        return model, checkpoint

    def freeze_profile_modules(self) -> None:
        self.vertical_model.freeze_teacher_forcing_modules()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def encode_patch(self, patch: TorchLateralPatchBatch) -> DirectionalEvidence:
        batch = patch.trace_batch
        batch_size = patch.batch_size
        width = patch.patch_width
        features = self.vertical_model.encode_trace(batch)
        jump_mean, jump_std, polarity_logits = self.vertical_model._interface_evidence(features, batch)
        if jump_mean is None or jump_std is None or polarity_logits is None:
            raise RuntimeError("lateral model did not produce interface evidence.")
        feature_channels = features.shape[1]
        vertical = features.shape[-1]
        feature_grid = features.reshape(batch_size, width, feature_channels, vertical)
        mixed = self.lateral_mixer(
            feature_grid,
            patch.relative_lateral_m,
            patch.lateral_valid,
            center_index=patch.center_index,
        )
        structure_features = self.vertical_model.structure_encoder(mixed)
        emission = self.vertical_model.emission_head(structure_features).transpose(1, 2)
        jump_mean_grid = jump_mean.reshape(batch_size, width, vertical)
        jump_std_grid = jump_std.reshape(batch_size, width, vertical)
        polarity_grid = polarity_logits.reshape(batch_size, width, 3, vertical)
        center_jump_mean = jump_mean_grid[:, patch.center_index]
        center_jump_std = jump_std_grid[:, patch.center_index]
        center_polarity = polarity_grid[:, patch.center_index]
        interface_cues = torch.cat(
            (
                (center_jump_mean / self.config.base.maximum_interface_jump_magnitude).unsqueeze(1),
                (center_jump_std / self.config.base.maximum_interface_jump_std).unsqueeze(1),
                torch.softmax(center_polarity, dim=1),
            ),
            dim=1,
        )
        boundary = self.vertical_model.boundary_head(
            torch.cat((structure_features, interface_cues), dim=1)
        ).squeeze(1)
        center_batch = center_trace_batch(patch)
        valid = center_batch.zone_valid
        return DirectionalEvidence(
            emission_log_potential=torch.where(valid.unsqueeze(-1), emission, torch.zeros_like(emission)),
            boundary_log_potential=torch.where(valid, boundary, torch.zeros_like(boundary)),
            interface_polarity_log_potential=torch.where(
                valid.unsqueeze(-1), center_polarity.transpose(1, 2), torch.zeros_like(center_polarity.transpose(1, 2))
            ),
            interface_jump_mean=torch.where(valid, center_jump_mean, torch.zeros_like(center_jump_mean)),
            interface_jump_std=torch.where(valid, center_jump_std, torch.ones_like(center_jump_std)),
            # The frozen Step-2 parameter head consumes the original center
            # encoder feature distribution.  Lateral mixing is reserved for
            # structure/HSMM evidence; passing ``mixed`` here silently
            # changes the profile-head coordinate system.
            center_feature_sequence=feature_grid[:, patch.center_index],
            lateral_feature_sequence=feature_grid,
            lateral_valid=patch.lateral_valid,
        )

    def parameterize_segments(
        self,
        feature_sequence: torch.Tensor,
        externally_supplied_segments: TorchTeacherForcingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vertical_model.parameterize_segments(feature_sequence, externally_supplied_segments)


def _teacher_output_from_lateral_evidence(
    model: LateralStructuredModel,
    evidence: DirectionalEvidence,
    batch: TorchTeacherForcingBatch,
) -> TeacherForcingOutput:
    mean, std = model.parameterize_segments(evidence.center_feature_sequence, batch)
    decoded = decode_lfm_anchored_torch(
        batch.background_highres,
        batch.segment_basis,
        batch.segment_mask,
        mean,
        batch.zone_valid,
        batch.ai_bounds,
    )
    projected, support = project_highres_torch(
        decoded,
        batch.zone_valid,
        factor=batch.projection_factor,
    )
    return TeacherForcingOutput(
        parameter_mean=mean,
        parameter_std=std,
        decoded_highres=decoded,
        projected_log_ai=projected,
        projection_support=support,
        interface_jump_mean=evidence.interface_jump_mean,
        interface_jump_std=evidence.interface_jump_std,
        interface_polarity_logits=evidence.interface_polarity_log_potential.transpose(1, 2),
    )


@dataclass(frozen=True)
class LateralLossConfig:
    structure: StructuredLossConfig
    feature_consistency_weight: float = 0.05
    state_consistency_weight: float = 0.10

    def __post_init__(self) -> None:
        for name in ("feature_consistency_weight", "state_consistency_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True)
class LateralLoss:
    total: torch.Tensor
    structure: StructuredLoss
    feature_consistency: torch.Tensor
    state_consistency: torch.Tensor


def _masked_tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(torch.any(mask).item()):
        return values.new_zeros(())
    return values[mask].mean()


def lateral_structured_training_loss(
    model: LateralStructuredModel,
    patch: TorchLateralPatchBatch,
    prior: HsmmPrior,
    config: LateralLossConfig,
) -> tuple[LateralLoss, DirectionalEvidence, TeacherForcingOutput]:
    """Train the center path and add only topology-valid lateral consistency."""
    center = center_trace_batch(patch)
    evidence = model.encode_patch(patch)
    teacher_output = _teacher_output_from_lateral_evidence(model, evidence, center)
    teacher_loss = teacher_forcing_loss(teacher_output, center, config.structure.teacher_forcing)
    state_mask = center.zone_valid & (center.truth_state_highres >= 0)
    emission_loss = balanced_emission_cross_entropy(
        evidence.emission_log_potential,
        center.truth_state_highres,
        state_mask,
    )
    boundary_target, boundary_valid = soft_boundary_supervision(center)
    zone_losses: list[torch.Tensor] = []
    for batch_index, zone_id in enumerate(center.zone_ids):
        zone = _zone_indices(center.zone_valid[batch_index])
        truth_segments = truth_hsmm_segments(center, batch_index)
        local_emission = evidence.emission_log_potential[batch_index, zone]
        local_boundary = torch.zeros_like(evidence.boundary_log_potential[batch_index, zone])
        zone_prior = prior.zone(zone_id)
        zone_losses.append(
            (
                hsmm_log_partition(local_emission, local_boundary, zone_prior)
                - hsmm_path_score(local_emission, local_boundary, zone_prior, truth_segments)
            )
            / float(zone.numel())
        )
    positive = boundary_valid & (boundary_target > 0.0)
    negative = boundary_valid & (boundary_target == 0.0)
    if not bool(torch.any(positive).item()) or not bool(torch.any(negative).item()):
        raise ValueError("lateral boundary supervision requires both classes.")
    boundary_terms = F.binary_cross_entropy_with_logits(
        evidence.boundary_log_potential,
        boundary_target,
        reduction="none",
    )
    boundary_loss = 0.5 * (boundary_terms[positive].mean() + boundary_terms[negative].mean())
    hsmm_nll = torch.mean(torch.stack(zone_losses))
    structure_total = (
        config.structure.teacher_forcing_weight * teacher_loss.total
        + config.structure.emission_weight * emission_loss
        + config.structure.boundary_weight * boundary_loss
        + config.structure.hsmm_nll_weight * hsmm_nll
    )
    structure = StructuredLoss(
        total=structure_total,
        emission_cross_entropy=emission_loss,
        boundary_binary_cross_entropy=boundary_loss,
        hsmm_negative_log_likelihood=hsmm_nll,
        teacher_forcing=teacher_loss,
        zone_count=len(zone_losses),
    )
    topology = patch.topology_mask
    center_state = center.truth_state_highres.unsqueeze(1).expand_as(topology)
    topology_any = topology.any(dim=1)
    state_terms = F.cross_entropy(
        evidence.emission_log_potential.transpose(1, 2),
        torch.clamp(center.truth_state_highres, min=0, max=2),
        reduction="none",
    )
    state_consistency = _masked_tensor_mean(state_terms, topology_any & state_mask)
    feature_consistency = evidence.center_feature_sequence.new_zeros(())
    if evidence.lateral_feature_sequence is not None and bool(torch.any(topology).item()):
        center_features = evidence.lateral_feature_sequence[:, patch.center_index].unsqueeze(1)
        feature_difference = (evidence.lateral_feature_sequence - center_features).square().mean(dim=2)
        feature_consistency = _masked_tensor_mean(feature_difference, topology)
    total = (
        structure.total
        + float(config.state_consistency_weight) * state_consistency
        + float(config.feature_consistency_weight) * feature_consistency
    )
    return (
        LateralLoss(
            total=total,
            structure=structure,
            feature_consistency=feature_consistency,
            state_consistency=state_consistency,
        ),
        evidence,
        teacher_output,
    )


def infer_lateral_patch(
    model: LateralStructuredModel,
    patch: TorchLateralPatchBatch,
    prior: HsmmPrior,
    *,
    evidence: DirectionalEvidence | None = None,
) -> CenterTracePosterior:
    """Run one fused-width patch through one center exact HSMM."""
    center = center_trace_batch(patch)
    directional = model.encode_patch(patch) if evidence is None else evidence
    batch_size, highres_size = center.zone_valid.shape
    map_state = torch.full((batch_size, highres_size), -1, dtype=torch.long, device=center.zone_valid.device)
    state_marginal = torch.zeros((batch_size, highres_size, 3), dtype=directional.emission_log_potential.dtype, device=center.zone_valid.device)
    boundary_marginal = torch.zeros((batch_size, highres_size), dtype=directional.emission_log_potential.dtype, device=center.zone_valid.device)
    results: list[HsmmResult] = []
    for batch_index, zone_id in enumerate(center.zone_ids):
        zone = _zone_indices(center.zone_valid[batch_index])
        result = exact_hsmm(
            directional.emission_log_potential[batch_index, zone],
            torch.zeros_like(directional.boundary_log_potential[batch_index, zone]),
            prior.zone(zone_id),
        )
        results.append(result)
        state_marginal[batch_index, zone] = result.state_marginal
        boundary_marginal[batch_index, zone] = result.boundary_marginal
        for segment in result.consensus_segments:
            map_state[batch_index, zone[segment.start : segment.stop]] = segment.state_id
    predicted_batch = build_predicted_segment_batch(center, [result.consensus_segments for result in results])
    predicted_output = _teacher_output_from_lateral_evidence(model, directional, predicted_batch)
    return CenterTracePosterior(
        evidence=directional,
        hsmm_results=tuple(results),
        map_state_highres=map_state,
        state_marginal_highres=state_marginal,
        boundary_marginal_highres=boundary_marginal,
        predicted_segment_batch=predicted_batch,
        predicted_profile=predicted_output,
    )


__all__ = [
    "LATERAL_PATCH_SCHEMA",
    "LATERAL_RUN_SCHEMA",
    "LateralLoss",
    "LateralLossConfig",
    "LateralModelConfig",
    "LateralPatch",
    "LateralPatchBatch",
    "LateralPatchDataModule",
    "LateralStructuredModel",
    "MaskedLateralMixer",
    "TorchLateralPatchBatch",
    "center_trace_batch",
    "collate_lateral_patches",
    "infer_lateral_patch",
    "lateral_patch_to_torch",
    "lateral_structured_training_loss",
    "validate_parent_event_identity",
]
