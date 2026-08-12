"""Frozen spatial and well-support identities for body inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

import numpy as np

from cup.seismic.geometry import SurveyLineGeometry
from cup.well.real_field_controls import WellControl, WellControlSet
from cup.well.scale_separation import gaussian_smooth_finite_runs_numpy
from ginn_v2.data import Orientation, PatchKey
from ginn_v2.projector import BodyScaleProjector, project_well_target


ValidationAnchor = Literal["maxmax", "maxmin", "minmax", "minmin", "center"]


@dataclass(frozen=True)
class SpatialSplit:
    train_keys: tuple[PatchKey, ...]
    validation_keys: tuple[PatchKey, ...]
    review_keys: tuple[PatchKey, ...]
    validation_centers: tuple[tuple[int, int], ...]
    block_xy_m: tuple[float, float, float, float]
    gap_m: float
    anchor: ValidationAnchor

    def __post_init__(self) -> None:
        train_centers = {(item.inline_index, item.xline_index) for item in self.train_keys}
        validation_centers = {(item.inline_index, item.xline_index) for item in self.validation_keys}
        review_centers = {(item.inline_index, item.xline_index) for item in self.review_keys}
        if train_centers & review_centers:
            raise ValueError("Spatial train and validation centers overlap.")
        if not self.train_keys or not self.validation_keys or not self.review_keys:
            raise ValueError("Spatial split must contain non-empty train and validation keys.")
        if not validation_centers <= review_centers:
            raise ValueError("Metric validation identities must be a subset of the review block.")
        if len(self.block_xy_m) != 4 or any(not np.isfinite(float(value)) for value in self.block_xy_m):
            raise ValueError("Spatial validation block must contain four finite metre bounds.")
        if self.gap_m < 0.0 or not np.isfinite(float(self.gap_m)):
            raise ValueError("Spatial split gap_m must be finite and non-negative.")


@dataclass(frozen=True)
class WellSampleSplit:
    well_name: str
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        train = set(self.train_indices)
        validation = set(self.validation_indices)
        if not train or train & validation:
            raise ValueError("Well split requires non-empty training indices and disjoint validation indices.")


@dataclass(frozen=True)
class WellTarget:
    well_name: str
    model_axis_target: np.ndarray
    valid_target_mask: np.ndarray
    native_body_target: np.ndarray

    def __post_init__(self) -> None:
        target = np.asarray(self.model_axis_target, dtype=np.float64)
        mask = np.asarray(self.valid_target_mask, dtype=bool)
        native = np.asarray(self.native_body_target, dtype=np.float64)
        if target.ndim != 1 or mask.shape != target.shape or native.ndim != 1:
            raise ValueError("WellTarget arrays have invalid dimensions.")
        if np.any(mask & ~np.isfinite(target)) or np.any(np.isfinite(target) & ~mask):
            raise ValueError("WellTarget mask must exactly describe finite model-axis targets.")
        object.__setattr__(self, "model_axis_target", target)
        object.__setattr__(self, "valid_target_mask", mask)
        object.__setattr__(self, "native_body_target", native)


@dataclass(frozen=True)
class WellPatchTarget:
    well_name: str
    patch_key: PatchKey
    target_values: np.ndarray
    target_mask: np.ndarray
    target_scale: float

    def __post_init__(self) -> None:
        if not str(self.well_name).strip():
            raise ValueError("WellPatchTarget.well_name must be non-empty.")
        values = np.asarray(self.target_values, dtype=np.float64)
        mask = np.asarray(self.target_mask, dtype=bool)
        if values.ndim != 1 or mask.shape != values.shape or not np.any(mask):
            raise ValueError("WellPatchTarget requires a non-empty one-dimensional target mask.")
        if np.any(~np.isfinite(values)):
            raise ValueError("WellPatchTarget.target_values must be finite; target_mask carries support.")
        if not np.isfinite(float(self.target_scale)) or float(self.target_scale) <= 0.0:
            raise ValueError("WellPatchTarget.target_scale must be finite and positive.")
        object.__setattr__(self, "target_values", values)
        object.__setattr__(self, "target_mask", mask)


def _center_xy(key: PatchKey, geometry: SurveyLineGeometry) -> tuple[float, float]:
    inline = geometry.inline_axis.line_at_index(key.inline_index)
    xline = geometry.xline_axis.line_at_index(key.xline_index)
    return geometry.line_to_coord(inline, xline)


def make_spatial_split(
    keys: Iterable[PatchKey],
    *,
    geometry: SurveyLineGeometry,
    validation_fraction: float,
    gap_m: float,
    anchor: ValidationAnchor,
) -> SpatialSplit:
    """Select a fixed XY block using physical coordinates rather than line steps."""

    candidates = tuple(keys)
    if not candidates:
        raise ValueError("Cannot split an empty patch identity set.")
    fraction = float(validation_fraction)
    gap = float(gap_m)
    if not 0.0 < fraction < 1.0 or not np.isfinite(fraction):
        raise ValueError("validation_fraction must be finite and within (0, 1).")
    if gap < 0.0 or not np.isfinite(gap):
        raise ValueError("gap_m must be finite and non-negative.")
    if anchor not in {"maxmax", "maxmin", "minmax", "minmin", "center"}:
        raise ValueError("Unsupported validation block anchor.")
    centers = tuple(sorted({(item.inline_index, item.xline_index) for item in candidates}))
    xy = np.asarray([_center_xy(PatchKey(i, j), geometry) for i, j in centers], dtype=np.float64)
    x_min, y_min = np.min(xy, axis=0)
    x_max, y_max = np.max(xy, axis=0)
    x_span = max(float(x_max - x_min), geometry.bin_spacing_m()["nominal_bin_spacing_m"])
    y_span = max(float(y_max - y_min), geometry.bin_spacing_m()["nominal_bin_spacing_m"])
    side = float(np.sqrt(fraction))
    block_x_span = x_span * side
    block_y_span = y_span * side
    if anchor == "center":
        x_start = float((x_min + x_max - block_x_span) / 2.0)
        y_start = float((y_min + y_max - block_y_span) / 2.0)
    else:
        x_start = float(x_max - block_x_span if anchor.startswith("max") else x_min)
        y_start = float(y_max - block_y_span if anchor.endswith("max") else y_min)
    x_stop = x_start + block_x_span
    y_stop = y_start + block_y_span
    center_validation: set[tuple[int, int]] = set()
    for (i, j), (x_value, y_value) in zip(centers, xy):
        if x_start <= x_value <= x_stop and y_start <= y_value <= y_stop:
            center_validation.add((i, j))
    if not center_validation or len(center_validation) == len(centers):
        raise ValueError("Spatial validation block does not produce both train and validation centers.")
    train_centers: set[tuple[int, int]] = set()
    for (i, j), (x_value, y_value) in zip(centers, xy):
        outside_gap = (
            x_value < x_start - gap
            or x_value > x_stop + gap
            or y_value < y_start - gap
            or y_value > y_stop + gap
        )
        if outside_gap:
            train_centers.add((i, j))
    if not train_centers:
        raise ValueError("Spatial gap removes every training center.")
    train_keys = tuple(item for item in candidates if (item.inline_index, item.xline_index) in train_centers)
    validation_keys = tuple(
        item for item in candidates if (item.inline_index, item.xline_index) in center_validation
    )
    return SpatialSplit(
        train_keys=train_keys,
        validation_keys=validation_keys,
        review_keys=validation_keys,
        validation_centers=tuple(sorted(center_validation)),
        block_xy_m=(x_start, x_stop, y_start, y_stop),
        gap_m=gap,
        anchor=anchor,
    )


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    return [tuple(item) for item in np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)]


def sample_lfm_trace(
    control: WellControl,
    *,
    lfm_log_ai: np.ndarray,
    lfm_valid_mask: np.ndarray,
    geometry: SurveyLineGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the volume LFM at the nearest physical survey center per well sample."""

    values = np.asarray(lfm_log_ai, dtype=np.float64)
    support = np.asarray(lfm_valid_mask, dtype=bool)
    expected = (
        geometry.inline_axis.count,
        geometry.xline_axis.count,
        control.sample_axis.values.size,
    )
    if values.shape != expected or support.shape != expected:
        raise ValueError(f"lfm_log_ai and lfm_valid_mask must have shape {expected}.")
    result = np.full(control.sample_axis.values.shape, np.nan, dtype=np.float64)
    result_mask = np.zeros(control.sample_axis.values.shape, dtype=bool)
    for sample_index, (x_m, y_m) in enumerate(zip(control.x_m_by_sample, control.y_m_by_sample)):
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            continue
        i_float, j_float = geometry.coord_to_index(float(x_m), float(y_m))
        i, j = int(round(i_float)), int(round(j_float))
        if (
            0 <= i < geometry.inline_axis.count
            and 0 <= j < geometry.xline_axis.count
            and abs(i_float - i) <= 0.5
            and abs(j_float - j) <= 0.5
            and support[i, j, sample_index]
            and np.isfinite(values[i, j, sample_index])
        ):
            result[sample_index] = values[i, j, sample_index]
            result_mask[sample_index] = True
    return result, result_mask


def well_target_zone_mask(
    control: WellControl,
    *,
    geometry: SurveyLineGeometry,
    target_zone_mask: np.ndarray,
) -> np.ndarray:
    """Sample the interpreted target-zone support along one well trajectory."""

    volume = np.asarray(target_zone_mask, dtype=bool)
    expected = (
        geometry.inline_axis.count,
        geometry.xline_axis.count,
        control.sample_axis.values.size,
    )
    if volume.shape != expected:
        raise ValueError(f"target_zone_mask must have shape {expected}.")
    result = np.zeros(control.sample_axis.values.shape, dtype=bool)
    for sample_index, (x_m, y_m) in enumerate(zip(control.x_m_by_sample, control.y_m_by_sample)):
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            continue
        i_float, j_float = geometry.coord_to_index(float(x_m), float(y_m))
        i, j = int(round(i_float)), int(round(j_float))
        if (
            0 <= i < geometry.inline_axis.count
            and 0 <= j < geometry.xline_axis.count
            and abs(i_float - i) <= 0.5
            and abs(j_float - j) <= 0.5
        ):
            result[sample_index] = volume[i, j, sample_index]
    return result


def build_well_body_target(
    control: WellControl,
    *,
    body_smoothing_fwhm_m: float,
    target_zone_support: np.ndarray,
    lfm_log_ai: np.ndarray,
    lfm_valid_mask: np.ndarray,
    geometry: SurveyLineGeometry,
    projector: BodyScaleProjector,
) -> WellTarget:
    """Build a trusted-well target in the model's configured body band."""

    native_coordinates = np.asarray(control.native.coordinates, dtype=np.float64)
    native_values = np.asarray(control.native.full_log_ai, dtype=np.float64)
    native_body = gaussian_smooth_finite_runs_numpy(
        native_values,
        native_coordinates,
        fwhm_m=body_smoothing_fwhm_m,
    )
    model_axis = np.asarray(control.sample_axis.values, dtype=np.float64)
    model_target = np.full(model_axis.shape, np.nan, dtype=np.float64)
    for start, stop in _finite_runs(np.isfinite(native_body)):
        inside = (model_axis >= native_coordinates[start]) & (model_axis <= native_coordinates[stop - 1])
        model_target[inside] = np.interp(
            model_axis[inside],
            native_coordinates[start:stop],
            native_body[start:stop],
        )
    observed = np.asarray(control.observed_valid_mask, dtype=bool)
    zone_support = np.asarray(target_zone_support, dtype=bool)
    if zone_support.shape != observed.shape:
        raise ValueError("target_zone_support must match the well model axis.")
    well_lfm, well_lfm_valid = sample_lfm_trace(
        control,
        lfm_log_ai=lfm_log_ai,
        lfm_valid_mask=lfm_valid_mask,
        geometry=geometry,
    )
    valid_target = observed & zone_support & well_lfm_valid & np.isfinite(model_target)
    model_target[~valid_target] = np.nan
    if np.count_nonzero(valid_target) < 4:
        raise ValueError(f"{control.well_name}: native body target has fewer than four observed model samples.")
    model_target = project_well_target(
        model_target,
        valid_target,
        well_lfm,
        well_lfm_valid,
        model_axis,
        projector,
    )
    return WellTarget(
        well_name=control.well_name,
        model_axis_target=model_target,
        valid_target_mask=valid_target,
        native_body_target=native_body,
    )


def build_well_splits(
    controls: WellControlSet,
    targets: Mapping[str, WellTarget],
) -> tuple[WellSampleSplit, ...]:
    """Use every trusted observed target sample as an anchor sample."""

    result: list[WellSampleSplit] = []
    for control in controls.controls:
        target = targets.get(control.well_name)
        if target is None:
            raise ValueError(f"Missing trusted well target: {control.well_name}")
        observed = target.valid_target_mask
        runs = _finite_runs(observed)
        if not runs:
            raise ValueError(f"{control.well_name}: no observed body-target run for the well split.")
        train_indices = tuple(int(index) for index in np.flatnonzero(observed))
        result.append(
            WellSampleSplit(
                well_name=control.well_name,
                train_indices=train_indices,
                validation_indices=(),
            )
        )
    return tuple(result)


def _nearest_patch_index(control: WellControl, sample_index: int, geometry: SurveyLineGeometry) -> tuple[int, int]:
    x_value = float(control.x_m_by_sample[sample_index])
    y_value = float(control.y_m_by_sample[sample_index])
    i_float, j_float = geometry.coord_to_index(x_value, y_value)
    i = int(round(i_float))
    j = int(round(j_float))
    if abs(i_float - i) > 0.5 or abs(j_float - j) > 0.5:
        raise ValueError(f"{control.well_name}: well sample is farther than half a trace from a seismic center.")
    return i, j


def build_well_patch_targets(
    controls: WellControlSet,
    targets: Mapping[str, WellTarget],
    splits: Iterable[WellSampleSplit],
    *,
    geometry: SurveyLineGeometry,
    subset: Literal["train", "validation"],
    orientations: Iterable[Orientation] = ("inline", "xline"),
) -> tuple[WellPatchTarget, ...]:
    """Group well targets by the actual center trace used by each sample."""

    control_by_name = {item.well_name: item for item in controls.controls}
    split_items = tuple(splits)
    selected_orientations = tuple(orientations)
    if not selected_orientations or any(item not in {"inline", "xline"} for item in selected_orientations):
        raise ValueError("well orientations must contain inline and/or xline.")
    result: list[WellPatchTarget] = []
    for split in split_items:
        control = control_by_name.get(split.well_name)
        target = targets.get(split.well_name)
        if control is None or target is None:
            raise ValueError(f"Well split references an unknown target: {split.well_name}")
        indices = split.train_indices if subset == "train" else split.validation_indices
        target_scale = float(np.std(target.model_axis_target[target.valid_target_mask]))
        if not np.isfinite(target_scale) or target_scale <= 0.0:
            raise ValueError(f"{split.well_name}: body target has no positive normalization scale.")
        grouped: dict[PatchKey, list[int]] = {}
        for sample_index in indices:
            i, j = _nearest_patch_index(control, sample_index, geometry)
            for orientation in selected_orientations:
                grouped.setdefault(PatchKey(i, j, orientation), []).append(int(sample_index))
        for patch_key, sample_indices in sorted(grouped.items()):
            values = np.zeros(target.model_axis_target.shape, dtype=np.float64)
            mask = np.zeros(target.model_axis_target.shape, dtype=bool)
            selected = np.asarray(sample_indices, dtype=np.int64)
            values[selected] = target.model_axis_target[selected]
            mask[selected] = True
            result.append(
                WellPatchTarget(
                    well_name=split.well_name,
                    patch_key=patch_key,
                    target_values=values,
                    target_mask=mask,
                    target_scale=target_scale,
                )
            )
    if not result:
        raise ValueError(f"No {subset} well patch targets were produced.")
    return tuple(result)


__all__ = [
    "SpatialSplit",
    "ValidationAnchor",
    "WellPatchTarget",
    "WellSampleSplit",
    "WellTarget",
    "build_well_body_target",
    "build_well_patch_targets",
    "build_well_splits",
    "sample_lfm_trace",
    "well_target_zone_mask",
    "make_spatial_split",
]
