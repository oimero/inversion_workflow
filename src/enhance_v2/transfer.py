"""Continuous dictionary regression and physical-space texture transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cup.seismic.geometry import LineAxis, SampleAxis, SurveyLineGeometry

from .contracts import (
    ResidualDiagnostics,
    ResidualResult,
    ResidualTextureLibrary,
    ResidualTransferPolicy,
    TransferGeometry,
)
from .keys import BodyKey, BodyKeyEncoder, weighted_key_distance
from .library import gaussian_smooth_finite_run


@dataclass
class _TargetGrid:
    body: np.ndarray
    axis: SampleAxis
    mode: str
    lateral_shape: tuple[int, ...]
    flat_body: np.ndarray
    flat_support: np.ndarray
    flat_zone_ids: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    flat_pinchout: np.ndarray
    orientation: str | None

    @property
    def n_lateral(self) -> int:
        return int(self.flat_body.shape[0])

    @property
    def n_sample(self) -> int:
        return int(self.flat_body.shape[1])


@dataclass
class _QueryNode:
    index: int
    lateral_index: int
    center_m: float
    sample_indices: np.ndarray
    zone_id: str
    key: BodyKey


@dataclass(frozen=True)
class _DictionaryArrays:
    """Vectorized dictionary representation shared by every target section."""

    profile: np.ndarray
    first_derivative: np.ndarray
    second_derivative: np.ndarray
    contrast: np.ndarray
    thickness: np.ndarray
    position: np.ndarray
    mu: np.ndarray
    center_m: np.ndarray
    residual_axis: np.ndarray
    residual: np.ndarray
    residual_size: np.ndarray
    zone_indices: Mapping[str, np.ndarray]

    @classmethod
    def from_library(cls, library: ResidualTextureLibrary) -> "_DictionaryArrays":
        atoms = library.atoms
        max_samples = max(atom.physical_axis.size for atom in atoms)
        residual_axis = np.full((len(atoms), max_samples), np.inf, dtype=np.float64)
        residual = np.zeros((len(atoms), max_samples), dtype=np.float64)
        sizes = np.empty(len(atoms), dtype=np.int64)
        for index, atom in enumerate(atoms):
            if not np.all(atom.valid_support):
                raise ValueError("Residual dictionary atoms must come from one continuous finite run.")
            size = atom.physical_axis.size
            residual_axis[index, :size] = atom.physical_axis
            residual[index, :size] = atom.residual_value
            sizes[index] = size
        return cls(
            profile=np.stack([atom.body_key.profile for atom in atoms]),
            first_derivative=np.stack([atom.body_key.first_derivative for atom in atoms]),
            second_derivative=np.stack([atom.body_key.second_derivative for atom in atoms]),
            contrast=np.asarray([atom.body_key.contrast for atom in atoms], dtype=np.float64),
            thickness=np.asarray([atom.body_key.thickness for atom in atoms], dtype=np.float64),
            position=np.asarray(
                [atom.body_key.normalized_zone_position for atom in atoms],
                dtype=np.float64,
            ),
            mu=np.asarray([atom.body_key.mu for atom in atoms], dtype=np.float64),
            center_m=np.asarray([atom.body_key.center_m for atom in atoms], dtype=np.float64),
            residual_axis=residual_axis,
            residual=residual,
            residual_size=sizes,
            zone_indices={
                zone_id: library.atom_indices_for_zone(zone_id)
                for zone_id in library.zone_ids
            },
        )

    def initial_weights(
        self,
        nodes: list[_QueryNode],
        library: ResidualTextureLibrary,
        *,
        temperature_multiplier: float,
    ) -> np.ndarray:
        weights = np.zeros((len(nodes), len(library.atoms)), dtype=np.float64)
        scales = library.feature_scales

        def profile_distance_square(query: np.ndarray, atoms: np.ndarray) -> np.ndarray:
            size = float(query.shape[1])
            square = (
                np.sum(np.square(query), axis=1)[:, None]
                + np.sum(np.square(atoms), axis=1)[None, :]
                - 2.0 * (query @ atoms.T)
            ) / size
            return np.maximum(square, 0.0)

        for zone_id, atom_indices in self.zone_indices.items():
            node_indices = np.asarray(
                [index for index, node in enumerate(nodes) if node.zone_id == zone_id],
                dtype=np.int64,
            )
            if node_indices.size == 0:
                continue
            keys = [nodes[index].key for index in node_indices]
            profile = np.stack([key.profile for key in keys])
            first = np.stack([key.first_derivative for key in keys])
            second = np.stack([key.second_derivative for key in keys])
            distance_square = (
                profile_distance_square(profile, self.profile[atom_indices])
                / np.square(float(scales["profile"]))
            )
            distance_square += (
                profile_distance_square(first, self.first_derivative[atom_indices])
                / np.square(float(scales["first_derivative"]))
            )
            distance_square += (
                profile_distance_square(second, self.second_derivative[atom_indices])
                / np.square(float(scales["second_derivative"]))
            )
            for query_values, atom_values, name in (
                (
                    np.asarray([key.contrast for key in keys], dtype=np.float64),
                    self.contrast,
                    "contrast",
                ),
                (
                    np.asarray([key.thickness for key in keys], dtype=np.float64),
                    self.thickness,
                    "thickness",
                ),
                (
                    np.asarray(
                        [key.normalized_zone_position for key in keys],
                        dtype=np.float64,
                    ),
                    self.position,
                    "position",
                ),
            ):
                distance_square += np.square(
                    (query_values[:, None] - atom_values[atom_indices][None, :])
                    / float(scales[name])
                )
            distances = np.sqrt(distance_square / 6.0)
            temperature = (
                float(temperature_multiplier)
                * float(library.zone_stats[zone_id]["temperature_base"])
            )
            logits = -np.square(distances / temperature)
            logits -= np.max(logits, axis=1, keepdims=True)
            local = np.exp(logits)
            local /= np.sum(local, axis=1, keepdims=True)
            weights[np.ix_(node_indices, atom_indices)] = local
        return weights

    def transform(
        self,
        node: _QueryNode,
        atom_indices: np.ndarray,
        query_axis: np.ndarray,
        *,
        denominator_floor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.asarray(atom_indices, dtype=np.int64)
        axis = np.asarray(query_axis, dtype=np.float64)
        stretch = np.clip(
            node.key.thickness
            / np.maximum(self.thickness[indices], float(denominator_floor)),
            0.75,
            1.33,
        )
        amplitude = np.clip(
            np.sqrt(
                node.key.contrast
                / np.maximum(self.contrast[indices], float(denominator_floor))
            ),
            0.75,
            1.33,
        )
        shift = node.key.mu - self.mu[indices]
        source = (
            self.center_m[indices, None]
            + self.mu[indices, None]
            + (axis[None, :] - node.key.center_m - node.key.mu) / stretch[:, None]
        )
        source_axes = self.residual_axis[indices]
        upper = self.residual_size[indices, None] - 1
        valid = (
            (source >= source_axes[:, :1])
            & (source <= np.take_along_axis(source_axes, upper, axis=1))
        )
        right = np.sum(
            source_axes[:, :, None] < source[:, None, :],
            axis=1,
        )
        right = np.minimum(right, upper)
        left = np.maximum(right - 1, 0)
        rows = indices[:, None]
        left_axis = self.residual_axis[rows, left]
        right_axis = self.residual_axis[rows, right]
        interval = right_axis - left_axis
        fraction = np.zeros(source.shape, dtype=np.float64)
        interpolated = interval > 0.0
        fraction[interpolated] = (
            (source - left_axis)[interpolated] / interval[interpolated]
        )
        values = (
            (1.0 - fraction) * self.residual[rows, left]
            + fraction * self.residual[rows, right]
        )
        values *= amplitude[:, None]
        values[~valid] = 0.0
        return values, valid, shift, stretch, amplitude


def _field(source: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _body_array(ginn_body: Any) -> tuple[np.ndarray, SampleAxis | None]:
    embedded_axis = _field(ginn_body, "sample_axis", "axis", default=None)
    value = _field(ginn_body, "ginn_body", "body", "values", "log_ai", default=ginn_body)
    if hasattr(value, "values") and not isinstance(value, np.ndarray):
        value = value.values
    body = np.asarray(value, dtype=np.float64)
    if body.ndim not in (1, 2, 3):
        raise ValueError("ginn_body must be a 1-D trace, 2-D section, or 3-D volume.")
    if np.any(np.isinf(body)):
        raise ValueError("ginn_body must not contain infinite values.")
    if embedded_axis is not None and not isinstance(embedded_axis, SampleAxis):
        raise TypeError("ginn_body.sample_axis must be SampleAxis when provided.")
    return body, embedded_axis


def _as_line_geometry(value: Any) -> SurveyLineGeometry | None:
    if isinstance(value, SurveyLineGeometry):
        return value
    if value is None:
        return None
    inline_axis = _field(value, "inline_axis", default=None)
    xline_axis = _field(value, "xline_axis", default=None)
    if isinstance(inline_axis, LineAxis) and isinstance(xline_axis, LineAxis):
        return value
    return None


def _build_line_geometry_from_mapping(mapping: Any, body: np.ndarray) -> SurveyLineGeometry | None:
    candidate = _as_line_geometry(_field(mapping, "line_geometry", "survey_geometry", "geometry", default=None))
    if candidate is not None:
        return candidate
    if _as_line_geometry(mapping) is not None:
        return _as_line_geometry(mapping)
    required = (
        "inline_min",
        "inline_max",
        "inline_step",
        "xline_min",
        "xline_max",
        "xline_step",
    )
    if not all(_field(mapping, key, default=None) is not None for key in required):
        return None
    inline_count = int(round((float(_field(mapping, "inline_max")) - float(_field(mapping, "inline_min"))) / float(_field(mapping, "inline_step")))) + 1
    xline_count = int(round((float(_field(mapping, "xline_max")) - float(_field(mapping, "xline_min"))) / float(_field(mapping, "xline_step")))) + 1
    if body.ndim == 3 and (inline_count != body.shape[0] or xline_count != body.shape[1]):
        raise ValueError("Geometry inline/xline counts do not match ginn_body volume dimensions.")
    affine = ("x0", "y0", "dx_inline", "dy_inline", "dx_xline", "dy_xline")
    missing_affine = [key for key in affine if _field(mapping, key, default=None) is None]
    if missing_affine:
        raise ValueError(
            "Mapping geometry must provide the complete affine XY transform: "
            + ", ".join(affine)
            + ". Missing: "
            + ", ".join(missing_affine)
        )
    x0 = float(_field(mapping, "x0"))
    y0 = float(_field(mapping, "y0"))
    dx_inline = float(_field(mapping, "dx_inline"))
    dy_inline = float(_field(mapping, "dy_inline"))
    dx_xline = float(_field(mapping, "dx_xline"))
    dy_xline = float(_field(mapping, "dy_xline"))
    return SurveyLineGeometry(
        inline_axis=LineAxis(float(_field(mapping, "inline_min")), float(_field(mapping, "inline_step")), inline_count, name="inline"),
        xline_axis=LineAxis(float(_field(mapping, "xline_min")), float(_field(mapping, "xline_step")), xline_count, name="xline"),
        x0=x0,
        y0=y0,
        dx_inline=dx_inline,
        dy_inline=dy_inline,
        dx_xline=dx_xline,
        dy_xline=dy_xline,
    )


def _reshape_lateral_field(value: Any, *, body_shape: tuple[int, ...], name: str, default: Any = None) -> np.ndarray | None:
    if value is None:
        return None if default is None else np.asarray(default)
    array = np.asarray(value)
    lateral_shape = body_shape[:-1]
    if array.shape == body_shape:
        return array
    if array.shape == lateral_shape:
        return np.broadcast_to(array[..., None], body_shape)
    if body_shape[0] == 1 and array.shape == (body_shape[-1],):
        return np.broadcast_to(array[None, :], body_shape)
    raise ValueError(f"{name} shape {array.shape} does not match body shape {body_shape} or lateral shape {lateral_shape}.")


def _zone_token(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return None
    text = str(value)
    if text.strip() in {"", "nan", "None", "none", "-1"}:
        return None
    return text


def _zone_array_from_intervals(axis: np.ndarray, intervals: Mapping[str, tuple[float, float]]) -> np.ndarray:
    result = np.empty(axis.shape, dtype=object)
    result[:] = None
    for zone_id, (top, bottom) in intervals.items():
        selected = (axis >= float(top)) & (axis <= float(bottom))
        # Explicit intervals are expected to be non-overlapping.  The first
        # assignment wins if a caller supplied touching/overlapping intervals.
        selected &= np.equal(result, None)
        result[selected] = str(zone_id)
    return result


def _prepare_target(ginn_body: Any, geometry: Any, library: ResidualTextureLibrary, policy: ResidualTransferPolicy) -> _TargetGrid:
    body, embedded_axis = _body_array(ginn_body)
    axis_from_geometry = _field(geometry, "sample_axis", "axis", default=None)
    if axis_from_geometry is None:
        geometry_samples = _field(geometry, "samples", default=None)
        if geometry_samples is not None:
            geometry_samples = np.asarray(geometry_samples, dtype=np.float64)
            axis_from_geometry = SampleAxis(
                geometry_samples,
                library.sample_axis.domain,
                library.sample_axis.unit,
                library.sample_axis.depth_basis,
            )
    axis = axis_from_geometry or embedded_axis or library.sample_axis
    if not isinstance(axis, SampleAxis):
        raise TypeError("Transfer geometry/sample axis must be SampleAxis.")
    if not np.array_equal(axis.values, library.sample_axis.values):
        library_values = library.sample_axis.values
        selected = np.searchsorted(library_values, axis.values, side="left")
        selected = np.clip(selected, 0, library_values.size - 1)
        if not np.allclose(library_values[selected], axis.values, rtol=0.0, atol=1.0e-9):
            raise ValueError("ginn_body SampleAxis must equal or be an exact subset of the library SampleAxis.")
    if body.shape[-1] != axis.values.size:
        raise ValueError(f"ginn_body sample dimension {body.shape[-1]} does not match SampleAxis size {axis.values.size}.")
    if axis.domain == "depth" and axis.depth_basis != "tvdss":
        raise ValueError("Depth residual transfer requires a TVDSS SampleAxis.")

    geometry_source = geometry
    line_geometry = _as_line_geometry(geometry)
    if line_geometry is None:
        line_geometry = _build_line_geometry_from_mapping(geometry, body)
    if isinstance(geometry, TransferGeometry):
        line_geometry = geometry.line_geometry
    orientation = _field(geometry_source, "orientation", "section_orientation", default=policy.section_orientation)
    orientation = None if orientation is None else str(orientation).casefold()
    if orientation not in {None, "inline", "xline"}:
        raise ValueError("section orientation must be 'inline' or 'xline'.")

    if body.ndim == 1:
        mode = "trace"
        lateral_shape: tuple[int, ...] = ()
        default_x = np.asarray([0.0], dtype=np.float64)
        default_y = np.asarray([0.0], dtype=np.float64)
    elif body.ndim == 2:
        mode = "section"
        lateral_shape = (body.shape[0],)
        x_value = _field(geometry_source, "x_m", "x", default=None)
        y_value = _field(geometry_source, "y_m", "y", default=None)
        if x_value is not None and y_value is not None:
            default_x = np.asarray(x_value, dtype=np.float64).reshape(-1)
            default_y = np.asarray(y_value, dtype=np.float64).reshape(-1)
            if default_x.shape != (body.shape[0],) or default_y.shape != default_x.shape:
                raise ValueError("Section x_m/y_m arrays must match the trace dimension.")
        elif line_geometry is not None:
            fixed_inline = _field(geometry_source, "fixed_inline", "inline", default=None)
            fixed_xline = _field(geometry_source, "fixed_xline", "xline", default=None)
            if orientation in (None, "inline"):
                if fixed_inline is None:
                    fixed_inline = line_geometry.inline_axis.values()[line_geometry.inline_axis.count // 2]
                trace_xlines = line_geometry.xline_axis.values()
                if trace_xlines.size != body.shape[0]:
                    raise ValueError("Inline section body trace count differs from geometry xline axis.")
                default_x, default_y = line_geometry.trace_xy_grids(np.asarray([fixed_inline]), trace_xlines)
                default_x = default_x.reshape(-1)
                default_y = default_y.reshape(-1)
                orientation = "inline"
            else:
                if fixed_xline is None:
                    fixed_xline = line_geometry.xline_axis.values()[line_geometry.xline_axis.count // 2]
                trace_ilines = line_geometry.inline_axis.values()
                if trace_ilines.size != body.shape[0]:
                    raise ValueError("Xline section body trace count differs from geometry inline axis.")
                default_x, default_y = line_geometry.trace_xy_grids(trace_ilines, np.asarray([fixed_xline]))
                default_x = default_x.reshape(-1)
                default_y = default_y.reshape(-1)
                orientation = "xline"
        else:
            raise ValueError("A section transfer requires physical x_m/y_m coordinates or SurveyLineGeometry.")
    else:
        mode = "volume"
        lateral_shape = body.shape[:-1]
        x_value = _field(geometry_source, "x_m", "x", default=None)
        y_value = _field(geometry_source, "y_m", "y", default=None)
        if x_value is not None and y_value is not None:
            default_x = np.asarray(x_value, dtype=np.float64)
            default_y = np.asarray(y_value, dtype=np.float64)
            if default_x.shape != lateral_shape or default_y.shape != lateral_shape:
                raise ValueError("Volume x_m/y_m arrays must match ginn_body lateral dimensions.")
        elif line_geometry is not None:
            ilines = np.asarray(_field(geometry_source, "ilines", default=line_geometry.inline_axis.values()), dtype=np.float64)
            xlines = np.asarray(_field(geometry_source, "xlines", default=line_geometry.xline_axis.values()), dtype=np.float64)
            if ilines.size != body.shape[0] or xlines.size != body.shape[1]:
                raise ValueError("Volume line axes differ from ginn_body lateral dimensions.")
            default_x, default_y = line_geometry.trace_xy_grids(ilines, xlines)
        else:
            raise ValueError("A volume transfer requires physical x_m/y_m grids or SurveyLineGeometry.")

    x_lateral = np.asarray(default_x, dtype=np.float64)
    y_lateral = np.asarray(default_y, dtype=np.float64)
    coordinate_shape = (1,) if body.ndim == 1 else lateral_shape
    if x_lateral.shape != coordinate_shape or y_lateral.shape != coordinate_shape:
        raise ValueError("Geometry x_m/y_m lateral shapes do not match ginn_body.")
    if np.any(~np.isfinite(x_lateral)) or np.any(~np.isfinite(y_lateral)):
        raise ValueError("Geometry x_m/y_m must be finite.")

    support_value = _field(geometry_source, "support", "valid_mask", "target_mask", "purpose_mask", default=None)
    support_full = _reshape_lateral_field(support_value, body_shape=body.shape, name="support", default=np.ones(body.shape, dtype=bool))
    support_full = np.asarray(support_full, dtype=bool) & np.isfinite(body)
    zone_value = _field(geometry_source, "zone_ids", "zone_id", "zone", default=None)
    zone_full = _reshape_lateral_field(zone_value, body_shape=body.shape, name="zone_ids", default=None)
    if zone_full is None:
        intervals = library.scale_contract.zone_intervals
        if not intervals:
            envelopes = {
                zone_id: tuple(stats["interval_envelope"])
                for zone_id, stats in library.zone_stats.items()
                if "interval_envelope" in stats
            }
            if len(envelopes) > 1:
                raise ValueError(
                    "Spatially varying zones require explicit per-sample zone_ids in TransferGeometry."
                )
            intervals = envelopes
        vertical_zone = _zone_array_from_intervals(axis.values, intervals) if intervals else np.full(axis.values.shape, library.zone_ids[0], dtype=object)
        if body.ndim == 1:
            zone_full = vertical_zone.copy()
        else:
            zone_full = np.broadcast_to(vertical_zone[None, ...], body.shape).copy()
    else:
        zone_full = np.asarray(zone_full, dtype=object)
    flat_zone = zone_full.reshape(-1, body.shape[-1])
    flat_support = support_full.reshape(-1, body.shape[-1])

    valid_library_zones = set(library.zone_ids)
    for lateral_index in range(flat_support.shape[0]):
        for sample_index in range(flat_support.shape[1]):
            zone = _zone_token(flat_zone[lateral_index, sample_index])
            if zone not in valid_library_zones:
                flat_support[lateral_index, sample_index] = False
                flat_zone[lateral_index, sample_index] = None
            else:
                flat_zone[lateral_index, sample_index] = zone
    pinch_value = _field(geometry_source, "pinchout_mask", "edge_break_mask", "pinchout", default=None)
    pinch_full = _reshape_lateral_field(pinch_value, body_shape=body.shape, name="pinchout_mask", default=np.zeros(body.shape, dtype=bool))
    flat_pinchout = np.asarray(pinch_full, dtype=bool).reshape(-1, body.shape[-1])
    return _TargetGrid(
        body=body,
        axis=axis,
        mode=mode,
        lateral_shape=lateral_shape,
        flat_body=body.reshape(-1, body.shape[-1]),
        flat_support=flat_support,
        flat_zone_ids=flat_zone,
        x_m=x_lateral.reshape(-1),
        y_m=y_lateral.reshape(-1),
        flat_pinchout=flat_pinchout,
        orientation=orientation,
    )


def _segments_by_zone(support: np.ndarray, zones: np.ndarray) -> list[tuple[int, int, str]]:
    support = np.asarray(support, dtype=bool)
    zones = np.asarray(zones, dtype=object)
    if support.shape != zones.shape:
        raise ValueError("support and zones must have matching shapes.")
    result: list[tuple[int, int, str]] = []
    start: int | None = None
    current_zone: str | None = None
    for index, (valid, raw_zone) in enumerate(zip(support, zones)):
        zone = _zone_token(raw_zone) if valid else None
        if zone is None:
            if start is not None:
                result.append((start, index, current_zone or ""))
                start = None
                current_zone = None
            continue
        if start is None:
            start = index
            current_zone = zone
        elif zone != current_zone:
            result.append((start, index, current_zone or ""))
            start = index
            current_zone = zone
    if start is not None:
        result.append((start, support.size, current_zone or ""))
    return [(start, stop, zone) for start, stop, zone in result if stop - start >= 2]


def _window_centers(start_coordinate: float, end_coordinate: float, spacing: float) -> np.ndarray:
    values = [float(start_coordinate)]
    current = float(start_coordinate) + float(spacing)
    while current < float(end_coordinate) - 1.0e-10:
        values.append(float(current))
        current += float(spacing)
    values.append(float(end_coordinate))
    return np.asarray(values, dtype=np.float64)


def _zone_position(library: ResidualTextureLibrary, zone_id: str, center: float, fallback: tuple[float, float]) -> float:
    interval = library.zone_stats.get(zone_id, {}).get("interval")
    if interval is None:
        interval = fallback
    top, bottom = float(interval[0]), float(interval[1])
    return float(np.clip((center - top) / max(bottom - top, np.finfo(np.float64).tiny), 0.0, 1.0))


def _build_query_nodes(
    target: _TargetGrid,
    library: ResidualTextureLibrary,
    policy: ResidualTransferPolicy,
) -> tuple[list[_QueryNode], BodyKeyEncoder, float, float]:
    sample_step = target.axis.step
    contract = library.scale_contract
    half_width = float(policy.window_half_width_m or contract.resolved_window_half_width_m(sample_step))
    library_half_width = contract.resolved_window_half_width_m(sample_step)
    if not np.isclose(half_width, library_half_width, rtol=1.0e-9, atol=1.0e-9):
        raise ValueError("Transfer policy window_half_width_m must match the library physical key window.")
    spacing = float(policy.window_center_spacing_m or contract.resolved_window_center_spacing_m(sample_step))
    minimum_samples = int(policy.min_window_samples or contract.min_window_samples)
    encoder = BodyKeyEncoder(
        window_half_width_m=half_width,
        profile_samples=contract.normalized_profile_samples,
        denominator_floor=contract.denominator_floor,
    )
    nodes: list[_QueryNode] = []
    for lateral_index in range(target.n_lateral):
        segments = _segments_by_zone(target.flat_support[lateral_index], target.flat_zone_ids[lateral_index])
        for segment_start, segment_stop, zone_id in segments:
            segment_axis = target.axis.values[segment_start:segment_stop]
            if segment_axis.size < minimum_samples:
                continue
            centers = _window_centers(float(segment_axis[0]), float(segment_axis[-1]), spacing)
            for center in centers:
                window_mask = (
                    (target.axis.values >= center - half_width - 1.0e-10)
                    & (target.axis.values <= center + half_width + 1.0e-10)
                    & (np.arange(target.n_sample) >= segment_start)
                    & (np.arange(target.n_sample) < segment_stop)
                    & target.flat_support[lateral_index]
                )
                indices = np.flatnonzero(window_mask)
                if indices.size < minimum_samples:
                    continue
                key = encoder.encode(
                    target.flat_body[lateral_index, indices],
                    target.axis.values[indices],
                    center_m=float(center),
                    zone_id=zone_id,
                    normalized_zone_position=_zone_position(library, zone_id, float(center), (float(segment_axis[0]), float(segment_axis[-1]))),
                )
                nodes.append(
                    _QueryNode(
                        index=len(nodes),
                        lateral_index=lateral_index,
                        center_m=float(center),
                        sample_indices=indices.astype(np.int64),
                        zone_id=zone_id,
                        key=key,
                    )
                )
    if not nodes:
        raise ValueError("No valid GINN body query windows exist inside the transfer support.")
    if policy.max_nodes is not None and len(nodes) > policy.max_nodes:
        raise ValueError(f"Transfer query has {len(nodes)} windows, exceeding policy.max_nodes={policy.max_nodes}.")
    return nodes, encoder, half_width, spacing


def _initial_weights(
    nodes: list[_QueryNode],
    library: ResidualTextureLibrary,
    dictionary: _DictionaryArrays,
    *,
    temperature_multiplier: float,
) -> np.ndarray:
    return dictionary.initial_weights(
        nodes,
        library,
        temperature_multiplier=temperature_multiplier,
    )


def _node_edges(
    nodes: list[_QueryNode],
    target: _TargetGrid,
    library: ResidualTextureLibrary,
    policy: ResidualTransferPolicy,
    half_width: float,
    spacing: float,
) -> list[tuple[int, int, float, str]]:
    """Build lateral and vertical physical graph edges once."""

    node_by_lateral: dict[int, list[int]] = {}
    for node in nodes:
        node_by_lateral.setdefault(node.lateral_index, []).append(node.index)
    for values in node_by_lateral.values():
        values.sort(key=lambda index: nodes[index].center_m)

    pairs: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, float, str]] = []

    def add_edge(left_index: int, right_index: int, distance_m: float, kind: str, overlap_factor: float = 1.0) -> None:
        if left_index == right_index:
            return
        first, second = sorted((left_index, right_index))
        if (first, second) in pairs:
            return
        left = nodes[first]
        right = nodes[second]
        if left.zone_id != right.zone_id:
            return
        if (
            np.any(target.flat_pinchout[left.lateral_index, left.sample_indices])
            or np.any(target.flat_pinchout[right.lateral_index, right.sample_indices])
        ):
            return
        key_distance = weighted_key_distance(left.key, right.key, library.feature_scales)
        if not np.isfinite(key_distance):
            return
        if policy.edge_key_distance_max is not None and key_distance > policy.edge_key_distance_max:
            return
        distance_weight = np.exp(-float(distance_m) / policy.lateral_correlation_length_m)
        key_weight = np.exp(-0.5 * np.square(key_distance / policy.key_edge_scale))
        graph_weight = float(distance_weight * key_weight * overlap_factor)
        if not np.isfinite(graph_weight) or graph_weight <= 0.0:
            return
        pairs.add((first, second))
        candidates.append((first, second, graph_weight, kind))

    def adjacent_lateral_indices(lateral_index: int) -> list[int]:
        if len(target.lateral_shape) == 0:
            return []
        if len(target.lateral_shape) == 1:
            index = lateral_index
            result = []
            if index > 0:
                result.append(index - 1)
            if index + 1 < target.lateral_shape[0]:
                result.append(index + 1)
            return result
        inline_count, xline_count = target.lateral_shape
        inline_index, xline_index = np.unravel_index(lateral_index, target.lateral_shape)
        result: list[int] = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = inline_index + di, xline_index + dj
            if 0 <= ni < inline_count and 0 <= nj < xline_count:
                result.append(int(np.ravel_multi_index((ni, nj), target.lateral_shape)))
        return result

    tolerance = max(0.5 * spacing, 0.75 * (target.axis.step or 1.0))
    for lateral_index in range(target.n_lateral):
        left_nodes = node_by_lateral.get(lateral_index, [])
        for neighbor_lateral in adjacent_lateral_indices(lateral_index):
            if neighbor_lateral <= lateral_index:
                continue
            right_nodes = node_by_lateral.get(neighbor_lateral, [])
            if not right_nodes:
                continue
            for left_index in left_nodes:
                distances = np.asarray([abs(nodes[right_index].center_m - nodes[left_index].center_m) for right_index in right_nodes])
                if distances.size == 0:
                    continue
                right_index = right_nodes[int(np.argmin(distances))]
                if float(np.min(distances)) > tolerance:
                    continue
                distance_m = float(np.hypot(target.x_m[neighbor_lateral] - target.x_m[lateral_index], target.y_m[neighbor_lateral] - target.y_m[lateral_index]))
                if not np.isfinite(distance_m) or distance_m <= 0.0:
                    continue
                add_edge(left_index, right_index, distance_m, "lateral")

    for lateral_index, lateral_nodes in node_by_lateral.items():
        for left_index, right_index in zip(lateral_nodes[:-1], lateral_nodes[1:]):
            left = nodes[left_index]
            right = nodes[right_index]
            if left.zone_id != right.zone_id:
                continue
            left_start = left.center_m - half_width
            left_end = left.center_m + half_width
            right_start = right.center_m - half_width
            right_end = right.center_m + half_width
            overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
            overlap_factor = min(1.0, overlap / max(2.0 * half_width, np.finfo(np.float64).tiny))
            if overlap_factor <= 0.0:
                continue
            # Vertical consistency uses a normalized overlap factor.  Its
            # graph coordinate is the physical vertical center separation;
            # the term is not a sample-index smoothing operation.
            add_edge(left_index, right_index, 0.0, "vertical", overlap_factor=overlap_factor)
    return candidates


def solve_spatial_weight_field(
    initial_weights: np.ndarray,
    edges: list[tuple[int, int, float, str]],
    *,
    lambda_lateral: float,
    lambda_vertical: float,
    iterations: int,
    vertical_residual_terms: list["_VerticalTerm"] | None = None,
) -> np.ndarray:
    """Solve the non-negative simplex weight field by deterministic steps.

    Lateral edges smooth dictionary weights directly.  When
    ``vertical_residual_terms`` is supplied, vertical overlap consistency is
    evaluated on the transformed residual windows themselves, rather than
    forcing adjacent vertical weight vectors to be equal.
    """

    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    if weights.ndim != 2 or weights.shape[0] == 0:
        raise ValueError("initial_weights must be a non-empty 2-D array.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("initial_weights must be finite and non-negative.")
    row_sum = np.sum(weights, axis=1)
    if np.any(row_sum <= 0.0):
        raise ValueError("Each initial weight row must have positive sum.")
    weights /= row_sum[:, None]
    unary = weights.copy()
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(weights.shape[0])]
    for left, right, value, kind in edges:
        if vertical_residual_terms is not None and kind == "vertical":
            continue
        coefficient = float(value) * (float(lambda_lateral) if kind == "lateral" else float(lambda_vertical))
        if coefficient <= 0.0:
            continue
        neighbors[left].append((right, coefficient))
        neighbors[right].append((left, coefficient))
    for _ in range(int(iterations)):
        updated = np.empty_like(weights)
        for row in range(weights.shape[0]):
            numerator = unary[row].copy()
            denominator = 1.0
            for neighbor, coefficient in neighbors[row]:
                numerator += coefficient * weights[neighbor]
                denominator += coefficient
            updated[row] = numerator / denominator
        if vertical_residual_terms:
            gradients = np.zeros_like(updated)
            for term in vertical_residual_terms:
                indices = term.atom_indices
                difference = (
                    updated[term.left, indices] @ term.left_matrix
                    - updated[term.right, indices] @ term.right_matrix
                )
                coefficient = float(term.weight) * float(lambda_vertical)
                gradients[term.left, indices] += coefficient * (
                    term.left_matrix @ difference
                )
                gradients[term.right, indices] -= coefficient * (
                    term.right_matrix @ difference
                )
            # Residual windows carry the same log-AI units as the body and are
            # typically small compared with the unary key fit.  A fixed small
            # step keeps the projected simplex iteration deterministic and
            # prevents the overlap term from becoming a free amplitude fit.
            updated -= 0.1 * gradients
        updated = np.maximum(updated, 0.0)
        updated /= np.maximum(np.sum(updated, axis=1, keepdims=True), np.finfo(np.float64).tiny)
        weights = updated
    return weights


def _sparsify_weights(weights: np.ndarray, *, top_k: int) -> np.ndarray:
    """Keep the strongest ``top_k`` atoms in each already-coupled weight row."""

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("weights must be a non-empty 2-D array.")
    if isinstance(top_k, bool) or int(top_k) != top_k or int(top_k) < 1:
        raise ValueError("top_k must be a positive integer.")
    count = min(int(top_k), values.shape[1])
    if count == 1:
        selected = np.argmax(values, axis=1)
        result = np.zeros_like(values)
        result[np.arange(values.shape[0]), selected] = 1.0
        return result
    selected = np.argpartition(values, -count, axis=1)[:, -count:]
    result = np.zeros_like(values)
    rows = np.arange(values.shape[0])[:, None]
    result[rows, selected] = values[rows, selected]
    row_sum = np.sum(result, axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("Sparse dictionary rows must retain positive weight.")
    return result / row_sum


def _lateral_label_switch_fraction(
    labels: np.ndarray,
    edges: list[tuple[int, int, float, str]],
) -> float:
    lateral = [(left, right) for left, right, _weight, kind in edges if kind == "lateral"]
    if not lateral:
        return 0.0
    switches = sum(int(labels[left] != labels[right]) for left, right in lateral)
    return float(switches / len(lateral))


def solve_spatial_label_field(
    initial_weights: np.ndarray,
    edges: list[tuple[int, int, float, str]],
    nodes: list[_QueryNode],
    library: ResidualTextureLibrary,
    *,
    continuity_strength: float,
    iterations: int,
) -> np.ndarray:
    """Select one atom per node with a lateral Potts continuity term.

    The unary term remains the body-key dictionary likelihood.  Only lateral
    graph edges receive the same-label preference; vertically adjacent
    windows retain independent labels and are joined by partition-of-unity
    stitching.
    """

    weights = np.asarray(initial_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape != (len(nodes), len(library.atoms)):
        raise ValueError("initial_weights shape differs from nodes/library.")
    strength = float(continuity_strength)
    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError("continuity_strength must be finite and positive.")
    if isinstance(iterations, bool) or int(iterations) != iterations or int(iterations) < 1:
        raise ValueError("iterations must be a positive integer.")

    labels = np.argmax(weights, axis=1).astype(np.int64)
    unary = -np.log(np.maximum(weights, np.finfo(np.float64).tiny))
    neighbors: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for left, right, edge_weight, kind in edges:
        if kind != "lateral":
            continue
        coefficient = strength * float(edge_weight)
        if coefficient <= 0.0:
            continue
        neighbors[left].append((right, coefficient))
        neighbors[right].append((left, coefficient))

    zone_indices = {
        zone_id: library.atom_indices_for_zone(zone_id)
        for zone_id in library.zone_ids
    }
    for iteration in range(int(iterations)):
        changed = 0
        order = range(len(nodes)) if iteration % 2 == 0 else range(len(nodes) - 1, -1, -1)
        for row in order:
            candidates = zone_indices[nodes[row].zone_id]
            costs = unary[row, candidates].copy()
            for neighbor, coefficient in neighbors[row]:
                costs += coefficient
                same = np.flatnonzero(candidates == labels[neighbor])
                if same.size:
                    costs[int(same[0])] -= coefficient
            selected = int(candidates[int(np.argmin(costs))])
            if selected != int(labels[row]):
                labels[row] = selected
                changed += 1
        if changed == 0:
            break
    return labels


@dataclass(frozen=True)
class _VerticalTerm:
    left: int
    right: int
    weight: float
    atom_indices: np.ndarray
    left_matrix: np.ndarray
    right_matrix: np.ndarray


def _vertical_residual_terms(
    edges: list[tuple[int, int, float, str]],
    nodes: list[_QueryNode],
    target: _TargetGrid,
    library: ResidualTextureLibrary,
    dictionary: _DictionaryArrays,
) -> list[_VerticalTerm]:
    terms: list[_VerticalTerm] = []
    transformed: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def node_matrix(node: _QueryNode) -> tuple[np.ndarray, np.ndarray]:
        cached = transformed.get(node.index)
        if cached is not None:
            return cached
        atom_indices = dictionary.zone_indices[node.zone_id]
        values, valid, _shift, _stretch, _amplitude = dictionary.transform(
            node,
            atom_indices,
            target.axis.values[node.sample_indices],
            denominator_floor=library.scale_contract.denominator_floor,
        )
        transformed[node.index] = (values, valid)
        return values, valid

    for left_index, right_index, edge_weight, kind in edges:
        if kind != "vertical":
            continue
        left = nodes[left_index]
        right = nodes[right_index]
        common, left_positions, right_positions = np.intersect1d(
            left.sample_indices,
            right.sample_indices,
            return_indices=True,
        )
        if common.size < 2:
            continue
        atom_indices = dictionary.zone_indices[left.zone_id]
        left_values, left_valid = node_matrix(left)
        right_values, right_valid = node_matrix(right)
        common_valid = np.all(
            left_valid[:, left_positions] & right_valid[:, right_positions],
            axis=0,
        )
        if np.count_nonzero(common_valid) < 2:
            continue
        terms.append(
            _VerticalTerm(
                left=left_index,
                right=right_index,
                weight=float(edge_weight),
                atom_indices=atom_indices,
                left_matrix=left_values[:, left_positions][:, common_valid],
                right_matrix=right_values[:, right_positions][:, common_valid],
            )
        )
    return terms


def _node_values_and_transforms(
    node: _QueryNode,
    weights: np.ndarray,
    library: ResidualTextureLibrary,
    dictionary: _DictionaryArrays,
    axis: np.ndarray,
    *,
    collect_details: bool = True,
    preserve_mixture_energy: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    local_axis = axis[node.sample_indices]
    residual = np.zeros(local_axis.shape, dtype=np.float64)
    active_weight = np.zeros(local_axis.shape, dtype=np.float64)
    active_weight_square = np.zeros(local_axis.shape, dtype=np.float64)
    transform_values = {"shift": 0.0, "stretch": 0.0, "amplitude": 0.0}
    source_weight: dict[str, float] = {}
    component_rms_weighted = 0.0
    component_weight = 0.0
    atom_indices = np.flatnonzero(weights > 0.0)
    transformed, valid, shifts, stretches, amplitudes = dictionary.transform(
        node,
        atom_indices,
        local_axis,
        denominator_floor=library.scale_contract.denominator_floor,
    )
    local_weights = np.asarray(weights[atom_indices], dtype=np.float64)
    weighted = local_weights[:, None] * valid
    residual = np.sum(weighted * transformed, axis=0)
    active_weight = np.sum(weighted, axis=0)
    active_weight_square = np.sum(np.square(local_weights[:, None]) * valid, axis=0)
    if preserve_mixture_energy:
        valid_count = np.sum(valid, axis=1)
        component_rms = np.zeros(atom_indices.size, dtype=np.float64)
        occupied_components = valid_count > 0
        component_rms[occupied_components] = np.sqrt(
            np.sum(np.square(transformed), axis=1)[occupied_components]
            / valid_count[occupied_components]
        )
        component_rms_weighted = float(np.sum(local_weights * component_rms))
        component_weight = float(np.sum(local_weights[occupied_components]))
    if collect_details:
        transform_values = {
            "shift": float(local_weights @ shifts),
            "stretch": float(local_weights @ stretches),
            "amplitude": float(local_weights @ amplitudes),
        }
        for atom_index, weight in zip(atom_indices, local_weights):
            source_well = library.atoms[int(atom_index)].source_well
            source_weight[source_well] = source_weight.get(source_well, 0.0) + float(weight)
    occupied = active_weight > np.finfo(np.float64).tiny
    residual[occupied] /= active_weight[occupied]
    if preserve_mixture_energy and component_weight > 0.0 and np.any(occupied):
        target_rms = component_rms_weighted / component_weight
        mixed_rms = float(np.sqrt(np.mean(np.square(residual[occupied]))))
        if mixed_rms > library.scale_contract.denominator_floor:
            residual[occupied] *= target_rms / mixed_rms
    effective = np.zeros(local_axis.shape, dtype=np.float64)
    effective[occupied] = np.square(active_weight[occupied]) / np.maximum(
        active_weight_square[occupied], np.finfo(np.float64).tiny
    )
    return residual, effective, transform_values, source_weight


def _stitch_nodes(
    nodes: list[_QueryNode],
    node_weights: np.ndarray,
    target: _TargetGrid,
    library: ResidualTextureLibrary,
    dictionary: _DictionaryArrays,
    half_width: float,
    *,
    project: bool,
    collect_node_details: bool = True,
    preserve_mixture_energy: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[dict[str, float]], list[dict[str, float]]]:
    values = np.zeros((target.n_lateral, target.n_sample), dtype=np.float64)
    denominator = np.zeros_like(values)
    effective = np.zeros_like(values)
    local_values: list[np.ndarray] = []
    transform_summaries: list[dict[str, float]] = []
    source_summaries: list[dict[str, float]] = []
    for node, weights in zip(nodes, node_weights):
        local, local_effective, transform_summary, source_summary = _node_values_and_transforms(
            node,
            weights,
            library,
            dictionary,
            target.axis.values,
            collect_details=collect_node_details,
            preserve_mixture_energy=preserve_mixture_energy,
        )
        if collect_node_details:
            local_values.append(local)
            transform_summaries.append(transform_summary)
            source_summaries.append(source_summary)
        distances = np.abs(target.axis.values[node.sample_indices] - node.center_m)
        window_weight = 0.5 * (1.0 + np.cos(np.pi * distances / max(half_width, np.finfo(np.float64).tiny)))
        window_weight = np.where(distances <= half_width + 1.0e-10, window_weight, 0.0)
        lateral = node.lateral_index
        values[lateral, node.sample_indices] += window_weight * local
        denominator[lateral, node.sample_indices] += window_weight
        effective[lateral, node.sample_indices] += window_weight * local_effective
    result = np.zeros_like(values)
    occupied = denominator > 1.0e-12
    result[occupied] = values[occupied] / denominator[occupied]
    effective_result = np.zeros_like(effective)
    effective_result[occupied] = effective[occupied] / denominator[occupied]
    if project:
        result = _project_residual(result, target, library, target.axis.values)
    result[~target.flat_support] = 0.0
    effective_result[~target.flat_support] = 0.0
    return result, effective_result, local_values, transform_summaries, source_summaries


def _project_residual(
    residual: np.ndarray,
    target: _TargetGrid,
    library: ResidualTextureLibrary,
    axis: np.ndarray,
) -> np.ndarray:
    output = np.asarray(residual, dtype=np.float64).copy()
    fwhm = library.scale_contract.body_smoothing_fwhm_m
    for lateral_index in range(target.n_lateral):
        for start, stop, _zone in _segments_by_zone(target.flat_support[lateral_index], target.flat_zone_ids[lateral_index]):
            if stop - start < 2:
                output[lateral_index, start:stop] = 0.0
                continue
            segment_axis = axis[start:stop]
            segment_values = output[lateral_index, start:stop]
            smooth = gaussian_smooth_finite_run(segment_values, segment_axis, fwhm_m=fwhm)
            output[lateral_index, start:stop] = segment_values - smooth
        output[lateral_index, ~target.flat_support[lateral_index]] = 0.0
    return output


def _field_from_flat(values: np.ndarray, target: _TargetGrid) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(target.body.shape)


def _node_source_weight_matrix(weights: np.ndarray, library: ResidualTextureLibrary) -> tuple[tuple[str, ...], np.ndarray]:
    wells = tuple(sorted(library.source_wells))
    matrix = np.zeros((weights.shape[0], len(wells)), dtype=np.float64)
    well_to_column = {well: column for column, well in enumerate(wells)}
    for atom_index, atom in enumerate(library.atoms):
        matrix[:, well_to_column[atom.source_well]] += weights[:, atom_index]
    return wells, matrix


def _weight_uniform_variance(weights: np.ndarray, nodes: list[_QueryNode], library: ResidualTextureLibrary) -> float:
    if not len(nodes):
        return 0.0
    values: list[float] = []
    for row, node in enumerate(nodes):
        indices = library.atom_indices_for_zone(node.zone_id)
        uniform = np.zeros(weights.shape[1], dtype=np.float64)
        uniform[indices] = 1.0 / float(indices.size)
        values.append(float(np.mean(np.square(weights[row] - uniform))))
    return float(np.mean(values))


def _summary_temperature_sweep(
    nodes: list[_QueryNode],
    library: ResidualTextureLibrary,
    dictionary: _DictionaryArrays,
    policy: ResidualTransferPolicy,
    edges: list[tuple[int, int, float, str]],
    vertical_residual_terms: list[_VerticalTerm] | None = None,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for multiplier in policy.temperature_multipliers:
        initial = _initial_weights(
            nodes,
            library,
            dictionary,
            temperature_multiplier=multiplier,
        )
        spatial = (
            solve_spatial_weight_field(
                initial,
                edges,
                lambda_lateral=policy.lambda_lateral,
                lambda_vertical=policy.lambda_vertical,
                iterations=policy.spatial_iterations,
                vertical_residual_terms=vertical_residual_terms,
            )
            if policy.spatial_coupling
            else initial
        )
        effective = 1.0 / np.maximum(np.sum(np.square(spatial), axis=1), np.finfo(np.float64).tiny)
        source_wells, source_matrix = _node_source_weight_matrix(spatial, library)
        source_share = np.sum(source_matrix, axis=0)
        source_share /= max(float(np.sum(source_share)), np.finfo(np.float64).tiny)
        result[str(float(multiplier))] = {
            "temperature_multiplier": float(multiplier),
            "effective_dictionary_count_mean": float(np.mean(effective)),
            "effective_dictionary_count_min": float(np.min(effective)),
            "effective_dictionary_count_max": float(np.max(effective)),
            "weight_uniform_variance": _weight_uniform_variance(initial, nodes, library),
            "source_well_count": float(len(source_wells)),
            "dominant_source_well_share": float(np.max(source_share)) if source_share.size else 0.0,
        }
    return result


def _interpolated_node_distance(
    left: _QueryNode,
    left_values: np.ndarray,
    right: _QueryNode,
    right_values: np.ndarray,
    axis: np.ndarray,
) -> float:
    left_axis = axis[left.sample_indices]
    right_axis = axis[right.sample_indices]
    common, left_positions, right_positions = np.intersect1d(
        left.sample_indices,
        right.sample_indices,
        return_indices=True,
    )
    if common.size:
        return float(np.sqrt(np.mean(np.square(left_values[left_positions] - right_values[right_positions]))))
    overlap_start = max(float(left_axis[0]), float(right_axis[0]))
    overlap_end = min(float(left_axis[-1]), float(right_axis[-1]))
    if overlap_end < overlap_start:
        return 0.0
    overlap_axis = np.unique(
        np.concatenate(
            [
                left_axis[(left_axis >= overlap_start) & (left_axis <= overlap_end)],
                right_axis[(right_axis >= overlap_start) & (right_axis <= overlap_end)],
            ]
        )
    )
    if overlap_axis.size == 0:
        return 0.0
    left_interp = np.interp(overlap_axis, left_axis, left_values)
    right_interp = np.interp(overlap_axis, right_axis, right_values)
    return float(np.sqrt(np.mean(np.square(left_interp - right_interp))))


def _continuity_metrics(
    edges: list[tuple[int, int, float, str]],
    nodes: list[_QueryNode],
    weights: np.ndarray,
    local_values: list[np.ndarray],
    library: ResidualTextureLibrary,
    axis: np.ndarray,
) -> dict[str, Any]:
    key_distances: list[float] = []
    weight_distances: list[float] = []
    residual_distances: list[float] = []
    for left, right, _edge_weight, kind in edges:
        if kind != "lateral":
            continue
        key_distance = weighted_key_distance(nodes[left].key, nodes[right].key, library.feature_scales)
        if not np.isfinite(key_distance):
            continue
        key_distances.append(float(key_distance))
        weight_distances.append(float(np.sqrt(np.mean(np.square(weights[left] - weights[right])))))
        residual_distances.append(
            _interpolated_node_distance(nodes[left], local_values[left], nodes[right], local_values[right], axis)
        )

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": float(array.size),
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "max": float(np.max(array)),
        }

    return {
        "key_distance": key_distances,
        "weight_distance": weight_distances,
        "residual_distance": residual_distances,
        "key_distance_stats": stats(key_distances),
        "weight_distance_stats": stats(weight_distances),
        "residual_distance_stats": stats(residual_distances),
    }


def _field_derivative_rms(field: np.ndarray, target: _TargetGrid) -> float:
    values: list[np.ndarray] = []
    for lateral_index in range(target.n_lateral):
        for start, stop, _zone in _segments_by_zone(target.flat_support[lateral_index], target.flat_zone_ids[lateral_index]):
            if stop - start < 2:
                continue
            values.append(np.gradient(field[lateral_index, start:stop], target.axis.values[start:stop], edge_order=1))
    if not values:
        return 0.0
    concatenated = np.concatenate(values)
    return float(np.sqrt(np.mean(np.square(concatenated))))


def _field_autocorrelation_half_width(field: np.ndarray, target: _TargetGrid) -> float:
    widths: list[float] = []
    for lateral_index in range(target.n_lateral):
        for start, stop, _zone in _segments_by_zone(target.flat_support[lateral_index], target.flat_zone_ids[lateral_index]):
            if stop - start < 3:
                continue
            trace = np.asarray(field[lateral_index, start:stop], dtype=np.float64)
            centered = trace - float(np.mean(trace))
            autocorrelation = np.correlate(centered, centered, mode="full")[trace.size - 1 :]
            if autocorrelation[0] <= np.finfo(np.float64).tiny:
                continue
            below_half = np.flatnonzero(autocorrelation <= 0.5 * autocorrelation[0])
            lag_index = int(below_half[0]) if below_half.size else trace.size - 1
            widths.append(float(target.axis.values[start + lag_index] - target.axis.values[start]))
    return float(np.mean(widths)) if widths else 0.0


def _amplitude_diagnostics(
    predicted: np.ndarray,
    weighted_donor: np.ndarray,
    target: _TargetGrid,
    library: ResidualTextureLibrary,
) -> dict[str, Any]:
    support = target.flat_support
    predicted_values = predicted[support]
    donor_values = weighted_donor[support]
    predicted_rms = float(np.sqrt(np.mean(np.square(predicted_values)))) if predicted_values.size else 0.0
    donor_rms = float(np.sqrt(np.mean(np.square(donor_values)))) if donor_values.size else 0.0
    predicted_derivative = _field_derivative_rms(predicted, target)
    donor_derivative = _field_derivative_rms(weighted_donor, target)
    predicted_width = _field_autocorrelation_half_width(predicted, target)
    donor_width = _field_autocorrelation_half_width(weighted_donor, target)
    abs_quantiles = np.quantile(np.abs(predicted_values), [0.01, 0.05, 0.5, 0.95, 0.99]).astype(float) if predicted_values.size else np.zeros(5, dtype=np.float64)
    return {
        "residual_rms": predicted_rms,
        "weighted_donor_rms": donor_rms,
        "residual_rms_ratio_to_weighted_donor": predicted_rms / max(donor_rms, np.finfo(np.float64).tiny),
        "residual_absolute_quantiles": {
            name: float(value)
            for name, value in zip(("p01", "p05", "p50", "p95", "p99"), abs_quantiles)
        },
        "first_derivative_rms": predicted_derivative,
        "weighted_donor_first_derivative_rms": donor_derivative,
        "first_derivative_rms_ratio_to_weighted_donor": predicted_derivative / max(donor_derivative, np.finfo(np.float64).tiny),
        "autocorrelation_half_width_m": predicted_width,
        "weighted_donor_autocorrelation_half_width_m": donor_width,
        "autocorrelation_half_width_ratio_to_weighted_donor": predicted_width / max(donor_width, np.finfo(np.float64).tiny),
        "source_well_residual_rms": dict(library.source_well_residual_rms),
    }


def _transfer_residual(
    ginn_body: Any,
    geometry: Any,
    library: ResidualTextureLibrary,
    transfer_policy: ResidualTransferPolicy,
    dictionary: _DictionaryArrays,
) -> ResidualResult:
    """Run the production spatial top-2, unprojected transfer."""

    if (
        transfer_policy.section_orientation is not None
        and _field(geometry, "orientation", "section_orientation", default=None) is None
    ):
        if isinstance(geometry, Mapping):
            geometry = dict(geometry)
            geometry["orientation"] = transfer_policy.section_orientation
        else:
            geometry = TransferGeometry(
                sample_axis=_field(geometry, "sample_axis", default=None),
                line_geometry=_as_line_geometry(geometry),
                orientation=transfer_policy.section_orientation,
            )
    target = _prepare_target(ginn_body, geometry, library, transfer_policy)
    nodes, _encoder, half_width, spacing = _build_query_nodes(
        target,
        library,
        transfer_policy,
    )
    edges = _node_edges(nodes, target, library, transfer_policy, half_width, spacing)
    initial_weights = _initial_weights(
        nodes,
        library,
        dictionary,
        temperature_multiplier=transfer_policy.temperature_multiplier,
    )
    if transfer_policy.spatial_coupling:
        vertical_residual_terms = (
            _vertical_residual_terms(edges, nodes, target, library, dictionary)
            if transfer_policy.lambda_vertical > 0.0
            else None
        )
        spatial_weights = solve_spatial_weight_field(
            initial_weights,
            edges,
            lambda_lateral=transfer_policy.lambda_lateral,
            lambda_vertical=transfer_policy.lambda_vertical,
            iterations=transfer_policy.spatial_iterations,
            vertical_residual_terms=vertical_residual_terms,
        )
    else:
        spatial_weights = initial_weights
    production_weights = _sparsify_weights(spatial_weights, top_k=2)
    predicted_flat, effective_flat, _local, _transforms, _sources = _stitch_nodes(
        nodes,
        production_weights,
        target,
        library,
        dictionary,
        half_width,
        project=False,
        collect_node_details=False,
    )
    predicted_flat[~target.flat_support] = 0.0
    return ResidualResult(
        predicted_residual=_field_from_flat(predicted_flat, target),
        effective_dictionary_count=_field_from_flat(effective_flat, target),
        support=_field_from_flat(target.flat_support, target).astype(bool),
        node_count=len(nodes),
        graph_edge_count=len(edges),
    )


def _diagnose_residual(
    ginn_body: Any,
    geometry: Any,
    library: ResidualTextureLibrary,
    transfer_policy: ResidualTransferPolicy,
    dictionary: _DictionaryArrays,
) -> ResidualDiagnostics:
    """Run production transfer plus research-only comparison branches.

    The function accepts a 1-D trace, a section, or a full volume.  A full
    volume should be accompanied by ``SurveyLineGeometry`` (or a
    ``TransferGeometry``/mapping carrying it); all graph distances are then
    computed from XY coordinates.  The xline number step is used only to
    locate the corresponding trace coordinate through ``SurveyLineGeometry``.
    """

    # A policy-level section orientation is useful when the geometry object is
    # just a SurveyLineGeometry, while an explicit geometry orientation wins.
    if transfer_policy.section_orientation is not None and _field(geometry, "orientation", "section_orientation", default=None) is None:
        if isinstance(geometry, Mapping):
            geometry = dict(geometry)
            geometry["orientation"] = transfer_policy.section_orientation
        else:
            geometry = TransferGeometry(
                sample_axis=_field(geometry, "sample_axis", default=None),
                line_geometry=_as_line_geometry(geometry),
                orientation=transfer_policy.section_orientation,
            )
    target = _prepare_target(ginn_body, geometry, library, transfer_policy)
    nodes, _encoder, half_width, spacing = _build_query_nodes(target, library, transfer_policy)
    edges = _node_edges(nodes, target, library, transfer_policy, half_width, spacing)
    vertical_residual_terms = _vertical_residual_terms(
        edges,
        nodes,
        target,
        library,
        dictionary,
    )

    initial_weights = _initial_weights(
        nodes,
        library,
        dictionary,
        temperature_multiplier=transfer_policy.temperature_multiplier,
    )
    if transfer_policy.spatial_coupling:
        spatial_weights = solve_spatial_weight_field(
            initial_weights,
            edges,
            lambda_lateral=transfer_policy.lambda_lateral,
            lambda_vertical=transfer_policy.lambda_vertical,
            iterations=transfer_policy.spatial_iterations,
            vertical_residual_terms=vertical_residual_terms,
        )
    else:
        spatial_weights = initial_weights.copy()

    soft_flat, _soft_effective, soft_local_values, _soft_transforms, _soft_sources = _stitch_nodes(
        nodes,
        initial_weights,
        target,
        library,
        dictionary,
        half_width,
        project=True,
    )
    spatial_soft_projected_flat, _spatial_soft_effective, _spatial_soft_local, _spatial_soft_transforms, _spatial_soft_sources = _stitch_nodes(
        nodes,
        spatial_weights,
        target,
        library,
        dictionary,
        half_width,
        project=True,
    )
    weighted_donor_flat, _donor_effective, _donor_local_values, _donor_transforms, _donor_sources = _stitch_nodes(
        nodes,
        spatial_weights,
        target,
        library,
        dictionary,
        half_width,
        project=False,
    )
    hard_weights = np.zeros_like(initial_weights)
    hard_indices = np.argmax(initial_weights, axis=1)
    hard_weights[np.arange(hard_weights.shape[0]), hard_indices] = 1.0
    hard_flat, _hard_effective, _hard_local_values, _hard_transforms, _hard_sources = _stitch_nodes(
        nodes,
        hard_weights,
        target,
        library,
        dictionary,
        half_width,
        project=True,
    )
    hard_unprojected_flat, _hard_raw_effective, _hard_raw_local, _hard_raw_transforms, _hard_raw_sources = _stitch_nodes(
        nodes,
        hard_weights,
        target,
        library,
        dictionary,
        half_width,
        project=False,
    )
    spatial_sparse_weights = _sparsify_weights(spatial_weights, top_k=2)
    spatial_sparse_flat, effective_flat, final_local_values, final_transforms, final_sources = _stitch_nodes(
        nodes,
        spatial_sparse_weights,
        target,
        library,
        dictionary,
        half_width,
        project=False,
    )
    spatial_sparse_energy_flat, _energy_effective, _energy_local, _energy_transforms, _energy_sources = _stitch_nodes(
        nodes,
        spatial_sparse_weights,
        target,
        library,
        dictionary,
        half_width,
        project=False,
        preserve_mixture_energy=True,
    )
    spatial_dominant_weights = _sparsify_weights(spatial_weights, top_k=1)
    spatial_dominant_flat, _dominant_effective, _dominant_local, _dominant_transforms, _dominant_sources = _stitch_nodes(
        nodes,
        spatial_dominant_weights,
        target,
        library,
        dictionary,
        half_width,
        project=False,
    )
    graph_labels = solve_spatial_label_field(
        initial_weights,
        edges,
        nodes,
        library,
        continuity_strength=(
            transfer_policy.lambda_lateral
            * transfer_policy.label_continuity_strength
        ),
        iterations=transfer_policy.label_iterations,
    )
    graph_dominant_weights = np.zeros_like(initial_weights)
    graph_dominant_weights[np.arange(graph_dominant_weights.shape[0]), graph_labels] = 1.0
    graph_dominant_flat, _graph_effective, _graph_local, _graph_transforms, _graph_sources = _stitch_nodes(
        nodes,
        graph_dominant_weights,
        target,
        library,
        dictionary,
        half_width,
        project=False,
    )
    uniform_weights = np.zeros_like(initial_weights)
    for row, node in enumerate(nodes):
        indices = library.atom_indices_for_zone(node.zone_id)
        uniform_weights[row, indices] = 1.0 / float(indices.size)
    uniform_flat, _uniform_effective, _uniform_local_values, _uniform_transforms, _uniform_sources = _stitch_nodes(
        nodes,
        uniform_weights,
        target,
        library,
        dictionary,
        half_width,
        project=True,
    )

    predicted_flat = spatial_sparse_flat
    predicted_flat[~target.flat_support] = 0.0
    spatial_soft_projected_flat[~target.flat_support] = 0.0
    soft_flat[~target.flat_support] = 0.0
    hard_flat[~target.flat_support] = 0.0
    hard_unprojected_flat[~target.flat_support] = 0.0
    spatial_sparse_flat[~target.flat_support] = 0.0
    spatial_sparse_energy_flat[~target.flat_support] = 0.0
    spatial_dominant_flat[~target.flat_support] = 0.0
    graph_dominant_flat[~target.flat_support] = 0.0
    uniform_flat[~target.flat_support] = 0.0
    enhanced_flat = target.flat_body.copy()
    enhanced_flat[target.flat_support] += predicted_flat[target.flat_support]

    source_wells, source_matrix = _node_source_weight_matrix(spatial_sparse_weights, library)
    source_totals = np.sum(source_matrix, axis=0)
    source_totals /= max(float(np.sum(source_totals)), np.finfo(np.float64).tiny)
    zone_weight_totals: dict[str, float] = {}
    for node, row in zip(nodes, spatial_sparse_weights):
        zone_weight_totals[node.zone_id] = zone_weight_totals.get(node.zone_id, 0.0) + float(np.sum(row))
    zone_total = max(float(sum(zone_weight_totals.values())), np.finfo(np.float64).tiny)
    zone_weight_totals = {key: float(value / zone_total) for key, value in zone_weight_totals.items()}

    transform_arrays = {
        name: np.asarray([item[name] for item in final_transforms], dtype=np.float64)
        for name in ("shift", "stretch", "amplitude")
    }
    transform_summary: dict[str, Any] = {
        **transform_arrays,
        "node_center_m": np.asarray([node.center_m for node in nodes], dtype=np.float64),
        "node_zone_id": np.asarray([node.zone_id for node in nodes], dtype=object),
        "summary": {
            name: {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in transform_arrays.items()
        },
    }
    node_records = [
        {
            "node_index": int(node.index),
            "lateral_index": int(node.lateral_index),
            "center_m": float(node.center_m),
            "zone_id": node.zone_id,
            "shift": float(final_transforms[index]["shift"]),
            "stretch": float(final_transforms[index]["stretch"]),
            "amplitude": float(final_transforms[index]["amplitude"]),
            "source_well_weight": dict(final_sources[index]),
        }
        for index, node in enumerate(nodes)
    ]
    continuity = _continuity_metrics(
        edges,
        nodes,
        spatial_sparse_weights,
        final_local_values,
        library,
        target.axis.values,
    )
    amplitude_diagnostics = _amplitude_diagnostics(predicted_flat, weighted_donor_flat, target, library)
    dictionary_summary: dict[str, Any] = {
        "atom_count": int(len(library.atoms)),
        "source_wells": source_wells,
        "source_well_weight_share": {well: float(source_totals[index]) for index, well in enumerate(source_wells)},
        "zone_weight_share": zone_weight_totals,
        "temperature_multiplier": float(transfer_policy.temperature_multiplier),
        "temperature_by_zone": {
            zone_id: float(stats["temperature_base"] * transfer_policy.temperature_multiplier)
            for zone_id, stats in library.zone_stats.items()
        },
        "zone_coverage": {
            zone_id: {
                key: value
                for key, value in stats.items()
                if key in {"atom_count", "source_well_count", "source_wells", "temperature_base", "interval"}
            }
            for zone_id, stats in library.zone_stats.items()
        },
        "initial_weights": initial_weights,
        "spatial_weights": spatial_weights,
        "production_weights": spatial_sparse_weights,
        "node_lateral_index": np.asarray([node.lateral_index for node in nodes], dtype=np.int64),
        "node_center_m": np.asarray([node.center_m for node in nodes], dtype=np.float64),
        "node_zone_id": np.asarray([node.zone_id for node in nodes], dtype=object),
        "node_records": node_records,
        "n_graph_edges": int(len(edges)),
        "initial_hard_lateral_label_switch_fraction": _lateral_label_switch_fraction(
            np.argmax(initial_weights, axis=1),
            edges,
        ),
        "spatial_dominant_lateral_label_switch_fraction": _lateral_label_switch_fraction(
            np.argmax(spatial_weights, axis=1),
            edges,
        ),
        "graph_dominant_lateral_label_switch_fraction": _lateral_label_switch_fraction(
            graph_labels,
            edges,
        ),
        "initial_weight_uniform_variance": _weight_uniform_variance(initial_weights, nodes, library),
        "temperature_sweep": _summary_temperature_sweep(
            nodes,
            library,
            dictionary,
            transfer_policy,
            edges,
            vertical_residual_terms,
        ),
        "amplitude_diagnostics": amplitude_diagnostics,
    }

    return ResidualDiagnostics(
        ginn_body=target.body,
        predicted_residual=_field_from_flat(predicted_flat, target),
        enhanced_log_ai=_field_from_flat(enhanced_flat, target),
        dictionary_weight_summary=dictionary_summary,
        effective_dictionary_count=_field_from_flat(effective_flat, target),
        transform_summary=transform_summary,
        lateral_continuity_metrics=continuity,
        support=_field_from_flat(target.flat_support, target).astype(bool),
        soft_residual=_field_from_flat(soft_flat, target),
        hard_nearest_residual=_field_from_flat(hard_flat, target),
        uniform_residual=_field_from_flat(uniform_flat, target),
        residual_variants={
            "spatial_soft_projected": _field_from_flat(
                spatial_soft_projected_flat,
                target,
            ),
            "hard_nearest_unprojected": _field_from_flat(hard_unprojected_flat, target),
            "spatial_top2_energy_preserved_unprojected": _field_from_flat(
                spatial_sparse_energy_flat,
                target,
            ),
            "spatial_dominant_unprojected": _field_from_flat(spatial_dominant_flat, target),
            "graph_dominant_unprojected": _field_from_flat(graph_dominant_flat, target),
        },
    )


class ResidualTransfer:
    """Transfer one residual dictionary through the production or diagnostic path.

    ``predict`` is the formal spatial top-2, unprojected method. ``diagnose``
    evaluates that same method together with research-only alternatives.
    """

    def __init__(
        self,
        library: ResidualTextureLibrary,
        policy: ResidualTransferPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(library, ResidualTextureLibrary):
            raise TypeError("library must be a ResidualTextureLibrary.")
        self.library = library
        self.policy = ResidualTransferPolicy.from_any(policy)
        self._dictionary = _DictionaryArrays.from_library(library)

    def predict(self, ginn_body: Any, geometry: Any) -> ResidualResult:
        """Return only the formal spatial top-2, unprojected residual."""

        return _transfer_residual(
            ginn_body,
            geometry,
            self.library,
            self.policy,
            self._dictionary,
        )

    def diagnose(self, ginn_body: Any, geometry: Any) -> ResidualDiagnostics:
        """Return the formal residual plus all research comparison branches."""

        return _diagnose_residual(
            ginn_body,
            geometry,
            self.library,
            self.policy,
            self._dictionary,
        )


__all__ = [
    "ResidualTransfer",
    "solve_spatial_label_field",
    "solve_spatial_weight_field",
]
