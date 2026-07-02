"""Framework-body scenario modifier for any unified LFM baseline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cup.seismic.lfm.builders import _surface_for_output
from cup.seismic.lfm.types import LfmContext, LfmVariantResult


BODY_COLUMNS = ["body_id", "framework_class", "u_top", "u_bottom", "vertex_order", "inline", "xline"]
_SAFE_CLASS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.cross(b - a, c - a))


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> bool:
    return bool(
        np.isclose(_orientation(a, b, p), 0.0, rtol=0.0, atol=1e-8)
        and min(a[0], b[0]) - 1e-8 <= p[0] <= max(a[0], b[0]) + 1e-8
        and min(a[1], b[1]) - 1e-8 <= p[1] <= max(a[1], b[1]) + 1e-8
    )


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    o1, o2, o3, o4 = (_orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b))
    if ((o1 > 0.0 and o2 < 0.0) or (o1 < 0.0 and o2 > 0.0)) and (
        (o3 > 0.0 and o4 < 0.0) or (o3 < 0.0 and o4 > 0.0)
    ):
        return True
    return _on_segment(a, b, c) or _on_segment(a, b, d) or _on_segment(c, d, a) or _on_segment(c, d, b)


def _validate_polygon(vertices: np.ndarray, *, body_id: str) -> None:
    if vertices.shape[0] < 3 or np.unique(vertices, axis=0).shape[0] < 3:
        raise ValueError(f"Framework body {body_id!r} polygon needs at least three distinct vertices.")
    if abs(_polygon_area(vertices)) <= 1e-8:
        raise ValueError(f"Framework body {body_id!r} polygon is degenerate.")
    n = vertices.shape[0]
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        for j in range(i + 1, n):
            if j in {i, (i + 1) % n} or i in {j, (j + 1) % n}:
                continue
            if i == 0 and j == n - 1:
                continue
            c, d = vertices[j], vertices[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                raise ValueError(f"Framework body {body_id!r} polygon self-intersects.")


def _polygon_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _points_in_polygon(x: np.ndarray, y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    flat_x, flat_y = np.asarray(x).ravel(), np.asarray(y).ravel()
    inside = np.zeros(flat_x.shape, dtype=bool)
    vx, vy = vertices[:, 0], vertices[:, 1]
    j = vertices.shape[0] - 1
    for i in range(vertices.shape[0]):
        crosses = ((vy[i] > flat_y) != (vy[j] > flat_y)) & (
            flat_x < (vx[j] - vx[i]) * (flat_y - vy[i]) / (vy[j] - vy[i] + np.finfo(float).eps) + vx[i]
        )
        inside ^= crosses
        j = i
    return inside.reshape(np.asarray(x).shape)


def _distance_to_boundary(x: np.ndarray, y: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    points = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
    distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    for index in range(vertices.shape[0]):
        start, end = vertices[index], vertices[(index + 1) % vertices.shape[0]]
        segment = end - start
        denom = float(np.dot(segment, segment))
        projection = np.clip(((points - start) @ segment) / denom, 0.0, 1.0)
        nearest = start + projection[:, None] * segment
        distance = np.minimum(distance, np.linalg.norm(points - nearest, axis=1))
    return distance.reshape(np.asarray(x).shape)


def _polygon_edges(vertices: np.ndarray):
    for index in range(vertices.shape[0]):
        yield vertices[index], vertices[(index + 1) % vertices.shape[0]]


def _polygons_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    if np.any(_points_in_polygon(first[:, 0], first[:, 1], second)):
        return True
    if np.any(_points_in_polygon(second[:, 0], second[:, 1], first)):
        return True
    return any(_segments_intersect(a, b, c, d) for a, b in _polygon_edges(first) for c, d in _polygon_edges(second))


def _output_footprint(context: LfmContext) -> np.ndarray | None:
    output = context.output_geometry
    if output.is_section:
        return None
    return np.asarray(
        [
            context.line_geometry.line_to_coord(output.ilines[0], output.xlines[0]),
            context.line_geometry.line_to_coord(output.ilines[-1], output.xlines[0]),
            context.line_geometry.line_to_coord(output.ilines[-1], output.xlines[-1]),
            context.line_geometry.line_to_coord(output.ilines[0], output.xlines[-1]),
        ],
        dtype=np.float64,
    )


def _intersects_output(vertices: np.ndarray, context: LfmContext) -> bool:
    output = context.output_geometry
    footprint = _output_footprint(context)
    if footprint is not None:
        return _polygons_intersect(vertices, footprint)
    if np.any(_points_in_polygon(output.x_m, output.y_m, vertices)):
        return True
    points = np.column_stack([output.x_m, output.y_m])
    return any(
        _segments_intersect(points[index], points[index + 1], a, b)
        for index in range(points.shape[0] - 1)
        for a, b in _polygon_edges(vertices)
    )


def _vertical_probability(v: np.ndarray, *, top_fraction: float, bottom_fraction: float) -> np.ndarray:
    probability = np.zeros(np.asarray(v).shape, dtype=np.float64)
    top = (v >= 0.0) & (v < top_fraction)
    middle = (v >= top_fraction) & (v <= 1.0 - bottom_fraction)
    bottom = (v > 1.0 - bottom_fraction) & (v <= 1.0)
    probability[top] = 0.5 * (1.0 - np.cos(np.pi * v[top] / top_fraction))
    probability[middle] = 1.0
    probability[bottom] = 0.5 * (1.0 - np.cos(np.pi * (1.0 - v[bottom]) / bottom_fraction))
    return probability


def _body_rows(path: Path, classes: Mapping[str, Any], context: LfmContext) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if list(frame.columns) != BODY_COLUMNS:
        raise ValueError(f"Framework body CSV columns must be exactly {BODY_COLUMNS}.")
    if frame.empty:
        raise ValueError("Framework body CSV is empty.")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_body_id, group in frame.groupby("body_id", sort=False, dropna=False):
        body_id = str(raw_body_id).strip()
        if not body_id or body_id.casefold() in {"nan", "none", "null"} or body_id.casefold() in seen_ids:
            raise ValueError(f"Invalid or duplicate framework body_id: {raw_body_id!r}.")
        seen_ids.add(body_id.casefold())
        class_values = group["framework_class"].astype(str).str.strip().unique()
        top_values = pd.to_numeric(group["u_top"], errors="coerce").unique()
        bottom_values = pd.to_numeric(group["u_bottom"], errors="coerce").unique()
        if class_values.size != 1 or top_values.size != 1 or bottom_values.size != 1:
            raise ValueError(f"Framework body {body_id!r} class/u_top/u_bottom must be identical on every row.")
        class_name = str(class_values[0])
        if class_name not in classes:
            raise ValueError(f"Framework body {body_id!r} references inactive class {class_name!r}.")
        u_top, u_bottom = float(top_values[0]), float(bottom_values[0])
        if not (np.isfinite(u_top) and np.isfinite(u_bottom) and 0.0 <= u_top < u_bottom <= 1.0):
            raise ValueError(f"Framework body {body_id!r} requires 0 <= u_top < u_bottom <= 1.")
        order_values = pd.to_numeric(group["vertex_order"], errors="coerce").to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(order_values)) or np.any(order_values != np.arange(len(group), dtype=np.float64)):
            raise ValueError(f"Framework body {body_id!r} vertex_order must be unique and contiguous from 0.")
        line_vertices = group[["inline", "xline"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(line_vertices)):
            raise ValueError(f"Framework body {body_id!r} has non-finite line vertices.")
        xy_vertices = []
        for il, xl in line_vertices:
            try:
                il_index, xl_index = context.line_geometry.line_to_index(float(il), float(xl))
                if not (
                    np.isclose(il_index, round(il_index), rtol=0.0, atol=1e-8)
                    and np.isclose(xl_index, round(xl_index), rtol=0.0, atol=1e-8)
                ):
                    raise ValueError("vertex is not on an explicit survey line")
                xy_vertices.append(context.line_geometry.line_to_coord(float(il), float(xl)))
            except ValueError as exc:
                raise ValueError(
                    f"Framework body {body_id!r} has a vertex outside or between explicit survey lines."
                ) from exc
        vertices = np.asarray(xy_vertices, dtype=np.float64)
        _validate_polygon(vertices, body_id=body_id)
        rows.append(
            {
                "body_id": body_id,
                "framework_class": class_name,
                "u_top": u_top,
                "u_bottom": u_bottom,
                "vertices": vertices,
                "vertex_count": int(vertices.shape[0]),
                "polygon_area_m2": _polygon_area(vertices),
            }
        )
    return rows


def _class_config(raw: Mapping[str, Any], *, class_name: str, context: LfmContext) -> dict[str, Any]:
    required = {
        "top_horizon",
        "bottom_horizon",
        "linear_ai_multiplier",
        "edge_taper_m",
        "top_taper_fraction",
        "bottom_taper_fraction",
    }
    if set(raw) != required:
        raise ValueError(f"framework class {class_name!r} must contain exactly {sorted(required)}.")
    top, bottom = str(raw["top_horizon"]), str(raw["bottom_horizon"])
    names = context.target_zone.horizon_names
    if top not in names or bottom not in names or names.index(bottom) != names.index(top) + 1:
        raise ValueError(f"framework class {class_name!r} must map to one adjacent TargetZone interval.")
    multiplier = float(raw["linear_ai_multiplier"])
    edge = float(raw["edge_taper_m"])
    top_fraction = float(raw["top_taper_fraction"])
    bottom_fraction = float(raw["bottom_taper_fraction"])
    if not np.isfinite(multiplier) or multiplier <= 0.0 or np.isclose(multiplier, 1.0, rtol=0.0, atol=0.0):
        raise ValueError(f"framework class {class_name!r} linear_ai_multiplier must be positive and not 1.")
    if not np.isfinite(edge) or edge <= 0.0:
        raise ValueError(f"framework class {class_name!r} edge_taper_m must be positive.")
    if not (0.0 < top_fraction < 0.5 and 0.0 < bottom_fraction < 0.5):
        raise ValueError(f"framework class {class_name!r} taper fractions must lie in (0, 0.5).")
    return {
        "top_horizon": top,
        "bottom_horizon": bottom,
        "linear_ai_multiplier": multiplier,
        "edge_taper_m": edge,
        "top_taper_fraction": top_fraction,
        "bottom_taper_fraction": bottom_fraction,
    }


class FrameworkModifier:
    method = "framework"

    def apply(
        self,
        *,
        modifier_id: str,
        config: Mapping[str, Any],
        parent: LfmVariantResult,
        context: LfmContext,
    ) -> LfmVariantResult:
        if str(config.get("method") or "") != self.method:
            raise ValueError("FrameworkModifier received a non-framework config.")
        if set(config) != {"method", "bodies_file", "classes"}:
            raise ValueError("framework modifier must contain exactly method/bodies_file/classes.")
        classes_raw = config.get("classes")
        if not isinstance(classes_raw, Mapping) or not classes_raw:
            raise ValueError("framework.classes must be a non-empty mapping.")
        invalid_class_names = [str(name) for name in classes_raw if not _SAFE_CLASS.fullmatch(str(name))]
        if invalid_class_names:
            raise ValueError(f"framework class names must be filesystem-safe identifiers: {invalid_class_names}")
        classes = {
            str(name): _class_config(dict(value), class_name=str(name), context=context)
            for name, value in classes_raw.items()
            if isinstance(value, Mapping)
        }
        if len(classes) != len(classes_raw):
            raise ValueError("Every framework class config must be a mapping.")
        body_path = Path(str(config.get("bodies_file") or ""))
        if not body_path.is_file():
            raise FileNotFoundError(body_path)
        bodies = _body_rows(body_path, classes, context)
        body_classes = {str(body["framework_class"]) for body in bodies}
        missing_classes = sorted(set(classes) - body_classes)
        if missing_classes:
            raise ValueError(f"Configured framework classes have no body definitions: {missing_classes}")
        output = context.output_geometry
        out_x, out_y = output.x_m, output.y_m
        class_probability = {
            class_name: np.zeros(output.volume_shape, dtype=np.float64) for class_name in classes
        }
        class_map = {class_name: np.zeros(output.lateral_shape, dtype=np.float64) for class_name in classes}
        body_qc: list[dict[str, Any]] = []
        for body in bodies:
            class_name = str(body["framework_class"])
            class_cfg = classes[class_name]
            vertices = np.asarray(body["vertices"], dtype=np.float64)
            inside = _points_in_polygon(out_x, out_y, vertices)
            distance = _distance_to_boundary(out_x, out_y, vertices)
            p_map = np.zeros(output.lateral_shape, dtype=np.float64)
            taper = float(class_cfg["edge_taper_m"])
            ramp = inside & (distance > 0.0) & (distance < taper)
            p_map[ramp] = 0.5 * (1.0 - np.cos(np.pi * distance[ramp] / taper))
            p_map[inside & (distance >= taper)] = 1.0
            class_map[class_name] = np.maximum(class_map[class_name], p_map)
            top = _surface_for_output(context, str(class_cfg["top_horizon"]))
            bottom = _surface_for_output(context, str(class_cfg["bottom_horizon"]))
            samples = output.samples
            if output.is_section:
                u_zone = (samples[None, :] - top[:, None]) / (bottom[:, None] - top[:, None])
                p_vertical = _vertical_probability(
                    (u_zone - float(body["u_top"])) / (float(body["u_bottom"]) - float(body["u_top"])),
                    top_fraction=float(class_cfg["top_taper_fraction"]),
                    bottom_fraction=float(class_cfg["bottom_taper_fraction"]),
                )
                p_body = p_map[:, None] * p_vertical
            else:
                u_zone = (samples[None, None, :] - top[:, :, None]) / (bottom[:, :, None] - top[:, :, None])
                p_vertical = _vertical_probability(
                    (u_zone - float(body["u_top"])) / (float(body["u_bottom"]) - float(body["u_top"])),
                    top_fraction=float(class_cfg["top_taper_fraction"]),
                    bottom_fraction=float(class_cfg["bottom_taper_fraction"]),
                )
                p_body = p_map[:, :, None] * p_vertical
            class_probability[class_name] = np.maximum(class_probability[class_name], p_body)
            intersects = _intersects_output(vertices, context)
            positive = int(np.count_nonzero(p_body > 0.0))
            effective = int(np.count_nonzero((p_body > 0.0) & parent.valid_mask_model))
            if (output.mode == "volume" or intersects) and effective == 0:
                raise ValueError(
                    f"Framework body {body['body_id']!r} intersects the requested output "
                    "but has no positive sample inside valid_mask_model."
                )
            thickness = (float(body["u_bottom"]) - float(body["u_top"])) * (bottom - top)
            thickness_values = thickness[inside & np.isfinite(thickness) & (thickness > 0.0)]
            body_qc.append(
                {
                    "body_id": body["body_id"],
                    "framework_class": class_name,
                    "u_top": body["u_top"],
                    "u_bottom": body["u_bottom"],
                    "relative_thickness": float(body["u_bottom"]) - float(body["u_top"]),
                    "vertex_count": body["vertex_count"],
                    "polygon_area_m2": body["polygon_area_m2"],
                    "window_intersects": intersects,
                    "map_positive_trace_count": int(np.count_nonzero(p_map > 0.0)),
                    "map_max": float(np.max(p_map)),
                    "map_mean_inside": float(np.mean(p_map[inside])) if np.any(inside) else np.nan,
                    "physical_thickness_min": float(np.min(thickness_values)) if thickness_values.size else np.nan,
                    "physical_thickness_median": float(np.median(thickness_values)) if thickness_values.size else np.nan,
                    "physical_thickness_max": float(np.max(thickness_values)) if thickness_values.size else np.nan,
                    "probability_max": float(np.max(p_body)),
                    "positive_probability_voxel_count": positive,
                    "actual_modified_sample_count": effective,
                }
            )

        modified = np.asarray(parent.log_ai, dtype=np.float64).copy()
        modifier_fields: dict[str, np.ndarray] = dict(parent.modifier_fields)
        class_rows = []
        bin_area_m2 = abs(
            context.line_geometry.dx_inline * context.line_geometry.dy_xline
            - context.line_geometry.dy_inline * context.line_geometry.dx_xline
        )
        for class_name, probability in class_probability.items():
            class_cfg = classes[class_name]
            delta = probability * np.log(float(class_cfg["linear_ai_multiplier"]))
            modified[parent.valid_mask_model] += delta[parent.valid_mask_model]
            modifier_fields[f"class_probability__{class_name}"] = probability.astype(np.float32)
            modifier_fields[f"class_map_qc__{class_name}"] = class_map[class_name].astype(np.float32)
            class_rows.append(
                {
                    "framework_class": class_name,
                    **class_cfg,
                    "map_max": float(np.max(class_map[class_name])),
                    "map_mean": float(np.mean(class_map[class_name])),
                    "map_area_gt_0_5_trace_count": int(np.count_nonzero(class_map[class_name] > 0.5)),
                    "map_area_gt_0_9_trace_count": int(np.count_nonzero(class_map[class_name] > 0.9)),
                    "map_area_gt_0_5_m2": float(np.count_nonzero(class_map[class_name] > 0.5) * bin_area_m2),
                    "map_area_gt_0_9_m2": float(np.count_nonzero(class_map[class_name] > 0.9) * bin_area_m2),
                    "probability_max": float(np.max(probability)),
                    "positive_probability_voxel_count": int(np.count_nonzero(probability > 0.0)),
                    "modified_sample_count": int(np.count_nonzero((probability > 0.0) & parent.valid_mask_model)),
                }
            )
        if not any(
            np.any((probability > 0.0) & parent.valid_mask_model)
            for probability in class_probability.values()
        ):
            raise ValueError("Framework modifier does not affect any sample in the requested output geometry.")
        modified[~parent.valid_mask_model] = np.nan
        result = LfmVariantResult(
            log_ai=modified,
            valid_mask_model=parent.valid_mask_model.copy(),
            baseline_id=parent.baseline_id,
            baseline_method=parent.baseline_method,
            method_fields=dict(parent.method_fields),
            modifier_fields=modifier_fields,
            qc_tables={
                **parent.qc_tables,
                "framework_body_qc": pd.DataFrame.from_records(body_qc),
                "framework_class_qc": pd.DataFrame.from_records(class_rows),
            },
            metadata={
                **parent.metadata,
                "modifier_chain": [*list(parent.metadata.get("modifier_chain", [])), modifier_id],
                "framework_scenario_model": True,
                "strictly_well_honoring": False,
                "framework_bodies": [
                    {
                        "body_id": body["body_id"],
                        "framework_class": body["framework_class"],
                        "u_top": body["u_top"],
                        "u_bottom": body["u_bottom"],
                        "vertices_xy_m": np.asarray(body["vertices"]).tolist(),
                    }
                    for body in bodies
                ],
            },
        )
        result.validate(context)
        return result


MODIFIERS = {"framework": FrameworkModifier()}


__all__ = ["BODY_COLUMNS", "FrameworkModifier", "MODIFIERS"]
