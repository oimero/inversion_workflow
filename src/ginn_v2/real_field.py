"""Real-field section adapter for the public Structured GINN interface.

The adapter owns data assembly only.  It samples one published LFM section's
exact physical path from the source seismic and emits ordinary
``ObservationTile`` objects.  Structured inference remains entirely inside
``ConditionalGenerator.predict``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cup.config.workflow import WorkflowConfig
from cup.lfm.artifacts import ResolvedLfmVariant, resolve_lfm_variant
from cup.petrel.load import import_interpretation_petrel
from cup.physics.numpy_backend import velocity_from_ai
from cup.seismic.horizon import normalize_interpretation_unit_for_geometry
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.target_zone import TargetZone
from cup.synthetic.core.records import SampleAxis
from cup.utils.io import load_yaml_config, resolve_relative_path

from ginn_v2.contracts import ObservationTile, zone_linear_lateral_support
from ginn_v2.representation import build_lfm_anchor, lfm_residual_from_anchor


@dataclass(frozen=True)
class RealSectionObservations:
    """One published real section and its zone-specific observation tiles."""

    tiles: tuple[ObservationTile, ...]
    zone_ids: tuple[str, ...]
    raw_seismic: np.ndarray
    transformed_seismic: np.ndarray
    full_lfm: np.ndarray
    observed_valid: np.ndarray
    ilines: np.ndarray
    xlines: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    lateral_m: np.ndarray
    seismic_scale_factor: float
    lfm_variant: ResolvedLfmVariant
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.tiles or len(self.tiles) != len(self.zone_ids):
            raise ValueError("real section must contain one tile per zone identity.")

    def select_traces(self, start: int, count: int) -> "RealSectionObservations":
        """Select one contiguous physical subsection for bounded experiments."""

        first = int(start)
        size = int(count)
        width = int(self.full_lfm.shape[0])
        if first < 0 or size <= 0 or first + size > width:
            raise ValueError(
                f"trace selection [{first}, {first + size}) exceeds width {width}."
            )
        selection = slice(first, first + size)
        tiles = tuple(
            ObservationTile(
                model_axis=tile.model_axis,
                highres_axis=tile.highres_axis,
                seismic=tile.seismic[selection],
                lfm=tile.lfm[selection],
                observed_valid=tile.observed_valid[selection],
                lateral_m=tile.lateral_m[selection],
                lateral_valid=tile.lateral_valid[selection],
                zone_top=tile.zone_top[selection],
                zone_bottom=tile.zone_bottom[selection],
                vp_model_mps=(
                    None
                    if tile.vp_model_mps is None
                    else tile.vp_model_mps[selection]
                ),
                x_m=None if tile.x_m is None else tile.x_m[selection],
                y_m=None if tile.y_m is None else tile.y_m[selection],
                identity=f"{tile.identity}:traces[{first}:{first + size}]",
            )
            for tile in self.tiles
        )
        return RealSectionObservations(
            tiles=tiles,
            zone_ids=self.zone_ids,
            raw_seismic=self.raw_seismic[selection],
            transformed_seismic=self.transformed_seismic[selection],
            full_lfm=self.full_lfm[selection],
            observed_valid=self.observed_valid[selection],
            ilines=self.ilines[selection],
            xlines=self.xlines[selection],
            x_m=self.x_m[selection],
            y_m=self.y_m[selection],
            lateral_m=self.lateral_m[selection],
            seismic_scale_factor=self.seismic_scale_factor,
            lfm_variant=self.lfm_variant,
            metadata={
                **dict(self.metadata),
                "trace_selection": {
                    "start": first,
                    "stop": first + size,
                    "count": size,
                    "source_trace_count": width,
                },
            },
        )


def _read_lfm_section(
    resolved: ResolvedLfmVariant,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(resolved.lfm_path, allow_pickle=False) as data:
        log_ai = np.asarray(data["log_ai"], dtype=np.float64)
        valid = np.asarray(data["valid_mask_model"], dtype=bool)
        ilines = np.asarray(data["ilines"], dtype=np.float64)
        xlines = np.asarray(data["xlines"], dtype=np.float64)
        samples = np.asarray(data["samples"], dtype=np.float64)
    if log_ai.ndim != 2:
        raise ValueError(
            "initial real-field inference requires a published section LFM; "
            f"got shape {log_ai.shape}."
        )
    if (
        valid.shape != log_ai.shape
        or ilines.shape != (log_ai.shape[0],)
        or xlines.shape != ilines.shape
        or samples.shape != (log_ai.shape[1],)
    ):
        raise ValueError("published section LFM axes do not match its values.")
    return log_ai, valid, ilines, xlines, samples


def _section_xy_and_distance(
    survey: Any,
    ilines: np.ndarray,
    xlines: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(
        [
            survey.line_geometry.line_to_coord(float(iline), float(xline))
            for iline, xline in zip(ilines, xlines, strict=True)
        ],
        dtype=np.float64,
    )
    lateral = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if lateral.size > 1 and np.any(np.diff(lateral) <= 0.0):
        raise ValueError("published section path must move forward in physical XY space.")
    return xy[:, 0], xy[:, 1], lateral


def _bilinear_plan(
    survey: Any,
    iline: float,
    xline: float,
) -> tuple[tuple[int, int, float], ...]:
    i_float, j_float = survey.line_geometry.line_to_index(iline, xline)
    i0, i1 = int(np.floor(i_float)), int(np.ceil(i_float))
    j0, j1 = int(np.floor(j_float)), int(np.ceil(j_float))
    wi, wj = float(i_float - i0), float(j_float - j0)
    candidates = (
        (i0, j0, (1.0 - wi) * (1.0 - wj)),
        (i0, j1, (1.0 - wi) * wj),
        (i1, j0, wi * (1.0 - wj)),
        (i1, j1, wi * wj),
    )
    combined: dict[tuple[int, int], float] = {}
    for i, j, weight in candidates:
        if weight <= 0.0:
            continue
        survey.trace_flat_index(i, j)
        combined[(i, j)] = combined.get((i, j), 0.0) + weight
    if not combined or not np.isclose(sum(combined.values()), 1.0, atol=1.0e-10):
        raise ValueError(f"invalid seismic sampling weights at {(iline, xline)}.")
    return tuple((i, j, weight) for (i, j), weight in sorted(combined.items()))


def _sample_section_seismic(
    survey: Any,
    ilines: np.ndarray,
    xlines: np.ndarray,
    samples: np.ndarray,
    *,
    domain: str,
) -> np.ndarray:
    plans = tuple(
        _bilinear_plan(survey, float(iline), float(xline))
        for iline, xline in zip(ilines, xlines, strict=True)
    )
    indices = sorted({(i, j) for plan in plans for i, j, _ in plan})
    traces = survey.read_traces_at_indices(
        indices,
        sample_start=float(samples[0]),
        sample_end=float(samples[-1]),
        domain=domain,
    )
    if not traces:
        raise ValueError("section seismic sampler resolved no source traces.")
    for trace in traces.values():
        if not np.allclose(trace.basis, samples, rtol=0.0, atol=1.0e-9):
            raise ValueError("section LFM and source seismic sample axes differ.")
    result = np.empty((len(plans), samples.size), dtype=np.float64)
    for trace_index, plan in enumerate(plans):
        result[trace_index] = sum(
            weight * np.asarray(traces[(i, j)].values, dtype=np.float64)
            for i, j, weight in plan
        )
    return result


def _target_zone(
    raw_config: Mapping[str, Any],
    workflow: WorkflowConfig,
    survey: Any,
    *,
    repo_root: Path,
) -> TargetZone:
    target = raw_config.get("target_interval")
    if not isinstance(target, Mapping) or not isinstance(target.get("horizons"), list):
        raise ValueError("workflow target_interval.horizons must be an ordered list.")
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    geometry = survey.describe_geometry(workflow.seismic.domain)
    frames: dict[str, Any] = {}
    names: list[str] = []
    for item in target["horizons"]:
        if not isinstance(item, Mapping):
            raise ValueError("each target horizon must be a mapping.")
        name = str(item.get("name") or "").strip()
        path = resolve_relative_path(str(item.get("file") or ""), root=data_root)
        if not name or name in frames or not path.is_file():
            raise ValueError(f"invalid target horizon {name!r}: {path}")
        frame = normalize_interpretation_unit_for_geometry(
            import_interpretation_petrel(path),
            geometry,
        )
        frame = frame.copy()
        frame["interpretation"] = np.abs(
            frame["interpretation"].to_numpy(dtype=np.float64)
        )
        frames[name] = frame
        names.append(name)
    if len(names) < 2:
        raise ValueError("real-field inference requires at least two horizons.")
    return TargetZone(
        frames,
        geometry,
        names,
        min_thickness=float(survey.sample_axis(workflow.seismic.domain).step),
    )


def _highres_axis(model_axis: SampleAxis, interval: float) -> SampleAxis:
    highres_interval = float(interval)
    ratio = model_axis.sample_interval / highres_interval
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1.0e-10):
        raise ValueError("high-resolution interval must integer-divide the model interval.")
    coordinates = np.linspace(
        float(model_axis.coordinates[0]),
        float(model_axis.coordinates[-1]),
        (model_axis.coordinates.size - 1) * factor + 1,
        dtype=np.float64,
    )
    return SampleAxis(
        sample_domain=model_axis.sample_domain,
        unit=model_axis.unit,
        coordinates=coordinates,
        sample_interval=highres_interval,
        positive_direction=model_axis.positive_direction,
        depth_basis=model_axis.depth_basis,
    )


def load_real_section_observations(
    *,
    workflow_config: str | Path,
    lfm_run_dir: str | Path,
    variant_id: str,
    well_control_run_dir: str | Path,
    forward_model_inputs: str | Path,
    target_seismic_scale: float,
    target_lfm_residual_scale: float,
    highres_interval: float,
    repo_root: str | Path,
) -> RealSectionObservations:
    """Assemble one published real LFM section into zone observation tiles."""

    root = Path(repo_root)
    raw = load_yaml_config(resolve_relative_path(workflow_config, root=root))
    workflow = WorkflowConfig.from_mapping(raw)
    if workflow.seismic.domain != "depth" or workflow.seismic.depth_basis != "tvdss":
        raise ValueError("current real-field path requires depth/TVDSS seismic.")
    resolved = resolve_lfm_variant(
        {
            "lfm_run_dir": str(lfm_run_dir),
            "variant_id": str(variant_id),
            "well_control_run_dir": str(well_control_run_dir),
        },
        repo_root=root,
    )
    if (
        resolved.variant_metadata.get("sample_domain") != workflow.seismic.domain
        or resolved.variant_metadata.get("depth_basis") != workflow.seismic.depth_basis
    ):
        raise ValueError("LFM variant and current seismic domain contracts differ.")
    full_lfm, lfm_valid, ilines, xlines, samples = _read_lfm_section(resolved)

    data_root = resolve_relative_path(workflow.data_root, root=root)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    survey = open_survey(
        seismic_path,
        workflow.seismic.type,
        segy_options=(
            segy_options_from_config(workflow.seismic.as_dict())
            if workflow.seismic.type == "segy"
            else None
        ),
    )
    survey_axis = survey.sample_axis(workflow.seismic.domain)
    model_interval = float(survey_axis.step)
    model_axis = SampleAxis(
        sample_domain=workflow.seismic.domain,
        unit=survey_axis.unit,
        coordinates=samples,
        sample_interval=model_interval,
        positive_direction="down",
        depth_basis=workflow.seismic.depth_basis,
    )
    highres_axis = _highres_axis(model_axis, highres_interval)
    raw_seismic = _sample_section_seismic(
        survey,
        ilines,
        xlines,
        samples,
        domain=workflow.seismic.domain,
    )
    observed_valid = lfm_valid & np.isfinite(raw_seismic) & np.isfinite(full_lfm)
    valid_amplitude = np.abs(raw_seismic[observed_valid])
    if valid_amplitude.size == 0:
        raise ValueError("real section has no jointly valid seismic/LFM samples.")
    source_scale = float(np.percentile(valid_amplitude, 95.0))
    requested_scale = float(target_seismic_scale)
    if not np.isfinite(source_scale) or source_scale <= 0.0:
        raise ValueError("real seismic p95 absolute amplitude is not positive.")
    if not np.isfinite(requested_scale) or requested_scale <= 0.0:
        raise ValueError("checkpoint target seismic scale must be positive.")
    scale_factor = requested_scale / source_scale
    transformed_seismic = raw_seismic * scale_factor

    x_m, y_m, lateral_m = _section_xy_and_distance(survey, ilines, xlines)
    target_zone = _target_zone(raw, workflow, survey, repo_root=root)
    horizon_values = {
        name: np.asarray(
            [
                target_zone.horizon_surfaces[name].sample_at_line(
                    float(iline),
                    float(xline),
                    nearest_fallback_max_line_distance=0.0,
                ).value
                for iline, xline in zip(ilines, xlines, strict=True)
            ],
            dtype=np.float64,
        )
        for name in target_zone.horizon_names
    }

    with resolve_relative_path(forward_model_inputs, root=root).open(
        "r", encoding="utf-8"
    ) as handle:
        forward = json.load(handle)
    relation = forward.get("ai_velocity_relation")
    if not isinstance(relation, Mapping):
        raise ValueError("forward_model_inputs lacks ai_velocity_relation.")
    velocity = np.full(full_lfm.shape, np.nan, dtype=np.float64)
    finite_lfm = np.isfinite(full_lfm)
    velocity[finite_lfm] = velocity_from_ai(
        np.exp(full_lfm[finite_lfm]),
        a=float(relation["a"]),
        b=float(relation["b"]),
    )

    tiles: list[ObservationTile] = []
    zone_ids: list[str] = []
    zone_input_scales: dict[str, dict[str, float]] = {}
    for top_name, bottom_name in target_zone.iter_zones():
        top = horizon_values[top_name]
        bottom = horizon_values[bottom_name]
        lateral_valid = zone_linear_lateral_support(
            model_axis,
            observed_valid,
            top,
            bottom,
        )
        zone_id = f"{top_name}__to__{bottom_name}"
        tile = ObservationTile(
            model_axis=model_axis,
            highres_axis=highres_axis,
            seismic=np.where(observed_valid, transformed_seismic, 0.0),
            lfm=np.where(observed_valid, full_lfm, 0.0),
            observed_valid=observed_valid,
            lateral_m=lateral_m,
            lateral_valid=lateral_valid,
            zone_top=top,
            zone_bottom=bottom,
            vp_model_mps=velocity,
            x_m=x_m,
            y_m=y_m,
            identity=f"real:{resolved.variant_id}:{zone_id}",
        )
        anchor = build_lfm_anchor(tile)
        residual = lfm_residual_from_anchor(tile, anchor)
        supported_residual = np.abs(residual[anchor.model_support])
        if supported_residual.size == 0:
            raise ValueError(f"real zone {zone_id!r} has no LFM residual support.")
        lfm_p95 = float(np.percentile(supported_residual, 95.0))
        expected_lfm_scale = float(target_lfm_residual_scale)
        if not np.isfinite(expected_lfm_scale) or expected_lfm_scale <= 0.0:
            raise ValueError("checkpoint LFM residual scale must be positive.")
        zone_input_scales[zone_id] = {
            "lfm_residual_abs_p95": lfm_p95,
            "checkpoint_lfm_residual_abs_p95": expected_lfm_scale,
            "lfm_residual_scale_ratio": lfm_p95 / expected_lfm_scale,
        }
        tiles.append(tile)
        zone_ids.append(zone_id)
    return RealSectionObservations(
        tiles=tuple(tiles),
        zone_ids=tuple(zone_ids),
        raw_seismic=raw_seismic,
        transformed_seismic=transformed_seismic,
        full_lfm=full_lfm,
        observed_valid=observed_valid,
        ilines=ilines,
        xlines=xlines,
        x_m=x_m,
        y_m=y_m,
        lateral_m=lateral_m,
        seismic_scale_factor=scale_factor,
        lfm_variant=resolved,
        metadata={
            "seismic_path": str(seismic_path),
            "seismic_transform": "absolute_p95_to_checkpoint_absolute_p95",
            "seismic_abs_p95_raw": source_scale,
            "checkpoint_seismic_abs_p95": requested_scale,
            "seismic_scale_factor": scale_factor,
            "zone_input_scales": zone_input_scales,
            "forward_model_inputs": str(
                resolve_relative_path(forward_model_inputs, root=root)
            ),
        },
    )


__all__ = ["RealSectionObservations", "load_real_section_observations"]
