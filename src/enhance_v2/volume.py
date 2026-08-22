"""Section-streamed full-volume conditional residual transfer."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import logging
import time
from typing import Mapping

import numpy as np

from cup.seismic.geometry import SampleAxis, SurveyLineGeometry

from .contracts import ResidualTextureLibrary, ResidualTransferPolicy, TransferGeometry
from .transfer import transfer_residual_field


@dataclass(frozen=True)
class ZoneSurface:
    """One named zone represented by top and bottom grids on the survey."""

    zone_id: str
    top: np.ndarray
    bottom: np.ndarray

    def __post_init__(self) -> None:
        top = np.asarray(self.top, dtype=np.float64)
        bottom = np.asarray(self.bottom, dtype=np.float64)
        if top.ndim != 2 or bottom.shape != top.shape:
            raise ValueError("ZoneSurface top and bottom must be matching 2-D grids.")
        object.__setattr__(self, "zone_id", str(self.zone_id))
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "bottom", bottom)


@dataclass(frozen=True)
class VolumeTransferConfig:
    """Execution settings for section-streamed transfer."""

    orientations: tuple[str, ...] = ("inline", "xline")
    log_every_sections: int = 10
    max_workers: int = 1

    def __post_init__(self) -> None:
        orientations = tuple(str(value).casefold() for value in self.orientations)
        if not orientations or len(set(orientations)) != len(orientations):
            raise ValueError("orientations must contain unique inline/xline values.")
        if any(value not in {"inline", "xline"} for value in orientations):
            raise ValueError("orientations must contain inline and/or xline.")
        for name in ("log_every_sections", "max_workers"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        object.__setattr__(self, "orientations", orientations)
        object.__setattr__(self, "log_every_sections", int(self.log_every_sections))
        object.__setattr__(self, "max_workers", int(self.max_workers))


@dataclass(frozen=True)
class VolumeTransferResult:
    """Fused residual and enhanced body fields."""

    predicted_residual_log_ai: np.ndarray
    enhanced_log_ai: np.ndarray
    direction_count: np.ndarray
    summary: Mapping[str, float | int | list[int]]

    def __post_init__(self) -> None:
        residual = np.asarray(self.predicted_residual_log_ai, dtype=np.float32)
        enhanced = np.asarray(self.enhanced_log_ai, dtype=np.float32)
        count = np.asarray(self.direction_count, dtype=np.uint8)
        if residual.ndim != 3 or enhanced.shape != residual.shape or count.shape != residual.shape:
            raise ValueError("VolumeTransferResult arrays must share [inline, xline, sample] shape.")
        if np.any(~np.isfinite(residual)) or np.any(count > 2):
            raise ValueError("Residual must be finite and direction_count must be in [0, 2].")
        if np.any(np.isfinite(enhanced) != (count > 0)):
            raise ValueError("Enhanced log-AI finite support must equal direction_count > 0.")
        object.__setattr__(self, "predicted_residual_log_ai", residual)
        object.__setattr__(self, "enhanced_log_ai", enhanced)
        object.__setattr__(self, "direction_count", count)
        object.__setattr__(self, "summary", dict(self.summary))


def _section_zone_geometry(
    body: np.ndarray,
    axis: SampleAxis,
    zones: tuple[ZoneSurface, ...],
    *,
    orientation: str,
    fixed_index: int,
    line_geometry: SurveyLineGeometry,
    ilines: np.ndarray,
    xlines: np.ndarray,
) -> TransferGeometry:
    if orientation == "inline":
        trace_count = body.shape[0]
        xy = np.asarray(
            [line_geometry.line_to_coord(ilines[fixed_index], value) for value in xlines],
            dtype=np.float64,
        )
    else:
        trace_count = body.shape[0]
        xy = np.asarray(
            [line_geometry.line_to_coord(value, xlines[fixed_index]) for value in ilines],
            dtype=np.float64,
        )
    if xy.shape != (trace_count, 2):
        raise ValueError("Section physical coordinates differ from body trace count.")

    zone_ids = np.empty(body.shape, dtype=object)
    zone_ids[:] = None
    support = np.zeros(body.shape, dtype=bool)
    samples = np.asarray(axis.values, dtype=np.float64)
    for zone_index, zone in enumerate(zones):
        if orientation == "inline":
            top = zone.top[fixed_index]
            bottom = zone.bottom[fixed_index]
        else:
            top = zone.top[:, fixed_index]
            bottom = zone.bottom[:, fixed_index]
        valid_trace = np.isfinite(top) & np.isfinite(bottom) & (top < bottom)
        local = valid_trace[:, None] & (samples[None, :] >= top[:, None])
        if zone_index == len(zones) - 1:
            local &= samples[None, :] <= bottom[:, None]
        else:
            local &= samples[None, :] < bottom[:, None]
        local &= np.isfinite(body)
        if np.any(local & support):
            raise ValueError("Zone surfaces overlap on the transfer sample axis.")
        support |= local
        zone_ids[local] = zone.zone_id
    return TransferGeometry(
        sample_axis=axis,
        x_m=xy[:, 0],
        y_m=xy[:, 1],
        support=support,
        zone_ids=zone_ids,
        pinchout_mask=np.zeros(body.shape, dtype=bool),
        orientation=orientation,
    )


_WORKER_LIBRARY: ResidualTextureLibrary | None = None
_WORKER_POLICY: ResidualTransferPolicy | None = None
_WORKER_SAMPLE_AXIS: SampleAxis | None = None
_WORKER_LINE_GEOMETRY: SurveyLineGeometry | None = None
_WORKER_ILINES: np.ndarray | None = None
_WORKER_XLINES: np.ndarray | None = None
_WORKER_ZONES: tuple[ZoneSurface, ...] | None = None


def _initialize_section_worker(
    library: ResidualTextureLibrary,
    policy: ResidualTransferPolicy,
    sample_axis: SampleAxis,
    line_geometry: SurveyLineGeometry,
    ilines: np.ndarray,
    xlines: np.ndarray,
    zones: tuple[ZoneSurface, ...],
) -> None:
    global _WORKER_LIBRARY
    global _WORKER_POLICY
    global _WORKER_SAMPLE_AXIS
    global _WORKER_LINE_GEOMETRY
    global _WORKER_ILINES
    global _WORKER_XLINES
    global _WORKER_ZONES
    _WORKER_LIBRARY = library
    _WORKER_POLICY = policy
    _WORKER_SAMPLE_AXIS = sample_axis
    _WORKER_LINE_GEOMETRY = line_geometry
    _WORKER_ILINES = np.asarray(ilines, dtype=np.float64)
    _WORKER_XLINES = np.asarray(xlines, dtype=np.float64)
    _WORKER_ZONES = zones


def _transfer_section_worker(
    task: tuple[str, int, np.ndarray],
) -> tuple[str, int, np.ndarray, np.ndarray, int, int]:
    orientation, fixed_index, section_body = task
    if (
        _WORKER_LIBRARY is None
        or _WORKER_POLICY is None
        or _WORKER_SAMPLE_AXIS is None
        or _WORKER_LINE_GEOMETRY is None
        or _WORKER_ILINES is None
        or _WORKER_XLINES is None
        or _WORKER_ZONES is None
    ):
        raise RuntimeError("Enhance V2 section worker was not initialized.")
    geometry = _section_zone_geometry(
        section_body,
        _WORKER_SAMPLE_AXIS,
        _WORKER_ZONES,
        orientation=orientation,
        fixed_index=fixed_index,
        line_geometry=_WORKER_LINE_GEOMETRY,
        ilines=_WORKER_ILINES,
        xlines=_WORKER_XLINES,
    )
    result = transfer_residual_field(
        section_body,
        geometry,
        _WORKER_LIBRARY,
        _WORKER_POLICY,
    )
    return (
        orientation,
        int(fixed_index),
        np.asarray(result.predicted_residual, dtype=np.float32),
        np.asarray(result.support, dtype=bool),
        int(result.node_count),
        int(result.graph_edge_count),
    )


class ResidualTextureVolumeTransfer:
    """Run the 1-D spatial dictionary solver along both survey directions."""

    def __init__(
        self,
        library: ResidualTextureLibrary,
        policy: ResidualTransferPolicy,
        config: VolumeTransferConfig,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(library, ResidualTextureLibrary):
            raise TypeError("library must be a ResidualTextureLibrary.")
        if not isinstance(policy, ResidualTransferPolicy):
            raise TypeError("policy must be a ResidualTransferPolicy.")
        self.library = library
        self.policy = policy
        self.config = config
        self.log = logger or logging.getLogger(__name__)

    def transfer(
        self,
        body_log_ai: np.ndarray,
        *,
        sample_axis: SampleAxis,
        line_geometry: SurveyLineGeometry,
        ilines: np.ndarray,
        xlines: np.ndarray,
        zones: tuple[ZoneSurface, ...],
    ) -> VolumeTransferResult:
        """Transfer residual texture to a complete body volume."""

        body = np.asarray(body_log_ai, dtype=np.float32)
        il_axis = np.asarray(ilines, dtype=np.float64)
        xl_axis = np.asarray(xlines, dtype=np.float64)
        expected_shape = (il_axis.size, xl_axis.size, sample_axis.values.size)
        if body.shape != expected_shape:
            raise ValueError(f"body_log_ai shape {body.shape} differs from axes {expected_shape}.")
        if not zones:
            raise ValueError("At least one ZoneSurface is required.")
        if any(zone.top.shape != body.shape[:2] for zone in zones):
            raise ValueError("ZoneSurface grids must match body lateral shape.")
        if set(zone.zone_id for zone in zones) != set(self.library.zone_ids):
            raise ValueError("Volume zone ids differ from the residual library zone ids.")

        residual_sum = np.zeros(body.shape, dtype=np.float32)
        direction_count = np.zeros(body.shape, dtype=np.uint8)
        total_sections = sum(
            body.shape[0] if orientation == "inline" else body.shape[1]
            for orientation in self.config.orientations
        )
        completed = 0
        node_count = 0
        edge_count = 0
        disagreement_square_sum = 0.0
        disagreement_count = 0
        started = time.perf_counter()

        def tasks():
            for orientation in self.config.orientations:
                section_count = body.shape[0] if orientation == "inline" else body.shape[1]
                for fixed_index in range(section_count):
                    section = body[fixed_index] if orientation == "inline" else body[:, fixed_index]
                    yield orientation, fixed_index, np.ascontiguousarray(section, dtype=np.float32)

        def consume(
            payload: tuple[str, int, np.ndarray, np.ndarray, int, int],
        ) -> None:
            nonlocal completed
            nonlocal node_count
            nonlocal edge_count
            nonlocal disagreement_square_sum
            nonlocal disagreement_count
            orientation, fixed_index, local_residual, local_support, local_nodes, local_edges = payload
            if orientation == "inline":
                previous = residual_sum[fixed_index]
                previous_count = direction_count[fixed_index]
            else:
                previous = residual_sum[:, fixed_index]
                previous_count = direction_count[:, fixed_index]
            paired = local_support & (previous_count == 1)
            if np.any(paired):
                difference = local_residual[paired] - previous[paired]
                disagreement_square_sum += float(np.sum(np.square(difference, dtype=np.float64)))
                disagreement_count += int(difference.size)
            previous[local_support] += local_residual[local_support]
            previous_count[local_support] += np.uint8(1)
            node_count += int(local_nodes)
            edge_count += int(local_edges)
            completed += 1
            if completed % self.config.log_every_sections == 0 or completed == total_sections:
                self.log.info(
                    "enhance volume %d/%d sections | orientation=%s | fixed_index=%d | workers=%d | elapsed=%.1fs",
                    completed,
                    total_sections,
                    orientation,
                    fixed_index,
                    self.config.max_workers,
                    time.perf_counter() - started,
                )

        worker_arguments = (
            self.library,
            self.policy,
            sample_axis,
            line_geometry,
            il_axis,
            xl_axis,
            zones,
        )
        if self.config.max_workers == 1:
            _initialize_section_worker(*worker_arguments)
            for task in tasks():
                consume(_transfer_section_worker(task))
        else:
            task_iterator = iter(tasks())
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers,
                initializer=_initialize_section_worker,
                initargs=worker_arguments,
            ) as executor:
                pending = set()
                for _ in range(2 * self.config.max_workers):
                    try:
                        pending.add(executor.submit(_transfer_section_worker, next(task_iterator)))
                    except StopIteration:
                        break
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        consume(future.result())
                        try:
                            pending.add(executor.submit(_transfer_section_worker, next(task_iterator)))
                        except StopIteration:
                            pass

        expected_support = np.isfinite(body)
        if np.any((direction_count > 0) != expected_support):
            missing = int(np.count_nonzero(expected_support & (direction_count == 0)))
            unexpected = int(np.count_nonzero(~expected_support & (direction_count > 0)))
            raise ValueError(
                "Enhance V2 section fusion support differs from GINN body support: "
                f"missing={missing}, unexpected={unexpected}."
            )
        residual = np.zeros(body.shape, dtype=np.float32)
        residual[expected_support] = (
            residual_sum[expected_support]
            / direction_count[expected_support].astype(np.float32)
        )
        enhanced = np.full(body.shape, np.nan, dtype=np.float32)
        enhanced[expected_support] = body[expected_support] + residual[expected_support]
        selected = residual[expected_support]
        summary: dict[str, float | int | list[int]] = {
            "shape": [int(value) for value in body.shape],
            "section_count": int(total_sections),
            "node_count": int(node_count),
            "graph_edge_count": int(edge_count),
            "support_count": int(np.count_nonzero(expected_support)),
            "single_direction_fraction": float(
                np.count_nonzero(direction_count == 1) / max(np.count_nonzero(expected_support), 1)
            ),
            "two_direction_fraction": float(
                np.count_nonzero(direction_count == 2) / max(np.count_nonzero(expected_support), 1)
            ),
            "residual_rms_log_ai": (
                float(np.sqrt(np.mean(np.square(selected, dtype=np.float64))))
                if selected.size
                else 0.0
            ),
            "direction_disagreement_rms_log_ai": (
                float(np.sqrt(disagreement_square_sum / disagreement_count))
                if disagreement_count
                else 0.0
            ),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        return VolumeTransferResult(
            predicted_residual_log_ai=residual,
            enhanced_log_ai=enhanced,
            direction_count=direction_count,
            summary=summary,
        )


__all__ = [
    "ResidualTextureVolumeTransfer",
    "VolumeTransferConfig",
    "VolumeTransferResult",
    "ZoneSurface",
]
