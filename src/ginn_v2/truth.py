"""Structured truth contracts, explicit adapters, and persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

from cup.synthetic.core.records import SampleAxis
from cup.synthetic.core.truth import SyntheticTruth


ARTIFACT_TYPE = "structured_truth_v1"
ARTIFACT_VERSION = 1
_IDENTITY_KEYS = ("producer", "calibration", "projection", "forward")
_ARRAY_NAMES = (
    "observed_axis",
    "observed_seismic",
    "model_consistent_seismic",
    "observed_lfm",
    "observed_valid",
    "latent_axis",
    "latent_log_ai_highres_truth",
    "latent_valid",
    "latent_state_id",
    "latent_object_id",
    "latent_object_xi",
    "latent_zone_id",
    "latent_clipping_mask",
    "zone_valid",
)


def _required_mapping(
    value: Mapping[str, Any],
    required: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    missing = sorted(set(required).difference(value))
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}.")
    return dict(value)


def _float_array(value: Any, *, label: str, ndim: int = 1) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a numeric array.") from exc
    if array.ndim != ndim:
        raise ValueError(f"{label} must be {ndim}D, got shape {array.shape}.")
    return np.array(array, dtype=np.float64, copy=True)


def _bool_array(value: Any, *, label: str, ndim: int = 1) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype != np.dtype(bool):
        raise TypeError(f"{label} must be a boolean {ndim}D array.")
    return np.array(array, dtype=bool, copy=True)


def _int_array(value: Any, *, label: str, ndim: int = 1) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{label} must be an integer {ndim}D array.")
    return np.array(array, dtype=np.int64, copy=True)


def _finite_scalar(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _axis_manifest(axis: SampleAxis) -> dict[str, Any]:
    return {
        "sample_domain": axis.sample_domain,
        "unit": axis.unit,
        "sample_interval": float(axis.sample_interval),
        "positive_direction": axis.positive_direction,
        "depth_basis": axis.depth_basis,
    }


def _axis_from_manifest(
    metadata: Mapping[str, Any],
    coordinates: np.ndarray,
    *,
    label: str,
) -> SampleAxis:
    fields = _required_mapping(
        metadata,
        (
            "sample_domain",
            "unit",
            "sample_interval",
            "positive_direction",
            "depth_basis",
        ),
        label=label,
    )
    return SampleAxis(
        sample_domain=str(fields["sample_domain"]),
        unit=str(fields["unit"]),
        coordinates=coordinates,
        sample_interval=float(fields["sample_interval"]),
        positive_direction=str(fields["positive_direction"]),
        depth_basis=None if fields["depth_basis"] is None else str(fields["depth_basis"]),
    )


def _assert_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    result = _required_mapping(identity, _IDENTITY_KEYS, label="structured truth identity")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("structured truth identity must be JSON serializable.") from exc
    return result


def _axis_nested(highres: SampleAxis, model: SampleAxis) -> int:
    if highres.sample_domain != model.sample_domain or highres.unit != model.unit:
        raise ValueError("high-resolution and model axes must share domain and unit.")
    if highres.depth_basis != model.depth_basis:
        raise ValueError("high-resolution and model axes must share depth_basis.")
    ratio = model.sample_interval / highres.sample_interval
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-12):
        raise ValueError("model axis is not an integer nested high-resolution axis.")
    nested = highres.coordinates[::factor]
    if nested.shape != model.coordinates.shape or not np.allclose(
        nested,
        model.coordinates,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("model axis is not exactly nested in the high-resolution axis.")
    return factor


@dataclass(frozen=True)
class ObservedTrace:
    """One model-grid observed trace with an explicit mask."""

    sample_axis: SampleAxis
    seismic: np.ndarray
    lfm: np.ndarray
    observed_valid: np.ndarray
    lfm_source_identity: Mapping[str, Any]
    model_consistent_seismic: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_axis, SampleAxis):
            raise TypeError("ObservedTrace.sample_axis must be SampleAxis.")
        size = self.sample_axis.coordinates.size
        seismic = _float_array(self.seismic, label="ObservedTrace.seismic")
        model_consistent = _float_array(
            seismic if self.model_consistent_seismic is None else self.model_consistent_seismic,
            label="ObservedTrace.model_consistent_seismic",
        )
        lfm = _float_array(self.lfm, label="ObservedTrace.lfm")
        valid = _bool_array(self.observed_valid, label="ObservedTrace.observed_valid")
        for name, array in (
            ("seismic", seismic),
            ("model_consistent_seismic", model_consistent),
            ("lfm", lfm),
            ("observed_valid", valid),
        ):
            if array.shape != (size,):
                raise ValueError(
                    f"ObservedTrace.{name} shape {array.shape} does not match axis {(size,)}."
                )
        if np.any(
            valid
            & (
                ~np.isfinite(seismic)
                | ~np.isfinite(model_consistent)
                | ~np.isfinite(lfm)
            )
        ):
            raise ValueError("ObservedTrace valid samples must be finite.")
        if not isinstance(self.lfm_source_identity, Mapping) or not self.lfm_source_identity:
            raise ValueError("ObservedTrace.lfm_source_identity must be explicit.")
        try:
            json.dumps(dict(self.lfm_source_identity), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("ObservedTrace.lfm_source_identity must be JSON serializable.") from exc
        object.__setattr__(self, "seismic", seismic)
        object.__setattr__(self, "model_consistent_seismic", model_consistent)
        object.__setattr__(self, "lfm", lfm)
        object.__setattr__(self, "observed_valid", valid)
        object.__setattr__(self, "lfm_source_identity", dict(self.lfm_source_identity))


@dataclass(frozen=True)
class LatentTrace:
    """One high-resolution latent trace and its independently named mask."""

    latent_axis: SampleAxis
    latent_valid: np.ndarray
    log_ai_highres_truth: np.ndarray
    state_id: np.ndarray
    object_id: np.ndarray
    object_xi: np.ndarray
    zone_id: np.ndarray
    clipping_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.latent_axis, SampleAxis):
            raise TypeError("LatentTrace.latent_axis must be SampleAxis.")
        size = self.latent_axis.coordinates.size
        valid = _bool_array(self.latent_valid, label="LatentTrace.latent_valid")
        values = _float_array(
            self.log_ai_highres_truth,
            label="LatentTrace.log_ai_highres_truth",
        )
        if valid.shape != (size,) or values.shape != (size,):
            raise ValueError("LatentTrace arrays must match latent_axis.")
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError("LatentTrace valid samples must have finite log AI truth.")
        state_id = _int_array(self.state_id, label="LatentTrace.state_id")
        object_id = _int_array(self.object_id, label="LatentTrace.object_id")
        zone_id = _int_array(self.zone_id, label="LatentTrace.zone_id")
        object_xi = _float_array(self.object_xi, label="LatentTrace.object_xi")
        clipping_mask = _bool_array(
            self.clipping_mask,
            label="LatentTrace.clipping_mask",
        )
        for name, array in (
            ("state_id", state_id),
            ("object_id", object_id),
            ("object_xi", object_xi),
            ("zone_id", zone_id),
            ("clipping_mask", clipping_mask),
        ):
            if array.shape != (size,):
                raise ValueError(f"LatentTrace.{name} must match latent_axis.")
        object_samples = object_id >= 0
        if np.any(object_samples & (~np.isfinite(object_xi) | (object_xi < 0.0) | (object_xi > 1.0))):
            raise ValueError("LatentTrace.object_xi is invalid on object samples.")
        if np.any(clipping_mask & ~np.isfinite(values)):
            raise ValueError("LatentTrace.clipping_mask marks non-finite samples.")
        object.__setattr__(self, "latent_valid", valid)
        object.__setattr__(self, "log_ai_highres_truth", values)
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "object_xi", object_xi)
        object.__setattr__(self, "zone_id", zone_id)
        object.__setattr__(self, "clipping_mask", clipping_mask)


@dataclass(frozen=True)
class ZoneTruth:
    """One explicit zone and its realization-level shared background."""

    zone_id: str
    top: float
    bottom: float
    background_a: float
    background_b: float
    zone_valid: np.ndarray

    def __post_init__(self) -> None:
        zone_id = str(self.zone_id).strip()
        if not zone_id:
            raise ValueError("ZoneTruth.zone_id must be non-empty.")
        top = _finite_scalar(self.top, label="ZoneTruth.top")
        bottom = _finite_scalar(self.bottom, label="ZoneTruth.bottom")
        if not top < bottom:
            raise ValueError("ZoneTruth.top must be smaller than bottom.")
        background_a = _finite_scalar(self.background_a, label="ZoneTruth.background_a")
        background_b = _finite_scalar(self.background_b, label="ZoneTruth.background_b")
        valid = _bool_array(self.zone_valid, label="ZoneTruth.zone_valid")
        if not np.any(valid):
            raise ValueError("ZoneTruth.zone_valid must contain at least one valid sample.")
        object.__setattr__(self, "zone_id", zone_id)
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "bottom", bottom)
        object.__setattr__(self, "background_a", background_a)
        object.__setattr__(self, "background_b", background_b)
        object.__setattr__(self, "zone_valid", valid)


@dataclass(frozen=True)
class RawSegmentParameters:
    """A complete object description with raw coefficients for the decoder."""

    zone_id: str
    object_id: int | str
    state: str
    state_id: int
    top: Any
    bottom: Any
    c0: Any
    c1: Any
    c2: Any


@dataclass(frozen=True)
class SegmentTruth:
    """One supervised semi-Markov object and all coefficient stages."""

    zone_id: str
    object_id: int | str
    state: str
    state_id: int
    top: float
    bottom: float
    duration_fraction: float
    duration_samples: int
    c0_raw: np.ndarray
    c1_raw: np.ndarray
    c2_raw: np.ndarray
    c0_projected: np.ndarray
    c1_projected: np.ndarray
    c2_projected: np.ndarray
    c0_effective: np.ndarray
    c1_effective: np.ndarray
    c2_effective: np.ndarray
    segment_supervision_valid: bool

    def __post_init__(self) -> None:
        zone_id = str(self.zone_id).strip()
        state = str(self.state).strip()
        if not zone_id or not state:
            raise ValueError("SegmentTruth.zone_id and state must be non-empty.")
        if isinstance(self.object_id, bool) or str(self.object_id).strip() == "":
            raise ValueError("SegmentTruth.object_id must be explicit and non-empty.")
        state_id = int(self.state_id)
        if state_id not in {0, 1, 2}:
            raise ValueError("SegmentTruth.state_id must be 0, 1, or 2.")
        top = _finite_scalar(self.top, label="SegmentTruth.top")
        bottom = _finite_scalar(self.bottom, label="SegmentTruth.bottom")
        if not top < bottom:
            raise ValueError("SegmentTruth.top must be smaller than bottom.")
        duration_fraction = _finite_scalar(
            self.duration_fraction,
            label="SegmentTruth.duration_fraction",
        )
        if not 0.0 < duration_fraction <= 1.0:
            raise ValueError("SegmentTruth.duration_fraction must be in (0, 1].")
        if (
            isinstance(self.duration_samples, bool)
            or int(self.duration_samples) != self.duration_samples
            or int(self.duration_samples) <= 0
        ):
            raise ValueError("SegmentTruth.duration_samples must be a positive integer.")
        arrays = []
        for name in (
            "c0_raw",
            "c1_raw",
            "c2_raw",
            "c0_projected",
            "c1_projected",
            "c2_projected",
            "c0_effective",
            "c1_effective",
            "c2_effective",
        ):
            value = _float_array(getattr(self, name), label=f"SegmentTruth.{name}")
            if value.size != 1:
                raise ValueError(f"SegmentTruth.{name} must contain exactly one scalar.")
            arrays.append(float(value.reshape(-1)[0]))
        object.__setattr__(self, "zone_id", zone_id)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "bottom", bottom)
        object.__setattr__(self, "duration_fraction", duration_fraction)
        object.__setattr__(self, "duration_samples", int(self.duration_samples))
        for name, value in zip(
            (
                "c0_raw",
                "c1_raw",
                "c2_raw",
                "c0_projected",
                "c1_projected",
                "c2_projected",
                "c0_effective",
                "c1_effective",
                "c2_effective",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, np.asarray([value], dtype=np.float64))
        object.__setattr__(
            self,
            "segment_supervision_valid",
            bool(self.segment_supervision_valid),
        )

    def raw_parameters(self) -> RawSegmentParameters:
        return RawSegmentParameters(
            zone_id=self.zone_id,
            object_id=self.object_id,
            state=self.state,
            state_id=self.state_id,
            top=self.top,
            bottom=self.bottom,
            c0=float(self.c0_raw[0]),
            c1=float(self.c1_raw[0]),
            c2=float(self.c2_raw[0]),
        )


@dataclass(frozen=True)
class StructuredSample:
    """The smallest Stage 1 object consumed by decoder and Oracle."""

    realization_id: str
    lateral_index: int
    lateral_m: float
    inline: float
    xline: float
    xline_step: float
    observed: ObservedTrace
    latent: LatentTrace
    zone: ZoneTruth
    segments: tuple[SegmentTruth, ...]
    identity: Mapping[str, Any]
    forward_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        realization_id = str(self.realization_id).strip()
        if not realization_id:
            raise ValueError("StructuredSample.realization_id must be non-empty.")
        if (
            isinstance(self.lateral_index, bool)
            or int(self.lateral_index) != self.lateral_index
            or int(self.lateral_index) < 0
        ):
            raise ValueError("StructuredSample.lateral_index must be a non-negative integer.")
        geometry = {
            name: _finite_scalar(getattr(self, name), label=f"StructuredSample.{name}")
            for name in ("lateral_m", "inline", "xline", "xline_step")
        }
        if geometry["xline_step"] == 0.0:
            raise ValueError("StructuredSample.xline_step must be non-zero.")
        if not isinstance(self.observed, ObservedTrace):
            raise TypeError("StructuredSample.observed must be ObservedTrace.")
        if not isinstance(self.latent, LatentTrace):
            raise TypeError("StructuredSample.latent must be LatentTrace.")
        if not isinstance(self.zone, ZoneTruth):
            raise TypeError("StructuredSample.zone must be ZoneTruth.")
        segments = tuple(self.segments)
        if not segments or not all(isinstance(item, SegmentTruth) for item in segments):
            raise TypeError("StructuredSample.segments must be a non-empty tuple of SegmentTruth.")
        _assert_identity(self.identity)
        if not isinstance(self.forward_context, Mapping):
            raise TypeError("StructuredSample.forward_context must be a mapping.")
        try:
            json.dumps(dict(self.forward_context), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("StructuredSample.forward_context must be JSON serializable.") from exc
        _axis_nested(self.latent.latent_axis, self.observed.sample_axis)
        if self.zone.zone_valid.shape != self.latent.latent_valid.shape:
            raise ValueError("ZoneTruth.zone_valid must match latent axis length.")
        latent_axis = self.latent.latent_axis.coordinates
        tolerance = max(self.latent.latent_axis.sample_interval * 1e-6, 1e-9)
        geometric_zone = (latent_axis >= self.zone.top - tolerance) & (
            latent_axis <= self.zone.bottom + tolerance
        )
        if np.any(self.zone.zone_valid & ~geometric_zone):
            raise ValueError("ZoneTruth.zone_valid contains samples outside zone endpoints.")
        if np.any(self.zone.zone_valid & ~self.latent.latent_valid):
            raise ValueError("ZoneTruth.zone_valid must be a subset of latent_valid.")
        previous_bottom: float | None = None
        object_ids: set[object] = set()
        for index, segment in enumerate(segments):
            if segment.zone_id != self.zone.zone_id:
                raise ValueError("Every SegmentTruth must belong to the StructuredSample zone.")
            if segment.object_id in object_ids:
                raise ValueError("StructuredSample contains duplicate object_id values.")
            object_ids.add(segment.object_id)
            if segment.top < self.zone.top - tolerance or segment.bottom > self.zone.bottom + tolerance:
                raise ValueError("SegmentTruth endpoints must lie inside the zone.")
            if previous_bottom is not None and not np.isclose(
                segment.top,
                previous_bottom,
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("SegmentTruth endpoints must be contiguous and ordered.")
            previous_bottom = segment.bottom
            if index > 0 and segment.top < segments[index - 1].top:
                raise ValueError("SegmentTruth rows must be ordered by top.")
        if not np.isclose(segments[0].top, self.zone.top, rtol=0.0, atol=tolerance):
            raise ValueError("First segment must start at the zone top.")
        if not np.isclose(segments[-1].bottom, self.zone.bottom, rtol=0.0, atol=tolerance):
            raise ValueError("Last segment must end at the zone bottom.")
        object.__setattr__(self, "realization_id", realization_id)
        object.__setattr__(self, "lateral_index", int(self.lateral_index))
        object.__setattr__(self, "lateral_m", geometry["lateral_m"])
        object.__setattr__(self, "inline", geometry["inline"])
        object.__setattr__(self, "xline", geometry["xline"])
        object.__setattr__(self, "xline_step", geometry["xline_step"])
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "identity", dict(self.identity))
        object.__setattr__(self, "forward_context", dict(self.forward_context))


class StructuredTruthAdapter:
    """Adapt explicit generator output plus a structured sidecar into one sample."""

    @classmethod
    def from_synthetic_truth(
        cls,
        truth: SyntheticTruth,
        *,
        lateral_index: int,
        model_axis: SampleAxis,
        seismic: Any,
        lfm: Any,
        observed_valid: Any,
        lfm_source_identity: Mapping[str, Any],
        latent_valid: Any,
        zone: Mapping[str, Any],
        segments: Sequence[Mapping[str, Any]],
        identity: Mapping[str, Any],
        xline_step: float,
        model_consistent_seismic: Any | None = None,
        forward_context: Mapping[str, Any] | None = None,
    ) -> StructuredSample:
        if not isinstance(truth, SyntheticTruth):
            raise TypeError("StructuredTruthAdapter requires cup.synthetic.core.truth.SyntheticTruth.")
        if not isinstance(model_axis, SampleAxis):
            raise TypeError("StructuredTruthAdapter.model_axis must be SampleAxis.")
        if (
            isinstance(lateral_index, bool)
            or int(lateral_index) != lateral_index
            or int(lateral_index) < 0
        ):
            raise ValueError("lateral_index must be a non-negative integer.")
        lateral_index = int(lateral_index)
        if lateral_index >= truth.lateral_m.size:
            raise IndexError("lateral_index is outside SyntheticTruth.lateral_m.")
        required_zone = (
            "zone_id",
            "zone_grid_value",
            "top",
            "bottom",
            "background_a",
            "background_b",
            "zone_valid",
        )
        zone_data = _required_mapping(zone, required_zone, label="structured zone sidecar")
        if len(segments) == 0:
            raise ValueError("structured segment sidecar must not be empty.")
        highres_axis = SampleAxis(
            sample_domain=truth.sample_domain,
            unit=truth.axis_unit,
            coordinates=truth.highres_axis,
            sample_interval=float(truth.highres_sample_interval),
            positive_direction="down" if truth.sample_domain == "depth" else "increasing_time",
            depth_basis="tvdss" if truth.sample_domain == "depth" else None,
        )
        _axis_nested(highres_axis, model_axis)
        log_ai = _float_array(
            np.asarray(truth.log_ai_highres)[lateral_index],
            label="SyntheticTruth.log_ai_highres[lateral_index]",
        )
        valid_latent = _bool_array(latent_valid, label="structured latent_valid")
        if valid_latent.shape != log_ai.shape:
            raise ValueError("structured latent_valid must match SyntheticTruth high-resolution trace.")
        zone_valid = _bool_array(zone_data["zone_valid"], label="structured zone_valid")
        if zone_valid.shape != log_ai.shape:
            raise ValueError("structured zone_valid must match SyntheticTruth high-resolution trace.")
        zone_grid = np.asarray(truth.zone_id_highres)[lateral_index]
        object_grid = np.asarray(truth.object_id_highres)[lateral_index]
        state_grid = np.asarray(truth.state_id_highres)[lateral_index]
        object_xi_grid = np.asarray(truth.object_xi_highres)[lateral_index]
        zone_grid_value = int(zone_data["zone_grid_value"])
        if np.any(zone_valid & (zone_grid != zone_grid_value)):
            raise ValueError("structured zone_valid disagrees with SyntheticTruth.zone_id_highres.")
        segment_rows: list[SegmentTruth] = []
        required_segment = (
            "zone_id",
            "object_id",
            "state",
            "state_id",
            "top",
            "bottom",
            "duration_fraction",
            "duration_samples",
            "c0_raw",
            "c1_raw",
            "c2_raw",
            "c0_projected",
            "c1_projected",
            "c2_projected",
            "c0_effective",
            "c1_effective",
            "c2_effective",
            "segment_supervision_valid",
        )
        tolerance = max(highres_axis.sample_interval * 1e-6, 1e-9)
        for row in segments:
            data = _required_mapping(row, required_segment, label="structured segment sidecar")
            object_id = data["object_id"]
            object_mask = zone_valid & (object_grid == object_id)
            if not np.any(object_mask):
                raise ValueError(
                    f"structured object_id {object_id!r} is absent from the selected SyntheticTruth trace."
                )
            state_id = int(data["state_id"])
            if np.any(state_grid[object_mask] != state_id):
                raise ValueError("structured segment state_id disagrees with SyntheticTruth state grid.")
            xi = np.asarray(object_xi_grid[object_mask], dtype=np.float64)
            if np.any(~np.isfinite(xi)) or np.any((xi < -tolerance) | (xi > 1.0 + tolerance)):
                raise ValueError("SyntheticTruth object_xi is invalid for a structured segment.")
            top = _finite_scalar(data["top"], label="structured segment top")
            bottom = _finite_scalar(data["bottom"], label="structured segment bottom")
            object_coordinates = highres_axis.coordinates[object_mask]
            if np.any(object_coordinates < top - tolerance) or np.any(object_coordinates > bottom + tolerance):
                raise ValueError("structured segment endpoints do not contain its SyntheticTruth object samples.")
            segment_rows.append(
                SegmentTruth(
                    zone_id=str(data["zone_id"]),
                    object_id=object_id,
                    state=str(data["state"]),
                    state_id=state_id,
                    top=top,
                    bottom=bottom,
                    duration_fraction=float(data["duration_fraction"]),
                    duration_samples=int(data["duration_samples"]),
                    c0_raw=data["c0_raw"],
                    c1_raw=data["c1_raw"],
                    c2_raw=data["c2_raw"],
                    c0_projected=data["c0_projected"],
                    c1_projected=data["c1_projected"],
                    c2_projected=data["c2_projected"],
                    c0_effective=data["c0_effective"],
                    c1_effective=data["c1_effective"],
                    c2_effective=data["c2_effective"],
                    segment_supervision_valid=bool(data["segment_supervision_valid"]),
                )
            )
        return StructuredSample(
            realization_id=truth.realization_id,
            lateral_index=lateral_index,
            lateral_m=float(truth.lateral_m[lateral_index]),
            inline=float(truth.inline_float[lateral_index]),
            xline=float(truth.xline_float[lateral_index]),
            xline_step=_finite_scalar(xline_step, label="xline_step"),
            observed=ObservedTrace(
                sample_axis=model_axis,
                seismic=seismic,
                lfm=lfm,
                observed_valid=observed_valid,
                lfm_source_identity=lfm_source_identity,
                model_consistent_seismic=model_consistent_seismic,
            ),
            latent=LatentTrace(
                latent_axis=highres_axis,
                latent_valid=valid_latent,
                log_ai_highres_truth=log_ai,
                state_id=np.asarray(truth.state_id_highres)[lateral_index],
                object_id=np.asarray(truth.object_id_highres)[lateral_index],
                object_xi=np.asarray(truth.object_xi_highres)[lateral_index],
                zone_id=np.asarray(truth.zone_id_highres)[lateral_index],
                clipping_mask=np.asarray(truth.clipping_mask_highres)[lateral_index],
            ),
            zone=ZoneTruth(
                zone_id=str(zone_data["zone_id"]),
                top=zone_data["top"],
                bottom=zone_data["bottom"],
                background_a=zone_data["background_a"],
                background_b=zone_data["background_b"],
                zone_valid=zone_valid,
            ),
            segments=tuple(segment_rows),
            identity=_assert_identity(identity),
            forward_context={} if forward_context is None else dict(forward_context),
        )

    @classmethod
    def from_structured_sample_record(
        cls,
        record: Any,
        *,
        zone_id: str,
        lateral_index: int,
        xline_step: float | None = None,
        identity: Mapping[str, Any] | None = None,
    ) -> StructuredSample:
        """Build Stage 1 truth from the producer-owned no-increment record."""
        from cup.synthetic.core.records import StructuredSampleRecord

        if not isinstance(record, StructuredSampleRecord):
            raise TypeError(
                "StructuredTruthAdapter requires cup.synthetic.core.records.StructuredSampleRecord."
            )
        truth = record.truth
        lateral_index = int(lateral_index)
        selected_zone = str(zone_id).strip()
        zone_rows = [
            row
            for row in truth.structured_zone_truth
            if str(row.get("zone_id")) == selected_zone
            and int(row.get("lateral_index", -1)) == lateral_index
        ]
        if len(zone_rows) != 1:
            raise ValueError(
                "StructuredTruthAdapter requires exactly one explicit producer zone row."
            )
        zone_row = zone_rows[0]
        zone_grid = np.asarray(truth.zone_id_highres)[lateral_index]
        zone_valid = zone_grid == int(zone_row["zone_grid_value"])
        segment_rows = [
            row
            for row in truth.structured_segment_truth
            if str(row.get("zone_id")) == selected_zone
            and int(row.get("lateral_index", -1)) == lateral_index
        ]
        segment_rows.sort(key=lambda row: (float(row["top"]), int(row["object_id"])))
        if not segment_rows:
            raise ValueError("StructuredTruthAdapter found no explicit producer segments.")
        expected_object_ids = {
            int(row["object_id"])
            for row in truth.structured_segment_truth
            if str(row.get("zone_id")) == selected_zone
        }
        actual_object_ids = {int(row["object_id"]) for row in segment_rows}
        if actual_object_ids != expected_object_ids:
            raise ValueError(
                "StructuredTruthAdapter selected lateral trace has an incomplete segment key set."
            )
        resolved_identity = (
            dict(identity)
            if identity is not None
            else dict(record.domain_metadata.get("structured_identity") or {})
        )
        resolved_xline_step = (
            float(xline_step)
            if xline_step is not None
            else float(record.domain_metadata.get("xline_step"))
        )
        return cls.from_synthetic_truth(
            truth,
            lateral_index=lateral_index,
            model_axis=record.projected.model_axis,
            seismic=np.asarray(record.forward.seismic_observed)[lateral_index],
            model_consistent_seismic=np.asarray(
                record.forward.seismic_model_consistent
            )[lateral_index],
            lfm=np.asarray(record.lfm.values)[lateral_index],
            observed_valid=np.asarray(record.valid_mask, dtype=bool)[lateral_index],
            lfm_source_identity=record.lfm.source_identity,
            latent_valid=zone_valid,
            zone={
                **dict(zone_row),
                "zone_valid": zone_valid,
            },
            segments=segment_rows,
            identity=resolved_identity,
            xline_step=resolved_xline_step,
            forward_context=dict(
                record.forward.metadata.get("structured_forward_context") or {}
            ),
        )


def _segment_manifest(segment: SegmentTruth) -> dict[str, Any]:
    return {
        "zone_id": segment.zone_id,
        "object_id": segment.object_id,
        "state": segment.state,
        "state_id": segment.state_id,
        "top": segment.top,
        "bottom": segment.bottom,
        "duration_fraction": segment.duration_fraction,
        "duration_samples": segment.duration_samples,
        "c0_raw": float(segment.c0_raw[0]),
        "c1_raw": float(segment.c1_raw[0]),
        "c2_raw": float(segment.c2_raw[0]),
        "c0_projected": float(segment.c0_projected[0]),
        "c1_projected": float(segment.c1_projected[0]),
        "c2_projected": float(segment.c2_projected[0]),
        "c0_effective": float(segment.c0_effective[0]),
        "c1_effective": float(segment.c1_effective[0]),
        "c2_effective": float(segment.c2_effective[0]),
        "segment_supervision_valid": segment.segment_supervision_valid,
    }


class StructuredTruthArtifactWriter:
    """Write one explicit structured_truth_v1 directory without overwriting."""

    def write(self, sample: StructuredSample, path: str | Path) -> Path:
        if not isinstance(sample, StructuredSample):
            raise TypeError("StructuredTruthArtifactWriter.write requires StructuredSample.")
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"Structured truth artifact already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "observed_axis": sample.observed.sample_axis.coordinates,
            "observed_seismic": sample.observed.seismic,
            "model_consistent_seismic": sample.observed.model_consistent_seismic,
            "observed_lfm": sample.observed.lfm,
            "observed_valid": sample.observed.observed_valid,
            "latent_axis": sample.latent.latent_axis.coordinates,
            "latent_log_ai_highres_truth": sample.latent.log_ai_highres_truth,
            "latent_valid": sample.latent.latent_valid,
            "latent_state_id": sample.latent.state_id,
            "latent_object_id": sample.latent.object_id,
            "latent_object_xi": sample.latent.object_xi,
            "latent_zone_id": sample.latent.zone_id,
            "latent_clipping_mask": sample.latent.clipping_mask,
            "zone_valid": sample.zone.zone_valid,
        }
        manifest = {
            "artifact_type": ARTIFACT_TYPE,
            "artifact_version": ARTIFACT_VERSION,
            "realization_id": sample.realization_id,
            "lateral_index": sample.lateral_index,
            "geometry": {
                "lateral_m": sample.lateral_m,
                "inline": sample.inline,
                "xline": sample.xline,
                "xline_step": sample.xline_step,
            },
            "identity": dict(sample.identity),
            "forward_context": dict(sample.forward_context),
            "lfm": {
                "source_identity": dict(sample.observed.lfm_source_identity),
            },
            "observed_axis": _axis_manifest(sample.observed.sample_axis),
            "latent_axis": _axis_manifest(sample.latent.latent_axis),
            "zone": {
                "zone_id": sample.zone.zone_id,
                "top": sample.zone.top,
                "bottom": sample.zone.bottom,
                "background_a": sample.zone.background_a,
                "background_b": sample.zone.background_b,
            },
            "segments": [_segment_manifest(item) for item in sample.segments],
            "array_file": "arrays.npz",
            "array_names": list(_ARRAY_NAMES),
        }
        try:
            json.dumps(manifest, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("Structured truth manifest is not JSON serializable.") from exc
        temporary_path = target.parent / f".{target.name}.{uuid.uuid4().hex}.staging"
        temporary_path.mkdir(parents=True, exist_ok=False)
        try:
            np.savez_compressed(temporary_path / "arrays.npz", **arrays)
            (temporary_path / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary_path.replace(target)
        finally:
            if temporary_path.exists():
                shutil.rmtree(temporary_path)
        return target


class StructuredTruthArtifactReader:
    """Read and fully validate one structured_truth_v1 directory."""

    def read(self, path: str | Path) -> StructuredSample:
        root = Path(path)
        manifest_path = root / "manifest.json"
        array_path = root / "arrays.npz"
        if not manifest_path.is_file() or not array_path.is_file():
            raise FileNotFoundError(
                f"Structured truth artifact requires {manifest_path} and {array_path}."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_type") != ARTIFACT_TYPE:
            raise ValueError("Structured truth artifact_type is unsupported.")
        if manifest.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("Structured truth artifact_version is unsupported.")
        if manifest.get("array_file") != "arrays.npz":
            raise ValueError("Structured truth artifact must use arrays.npz.")
        if list(manifest.get("array_names") or []) != list(_ARRAY_NAMES):
            raise ValueError("Structured truth artifact array contract is invalid.")
        with np.load(array_path, allow_pickle=False) as data:
            if set(data.files) != set(_ARRAY_NAMES):
                raise ValueError("Structured truth artifact arrays do not match its manifest.")
            arrays = {name: np.array(data[name], copy=True) for name in _ARRAY_NAMES}
        geometry = _required_mapping(
            manifest.get("geometry"),
            ("lateral_m", "inline", "xline", "xline_step"),
            label="structured artifact geometry",
        )
        forward_context = manifest.get("forward_context")
        if not isinstance(forward_context, Mapping):
            raise TypeError("structured artifact forward_context must be a mapping.")
        zone_data = _required_mapping(
            manifest.get("zone"),
            ("zone_id", "top", "bottom", "background_a", "background_b"),
            label="structured artifact zone",
        )
        lfm_data = _required_mapping(
            manifest.get("lfm"),
            ("source_identity",),
            label="structured artifact lfm",
        )
        segments = manifest.get("segments")
        if not isinstance(segments, list):
            raise TypeError("structured artifact segments must be a list.")
        zone_grid_value = zone_data.get("zone_grid_value")
        if zone_grid_value is not None:
            zone_grid_value = int(zone_grid_value)
            zone_mask = arrays["zone_valid"].astype(bool)
            latent_zone = arrays["latent_zone_id"]
            if np.any(zone_mask & (latent_zone != zone_grid_value)):
                raise ValueError(
                    "structured artifact zone_valid disagrees with latent_zone_id."
                )
            for row in segments:
                object_mask = zone_mask & (
                    arrays["latent_object_id"] == int(row["object_id"])
                )
                if not np.any(object_mask):
                    raise ValueError(
                        f"structured artifact object {row['object_id']!r} is absent."
                    )
                if np.any(
                    arrays["latent_state_id"][object_mask] != int(row["state_id"])
                ):
                    raise ValueError(
                        "structured artifact segment state disagrees with latent grid."
                    )
        sample = StructuredSample(
            realization_id=str(manifest["realization_id"]),
            lateral_index=int(manifest["lateral_index"]),
            lateral_m=float(geometry["lateral_m"]),
            inline=float(geometry["inline"]),
            xline=float(geometry["xline"]),
            xline_step=float(geometry["xline_step"]),
            observed=ObservedTrace(
                sample_axis=_axis_from_manifest(
                    manifest["observed_axis"],
                    arrays["observed_axis"],
                    label="structured artifact observed_axis",
                ),
                seismic=arrays["observed_seismic"],
                model_consistent_seismic=arrays["model_consistent_seismic"],
                lfm=arrays["observed_lfm"],
                observed_valid=arrays["observed_valid"],
                lfm_source_identity=lfm_data["source_identity"],
            ),
            latent=LatentTrace(
                latent_axis=_axis_from_manifest(
                    manifest["latent_axis"],
                    arrays["latent_axis"],
                    label="structured artifact latent_axis",
                ),
                latent_valid=arrays["latent_valid"],
                log_ai_highres_truth=arrays["latent_log_ai_highres_truth"],
                state_id=arrays["latent_state_id"],
                object_id=arrays["latent_object_id"],
                object_xi=arrays["latent_object_xi"],
                zone_id=arrays["latent_zone_id"],
                clipping_mask=arrays["latent_clipping_mask"],
            ),
            zone=ZoneTruth(
                zone_id=str(zone_data["zone_id"]),
                top=float(zone_data["top"]),
                bottom=float(zone_data["bottom"]),
                background_a=float(zone_data["background_a"]),
                background_b=float(zone_data["background_b"]),
                zone_valid=arrays["zone_valid"],
            ),
            segments=tuple(
                SegmentTruth(
                    zone_id=str(row["zone_id"]),
                    object_id=row["object_id"],
                    state=str(row["state"]),
                    state_id=int(row["state_id"]),
                    top=float(row["top"]),
                    bottom=float(row["bottom"]),
                    duration_fraction=float(row["duration_fraction"]),
                    duration_samples=int(row["duration_samples"]),
                    c0_raw=[row["c0_raw"]],
                    c1_raw=[row["c1_raw"]],
                    c2_raw=[row["c2_raw"]],
                    c0_projected=[row["c0_projected"]],
                    c1_projected=[row["c1_projected"]],
                    c2_projected=[row["c2_projected"]],
                    c0_effective=[row["c0_effective"]],
                    c1_effective=[row["c1_effective"]],
                    c2_effective=[row["c2_effective"]],
                    segment_supervision_valid=bool(row["segment_supervision_valid"]),
                )
                for row in segments
            ),
            identity=_assert_identity(manifest.get("identity")),
            forward_context=dict(forward_context),
        )
        return sample


def assert_structured_sample_equal(
    expected: StructuredSample,
    actual: StructuredSample,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Raise AssertionError unless two structured samples share all semantics."""
    if not isinstance(expected, StructuredSample) or not isinstance(actual, StructuredSample):
        raise TypeError("assert_structured_sample_equal requires StructuredSample values.")
    scalar_fields = (
        "realization_id",
        "lateral_index",
        "lateral_m",
        "inline",
        "xline",
        "xline_step",
    )
    for name in scalar_fields:
        left, right = getattr(expected, name), getattr(actual, name)
        if isinstance(left, str) or isinstance(left, int):
            if left != right:
                raise AssertionError(f"structured sample field differs: {name}")
        elif not np.isclose(float(left), float(right), rtol=rtol, atol=atol):
            raise AssertionError(f"structured sample field differs: {name}")
    if expected.identity != actual.identity:
        raise AssertionError("structured sample identity differs.")
    if expected.forward_context != actual.forward_context:
        raise AssertionError("structured forward context differs.")
    if expected.observed.lfm_source_identity != actual.observed.lfm_source_identity:
        raise AssertionError("structured LFM source identity differs.")
    for name in ("zone_id",):
        if getattr(expected.zone, name) != getattr(actual.zone, name):
            raise AssertionError(f"structured zone field differs: {name}")
    for name in ("top", "bottom", "background_a", "background_b"):
        if not np.isclose(
            float(getattr(expected.zone, name)),
            float(getattr(actual.zone, name)),
            rtol=rtol,
            atol=atol,
        ):
            raise AssertionError(f"structured zone field differs: {name}")
    if not np.array_equal(expected.zone.zone_valid, actual.zone.zone_valid):
        raise AssertionError("structured zone field differs: zone_valid")
    for left_trace, right_trace, names in (
        (
            expected.observed,
            actual.observed,
            ("seismic", "model_consistent_seismic", "lfm", "observed_valid"),
        ),
        (
            expected.latent,
            actual.latent,
            ("latent_valid", "log_ai_highres_truth"),
        ),
    ):
        for name in names:
            left, right = getattr(left_trace, name), getattr(right_trace, name)
            if left.dtype == np.dtype(bool):
                if not np.array_equal(left, right):
                    raise AssertionError(f"structured trace field differs: {name}")
            elif not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
                raise AssertionError(f"structured trace field differs: {name}")
        if not np.array_equal(
            left_trace.sample_axis.coordinates,
            right_trace.sample_axis.coordinates,
        ):
            raise AssertionError("structured trace axis coordinates differ.")
    for name in (
        "state_id",
        "object_id",
        "object_xi",
        "zone_id",
        "clipping_mask",
    ):
        left = getattr(expected.latent, name)
        right = getattr(actual.latent, name)
        if left.dtype == np.dtype(bool) or np.issubdtype(left.dtype, np.integer):
            if not np.array_equal(left, right):
                raise AssertionError(f"structured latent field differs: {name}")
        elif not np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True):
            raise AssertionError(f"structured latent field differs: {name}")
    if len(expected.segments) != len(actual.segments):
        raise AssertionError("structured sample segment count differs.")
    for index, (left, right) in enumerate(zip(expected.segments, actual.segments, strict=True)):
        for name in (
            "zone_id",
            "object_id",
            "state",
            "state_id",
            "duration_samples",
            "segment_supervision_valid",
        ):
            if getattr(left, name) != getattr(right, name):
                raise AssertionError(f"segment {index} field differs: {name}")
        for name in (
            "top",
            "bottom",
            "duration_fraction",
            "c0_raw",
            "c1_raw",
            "c2_raw",
            "c0_projected",
            "c1_projected",
            "c2_projected",
            "c0_effective",
            "c1_effective",
            "c2_effective",
        ):
            if not np.allclose(
                getattr(left, name),
                getattr(right, name),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                raise AssertionError(f"segment {index} field differs: {name}")


__all__ = [
    "ARTIFACT_TYPE",
    "ARTIFACT_VERSION",
    "LatentTrace",
    "ObservedTrace",
    "RawSegmentParameters",
    "SegmentTruth",
    "StructuredSample",
    "StructuredTruthAdapter",
    "StructuredTruthArtifactReader",
    "StructuredTruthArtifactWriter",
    "ZoneTruth",
    "assert_structured_sample_equal",
]
