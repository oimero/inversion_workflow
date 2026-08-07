"""Small public contracts for the Structured GINN V2 generator.

The public seam carries axes, physical geometry, observable evidence, and
complete section results.  Event decoding, semi-Markov scoring, and sampling
remain implementation details of the generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cup.synthetic.core.records import SampleAxis


class InputContractError(ValueError):
    """Raised when axes, geometry, masks, or values violate the input contract."""


class DomainMismatchError(InputContractError):
    """Raised when two vertical objects use different sample domains."""


class NumericalFailure(FloatingPointError):
    """Raised when a scientific result is non-finite on declared support."""


def _array(value: object, *, dtype: object, ndim: int, name: str) -> np.ndarray:
    parsed = np.asarray(value, dtype=dtype)
    if parsed.ndim != ndim:
        raise InputContractError(f"{name} must have {ndim} dimensions.")
    return parsed


def _validate_nested_axes(model: SampleAxis, highres: SampleAxis) -> None:
    if not isinstance(model, SampleAxis) or not isinstance(highres, SampleAxis):
        raise TypeError("model and high-resolution axes must be SampleAxis objects.")
    if (
        model.sample_domain != highres.sample_domain
        or model.unit != highres.unit
        or model.depth_basis != highres.depth_basis
    ):
        raise DomainMismatchError("model and high-resolution axes must share domain.")
    ratio = model.sample_interval / highres.sample_interval
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1.0e-10):
        raise InputContractError("model and high-resolution axes must be integer nested.")
    nested = highres.coordinates[::factor]
    if nested.shape != model.coordinates.shape or not np.allclose(
        nested, model.coordinates, rtol=0.0, atol=1.0e-9
    ):
        raise InputContractError("model axis must be nested in high-resolution axis.")


def _validate_xy(
    lateral: np.ndarray,
    x_m: np.ndarray | None,
    y_m: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    parsed: list[np.ndarray | None] = []
    for name, value in (("x_m", x_m), ("y_m", y_m)):
        if value is None:
            parsed.append(None)
            continue
        item = _array(value, dtype=np.float64, ndim=1, name=name)
        if item.shape != lateral.shape or np.any(~np.isfinite(item)):
            raise InputContractError(f"{name} must be finite and match lateral_m.")
        parsed.append(item)
    if (parsed[0] is None) != (parsed[1] is None):
        raise InputContractError("x_m and y_m must either both be present or absent.")
    return parsed[0], parsed[1]


@dataclass(frozen=True)
class ObservationTile:
    """A lateral tile with explicit vertical and physical geometry."""

    model_axis: SampleAxis
    highres_axis: SampleAxis
    seismic: np.ndarray
    lfm: np.ndarray
    observed_valid: np.ndarray
    lateral_m: np.ndarray
    lateral_valid: np.ndarray
    zone_top: np.ndarray
    zone_bottom: np.ndarray
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None
    identity: str = ""

    def __post_init__(self) -> None:
        _validate_nested_axes(self.model_axis, self.highres_axis)
        seismic = _array(self.seismic, dtype=np.float64, ndim=2, name="seismic")
        lfm = _array(self.lfm, dtype=np.float64, ndim=2, name="lfm")
        observed_valid = _array(
            self.observed_valid, dtype=bool, ndim=2, name="observed_valid"
        )
        expected = (seismic.shape[0], self.model_axis.coordinates.size)
        if lfm.shape != expected or observed_valid.shape != expected or seismic.shape != expected:
            raise InputContractError(
                "seismic, lfm, and observed_valid must be [lateral, model_sample]."
            )
        if np.any(observed_valid & (~np.isfinite(seismic) | ~np.isfinite(lfm))):
            raise InputContractError("valid observation samples must be finite.")

        lateral = _array(self.lateral_m, dtype=np.float64, ndim=1, name="lateral_m")
        lateral_valid = _array(
            self.lateral_valid, dtype=bool, ndim=1, name="lateral_valid"
        )
        top = _array(self.zone_top, dtype=np.float64, ndim=1, name="zone_top")
        bottom = _array(self.zone_bottom, dtype=np.float64, ndim=1, name="zone_bottom")
        if any(item.shape != (seismic.shape[0],) for item in (lateral, lateral_valid, top, bottom)):
            raise InputContractError("lateral geometry must match tile width.")
        if np.any(~np.isfinite(lateral)) or (
            lateral.size > 1 and np.any(np.diff(lateral) <= 0.0)
        ):
            raise InputContractError("lateral_m must be finite and strictly increasing.")
        if np.any(lateral_valid & (~np.isfinite(top) | ~np.isfinite(bottom))):
            raise InputContractError("valid zone coordinates must be finite.")
        if np.any(lateral_valid & (bottom <= top)):
            raise InputContractError("valid zone bottoms must be greater than tops.")
        x_m, y_m = _validate_xy(lateral, self.x_m, self.y_m)

        object.__setattr__(self, "seismic", seismic)
        object.__setattr__(self, "lfm", lfm)
        object.__setattr__(self, "observed_valid", observed_valid)
        object.__setattr__(self, "lateral_m", lateral)
        object.__setattr__(self, "lateral_valid", lateral_valid)
        object.__setattr__(self, "zone_top", top)
        object.__setattr__(self, "zone_bottom", bottom)
        object.__setattr__(self, "x_m", x_m)
        object.__setattr__(self, "y_m", y_m)

    @property
    def sample_domain(self) -> str:
        return self.model_axis.sample_domain

    @property
    def width(self) -> int:
        return int(self.seismic.shape[0])


@dataclass(frozen=True)
class BandlimitedEvidence:
    """Band-limited evidence exposed to the event generator."""

    model_axis: SampleAxis
    highres_axis: SampleAxis
    background_lfm_linear: np.ndarray
    background_lfm_linear_highres: np.ndarray
    projected_log_ai_increment_mean: np.ndarray
    projected_log_ai_increment_scale: np.ndarray
    signed_reflectivity_mean: np.ndarray
    signed_reflectivity_scale: np.ndarray
    state_fraction: np.ndarray
    local_tuning_scale: np.ndarray
    support: np.ndarray
    highres_support: np.ndarray
    lateral_m: np.ndarray
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None
    identity: str = ""

    @property
    def sample_domain(self) -> str:
        """Return the vertical domain carried by the model axis."""

        return self.model_axis.sample_domain

    def __post_init__(self) -> None:
        _validate_nested_axes(self.model_axis, self.highres_axis)
        mean = _array(
            self.projected_log_ai_increment_mean,
            dtype=np.float64,
            ndim=2,
            name="projected_log_ai_increment_mean",
        )
        shape = mean.shape
        fields = {
            "background_lfm_linear": _array(
                self.background_lfm_linear, dtype=np.float64, ndim=2, name="background_lfm_linear"
            ),
            "projected_log_ai_increment_scale": _array(
                self.projected_log_ai_increment_scale, dtype=np.float64, ndim=2, name="projected_log_ai_increment_scale"
            ),
            "signed_reflectivity_mean": _array(
                self.signed_reflectivity_mean, dtype=np.float64, ndim=2, name="signed_reflectivity_mean"
            ),
            "signed_reflectivity_scale": _array(
                self.signed_reflectivity_scale, dtype=np.float64, ndim=2, name="signed_reflectivity_scale"
            ),
            "local_tuning_scale": _array(
                self.local_tuning_scale, dtype=np.float64, ndim=2, name="local_tuning_scale"
            ),
            "support": _array(self.support, dtype=bool, ndim=2, name="support"),
        }
        if any(value.shape != shape for value in fields.values()):
            raise InputContractError("model-grid evidence fields must share one shape.")
        state_fraction = _array(
            self.state_fraction, dtype=np.float64, ndim=3, name="state_fraction"
        )
        if state_fraction.shape != shape + (3,):
            raise InputContractError("state_fraction must be [lateral, sample, 3].")
        highres_shape = (shape[0], self.highres_axis.coordinates.size)
        background_highres = _array(
            self.background_lfm_linear_highres,
            dtype=np.float64,
            ndim=2,
            name="background_lfm_linear_highres",
        )
        highres_support = _array(
            self.highres_support, dtype=bool, ndim=2, name="highres_support"
        )
        if background_highres.shape != highres_shape or highres_support.shape != highres_shape:
            raise InputContractError("high-resolution fields must match [lateral, sample].")
        support = fields["support"]
        if np.any(support & ~np.isfinite(mean)):
            raise NumericalFailure("supported evidence mean must be finite.")
        for name in (
            "projected_log_ai_increment_scale",
            "signed_reflectivity_scale",
            "local_tuning_scale",
        ):
            value = fields[name]
            if np.any(support & (~np.isfinite(value) | (value <= 0.0))):
                raise NumericalFailure(f"supported {name} must be finite and positive.")
        if np.any(highres_support & ~np.isfinite(background_highres)):
            raise NumericalFailure("supported high-resolution anchor must be finite.")
        state_sum = np.sum(state_fraction, axis=-1)
        if np.any(
            support
            & (
                np.any(~np.isfinite(state_fraction), axis=-1)
                | np.any(state_fraction < 0.0, axis=-1)
                | ~np.isclose(state_sum, 1.0, rtol=0.0, atol=1.0e-5)
            )
        ):
            raise InputContractError(
                "supported state_fraction must be finite probabilities summing to one."
            )
        lateral = _array(self.lateral_m, dtype=np.float64, ndim=1, name="lateral_m")
        if lateral.shape != (shape[0],) or np.any(~np.isfinite(lateral)) or (
            lateral.size > 1 and np.any(np.diff(lateral) <= 0.0)
        ):
            raise InputContractError("evidence lateral_m must be finite and increasing.")
        x_m, y_m = _validate_xy(lateral, self.x_m, self.y_m)
        identity = str(self.identity).strip()
        if not identity:
            raise InputContractError("band-limited evidence identity must be non-empty.")

        object.__setattr__(self, "background_lfm_linear", fields["background_lfm_linear"])
        object.__setattr__(self, "background_lfm_linear_highres", background_highres)
        object.__setattr__(self, "projected_log_ai_increment_mean", mean)
        for name in (
            "projected_log_ai_increment_scale",
            "signed_reflectivity_mean",
            "signed_reflectivity_scale",
            "local_tuning_scale",
            "support",
        ):
            object.__setattr__(self, name, fields[name])
        object.__setattr__(self, "state_fraction", state_fraction)
        object.__setattr__(self, "highres_support", highres_support)
        object.__setattr__(self, "lateral_m", lateral)
        object.__setattr__(self, "x_m", x_m)
        object.__setattr__(self, "y_m", y_m)
        object.__setattr__(self, "identity", identity)


@dataclass(frozen=True)
class EventTrack:
    """One ordered event and its lateral fields."""

    event_id: str
    state_id: int
    presence: np.ndarray
    duration_fraction: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        if self.state_id not in {0, 1, 2}:
            raise InputContractError("event state_id must be 0, 1, or 2.")
        presence = _array(self.presence, dtype=bool, ndim=1, name="presence")
        duration = _array(
            self.duration_fraction, dtype=np.float64, ndim=1, name="duration_fraction"
        )
        coefficients = _array(
            self.coefficients, dtype=np.float64, ndim=2, name="coefficients"
        )
        if duration.shape != presence.shape or coefficients.shape != (presence.size, 3):
            raise InputContractError("event lateral fields have inconsistent shapes.")
        if np.any(~np.isfinite(duration)) or np.any(duration < 0.0):
            raise NumericalFailure("event duration_fraction must be finite and non-negative.")
        if np.any(presence & (duration <= 0.0)):
            raise InputContractError("present events must have positive duration.")
        if np.any(~np.isfinite(coefficients[presence])):
            raise NumericalFailure("present event coefficients must be finite.")
        identity = str(self.event_id).strip()
        if not identity:
            raise InputContractError("event_id cannot be empty.")
        object.__setattr__(self, "event_id", identity)
        object.__setattr__(self, "presence", presence)
        object.__setattr__(self, "duration_fraction", duration)
        object.__setattr__(self, "coefficients", coefficients)


@dataclass(frozen=True)
class EventTrackTruth:
    """One producer-owned high-resolution event across a lateral zone tile."""

    zone_id: str
    event_id: int
    state_id: int
    presence: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    duration_fraction: np.ndarray
    raw_coefficients: np.ndarray
    projected_coefficients: np.ndarray
    effective_coefficients: np.ndarray
    segment_supervision_valid: np.ndarray

    def __post_init__(self) -> None:
        zone_id = str(self.zone_id).strip()
        if not zone_id or self.event_id < 0 or self.state_id not in {0, 1, 2}:
            raise InputContractError("event truth identity or state is invalid.")
        presence = _array(self.presence, dtype=bool, ndim=1, name="presence")
        width = presence.size
        one_dimensional = {
            "top": _array(self.top, dtype=np.float64, ndim=1, name="top"),
            "bottom": _array(self.bottom, dtype=np.float64, ndim=1, name="bottom"),
            "duration_fraction": _array(
                self.duration_fraction,
                dtype=np.float64,
                ndim=1,
                name="duration_fraction",
            ),
            "segment_supervision_valid": _array(
                self.segment_supervision_valid,
                dtype=bool,
                ndim=1,
                name="segment_supervision_valid",
            ),
        }
        if any(value.shape != (width,) for value in one_dimensional.values()):
            raise InputContractError("event truth lateral fields have inconsistent shapes.")
        coefficient_fields = {
            "raw_coefficients": _array(
                self.raw_coefficients, dtype=np.float64, ndim=2, name="raw_coefficients"
            ),
            "projected_coefficients": _array(
                self.projected_coefficients,
                dtype=np.float64,
                ndim=2,
                name="projected_coefficients",
            ),
            "effective_coefficients": _array(
                self.effective_coefficients,
                dtype=np.float64,
                ndim=2,
                name="effective_coefficients",
            ),
        }
        if any(value.shape != (width, 3) for value in coefficient_fields.values()):
            raise InputContractError("event truth coefficients must be [lateral, 3].")
        top = one_dimensional["top"]
        bottom = one_dimensional["bottom"]
        duration = one_dimensional["duration_fraction"]
        if np.any(presence & (~np.isfinite(top) | ~np.isfinite(bottom) | (bottom <= top))):
            raise InputContractError("present event truth extents must be finite and positive.")
        if np.any(presence & (~np.isfinite(duration) | (duration <= 0.0))):
            raise InputContractError("present event truth durations must be finite and positive.")
        if np.any(~presence & (np.isfinite(top) | np.isfinite(bottom) | (duration != 0.0))):
            raise InputContractError("absent event truth must not carry extents or duration.")
        for name, values in coefficient_fields.items():
            if np.any(~np.isfinite(values[presence])) or np.any(np.isfinite(values[~presence])):
                raise InputContractError(
                    f"{name} must be finite exactly where the event is present."
                )
        object.__setattr__(self, "zone_id", zone_id)
        object.__setattr__(self, "presence", presence)
        for name, value in {**one_dimensional, **coefficient_fields}.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class Segment:
    """Internal rasterized view of one event on one trace."""

    trace_index: int
    state_id: int
    start_index: int
    stop_index: int
    c0: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        if self.trace_index < 0 or self.state_id not in {0, 1, 2}:
            raise InputContractError("segment trace and state identifiers are invalid.")
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise InputContractError("segment extent must be non-empty and ordered.")
        coefficients = np.asarray((self.c0, self.c1, self.c2), dtype=np.float64)
        if np.any(~np.isfinite(coefficients)):
            raise NumericalFailure("segment coefficients must be finite.")


@dataclass(frozen=True)
class GenerationPolicy:
    realization_count: int = 16
    random_identity: int = 0
    retain_realizations: bool = True
    lateral_correlation_m: float = 900.0
    event_density_multiplier: float = 1.0
    structure_sampling_temperature: float = 0.65
    profile_sampling_temperature: float = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.realization_count, bool)
            or not isinstance(self.realization_count, (int, np.integer))
            or self.realization_count <= 0
        ):
            raise InputContractError("realization_count must be a positive integer.")
        if isinstance(self.random_identity, bool) or not isinstance(
            self.random_identity, (int, np.integer)
        ):
            raise InputContractError("random_identity must be an integer.")
        if not isinstance(self.retain_realizations, (bool, np.bool_)):
            raise InputContractError("retain_realizations must be boolean.")
        if not np.isfinite(self.lateral_correlation_m) or self.lateral_correlation_m <= 0.0:
            raise InputContractError("lateral_correlation_m must be finite and positive.")
        if (
            not np.isfinite(self.event_density_multiplier)
            or self.event_density_multiplier <= 0.0
        ):
            raise InputContractError(
                "event_density_multiplier must be finite and positive."
            )
        for name in (
            "structure_sampling_temperature",
            "profile_sampling_temperature",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise InputContractError(f"{name} must be finite and positive.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GenerationPolicy":
        return cls(**dict(value))


@dataclass(frozen=True)
class EventTrackRealization:
    """One legal ordered EventTrack system rasterized on both vertical axes."""

    zone_id: str
    tracks: tuple[EventTrack, ...]
    log_ai_highres: np.ndarray
    state_highres: np.ndarray
    model_log_ai: np.ndarray
    model_support: np.ndarray
    identity: str

    def __post_init__(self) -> None:
        zone_id = str(self.zone_id).strip()
        identity = str(self.identity).strip()
        if not zone_id or not identity or not self.tracks:
            raise InputContractError("event realization requires zone, identity, and tracks.")
        highres = _array(
            self.log_ai_highres, dtype=np.float64, ndim=2, name="log_ai_highres"
        )
        state = _array(self.state_highres, dtype=np.int8, ndim=2, name="state_highres")
        model = _array(self.model_log_ai, dtype=np.float64, ndim=2, name="model_log_ai")
        support = _array(self.model_support, dtype=bool, ndim=2, name="model_support")
        if state.shape != highres.shape or support.shape != model.shape:
            raise InputContractError("event realization raster fields have inconsistent shapes.")
        if any(track.presence.size != highres.shape[0] for track in self.tracks):
            raise InputContractError("event realization track width differs from its raster.")
        if np.any(np.isfinite(highres) & ~np.isin(state, (0, 1, 2))):
            raise InputContractError("finite high-resolution realization samples require a state.")
        if np.any(support & ~np.isfinite(model)):
            raise NumericalFailure("supported projected realization samples must be finite.")
        object.__setattr__(self, "zone_id", zone_id)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "tracks", tuple(self.tracks))
        object.__setattr__(self, "log_ai_highres", highres)
        object.__setattr__(self, "state_highres", state)
        object.__setattr__(self, "model_log_ai", model)
        object.__setattr__(self, "model_support", support)


@dataclass(frozen=True)
class StructuredEnsemble:
    """Complete section result returned by the generator seam."""

    evidence: BandlimitedEvidence
    representative: EventTrackRealization
    realizations: tuple[EventTrackRealization, ...]
    summary: Mapping[str, Any]
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        realizations = tuple(self.realizations)
        if realizations and self.representative.identity not in {
            item.identity for item in realizations
        }:
            raise InputContractError(
                "representative must be one of the retained complete realizations."
            )
        summary = dict(self.summary)
        diagnostics = {str(name): float(value) for name, value in self.diagnostics.items()}
        if any(not np.isfinite(value) for value in diagnostics.values()):
            raise NumericalFailure("ensemble diagnostics must be finite.")
        object.__setattr__(self, "realizations", realizations)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True)
class VolumeInferenceResult:
    tiles: Mapping[str, StructuredEnsemble]
    representative_member_index: int
    representative_identity: int

    def __post_init__(self) -> None:
        if not self.tiles:
            raise InputContractError("volume inference must contain at least one tile.")
        if self.representative_member_index < 0:
            raise InputContractError("representative_member_index cannot be negative.")
        object.__setattr__(self, "tiles", dict(sorted(self.tiles.items())))


__all__ = [
    "BandlimitedEvidence",
    "DomainMismatchError",
    "EventTrack",
    "EventTrackRealization",
    "EventTrackTruth",
    "GenerationPolicy",
    "InputContractError",
    "NumericalFailure",
    "ObservationTile",
    "Segment",
    "StructuredEnsemble",
    "VolumeInferenceResult",
]
