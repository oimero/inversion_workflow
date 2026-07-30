"""Public scientific contracts for Structured GINN V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cup.synthetic.core.records import SampleAxis


class InputContractError(ValueError):
    """Raised when axes, geometry, masks, or values violate the input contract."""


class DomainMismatchError(InputContractError):
    """Raised when a checkpoint and an observation use different vertical domains."""


class NumericalFailure(FloatingPointError):
    """Raised when a numerical operation produces non-finite scientific output."""


def _array(value: object, *, dtype: object, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != ndim:
        raise InputContractError(f"{name} must have {ndim} dimensions.")
    return result


def _same_axis(left: SampleAxis, right: SampleAxis) -> bool:
    return (
        left.sample_domain == right.sample_domain
        and left.unit == right.unit
        and left.depth_basis == right.depth_basis
        and left.coordinates.shape == right.coordinates.shape
        and np.allclose(left.coordinates, right.coordinates, rtol=0.0, atol=1.0e-10)
    )


@dataclass(frozen=True)
class ObservationTile:
    """One 1D lateral tile with explicit vertical and physical geometry."""

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
        if not isinstance(self.model_axis, SampleAxis) or not isinstance(
            self.highres_axis, SampleAxis
        ):
            raise TypeError("model_axis and highres_axis must be SampleAxis objects.")
        if (
            self.model_axis.sample_domain != self.highres_axis.sample_domain
            or self.model_axis.unit != self.highres_axis.unit
            or self.model_axis.depth_basis != self.highres_axis.depth_basis
        ):
            raise DomainMismatchError("model and high-resolution axes must share domain.")
        ratio = self.model_axis.sample_interval / self.highres_axis.sample_interval
        factor = int(round(ratio))
        if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1.0e-10):
            raise InputContractError("model and high-resolution axes must be integer nested.")
        nested = self.highres_axis.coordinates[::factor]
        if nested.shape != self.model_axis.coordinates.shape or not np.allclose(
            nested,
            self.model_axis.coordinates,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise InputContractError("model axis must be nested in high-resolution axis.")

        seismic = _array(self.seismic, dtype=np.float64, ndim=2, name="seismic")
        lfm = _array(self.lfm, dtype=np.float64, ndim=2, name="lfm")
        valid = _array(
            self.observed_valid,
            dtype=bool,
            ndim=2,
            name="observed_valid",
        )
        expected = (seismic.shape[0], self.model_axis.coordinates.size)
        if seismic.shape != expected or lfm.shape != expected or valid.shape != expected:
            raise InputContractError(
                "seismic, lfm, and observed_valid must be [lateral, model_sample]."
            )
        if np.any(valid & (~np.isfinite(seismic) | ~np.isfinite(lfm))):
            raise InputContractError("valid observation samples must be finite.")

        lateral = _array(self.lateral_m, dtype=np.float64, ndim=1, name="lateral_m")
        lateral_valid = _array(
            self.lateral_valid,
            dtype=bool,
            ndim=1,
            name="lateral_valid",
        )
        top = _array(self.zone_top, dtype=np.float64, ndim=1, name="zone_top")
        bottom = _array(self.zone_bottom, dtype=np.float64, ndim=1, name="zone_bottom")
        if any(
            item.shape != (seismic.shape[0],)
            for item in (lateral, lateral_valid, top, bottom)
        ):
            raise InputContractError("lateral geometry must match the tile width.")
        if np.any(~np.isfinite(lateral)) or (
            lateral.size > 1 and np.any(np.diff(lateral) <= 0.0)
        ):
            raise InputContractError("lateral_m must be finite and strictly increasing.")
        if np.any(lateral_valid & (~np.isfinite(top) | ~np.isfinite(bottom))):
            raise InputContractError("valid zone coordinates must be finite.")
        if np.any(lateral_valid & (bottom <= top)):
            raise InputContractError("valid zone bottoms must be greater than tops.")

        coordinates: list[np.ndarray | None] = [self.x_m, self.y_m]
        parsed_coordinates: list[np.ndarray | None] = []
        for name, value in zip(("x_m", "y_m"), coordinates):
            if value is None:
                parsed_coordinates.append(None)
                continue
            parsed = _array(value, dtype=np.float64, ndim=1, name=name)
            if parsed.shape != lateral.shape or np.any(~np.isfinite(parsed)):
                raise InputContractError(f"{name} must be finite and match lateral_m.")
            parsed_coordinates.append(parsed)
        if (parsed_coordinates[0] is None) != (parsed_coordinates[1] is None):
            raise InputContractError("x_m and y_m must either both be present or absent.")

        object.__setattr__(self, "seismic", seismic)
        object.__setattr__(self, "lfm", lfm)
        object.__setattr__(self, "observed_valid", valid)
        object.__setattr__(self, "lateral_m", lateral)
        object.__setattr__(self, "lateral_valid", lateral_valid)
        object.__setattr__(self, "zone_top", top)
        object.__setattr__(self, "zone_bottom", bottom)
        object.__setattr__(self, "x_m", parsed_coordinates[0])
        object.__setattr__(self, "y_m", parsed_coordinates[1])

    @property
    def sample_domain(self) -> str:
        return self.model_axis.sample_domain

    @property
    def width(self) -> int:
        return int(self.seismic.shape[0])


@dataclass(frozen=True)
class BandlimitedEvidence:
    """Model-grid evidence that is allowed to condition the structured generator."""

    model_axis: SampleAxis
    highres_axis: SampleAxis
    background_lfm_linear: np.ndarray
    bandlimited_increment_mean: np.ndarray
    bandlimited_increment_scale: np.ndarray
    state_occupancy: np.ndarray
    interface_activity: np.ndarray
    local_tuning_scale: np.ndarray
    support: np.ndarray
    lateral_m: np.ndarray
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_axis, SampleAxis) or not isinstance(
            self.highres_axis, SampleAxis
        ):
            raise TypeError("model_axis and highres_axis must be SampleAxis objects.")
        if (
            self.model_axis.sample_domain != self.highres_axis.sample_domain
            or self.model_axis.unit != self.highres_axis.unit
            or self.model_axis.depth_basis != self.highres_axis.depth_basis
        ):
            raise DomainMismatchError("evidence axes must share domain, unit, and basis.")
        ratio = self.model_axis.sample_interval / self.highres_axis.sample_interval
        factor = int(round(ratio))
        if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1.0e-10):
            raise InputContractError("evidence axes must be integer nested.")
        nested = self.highres_axis.coordinates[::factor]
        if nested.shape != self.model_axis.coordinates.shape or not np.allclose(
            nested,
            self.model_axis.coordinates,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise InputContractError("evidence model axis must nest in highres_axis.")
        mean = _array(
            self.bandlimited_increment_mean,
            dtype=np.float64,
            ndim=2,
            name="bandlimited_increment_mean",
        )
        shape = mean.shape
        if shape[1] != self.model_axis.coordinates.size:
            raise InputContractError("evidence sample dimension must match model_axis.")
        background = _array(
            self.background_lfm_linear,
            dtype=np.float64,
            ndim=2,
            name="background_lfm_linear",
        )
        scale = _array(
            self.bandlimited_increment_scale,
            dtype=np.float64,
            ndim=2,
            name="bandlimited_increment_scale",
        )
        activity = _array(
            self.interface_activity,
            dtype=np.float64,
            ndim=2,
            name="interface_activity",
        )
        tuning = _array(
            self.local_tuning_scale,
            dtype=np.float64,
            ndim=2,
            name="local_tuning_scale",
        )
        support = _array(self.support, dtype=bool, ndim=2, name="support")
        occupancy = _array(
            self.state_occupancy,
            dtype=np.float64,
            ndim=3,
            name="state_occupancy",
        )
        if any(item.shape != shape for item in (background, scale, activity, tuning, support)):
            raise InputContractError("all scalar evidence fields must share one shape.")
        if occupancy.shape != shape + (3,):
            raise InputContractError("state_occupancy must have shape [lateral, sample, 3].")
        if np.any(support & ~np.isfinite(mean)) or np.any(
            support & (~np.isfinite(scale) | (scale <= 0.0))
        ):
            raise NumericalFailure("supported evidence mean/scale must be finite and positive.")
        if np.any(support & (~np.isfinite(activity) | (activity < 0.0) | (activity > 1.0))):
            raise InputContractError("interface_activity must be in [0, 1].")
        if np.any(support & (~np.isfinite(tuning) | (tuning <= 0.0))):
            raise InputContractError("local_tuning_scale must be finite and positive.")
        sums = np.sum(occupancy, axis=-1)
        if np.any(support & ~np.all(np.isfinite(occupancy), axis=-1)) or np.any(
            support & ~np.isclose(sums, 1.0, rtol=1.0e-5, atol=1.0e-6)
        ):
            raise InputContractError("supported state_occupancy must be finite and sum to one.")
        if np.any(occupancy[support] < 0.0):
            raise InputContractError("state_occupancy cannot be negative.")
        lateral = _array(self.lateral_m, dtype=np.float64, ndim=1, name="lateral_m")
        if (
            lateral.shape != (shape[0],)
            or np.any(~np.isfinite(lateral))
            or (lateral.size > 1 and np.any(np.diff(lateral) <= 0.0))
        ):
            raise InputContractError(
                "lateral_m must be finite, increasing, and match evidence width."
            )
        parsed_coordinates: list[np.ndarray | None] = []
        for name, value in (("x_m", self.x_m), ("y_m", self.y_m)):
            if value is None:
                parsed_coordinates.append(None)
                continue
            parsed = _array(value, dtype=np.float64, ndim=1, name=name)
            if parsed.shape != lateral.shape or np.any(~np.isfinite(parsed)):
                raise InputContractError(f"{name} must be finite and match lateral_m.")
            parsed_coordinates.append(parsed)
        if (parsed_coordinates[0] is None) != (parsed_coordinates[1] is None):
            raise InputContractError("x_m and y_m must both be present or absent.")

        object.__setattr__(self, "background_lfm_linear", background)
        object.__setattr__(self, "bandlimited_increment_mean", mean)
        object.__setattr__(self, "bandlimited_increment_scale", scale)
        object.__setattr__(self, "state_occupancy", occupancy)
        object.__setattr__(self, "interface_activity", activity)
        object.__setattr__(self, "local_tuning_scale", tuning)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "lateral_m", lateral)
        object.__setattr__(self, "x_m", parsed_coordinates[0])
        object.__setattr__(self, "y_m", parsed_coordinates[1])


@dataclass(frozen=True)
class Segment:
    trace_index: int
    state_id: int
    start_index: int
    stop_index: int
    c0: float
    c1: float
    c2: float
    log_score: float = 0.0


@dataclass(frozen=True)
class StructuredRealization:
    identity: int
    log_ai_highres: np.ndarray
    state_highres: np.ndarray
    projected_log_ai: np.ndarray
    segments: tuple[Segment, ...]
    conditional_log_score: float


@dataclass(frozen=True)
class EnsembleSummary:
    log_ai_highres_mean: np.ndarray
    log_ai_highres_std: np.ndarray
    projected_log_ai_mean: np.ndarray
    projected_log_ai_std: np.ndarray
    state_occupancy_highres: np.ndarray
    interface_density_highres: np.ndarray
    segment_count_mean: np.ndarray
    segment_count_std: np.ndarray
    segment_duration_fraction_mean: np.ndarray
    segment_duration_fraction_std: np.ndarray


@dataclass(frozen=True)
class GenerationPolicy:
    realization_count: int = 16
    random_identity: int = 0
    retain_realizations: bool = True
    lateral_correlation_m: float = 900.0

    def __post_init__(self) -> None:
        if isinstance(self.realization_count, bool) or self.realization_count <= 0:
            raise ValueError("realization_count must be positive.")
        if not np.isfinite(self.lateral_correlation_m) or self.lateral_correlation_m <= 0:
            raise ValueError("lateral_correlation_m must be finite and positive.")


@dataclass(frozen=True)
class StructuredPrediction:
    evidence: BandlimitedEvidence
    representative: StructuredRealization
    summary: EnsembleSummary
    realization_identities: tuple[int, ...]
    realizations: tuple[StructuredRealization, ...] | None
    diagnostics: Mapping[str, float]


@dataclass(frozen=True)
class VolumeInferenceResult:
    """Two-pass volume result using one member identity for every tile."""

    tiles: Mapping[str, StructuredPrediction]
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
    "EnsembleSummary",
    "GenerationPolicy",
    "InputContractError",
    "NumericalFailure",
    "ObservationTile",
    "Segment",
    "StructuredPrediction",
    "StructuredRealization",
    "VolumeInferenceResult",
]
