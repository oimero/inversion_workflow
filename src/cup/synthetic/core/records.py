"""Strict in-memory records shared by Synthoseis scientific and benchmark Modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cup.synthetic.core.truth import SyntheticTruth


@dataclass(frozen=True)
class SampleAxis:
    sample_domain: str
    unit: str
    coordinates: np.ndarray
    sample_interval: float
    positive_direction: str
    depth_basis: str | None = None

    def __post_init__(self) -> None:
        domain = self.sample_domain.strip().lower()
        expected_unit = {"time": "s", "depth": "m"}.get(domain)
        if expected_unit is None or self.unit != expected_unit:
            raise ValueError("sample_domain and unit must be time/s or depth/m.")
        if domain == "depth" and self.depth_basis != "tvdss":
            raise ValueError("depth SampleAxis requires depth_basis='tvdss'.")
        if domain == "time" and self.depth_basis is not None:
            raise ValueError("time SampleAxis must not define depth_basis.")
        coordinates = np.asarray(self.coordinates, dtype=np.float64).reshape(-1)
        if coordinates.size < 2 or np.any(~np.isfinite(coordinates)):
            raise ValueError("sample axis must contain at least two finite coordinates.")
        interval = float(self.sample_interval)
        if interval <= 0.0 or not np.allclose(
            np.diff(coordinates), interval, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("sample axis must be regular with the declared positive interval.")
        if self.positive_direction not in {"down", "increasing_time"}:
            raise ValueError("unsupported sample-axis positive direction.")
        object.__setattr__(self, "sample_domain", domain)
        object.__setattr__(self, "coordinates", coordinates)


@dataclass(frozen=True)
class ProjectedTruth:
    model_axis: SampleAxis
    model_target_log_ai: np.ndarray
    rgt_model: np.ndarray
    state_fraction_model: np.ndarray
    dominant_object_id_model: np.ndarray
    zone_id_model: np.ndarray
    boundary_fraction_model: np.ndarray
    boundary_mask_model: np.ndarray
    geometric_valid_mask_model: np.ndarray
    categorical_valid_mask_model: np.ndarray
    projection_support_highres: np.ndarray
    projection_support_model: np.ndarray


@dataclass(frozen=True)
class DomainPreparation:
    model_axis: SampleAxis
    required_context_extent: float
    forward_configuration: object

    def __post_init__(self) -> None:
        if not np.isfinite(self.required_context_extent) or self.required_context_extent < 0.0:
            raise ValueError("required_context_extent must be finite and non-negative.")


@dataclass(frozen=True)
class ForwardSupport:
    highres: np.ndarray
    model: np.ndarray
    observed: np.ndarray
    physics: np.ndarray


@dataclass(frozen=True)
class TimeForwardExtras:
    reflectivity_highres: np.ndarray
    reflectivity_model: np.ndarray
    forward_valid_mask_highres: np.ndarray
    forward_valid_mask_model: np.ndarray


@dataclass(frozen=True)
class DepthForwardExtras:
    vp_highres_mps: np.ndarray
    vp_model_mps: np.ndarray


@dataclass(frozen=True)
class ForwardResult:
    seismic_observed: np.ndarray
    seismic_model_consistent: np.ndarray
    subgrid_forward_residual: np.ndarray
    support: ForwardSupport
    qc: Mapping[str, Any]
    metadata: Mapping[str, Any]
    extras: TimeForwardExtras | DepthForwardExtras

    def __post_init__(self) -> None:
        object.__setattr__(self, "qc", MappingProxyType(dict(self.qc)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class LfmObservation:
    """A low-frequency observation with its axis and provenance."""

    values: np.ndarray
    sample_axis: SampleAxis
    valid_mask: np.ndarray
    source_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_axis, SampleAxis):
            raise TypeError("LfmObservation.sample_axis must be SampleAxis.")
        values = np.asarray(self.values, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        expected = (values.shape[0], self.sample_axis.coordinates.size)
        if values.ndim != 2 or values.shape != expected or valid.shape != expected:
            raise ValueError(
                "LfmObservation values and valid_mask must be [lateral, sample]."
            )
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError("LfmObservation valid values must be finite.")
        if not isinstance(self.source_identity, Mapping) or not self.source_identity:
            raise ValueError("LfmObservation.source_identity must be explicit.")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "valid_mask", valid.copy())
        object.__setattr__(self, "source_identity", MappingProxyType(dict(self.source_identity)))


@dataclass(frozen=True)
class StructuredSampleRecord:
    """Producer-owned sample record without increment semantics."""

    truth: SyntheticTruth
    projected: ProjectedTruth
    forward: ForwardResult
    lfm: LfmObservation
    valid_mask: np.ndarray
    qc: Mapping[str, Any] = field(default_factory=dict)
    domain_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid_mask, dtype=bool)
        expected = (
            self.truth.lateral_m.size,
            self.projected.model_axis.coordinates.size,
        )
        if valid.shape != expected:
            raise ValueError("StructuredSampleRecord.valid_mask does not match model grid.")
        if self.lfm.sample_axis is not self.projected.model_axis and not np.array_equal(
            self.lfm.sample_axis.coordinates,
            self.projected.model_axis.coordinates,
        ):
            raise ValueError("StructuredSampleRecord LFM axis differs from model axis.")
        if not np.array_equal(self.lfm.valid_mask, valid):
            raise ValueError("StructuredSampleRecord LFM mask differs from valid_mask.")
        for name, values in (
            ("lfm", self.lfm.values),
            ("seismic_observed", self.forward.seismic_observed),
        ):
            if np.asarray(values).shape != expected:
                raise ValueError(f"StructuredSampleRecord {name} shape is invalid.")
        object.__setattr__(self, "valid_mask", valid.copy())
        object.__setattr__(self, "qc", MappingProxyType(dict(self.qc)))
        object.__setattr__(self, "domain_metadata", MappingProxyType(dict(self.domain_metadata)))

    @classmethod
    def from_benchmark_sample(cls, sample: "BenchmarkSample") -> "StructuredSampleRecord":
        """Strip the legacy benchmark decomposition at the new seam."""
        if not isinstance(sample, BenchmarkSample):
            raise TypeError("StructuredSampleRecord requires a BenchmarkSample source.")
        return cls(
            truth=sample.truth,
            projected=sample.projected,
            forward=sample.forward,
            lfm=LfmObservation(
                values=sample.input_lfm_canonical_log_ai,
                sample_axis=sample.projected.model_axis,
                valid_mask=sample.valid_mask,
                source_identity=dict(sample.domain_metadata["lfm_source_identity"]),
            ),
            valid_mask=sample.valid_mask,
            qc=sample.qc,
            domain_metadata=sample.domain_metadata,
        )


@dataclass(frozen=True)
class BenchmarkSample:
    truth: SyntheticTruth
    projected: ProjectedTruth
    forward: ForwardResult
    canonical_background_log_ai: np.ndarray
    target_increment_log_ai: np.ndarray
    input_lfm_canonical_log_ai: np.ndarray
    valid_mask: np.ndarray
    qc: Mapping[str, Any] = field(default_factory=dict)
    domain_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "qc", MappingProxyType(dict(self.qc)))
        object.__setattr__(self, "domain_metadata", MappingProxyType(dict(self.domain_metadata)))

    def to_structured_record(self) -> StructuredSampleRecord:
        return StructuredSampleRecord.from_benchmark_sample(self)


@dataclass(frozen=True)
class BenchmarkView:
    owner_realization_id: str
    view_id: str
    seismic_observed: np.ndarray
    positive_gain: np.ndarray
    additive_noise: np.ndarray
    metadata: Mapping[str, object]
    qc: Mapping[str, object] = field(default_factory=dict)
    sample_domain: str = "time"

    def __post_init__(self) -> None:
        if not self.owner_realization_id.strip() or not self.view_id.strip():
            raise ValueError("Benchmark view owner and view id must be non-empty.")
        domain = self.sample_domain.strip().casefold()
        if domain not in {"time", "depth"}:
            raise ValueError("Benchmark view domain must be time or depth.")
        observed = np.asarray(self.seismic_observed, dtype=np.float64)
        gain = np.asarray(self.positive_gain, dtype=np.float64)
        noise = np.asarray(self.additive_noise, dtype=np.float64)
        if observed.ndim != 2 or gain.shape != observed.shape or noise.shape != observed.shape:
            raise ValueError("Benchmark view arrays must share one 2-D shape.")
        if np.any(~np.isfinite(gain)) or np.any(gain <= 0.0):
            raise ValueError("Benchmark view positive_gain must be finite and positive.")
        object.__setattr__(self, "sample_domain", domain)
        object.__setattr__(self, "seismic_observed", observed)
        object.__setattr__(self, "positive_gain", gain)
        object.__setattr__(self, "additive_noise", noise)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "qc", MappingProxyType(dict(self.qc)))


__all__ = [
    "BenchmarkSample",
    "BenchmarkView",
    "DepthForwardExtras",
    "DomainPreparation",
    "ForwardResult",
    "ForwardSupport",
    "LfmObservation",
    "ProjectedTruth",
    "SampleAxis",
    "StructuredSampleRecord",
    "TimeForwardExtras",
]
