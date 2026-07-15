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
class ProjectionPolicy:
    continuous_method: str
    edge_mode: str
    support_mode: str
    antialias_taps_per_factor: int
    cutoff_output_nyquist_fraction: float
    kaiser_beta: float
    categorical_window_mode: str = "centered_factor_plus_one"
    geometric_valid_mode: str = "categorical_window_any"

    def __post_init__(self) -> None:
        supported = {
            ("scipy_resample_poly", "line", "full"),
            ("valid_fir_decimate", "finite_support", "valid_fir"),
        }
        if (self.continuous_method, self.edge_mode, self.support_mode) not in supported:
            raise ValueError("unsupported projection policy combination.")
        if self.antialias_taps_per_factor < 1:
            raise ValueError("antialias_taps_per_factor must be positive.")
        if not 0.0 < self.cutoff_output_nyquist_fraction < 1.0:
            raise ValueError("projection cutoff fraction must lie in (0, 1).")
        if self.categorical_window_mode != "centered_factor_plus_one":
            raise ValueError("unsupported categorical window mode.")
        if self.geometric_valid_mode not in {
            "categorical_window_any",
            "point_sample_highres",
        }:
            raise ValueError("unsupported geometric valid mode.")


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
    projection_policy: ProjectionPolicy
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
class BenchmarkResiduals:
    residual_vs_lfm_ideal: np.ndarray
    residual_vs_lfm_controlled_degraded: np.ndarray


@dataclass(frozen=True)
class BenchmarkSample:
    truth: SyntheticTruth
    projected: ProjectedTruth
    forward: ForwardResult
    canonical_background_log_ai: np.ndarray
    target_increment_log_ai: np.ndarray
    input_lfm_canonical_log_ai: np.ndarray
    input_lfm_controlled_degraded_log_ai: np.ndarray
    residuals: BenchmarkResiduals
    valid_mask: np.ndarray
    qc: Mapping[str, Any] = field(default_factory=dict)
    domain_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "qc", MappingProxyType(dict(self.qc)))
        object.__setattr__(self, "domain_metadata", MappingProxyType(dict(self.domain_metadata)))


@dataclass(frozen=True)
class BenchmarkVariant:
    owner_realization_id: str
    variant_id: str
    sample_kind: str
    seismic_observed: np.ndarray
    seismic_model_consistent: np.ndarray
    positive_gain: np.ndarray
    additive_noise: np.ndarray
    metadata: Mapping[str, object]
    residuals: BenchmarkResiduals | None = None
    qc: Mapping[str, object] = field(default_factory=dict)
    sample_domain: str = "time"

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "qc", MappingProxyType(dict(self.qc)))


__all__ = [
    "BenchmarkResiduals",
    "BenchmarkSample",
    "BenchmarkVariant",
    "DepthForwardExtras",
    "DomainPreparation",
    "ForwardResult",
    "ForwardSupport",
    "ProjectedTruth",
    "ProjectionPolicy",
    "SampleAxis",
    "TimeForwardExtras",
]
