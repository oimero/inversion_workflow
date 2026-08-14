"""Public contracts for deterministic conditional residual texture transfer.

The transfer implementation deliberately keeps the data contracts small.  A
``ResidualTextureLibrary`` owns the native well dictionary, while a
``ResidualTransferResult`` owns one deterministic realization on the target
body grid.  No neural-network or serialization details are part of these
interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from cup.seismic.geometry import SampleAxis, SurveyLineGeometry


_FEATURE_NAMES = (
    "profile",
    "first_derivative",
    "second_derivative",
    "contrast",
    "thickness",
    "position",
)


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}.")
    return result


def _normalise_zone_intervals(value: Mapping[str, Any] | None) -> dict[str, tuple[float, float]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("zone_intervals must be a mapping from zone id to (top, bottom).")
    result: dict[str, tuple[float, float]] = {}
    for raw_name, raw_interval in value.items():
        name = str(raw_name)
        if isinstance(raw_interval, Mapping):
            top = raw_interval.get("top", raw_interval.get("start"))
            bottom = raw_interval.get("bottom", raw_interval.get("end"))
            raw_interval = (top, bottom)
        try:
            top, bottom = raw_interval  # type: ignore[misc]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"zone_intervals[{name!r}] must contain top and bottom coordinates.") from exc
        top = float(top)
        bottom = float(bottom)
        if not np.isfinite(top) or not np.isfinite(bottom) or not top < bottom:
            raise ValueError(f"zone_intervals[{name!r}] must satisfy finite top < bottom.")
        if name in result:
            raise ValueError(f"Duplicate zone id {name!r}.")
        result[name] = (top, bottom)
    return result


def _normalise_zone_intervals_by_well(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, tuple[float, float]]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("zone_intervals_by_well must map well names to zone intervals.")
    result: dict[str, dict[str, tuple[float, float]]] = {}
    seen: set[str] = set()
    for raw_well_name, raw_intervals in value.items():
        well_name = str(raw_well_name)
        token = well_name.casefold()
        if token in seen:
            raise ValueError(f"Duplicate well name {well_name!r} in zone_intervals_by_well.")
        intervals = _normalise_zone_intervals(raw_intervals)
        if not intervals:
            raise ValueError(f"zone_intervals_by_well[{well_name!r}] must not be empty.")
        result[well_name] = intervals
        seen.add(token)
    return result


@dataclass(frozen=True)
class ScaleContract:
    """Physical scale contract shared by dictionary construction and transfer.

    ``zone_intervals_by_well`` carries spatially varying interpreted horizons
    along well paths.  ``zone_intervals`` remains useful for small fixtures in
    which every control truly shares the same vertical interval.
    """

    body_smoothing_fwhm_m: float = 15.0
    window_half_width_m: float | None = None
    window_center_spacing_m: float | None = None
    normalized_profile_samples: int = 33
    min_window_samples: int = 4
    denominator_floor: float = 1.0e-12
    feature_scale_floor: float = 1.0e-12
    zone_intervals: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    zone_intervals_by_well: Mapping[str, Mapping[str, tuple[float, float]]] = field(default_factory=dict)
    sample_axis: SampleAxis | None = None
    domain: str | None = None
    depth_basis: str | None = None

    def __post_init__(self) -> None:
        fwhm = _finite_positive(self.body_smoothing_fwhm_m, name="body_smoothing_fwhm_m")
        object.__setattr__(self, "body_smoothing_fwhm_m", fwhm)
        for name in ("window_half_width_m", "window_center_spacing_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_positive(value, name=name))
        if isinstance(self.normalized_profile_samples, bool) or int(self.normalized_profile_samples) != self.normalized_profile_samples:
            raise ValueError("normalized_profile_samples must be an integer.")
        if int(self.normalized_profile_samples) < 3:
            raise ValueError("normalized_profile_samples must be at least three.")
        object.__setattr__(self, "normalized_profile_samples", int(self.normalized_profile_samples))
        if isinstance(self.min_window_samples, bool) or int(self.min_window_samples) != self.min_window_samples:
            raise ValueError("min_window_samples must be an integer.")
        if int(self.min_window_samples) < 3:
            raise ValueError("min_window_samples must be at least three.")
        object.__setattr__(self, "min_window_samples", int(self.min_window_samples))
        object.__setattr__(self, "denominator_floor", _finite_positive(self.denominator_floor, name="denominator_floor"))
        object.__setattr__(self, "feature_scale_floor", _finite_positive(self.feature_scale_floor, name="feature_scale_floor"))

        intervals = _normalise_zone_intervals(self.zone_intervals)
        object.__setattr__(self, "zone_intervals", intervals)
        intervals_by_well = _normalise_zone_intervals_by_well(self.zone_intervals_by_well)
        object.__setattr__(self, "zone_intervals_by_well", intervals_by_well)

        if self.sample_axis is not None and not isinstance(self.sample_axis, SampleAxis):
            raise TypeError("sample_axis must be a cup.seismic.geometry.SampleAxis.")
        if self.domain is not None:
            domain = str(self.domain).casefold()
            if domain not in {"time", "depth"}:
                raise ValueError("domain must be 'time' or 'depth'.")
            object.__setattr__(self, "domain", domain)
        if self.sample_axis is not None:
            if self.domain is not None and self.domain != self.sample_axis.domain:
                raise ValueError("ScaleContract domain differs from sample_axis.domain.")
            if self.sample_axis.domain == "depth" and self.depth_basis not in (None, "tvdss"):
                raise ValueError("Depth ScaleContract requires depth_basis='tvdss'.")
            if self.sample_axis.domain == "time" and self.depth_basis is not None:
                raise ValueError("Time ScaleContract cannot declare depth_basis.")
            object.__setattr__(self, "domain", self.sample_axis.domain)
            object.__setattr__(self, "depth_basis", self.sample_axis.depth_basis)

    @classmethod
    def from_any(cls, value: "ScaleContract | Mapping[str, Any] | None") -> "ScaleContract":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("scale_contract must be ScaleContract, a mapping, or None.")
        aliases = {
            "body_fwhm_m": "body_smoothing_fwhm_m",
            "smoothing_fwhm_m": "body_smoothing_fwhm_m",
            "window_half_width": "window_half_width_m",
            "window_spacing_m": "window_center_spacing_m",
            "profile_samples": "normalized_profile_samples",
            "zone_bounds": "zone_intervals",
            "well_zone_bounds": "zone_intervals_by_well",
        }
        data: dict[str, Any] = {}
        for key, item in value.items():
            data[aliases.get(str(key), str(key))] = item
        return cls(**data)

    def resolved_window_half_width_m(self, sample_step_m: float) -> float:
        if self.window_half_width_m is not None:
            return float(self.window_half_width_m)
        return max(2.0 * self.body_smoothing_fwhm_m, 4.0 * float(sample_step_m))

    def resolved_window_center_spacing_m(self, sample_step_m: float) -> float:
        if self.window_center_spacing_m is not None:
            return float(self.window_center_spacing_m)
        return self.resolved_window_half_width_m(sample_step_m)


@dataclass(frozen=True)
class ResidualTransferPolicy:
    """Deterministic run-time policy for one texture transfer."""

    temperature_multiplier: float = 1.0
    temperature_multipliers: tuple[float, ...] = (0.75, 1.0, 1.5)
    lateral_correlation_length_m: float = 50.0
    key_edge_scale: float = 1.0
    lambda_lateral: float = 0.75
    lambda_vertical: float = 0.25
    spatial_iterations: int = 12
    spatial_coupling: bool = True
    window_half_width_m: float | None = None
    window_center_spacing_m: float | None = None
    min_window_samples: int | None = None
    max_nodes: int | None = None
    edge_key_distance_max: float | None = None
    section_orientation: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_multiplier", _finite_positive(self.temperature_multiplier, name="temperature_multiplier"))
        multipliers = tuple(_finite_positive(item, name="temperature_multipliers") for item in self.temperature_multipliers)
        if not multipliers:
            raise ValueError("temperature_multipliers must not be empty.")
        object.__setattr__(self, "temperature_multipliers", multipliers)
        object.__setattr__(self, "lateral_correlation_length_m", _finite_positive(self.lateral_correlation_length_m, name="lateral_correlation_length_m"))
        object.__setattr__(self, "key_edge_scale", _finite_positive(self.key_edge_scale, name="key_edge_scale"))
        for name in ("lambda_lateral", "lambda_vertical"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if isinstance(self.spatial_iterations, bool) or int(self.spatial_iterations) != self.spatial_iterations or int(self.spatial_iterations) < 0:
            raise ValueError("spatial_iterations must be a non-negative integer.")
        object.__setattr__(self, "spatial_iterations", int(self.spatial_iterations))
        for name in ("window_half_width_m", "window_center_spacing_m", "edge_key_distance_max"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_positive(value, name=name))
        if self.min_window_samples is not None:
            if isinstance(self.min_window_samples, bool) or int(self.min_window_samples) != self.min_window_samples or int(self.min_window_samples) < 3:
                raise ValueError("min_window_samples must be an integer >= 3.")
            object.__setattr__(self, "min_window_samples", int(self.min_window_samples))
        if self.max_nodes is not None:
            if isinstance(self.max_nodes, bool) or int(self.max_nodes) != self.max_nodes or int(self.max_nodes) < 1:
                raise ValueError("max_nodes must be a positive integer.")
            object.__setattr__(self, "max_nodes", int(self.max_nodes))
        if self.section_orientation is not None:
            orientation = str(self.section_orientation).casefold()
            if orientation not in {"inline", "xline"}:
                raise ValueError("section_orientation must be 'inline' or 'xline'.")
            object.__setattr__(self, "section_orientation", orientation)

    @classmethod
    def from_any(cls, value: "ResidualTransferPolicy | Mapping[str, Any] | None") -> "ResidualTransferPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("policy must be ResidualTransferPolicy, a mapping, or None.")
        aliases = {
            "temperature": "temperature_multiplier",
            "temperature_scale": "temperature_multiplier",
            "lateral_length_m": "lateral_correlation_length_m",
            "key_scale": "key_edge_scale",
            "lambda_horizontal": "lambda_lateral",
            "lambda_vertical_overlap": "lambda_vertical",
            "iterations": "spatial_iterations",
        }
        data: dict[str, Any] = {}
        for key, item in value.items():
            data[aliases.get(str(key), str(key))] = item
        return cls(**data)


@dataclass(frozen=True)
class DictionaryAtom:
    """One paired body-key/residual-value window from one native well."""

    body_key: Any
    residual_value: np.ndarray
    physical_axis: np.ndarray
    zone_id: str
    normalized_zone_position: float
    valid_support: np.ndarray
    source_well: str
    source_interval: tuple[float, float]
    body_value: np.ndarray | None = None

    def __post_init__(self) -> None:
        axis = np.asarray(self.physical_axis, dtype=np.float64)
        residual = np.asarray(self.residual_value, dtype=np.float64)
        support = np.asarray(self.valid_support, dtype=bool)
        if axis.ndim != 1 or axis.size < 2 or residual.shape != axis.shape or support.shape != axis.shape:
            raise ValueError("DictionaryAtom axis, residual_value and valid_support must be matching 1-D arrays.")
        if np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
            raise ValueError("DictionaryAtom physical_axis must be finite and strictly increasing.")
        if not np.array_equal(support, np.isfinite(residual)):
            raise ValueError("DictionaryAtom valid_support must exactly describe finite residual_value samples.")
        if not np.any(support):
            raise ValueError("DictionaryAtom has no valid residual support.")
        position = float(self.normalized_zone_position)
        if not np.isfinite(position):
            raise ValueError("DictionaryAtom normalized_zone_position must be finite.")
        start, end = (float(self.source_interval[0]), float(self.source_interval[1]))
        if not np.isfinite(start) or not np.isfinite(end) or not start < end:
            raise ValueError("DictionaryAtom source_interval must satisfy finite start < end.")
        if self.body_value is not None:
            body = np.asarray(self.body_value, dtype=np.float64)
            if body.shape != axis.shape or np.any(~np.isfinite(body)):
                raise ValueError("DictionaryAtom body_value must be finite and aligned to physical_axis.")
            object.__setattr__(self, "body_value", body)
        object.__setattr__(self, "physical_axis", axis)
        object.__setattr__(self, "residual_value", residual)
        object.__setattr__(self, "valid_support", support)
        object.__setattr__(self, "zone_id", str(self.zone_id))
        object.__setattr__(self, "normalized_zone_position", position)
        object.__setattr__(self, "source_interval", (start, end))
        object.__setattr__(self, "source_well", str(self.source_well))


@dataclass(frozen=True)
class ResidualTextureLibrary:
    """Validated deterministic residual dictionary and its published scales."""

    sample_axis: SampleAxis
    atoms: tuple[DictionaryAtom, ...]
    feature_scales: Mapping[str, float]
    zone_stats: Mapping[str, Mapping[str, Any]]
    scale_contract: ScaleContract
    profile_axis: np.ndarray
    source_well_residual_rms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_axis, SampleAxis):
            raise TypeError("ResidualTextureLibrary.sample_axis must be SampleAxis.")
        atoms = tuple(self.atoms)
        if not atoms:
            raise ValueError("ResidualTextureLibrary requires at least one atom.")
        if not isinstance(self.scale_contract, ScaleContract):
            raise TypeError("scale_contract must be ScaleContract.")
        profile_axis = np.asarray(self.profile_axis, dtype=np.float64)
        if profile_axis.ndim != 1 or profile_axis.size < 3 or np.any(~np.isfinite(profile_axis)) or np.any(np.diff(profile_axis) <= 0.0):
            raise ValueError("profile_axis must be a finite strictly increasing 1-D axis.")
        for atom in atoms:
            if atom.body_key is None or not hasattr(atom.body_key, "profile"):
                raise TypeError("DictionaryAtom.body_key must be a BodyKey-like object.")
            if str(atom.body_key.zone_id) != str(atom.zone_id):
                raise ValueError("DictionaryAtom zone_id differs from body_key.zone_id.")
            if np.asarray(atom.body_key.profile).shape != profile_axis.shape:
                raise ValueError("All DictionaryAtom BodyKey profiles must share the library profile axis.")
            if not np.isclose(float(atom.body_key.normalized_zone_position), float(atom.normalized_zone_position), rtol=0.0, atol=1.0e-12):
                raise ValueError("DictionaryAtom normalized zone position differs from body_key.")
        scales: dict[str, float] = {}
        for name in _FEATURE_NAMES:
            if name not in self.feature_scales:
                raise ValueError(f"ResidualTextureLibrary is missing feature scale {name!r}.")
            scales[name] = _finite_positive(self.feature_scales[name], name=f"feature_scales[{name}]")
        zones = {str(atom.zone_id) for atom in atoms}
        if not zones.issubset({str(name) for name in self.zone_stats}):
            missing = sorted(zones - {str(name) for name in self.zone_stats})
            raise ValueError(f"zone_stats is missing dictionary zones: {missing}")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "feature_scales", scales)
        object.__setattr__(self, "profile_axis", profile_axis)
        object.__setattr__(self, "zone_stats", {str(key): dict(value) for key, value in self.zone_stats.items()})
        object.__setattr__(self, "source_well_residual_rms", {str(key): float(value) for key, value in self.source_well_residual_rms.items()})

    @property
    def zone_ids(self) -> tuple[str, ...]:
        return tuple(sorted({atom.zone_id for atom in self.atoms}))

    @property
    def source_wells(self) -> tuple[str, ...]:
        return tuple(sorted({atom.source_well for atom in self.atoms}))

    def atom_indices_for_zone(self, zone_id: Any) -> np.ndarray:
        zone = str(zone_id)
        return np.asarray([index for index, atom in enumerate(self.atoms) if atom.zone_id == zone], dtype=np.int64)

    def describe(self) -> dict[str, Any]:
        return {
            "n_atoms": int(len(self.atoms)),
            "n_zones": int(len(self.zone_ids)),
            "zones": list(self.zone_ids),
            "source_wells": list(self.source_wells),
            "feature_scales": dict(self.feature_scales),
            "body_smoothing_fwhm_m": float(self.scale_contract.body_smoothing_fwhm_m),
            "window_half_width_m": float(self.scale_contract.resolved_window_half_width_m(self.sample_axis.step or 1.0)),
            "zone_stats": {key: dict(value) for key, value in self.zone_stats.items()},
        }


@dataclass(frozen=True)
class TransferGeometry:
    """Optional explicit geometry bundle accepted by ``transfer_residual_texture``.

    Most callers can pass a ``SurveyLineGeometry`` directly.  This bundle is
    useful when a section or a target mask has already been assembled.
    """

    sample_axis: SampleAxis | None = None
    line_geometry: SurveyLineGeometry | None = None
    ilines: np.ndarray | None = None
    xlines: np.ndarray | None = None
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None
    support: np.ndarray | None = None
    zone_ids: np.ndarray | None = None
    pinchout_mask: np.ndarray | None = None
    orientation: str | None = None


@dataclass
class ResidualTransferResult:
    """Output of one deterministic conditional residual transfer."""

    ginn_body: np.ndarray
    predicted_residual: np.ndarray
    enhanced_log_ai: np.ndarray
    dictionary_weight_summary: dict[str, Any]
    effective_dictionary_count: np.ndarray
    transform_summary: dict[str, Any]
    lateral_continuity_metrics: dict[str, Any]
    support: np.ndarray
    soft_residual: np.ndarray | None = None
    hard_nearest_residual: np.ndarray | None = None
    uniform_residual: np.ndarray | None = None

    def __post_init__(self) -> None:
        body = np.asarray(self.ginn_body, dtype=np.float64)
        residual = np.asarray(self.predicted_residual, dtype=np.float64)
        enhanced = np.asarray(self.enhanced_log_ai, dtype=np.float64)
        effective = np.asarray(self.effective_dictionary_count, dtype=np.float64)
        support = np.asarray(self.support, dtype=bool)
        if residual.shape != body.shape or enhanced.shape != body.shape or effective.shape != body.shape or support.shape != body.shape:
            raise ValueError("ResidualTransferResult arrays must all match ginn_body shape.")
        if np.any(~np.isfinite(residual)) or np.any(~np.isfinite(effective[support])):
            raise ValueError("ResidualTransferResult residual/effective count must be finite.")
        if np.any(residual[~support] != 0.0):
            raise ValueError("ResidualTransferResult predicted_residual must be zero outside support.")
        if np.any(effective[support] < 0.0):
            raise ValueError("ResidualTransferResult effective_dictionary_count must be non-negative.")
        object.__setattr__(self, "ginn_body", body)
        object.__setattr__(self, "predicted_residual", residual)
        object.__setattr__(self, "enhanced_log_ai", enhanced)
        object.__setattr__(self, "effective_dictionary_count", effective)
        object.__setattr__(self, "support", support)
        if self.soft_residual is not None:
            soft = np.asarray(self.soft_residual, dtype=np.float64)
            if soft.shape != body.shape:
                raise ValueError("soft_residual must match ginn_body shape.")
            object.__setattr__(self, "soft_residual", soft)
        if self.hard_nearest_residual is not None:
            hard = np.asarray(self.hard_nearest_residual, dtype=np.float64)
            if hard.shape != body.shape:
                raise ValueError("hard_nearest_residual must match ginn_body shape.")
            object.__setattr__(self, "hard_nearest_residual", hard)
        if self.uniform_residual is not None:
            uniform = np.asarray(self.uniform_residual, dtype=np.float64)
            if uniform.shape != body.shape:
                raise ValueError("uniform_residual must match ginn_body shape.")
            object.__setattr__(self, "uniform_residual", uniform)

    @property
    def summary(self) -> dict[str, Any]:
        continuity = {
            key: value
            for key, value in self.lateral_continuity_metrics.items()
            if key.endswith("_stats") or np.isscalar(value)
        }
        return {
            "shape": list(self.ginn_body.shape),
            "support_count": int(np.count_nonzero(self.support)),
            "residual_rms": float(np.sqrt(np.mean(np.square(self.predicted_residual[self.support])))) if np.any(self.support) else 0.0,
            "effective_dictionary_count_median": float(np.median(self.effective_dictionary_count[self.support])) if np.any(self.support) else 0.0,
            "lateral_continuity_metrics": continuity,
        }


__all__ = [
    "DictionaryAtom",
    "ResidualTextureLibrary",
    "ResidualTransferPolicy",
    "ResidualTransferResult",
    "ScaleContract",
    "TransferGeometry",
]
