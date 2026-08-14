"""Interpretable body keys and continuous analytic residual transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import DictionaryAtom


FEATURE_NAMES = (
    "profile",
    "first_derivative",
    "second_derivative",
    "contrast",
    "thickness",
    "position",
)


def _finite_axis(axis: np.ndarray, *, name: str = "axis") -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be a finite strictly increasing 1-D axis with at least two samples.")
    return values


@dataclass(frozen=True)
class BodyKey:
    """Continuous retrieval key for one local body window."""

    profile: np.ndarray
    first_derivative: np.ndarray
    second_derivative: np.ndarray
    contrast: float
    thickness: float
    zone_id: str
    normalized_zone_position: float
    local_mean: float
    mu: float
    center_m: float
    window_half_width_m: float

    def __post_init__(self) -> None:
        profile = np.asarray(self.profile, dtype=np.float64)
        first = np.asarray(self.first_derivative, dtype=np.float64)
        second = np.asarray(self.second_derivative, dtype=np.float64)
        if profile.ndim != 1 or profile.size < 3 or first.shape != profile.shape or second.shape != profile.shape:
            raise ValueError("BodyKey profiles and derivatives must be matching 1-D arrays.")
        for name, value in (("profile", profile), ("first_derivative", first), ("second_derivative", second)):
            if np.any(~np.isfinite(value)):
                raise ValueError(f"BodyKey {name} contains non-finite values.")
            object.__setattr__(self, name, value)
        for name in ("contrast", "thickness", "normalized_zone_position", "local_mean", "mu", "center_m", "window_half_width_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"BodyKey {name} must be finite.")
            object.__setattr__(self, name, value)
        if self.contrast <= 0.0:
            raise ValueError("BodyKey contrast must be positive.")
        if self.thickness <= 0.0:
            raise ValueError("BodyKey thickness must be positive.")
        if self.window_half_width_m <= 0.0:
            raise ValueError("BodyKey window_half_width_m must be positive.")
        object.__setattr__(self, "zone_id", str(self.zone_id))

    @property
    def normalized_profile(self) -> np.ndarray:
        return self.profile

    @property
    def normalized_local_body_profile(self) -> np.ndarray:
        return self.profile

    @property
    def d1(self) -> np.ndarray:
        return self.first_derivative

    @property
    def first_physical_derivative(self) -> np.ndarray:
        return self.first_derivative

    @property
    def d2(self) -> np.ndarray:
        return self.second_derivative

    @property
    def second_physical_derivative(self) -> np.ndarray:
        return self.second_derivative

    @property
    def position(self) -> float:
        return self.normalized_zone_position

    @property
    def local_contrast(self) -> float:
        return self.contrast

    @property
    def dominant_body_thickness(self) -> float:
        return self.thickness


@dataclass(frozen=True)
class TransformParameters:
    """Bounded continuous transform from an atom window to a query window."""

    shift: float
    stretch: float
    amplitude: float

    def as_dict(self) -> dict[str, float]:
        return {
            "shift": float(self.shift),
            "stretch": float(self.stretch),
            "amplitude": float(self.amplitude),
        }


class BodyKeyEncoder:
    """Encode body windows using one shared physical and normalized contract."""

    def __init__(
        self,
        *,
        window_half_width_m: float = 30.0,
        profile_samples: int = 33,
        denominator_floor: float = 1.0e-12,
    ) -> None:
        self.window_half_width_m = float(window_half_width_m)
        self.profile_samples = int(profile_samples)
        self.denominator_floor = float(denominator_floor)
        if not np.isfinite(self.window_half_width_m) or self.window_half_width_m <= 0.0:
            raise ValueError("window_half_width_m must be finite and positive.")
        if self.profile_samples < 3:
            raise ValueError("profile_samples must be at least three.")
        if not np.isfinite(self.denominator_floor) or self.denominator_floor <= 0.0:
            raise ValueError("denominator_floor must be finite and positive.")
        self.profile_axis = np.linspace(-1.0, 1.0, self.profile_samples, dtype=np.float64)

    def encode(
        self,
        body_values: np.ndarray,
        physical_axis: np.ndarray,
        *,
        center_m: float | None = None,
        zone_id: Any = "global",
        normalized_zone_position: float = 0.5,
    ) -> BodyKey:
        axis = _finite_axis(physical_axis, name="physical_axis")
        body = np.asarray(body_values, dtype=np.float64)
        if body.shape != axis.shape or np.any(~np.isfinite(body)):
            raise ValueError("body_values must be finite and aligned to physical_axis.")
        center = float(np.mean((axis[0], axis[-1])) if center_m is None else center_m)
        if not np.isfinite(center):
            raise ValueError("center_m must be finite.")

        half_width = self.window_half_width_m
        normalized_axis = (axis - center) / half_width
        # The edge values are deliberately held at the boundary.  This is only
        # used for a window clipped by a finite-run boundary; no value is read
        # across a gap or a zone boundary.
        profile_raw = np.interp(
            self.profile_axis,
            normalized_axis,
            body,
            left=float(body[0]),
            right=float(body[-1]),
        )
        local_mean = float(np.mean(body))
        contrast = float(np.sqrt(np.mean(np.square(body - local_mean))))
        contrast = max(contrast, self.denominator_floor)
        profile = (profile_raw - float(np.mean(profile_raw))) / max(
            float(np.sqrt(np.mean(np.square(profile_raw - np.mean(profile_raw))))),
            self.denominator_floor,
        )
        physical_profile_axis = self.profile_axis * half_width
        first = np.gradient(profile, physical_profile_axis, edge_order=1)
        second = np.gradient(first, physical_profile_axis, edge_order=1)

        if axis.size == 2:
            derivative = np.asarray([(body[1] - body[0]) / (axis[1] - axis[0])] * 2, dtype=np.float64)
        else:
            derivative = np.gradient(body, axis, edge_order=1)
        derivative_weight = np.abs(derivative)
        derivative_sum = max(float(np.sum(derivative_weight)), self.denominator_floor)
        local_coordinates = axis - center
        mu = float(np.sum(derivative_weight * local_coordinates) / derivative_sum)
        second_moment = float(np.sum(derivative_weight * np.square(local_coordinates - mu)) / derivative_sum)
        thickness = 2.0 * float(np.sqrt(max(second_moment, self.denominator_floor)))
        thickness = max(thickness, float(np.median(np.diff(axis))), self.denominator_floor)

        return BodyKey(
            profile=profile,
            first_derivative=first,
            second_derivative=second,
            contrast=contrast,
            thickness=thickness,
            zone_id=str(zone_id),
            normalized_zone_position=float(normalized_zone_position),
            local_mean=local_mean,
            mu=mu,
            center_m=center,
            window_half_width_m=half_width,
        )


def _rms_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("BodyKey profile shapes differ; use one shared BodyKeyEncoder.")
    return float(np.sqrt(np.mean(np.square(left - right))))


def key_distance_components(
    query: BodyKey,
    atom: BodyKey,
    feature_scales: Mapping[str, float],
) -> dict[str, float]:
    """Return the six normalized feature-group distances."""

    if query.zone_id != atom.zone_id:
        return {name: float("inf") for name in FEATURE_NAMES}
    required = set(FEATURE_NAMES)
    missing = required - set(feature_scales)
    if missing:
        raise ValueError(f"feature_scales is missing {sorted(missing)}.")
    values = {
        "profile": _rms_difference(query.profile, atom.profile),
        "first_derivative": _rms_difference(query.first_derivative, atom.first_derivative),
        "second_derivative": _rms_difference(query.second_derivative, atom.second_derivative),
        "contrast": abs(query.contrast - atom.contrast),
        "thickness": abs(query.thickness - atom.thickness),
        "position": abs(query.normalized_zone_position - atom.normalized_zone_position),
    }
    return {
        name: float(values[name]) / max(float(feature_scales[name]), np.finfo(np.float64).tiny)
        for name in FEATURE_NAMES
    }


def weighted_key_distance(
    query: BodyKey,
    atom: BodyKey,
    feature_scales: Mapping[str, float],
) -> float:
    """Compute the equal-weight six-group distance from the specification."""

    components = key_distance_components(query, atom, feature_scales)
    if any(not np.isfinite(value) for value in components.values()):
        return float("inf")
    return float(np.sqrt(sum(value * value for value in components.values()) / len(FEATURE_NAMES)))


def compute_feature_scales(
    atoms: Iterable[DictionaryAtom],
    *,
    scale_floor: float = 1.0e-12,
) -> dict[str, float]:
    """Publish median non-zero pairwise distances for each feature group."""

    atom_list = tuple(atoms)
    if not atom_list:
        raise ValueError("Cannot compute feature scales for an empty atom list.")
    keys = [atom.body_key for atom in atom_list]
    values: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}
    for left_index in range(len(keys)):
        for right_index in range(left_index + 1, len(keys)):
            left = keys[left_index]
            right = keys[right_index]
            raw = {
                "profile": _rms_difference(left.profile, right.profile),
                "first_derivative": _rms_difference(left.first_derivative, right.first_derivative),
                "second_derivative": _rms_difference(left.second_derivative, right.second_derivative),
                "contrast": abs(left.contrast - right.contrast),
                "thickness": abs(left.thickness - right.thickness),
                "position": abs(left.normalized_zone_position - right.normalized_zone_position),
            }
            for name, value in raw.items():
                if np.isfinite(value) and value > scale_floor:
                    values[name].append(float(value))
    result: dict[str, float] = {}
    for name in FEATURE_NAMES:
        if not values[name]:
            raise ValueError(
                f"Feature scale {name!r} has no non-zero pairwise dictionary distance; "
                "the dictionary does not satisfy the retrieval-scale contract."
            )
        result[name] = float(np.median(values[name]))
        if not np.isfinite(result[name]) or result[name] <= 0.0:
            raise ValueError(f"Feature scale {name!r} is not finite and positive.")
    return result


def compute_zone_temperature_bases(
    atoms: Iterable[DictionaryAtom],
    feature_scales: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Compute third-neighbour temperature bases and coverage diagnostics."""

    atom_list = tuple(atoms)
    zone_names = sorted({atom.zone_id for atom in atom_list})
    result: dict[str, dict[str, Any]] = {}
    for zone in zone_names:
        indices = [index for index, atom in enumerate(atom_list) if atom.zone_id == zone]
        source_wells = sorted({atom_list[index].source_well for index in indices})
        if len(indices) < 4 or len(source_wells) < 2:
            raise ValueError(
                f"Zone {zone!r} violates residual dictionary coverage: "
                f"atom_count={len(indices)}, source_well_count={len(source_wells)}; "
                "requires at least four atoms from at least two wells."
            )
        distances: list[float] = []
        for index in indices:
            atom_key = atom_list[index].body_key
            others = sorted(
                weighted_key_distance(atom_key, atom_list[other].body_key, feature_scales)
                for other in indices
                if other != index
            )
            if len(others) < 3 or not np.isfinite(others[2]) or others[2] <= 0.0:
                raise ValueError(f"Zone {zone!r} has invalid third-neighbour distance for atom {index}.")
            distances.append(float(others[2]))
        base = float(np.median(np.asarray(distances, dtype=np.float64)))
        if not np.isfinite(base) or base <= 0.0:
            raise ValueError(f"Zone {zone!r} has invalid temperature_base={base!r}.")
        result[zone] = {
            "atom_count": int(len(indices)),
            "source_well_count": int(len(source_wells)),
            "source_wells": source_wells,
            "temperature_base": base,
            "third_neighbour_distances": distances,
        }
    return result


def analytic_transform(query: BodyKey, atom: BodyKey, *, denominator_floor: float = 1.0e-12) -> TransformParameters:
    """Compute the bounded continuous shift/stretch/amplitude transform."""

    floor = float(denominator_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("denominator_floor must be finite and positive.")
    stretch = np.clip(query.thickness / max(atom.thickness, floor), 0.75, 1.33)
    amplitude = np.clip(np.sqrt(query.contrast / max(atom.contrast, floor)), 0.75, 1.33)
    shift = query.mu - atom.mu
    return TransformParameters(float(shift), float(stretch), float(amplitude))


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in changes.reshape(-1, 2)]


def transform_residual(
    atom: DictionaryAtom,
    query: BodyKey,
    query_axis: np.ndarray,
    *,
    denominator_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, TransformParameters]:
    """Apply one analytic transform with no cross-gap interpolation."""

    axis = _finite_axis(query_axis, name="query_axis")
    params = analytic_transform(query, atom.body_key, denominator_floor=denominator_floor)
    source_coordinates = atom.physical_axis
    source_values = atom.residual_value
    source_relative = atom.body_key.center_m + atom.body_key.mu + (axis - query.center_m - query.mu) / params.stretch
    transformed = np.zeros(axis.shape, dtype=np.float64)
    valid = np.zeros(axis.shape, dtype=bool)
    for start, stop in _finite_runs(atom.valid_support):
        if stop - start < 2:
            exact = np.isclose(source_relative, source_coordinates[start], rtol=0.0, atol=1.0e-10)
            transformed[exact] = source_values[start]
            valid[exact] = True
            continue
        source_axis = source_coordinates[start:stop]
        inside = (source_relative >= source_axis[0]) & (source_relative <= source_axis[-1])
        if np.any(inside):
            transformed[inside] = np.interp(source_relative[inside], source_axis, source_values[start:stop])
            valid[inside] = True
    transformed *= params.amplitude
    return transformed, valid, params


__all__ = [
    "BodyKey",
    "BodyKeyEncoder",
    "FEATURE_NAMES",
    "TransformParameters",
    "analytic_transform",
    "compute_feature_scales",
    "compute_zone_temperature_bases",
    "key_distance_components",
    "transform_residual",
    "weighted_key_distance",
]
