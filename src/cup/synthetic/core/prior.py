"""Versioned producer prior shared by Synthoseis-lite and Structured GINN."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cup.synthetic.core.calibration import ImpedanceCalibration, STATE_NAMES


PRODUCER_PRIOR_SCHEMA = "synthoseis_producer_prior_v1"


@dataclass(frozen=True)
class BoundedDistribution:
    median: float
    robust_sigma: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (self.median, self.robust_sigma, self.lower, self.upper),
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)) or self.robust_sigma <= 0.0:
            raise ValueError("producer distributions must be finite with positive scale.")
        if self.lower > self.median or self.median > self.upper:
            raise ValueError("producer distribution median must lie within its bounds.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BoundedDistribution":
        return cls(
            median=float(value["median"]),
            robust_sigma=float(value["robust_sigma"]),
            lower=float(value["lower"]),
            upper=float(value["upper"]),
        )

    def to_mapping(self) -> dict[str, float]:
        return {
            "median": float(self.median),
            "robust_sigma": float(self.robust_sigma),
            "lower": float(self.lower),
            "upper": float(self.upper),
        }


@dataclass(frozen=True)
class StateProducerPrior:
    state_id: int
    state_name: str
    log_duration: BoundedDistribution
    profile_distributions: Mapping[str, BoundedDistribution]

    def __post_init__(self) -> None:
        if self.state_id not in {0, 1, 2}:
            raise ValueError("producer state_id must be 0, 1, or 2.")
        if self.state_name != STATE_NAMES[self.state_id]:
            raise ValueError("producer state name and state_id differ.")
        required = {
            "c0",
            "c1",
            "c2",
            "profile_mean",
            "endpoint_difference",
            "peak_to_peak",
            "internal_extreme_amplitude",
        }
        parsed = dict(self.profile_distributions)
        missing = sorted(required - set(parsed))
        if missing:
            raise ValueError(f"producer state prior lacks distributions: {missing}")
        object.__setattr__(self, "profile_distributions", MappingProxyType(parsed))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StateProducerPrior":
        distributions = value.get("profile_distributions")
        if not isinstance(distributions, Mapping):
            raise ValueError("producer profile_distributions must be a mapping.")
        return cls(
            state_id=int(value["state_id"]),
            state_name=str(value["state_name"]),
            log_duration=BoundedDistribution.from_mapping(value["log_duration"]),
            profile_distributions={
                str(name): BoundedDistribution.from_mapping(item)
                for name, item in distributions.items()
            },
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "state_id": int(self.state_id),
            "state_name": self.state_name,
            "log_duration": self.log_duration.to_mapping(),
            "profile_distributions": {
                name: value.to_mapping()
                for name, value in sorted(self.profile_distributions.items())
            },
        }


@dataclass(frozen=True)
class ZoneProducerPrior:
    zone_id: str
    initial_probability: np.ndarray
    transition_probability: np.ndarray
    states: tuple[StateProducerPrior, ...]
    background_distributions: Mapping[str, BoundedDistribution]
    log_ai_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        identity = str(self.zone_id).strip()
        initial = np.asarray(self.initial_probability, dtype=np.float64)
        transition = np.asarray(self.transition_probability, dtype=np.float64)
        if not identity:
            raise ValueError("producer zone_id cannot be empty.")
        if initial.shape != (3,) or transition.shape != (3, 3):
            raise ValueError("producer transition contract requires three states.")
        if np.any(~np.isfinite(initial)) or np.any(initial <= 0.0):
            raise ValueError("producer initial probabilities must be finite and positive.")
        if np.any(~np.isfinite(transition)) or np.any(transition < 0.0):
            raise ValueError("producer transition probabilities must be finite and nonnegative.")
        if not np.allclose(np.diag(transition), 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("producer transition diagonal must be zero.")
        if not np.isclose(np.sum(initial), 1.0, rtol=0.0, atol=1.0e-10):
            raise ValueError("producer initial probabilities must sum to one.")
        if not np.allclose(
            np.sum(transition, axis=1), 1.0, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("producer transition rows must sum to one.")
        if tuple(item.state_id for item in self.states) != (0, 1, 2):
            raise ValueError("producer zone states must be ordered 0, 1, 2.")
        background = dict(self.background_distributions)
        if set(background) != {"background_a", "background_b", "zone_extent"}:
            raise ValueError("producer background distributions are incomplete.")
        bounds = tuple(float(item) for item in self.log_ai_bounds)
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("producer log-AI bounds are invalid.")
        object.__setattr__(self, "zone_id", identity)
        object.__setattr__(self, "initial_probability", initial)
        object.__setattr__(self, "transition_probability", transition)
        object.__setattr__(self, "background_distributions", MappingProxyType(background))
        object.__setattr__(self, "log_ai_bounds", bounds)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ZoneProducerPrior":
        background = value.get("background_distributions")
        if not isinstance(background, Mapping):
            raise ValueError("producer background_distributions must be a mapping.")
        return cls(
            zone_id=str(value["zone_id"]),
            initial_probability=np.asarray(value["initial_probability"]),
            transition_probability=np.asarray(value["transition_probability"]),
            states=tuple(
                StateProducerPrior.from_mapping(item) for item in value["states"]
            ),
            background_distributions={
                str(name): BoundedDistribution.from_mapping(item)
                for name, item in background.items()
            },
            log_ai_bounds=tuple(float(item) for item in value["log_ai_bounds"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "initial_probability": self.initial_probability.tolist(),
            "transition_probability": self.transition_probability.tolist(),
            "states": [item.to_mapping() for item in self.states],
            "background_distributions": {
                name: value.to_mapping()
                for name, value in sorted(self.background_distributions.items())
            },
            "log_ai_bounds": list(self.log_ai_bounds),
        }


@dataclass(frozen=True)
class ProducerPrior:
    generator_family: str
    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    zones: tuple[ZoneProducerPrior, ...]
    correlation_lengths_m: tuple[float, ...]
    coefficient_sigma_multipliers: tuple[float, ...]
    thickness_log_sigma_values: tuple[float, ...]
    geometry_families: tuple[str, ...]
    geometry_directions: tuple[str, ...]

    def __post_init__(self) -> None:
        domain = str(self.sample_domain).strip().casefold()
        unit = str(self.sample_unit).strip()
        if (domain, unit) not in {("time", "s"), ("depth", "m")}:
            raise ValueError("producer prior domain must be time/s or depth/m.")
        if domain == "depth" and self.depth_basis != "tvdss":
            raise ValueError("depth producer prior requires TVDSS.")
        if domain == "time" and self.depth_basis is not None:
            raise ValueError("time producer prior cannot define a depth basis.")
        if not str(self.generator_family).strip() or not self.zones:
            raise ValueError("producer prior requires a generator family and zones.")
        zone_ids = [item.zone_id for item in self.zones]
        if len(set(zone_ids)) != len(zone_ids):
            raise ValueError("producer prior contains duplicate zones.")
        correlation = tuple(float(item) for item in self.correlation_lengths_m)
        coefficient = tuple(float(item) for item in self.coefficient_sigma_multipliers)
        thickness = tuple(float(item) for item in self.thickness_log_sigma_values)
        if not correlation or any(not np.isfinite(item) or item <= 0.0 for item in correlation):
            raise ValueError("producer correlation lengths must be positive metres.")
        if (
            not coefficient
            or len(coefficient) != len(thickness)
            or any(not np.isfinite(item) or item < 0.0 for item in (*coefficient, *thickness))
        ):
            raise ValueError("producer lateral variation controls are invalid.")
        families = tuple(str(item) for item in self.geometry_families)
        directions = tuple(str(item) for item in self.geometry_directions)
        if not families or not directions:
            raise ValueError("producer prior requires geometry families and directions.")
        object.__setattr__(self, "sample_domain", domain)
        object.__setattr__(self, "sample_unit", unit)
        object.__setattr__(self, "correlation_lengths_m", correlation)
        object.__setattr__(self, "coefficient_sigma_multipliers", coefficient)
        object.__setattr__(self, "thickness_log_sigma_values", thickness)
        object.__setattr__(self, "geometry_families", families)
        object.__setattr__(self, "geometry_directions", directions)

    def zone(self, zone_id: str) -> ZoneProducerPrior:
        matches = [item for item in self.zones if item.zone_id == str(zone_id)]
        if len(matches) != 1:
            raise KeyError(f"unknown producer-prior zone: {zone_id}")
        return matches[0]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProducerPrior":
        if value.get("schema") != PRODUCER_PRIOR_SCHEMA:
            raise ValueError("unsupported producer prior schema.")
        return cls(
            generator_family=str(value["generator_family"]),
            sample_domain=str(value["sample_domain"]),
            sample_unit=str(value["sample_unit"]),
            depth_basis=(None if value.get("depth_basis") is None else str(value["depth_basis"])),
            zones=tuple(ZoneProducerPrior.from_mapping(item) for item in value["zones"]),
            correlation_lengths_m=tuple(value["correlation_lengths_m"]),
            coefficient_sigma_multipliers=tuple(value["coefficient_sigma_multipliers"]),
            thickness_log_sigma_values=tuple(value["thickness_log_sigma_values"]),
            geometry_families=tuple(value["geometry_families"]),
            geometry_directions=tuple(value["geometry_directions"]),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": PRODUCER_PRIOR_SCHEMA,
            "generator_family": self.generator_family,
            "sample_domain": self.sample_domain,
            "sample_unit": self.sample_unit,
            "depth_basis": self.depth_basis,
            "zones": [item.to_mapping() for item in self.zones],
            "correlation_lengths_m": list(self.correlation_lengths_m),
            "coefficient_sigma_multipliers": list(self.coefficient_sigma_multipliers),
            "thickness_log_sigma_values": list(self.thickness_log_sigma_values),
            "geometry_families": list(self.geometry_families),
            "geometry_directions": list(self.geometry_directions),
        }


def _distribution(value: Mapping[str, Any]) -> BoundedDistribution:
    return BoundedDistribution(
        median=float(value["median"]),
        robust_sigma=float(value["robust_sigma"]),
        lower=float(value["lower"] if "lower" in value else value["p01"]),
        upper=float(value["upper"] if "upper" in value else value["p99"]),
    )


def build_producer_prior(
    calibration: ImpedanceCalibration,
    *,
    object_core_controls: Mapping[str, Any],
    geometry_families: Sequence[str],
    geometry_directions: Sequence[str],
) -> ProducerPrior:
    """Materialize the exact prior consumed by the object producer."""

    states: list[ZoneProducerPrior] = []
    for zone in calibration.zones:
        zone_id = str(zone["zone_id"])
        model = calibration.zone_models[zone_id]
        state_priors = []
        for state_id, state_name in enumerate(STATE_NAMES):
            state_model = model["states"][state_name]
            state_priors.append(
                StateProducerPrior(
                    state_id=state_id,
                    state_name=state_name,
                    log_duration=_distribution(state_model["log_duration"]),
                    profile_distributions={
                        name: _distribution(item)
                        for name, item in state_model["coefficients"].items()
                    },
                )
            )
        background = model["background"]
        states.append(
            ZoneProducerPrior(
                zone_id=zone_id,
                initial_probability=np.asarray(model["initial_probabilities"]),
                transition_probability=np.asarray(model["transition_matrix"]),
                states=tuple(state_priors),
                background_distributions={
                    "background_a": _distribution(background["background_a"]),
                    "background_b": _distribution(background["background_b"]),
                    "zone_extent": _distribution(background["zone_extent"]),
                },
                log_ai_bounds=(
                    float(model["ai_bounds"]["p01"]),
                    float(model["ai_bounds"]["p99"]),
                ),
            )
        )
    return ProducerPrior(
        generator_family=calibration.generator_family,
        sample_domain="time" if calibration.axis_unit == "s" else "depth",
        sample_unit=calibration.axis_unit,
        depth_basis=calibration.depth_basis,
        zones=tuple(states),
        correlation_lengths_m=tuple(object_core_controls["correlation_length_m"]),
        coefficient_sigma_multipliers=tuple(
            object_core_controls["coefficient_sigma_multipliers"]
        ),
        thickness_log_sigma_values=tuple(
            object_core_controls["thickness_log_sigma_values"]
        ),
        geometry_families=tuple(str(item) for item in geometry_families),
        geometry_directions=tuple(str(item) for item in geometry_directions),
    )


def write_producer_prior(prior: ProducerPrior, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".staging")
    temporary.write_text(
        json.dumps(prior.to_mapping(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_producer_prior(value: Mapping[str, Any] | str | Path) -> ProducerPrior:
    if isinstance(value, Mapping):
        payload = value
    else:
        with Path(value).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("producer prior root must be a mapping.")
    return ProducerPrior.from_mapping(payload)


__all__ = [
    "BoundedDistribution",
    "PRODUCER_PRIOR_SCHEMA",
    "ProducerPrior",
    "StateProducerPrior",
    "ZoneProducerPrior",
    "build_producer_prior",
    "load_producer_prior",
    "write_producer_prior",
]
