"""Scientific contracts for band-limited seismic evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cup.synthetic.core.records import SampleAxis


class InputContractError(ValueError):
    """Raised when axes, geometry, masks, or values violate the interface."""


class DomainMismatchError(InputContractError):
    """Raised when a model and an observation use different vertical domains."""


def _array(value: object, *, dtype: object, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != ndim:
        raise InputContractError(f"{name} must have {ndim} dimensions.")
    return result


def zone_linear_lateral_support(
    sample_axis: SampleAxis,
    observed_valid: np.ndarray,
    zone_top: np.ndarray,
    zone_bottom: np.ndarray,
) -> np.ndarray:
    """Return traces with rank-two support for a zone-linear LFM anchor."""

    if not isinstance(sample_axis, SampleAxis):
        raise TypeError("sample_axis must be a SampleAxis object.")
    valid = _array(observed_valid, dtype=bool, ndim=2, name="observed_valid")
    top = _array(zone_top, dtype=np.float64, ndim=1, name="zone_top")
    bottom = _array(zone_bottom, dtype=np.float64, ndim=1, name="zone_bottom")
    if valid.shape != (top.size, sample_axis.coordinates.size) or bottom.shape != top.shape:
        raise InputContractError(
            "zone-linear support arrays must match lateral and sample axes."
        )
    geometry_valid = np.isfinite(top) & np.isfinite(bottom) & (bottom > top)
    inside = (
        valid
        & (sample_axis.coordinates[None, :] >= top[:, None])
        & (sample_axis.coordinates[None, :] <= bottom[:, None])
    )
    return geometry_valid & (np.count_nonzero(inside, axis=1) >= 2)


@dataclass(frozen=True)
class EvidenceInput:
    """One lateral tile with an explicit vertical axis and physical geometry."""

    sample_axis: SampleAxis
    seismic: np.ndarray
    lfm: np.ndarray
    observed_valid: np.ndarray
    lateral_m: np.ndarray
    lateral_valid: np.ndarray
    zone_top: np.ndarray
    zone_bottom: np.ndarray
    vp_model_mps: np.ndarray | None = None
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None
    identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.sample_axis, SampleAxis):
            raise TypeError("sample_axis must be a SampleAxis object.")
        seismic = _array(self.seismic, dtype=np.float64, ndim=2, name="seismic")
        lfm = _array(self.lfm, dtype=np.float64, ndim=2, name="lfm")
        valid = _array(self.observed_valid, dtype=bool, ndim=2, name="observed_valid")
        expected = (seismic.shape[0], self.sample_axis.coordinates.size)
        if seismic.shape != expected or lfm.shape != expected or valid.shape != expected:
            raise InputContractError(
                "seismic, lfm, and observed_valid must be [lateral, sample]."
            )
        if np.any(valid & (~np.isfinite(seismic) | ~np.isfinite(lfm))):
            raise InputContractError("valid observation samples must be finite.")

        velocity: np.ndarray | None = None
        if self.sample_axis.sample_domain == "depth":
            if self.vp_model_mps is None:
                raise InputContractError("depth observations require vp_model_mps.")
            velocity = _array(
                self.vp_model_mps,
                dtype=np.float64,
                ndim=2,
                name="vp_model_mps",
            )
            if velocity.shape != expected or np.any(
                valid & (~np.isfinite(velocity) | (velocity <= 0.0))
            ):
                raise InputContractError(
                    "vp_model_mps must be finite, positive, and match depth observations."
                )
        elif self.vp_model_mps is not None:
            raise InputContractError("time observations cannot carry vp_model_mps.")

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
        linear_support = zone_linear_lateral_support(
            self.sample_axis,
            valid,
            top,
            bottom,
        )
        if np.any(lateral_valid & ~linear_support):
            raise InputContractError(
                "valid lateral traces require at least two observed samples inside the zone."
            )

        coordinates: list[np.ndarray | None] = []
        for name, value in (("x_m", self.x_m), ("y_m", self.y_m)):
            if value is None:
                coordinates.append(None)
                continue
            parsed = _array(value, dtype=np.float64, ndim=1, name=name)
            if parsed.shape != lateral.shape or np.any(~np.isfinite(parsed)):
                raise InputContractError(f"{name} must be finite and match lateral_m.")
            coordinates.append(parsed)
        if (coordinates[0] is None) != (coordinates[1] is None):
            raise InputContractError("x_m and y_m must both be present or absent.")
        identity = str(self.identity).strip()
        if not identity:
            raise InputContractError("evidence input identity must be non-empty.")

        object.__setattr__(self, "seismic", seismic)
        object.__setattr__(self, "lfm", lfm)
        object.__setattr__(self, "observed_valid", valid)
        object.__setattr__(self, "lateral_m", lateral)
        object.__setattr__(self, "lateral_valid", lateral_valid)
        object.__setattr__(self, "zone_top", top)
        object.__setattr__(self, "zone_bottom", bottom)
        object.__setattr__(self, "vp_model_mps", velocity)
        object.__setattr__(self, "x_m", coordinates[0])
        object.__setattr__(self, "y_m", coordinates[1])
        object.__setattr__(self, "identity", identity)

    @property
    def sample_domain(self) -> str:
        return self.sample_axis.sample_domain

    @property
    def width(self) -> int:
        return int(self.seismic.shape[0])


@dataclass(frozen=True)
class EvidenceTargetContract:
    """Frozen scales and vertical semantics shared by training and inference."""

    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    global_scales: Mapping[str, float]
    source: str

    SCHEMA = "bandlimited_evidence_target_contract_v2"
    TARGETS = (
        "projected_log_ai_increment",
        "signed_reflectivity",
    )
    REQUIRED_SCALES = (
        "seismic",
        "lfm_residual",
        "projected_log_ai_increment",
        "signed_reflectivity",
    )

    def __post_init__(self) -> None:
        domain = str(self.sample_domain).strip().casefold()
        unit = str(self.sample_unit).strip()
        basis = None if self.depth_basis in {None, ""} else str(self.depth_basis)
        if domain not in {"time", "depth"}:
            raise InputContractError("target domain must be time or depth.")
        if not unit:
            raise InputContractError("target sample_unit cannot be empty.")
        if domain == "time" and basis is not None:
            raise InputContractError("time target contract cannot declare depth_basis.")
        if domain == "depth" and basis is None:
            raise InputContractError("depth target contract requires depth_basis.")
        scales = {str(key): float(value) for key, value in self.global_scales.items()}
        if set(scales) != set(self.REQUIRED_SCALES):
            raise InputContractError("target scales do not match the evidence interface.")
        if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
            raise InputContractError("target scales must be finite and positive.")
        source = str(self.source).strip()
        if not source:
            raise InputContractError("target contract source cannot be empty.")
        object.__setattr__(self, "sample_domain", domain)
        object.__setattr__(self, "sample_unit", unit)
        object.__setattr__(self, "depth_basis", basis)
        object.__setattr__(self, "global_scales", scales)
        object.__setattr__(self, "source", source)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceTargetContract":
        if value.get("schema") != cls.SCHEMA or value.get("status") != "ready":
            raise InputContractError("unsupported evidence target contract.")
        targets = tuple(value.get("targets") or ())
        if targets != cls.TARGETS:
            raise InputContractError("target contract fields are incomplete or reordered.")
        return cls(
            sample_domain=str(value.get("sample_domain") or ""),
            sample_unit=str(value.get("sample_unit") or ""),
            depth_basis=value.get("depth_basis"),
            global_scales=dict(value.get("global_scales") or {}),
            source=str(value.get("source") or ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "status": "ready",
            "sample_domain": self.sample_domain,
            "sample_unit": self.sample_unit,
            "depth_basis": self.depth_basis,
            "targets": list(self.TARGETS),
            "global_scales": dict(self.global_scales),
            "source": self.source,
        }


@dataclass(frozen=True)
class BandlimitedEvidence:
    """Model-grid evidence inferred from seismic and the full LFM."""

    sample_axis: SampleAxis
    background_lfm_linear: np.ndarray
    projected_log_ai_increment_mean: np.ndarray
    projected_log_ai_increment_scale: np.ndarray
    signed_reflectivity_mean: np.ndarray
    signed_reflectivity_scale: np.ndarray
    local_tuning_scale: np.ndarray
    support: np.ndarray
    lateral_m: np.ndarray
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None
    identity: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.sample_axis, SampleAxis):
            raise TypeError("sample_axis must be a SampleAxis object.")
        mean = _array(
            self.projected_log_ai_increment_mean,
            dtype=np.float64,
            ndim=2,
            name="projected_log_ai_increment_mean",
        )
        shape = mean.shape
        if shape[1] != self.sample_axis.coordinates.size:
            raise InputContractError("evidence samples must match sample_axis.")
        scalar_fields = {
            "background_lfm_linear": self.background_lfm_linear,
            "projected_log_ai_increment_scale": self.projected_log_ai_increment_scale,
            "signed_reflectivity_mean": self.signed_reflectivity_mean,
            "signed_reflectivity_scale": self.signed_reflectivity_scale,
            "local_tuning_scale": self.local_tuning_scale,
        }
        parsed = {
            name: _array(value, dtype=np.float64, ndim=2, name=name)
            for name, value in scalar_fields.items()
        }
        support = _array(self.support, dtype=bool, ndim=2, name="support")
        if support.shape != shape or any(value.shape != shape for value in parsed.values()):
            raise InputContractError("all scalar evidence fields must share one shape.")
        if np.any(support & ~np.isfinite(mean)):
            raise InputContractError("supported increment means must be finite.")
        for name in (
            "projected_log_ai_increment_scale",
            "signed_reflectivity_scale",
            "local_tuning_scale",
        ):
            if np.any(support & (~np.isfinite(parsed[name]) | (parsed[name] <= 0.0))):
                raise InputContractError(f"supported {name} must be finite and positive.")
        for name in ("background_lfm_linear", "signed_reflectivity_mean"):
            if np.any(support & ~np.isfinite(parsed[name])):
                raise InputContractError(f"supported {name} must be finite.")
        lateral = _array(self.lateral_m, dtype=np.float64, ndim=1, name="lateral_m")
        if lateral.shape != (shape[0],) or np.any(~np.isfinite(lateral)) or (
            lateral.size > 1 and np.any(np.diff(lateral) <= 0.0)
        ):
            raise InputContractError("lateral_m must be finite, increasing, and match width.")
        coordinates: list[np.ndarray | None] = []
        for name, value in (("x_m", self.x_m), ("y_m", self.y_m)):
            if value is None:
                coordinates.append(None)
                continue
            item = _array(value, dtype=np.float64, ndim=1, name=name)
            if item.shape != lateral.shape or np.any(~np.isfinite(item)):
                raise InputContractError(f"{name} must be finite and match lateral_m.")
            coordinates.append(item)
        if (coordinates[0] is None) != (coordinates[1] is None):
            raise InputContractError("x_m and y_m must both be present or absent.")
        identity = str(self.identity).strip()
        if not identity:
            raise InputContractError("evidence identity must be non-empty.")

        object.__setattr__(self, "projected_log_ai_increment_mean", mean)
        for name, value in parsed.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "lateral_m", lateral)
        object.__setattr__(self, "x_m", coordinates[0])
        object.__setattr__(self, "y_m", coordinates[1])
        object.__setattr__(self, "identity", identity)


__all__ = [
    "BandlimitedEvidence",
    "DomainMismatchError",
    "EvidenceInput",
    "EvidenceTargetContract",
    "InputContractError",
    "zone_linear_lateral_support",
]
