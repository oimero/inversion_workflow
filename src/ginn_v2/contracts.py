"""Public scientific contracts for Structured GINN V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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


def zone_linear_lateral_support(
    model_axis: SampleAxis,
    observed_valid: np.ndarray,
    zone_top: np.ndarray,
    zone_bottom: np.ndarray,
) -> np.ndarray:
    """Return traces with rank-two support for a zone-linear anchor."""

    if not isinstance(model_axis, SampleAxis):
        raise TypeError("model_axis must be a SampleAxis object.")
    valid = _array(
        observed_valid,
        dtype=bool,
        ndim=2,
        name="observed_valid",
    )
    top = _array(zone_top, dtype=np.float64, ndim=1, name="zone_top")
    bottom = _array(zone_bottom, dtype=np.float64, ndim=1, name="zone_bottom")
    if valid.shape != (top.size, model_axis.coordinates.size) or (
        bottom.shape != top.shape
    ):
        raise InputContractError(
            "zone-linear support arrays must match lateral and model axes."
        )
    geometry_valid = np.isfinite(top) & np.isfinite(bottom) & (bottom > top)
    inside = (
        valid
        & (model_axis.coordinates[None, :] >= top[:, None])
        & (model_axis.coordinates[None, :] <= bottom[:, None])
    )
    return geometry_valid & (np.count_nonzero(inside, axis=1) >= 2)


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
    vp_model_mps: np.ndarray | None = None
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

        velocity: np.ndarray | None = None
        if self.model_axis.sample_domain == "depth":
            if self.vp_model_mps is None:
                raise InputContractError(
                    "depth observations require model-grid vp_model_mps."
                )
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
            self.model_axis,
            valid,
            top,
            bottom,
        )
        if np.any(lateral_valid & ~linear_support):
            raise InputContractError(
                "valid lateral traces require at least two observed model samples "
                "inside the zone."
            )

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
        object.__setattr__(self, "vp_model_mps", velocity)
        object.__setattr__(self, "x_m", parsed_coordinates[0])
        object.__setattr__(self, "y_m", parsed_coordinates[1])

    @property
    def sample_domain(self) -> str:
        return self.model_axis.sample_domain

    @property
    def width(self) -> int:
        return int(self.seismic.shape[0])


@dataclass(frozen=True)
class ObservableTargetContract:
    """Passed L0-L2 target admission contract embedded in every checkpoint."""

    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    targets: tuple[str, ...]
    global_scales: Mapping[str, float]
    audit_report: str

    SCHEMA = "structured_ginn_v2_observable_target_contract_v1"
    REQUIRED_TARGETS = (
        "projected_log_ai_increment",
        "signed_reflectivity",
        "state_emission",
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
        targets = tuple(str(value) for value in self.targets)
        if domain not in {"time", "depth"}:
            raise InputContractError("target contract domain must be time or depth.")
        if not unit:
            raise InputContractError("target contract sample_unit cannot be empty.")
        if domain == "time" and basis is not None:
            raise InputContractError("time target contract cannot declare depth_basis.")
        if domain == "depth" and basis is None:
            raise InputContractError("depth target contract requires depth_basis.")
        if targets != self.REQUIRED_TARGETS:
            raise InputContractError(
                "target contract must contain the three audited targets in order."
            )
        scales = {str(key): float(value) for key, value in self.global_scales.items()}
        if set(scales) != set(self.REQUIRED_SCALES):
            raise InputContractError(
                "target contract global_scales do not match the audited interface."
            )
        if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
            raise InputContractError("target contract scales must be finite and positive.")
        report = str(self.audit_report).strip()
        if not report:
            raise InputContractError("target contract must identify its audit report.")
        object.__setattr__(self, "sample_domain", domain)
        object.__setattr__(self, "sample_unit", unit)
        object.__setattr__(self, "depth_basis", basis)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "global_scales", scales)
        object.__setattr__(self, "audit_report", report)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObservableTargetContract":
        if value.get("schema") != cls.SCHEMA or value.get("status") != "passed":
            raise InputContractError("observable target contract has not passed L0-L2.")
        return cls(
            sample_domain=str(value.get("sample_domain") or ""),
            sample_unit=str(value.get("sample_unit") or ""),
            depth_basis=value.get("depth_basis"),
            targets=tuple(value.get("targets") or ()),
            global_scales=dict(value.get("global_scales") or {}),
            audit_report=str(value.get("audit_report") or ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "status": "passed",
            "sample_domain": self.sample_domain,
            "sample_unit": self.sample_unit,
            "depth_basis": self.depth_basis,
            "targets": list(self.targets),
            "global_scales": dict(self.global_scales),
            "audit_report": self.audit_report,
        }


@dataclass(frozen=True)
class ObservableEvidence:
    """Audited model-grid evidence; no micro-boundary probability is exposed."""

    model_axis: SampleAxis
    highres_axis: SampleAxis
    background_lfm_linear: np.ndarray
    background_lfm_linear_highres: np.ndarray
    projected_log_ai_increment_mean: np.ndarray
    projected_log_ai_increment_scale: np.ndarray
    signed_reflectivity_mean: np.ndarray
    signed_reflectivity_scale: np.ndarray
    state_log_potential: np.ndarray
    local_tuning_scale: np.ndarray
    support: np.ndarray
    highres_support: np.ndarray
    lateral_m: np.ndarray
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
            self.projected_log_ai_increment_mean,
            dtype=np.float64,
            ndim=2,
            name="projected_log_ai_increment_mean",
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
        background_highres = _array(
            self.background_lfm_linear_highres,
            dtype=np.float64,
            ndim=2,
            name="background_lfm_linear_highres",
        )
        scale = _array(
            self.projected_log_ai_increment_scale,
            dtype=np.float64,
            ndim=2,
            name="projected_log_ai_increment_scale",
        )
        reflectivity = _array(
            self.signed_reflectivity_mean,
            dtype=np.float64,
            ndim=2,
            name="signed_reflectivity_mean",
        )
        reflectivity_scale = _array(
            self.signed_reflectivity_scale,
            dtype=np.float64,
            ndim=2,
            name="signed_reflectivity_scale",
        )
        tuning = _array(
            self.local_tuning_scale,
            dtype=np.float64,
            ndim=2,
            name="local_tuning_scale",
        )
        support = _array(self.support, dtype=bool, ndim=2, name="support")
        highres_support = _array(
            self.highres_support,
            dtype=bool,
            ndim=2,
            name="highres_support",
        )
        state_log_potential = _array(
            self.state_log_potential,
            dtype=np.float64,
            ndim=3,
            name="state_log_potential",
        )
        if any(
            item.shape != shape
            for item in (
                background,
                scale,
                reflectivity,
                reflectivity_scale,
                tuning,
                support,
            )
        ):
            raise InputContractError("all scalar evidence fields must share one shape.")
        if state_log_potential.shape != shape + (3,):
            raise InputContractError(
                "state_log_potential must have shape [lateral, sample, 3]."
            )
        highres_shape = (shape[0], self.highres_axis.coordinates.size)
        if (
            background_highres.shape != highres_shape
            or highres_support.shape != highres_shape
        ):
            raise InputContractError(
                "high-resolution anchor/support must be [lateral, highres_sample]."
            )
        if np.any(highres_support & ~np.isfinite(background_highres)):
            raise InputContractError(
                "supported high-resolution LFM anchor must be finite."
            )
        if np.any(support & ~np.isfinite(mean)) or np.any(
            support & (~np.isfinite(scale) | (scale <= 0.0))
        ):
            raise NumericalFailure("supported evidence mean/scale must be finite and positive.")
        if np.any(support & ~np.isfinite(reflectivity)) or np.any(
            support & (~np.isfinite(reflectivity_scale) | (reflectivity_scale <= 0.0))
        ):
            raise NumericalFailure(
                "supported reflectivity mean/scale must be finite and positive."
            )
        if np.any(support & (~np.isfinite(tuning) | (tuning <= 0.0))):
            raise InputContractError("local_tuning_scale must be finite and positive.")
        if np.any(support & ~np.all(np.isfinite(state_log_potential), axis=-1)):
            raise NumericalFailure("supported state_log_potential must be finite.")
        log_normalizer = np.logaddexp.reduce(state_log_potential, axis=-1)
        if np.any(support & ~np.isclose(log_normalizer, 0.0, rtol=0.0, atol=1.0e-5)):
            raise InputContractError("state_log_potential must be log-normalized.")
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
        identity = str(self.identity).strip()
        if not identity:
            raise InputContractError("observable evidence identity must be non-empty.")

        object.__setattr__(self, "background_lfm_linear", background)
        object.__setattr__(
            self, "background_lfm_linear_highres", background_highres
        )
        object.__setattr__(self, "projected_log_ai_increment_mean", mean)
        object.__setattr__(self, "projected_log_ai_increment_scale", scale)
        object.__setattr__(self, "signed_reflectivity_mean", reflectivity)
        object.__setattr__(self, "signed_reflectivity_scale", reflectivity_scale)
        object.__setattr__(self, "state_log_potential", state_log_potential)
        object.__setattr__(self, "local_tuning_scale", tuning)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "highres_support", highres_support)
        object.__setattr__(self, "lateral_m", lateral)
        object.__setattr__(self, "x_m", parsed_coordinates[0])
        object.__setattr__(self, "y_m", parsed_coordinates[1])
        object.__setattr__(self, "identity", identity)


@dataclass(frozen=True)
class SegmentExtent:
    """One externally supplied high-resolution segment extent."""

    trace_index: int
    state_id: int
    start_index: int
    stop_index: int
    duration_fraction: float

    def __post_init__(self) -> None:
        if self.trace_index < 0:
            raise InputContractError("segment trace_index cannot be negative.")
        if self.state_id not in {0, 1, 2}:
            raise InputContractError("segment state_id must be 0, 1, or 2.")
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise InputContractError("segment extent must be non-empty and ordered.")
        if (
            not np.isfinite(self.duration_fraction)
            or self.duration_fraction <= 0.0
            or self.duration_fraction > 1.0
        ):
            raise InputContractError(
                "segment duration_fraction must be finite in (0, 1]."
            )


@dataclass(frozen=True)
class CoefficientVarianceCalibration:
    """Post-hoc diagonal scale temperature for c0/c1/c2 sampling."""

    SCHEMA = "structured_ginn_v2_coefficient_variance_calibration_v1"

    temperature: tuple[float, float, float]

    def __post_init__(self) -> None:
        values = np.asarray(self.temperature, dtype=np.float64)
        if values.shape != (3,) or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise InputContractError(
                "coefficient variance temperatures must be three finite positive values."
            )
        object.__setattr__(self, "temperature", tuple(float(value) for value in values))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "temperature": list(self.temperature),
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "CoefficientVarianceCalibration":
        if value.get("schema") != cls.SCHEMA:
            raise InputContractError(
                "unsupported coefficient variance calibration schema."
            )
        temperature = tuple(value.get("temperature") or ())
        if len(temperature) != 3:
            raise InputContractError(
                "coefficient variance calibration requires c0/c1/c2 temperatures."
            )
        return cls(temperature=temperature)


@dataclass(frozen=True)
class SegmentParameterDistribution:
    """Diagonal coefficient distribution for one supplied segment extent."""

    extent: SegmentExtent
    mean: tuple[float, float, float]
    scale: tuple[float, float, float]
    parameter_identifiability_rank: int
    parameter_basis_condition: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if mean.shape != (3,) or scale.shape != (3,):
            raise InputContractError("segment parameter mean/scale must have length 3.")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise NumericalFailure("segment parameter distribution must be finite.")
        if self.parameter_identifiability_rank not in {1, 2, 3}:
            raise InputContractError("parameter identifiability rank must be 1, 2, or 3.")
        if (
            not np.isfinite(self.parameter_basis_condition)
            and not np.isinf(self.parameter_basis_condition)
        ) or self.parameter_basis_condition <= 0.0:
            raise InputContractError("parameter basis condition must be positive.")


@dataclass(frozen=True)
class Segment:
    trace_index: int
    state_id: int
    start_index: int
    stop_index: int
    c0: float
    c1: float
    c2: float

    def __post_init__(self) -> None:
        if self.trace_index < 0 or self.state_id not in {0, 1, 2}:
            raise InputContractError("segment trace/state identity is invalid.")
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise InputContractError("segment extent must be non-empty and ordered.")
        values = np.asarray((self.c0, self.c1, self.c2), dtype=np.float64)
        if np.any(~np.isfinite(values)):
            raise NumericalFailure("segment coefficients must be finite.")


@dataclass(frozen=True)
class StructuredRealization:
    log_ai_highres: np.ndarray
    state_highres: np.ndarray
    projected_log_ai: np.ndarray
    segments: tuple[Segment, ...]
    conditional_log_score: float

    def __post_init__(self) -> None:
        highres = _array(
            self.log_ai_highres, dtype=np.float64, ndim=2, name="log_ai_highres"
        )
        state = _array(
            self.state_highres, dtype=np.int8, ndim=2, name="state_highres"
        )
        projected = _array(
            self.projected_log_ai,
            dtype=np.float64,
            ndim=2,
            name="projected_log_ai",
        )
        segments = tuple(self.segments)
        if state.shape != highres.shape or projected.shape[0] != highres.shape[0]:
            raise InputContractError("structured realization grids are inconsistent.")
        if not segments or any(not isinstance(item, Segment) for item in segments):
            raise InputContractError("structured realization requires valid segments.")
        if np.any(np.isinf(highres)) or np.any(np.isinf(projected)):
            raise NumericalFailure("structured realization cannot contain infinite values.")
        finite_highres = np.isfinite(highres)
        if np.any(finite_highres & ((state < 0) | (state > 2))):
            raise InputContractError("finite high-resolution samples require valid states.")
        score = float(self.conditional_log_score)
        if not np.isfinite(score):
            raise NumericalFailure("conditional_log_score must be finite.")
        object.__setattr__(self, "log_ai_highres", highres)
        object.__setattr__(self, "state_highres", state)
        object.__setattr__(self, "projected_log_ai", projected)
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "conditional_log_score", score)


@dataclass(frozen=True)
class StructuredPrediction:
    """One deterministic structured prediction for one observation tile."""

    evidence: ObservableEvidence
    realization: StructuredRealization
    state_probability: np.ndarray
    renewal_probability: np.ndarray
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, ObservableEvidence) or not isinstance(
            self.realization, StructuredRealization
        ):
            raise TypeError("prediction requires ObservableEvidence and StructuredRealization.")
        model_shape = self.evidence.support.shape
        highres_shape = self.evidence.highres_support.shape
        state_probability = _array(
            self.state_probability,
            dtype=np.float64,
            ndim=3,
            name="state_probability",
        )
        renewal_probability = _array(
            self.renewal_probability,
            dtype=np.float64,
            ndim=2,
            name="renewal_probability",
        )
        if state_probability.shape != model_shape + (3,) or (
            renewal_probability.shape != model_shape
        ):
            raise InputContractError("prediction marginal shapes differ from evidence.")
        support = self.evidence.support
        if np.any(
            support
            & ~np.all(np.isfinite(state_probability), axis=-1)
        ) or np.any(
            support
            & (~np.isfinite(renewal_probability))
        ):
            raise NumericalFailure("supported prediction marginals must be finite.")
        if np.any(
            support
            & ~np.isclose(
                np.sum(state_probability, axis=-1),
                1.0,
                rtol=0.0,
                atol=1.0e-6,
            )
        ) or np.any(
            support
            & ((renewal_probability < 0.0) | (renewal_probability > 1.0))
        ):
            raise InputContractError("prediction marginals are not normalized probabilities.")
        if self.realization.log_ai_highres.shape != highres_shape or (
            self.realization.projected_log_ai.shape != model_shape
        ):
            raise InputContractError("prediction realization axes differ from evidence.")
        if np.any(
            self.evidence.highres_support
            & ~np.isfinite(self.realization.log_ai_highres)
        ):
            raise NumericalFailure("supported high-resolution prediction must be finite.")
        diagnostics = {str(key): float(value) for key, value in self.diagnostics.items()}
        if any(not key or not np.isfinite(value) for key, value in diagnostics.items()):
            raise NumericalFailure("prediction diagnostics must be finite named scalars.")
        object.__setattr__(self, "state_probability", state_probability)
        object.__setattr__(self, "renewal_probability", renewal_probability)
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True)
class VolumeInferenceResult:
    """Deterministic predictions keyed by stable volume-tile identity."""

    tiles: Mapping[str, StructuredPrediction]

    def __post_init__(self) -> None:
        if not self.tiles:
            raise InputContractError("volume inference must contain at least one tile.")
        object.__setattr__(self, "tiles", dict(sorted(self.tiles.items())))



__all__ = [
    "CoefficientVarianceCalibration",
    "DomainMismatchError",
    "InputContractError",
    "NumericalFailure",
    "ObservableEvidence",
    "ObservableTargetContract",
    "ObservationTile",
    "Segment",
    "SegmentExtent",
    "SegmentParameterDistribution",
    "StructuredPrediction",
    "StructuredRealization",
    "VolumeInferenceResult",
    "zone_linear_lateral_support",
]
