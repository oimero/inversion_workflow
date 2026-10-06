"""Canonical per-well support used by Step 6 and Step 8 waveform evaluation.

The support is a property of the interpreted target interval and the available
observed/well curves.  A later prediction may cover less of it, but that does
not redefine the evaluation window.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from cup.seismic.geometry import SampleAxis


EVALUATION_SUPPORT_SCHEMA = "well_evaluation_support_v1"
MIN_EVALUATION_SUPPORT_SAMPLES = 8
SUPPORT_RULE = (
    "target_interval AND observed_support AND well_curve_support AND "
    "additional_finite_support; choose the longest contiguous run"
)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("support masks must be one-dimensional.")
    padded = np.r_[False, values, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    if changes.size == 0:
        return None
    runs = [(int(start), int(stop)) for start, stop in changes.reshape(-1, 2) if stop > start]
    if not runs:
        return None
    return max(runs, key=lambda item: (item[1] - item[0], -item[0]))


def _validate_axis(axis: SampleAxis) -> np.ndarray:
    if not isinstance(axis, SampleAxis):
        raise TypeError("sample_axis must be a SampleAxis.")
    values = np.asarray(axis.values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("sample_axis must contain finite one-dimensional coordinates.")
    return values


def _validate_mask(mask: np.ndarray, *, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}.")
    return values


@dataclass(frozen=True)
class EvaluationSupport:
    """One fixed half-open sample-index interval for one well."""

    well_name: str
    sample_domain: str
    sample_unit: str
    depth_basis: str | None
    target_interval_start: float
    target_interval_stop: float
    support_start: float
    support_stop: float
    start_index: int
    stop_index: int
    target_samples: int
    candidate_samples: int
    support_samples: int
    support_fraction: float
    rule: str = SUPPORT_RULE

    def __post_init__(self) -> None:
        if not str(self.well_name).strip():
            raise ValueError("EvaluationSupport.well_name must be non-empty.")
        domain = str(self.sample_domain).strip().casefold()
        unit = str(self.sample_unit).strip().casefold()
        expected_unit = "s" if domain == "time" else "m" if domain == "depth" else None
        if expected_unit is None or unit != expected_unit:
            raise ValueError("EvaluationSupport sample domain/unit is invalid.")
        basis = None if self.depth_basis in (None, "") else str(self.depth_basis).strip().casefold()
        if (domain == "depth" and basis != "tvdss") or (domain == "time" and basis is not None):
            raise ValueError("EvaluationSupport depth_basis is inconsistent with sample domain.")
        if not np.isfinite(float(self.target_interval_start)) or not np.isfinite(float(self.target_interval_stop)):
            raise ValueError("EvaluationSupport target interval must be finite.")
        if float(self.target_interval_start) >= float(self.target_interval_stop):
            raise ValueError("EvaluationSupport target interval must be increasing.")
        if int(self.start_index) < 0 or int(self.stop_index) <= int(self.start_index):
            raise ValueError("EvaluationSupport indices must define a non-empty half-open interval.")
        if int(self.target_samples) < int(self.support_samples) or int(self.support_samples) < 1:
            raise ValueError("EvaluationSupport sample counts are inconsistent.")
        if int(self.candidate_samples) < int(self.support_samples):
            raise ValueError("EvaluationSupport candidate_samples is too small.")
        if not np.isfinite(float(self.support_fraction)) or not 0.0 < float(self.support_fraction) <= 1.0:
            raise ValueError("EvaluationSupport support_fraction must be within (0, 1].")
        object.__setattr__(self, "sample_domain", domain)
        object.__setattr__(self, "sample_unit", unit)
        object.__setattr__(self, "depth_basis", basis)

    @property
    def indices(self) -> np.ndarray:
        """Return the canonical half-open support indices."""

        return np.arange(int(self.start_index), int(self.stop_index), dtype=np.int64)

    def validate_axis(self, axis: SampleAxis) -> None:
        values = _validate_axis(axis)
        if (axis.domain, axis.unit, axis.depth_basis) != (
            self.sample_domain,
            self.sample_unit,
            self.depth_basis,
        ):
            raise ValueError(f"{self.well_name}: evaluation support domain/unit differs from the current axis.")
        if int(self.stop_index) > values.size:
            raise ValueError(f"{self.well_name}: evaluation support exceeds the current sample axis.")
        if not np.isclose(values[int(self.start_index)], float(self.support_start), rtol=0.0, atol=1e-10):
            raise ValueError(f"{self.well_name}: evaluation support start differs from the current sample axis.")
        if not np.isclose(
            values[int(self.stop_index) - 1],
            float(self.support_stop),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(f"{self.well_name}: evaluation support stop differs from the current sample axis.")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "well_name": self.well_name,
            "sample_domain": self.sample_domain,
            "sample_unit": self.sample_unit,
            "depth_basis": self.depth_basis,
            "target_interval_start": float(self.target_interval_start),
            "target_interval_stop": float(self.target_interval_stop),
            "support_start": float(self.support_start),
            "support_stop": float(self.support_stop),
            "start_index": int(self.start_index),
            "stop_index": int(self.stop_index),
            "target_samples": int(self.target_samples),
            "candidate_samples": int(self.candidate_samples),
            "support_samples": int(self.support_samples),
            "support_fraction": float(self.support_fraction),
            "rule": str(self.rule),
        }


def derive_evaluation_support(
    *,
    well_name: str,
    sample_axis: SampleAxis,
    target_interval: tuple[float, float],
    observed_support: np.ndarray,
    well_curve_support: np.ndarray,
    additional_support: Iterable[np.ndarray] = (),
    minimum_samples: int = MIN_EVALUATION_SUPPORT_SAMPLES,
) -> EvaluationSupport:
    """Derive one fixed support interval from explicit masks.

    ``target_interval`` is inclusive in coordinate space.  The returned
    indices are half-open so the same interval can be applied to Step 6 and
    Step 8 arrays without recomputing a longest run from predictions.
    """

    axis = _validate_axis(sample_axis)
    if (
        not isinstance(target_interval, tuple)
        or len(target_interval) != 2
        or not np.isfinite(float(target_interval[0]))
        or not np.isfinite(float(target_interval[1]))
    ):
        raise ValueError("target_interval must be a finite (start, stop) tuple.")
    target_start, target_stop = (float(target_interval[0]), float(target_interval[1]))
    if target_start >= target_stop:
        raise ValueError("target_interval must be strictly increasing.")
    required = [
        _validate_mask(observed_support, expected_shape=axis.shape, name="observed_support"),
        _validate_mask(well_curve_support, expected_shape=axis.shape, name="well_curve_support"),
    ]
    for index, mask in enumerate(additional_support):
        required.append(_validate_mask(mask, expected_shape=axis.shape, name=f"additional_support[{index}]"))
    target = (axis >= target_start) & (axis <= target_stop)
    candidate = target.copy()
    for mask in required:
        candidate &= mask
    target_samples = int(np.count_nonzero(target))
    candidate_samples = int(np.count_nonzero(candidate))
    run = _longest_true_run(candidate)
    minimum = int(minimum_samples)
    if minimum < 1:
        raise ValueError("minimum_samples must be positive.")
    if run is None:
        raise ValueError(f"{well_name}: target interval has no common evaluation support.")
    start, stop = run
    support_samples = stop - start
    if support_samples < minimum:
        raise ValueError(
            f"{well_name}: evaluation support has {support_samples} samples; "
            f"at least {minimum} are required."
        )
    if target_samples <= 0:
        raise ValueError(f"{well_name}: target interval contains no samples on the current axis.")
    return EvaluationSupport(
        well_name=str(well_name),
        sample_domain=sample_axis.domain,
        sample_unit=sample_axis.unit,
        depth_basis=sample_axis.depth_basis,
        target_interval_start=target_start,
        target_interval_stop=target_stop,
        support_start=float(axis[start]),
        support_stop=float(axis[stop - 1]),
        start_index=start,
        stop_index=stop,
        target_samples=target_samples,
        candidate_samples=candidate_samples,
        support_samples=support_samples,
        support_fraction=float(support_samples / target_samples),
    )


def write_evaluation_support_manifest(
    path: Path,
    *,
    sample_axis: SampleAxis,
    supports: Mapping[str, EvaluationSupport],
    source: str,
) -> dict[str, Any]:
    """Persist the canonical supports produced by Step 6."""

    _validate_axis(sample_axis)
    rows: dict[str, dict[str, Any]] = {}
    for name, support in supports.items():
        if str(name).casefold() != support.well_name.casefold():
            raise ValueError(f"Support mapping key does not match well name: {name!r}.")
        support.validate_axis(sample_axis)
        rows[support.well_name] = support.to_mapping()
    payload: dict[str, Any] = {
        "schema_version": EVALUATION_SUPPORT_SCHEMA,
        "status": "ok",
        "sample_axis": sample_axis.describe(),
        "source": str(source),
        "rule": SUPPORT_RULE,
        "wells": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def load_evaluation_support_manifest(
    path: Path,
    *,
    sample_axis: SampleAxis | None = None,
) -> dict[str, EvaluationSupport]:
    """Load and validate a Step-6 support manifest."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != EVALUATION_SUPPORT_SCHEMA or payload.get("status") != "ok":
        raise ValueError(f"Unsupported evaluation-support manifest: {path}")
    if payload.get("rule") != SUPPORT_RULE:
        raise ValueError(f"Evaluation-support rule differs from the current rule: {path}")
    if sample_axis is not None:
        axis_info = dict(payload.get("sample_axis") or {})
        described = sample_axis.describe()
        for key in ("n_sample", "sample_min", "sample_max", "sample_step", "sample_domain", "sample_unit", "depth_basis"):
            if key not in axis_info or axis_info[key] != described[key]:
                if isinstance(described[key], float) and key in axis_info:
                    matches = np.isclose(float(axis_info[key]), described[key], rtol=0.0, atol=1e-10)
                else:
                    matches = False
                if not matches:
                    raise ValueError(f"Evaluation-support manifest axis differs at {key!r}: {path}")
    raw_wells = payload.get("wells")
    if not isinstance(raw_wells, Mapping) or not raw_wells:
        raise ValueError(f"Evaluation-support manifest has no wells: {path}")
    supports: dict[str, EvaluationSupport] = {}
    for name, raw in raw_wells.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Evaluation-support entry is not a mapping: {name!r}")
        support = EvaluationSupport(
            well_name=str(raw.get("well_name") or name),
            sample_domain=str(raw.get("sample_domain") or ""),
            sample_unit=str(raw.get("sample_unit") or ""),
            depth_basis=raw.get("depth_basis"),
            target_interval_start=float(raw["target_interval_start"]),
            target_interval_stop=float(raw["target_interval_stop"]),
            support_start=float(raw["support_start"]),
            support_stop=float(raw["support_stop"]),
            start_index=int(raw["start_index"]),
            stop_index=int(raw["stop_index"]),
            target_samples=int(raw["target_samples"]),
            candidate_samples=int(raw["candidate_samples"]),
            support_samples=int(raw["support_samples"]),
            support_fraction=float(raw["support_fraction"]),
            rule=str(raw.get("rule") or ""),
        )
        if support.rule != SUPPORT_RULE:
            raise ValueError(f"Evaluation-support entry has an unknown rule: {support.well_name}")
        if sample_axis is not None:
            support.validate_axis(sample_axis)
        supports[support.well_name] = support
    return supports


__all__ = [
    "EVALUATION_SUPPORT_SCHEMA",
    "EvaluationSupport",
    "MIN_EVALUATION_SUPPORT_SAMPLES",
    "SUPPORT_RULE",
    "derive_evaluation_support",
    "load_evaluation_support_manifest",
    "write_evaluation_support_manifest",
]
