"""Shared GINN well-constraint file schema and loader utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "ginn_well_constraints_v1"
VALID_SAMPLE_DOMAINS = {"time", "depth"}
VALID_SAMPLE_UNITS = {"s", "m"}


@dataclass(frozen=True)
class WellConstraintBundle:
    """Model-ready well constraints sampled on a GINN trace axis."""

    sample_domain: str
    sample_unit: str
    samples: np.ndarray
    flat_indices: np.ndarray
    well_log_ai_target: np.ndarray
    well_ai_target: np.ndarray
    well_mask: np.ndarray
    well_weight: np.ndarray
    well_names: np.ndarray
    inline: np.ndarray
    xline: np.ndarray
    metadata: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    @property
    def n_wells(self) -> int:
        return int(self.flat_indices.size)

    @property
    def n_sample(self) -> int:
        return int(self.samples.size)


def load_well_constraints_npz(path: str | Path) -> WellConstraintBundle:
    """Load a ``ginn_well_constraints_v1`` NPZ file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "sample_domain",
            "sample_unit",
            "samples",
            "flat_indices",
            "well_log_ai_target",
            "well_ai_target",
            "well_mask",
            "well_weight",
            "well_names",
            "inline",
            "xline",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Well-constraint NPZ is missing keys: {sorted(missing)}")

        schema_version = _as_str(data["schema_version"])
        metadata = _json_to_dict(data["metadata_json"])
        bundle = WellConstraintBundle(
            schema_version=schema_version,
            sample_domain=_as_str(data["sample_domain"]),
            sample_unit=_as_str(data["sample_unit"]),
            samples=np.asarray(data["samples"], dtype=np.float32),
            flat_indices=np.asarray(data["flat_indices"], dtype=np.int64),
            well_log_ai_target=np.asarray(data["well_log_ai_target"], dtype=np.float32),
            well_ai_target=np.asarray(data["well_ai_target"], dtype=np.float32),
            well_mask=np.asarray(data["well_mask"], dtype=bool),
            well_weight=np.asarray(data["well_weight"], dtype=np.float32),
            well_names=np.asarray(data["well_names"]).astype(str),
            inline=np.asarray(data["inline"], dtype=np.float32),
            xline=np.asarray(data["xline"], dtype=np.float32),
            metadata=metadata,
        )

    validate_well_constraints(bundle)
    return bundle


def save_well_constraints_npz(path: str | Path, bundle: WellConstraintBundle) -> Path:
    """Validate and save a ``WellConstraintBundle`` as a compressed NPZ file."""
    validate_well_constraints(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(bundle.schema_version),
        sample_domain=np.asarray(bundle.sample_domain),
        sample_unit=np.asarray(bundle.sample_unit),
        samples=np.asarray(bundle.samples, dtype=np.float32),
        flat_indices=np.asarray(bundle.flat_indices, dtype=np.int64),
        well_log_ai_target=np.asarray(bundle.well_log_ai_target, dtype=np.float32),
        well_ai_target=np.asarray(bundle.well_ai_target, dtype=np.float32),
        well_mask=np.asarray(bundle.well_mask, dtype=bool),
        well_weight=np.asarray(bundle.well_weight, dtype=np.float32),
        well_names=np.asarray(bundle.well_names).astype(str),
        inline=np.asarray(bundle.inline, dtype=np.float32),
        xline=np.asarray(bundle.xline, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(bundle.metadata, ensure_ascii=False)),
    )
    return path


def validate_well_constraints(
    bundle: WellConstraintBundle,
    sample_domain: str | None = None,
    n_sample: int | None = None,
    n_traces: int | None = None,
) -> None:
    """Validate schema, shapes, domains, and optional workflow compatibility."""
    if bundle.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported well-constraint schema_version={bundle.schema_version!r}.")
    if bundle.sample_domain not in VALID_SAMPLE_DOMAINS:
        raise ValueError(f"sample_domain must be one of {sorted(VALID_SAMPLE_DOMAINS)}, got {bundle.sample_domain!r}.")
    if bundle.sample_unit not in VALID_SAMPLE_UNITS:
        raise ValueError(f"sample_unit must be one of {sorted(VALID_SAMPLE_UNITS)}, got {bundle.sample_unit!r}.")
    if sample_domain is not None and bundle.sample_domain != sample_domain:
        raise ValueError(f"Well constraints are for {bundle.sample_domain!r}, expected {sample_domain!r}.")

    samples = np.asarray(bundle.samples)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty 1D array.")
    if samples.size > 1 and np.any(np.diff(samples.astype(np.float64)) <= 0.0):
        raise ValueError("samples must be strictly increasing.")
    if n_sample is not None and samples.size != int(n_sample):
        raise ValueError(f"Well constraints n_sample={samples.size} does not match expected {int(n_sample)}.")

    flat_indices = np.asarray(bundle.flat_indices)
    if flat_indices.ndim != 1:
        raise ValueError("flat_indices must be a 1D array.")
    n_wells = int(flat_indices.size)
    if np.unique(flat_indices).size != n_wells:
        raise ValueError("Duplicate flat_indices are not supported in well constraints v1.")
    if n_traces is not None and n_wells:
        if flat_indices.min() < 0 or flat_indices.max() >= int(n_traces):
            raise ValueError(
                f"flat_indices must be within [0, {int(n_traces)}), "
                f"got min={int(flat_indices.min())}, max={int(flat_indices.max())}."
            )

    expected_2d = (n_wells, samples.size)
    for name in ("well_log_ai_target", "well_ai_target", "well_mask", "well_weight"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != expected_2d:
            raise ValueError(f"{name} shape {value.shape} does not match expected {expected_2d}.")

    for name in ("well_names", "inline", "xline"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != (n_wells,):
            raise ValueError(f"{name} shape {value.shape} does not match expected {(n_wells,)}.")

    if not np.all(np.isfinite(np.asarray(bundle.well_weight, dtype=np.float64))):
        raise ValueError("well_weight must be finite.")
    if np.any(np.asarray(bundle.well_weight) < 0.0):
        raise ValueError("well_weight must be non-negative.")


def _as_str(value: object) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array.reshape(-1)[0])


def _json_to_dict(value: object) -> dict[str, Any]:
    text = _as_str(value)
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("metadata_json must decode to a JSON object.")
    return parsed
