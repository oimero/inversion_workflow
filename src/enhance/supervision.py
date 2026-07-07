"""Time-domain high-frequency well supervision schema.

This module is intentionally separate from :mod:`ginn_depth.prior`.  The latter
keeps the legacy depth-domain resolution-prior format, while this schema only
describes well high-frequency supervision produced by the time-domain step 06.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cup.utils.io import to_json_compatible

SCHEMA_VERSION = "enhance_residual_supervision_v2"
VALID_SAMPLE_DOMAINS = {"time", "depth"}
VALID_SAMPLE_UNITS = {"s", "m"}

__all__ = [
    "SCHEMA_VERSION",
    "VALID_SAMPLE_DOMAINS",
    "VALID_SAMPLE_UNITS",
    "WellHighSupervisionBundle",
    "load_well_high_supervision_npz",
    "save_well_high_supervision_npz",
    "validate_well_high_supervision",
]


@dataclass(frozen=True)
class WellHighSupervisionBundle:
    """High-frequency log-AI supervision sampled at well-controlled traces."""

    sample_domain: str
    sample_unit: str
    samples: np.ndarray
    flat_indices: np.ndarray
    well_names: np.ndarray
    inline: np.ndarray
    xline: np.ndarray
    reference_log_ai: np.ndarray
    ginn_target_log_ai: np.ndarray
    enhance_residual_log_ai: np.ndarray
    well_mask: np.ndarray
    well_weight: np.ndarray
    native_samples: np.ndarray
    native_reference_log_ai: np.ndarray
    native_ginn_target_log_ai: np.ndarray
    native_enhance_residual_log_ai: np.ndarray
    native_well_mask: np.ndarray
    summary: dict[str, Any]
    metadata: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    @property
    def n_traces(self) -> int:
        return int(self.flat_indices.size)

    @property
    def n_sample(self) -> int:
        return int(self.samples.size)

    @property
    def n_native_sample(self) -> int:
        arr = np.asarray(self.native_samples)
        if arr.ndim == 1:
            return int(arr.size)
        if arr.ndim == 2:
            return int(arr.shape[1])
        return 0


def load_well_high_supervision_npz(path: str | Path) -> WellHighSupervisionBundle:
    """Load an ``enhance_residual_supervision_v2`` NPZ file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "sample_domain",
            "sample_unit",
            "samples",
            "flat_indices",
            "well_names",
            "inline",
            "xline",
            "reference_log_ai",
            "ginn_target_log_ai",
            "enhance_residual_log_ai",
            "well_mask",
            "well_weight",
            "native_samples",
            "native_reference_log_ai",
            "native_ginn_target_log_ai",
            "native_enhance_residual_log_ai",
            "native_well_mask",
            "summary_json",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Well high-supervision NPZ is missing keys: {sorted(missing)}")
        bundle = WellHighSupervisionBundle(
            schema_version=_as_str(data["schema_version"]),
            sample_domain=_as_str(data["sample_domain"]),
            sample_unit=_as_str(data["sample_unit"]),
            samples=np.asarray(data["samples"], dtype=np.float32),
            flat_indices=np.asarray(data["flat_indices"], dtype=np.int64),
            well_names=np.asarray(data["well_names"]).astype(str),
            inline=np.asarray(data["inline"], dtype=np.float32),
            xline=np.asarray(data["xline"], dtype=np.float32),
            reference_log_ai=np.asarray(data["reference_log_ai"], dtype=np.float32),
            ginn_target_log_ai=np.asarray(data["ginn_target_log_ai"], dtype=np.float32),
            enhance_residual_log_ai=np.asarray(data["enhance_residual_log_ai"], dtype=np.float32),
            well_mask=np.asarray(data["well_mask"], dtype=bool),
            well_weight=np.asarray(data["well_weight"], dtype=np.float32),
            native_samples=np.asarray(data["native_samples"], dtype=np.float32),
            native_reference_log_ai=np.asarray(data["native_reference_log_ai"], dtype=np.float32),
            native_ginn_target_log_ai=np.asarray(data["native_ginn_target_log_ai"], dtype=np.float32),
            native_enhance_residual_log_ai=np.asarray(data["native_enhance_residual_log_ai"], dtype=np.float32),
            native_well_mask=np.asarray(data["native_well_mask"], dtype=bool),
            summary=_json_to_dict(data["summary_json"]),
            metadata=_json_to_dict(data["metadata_json"]),
        )
    validate_well_high_supervision(bundle)
    return bundle


def save_well_high_supervision_npz(path: str | Path, bundle: WellHighSupervisionBundle) -> Path:
    """Validate and save a ``WellHighSupervisionBundle`` as compressed NPZ."""
    validate_well_high_supervision(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(bundle.schema_version),
        sample_domain=np.asarray(bundle.sample_domain),
        sample_unit=np.asarray(bundle.sample_unit),
        samples=np.asarray(bundle.samples, dtype=np.float32),
        flat_indices=np.asarray(bundle.flat_indices, dtype=np.int64),
        well_names=np.asarray(bundle.well_names).astype(str),
        inline=np.asarray(bundle.inline, dtype=np.float32),
        xline=np.asarray(bundle.xline, dtype=np.float32),
        reference_log_ai=np.asarray(bundle.reference_log_ai, dtype=np.float32),
        ginn_target_log_ai=np.asarray(bundle.ginn_target_log_ai, dtype=np.float32),
        enhance_residual_log_ai=np.asarray(bundle.enhance_residual_log_ai, dtype=np.float32),
        well_mask=np.asarray(bundle.well_mask, dtype=bool),
        well_weight=np.asarray(bundle.well_weight, dtype=np.float32),
        native_samples=np.asarray(bundle.native_samples, dtype=np.float32),
        native_reference_log_ai=np.asarray(bundle.native_reference_log_ai, dtype=np.float32),
        native_ginn_target_log_ai=np.asarray(bundle.native_ginn_target_log_ai, dtype=np.float32),
        native_enhance_residual_log_ai=np.asarray(bundle.native_enhance_residual_log_ai, dtype=np.float32),
        native_well_mask=np.asarray(bundle.native_well_mask, dtype=bool),
        summary_json=np.asarray(json.dumps(to_json_compatible(bundle.summary), ensure_ascii=False)),
        metadata_json=np.asarray(json.dumps(to_json_compatible(bundle.metadata), ensure_ascii=False)),
    )
    return path


def validate_well_high_supervision(
    bundle: WellHighSupervisionBundle,
    *,
    sample_domain: str | None = None,
    n_sample: int | None = None,
    n_traces: int | None = None,
) -> None:
    """Validate schema, shapes, finite supervised values, and weights."""
    if bundle.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported well high-supervision schema_version={bundle.schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}. Rebuild the well high-supervision package."
        )
    if bundle.sample_domain not in VALID_SAMPLE_DOMAINS:
        raise ValueError(f"sample_domain must be one of {sorted(VALID_SAMPLE_DOMAINS)}, got {bundle.sample_domain!r}.")
    if bundle.sample_unit not in VALID_SAMPLE_UNITS:
        raise ValueError(f"sample_unit must be one of {sorted(VALID_SAMPLE_UNITS)}, got {bundle.sample_unit!r}.")
    if sample_domain is not None and bundle.sample_domain != sample_domain:
        raise ValueError(f"Well high-supervision is for {bundle.sample_domain!r}, expected {sample_domain!r}.")

    samples = np.asarray(bundle.samples)
    _validate_1d_axis(samples, "samples")
    if n_sample is not None and samples.size != int(n_sample):
        raise ValueError(f"Well high-supervision n_sample={samples.size} does not match expected {int(n_sample)}.")

    flat_indices = np.asarray(bundle.flat_indices)
    if flat_indices.ndim != 1:
        raise ValueError("flat_indices must be a 1D array.")
    n_rows = int(flat_indices.size)
    if np.unique(flat_indices).size != n_rows:
        raise ValueError(f"Duplicate flat_indices are not supported in {SCHEMA_VERSION}.")
    if n_traces is not None and n_rows:
        if flat_indices.min() < 0 or flat_indices.max() >= int(n_traces):
            raise ValueError(
                f"flat_indices must be within [0, {int(n_traces)}), "
                f"got min={int(flat_indices.min())}, max={int(flat_indices.max())}."
            )

    for name in ("well_names", "inline", "xline"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != (n_rows,):
            raise ValueError(f"{name} shape {value.shape} does not match expected {(n_rows,)}.")

    expected_2d = (n_rows, samples.size)
    for name in ("reference_log_ai", "ginn_target_log_ai", "enhance_residual_log_ai", "well_mask", "well_weight"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != expected_2d:
            raise ValueError(f"{name} shape {value.shape} does not match expected {expected_2d}.")

    mask = np.asarray(bundle.well_mask, dtype=bool)
    weight = np.asarray(bundle.well_weight, dtype=np.float64)
    if not np.all(np.isfinite(weight)):
        raise ValueError("well_weight must be finite.")
    if np.any(weight < 0.0):
        raise ValueError("well_weight must be non-negative.")
    for name in ("reference_log_ai", "ginn_target_log_ai", "enhance_residual_log_ai"):
        value = np.asarray(getattr(bundle, name), dtype=np.float64)
        if np.any(~np.isfinite(value[mask])):
            raise ValueError(f"{name} contains non-finite values inside well_mask.")

    native_samples = np.asarray(bundle.native_samples)
    if native_samples.ndim == 1:
        _validate_1d_axis(native_samples, "native_samples")
        expected_native = (n_rows, native_samples.size)
    elif native_samples.ndim == 2 and native_samples.shape[0] == n_rows:
        for row in range(n_rows):
            row_mask = np.asarray(bundle.native_well_mask[row], dtype=bool)
            row_samples = native_samples[row, row_mask]
            _validate_1d_axis(row_samples, f"native_samples row {row}")
        expected_native = native_samples.shape
    else:
        raise ValueError(f"native_samples must be 1D or shape (n_traces, n_native_sample), got {native_samples.shape}.")

    for name in (
        "native_reference_log_ai",
        "native_ginn_target_log_ai",
        "native_enhance_residual_log_ai",
        "native_well_mask",
    ):
        value = np.asarray(getattr(bundle, name))
        if value.shape != expected_native:
            raise ValueError(f"{name} shape {value.shape} does not match expected {expected_native}.")
    native_mask = np.asarray(bundle.native_well_mask, dtype=bool)
    for name in ("native_reference_log_ai", "native_ginn_target_log_ai", "native_enhance_residual_log_ai"):
        value = np.asarray(getattr(bundle, name), dtype=np.float64)
        if np.any(~np.isfinite(value[native_mask])):
            raise ValueError(f"{name} contains non-finite values inside native_well_mask.")


def _validate_1d_axis(values: np.ndarray, name: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    if arr.size > 1 and np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")


def _as_str(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(arr)


def _json_to_dict(value: np.ndarray) -> dict[str, Any]:
    text = _as_str(value)
    if not text:
        return {}
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("JSON payload must decode to a dictionary.")
    return loaded
