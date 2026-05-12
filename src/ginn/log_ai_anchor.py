"""Log-AI anchor schema for first-stage GINN constraints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cup.utils.io import to_json_compatible

SCHEMA_VERSION = "ginn_log_ai_anchor_v1"
VALID_ANCHOR_TYPES = {"well", "facies_control"}
VALID_SAMPLE_DOMAINS = {"time", "depth"}
VALID_SAMPLE_UNITS = {"s", "m"}


@dataclass(frozen=True)
class LogAIAnchorBundle:
    """Target log-AI traces sampled on a GINN trace axis.

    Unlike the well-resolution prior used by the enhancement stage, this bundle
    contains no residual or LFM-difference fields. It is only a first-stage
    supervision target: where, over which samples, and with what weight the
    predicted AI should be anchored.
    """

    sample_domain: str
    sample_unit: str
    samples: np.ndarray
    flat_indices: np.ndarray
    target_ai: np.ndarray
    target_log_ai: np.ndarray
    anchor_mask: np.ndarray
    anchor_weight: np.ndarray
    anchor_names: np.ndarray
    anchor_types: np.ndarray
    inline: np.ndarray
    xline: np.ndarray
    summary: dict[str, Any]
    metadata: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    @property
    def n_anchors(self) -> int:
        return int(self.flat_indices.size)

    @property
    def n_sample(self) -> int:
        return int(self.samples.size)


def build_log_ai_anchor_bundle(
    *,
    sample_domain: str,
    sample_unit: str,
    samples: np.ndarray,
    flat_indices: np.ndarray,
    target_ai: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_weight: np.ndarray,
    anchor_names: np.ndarray,
    anchor_types: np.ndarray,
    inline: np.ndarray,
    xline: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> LogAIAnchorBundle:
    """Create and validate a log-AI anchor bundle from positive AI targets."""
    target_ai_arr = np.asarray(target_ai, dtype=np.float32)
    mask = np.asarray(anchor_mask, dtype=bool)
    target_log_ai = np.zeros_like(target_ai_arr, dtype=np.float32)
    valid = mask & np.isfinite(target_ai_arr) & (target_ai_arr > 0.0)
    target_log_ai[valid] = np.log(np.clip(target_ai_arr[valid], 1e-6, None)).astype(np.float32)
    summary = summarize_log_ai_anchor(target_log_ai, mask, anchor_weight=anchor_weight)
    bundle = LogAIAnchorBundle(
        sample_domain=sample_domain,
        sample_unit=sample_unit,
        samples=np.asarray(samples, dtype=np.float32),
        flat_indices=np.asarray(flat_indices, dtype=np.int64),
        target_ai=target_ai_arr,
        target_log_ai=target_log_ai,
        anchor_mask=mask,
        anchor_weight=np.asarray(anchor_weight, dtype=np.float32),
        anchor_names=np.asarray(anchor_names).astype(str),
        anchor_types=np.asarray(anchor_types).astype(str),
        inline=np.asarray(inline, dtype=np.float32),
        xline=np.asarray(xline, dtype=np.float32),
        summary=summary,
        metadata={} if metadata is None else dict(metadata),
    )
    validate_log_ai_anchor(bundle)
    return bundle


def load_log_ai_anchor_npz(path: str | Path) -> LogAIAnchorBundle:
    """Load a ``ginn_log_ai_anchor_v1`` NPZ file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "sample_domain",
            "sample_unit",
            "samples",
            "flat_indices",
            "target_ai",
            "target_log_ai",
            "anchor_mask",
            "anchor_weight",
            "anchor_names",
            "anchor_types",
            "inline",
            "xline",
            "summary_json",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Log-AI anchor NPZ is missing keys: {sorted(missing)}")

        bundle = LogAIAnchorBundle(
            schema_version=_as_str(data["schema_version"]),
            sample_domain=_as_str(data["sample_domain"]),
            sample_unit=_as_str(data["sample_unit"]),
            samples=np.asarray(data["samples"], dtype=np.float32),
            flat_indices=np.asarray(data["flat_indices"], dtype=np.int64),
            target_ai=np.asarray(data["target_ai"], dtype=np.float32),
            target_log_ai=np.asarray(data["target_log_ai"], dtype=np.float32),
            anchor_mask=np.asarray(data["anchor_mask"], dtype=bool),
            anchor_weight=np.asarray(data["anchor_weight"], dtype=np.float32),
            anchor_names=np.asarray(data["anchor_names"]).astype(str),
            anchor_types=np.asarray(data["anchor_types"]).astype(str),
            inline=np.asarray(data["inline"], dtype=np.float32),
            xline=np.asarray(data["xline"], dtype=np.float32),
            summary=_json_to_dict(data["summary_json"]),
            metadata=_json_to_dict(data["metadata_json"]),
        )

    validate_log_ai_anchor(bundle)
    return bundle


def save_log_ai_anchor_npz(path: str | Path, bundle: LogAIAnchorBundle) -> Path:
    """Validate and save a log-AI anchor bundle as compressed NPZ."""
    validate_log_ai_anchor(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(bundle.schema_version),
        sample_domain=np.asarray(bundle.sample_domain),
        sample_unit=np.asarray(bundle.sample_unit),
        samples=np.asarray(bundle.samples, dtype=np.float32),
        flat_indices=np.asarray(bundle.flat_indices, dtype=np.int64),
        target_ai=np.asarray(bundle.target_ai, dtype=np.float32),
        target_log_ai=np.asarray(bundle.target_log_ai, dtype=np.float32),
        anchor_mask=np.asarray(bundle.anchor_mask, dtype=bool),
        anchor_weight=np.asarray(bundle.anchor_weight, dtype=np.float32),
        anchor_names=np.asarray(bundle.anchor_names).astype(str),
        anchor_types=np.asarray(bundle.anchor_types).astype(str),
        inline=np.asarray(bundle.inline, dtype=np.float32),
        xline=np.asarray(bundle.xline, dtype=np.float32),
        summary_json=np.asarray(json.dumps(to_json_compatible(bundle.summary), ensure_ascii=False)),
        metadata_json=np.asarray(json.dumps(to_json_compatible(bundle.metadata), ensure_ascii=False)),
    )
    return path


def validate_log_ai_anchor(
    bundle: LogAIAnchorBundle,
    sample_domain: str | None = None,
    n_sample: int | None = None,
    n_traces: int | None = None,
) -> None:
    """Validate schema, shapes, finite values, and workflow compatibility."""
    if bundle.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported log-AI anchor schema_version={bundle.schema_version!r}.")
    if bundle.sample_domain not in VALID_SAMPLE_DOMAINS:
        raise ValueError(f"sample_domain must be one of {sorted(VALID_SAMPLE_DOMAINS)}, got {bundle.sample_domain!r}.")
    if bundle.sample_unit not in VALID_SAMPLE_UNITS:
        raise ValueError(f"sample_unit must be one of {sorted(VALID_SAMPLE_UNITS)}, got {bundle.sample_unit!r}.")
    if sample_domain is not None and bundle.sample_domain != sample_domain:
        raise ValueError(f"Log-AI anchor is for {bundle.sample_domain!r}, expected {sample_domain!r}.")

    samples = np.asarray(bundle.samples)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty 1D array.")
    if samples.size > 1 and np.any(np.diff(samples.astype(np.float64)) <= 0.0):
        raise ValueError("samples must be strictly increasing.")
    if n_sample is not None and samples.size != int(n_sample):
        raise ValueError(f"Log-AI anchor n_sample={samples.size} does not match expected {int(n_sample)}.")

    flat_indices = np.asarray(bundle.flat_indices)
    if flat_indices.ndim != 1:
        raise ValueError("flat_indices must be a 1D array.")
    n_anchors = int(flat_indices.size)
    if np.unique(flat_indices).size != n_anchors:
        raise ValueError("Duplicate flat_indices are not supported in log-AI anchor v1.")
    if n_traces is not None and n_anchors:
        if flat_indices.min() < 0 or flat_indices.max() >= int(n_traces):
            raise ValueError(
                f"flat_indices must be within [0, {int(n_traces)}), "
                f"got min={int(flat_indices.min())}, max={int(flat_indices.max())}."
            )

    expected_2d = (n_anchors, samples.size)
    for name in ("target_ai", "target_log_ai", "anchor_mask", "anchor_weight"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != expected_2d:
            raise ValueError(f"{name} shape {value.shape} does not match expected {expected_2d}.")

    for name in ("anchor_names", "anchor_types", "inline", "xline"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != (n_anchors,):
            raise ValueError(f"{name} shape {value.shape} does not match expected {(n_anchors,)}.")

    unknown_types = set(np.asarray(bundle.anchor_types).astype(str)) - VALID_ANCHOR_TYPES
    if unknown_types:
        raise ValueError(f"anchor_types contains unsupported values: {sorted(unknown_types)}.")

    mask = np.asarray(bundle.anchor_mask, dtype=bool)
    weight = np.asarray(bundle.anchor_weight, dtype=np.float64)
    if not np.all(np.isfinite(weight)):
        raise ValueError("anchor_weight must be finite.")
    if np.any(weight < 0.0):
        raise ValueError("anchor_weight must be non-negative.")

    target_ai = np.asarray(bundle.target_ai, dtype=np.float64)
    target_log_ai = np.asarray(bundle.target_log_ai, dtype=np.float64)
    if np.any(~np.isfinite(target_ai[mask])) or np.any(target_ai[mask] <= 0.0):
        raise ValueError("target_ai must be finite and positive inside anchor_mask.")
    if np.any(~np.isfinite(target_log_ai[mask])):
        raise ValueError("target_log_ai must be finite inside anchor_mask.")


def summarize_log_ai_anchor(
    target_log_ai: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    anchor_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute lightweight global statistics for a log-AI anchor bundle."""
    target = np.asarray(target_log_ai, dtype=np.float64)
    mask = np.asarray(anchor_mask, dtype=bool)
    if target.shape != mask.shape:
        raise ValueError(f"target shape {target.shape} does not match mask shape {mask.shape}.")
    valid = mask & np.isfinite(target)
    summary: dict[str, Any] = {
        "n_anchors": int(target.shape[0]) if target.ndim == 2 else 0,
        "n_valid_samples": int(np.count_nonzero(valid)),
        "target_log_ai": _robust_stats(target[valid]),
    }
    if anchor_weight is not None:
        weight = np.asarray(anchor_weight, dtype=np.float64)
        if weight.shape != target.shape:
            raise ValueError("anchor_weight shape must match target_log_ai shape.")
        summary["weight"] = _robust_stats(weight[valid])
    return summary


def _robust_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


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
        raise ValueError("JSON scalar must decode to an object.")
    return parsed
