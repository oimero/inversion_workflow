"""Log-AI anchor schema, well-control data, and batch sampler for GINN trainers.

Public entrypoint: ``from ginn.anchor import ...``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Sampler

from cup.seismic.geometry import nominal_bin_spacing_m, xy_circle_mask
from cup.utils.io import to_json_compatible

logger = logging.getLogger(__name__)

# Schema constants.

SCHEMA_VERSION = "ginn_log_ai_anchor_v2"
VALID_ANCHOR_TYPES = {"well", "facies_control"}
VALID_SAMPLE_DOMAINS = {"time", "depth"}
VALID_SAMPLE_UNITS = {"s", "m"}
TIME_TARGET_SEMANTICS = {
    "frequency_lowpass": "twt_log_ai_lowpass_at_diagnostic_ginn_cutoff",
    "auto_tie_filtered_las": "fourth_step_auto_tie_filtered_las_projected_with_optimized_tdt",
}


# Schema dataclass.


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


# Schema helpers.


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
    """Load a ``ginn_log_ai_anchor_v2`` NPZ file."""
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

    if (
        bundle.sample_domain == "time"
        and str(bundle.metadata.get("artifact_family", "")) == "well_constraints_time"
    ):
        target_source = str(bundle.metadata.get("ginn_target_source", ""))
        if target_source not in TIME_TARGET_SEMANTICS:
            raise ValueError(
                "Time-domain well anchor metadata ginn_target_source must be one of "
                f"{sorted(TIME_TARGET_SEMANTICS)}, got {target_source!r}."
            )
        expected_semantics = TIME_TARGET_SEMANTICS[target_source]
        if bundle.metadata.get("ginn_target_semantics") != expected_semantics:
            raise ValueError(
                "Time-domain well anchor metadata ginn_target_semantics must be "
                f"{expected_semantics!r} for source {target_source!r}."
            )
        diagnostic_cutoff = bundle.metadata.get("diagnostic_ginn_cutoff_hz")
        if (
            diagnostic_cutoff is None
            or not np.isfinite(float(diagnostic_cutoff))
            or float(diagnostic_cutoff) <= 0.0
        ):
            raise ValueError(
                "Time-domain well anchor metadata must contain positive finite "
                "diagnostic_ginn_cutoff_hz."
            )
        if target_source == "auto_tie_filtered_las":
            filtered_sources = bundle.metadata.get("filtered_las_sources")
            if not isinstance(filtered_sources, dict) or not filtered_sources:
                raise ValueError(
                    "auto_tie_filtered_las anchor metadata must contain non-empty filtered_las_sources."
                )
        frequency_bands = bundle.metadata.get("frequency_bands")
        if not isinstance(frequency_bands, dict):
            raise ValueError("Time-domain well anchor metadata must contain frequency_bands.")
        for key in ("lfm_cutoff_hz", "ginn_cutoff_hz", "reference_cutoff_hz"):
            value = frequency_bands.get(key)
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"Time-domain well anchor metadata is missing finite {key}.")


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


# Well-control data.


@dataclass(frozen=True)
class WellControlData:
    """Sparse per-trace well-control targets for in-batch GINN supervision."""

    flat_indices: np.ndarray
    target_log_ai: np.ndarray
    anchor_weight: np.ndarray
    well_influence: np.ndarray
    waveform_weight_scale: np.ndarray
    summary: dict[str, Any]

    @classmethod
    def empty(cls, n_sample: int, *, summary: dict[str, Any] | None = None) -> "WellControlData":
        return cls(
            flat_indices=np.empty(0, dtype=np.int64),
            target_log_ai=np.zeros((0, int(n_sample)), dtype=np.float32),
            anchor_weight=np.zeros((0, int(n_sample)), dtype=np.float32),
            well_influence=np.zeros((0,), dtype=np.float32),
            waveform_weight_scale=np.ones((0,), dtype=np.float32),
            summary={} if summary is None else dict(summary),
        )


def compute_anchor_influence(
    distance_m: float,
    radius_xy_m: float,
    decay: str,
    bin_spacing_m: float,
) -> float:
    """Compute well influence factor from XY distance using the configured decay."""
    if distance_m < 1e-8:
        return 1.0
    if decay == "linear":
        return max(0.0, 1.0 - distance_m / max(radius_xy_m, 1e-6))
    sigma = max(radius_xy_m / 2.0, 0.5 * bin_spacing_m)
    gaussian_edge = math.exp(-(radius_xy_m**2) / (2.0 * sigma**2)) if radius_xy_m > 0.0 else 0.0
    raw = math.exp(-(distance_m**2) / (2.0 * sigma**2))
    influence = (raw - gaussian_edge) / max(1.0 - gaussian_edge, 1e-6)
    return max(0.0, min(1.0, influence))


def build_well_control_data(
    *,
    anchor_file: Path | None,
    selected_indices: np.ndarray,
    sample_domain: str,
    n_sample: int,
    n_traces: int,
    geometry: dict[str, Any],
    lambda_log_ai_anchor: float,
    radius_xy_m: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    well_waveform_min_weight: float,
    distance_decay: str,
    log_prefix: str = "Well-control",
) -> WellControlData:
    """Build a sparse map of well-controlled traces for in-batch anchor loss."""
    selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    radius_xy_m = float(radius_xy_m)
    if lambda_log_ai_anchor <= 0.0 or anchor_file is None or selected_indices.size == 0 or radius_xy_m < 0.0:
        return WellControlData.empty(
            n_sample,
            summary={
                "enabled": False,
                "anchor_file": anchor_file,
                "reason": "disabled_or_empty",
            },
        )

    if distance_decay not in {"gaussian", "linear"}:
        raise ValueError(f"Unsupported anchor distance decay: {distance_decay!r}.")

    anchor_bundle = load_log_ai_anchor_npz(anchor_file)
    validate_log_ai_anchor(
        anchor_bundle,
        sample_domain=sample_domain,
        n_sample=n_sample,
        n_traces=n_traces,
    )

    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    selected_set = {int(flat_idx) for flat_idx in selected_indices}
    n_il = int(geometry["n_il"])
    n_xl = int(geometry["n_xl"])
    if x_grid.shape != (n_il, n_xl) or y_grid.shape != (n_il, n_xl):
        raise ValueError(
            f"x_grid/y_grid shape must be {(n_il, n_xl)}, got {x_grid.shape} and {y_grid.shape}."
        )
    bin_spacing_m = nominal_bin_spacing_m(x_grid, y_grid)

    best_by_flat: dict[int, tuple[float, int]] = {}
    for row, flat_idx in enumerate(np.asarray(anchor_bundle.flat_indices, dtype=np.int64)):
        target = np.asarray(anchor_bundle.target_log_ai[row], dtype=np.float32)
        target_mask = np.asarray(anchor_bundle.anchor_mask[row], dtype=bool) & np.isfinite(target)
        if not np.any(target_mask):
            continue

        flat_ref = int(flat_idx)
        il_ref = flat_ref // n_xl
        xl_ref = flat_ref % n_xl
        if not (0 <= il_ref < n_il and 0 <= xl_ref < n_xl):
            continue
        circle_mask, distance_m = xy_circle_mask(
            x_grid,
            y_grid,
            center_x=float(x_grid[il_ref, xl_ref]),
            center_y=float(y_grid[il_ref, xl_ref]),
            radius_xy_m=radius_xy_m,
        )
        for il, xl in np.argwhere(circle_mask):
            il_int = int(il)
            xl_int = int(xl)
            flat = il_int * n_xl + xl_int
            if flat not in selected_set:
                continue
            dist = float(distance_m[il_int, xl_int])
            influence = compute_anchor_influence(dist, radius_xy_m, distance_decay, bin_spacing_m)
            if influence <= 0.0:
                continue
            current = best_by_flat.get(flat)
            if current is None or influence > current[0]:
                best_by_flat[flat] = (float(influence), int(row))

    if not best_by_flat:
        return WellControlData.empty(
            n_sample,
            summary={
                "enabled": False,
                "anchor_file": anchor_file,
                "reason": "no_anchor_trace_in_selected_indices",
                "n_input_anchors": int(anchor_bundle.n_anchors),
            },
        )

    flat_indices = np.array(sorted(best_by_flat), dtype=np.int64)
    influences = np.array([best_by_flat[int(flat)][0] for flat in flat_indices], dtype=np.float32)
    rows = np.array([best_by_flat[int(flat)][1] for flat in flat_indices], dtype=np.int64)
    target_log_ai = np.asarray(anchor_bundle.target_log_ai[rows], dtype=np.float32)
    valid = np.asarray(anchor_bundle.anchor_mask[rows], dtype=bool) & np.isfinite(target_log_ai)
    anchor_weight = np.asarray(anchor_bundle.anchor_weight[rows], dtype=np.float32)
    anchor_weight = np.where(np.isfinite(anchor_weight) & (anchor_weight > 0.0), anchor_weight, 0.0)
    anchor_weight = (anchor_weight * valid.astype(np.float32)).astype(np.float32)
    waveform_weight_scale = (1.0 - (1.0 - float(well_waveform_min_weight)) * influences).astype(np.float32)

    anchor_names = np.asarray(anchor_bundle.anchor_names).astype(str)
    unique_rows, counts = np.unique(rows, return_counts=True)
    summary = {
        "enabled": True,
        "anchor_file": Path(anchor_file),
        "n_input_anchors": int(anchor_bundle.n_anchors),
        "n_controlled_traces": int(flat_indices.size),
        "radius_xy_m": radius_xy_m,
        "nominal_bin_spacing_m": float(bin_spacing_m),
        "distance_domain": "xy_m",
        "distance_decay": distance_decay,
        "well_waveform_min_weight": float(well_waveform_min_weight),
        "influence_min": float(np.min(influences)),
        "influence_mean": float(np.mean(influences)),
        "influence_max": float(np.max(influences)),
        "waveform_weight_min": float(np.min(waveform_weight_scale)),
        "waveform_weight_mean": float(np.mean(waveform_weight_scale)),
        "waveform_weight_max": float(np.max(waveform_weight_scale)),
        "controlled_traces_by_anchor": {
            str(anchor_names[int(row)]): int(count) for row, count in zip(unique_rows, counts)
        },
    }
    logger.info(
        "%s map: controlled_traces=%d, radius_xy_m=%.3f, influence=[%.3f, %.3f], waveform_weight=[%.3f, %.3f]",
        log_prefix,
        int(flat_indices.size),
        radius_xy_m,
        float(np.min(influences)),
        float(np.max(influences)),
        float(np.min(waveform_weight_scale)),
        float(np.max(waveform_weight_scale)),
    )
    return WellControlData(
        flat_indices=flat_indices,
        target_log_ai=target_log_ai,
        anchor_weight=anchor_weight,
        well_influence=influences,
        waveform_weight_scale=waveform_weight_scale,
        summary=summary,
    )


class MixedWellBatchSampler(Sampler[list[int]]):
    """Batch sampler that replaces a fraction of each batch with well-controlled traces."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        well_fraction: float,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not 0.0 <= well_fraction <= 1.0:
            raise ValueError(f"well_fraction must be within [0, 1], got {well_fraction}.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.well_fraction = float(well_fraction)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0
        self.anchor_indices = np.asarray(dataset.anchor_dataset_indices, dtype=np.int64)
        self.ordinary_indices = np.asarray(dataset.ordinary_dataset_indices, dtype=np.int64)
        self.all_indices = np.arange(len(dataset), dtype=np.int64)

        if self.anchor_indices.size and self.well_fraction > 0.0:
            self.n_well_per_batch = int(round(self.batch_size * self.well_fraction))
            self.n_well_per_batch = min(max(self.n_well_per_batch, 1), self.batch_size)
        else:
            self.n_well_per_batch = 0
        self.n_ordinary_per_batch = self.batch_size - self.n_well_per_batch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return int(len(self.dataset) // self.batch_size)
        return int(math.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        n_batches = len(self)
        if self.n_well_per_batch <= 0 or self.anchor_indices.size == 0:
            shuffled = rng.permutation(self.all_indices)
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                stop = min(start + self.batch_size, shuffled.size)
                batch = shuffled[start:stop]
                if batch.size == self.batch_size or not self.drop_last:
                    yield batch.astype(np.int64).tolist()
            return

        ordinary_pool = self.ordinary_indices if self.ordinary_indices.size else self.all_indices
        ordinary_needed = max(self.n_ordinary_per_batch, 0) * n_batches
        if ordinary_needed:
            repeats = int(math.ceil(ordinary_needed / max(ordinary_pool.size, 1)))
            ordinary_draws = np.concatenate([rng.permutation(ordinary_pool) for _ in range(repeats)])
        else:
            ordinary_draws = np.empty(0, dtype=np.int64)

        ordinary_cursor = 0
        for _ in range(n_batches):
            parts: list[np.ndarray] = []
            if self.n_ordinary_per_batch > 0:
                ordinary = ordinary_draws[ordinary_cursor : ordinary_cursor + self.n_ordinary_per_batch]
                ordinary_cursor += self.n_ordinary_per_batch
                parts.append(ordinary)
            well = rng.choice(self.anchor_indices, size=self.n_well_per_batch, replace=True)
            parts.append(np.asarray(well, dtype=np.int64))
            batch = np.concatenate(parts)
            rng.shuffle(batch)
            yield batch.astype(np.int64).tolist()
