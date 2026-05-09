"""Well-prior schema and reusable log-AI utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cup.utils.io import to_json_compatible
from cup.utils.raw_trace import centered_moving_average

SCHEMA_VERSION = "ginn_well_resolution_prior_v1"
VALID_SAMPLE_DOMAINS = {"time", "depth"}
VALID_SAMPLE_UNITS = {"s", "m"}


@dataclass(frozen=True)
class WellResolutionPriorBundle:
    """Well residuals and summary statistics sampled on a GINN trace axis."""

    sample_domain: str
    sample_unit: str
    samples: np.ndarray
    flat_indices: np.ndarray
    well_log_ai: np.ndarray
    lfm_log_ai: np.ndarray
    residual_log_ai: np.ndarray
    well_ai: np.ndarray
    lfm_ai: np.ndarray
    well_mask: np.ndarray
    well_weight: np.ndarray
    well_names: np.ndarray
    inline: np.ndarray
    xline: np.ndarray
    summary: dict[str, Any]
    metadata: dict[str, Any]
    schema_version: str = SCHEMA_VERSION

    @property
    def n_wells(self) -> int:
        return int(self.flat_indices.size)

    @property
    def n_sample(self) -> int:
        return int(self.samples.size)


def load_well_resolution_prior_npz(path: str | Path) -> WellResolutionPriorBundle:
    """Load a ``ginn_well_resolution_prior_v1`` NPZ file."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "sample_domain",
            "sample_unit",
            "samples",
            "flat_indices",
            "well_log_ai",
            "lfm_log_ai",
            "residual_log_ai",
            "well_ai",
            "lfm_ai",
            "well_mask",
            "well_weight",
            "well_names",
            "inline",
            "xline",
            "summary_json",
            "metadata_json",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Well resolution-prior NPZ is missing keys: {sorted(missing)}")

        bundle = WellResolutionPriorBundle(
            schema_version=_as_str(data["schema_version"]),
            sample_domain=_as_str(data["sample_domain"]),
            sample_unit=_as_str(data["sample_unit"]),
            samples=np.asarray(data["samples"], dtype=np.float32),
            flat_indices=np.asarray(data["flat_indices"], dtype=np.int64),
            well_log_ai=np.asarray(data["well_log_ai"], dtype=np.float32),
            lfm_log_ai=np.asarray(data["lfm_log_ai"], dtype=np.float32),
            residual_log_ai=np.asarray(data["residual_log_ai"], dtype=np.float32),
            well_ai=np.asarray(data["well_ai"], dtype=np.float32),
            lfm_ai=np.asarray(data["lfm_ai"], dtype=np.float32),
            well_mask=np.asarray(data["well_mask"], dtype=bool),
            well_weight=np.asarray(data["well_weight"], dtype=np.float32),
            well_names=np.asarray(data["well_names"]).astype(str),
            inline=np.asarray(data["inline"], dtype=np.float32),
            xline=np.asarray(data["xline"], dtype=np.float32),
            summary=_json_to_dict(data["summary_json"]),
            metadata=_json_to_dict(data["metadata_json"]),
        )

    validate_well_resolution_prior(bundle)
    return bundle


def save_well_resolution_prior_npz(path: str | Path, bundle: WellResolutionPriorBundle) -> Path:
    """Validate and save a ``WellResolutionPriorBundle`` as a compressed NPZ file."""
    validate_well_resolution_prior(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(bundle.schema_version),
        sample_domain=np.asarray(bundle.sample_domain),
        sample_unit=np.asarray(bundle.sample_unit),
        samples=np.asarray(bundle.samples, dtype=np.float32),
        flat_indices=np.asarray(bundle.flat_indices, dtype=np.int64),
        well_log_ai=np.asarray(bundle.well_log_ai, dtype=np.float32),
        lfm_log_ai=np.asarray(bundle.lfm_log_ai, dtype=np.float32),
        residual_log_ai=np.asarray(bundle.residual_log_ai, dtype=np.float32),
        well_ai=np.asarray(bundle.well_ai, dtype=np.float32),
        lfm_ai=np.asarray(bundle.lfm_ai, dtype=np.float32),
        well_mask=np.asarray(bundle.well_mask, dtype=bool),
        well_weight=np.asarray(bundle.well_weight, dtype=np.float32),
        well_names=np.asarray(bundle.well_names).astype(str),
        inline=np.asarray(bundle.inline, dtype=np.float32),
        xline=np.asarray(bundle.xline, dtype=np.float32),
        summary_json=np.asarray(json.dumps(to_json_compatible(bundle.summary), ensure_ascii=False)),
        metadata_json=np.asarray(json.dumps(to_json_compatible(bundle.metadata), ensure_ascii=False)),
    )
    return path


def validate_well_resolution_prior(
    bundle: WellResolutionPriorBundle,
    sample_domain: str | None = None,
    n_sample: int | None = None,
    n_traces: int | None = None,
) -> None:
    """Validate schema, shapes, finite values, and optional workflow compatibility."""
    if bundle.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported well-resolution-prior schema_version={bundle.schema_version!r}.")
    if bundle.sample_domain not in VALID_SAMPLE_DOMAINS:
        raise ValueError(f"sample_domain must be one of {sorted(VALID_SAMPLE_DOMAINS)}, got {bundle.sample_domain!r}.")
    if bundle.sample_unit not in VALID_SAMPLE_UNITS:
        raise ValueError(f"sample_unit must be one of {sorted(VALID_SAMPLE_UNITS)}, got {bundle.sample_unit!r}.")
    if sample_domain is not None and bundle.sample_domain != sample_domain:
        raise ValueError(f"Well prior is for {bundle.sample_domain!r}, expected {sample_domain!r}.")

    samples = np.asarray(bundle.samples)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty 1D array.")
    if samples.size > 1 and np.any(np.diff(samples.astype(np.float64)) <= 0.0):
        raise ValueError("samples must be strictly increasing.")
    if n_sample is not None and samples.size != int(n_sample):
        raise ValueError(f"Well prior n_sample={samples.size} does not match expected {int(n_sample)}.")

    flat_indices = np.asarray(bundle.flat_indices)
    if flat_indices.ndim != 1:
        raise ValueError("flat_indices must be a 1D array.")
    n_wells = int(flat_indices.size)
    if np.unique(flat_indices).size != n_wells:
        raise ValueError("Duplicate flat_indices are not supported in well resolution-prior v1.")
    if n_traces is not None and n_wells:
        if flat_indices.min() < 0 or flat_indices.max() >= int(n_traces):
            raise ValueError(
                f"flat_indices must be within [0, {int(n_traces)}), "
                f"got min={int(flat_indices.min())}, max={int(flat_indices.max())}."
            )

    expected_2d = (n_wells, samples.size)
    for name in ("well_log_ai", "lfm_log_ai", "residual_log_ai", "well_ai", "lfm_ai", "well_mask", "well_weight"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != expected_2d:
            raise ValueError(f"{name} shape {value.shape} does not match expected {expected_2d}.")

    for name in ("well_names", "inline", "xline"):
        value = np.asarray(getattr(bundle, name))
        if value.shape != (n_wells,):
            raise ValueError(f"{name} shape {value.shape} does not match expected {(n_wells,)}.")

    mask = np.asarray(bundle.well_mask, dtype=bool)
    weight = np.asarray(bundle.well_weight, dtype=np.float64)
    if not np.all(np.isfinite(weight)):
        raise ValueError("well_weight must be finite.")
    if np.any(weight < 0.0):
        raise ValueError("well_weight must be non-negative.")

    for name in ("well_log_ai", "lfm_log_ai", "residual_log_ai", "well_ai", "lfm_ai"):
        value = np.asarray(getattr(bundle, name), dtype=np.float64)
        if np.any(~np.isfinite(value[mask])):
            raise ValueError(f"{name} contains non-finite values inside well_mask.")


def summarize_well_resolution_prior(
    residual_log_ai: np.ndarray,
    well_mask: np.ndarray,
    *,
    sample_step: float | None = None,
    well_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute lightweight global statistics for well residual resolution prior."""
    residual = np.asarray(residual_log_ai, dtype=np.float64)
    mask = np.asarray(well_mask, dtype=bool)
    if residual.shape != mask.shape:
        raise ValueError(f"residual shape {residual.shape} does not match mask shape {mask.shape}.")
    if well_weight is not None and np.asarray(well_weight).shape != residual.shape:
        raise ValueError("well_weight shape must match residual shape.")

    valid = mask & np.isfinite(residual)
    values = residual[valid]
    summary: dict[str, Any] = {
        "n_wells": int(residual.shape[0]) if residual.ndim == 2 else 0,
        "n_valid_samples": int(values.size),
        "sample_step": None if sample_step is None else float(sample_step),
        "residual": _robust_stats(values),
        "event_density": _event_density_by_well(residual, valid, sample_step=sample_step),
        "reflectivity": _robust_stats(_reflectivity_values(residual, valid)),
        "spectrum": _spectrum_summary(residual, valid),
    }
    if well_weight is not None:
        summary["weight"] = _robust_stats(np.asarray(well_weight, dtype=np.float64)[valid])
    return summary


def random_reflectivity(
    n_interface: int,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> np.ndarray:
    """Create bounded sparse reflectivity interfaces for synthetic AI construction."""
    if n_interface <= 0:
        raise ValueError(f"n_interface must be positive, got {n_interface}.")
    if not 0.0 < max_abs < 1.0:
        raise ValueError(f"max_abs must be in (0, 1), got {max_abs}.")

    reflectivity = np.zeros((n_interface,), dtype=np.float32)
    generators = [_add_sparse_interfaces, _add_bed_boundaries, _add_thin_bed_pairs, _add_reflectivity_spike]
    n_components = int(np.random.randint(1, 3))
    for _ in range(n_components):
        generator = generators[int(np.random.randint(0, len(generators)))]
        generator(
            reflectivity,
            max_abs=max_abs,
            thin_bed_min_samples=thin_bed_min_samples,
            thin_bed_max_samples=thin_bed_max_samples,
        )

    if not np.any(reflectivity):
        reflectivity[int(np.random.randint(0, n_interface))] = float(np.random.uniform(-max_abs, max_abs))
    return np.clip(reflectivity, -max_abs, max_abs).astype(np.float32, copy=False)


def random_reflectivity_in_taper(
    n_interface: int,
    *,
    taper: np.ndarray,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> np.ndarray:
    """Create reflectivity only at interfaces inside positive taper support."""
    if n_interface <= 0:
        raise ValueError(f"n_interface must be positive, got {n_interface}.")

    taper_1d = np.asarray(taper, dtype=np.float32).reshape(-1)
    if taper_1d.size == n_interface + 1:
        active = (taper_1d[:-1] > 0.0) & (taper_1d[1:] > 0.0)
    elif taper_1d.size == n_interface:
        active = taper_1d > 0.0
    else:
        raise ValueError(
            "taper sample count must match n_interface or n_interface + 1, "
            f"got taper.size={taper_1d.size}, n_interface={n_interface}."
        )

    reflectivity = np.zeros((n_interface,), dtype=np.float32)
    for start, stop in true_runs(active):
        reflectivity[start:stop] = random_reflectivity(
            stop - start,
            max_abs=max_abs,
            thin_bed_min_samples=thin_bed_min_samples,
            thin_bed_max_samples=thin_bed_max_samples,
        )
    return reflectivity


def reflectivity_to_log_ai(reflectivity: np.ndarray, *, initial_log_ai: float = 0.0) -> np.ndarray:
    """Integrate normal-incidence reflectivity into a log-AI trace."""
    r = np.asarray(reflectivity, dtype=np.float32).reshape(-1)
    if r.size <= 0:
        raise ValueError("reflectivity must contain at least one interface.")
    if np.any(~np.isfinite(r)):
        raise ValueError("reflectivity contains non-finite values.")
    r = np.clip(r, -0.95, 0.95).astype(np.float64)
    increments = np.log1p(r) - np.log1p(-r)
    log_ai = np.empty((r.size + 1,), dtype=np.float64)
    log_ai[0] = float(initial_log_ai)
    log_ai[1:] = float(initial_log_ai) + np.cumsum(increments)
    return log_ai.astype(np.float32)


def highpass_log_ai_residual(log_ai_raw: np.ndarray, *, window: int, max_abs: float) -> np.ndarray:
    """Keep only the high-frequency part of a raw log-AI or residual trace."""
    if max_abs <= 0.0:
        raise ValueError(f"max_abs must be positive, got {max_abs}.")
    values = np.asarray(log_ai_raw, dtype=np.float32).reshape(-1)
    if values.size <= 0:
        raise ValueError("log_ai_raw must contain at least one sample.")
    if np.any(~np.isfinite(values)):
        raise ValueError("log_ai_raw contains non-finite values.")

    low = centered_moving_average(values, int(window))
    residual = values - low
    scale_ref = float(np.percentile(np.abs(residual), 99.0)) if residual.size else 0.0
    if scale_ref > max_abs:
        residual = residual * (float(max_abs) / scale_ref)
    return np.clip(residual, -max_abs, max_abs).astype(np.float32, copy=False)


def ai_to_reflectivity(ai: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute normal-incidence reflectivity from a 1D AI trace."""
    ai = np.asarray(ai, dtype=np.float32).reshape(-1)
    if ai.size < 2:
        return np.asarray([], dtype=np.float32)
    upper = ai[:-1]
    lower = ai[1:]
    return ((lower - upper) / (lower + upper + float(eps))).astype(np.float32)


def fit_delta_to_base_ai_bounds(
    delta_log_ai: np.ndarray,
    *,
    safe_base_ai: np.ndarray,
    ai_min: float,
    ai_max: float,
    max_abs: float,
) -> np.ndarray:
    """Clip delta log-AI so ``base_ai * exp(delta)`` stays inside AI bounds."""
    delta_log_ai = np.asarray(delta_log_ai, dtype=np.float32)
    lower = np.log(float(ai_min) / safe_base_ai)
    upper = np.log(float(ai_max) / safe_base_ai)
    lower = np.maximum(lower, -float(max_abs))
    upper = np.minimum(upper, float(max_abs))
    clipped = np.clip(delta_log_ai, lower, upper)
    impossible = lower > upper
    if np.any(impossible):
        clipped[impossible] = np.clip(
            delta_log_ai[impossible],
            np.log(float(ai_min) / safe_base_ai[impossible]),
            np.log(float(ai_max) / safe_base_ai[impossible]),
        )
    return clipped.astype(np.float32, copy=False)


def fit_residual_to_lfm_bounds(
    residual: np.ndarray,
    *,
    safe_lfm: np.ndarray,
    ai_min: float,
    ai_max: float,
    max_abs: float,
) -> np.ndarray:
    """Compatibility alias for prior NPZ code that still uses LFM terminology."""
    return fit_delta_to_base_ai_bounds(
        residual,
        safe_base_ai=safe_lfm,
        ai_min=ai_min,
        ai_max=ai_max,
        max_abs=max_abs,
    )


def edge_taper(length: int) -> np.ndarray:
    """Return a short symmetric taper for stitching sampled residual patches."""
    if length <= 2:
        return np.ones((length,), dtype=np.float32)
    edge = max(1, min(length // 4, 8))
    taper = np.ones((length,), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, edge + 2, dtype=np.float32)[1:-1]
    taper[:edge] = ramp
    taper[-edge:] = ramp[::-1]
    return taper


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open ``[start, stop)`` runs where a 1D boolean mask is true."""
    mask_1d = np.asarray(mask, dtype=bool).reshape(-1)
    if mask_1d.size == 0 or not np.any(mask_1d):
        return []
    padded = np.concatenate(([False], mask_1d, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, edges.size, 2)]


def validate_ai_bounds(ai_min: float, ai_max: float) -> None:
    """Validate positive increasing AI bounds."""
    if ai_min <= 0.0 or ai_max <= ai_min:
        raise ValueError(f"Invalid AI bounds: ai_min={ai_min}, ai_max={ai_max}.")


def _event_density_by_well(
    residual: np.ndarray,
    mask: np.ndarray,
    *,
    sample_step: float | None,
    threshold_quantile: float = 90.0,
) -> dict[str, Any]:
    values = np.abs(residual[mask])
    if values.size == 0:
        return {"threshold": None, "events_per_sample": None, "events_per_unit": None, "per_well": []}
    threshold = float(np.percentile(values, threshold_quantile))
    per_well = []
    for row in range(residual.shape[0]):
        row_mask = mask[row]
        n = int(row_mask.sum())
        if n == 0:
            per_well.append(None)
            continue
        n_events = int((np.abs(residual[row, row_mask]) >= threshold).sum())
        per_well.append(float(n_events / n))
    finite = np.asarray([v for v in per_well if v is not None], dtype=np.float64)
    mean_per_sample = float(finite.mean()) if finite.size else None
    events_per_unit = None
    if mean_per_sample is not None and sample_step is not None and sample_step > 0.0:
        events_per_unit = float(mean_per_sample / sample_step)
    return {
        "threshold": threshold,
        "events_per_sample": mean_per_sample,
        "events_per_unit": events_per_unit,
        "per_well": per_well,
    }


def _reflectivity_values(residual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = []
    for row in range(residual.shape[0]):
        row_mask = mask[row]
        if int(row_mask.sum()) < 2:
            continue
        log_ai = residual[row, row_mask]
        diff = np.diff(log_ai)
        refl = np.tanh(0.5 * diff)
        values.append(refl[np.isfinite(refl)])
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(values)


def _spectrum_summary(residual: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    spectra = []
    frequencies = None
    for row in range(residual.shape[0]):
        row_mask = mask[row]
        values = residual[row, row_mask]
        values = values[np.isfinite(values)]
        if values.size < 8:
            continue
        centered = values - float(np.mean(values))
        amp = np.abs(np.fft.rfft(centered))
        freq = np.fft.rfftfreq(centered.size, d=1.0)
        if frequencies is None:
            frequencies = np.linspace(0.0, 0.5, num=amp.size, dtype=np.float64)
        spectra.append(np.interp(frequencies, freq, amp))
    if not spectra or frequencies is None:
        return {"frequency_cycles_per_sample": [], "amplitude_mean": [], "amplitude_p50": [], "amplitude_p90": []}
    stacked = np.stack(spectra, axis=0)
    return {
        "frequency_cycles_per_sample": frequencies.tolist(),
        "amplitude_mean": np.mean(stacked, axis=0).tolist(),
        "amplitude_p50": np.percentile(stacked, 50, axis=0).tolist(),
        "amplitude_p90": np.percentile(stacked, 90, axis=0).tolist(),
    }


def _robust_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "rms": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "abs_p95": None,
            "abs_p99": None,
            "max_abs": None,
        }
    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "abs_p95": float(np.percentile(abs_arr, 95)),
        "abs_p99": float(np.percentile(abs_arr, 99)),
        "max_abs": float(np.max(abs_arr)),
    }


def _add_sparse_interfaces(
    reflectivity: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    del thin_bed_min_samples
    n = reflectivity.size
    n_events = int(np.random.randint(1, max(2, n // max(thin_bed_max_samples * 3, 1))))
    for _ in range(n_events):
        idx = int(np.random.randint(0, n))
        reflectivity[idx] += float(np.random.uniform(-0.7 * max_abs, 0.7 * max_abs))


def _add_bed_boundaries(
    reflectivity: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    n = reflectivity.size
    pos = int(np.random.randint(0, max(1, thin_bed_max_samples)))
    segment_max = max(thin_bed_max_samples * 5, thin_bed_min_samples + 1)
    while pos < n:
        reflectivity[pos] += float(np.random.uniform(-max_abs, max_abs))
        width = int(np.random.randint(thin_bed_min_samples, segment_max + 1))
        pos += width


def _add_thin_bed_pairs(
    reflectivity: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    n = reflectivity.size
    n_beds = int(np.random.randint(1, max(2, n // max(thin_bed_max_samples * 8, 1))))
    for _ in range(n_beds):
        start = int(np.random.randint(0, n))
        width = int(np.random.randint(thin_bed_min_samples, thin_bed_max_samples + 1))
        stop = start + width
        if stop >= n:
            continue
        amp = float(np.random.uniform(-max_abs, max_abs))
        reflectivity[start] += amp
        reflectivity[stop] -= amp * float(np.random.uniform(0.5, 1.0))


def _add_reflectivity_spike(
    reflectivity: np.ndarray,
    *,
    max_abs: float,
    thin_bed_min_samples: int,
    thin_bed_max_samples: int,
) -> None:
    del thin_bed_min_samples, thin_bed_max_samples
    idx = int(np.random.randint(0, reflectivity.size))
    reflectivity[idx] += float(np.random.uniform(-max_abs, max_abs))


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


