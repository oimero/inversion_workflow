"""cup.seismic.gain: time-domain dynamic gain computation.

This module houses the business logic for estimating, fitting, and applying a
time-domain dynamic gain volume. The gain maps unit-wavelet GINN-target
synthetics at well anchors to the same normalized seismic domain used by
GINN training:

    seismic_norm = seismic_raw / train_mask_rms

Boundary
--------
- Functions assume a time-domain seismic volume and an LFM volume already loaded
  in memory.  I/O, CLI, and figure-rendering are kept in ``scripts/dynamic_gain.py``.
- NPZ export uses the ``dynamic_gain_v3`` schema.

Core public objects
-------------------
1. positive_ls_gain: local least-squares gain for a sample segment.
2. segment_attribute_values: compute seismic attributes from sample values.
3. assign_spatial_clusters: cluster well segments by XY proximity.
4. recommended_fixed_gain: spatial-debiased fixed-gain recommendation.
5. choose_attribute: select the best seismic attribute for gain prediction.
6. fit_gain_relationship: OLS log-gain vs log-attribute.
7. compute_attribute_axis: compute a moving-window attribute over a flat trace array.
8. build_gain_volume: apply the fitted relationship to the full seismic volume.
9. write_gain_npz: write the dynamic_gain_v3 NPZ file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cup.utils.io import to_json_compatible
from cup.utils.raw_trace import centered_moving_average, centered_moving_rms_axis, centered_moving_sum_axis
from cup.utils.statistics import ols_fit, pearson_r, radius_connected_components, spearman_rho

# ── Candidate attributes ──
# These are the valid keys for the attribute-selection loop; unknown keys
# are rejected with a clear error message.
CANDIDATE_ATTRIBUTES = ("seismic_rms", "seismic_abs_mean", "seismic_abs_p90")

SCHEMA_VERSION = "dynamic_gain_v3"
GAIN_REFERENCE = "unit_wavelet_ginn_target_synthetic_to_normalized_observation"
NORMALIZATION = "seismic_raw_divided_by_train_mask_rms"


# ── Local gain estimation ──


def positive_ls_gain(
    seismic_values: np.ndarray,
    synthetic_values: np.ndarray,
    *,
    eps: float = 1e-12,
    min_valid_samples: int = 8,
) -> float:
    """Positive least-squares gain mapping *synthetic_values* to *seismic_values*.

    Returns ``NaN`` if the segment has fewer than *min_valid_samples* finite
    values or if the computed gain is non-positive.
    """
    seismic_values = np.asarray(seismic_values, dtype=np.float64)
    synthetic_values = np.asarray(synthetic_values, dtype=np.float64)
    valid = np.isfinite(seismic_values) & np.isfinite(synthetic_values)
    if int(valid.sum()) < int(min_valid_samples):
        return float("nan")
    numerator = float(np.sum(seismic_values[valid] * synthetic_values[valid]))
    denominator = float(np.sum(synthetic_values[valid] ** 2))
    gain = numerator / (denominator + float(eps) * max(int(valid.sum()), 1))
    return float(gain) if np.isfinite(gain) and gain > 0.0 else float("nan")


def segment_attribute_values(seismic_norm_values: np.ndarray) -> dict[str, float]:
    """Compute per-attribute summaries for a 1-D segment of normalized seismic."""
    values = np.asarray(seismic_norm_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {name: float("nan") for name in CANDIDATE_ATTRIBUTES}
    abs_values = np.abs(values)
    return {
        "seismic_rms": float(np.sqrt(np.mean(values ** 2))),
        "seismic_abs_mean": float(np.mean(abs_values)),
        "seismic_abs_p90": float(np.percentile(abs_values, 90.0)),
    }


# ── Spatial debias ──


def assign_spatial_clusters(sample_df: pd.DataFrame, *, radius_m: float, enabled: bool) -> pd.DataFrame:
    """Tag samples with spatial cluster ids and sizes.

    When *enabled* is False every well becomes its own singleton cluster.
    """
    wells = (
        sample_df.groupby("well_name", as_index=False)
        .agg(x_m=("x_m", "median"), y_m=("y_m", "median"))
        .sort_values("well_name")
        .reset_index(drop=True)
    )
    if enabled:
        wells["spatial_cluster_id"] = radius_connected_components(
            wells[["x_m", "y_m"]].to_numpy(dtype=np.float64), float(radius_m)
        )
    else:
        wells["spatial_cluster_id"] = np.arange(len(wells), dtype=np.int64)
    wells["spatial_cluster_size"] = (
        wells.groupby("spatial_cluster_id")["well_name"].transform("count").astype(np.int64)
    )
    return sample_df.merge(
        wells[["well_name", "spatial_cluster_id", "spatial_cluster_size"]],
        on="well_name",
        how="left",
        validate="many_to_one",
    )


# ── Fixed gain recommendation ──


def recommended_fixed_gain(sample_df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Spatial-debiased fixed-gain recommendation from well-segment samples."""
    well_gain = (
        sample_df.groupby(["well_name", "spatial_cluster_id", "spatial_cluster_size"], as_index=False)
        .agg(gain=("gain", "median"), n_segments=("gain", "count"))
        .sort_values(["spatial_cluster_id", "well_name"])
    )
    cluster_gain = (
        well_gain.groupby("spatial_cluster_id", as_index=False)
        .agg(gain=("gain", "median"), n_wells=("well_name", "count"), n_segments=("n_segments", "sum"))
        .sort_values("spatial_cluster_id")
    )
    values = cluster_gain["gain"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("Cannot recommend fixed gain because no positive finite cluster gain exists.")
    payload = {
        "recommended_fixed_gain": float(np.median(values)),
        "n_wells": int(well_gain["well_name"].nunique()),
        "n_segments": int(sample_df.shape[0]),
        "n_spatial_clusters": int(cluster_gain.shape[0]),
        "spatial_debias_method": "segment_median_to_well_median_to_cluster_median_to_global_median",
        "normalization": NORMALIZATION,
        "gain_reference": GAIN_REFERENCE,
    }
    return payload, well_gain, cluster_gain


# ── Attribute selection ──


def _add_log_attribute_columns(sample_df: pd.DataFrame, candidate_attributes: list[str]) -> pd.DataFrame:
    out = sample_df.copy()
    for attr in candidate_attributes:
        values = out[attr].to_numpy(dtype=np.float64)
        log_values = np.full(values.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(values) & (values > 0.0)
        log_values[valid] = np.log(values[valid])
        out[f"log_{attr}"] = log_values
    return out


def choose_attribute(
    sample_df: pd.DataFrame,
    *,
    candidate_attributes: list[str],
    attr_tie_threshold: float = 0.05,
) -> tuple[str, pd.DataFrame]:
    """Select the best seismic attribute for log-log gain prediction.

    Prefers ``seismic_rms`` unless another attribute has a Pearson |r|
    substantially higher (above *attr_tie_threshold*).
    """
    unknown = set(candidate_attributes) - set(CANDIDATE_ATTRIBUTES)
    if unknown:
        raise ValueError(f"Unsupported dynamic gain attributes: {sorted(unknown)}")
    rows = []
    y = sample_df["log_gain"].to_numpy(dtype=np.float64)
    for attr in candidate_attributes:
        x = sample_df[f"log_{attr}"].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        rows.append(
            {
                "attribute": attr,
                "n_samples": int(valid.sum()),
                "pearson": pearson_r(x, y),
                "spearman": spearman_rho(x, y),
            }
        )
    metrics = pd.DataFrame.from_records(rows)
    if metrics.empty:
        raise ValueError("No candidate attributes were configured.")
    score = metrics["pearson"].abs().fillna(-np.inf)
    best_attr = str(metrics.loc[int(score.idxmax()), "attribute"])
    if "seismic_rms" in set(candidate_attributes):
        rms_score = float(
            metrics.loc[metrics["attribute"] == "seismic_rms", "pearson"]
            .abs()
            .fillna(-np.inf)
            .iloc[0]
        )
        best_score = float(score.max())
        if np.isfinite(rms_score) and (best_score - rms_score) < float(attr_tie_threshold):
            best_attr = "seismic_rms"
    return best_attr, metrics


# ── Fitting ──


def fit_gain_relationship(
    sample_df: pd.DataFrame,
    *,
    candidate_attributes: list[str],
    attr_tie_threshold: float = 0.05,
    clip_percentiles: tuple[float, float] = (5.0, 95.0),
    attribute_floor_fraction: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Fit the log-log gain-attribute relationship on well segments."""
    if len(clip_percentiles) != 2:
        raise ValueError(f"clip_percentiles must contain two values, got {clip_percentiles}.")
    clip_low, clip_high = float(clip_percentiles[0]), float(clip_percentiles[1])
    if not (0.0 <= clip_low <= clip_high <= 100.0):
        raise ValueError(f"clip_percentiles must be ordered within [0, 100], got {clip_percentiles}.")
    if float(attribute_floor_fraction) <= 0.0 or not np.isfinite(float(attribute_floor_fraction)):
        raise ValueError(f"attribute_floor_fraction must be positive and finite, got {attribute_floor_fraction}.")

    sample_df = _add_log_attribute_columns(sample_df, list(candidate_attributes))
    selected_attr, attr_metrics = choose_attribute(
        sample_df,
        candidate_attributes=candidate_attributes,
        attr_tie_threshold=attr_tie_threshold,
    )
    fit = ols_fit(
        sample_df[f"log_{selected_attr}"].to_numpy(dtype=np.float64),
        sample_df["log_gain"].to_numpy(dtype=np.float64),
    )
    gain_values = sample_df["gain"].to_numpy(dtype=np.float64)
    attr_values = sample_df[selected_attr].to_numpy(dtype=np.float64)
    attr_positive = attr_values[np.isfinite(attr_values) & (attr_values > 0.0)]
    if attr_positive.size == 0:
        raise ValueError(f"Selected attribute {selected_attr!r} has no positive finite samples.")
    gain_positive = gain_values[np.isfinite(gain_values) & (gain_values > 0.0)]
    if gain_positive.size == 0:
        raise ValueError("Cannot fit dynamic gain because no positive finite gain samples exist.")
    gain_clip = np.percentile(
        gain_positive,
        [clip_low, clip_high],
    )
    if gain_clip[0] <= 0.0 or gain_clip[1] <= 0.0 or gain_clip[0] > gain_clip[1]:
        raise ValueError(f"Invalid gain clip bounds from samples: {gain_clip}.")
    attribute_floor = float(
        np.percentile(attr_positive, 1.0) * float(attribute_floor_fraction)
    )
    attribute_floor = max(attribute_floor, float(np.finfo(np.float32).tiny))
    fit_payload = {
        **fit,
        "attribute_name": selected_attr,
        "gain_clip_low": float(gain_clip[0]),
        "gain_clip_high": float(gain_clip[1]),
        "clip_percentiles": [clip_low, clip_high],
        "attribute_floor": float(attribute_floor),
        "attribute_floor_fraction": float(attribute_floor_fraction),
        "candidate_attributes": candidate_attributes,
    }
    return sample_df, fit_payload, attr_metrics


# ── Attribute volume computation ──


def _moving_abs_mean_axis(values: np.ndarray, window: int) -> np.ndarray:
    """Moving-window absolute-mean along axis 1."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    numerator = centered_moving_sum_axis(np.where(valid, np.abs(values), 0.0), int(window))
    denominator = centered_moving_sum_axis(valid.astype(np.float32), int(window))
    out = np.full(values.shape, np.nan, dtype=np.float32)
    positive = denominator > 0.0
    out[positive] = (numerator[positive] / denominator[positive]).astype(np.float32)
    return out


def _moving_abs_p90_axis(values: np.ndarray, window: int) -> np.ndarray:
    """Moving-window absolute 90th percentile along axis 1."""
    values = np.asarray(values, dtype=np.float32)
    left = int(window) // 2
    right = int(window) - 1 - left
    out = np.full(values.shape, np.nan, dtype=np.float32)
    for row in range(values.shape[0]):
        padded = np.pad(
            np.abs(values[row]).astype(np.float32),
            (left, right),
            mode="constant",
            constant_values=np.nan,
        )
        windows = np.lib.stride_tricks.sliding_window_view(padded, int(window))
        with np.errstate(all="ignore"):
            out[row] = np.nanpercentile(windows, 90.0, axis=1).astype(np.float32)
    return out


def compute_attribute_axis(
    values: np.ndarray, *, attribute_name: str, window_samples: int
) -> np.ndarray:
    """Compute a moving-window seismic attribute on a flat trace array."""
    if int(window_samples) < 1:
        raise ValueError(f"window_samples must be >= 1, got {window_samples}.")
    if attribute_name == "seismic_rms":
        return centered_moving_rms_axis(values, int(window_samples))
    if attribute_name == "seismic_abs_mean":
        return _moving_abs_mean_axis(values, int(window_samples))
    if attribute_name == "seismic_abs_p90":
        return _moving_abs_p90_axis(values, int(window_samples))
    raise ValueError(f"Unsupported attribute_name={attribute_name!r}.")


# ── Gain volume ──


def build_gain_volume(
    seismic_norm_flat: np.ndarray,
    *,
    fit: dict[str, Any],
    sample_step_s: float,
    seismic_shape: tuple[int, int, int],
    window_samples: int,
    gain_smoothing_samples: int = 1,
    batch_traces: int = 512,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a complete dynamic gain volume from normalized seismic traces.

    Parameters
    ----------
    seismic_norm_flat : (n_trace, n_sample) float32
        ``seismic_raw / train_mask_rms``.
    fit : dict
        Output of :func:`fit_gain_relationship`.
    sample_step_s : float
        Seismic sampling interval in seconds.
    window_samples : int
        Odd number of samples for the moving-window attribute.
    gain_smoothing_samples : int
        Odd number of samples for post-prediction gain smoothing (1 = disabled).
    batch_traces : int
        Number of traces processed in each batch.

    Returns
    -------
    volume : (n_il, n_xl, n_sample) float32
    stats : dict
    """
    attr_name = str(fit["attribute_name"])
    attr_floor = float(fit["attribute_floor"])
    batch_traces = int(batch_traces)
    if batch_traces < 1:
        raise ValueError(f"batch_traces must be >= 1, got {batch_traces}.")
    window_samples = int(window_samples)
    if window_samples < 1:
        raise ValueError(f"window_samples must be >= 1, got {window_samples}.")
    log_clip = (float(np.log(fit["gain_clip_low"])), float(np.log(fit["gain_clip_high"])))
    smoothing = int(gain_smoothing_samples)
    if smoothing < 1:
        smoothing = 1
    if smoothing % 2 == 0:
        smoothing += 1
    interpolate_after_smooth = smoothing > 1

    n_sample = seismic_norm_flat.shape[-1]
    gain_flat = np.empty_like(seismic_norm_flat, dtype=np.float32)
    raw_below_clip = 0
    raw_above_clip = 0
    finite_total = 0
    for start in range(0, seismic_norm_flat.shape[0], batch_traces):
        end = min(start + batch_traces, seismic_norm_flat.shape[0])
        batch = seismic_norm_flat[start:end]
        attr = compute_attribute_axis(batch, attribute_name=attr_name, window_samples=window_samples)
        attr_safe = np.maximum(attr, attr_floor)
        log_attr = np.where(np.isfinite(attr_safe) & (attr_safe > 0.0), np.log(attr_safe), np.nan)
        log_gain = float(fit["intercept"]) + float(fit["slope"]) * log_attr
        finite = np.isfinite(log_gain)
        raw_below_clip += int(np.sum(finite & (log_gain < log_clip[0])))
        raw_above_clip += int(np.sum(finite & (log_gain > log_clip[1])))
        finite_total += int(np.sum(finite))
        log_gain = np.clip(log_gain, log_clip[0], log_clip[1])
        gain = np.exp(log_gain).astype(np.float32)
        fill = float(np.sqrt(float(fit["gain_clip_low"]) * float(fit["gain_clip_high"])))
        gain = np.where(np.isfinite(gain) & (gain > 0.0), gain, fill).astype(np.float32)
        if interpolate_after_smooth:
            for row in range(gain.shape[0]):
                gain[row] = centered_moving_average(gain[row], smoothing)
            gain = np.where(np.isfinite(gain) & (gain > 0.0), gain, fill).astype(np.float32)
        gain_flat[start:end] = gain

    volume = gain_flat.reshape(seismic_shape)
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("Predicted dynamic gain volume contains non-finite or non-positive values.")
    stats = {
        "shape": list(volume.shape),
        "size": int(volume.size),
        "finite_count": int(volume.size),
        "min": float(np.min(volume)),
        "p05": float(np.percentile(volume, 5.0)),
        "median": float(np.median(volume)),
        "p95": float(np.percentile(volume, 95.0)),
        "max": float(np.max(volume)),
        "mean": float(np.mean(volume)),
        "attribute_window_samples": int(window_samples),
        "attribute_window_s": float(window_samples * sample_step_s),
        "gain_smoothing_samples": int(smoothing),
        "raw_below_clip_count": int(raw_below_clip),
        "raw_above_clip_count": int(raw_above_clip),
        "raw_clip_fraction": float((raw_below_clip + raw_above_clip) / max(finite_total, 1)),
    }
    return volume, stats


# ── NPZ export ──


def write_gain_npz(
    path: Path,
    gain_volume: np.ndarray,
    *,
    samples: np.ndarray,
    ilines: np.ndarray,
    xlines: np.ndarray,
    geometry: dict[str, Any],
    wavelet_file: str,
    lfm_file: str,
    anchor_file: str,
    ginn_target_source: str,
    ginn_target_semantics: str,
    diagnostic_ginn_cutoff_hz: float,
    filtered_las_sources: dict[str, str],
    train_mask_rms: float,
    fit: dict[str, Any],
    volume_stats: dict[str, Any],
    lfm_metadata: dict[str, Any],
) -> None:
    """Write ``dynamic_gain.npz`` conforming to the ``dynamic_gain_v3`` schema."""
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sample_domain": "time",
        "sample_unit": "s",
        "gain_reference": GAIN_REFERENCE,
        "normalization": NORMALIZATION,
        "train_mask_rms": float(train_mask_rms),
        "gain_model_is_relative_to_fixed_gain": False,
        "unit_wavelet_file": wavelet_file,
        "ai_lfm_file": lfm_file,
        "log_ai_anchor_file": anchor_file,
        "ginn_target_source": ginn_target_source,
        "ginn_target_semantics": ginn_target_semantics,
        "diagnostic_ginn_cutoff_hz": float(diagnostic_ginn_cutoff_hz),
        "filtered_las_sources": dict(filtered_las_sources),
        "target_layer": lfm_metadata.get("target_layer"),
        "horizons": lfm_metadata.get("horizons"),
        "fit": fit,
        "volume_stats": volume_stats,
        "path_style": "repo_relative",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        volume=gain_volume.astype(np.float32),
        samples=np.asarray(samples, dtype=np.float64),
        inline=np.asarray(ilines, dtype=np.float64),
        xline=np.asarray(xlines, dtype=np.float64),
        geometry_json=np.asarray(json.dumps(to_json_compatible(geometry), ensure_ascii=False)),
        metadata_json=np.asarray(json.dumps(to_json_compatible(metadata), ensure_ascii=False)),
    )
