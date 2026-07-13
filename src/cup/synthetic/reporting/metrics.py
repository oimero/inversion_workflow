"""Baseline metrics for synthoseis-lite consumers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def finite_mask(*arrays: np.ndarray, base_mask: np.ndarray | None = None) -> np.ndarray:
    if not arrays:
        raise ValueError("finite_mask needs at least one array.")
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    if base_mask is not None:
        mask &= np.asarray(base_mask, dtype=bool)
    for array in arrays:
        values = np.asarray(array)
        if values.shape != mask.shape:
            raise ValueError(f"shape mismatch: {values.shape} vs {mask.shape}")
        mask &= np.isfinite(values)
    return mask


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    mask = finite_mask(target, prediction, base_mask=valid_mask)
    n = int(np.count_nonzero(mask))
    if n < 2:
        return {
            "status": "insufficient_valid_samples",
            "n_valid": n,
            "bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "nrmse": float("nan"),
            "corr": float("nan"),
            "target_rms": float("nan"),
            "prediction_rms": float("nan"),
            "target_std": float("nan"),
        }
    y = np.asarray(target, dtype=np.float64)[mask]
    p = np.asarray(prediction, dtype=np.float64)[mask]
    residual = p - y
    centered_y = y - float(np.mean(y))
    centered_p = p - float(np.mean(p))
    target_std = float(np.sqrt(np.mean(centered_y**2)))
    pred_std = float(np.sqrt(np.mean(centered_p**2)))
    denom = target_std * pred_std
    corr = float(np.mean(centered_y * centered_p) / denom) if denom > 0.0 else float("nan")
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "status": "ok",
        "n_valid": n,
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": rmse,
        "nrmse": rmse / target_std if target_std > 0.0 else float("nan"),
        "corr": corr,
        "target_rms": float(np.sqrt(np.mean(y**2))),
        "prediction_rms": float(np.sqrt(np.mean(p**2))),
        "target_std": target_std,
    }


def energy_rms(values: np.ndarray, *, valid_mask: np.ndarray | None = None) -> float:
    mask = finite_mask(values, base_mask=valid_mask)
    if not np.any(mask):
        return float("nan")
    data = np.asarray(values, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(data**2)))


def metric_row(
    *,
    sample_row: Mapping[str, Any],
    baseline_id: str,
    target: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    keys = [
        "sample_id",
        "sample_kind",
        "suite",
        "section_id",
        "scenario_id",
        "geometry_family",
        "duration_mode",
        "split",
        "status",
        "probe_frequency_hz",
        "probe_phase",
        "probe_lateral_shape",
        "probe_amplitude_multiplier",
        "paired_zero_sample_id",
        "seismic_variant_id",
        "seismic_mismatch_family",
    ]
    row = {key: sample_row.get(key, "") for key in keys}
    row["baseline_id"] = baseline_id
    row.update(regression_metrics(target, prediction, valid_mask=valid_mask))
    return row


def aggregate_metric_rows(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not rows:
        return result
    baselines = sorted({str(row["baseline_id"]) for row in rows})
    for baseline in baselines:
        subset = [row for row in rows if str(row["baseline_id"]) == baseline and row.get("status") == "ok"]
        result[baseline] = {
            "n_samples": len(subset),
            "mean_rmse": _mean(subset, "rmse"),
            "mean_nrmse": _mean(subset, "nrmse"),
            "median_corr": _median(subset, "corr"),
            "mean_bias": _mean(subset, "bias"),
        }
    return result


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _median(rows: list[Mapping[str, Any]], key: str) -> float:
    values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")
