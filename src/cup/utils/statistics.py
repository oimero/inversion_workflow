"""cup.utils.statistics: 通用统计与拟合辅助工具。

本模块提供 Pearson 相关系数等基础统计函数。

边界说明
--------
- 本模块不依赖地球物理库，仅使用 numpy/pandas。

核心公开对象
------------
1. pearson_r: Pearson 相关系数，自动忽略非有限值。
2. radius_connected_components: 半径连边的连通分量。
3. aggregate_cluster_then_global: 先簇内聚合、再簇间聚合。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def pearson_r(
    x: np.ndarray, y: np.ndarray, *, mask: np.ndarray | None = None,
) -> float:
    """Pearson correlation coefficient, ignoring non-finite entries.

    Parameters
    ----------
    x, y : array-like
    mask : bool array, optional
        Extra boolean mask applied on top of the finite check.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    n = int(valid.sum())
    if n < 5:
        return np.nan
    xs = x[valid]
    ys = y[valid]
    if np.std(xs) <= 0.0 or np.std(ys) <= 0.0:
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient, ignoring non-finite entries."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 5:
        return np.nan
    rx = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=float)
    return pearson_r(rx, ry)


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Ordinary least-squares fit y = a + b*x.

    Returns a dict with keys *intercept*, *slope*, *n_samples*, *pearson_r*,
    *r2*, *mae*, *rmse*.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 5:
        raise ValueError("Need at least five finite samples for OLS fitting.")
    xv, yv = x[valid], y[valid]
    design = np.column_stack([np.ones_like(xv), xv])
    intercept, slope = np.linalg.lstsq(design, yv, rcond=None)[0]
    pred = intercept + slope * xv
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = np.nan if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "n_samples": int(valid.sum()),
        "pearson_r": pearson_r(x, y),
        "r2": float(r2),
        "mae": float(np.mean(np.abs(yv - pred))),
        "rmse": float(np.sqrt(np.mean((yv - pred) ** 2))),
    }


def rms(values: np.ndarray) -> float:
    """Root mean square of finite values; returns NaN for an empty array."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def centered_rms(values: np.ndarray, mask: np.ndarray, *, min_count: int = 1) -> float:
    """Root mean square after removing the masked finite mean."""
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if np.count_nonzero(finite) < int(min_count):
        return float("nan")
    selected = np.asarray(values, dtype=np.float64)[finite]
    centered = selected - float(np.mean(selected))
    return float(np.sqrt(np.mean(centered * centered)))


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation between two 1-D arrays."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan")
    a = a[valid] - float(np.mean(a[valid]))
    b = b[valid] - float(np.mean(b[valid]))
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(np.sum(a * b) / denom)


def normalized_mae(
    reference: np.ndarray, estimate: np.ndarray, *, mask: np.ndarray | None = None,
) -> float:
    """Normalised mean absolute error: |ref - est| / |ref| (L1)."""
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    valid = np.isfinite(ref) & np.isfinite(est)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if int(valid.sum()) < 5:
        return np.nan
    denom = float(np.sum(np.abs(ref[valid])))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(np.abs(ref[valid] - est[valid])) / denom)


def radius_connected_components(points_xy: np.ndarray, radius: float) -> np.ndarray:
    """Label connected components formed by point pairs within ``radius``.

    Non-finite points are kept as singleton clusters so dense valid wells do not
    accidentally inherit missing-coordinate rows.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape [n, 2], got {points.shape}.")
    radius = float(radius)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError(f"radius must be a finite non-negative number, got {radius}.")

    n_points = int(points.shape[0])
    labels = np.full(n_points, -1, dtype=np.int64)
    finite = np.isfinite(points).all(axis=1)
    finite_indices = np.flatnonzero(finite)
    radius_sq = radius * radius
    next_label = 0

    for start in range(n_points):
        if labels[start] >= 0:
            continue
        labels[start] = next_label
        if not finite[start]:
            next_label += 1
            continue
        stack = [start]
        while stack:
            current = stack.pop()
            candidate_indices = finite_indices[labels[finite_indices] < 0]
            if candidate_indices.size == 0:
                continue
            deltas = points[candidate_indices] - points[current]
            distances_sq = np.sum(deltas * deltas, axis=1)
            neighbors = candidate_indices[distances_sq <= radius_sq]
            for neighbor in neighbors:
                labels[neighbor] = next_label
                stack.append(int(neighbor))
        next_label += 1

    return labels


def aggregate_cluster_then_global(
    df: pd.DataFrame,
    *,
    value_columns: list[str],
    cluster_column: str,
    group_columns: list[str] | None = None,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Aggregate metrics as cluster median first, then global median/quantiles."""
    group_columns = list(group_columns or [])
    quantiles = list(quantiles or [0.1])
    required = set(group_columns + [cluster_column] + list(value_columns))
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"aggregate input is missing columns: {sorted(missing)}")
    for quantile in quantiles:
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError(f"quantiles must be within [0, 1], got {quantile}.")

    rows: list[dict[str, Any]] = []
    grouped = [((), df)] if not group_columns else df.groupby(group_columns, dropna=False)
    for group_key, group_df in grouped:
        if group_columns:
            key_values = group_key if isinstance(group_key, tuple) else (group_key,)
            base = dict(zip(group_columns, key_values))
        else:
            base = {}
        cluster_values = group_df.groupby(cluster_column, dropna=False)[value_columns].median(numeric_only=True)
        row: dict[str, Any] = {
            **base,
            "n_samples": int(len(group_df)),
            "n_clusters": int(len(cluster_values)),
        }
        for column in value_columns:
            values = cluster_values[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            row[f"spatial_debiased_median_{column}"] = float(np.median(values)) if values.size else np.nan
            for quantile in quantiles:
                q_name = f"p{int(round(float(quantile) * 100)):02d}"
                row[f"spatial_debiased_{q_name}_{column}"] = (
                    float(np.quantile(values, float(quantile))) if values.size else np.nan
                )
        rows.append(row)

    return pd.DataFrame.from_records(rows)
